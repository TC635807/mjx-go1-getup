#!/usr/bin/env python
"""评估起身策略: 成功率 / 分姿态桶 / 到站时间 (调度器方案的真信号)。

训练日志里的 eval_reward 只当参考 —— 起身任务的判定必须按**查看器真正用的那套**
来: up_z > 0.99 且躯干**相对地面**高度进入 [z_lo, z_hi] 并**保持 0.5s** (10 步)。
本脚本同时给"严格/实用"两个倾角闸门, 以及按**初始姿态**分桶的成功率 —— 后者是
判断"哪一类躺法还不会"的关键 (也是关键帧方案唯一有效的那一档的对照)。

用法:
  python sim/eval_getup.py --pkl policies/go1_getup_policy.pkl
  python sim/eval_getup.py --pkl policies/go1_walk_policy.pkl \\
      --label 走路策略(对照)
"""
import argparse
import os
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.30")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import jax
import jax.numpy as jp
import numpy as np

from envs.go1_getup import Go1Getup, default_config
from sim.common import load_policy


def rpy_deg(q):
  """(w,x,y,z) -> (roll, pitch) 度 (与 sim/probe_getup_mjx.py 同一套公式)。"""
  w, x, y, z = [float(v) for v in q]
  R = np.array([
      [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
  ])
  return (float(np.degrees(np.arctan2(R[2, 1], R[2, 2]))),
          float(np.degrees(np.arcsin(np.clip(-R[2, 0], -1, 1)))))


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--pkl", default=None,
                  help="策略 pkl/checkpoint。与 --zero 二选一")
  ap.add_argument("--zero", action="store_true",
                  help="**先验基线**: 动作恒为 0 (getup_res 下 = 纯关键帧先验; "
                       "getup 下 = 保持当前关节角)。用来在同一批 256 个出生姿态上, "
                       "和学习到的策略做**同协议**对比")
  ap.add_argument("--envs", type=int, default=256)
  ap.add_argument("--steps", type=int, default=300)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--label", type=str, default=None)
  ap.add_argument("--hold", type=int, default=25,
                  help="判定'站住'需要**连续满足的控制步数** (25 步 = 0.5s, "
                       "与查看器调度器的判据一致)")
  ap.add_argument("--check_every", type=int, default=5,
                  help="每多少步结算一次判据 (每步都 np.asarray 会拖慢评估)")
  ap.add_argument("--up_th", type=float, default=0.99,
                  help="站住判据的竖直度阈值: up_z > 该值。0.99 = 倾角 < 8.1 度 (原口径); "
                       "0.95 = < 18.2 度。本项目实测学习型策略稳定站在 ~13 度倾角上, "
                       "所以这个阈值决定了\"成功率\"差一个数量级")
  ap.add_argument("--z_lo", type=float, default=0.24,
                  help="躯干相对地面高度下界 (站直约 0.277)")
  ap.add_argument("--z_hi", type=float, default=0.32)
  ap.add_argument("--graph_mode", default="WARP")
  ap.add_argument("--getup2_action_clip", type=float, default=None,
                  help="[--task getup_v2] **必须与训练时一致**: target = default + "
                       "scale*clip(a,±C)。默认配置是 3.0, 但本项目实测 Go1 需要 8.0"
                       "(thigh 行程 5.2 rad); 不一致会把策略动作夹掉 -> 成功率假 0")
  ap.add_argument("--task", choices=["getup", "getup_res", "getup_v2"], default="getup",
                  help="getup = 增量动作的纯起身环境; getup_res = 关键帧先验 + 学习残差 "
                       "(envs/go1_getup_residual.py)。两者判据/分桶完全一致, 只有动作语义不同")
  ap.add_argument("--difficulty", type=float, default=1.0,
                  help="§29.15 出生难度 α (关节随机度)。**评课程中间档时要跟训练时"
                       "一致**, 否则用全随机摔倒去考一个只练过简单起手的策略, 成功率虚低")
  ap.add_argument("--drop_height", type=float, default=None,
                  help="覆盖 cfg.getup.drop_height。'真实摔倒'档给 0.15 "
                       "(走路中摔倒不是 0.5m 自由落体)")
  ap.add_argument("--orient_rand", type=float, default=1.0,
                  help="§29.15 第二根轴 β (朝向随机度)。β=0 表示身体竖直出生")
  ap.add_argument("--include_standing", action="store_true",
                  help="也评估'站立出生'那一档 (训练用 drop_prob=0.75)。"
                       "**默认关**: 测'能不能爬起来'必须只用真摔倒, 否则站立出生"
                       "开局就满足判据, 成功率被污染, 姿态桶也串了")
  args = ap.parse_args()

  t0 = time.time()
  if args.task == "getup":
    from envs.go1_getup import Go1Getup as _Env
    from envs.go1_getup import default_config as _cfg_fn
  elif args.task == "getup_res":
    from envs.go1_getup_residual import Go1GetupResidual as _Env
    from envs.go1_getup_residual import default_config as _cfg_fn
  else:
    from envs.go1_getup_v2 import Go1GetupV2 as _Env
    from envs.go1_getup_v2 import default_config as _cfg_fn
  cfg = _cfg_fn()
  cfg.graph_mode = args.graph_mode
  if args.task == "getup_v2" and args.getup2_action_clip is not None:
    cfg.getup2.action_clip = args.getup2_action_clip
  if args.task != "getup_v2":
    cfg.getup.difficulty = args.difficulty
    cfg.getup.orient_rand = args.orient_rand
    if args.drop_height is not None:
      cfg.getup.drop_height = args.drop_height
    if not args.include_standing:
      cfg.getup.drop_prob = 1.0      # 全部撒摔倒姿态
  env = _Env(cfg)
  if args.task == "getup_v2":
    g2 = cfg.getup2
    print(f"  getup_v2 出生: z=terrain+{g2.spawn_height}  xy±{g2.spawn_xy_range}  "
          f"随机关节+随机朝向;  动作 target=default+{g2.action_scale}*clip(a,±{g2.action_clip})")
  else:
    print(f"  摔倒采样 drop_prob={cfg.getup.drop_prob} "
          f"(1.0 = 全部真摔倒)  difficulty={cfg.getup.difficulty:.2f} "
          f"orient_rand={cfg.getup.orient_rand:.2f}")
  n_state = env.observation_size["state"][0]
  if args.zero:
    policy, ns = None, n_state
    label = args.label or "零动作基线"
    print(f"[{time.time()-t0:5.1f}s] {label}: 动作恒 0  env obs={n_state}")
  else:
    if not args.pkl:
      print("[错误] 需要 --pkl 或 --zero")
      return 1
    policy, ns = load_policy(args.pkl)
    label = args.label or os.path.basename(args.pkl)
    print(f"[{time.time()-t0:5.1f}s] {label}: 策略 obs={ns}  env obs={n_state}")
    if ns != n_state:
      print(f"[错误] 维度不符: 策略 {ns} != 环境 {n_state} (是不是拿走路策略跑起身?)")
      return 1

  N = args.envs
  reset = jax.jit(jax.vmap(env.reset))
  step = jax.jit(jax.vmap(env.step))
  dummy = jp.zeros((2,), dtype=jp.uint32)
  if args.zero:
    act_fn = jax.jit(jax.vmap(lambda o: jp.zeros(env.action_size)))
  else:
    act_fn = jax.jit(jax.vmap(lambda o: policy(o, dummy)[0]))

  st = reset(jax.random.split(jax.random.PRNGKey(args.seed), N))
  st = jax.block_until_ready(st)

  def z_ref_np(qpos):
    if env._hfield_ok:
      return np.asarray(env.terrain_height(jp.asarray(qpos[:, :2])))
    return np.zeros(len(qpos))

  q0 = np.asarray(st.data.qpos)
  up0 = 1.0 - 2.0 * (q0[:, 4] ** 2 + q0[:, 5] ** 2)
  # 注意: 站立出生 (up_z≈1) 必须单列一桶 —— 归进'俯卧'会把两类完全不同的
  # 初态混在一起 (站着开局本来就满足判据), 成功率会被虚高。
  buckets = {
      "仰卧 (up_z<-0.5)": up0 < -0.5,
      "侧躺 (|up_z|<=0.5)": np.abs(up0) <= 0.5,
      "俯卧 (0.5<up_z<=0.9)": (up0 > 0.5) & (up0 <= 0.9),
      "站立出生 (up_z>0.9)": up0 > 0.9,
  }

  # hold_u / best_hold 都按**控制步**计 (每次检查推进 check_every 步)
  done_seen = np.zeros(N, bool)     # 出界/陷落终止 (env 的 _is_healthy)
  hold_u = np.zeros(N, np.int32)
  best_hold = np.zeros(N, np.int32)
  first_at = np.full(N, -1, np.int32)
  ce = args.check_every
  for i in range(args.steps):
    st = step(st, act_fn(st.obs["state"]))
    done_seen |= np.asarray(st.done) != 0
    if (i + 1) % ce == 0 or i == args.steps - 1:
      q = np.asarray(st.data.qpos)
      gh = z_ref_np(q)
      up = 1.0 - 2.0 * (q[:, 4] ** 2 + q[:, 5] ** 2)
      zr = q[:, 2] - gh
      ok = (up > args.up_th) & (zr >= args.z_lo) & (zr <= args.z_hi)
      hold_u = np.where(ok, hold_u + ce, 0)
      best_hold = np.maximum(best_hold, hold_u)
      newly = (hold_u >= args.hold) & (first_at < 0)
      first_at = np.where(newly, i + 1, first_at)
  q = np.asarray(st.data.qpos)
  gh = z_ref_np(q)
  up = 1.0 - 2.0 * (q[:, 4] ** 2 + q[:, 5] ** 2)
  zr = q[:, 2] - gh
  tilt = np.array([max(abs(a), abs(b))
                   for a, b in (rpy_deg(q[i, 3:7]) for i in range(N))])
  stable = best_hold >= args.hold
  stood = first_at >= 0
  strict = stable & (tilt < 2.0)
  practical = stable & (tilt < 20.0)
  # "曾站住"不够: 站住又倒下的话, 调度器会在那个 0.5s 窗口里交还给走路策略,
  # 然后靠走路策略去救 —— 这正是要避免的。所以还要看**末态仍站着**。
  final_ok = (up > args.up_th) & (zr >= args.z_lo) & (zr <= args.z_hi)
  final_practical = final_ok & (tilt < 20.0)

  print(f"  trials={N}  steps={args.steps} ({args.steps*env.dt:.1f}s)  "
        f"判定: up_z>0.99 且相对高度∈[{args.z_lo},{args.z_hi}] "
        f"连续 {args.hold} 步 ({args.hold*env.dt:.2f}s); 每 {ce} 步结算一次")
  print(f"  站立成功 (严格 |tilt|<2°)   : {int(strict.sum())}/{N} = "
        f"{100*strict.mean():.1f}%")
  print(f"  站立成功 (实用 |tilt|<20°)  : {int(practical.sum())}/{N} = "
        f"{100*practical.mean():.1f}%")
  print(f"  至少站住过一次              : {int(stood.sum())}/{N} = "
        f"{100*stood.mean():.1f}%")
  print(f"  末态仍满足站住判据          : {int(final_ok.sum())}/{N} = "
        f"{100*final_ok.mean():.1f}%  (含倾角<20°: "
        f"{int(final_practical.sum())}/{N} = {100*final_practical.mean():.1f}%)")
  if stood.any():
    t_at = first_at[stood] * env.dt
    print(f"  到站时间 (成功者)           : 均值 {t_at.mean():.2f}s "
          f"(min {t_at.min():.2f}s)")
  print("  分初始姿态 (曾站住, 实用判据 |tilt|<20°):")
  for name, m in buckets.items():
    if m.sum():
      print(f"    {name:<22}: {int(practical[m].sum()):>4}/{int(m.sum()):>4} "
            f"= {100*practical[m].mean():5.1f}%")
  good = ~done_seen                      # 排掉被甩出地形/陷下去的样本
  print(f"  出界/陷落终止               : {int(done_seen.sum())}/{N} = "
        f"{100*done_seen.mean():.1f}%  (这些样本不进下面的统计)")
  print(f"  --- 末态分布 (n={int(good.sum())}) ---")
  if good.any():
    print(f"  末态 up_z   : 均值 {up[good].mean():+.3f}  中位 "
          f"{np.median(up[good]):+.3f}  |tilt|<20° 的比例 "
          f"{100*(tilt[good] < 20).mean():.0f}%  up_z>0.99 的比例 "
          f"{100*(up[good] > 0.99).mean():.0f}%")
    print(f"  末态相对高度: 均值 {zr[good].mean():+.3f}m  中位 "
          f"{np.median(zr[good]):+.3f}m  "
          f"p10={np.percentile(zr[good], 10):+.3f} "
          f"p90={np.percentile(zr[good], 90):+.3f}  "
          f"落在[0.20,0.34]的比例 {100*((zr[good] >= 0.20) & (zr[good] <= 0.34)).mean():.0f}%")
    near = (up[good] > 0.99) & (zr[good] >= 0.20) & (zr[good] <= 0.34)
    print(f"  接近站立(up_z>0.99 且高度 0.20~0.34): {int(near.sum())}/"
          f"{int(good.sum())} = {100*near.mean():.1f}%  "
          f"<- 比'严格站住 0.5s'宽一档, 用来看还差多远")
    print(f"  末态 |tilt| : 均值 {tilt[good].mean():.1f}°  中位 "
          f"{np.median(tilt[good]):.1f}°")
  print(f"[{time.time()-t0:5.1f}s] 完成")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
