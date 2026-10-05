#!/usr/bin/env python
"""v20 高帧率观测脚本: 同一个 Go1Walk 环境, 但把查看器循环的耗时项全部修掉。

为什么另起一个脚本 (实测归因, 2026-09-18)
------------------------------------------
`sim/view_go1.py` 看 v20 模型是 3 fps (step=373ms)。逐项拆开后, 373ms 里
**没有一项是物理本身贵**:

| 项 | 实测 | 结论 |
|---|---|---|
| 单环境一步物理 (jit + WARP_STAGED) | 13-15 ms | 物理只值这么多 |
| 离屏渲染 llvmpipe | 491 ms/帧 | ✗ -> 换 d3d12 后 **15.3 ms** (32x) |
| 离屏渲染 `GALLIUM_DRIVER=d3d12` (AMD 780M) | 15.3 ms | 硬件路径, 不吃 CPU |
| 策略前向 **eager 未 jit** (原脚本行为) | **11.0 ms** | jit 后 **0.08 ms** (137x) |
| 扫描点计算 (已 jit) | 3.5 ms | 可接受 |

本脚本相对 `view_go1.py` 的改动 (每项都有上面的实测支撑):

1. **策略前向 jit**。原脚本每帧 `policy(obs, rng)` 是 eager 调用, 每次都要走
   Python 层的 dispatch; 小网络 (128x128) 的 dispatch 开销远大于计算本身。
   jit 后 11.0ms -> 0.08ms。
2. **默认 `GALLIUM_DRIVER=d3d12`**。WSL 无 `/dev/dri`, Mesa 默认落 llvmpipe
   软件渲染: 既慢 (491ms/帧) 又吃满 16 核, 饿死 jax 的 kernel 分发线程
   (§15.4)。`/dev/dxg` + `d3d12_dri.so` 在本机可用, 走 D3D12 (AMD 780M)
   硬件路径后 15.3ms/帧且几乎不占 CPU。设 `GALLIUM_DRIVER=` 可回退 llvmpipe
   (此时自动退回 LP_NUM_THREADS=4 限流)。
3. **开窗口前预热到稳态**。查看器前 ~400 帧 step 会停在 373ms 不恢复; 本脚本
   在 `launch_passive` 之前把同样的循环跑 `--warmup` 帧 (含策略+step+扫描点),
   让编译与首帧开销都在"还没有窗口"时结清, 窗口一出现就是稳态帧率。

**没有采用的优化 (实测否决, 留作记录)**: 把 `naconmax` 从 65536 降到 512-2048
虽然能让步耗时从 14.9ms 降到 12.3ms (18%), 但固定动作序列 500 步的轨迹与默认
**从第 1 步就分叉** (最大偏差 1.7e-3, 随后混沌放大到 0.5m)。接触数组容量会
改变求解器里的求和顺序 -> 物理不再等价, 故不用。

用法:
  python sim/watch_v20_fast.py                    # 默认 v20 权重
  python sim/watch_v20_fast.py --pkl policies/go1_walk_policy.pkl
  python sim/watch_v20_fast.py --fast             # 不做实时节流
  python sim/watch_v20_fast.py --no_scan_points   # 省 3.5ms/帧
  GALLIUM_DRIVER= python sim/watch_v20_fast.py    # 强制 llvmpipe

键盘: W/S 调 vx, A/D 调 wz (每按一下 ±0.2), R 重置, T 切扫描点, Q/Esc 退出。
"""
import argparse
import os
import sys
import time

# --- GL 后端选择 (必须在 import mujoco 之前) ---
# WSL 无 /dev/dri -> Mesa 默认 llvmpipe 软件渲染 (491ms/帧, 且吃满 16 核 ->
# 饿死 jax 分发线程, 见 §15.4)。本机有 /dev/dxg + d3d12_dri.so, 走 D3D12 硬件
# 路径是 15.3ms/帧。用 setdefault: 用户显式设 GALLIUM_DRIVER 时不覆盖
# (设成空串可强制 llvmpipe 回退)。
os.environ.setdefault("GALLIUM_DRIVER", "d3d12")
# llvmpipe 的线程限流: 只有回退到软件渲染时才起作用 (d3d12 路径读不到该变量)。
# llvmpipe 只在首次初始化时读它, 所以必须在 import mujoco 之前设。
os.environ.setdefault("LP_NUM_THREADS", "4")
# --- 显存池上限 (必须在 import jax 之前设置!) ---
# XLA 默认预占 ~75% 显存 (8G 卡 -> ~6G)。本机实测 (2026-09-18):
#   单进程默认预占 -> 15.7 ms/帧 (64fps) 正常
#   两个进程 (例: 上一个查看器窗口没关) -> 显存 7674/8192 MiB, SM 时钟掉到
#   180 MHz, 其中一个进程 **step 暴涨到 307-550ms (2-3fps)**, 另一个 32ms
#   -> 表现为"查看器莫名只有 3fps", 与 CUDA graph / 地形 / 策略都无关。
# 0.25 实测: 单进程 15.7ms, 双进程各 27ms (37fps) —— 并发也不再塌方。
# 其它 19 个 sim 脚本本来就设了 0.3-0.55, 只有本脚本与 view_go1.py 漏了。
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.25")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import jax
import jax.numpy as jp
import mujoco
import mujoco.viewer
import numpy as np

from envs.go1_walk import Go1Walk, default_config
from sim import getup_keyframe as gk

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, "..")
DEFAULT_PKL = os.path.join(_ROOT, "policies", "go1_walk_policy.pkl")


# ---------------------------------------------------------------- 策略加载
# 与 view_go1.py 同源: pkl 不存网络配置, 维度与隐层都从参数形状反推, 否则
# 加载 v19/v20/v21 权重会 ScopeParamShapeError。
def _infer_layers(params):
  roots = [params]
  if isinstance(params, (tuple, list)) and len(params) > 1:
    roots = [params[1]]
  found = {}

  def walk(node):
    if isinstance(node, (tuple, list)):
      for v in node:
        walk(v)
      return
    if not hasattr(node, "items"):
      return
    for k, v in node.items():
      if (k.startswith("hidden_") and hasattr(v, "get") and "kernel" in v
          and hasattr(v["kernel"], "shape")):
        found[int(k.split("_")[1])] = int(v["kernel"].shape[1])
      walk(v)

  for r in roots:
    walk(r)
  return tuple(found[i] for i in sorted(found)) if found else (128, 128)


def _infer_dims(params):
  """返回 (state_dim, privileged_dim)。v20 是 91/162 (含高度扫描 35 + 相位 8)。"""
  def first_dim(node):
    best = {}

    def walk(n):
      if isinstance(n, (tuple, list)):
        for v in n:
          walk(v)
        return
      if not hasattr(n, "items"):
        return
      for k, v in n.items():
        if (k == "hidden_0" and hasattr(v, "get") and "kernel" in v
            and hasattr(v["kernel"], "shape")):
          best.setdefault("dim", int(v["kernel"].shape[0]))
        walk(v)

    walk(node)
    return best.get("dim")

  if isinstance(params, (tuple, list)) and len(params) >= 3:
    pol, val = first_dim(params[1]), first_dim(params[2])
  else:
    pol, val = first_dim(params), None
  return pol or 48, val or 119


def load_policy(pkl_path):
  import pickle
  from brax.training.agents.ppo import networks as ppo_networks
  with open(pkl_path, "rb") as f:
    params = pickle.load(f)
  layers = _infer_layers(params)
  n_state, n_priv = _infer_dims(params)
  network = ppo_networks.make_ppo_networks(
      observation_size={"state": (n_state,), "privileged_state": (n_priv,)},
      action_size=12,
      policy_hidden_layer_sizes=layers,
      value_hidden_layer_sizes=layers,
      policy_obs_key="state",
      value_obs_key="privileged_state",
      distribution_type="normal",
      activation=jax.nn.tanh,
  )
  inference = ppo_networks.make_inference_fn(network)(params, deterministic=True)
  return inference, (n_state, n_priv)


def active_gl_renderer():
  """读当前 GL 渲染器名, 用于确认 d3d12 是否真的生效。

  必须**自己建一个临时上下文**再查: 查看器的 GL 上下文在它自己的 C++ 线程里,
  主线程直接 glGetString 会因为没有 current context 而返回 None (曾误报 "?")。
  """
  try:
    import ctypes
    ctx = mujoco.GLContext(64, 64)
    ctx.make_current()
    gl = ctypes.CDLL("libGL.so.1")
    gl.glGetString.restype = ctypes.c_char_p
    gl.glGetString.argtypes = [ctypes.c_uint]
    v = gl.glGetString(0x1F01)          # GL_RENDERER
    return v.decode() if v else "(空)"
  except Exception as e:                # 探测失败不影响主流程
    return f"(取不到: {type(e).__name__})"


def print_spawn(env, state, seed):
  """打印出生点与**原地坡度** (2026-09-18 新增)。

  为什么必须打印: 观测脚本是单环境 + 固定 seed -> **每次启动都落在同一个点**。
  实测默认 seed=0 固定落在 (-4.17, +2.81) 的陡坡上: 地面高 0.2147m (全图前 5%),
  纵向坡度 25%, 左右脚地面高差 **3.5cm** —— 狗一出生就得应付这么大的横向高差,
  看起来"姿态歪", 而且每次重启都一样。训练侧不受影响 (768 环境各自不同 key)。
  现在 seed 默认随机, 并在启动时把这个点打出来, 免得再被"同一个山坡"误导。
  """
  try:
    q0 = np.asarray(state.data.qpos)
    w, x, y, z = q0[3:7]
    yaw = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    h0 = float(env.terrain_height(jp.asarray(q0[None, :2]))[0])
    py = np.array([-np.sin(yaw), np.cos(yaw)])          # 机体系 +y (左侧) 的世界方向
    hl = float(env.terrain_height(jp.asarray((q0[:2] + 0.25 * py)[None, :]))[0])
    hr = float(env.terrain_height(jp.asarray((q0[:2] - 0.25 * py)[None, :]))[0])
    print(f"[出生] seed={seed}  xy=({q0[0]:+.2f}, {q0[1]:+.2f})  地面高={h0:.4f}m  "
          f"躯干z={q0[2]:.3f}  yaw={np.degrees(yaw):+.1f}°", flush=True)
    print(f"       横向坡度={100 * (hl - hr) / 0.5:+.1f}%   "
          f"左右地面高差={hl - hr:+.4f}m (±0.25m 内)", flush=True)
  except Exception as e:
    print(f"[出生] 位置信息取不到: {type(e).__name__}", flush=True)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--pkl", type=str, default=DEFAULT_PKL)
  ap.add_argument("--cmd", type=float, default=0.5, help="vx 指令 (W/S 调)")
  ap.add_argument("--wz", type=float, default=0.0, help="wz 指令 (A/D 调)")
  ap.add_argument("--seed", type=int, default=-1,
                  help="随机出生的 seed。-1 (默认) = 每次启动随机取, 避免每次都"
                       "落在同一个陡坡上 (实测 seed=0 固定落在 (-4.17,+2.81)); "
                       "给具体整数则复现同一起点, 便于 A/B 对照")
  ap.add_argument("--fast", action="store_true",
                  help="关实时节流, 全速跑 (默认按 env.dt=20ms 节流 = 50fps 实时)")
  ap.add_argument("--warmup", type=int, default=150,
                  help="开窗口前预热的帧数 (默认 150; 见文件头第 3 条)")
  ap.add_argument("--no_scan_points", dest="scan_points", action="store_false",
                  help="不画 35 点高度扫描 (省 ~3.5ms/帧)")
  ap.set_defaults(scan_points=True)
  ap.add_argument("--no_random_init", action="store_true",
                  help="固定出生点 (默认随机, 与训练一致)")
  ap.add_argument("--overview", type=float, default=None, metavar="DIST",
                  help="俯瞰地形全貌 (相机距离, 米; 建议 7~9)")
  ap.add_argument("--shadowsize", type=int, default=512,
                  help="阴影贴图尺寸 (默认 512; d3d12 下不再是瓶颈, 但小一点更省)")
  # --- 摔倒处理 (2026-09-19): 默认**不再自动重置**, 改为关键帧起身 ---
  # 注意本查看器的地面是 hfield (地形场景), §15 的 kip 在 hfield 上成功率明显更低
  # (接触生成方式不同), 所以 getup 失败会回退到重置。详见 sim/probe_getup_mjx.py。
  ap.add_argument("--fall_action", choices=["getup", "getup_hold", "reset"],
                  default="getup",
                  help="摔倒后: getup=关键帧起身, 失败回退重置 (默认); "
                       "getup_hold=起身失败也躺那不动; reset=旧行为直接重置")
  ap.add_argument("--fall_z", type=float, default=0.16,
                  help="摔倒判据 1: 躯干离地高度低于此值 (米); 用相对地面高度, "
                       "绝对 z 在高台上永远不触发")
  ap.add_argument("--fall_debounce", type=int, default=18,
                  help="摔倒判据要连续满足这么多帧才触发 (去抖, 见 view_go1.py)")
  ap.add_argument("--fall_uz", type=float, default=0.30,
                  help="摔倒判据 2: 躯干 up 轴 z 投影低于此值 (0.45 = 倾斜约 63 度)")
  ap.add_argument("--getup_attempts", type=int, default=gk.DEFAULT_MAX_ATTEMPTS,
                  help="关键帧起身最多试几个 kip 方案")
  # --- 调度器方案: 学到的起身策略 (两个网络, 一份物理, 状态机路由) ---
  ap.add_argument("--getup_pkl", type=str, default=None,
                  help="学到的起身策略权重 (train/train_getup.py)。给了就用策略起身; "
                       "不给则退回关键帧序列。要求 obs 与走路策略同形")
  ap.add_argument("--getup_action", choices=["incremental", "absolute"],
                  default="incremental",
                  help="起身策略的动作语义。默认增量 (qpos[7:]+a*scale), 与 "
                       "envs/go1_getup.py 一致")
  ap.add_argument("--getup_action_scale", type=float, default=0.5,
                  help="增量动作缩放 (与训练 cfg.getup.action_scale 一致)")
  ap.add_argument("--getup_up_th", type=float, default=0.95,
                  help="交还走路策略的竖直度阈值 up_z。0.99 = 倾角<8.1 度 (旧硬编码, "
                       "实测学习到的策略稳定站在 ~13 度 -> 永远超时失败); "
                       "0.95 = 倾角<18.2 度 (默认, 与 sim/eval_getup.py --up_th 同口径)")
  ap.add_argument("--getup_zlo", type=float, default=0.20,
                  help="交还走路策略的躯干相对地面高度下界 (站直约 0.277)")
  ap.add_argument("--getup_zhi", type=float, default=0.34,
                  help="交还走路策略的躯干相对地面高度上界")
  ap.add_argument("--getup_hold", type=int, default=25,
                  help="起身成功后还要保持多少帧才交还 (25 = 0.5s, 防乒乓)")
  ap.add_argument("--getup_timeout", type=int, default=400,
                  help="起身最多等多少帧 (400 = 8s), 超时按失败处理")
  ap.add_argument("--getup_cooldown", type=int, default=25,
                  help="交接冷却 (25 帧 = 0.5s)")
  args = ap.parse_args()

  pkl = args.pkl
  if not os.path.isabs(pkl):
    pkl = os.path.join(_ROOT, pkl)
  if not os.path.exists(pkl):
    print(f"找不到权重: {pkl}", flush=True)
    return 1

  t_start = time.time()
  print(f"[GL] GALLIUM_DRIVER={os.environ.get('GALLIUM_DRIVER') or '(空=llvmpipe)'}  "
        f"LP_NUM_THREADS={os.environ.get('LP_NUM_THREADS')}", flush=True)
  # 在这里探测 (开窗口前, 主线程可以自己建上下文); 确认 d3d12 真的生效,
  # 而不是静默回退到 llvmpipe —— 后者会慢 32 倍并且抢 CPU 饿死 jax 分发线程。
  print(f"[GL] 实际渲染器: {active_gl_renderer()}", flush=True)

  policy, dims = load_policy(pkl)
  print(f"[策略] {os.path.basename(pkl)}  obs state={dims[0]} priv={dims[1]}",
        flush=True)

  # ---- 调度器方案: 第二个网络 (起身策略)。obs 必须与走路同形 ----
  getup_policy = None
  if args.getup_pkl:
    gp = args.getup_pkl
    if not os.path.isabs(gp):
      gp = os.path.join(_ROOT, gp)
    if not os.path.exists(gp):
      print(f"找不到起身权重: {gp}", flush=True)
      return 1
    getup_policy, gdims = load_policy(gp)
    if tuple(gdims) != tuple(dims):
      print(f"[错误] 起身策略 obs {tuple(gdims)} != 走路策略 {tuple(dims)}; "
            f"调度器要求两个策略吃同一个 obs 向量", flush=True)
      return 1
    print(f"[起身策略] {os.path.basename(gp)}  obs state={gdims[0]} "
          f"priv={gdims[1]} (与走路同形 OK)", flush=True)

  # ---- 环境: 与 v20 训练严格对齐 ----
  cfg = default_config()
  cfg.foot_soft_contact = False            # v5+ 硬接触
  cfg.terrain = True
  cfg.height_scan.enable = True
  cfg.random_init.enable = not args.no_random_init
  cfg.command_config.frame = "body"        # v17+ 机体系指令
  cfg.healthy_roll_range = 17.5            # v17+ 终止倾角
  cfg.healthy_pitch_range = 17.5
  # v20: 相位奖励打开 -> obs 多 8 维 (cos,sin x 4 足)。不开则 91 维策略形状不符。
  cfg.reward_config.scales.feet_phase = 2.0
  # v20 训练侧的联动 (train_go1.py): feet_phase 与 feet_height 目标必须一致
  cfg.reward_config.max_foot_height = cfg.reward_config.gait_swing_height
  # **交互式查看器必须显式设 WARP_STAGED** (2026-09-18 修复):
  # 不设 -> config.graph_mode=None -> MJX 默认 WARP (按 buffer 地址缓存 CUDA graph)。
  # 本循环每帧新建 command 数组 (见 frame() 的注释) -> 地址每次都变 -> 每帧重新
  # capture -> 实测 step=302ms = 3fps, 且跑几千帧会崩 (Warp error: unknown stream)。
  # 实测对照 (envs/go1_walk.py 的 graph_mode 注释): WARP=185.5ms vs
  # WARP_STAGED=16.3ms。view_go1.py 早就默认 WARP_STAGED, 本脚本漏了。
  # 规则见 观测器脚本撰写指南.md 陷阱 2 / 检查清单最后一条。
  cfg.graph_mode = "WARP_STAGED"
  env = Go1Walk(cfg)
  env.mj_model.vis.quality.shadowsize = args.shadowsize

  env_dims = (env.observation_size["state"][0],
              env.observation_size["privileged_state"][0])
  if tuple(dims) != tuple(env_dims):
    print(f"[错误] 策略输入 {tuple(dims)} != 环境 obs {tuple(env_dims)}",
          flush=True)
    return 1
  print(f"[环境] obs={env_dims[0]}/{env_dims[1]}  "
        f"terrain+scan+相位8维  随机出生={cfg.random_init.enable}  "
        f"dt={env.dt}s ({1/env.dt:.0f}Hz)", flush=True)

  # ---- 关阴影/反射/天空盒 (软件渲染时代的大头; d3d12 下收益小但无害) ----
  _flags = mujoco.mjtRndFlag
  env.mj_model.light_castshadow[:] = 0

  # ---- JIT: 策略前向 + 环境 step/reset ----
  # 策略必须 jit —— 原查看器是 eager 调用, 实测 11.0ms vs 0.08ms。
  _dummy_rng = jp.zeros((2,), dtype=jp.uint32)
  act_jit = jax.jit(lambda obs: policy(obs, _dummy_rng)[0])
  reset_jit = jax.jit(env.reset)
  step_jit = jax.jit(env.step, donate_argnums=0)

  # ---- 关键帧起身 (2026-09-19): 摔倒不再自动重置 ----
  # 动作语义 = ctrl 目标 (env 里 motor_targets=clip(action, ctrlrange)), 所以状态机
  # 输出的 12 维向量可以直接当 action。地面是 hfield -> 用 hfield 方案表。
  home_jp = jp.asarray(np.asarray(env._init_q)[7:], jp.float32)
  getup_jit = jax.jit(gk.fsm_step)
  hfield_floor = bool(getattr(env, "_hfield_ok", False))

  def ground_h(q):
    """躯干正下方地面高度 (米)。hfield 场景必须用相对高度做摔倒判据。"""
    if not hfield_floor:
      return 0.0
    return float(env.terrain_height(jp.asarray(q[None, :2]))[0])

  def z_rel_of(q):
    return q[2] - ground_h(q)

  # 起身策略推理 (obs 与走路同一个 key -> 调度器只需换 act 来源)
  getup_pol_jit = None
  if getup_policy is not None:
    getup_pol_jit = jax.jit(
        lambda obs: getup_policy(obs, jp.zeros((2,), jp.uint32))[0])
  print(f"[摔倒] 处理={args.fall_action} | 起身方式="
        f"{'学到的策略' if getup_pol_jit is not None else '关键帧序列'}"
        f" | 交还判据 up_z>{args.getup_up_th} 且 相对高度∈[{args.getup_zlo},"
        f"{args.getup_zhi}] 保持 {args.getup_hold} 帧", flush=True)

  # ---- 扫描点可视化: 三步 (采样点 xy / 地面高 / 特征) jit 成一个 kernel ----
  # eager 逐条调用会变成每帧 100+ 次 kernel launch (实测 51.6ms/帧 -> 0.32ms)。
  n_scan = env._scan_n if (env._hfield_ok and cfg.height_scan.enable) else 0
  show_scan = bool(args.scan_points) and n_scan > 0
  _hs = cfg.height_scan
  _scan_xy_j = jp.asarray(np.asarray(env._scan_xy), dtype=jp.float32)
  _scan_lim = max(_hs.clip_m * _hs.scale, 1e-6)

  if show_scan:
    @jax.jit
    def scan_viz(data, qpos):
      feat = env.height_scan_features(data, rng=None)
      q = qpos[3:7] / jp.maximum(jp.linalg.norm(qpos[3:7]), 1e-8)
      w, x, y, z = q
      yaw = jp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
      c, s = jp.cos(yaw), jp.sin(yaw)
      xy = jp.stack([c * _scan_xy_j[:, 0] - s * _scan_xy_j[:, 1] + qpos[0],
                     s * _scan_xy_j[:, 0] + c * _scan_xy_j[:, 1] + qpos[1]], -1)
      return xy, env.terrain_height(xy), feat

    def scan_rgba(f):
      t = float(np.clip(f / _scan_lim, -1.0, 1.0))
      return np.array([0.15 + 0.75 * max(-t, 0.0),
                       0.25 + 0.55 * (1.0 - abs(t)),
                       0.15 + 0.75 * max(t, 0.0), 0.9], dtype=np.float32)

    def draw_scan(scn, qpos, data):
      xy, gz, feats = scan_viz(data, qpos)
      xy, gz, feats = np.asarray(xy), np.asarray(gz), np.asarray(feats)
      base = np.asarray(qpos[:3], dtype=float)
      mat = np.eye(3).reshape(9)
      scn.ngeom = 0
      for i in range(n_scan):
        if scn.ngeom >= scn.maxgeom:
          break
        pos = np.array([xy[i, 0], xy[i, 1], gz[i]])
        mujoco.mjv_initGeom(
            scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([0.012, 0.0, 0.0]), pos, mat, scan_rgba(feats[i]))
        scn.ngeom += 1
        if scn.ngeom < scn.maxgeom:
          mujoco.mjv_initGeom(
              scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_LINE,
              np.zeros(3), np.zeros(3), mat,
              np.array([0.7, 0.7, 0.2, 0.30], dtype=np.float32))
          mujoco.mjv_connector(scn.geoms[scn.ngeom],
                               mujoco.mjtGeom.mjGEOM_LINE, 1.0,
                               np.array([base[0], base[1], base[2]]), pos)
          scn.ngeom += 1
      return feats

  # ---- 每帧动作 (预热与主循环共用同一份, 保证预热覆盖真实路径) ----
  command = np.array([args.cmd, 0.0, args.wz], dtype=np.float32)
  d_view = mujoco.MjData(env.mj_model)
  seed = args.seed if args.seed >= 0 else int(time.time() * 1000) % (2**31 - 1)
  rng = jax.random.PRNGKey(seed)
  state = reset_jit(rng)
  print_spawn(env, state, seed)
  getup_on = [False]
  getup_fsm = None
  getup_f0 = 0
  force_getup = [False]
  fall_count = [0]                   # 摔倒判据连续满足帧数 (去抖)
  getup_hold_cnt = [0]               # 起身后"站住"连续帧数 (判交还)
  getup_best = [0.0]                 # 本次起身到过的最大 up_z (诊断)
  getup_best_ok = [0.0]              # 高度带内到过的最大 up_z (诊断)
  cooldown = [0]                     # 交接冷却, 防乒乓

  def frame(state, d_view, act=None):
    """跑一帧: 策略(或给定 act) -> step -> 同步 -> 灌 CPU MuJoCo。

    act 非 None 时**跳过策略**, 直接把它当动作 —— 关键帧起身用它接管控制。
    返回 (state, qpos, qvel, 分项)。
    """
    # 每帧重建 command 数组: step_jit 的 donate_argnums=0 会把 state.info 里的
    # 数组合并捐赠, 复用同一数组会被 jax 判为 deleted。(3,) 重建开销可忽略。
    t0 = time.perf_counter()
    state.info["command"] = jp.asarray(np.array(command, dtype=np.float32))
    a = act_jit(state.obs["state"]) if act is None else act
    state = step_jit(state, a)
    # np.asarray 同时完成设备同步 -> step 的真实耗时在这里结算
    qpos = np.asarray(state.data.qpos)
    qvel = np.asarray(state.data.qvel)
    t_step = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    d_view.qpos[:] = qpos
    d_view.qvel[:] = qvel
    mujoco.mj_forward(env.mj_model, d_view)
    t_copy = (time.perf_counter() - t1) * 1000.0
    return state, qpos, qvel, t_step, t_copy

  # ---- 开窗口前预热: 把编译 + 首帧暂态在"没有窗口"时结清 ----
  # 查看器前 ~400 帧 step=373ms 不恢复; 预热用与主循环完全相同的调用序列,
  # 让正式循环从一开始就是稳态。
  print(f"[预热] {args.warmup} 帧 (编译 + 首帧暂态, 完成后才开窗口)...",
        flush=True)
  for _ in range(max(args.warmup, 0)):
    state, qpos, qvel, _, _ = frame(state, d_view)
  if show_scan:
    jax.block_until_ready(scan_viz(state.data, state.data.qpos))
  # 不打印"无窗口稳态 step": 实测它比窗口内偏高约 2 倍 (开窗口前 24ms vs
  # 窗口内 13ms), 曾据此误判"窗口外反而慢"。该数字受开窗口前的驱动/排队状态
  # 影响, **不能预测窗口内帧率**。真正的判据是主循环的 fps 分项。
  print(f"[{time.time()-t_start:5.1f}s] 预热完成, 开窗口", flush=True)

  with mujoco.viewer.launch_passive(env.mj_model, d_view) as v:
    v.user_scn.flags[_flags.mjRND_REFLECTION] = 0
    v.user_scn.flags[_flags.mjRND_SKYBOX] = 0
    v.user_scn.flags[_flags.mjRND_SHADOW] = 0
    if args.overview is not None:
      cam = v.cam
      cam.type = mujoco.mjtCamera.mjCAMERA_FREE
      cam.lookat[:] = [0.0, 0.0, 0.25]
      cam.distance = float(args.overview)
      cam.azimuth = 135.0
      cam.elevation = -35.0

    # 键盘 (pynput 在窗口就绪后再起, 避免与 GL 初始化争线程)
    tap = set()
    keys_down = {}
    reset_req = [False]
    scan_on = [True]
    try:
      from pynput import keyboard

      def _norm(k):
        if isinstance(k, keyboard.KeyCode) and k.char is not None:
          return k.char.lower()
        return k

      def on_press(k):
        kk = _norm(k)
        was = keys_down.get(kk, False)
        keys_down[kk] = True
        if not was and kk in ("w", "s", "a", "d", "t", "g"):
          tap.add(kk)                 # 边沿触发, 滤掉 OS autorepeat
        if kk == "r":
          reset_req[0] = True

      keyboard.Listener(on_press=on_press,
                        on_release=lambda k: keys_down.__setitem__(_norm(k), False)
                        ).start()
      print("[键盘] W/S vx±0.2 | A/D wz±0.2 | R 重置"
            + (" | T 扫描点" if show_scan else "") + " | Q/Esc 退出", flush=True)
    except Exception as e:
      print(f"[键盘] 起不来 ({type(e).__name__}); 用命令行 --cmd/--wz", flush=True)

    print(f"[{time.time()-t_start:5.1f}s] 就绪, 开始 "
          f"(目标 {1/env.dt:.0f}fps 实时; {'全速' if args.fast else '实时节流'})",
          flush=True)

    ema = {"step": 0.0, "copy": 0.0, "sync": 0.0, "scan": 0.0, "total": 0.0}
    alpha, fi, t_prev = 0.1, 0, time.perf_counter()

    def ema_u(k, x):
      ema[k] = x if ema[k] == 0.0 else (1 - alpha) * ema[k] + alpha * x

    while v.is_running():
      sim = v._get_sim()
      if sim is not None and not sim.run:      # 暂停按钮
        v.sync()
        time.sleep(0.01)
        continue

      f0 = time.perf_counter()
      # 单发按键
      if tap:
        if "w" in tap:
          command[0] = min(command[0] + 0.2, 1.5)
        if "s" in tap:
          command[0] = max(command[0] - 0.2, 0.0)
        if "a" in tap:
          command[2] = min(command[2] + 0.2, 1.5)
        if "d" in tap:
          command[2] = max(command[2] - 0.2, -1.5)
        if "t" in tap:
          scan_on[0] = not scan_on[0]
        if "g" in tap:
          force_getup[0] = True          # G 键: 手动触发一次起身
        tap.clear()
      if reset_req[0]:
        reset_req[0] = False
        rng, key = jax.random.split(rng)
        state = reset_jit(key)
        getup_on[0] = False

      # ---- 摔倒判定: 默认不再自动重置, 交给起身 (策略或关键帧) ----
      q_now = np.asarray(state.data.qpos)
      if not getup_on[0] and cooldown[0] <= 0:
        upz = gk.up_z_from_quat(q_now[3:7])
        zrel = z_rel_of(q_now)
        fallen = (zrel < args.fall_z) or (upz < args.fall_uz)
        fall_count[0] = fall_count[0] + 1 if fallen else 0
        armed = fall_count[0] >= args.fall_debounce
        if force_getup[0] or (armed and args.fall_action != "reset"):
          fall_count[0] = 0
          getup_on[0] = True
          getup_hold_cnt[0] = 0
          getup_best[0] = 0.0
          getup_best_ok[0] = 0.0
          getup_f0 = fi
          if getup_pol_jit is None:
            getup_fsm = gk.init_fsm(hfield=hfield_floor,
                                    max_attempts=args.getup_attempts)
            how = (f"关键帧 方案表="
                   f"{'hfield' if hfield_floor else 'plane'} "
                   f"x{args.getup_attempts}")
          else:
            how = "学到的策略"
          print(f"   [起身] 触发 (up_z={upz:+.2f} 离地={zrel:.3f}m) -> {how}",
                flush=True)
        elif armed:
          fall_count[0] = 0
          rng, key = jax.random.split(rng)
          state = reset_jit(key)
          cooldown[0] = args.getup_cooldown
      force_getup[0] = False

      st_now = jp.int32(0)
      if getup_on[0]:
        if getup_pol_jit is not None:
          act = getup_pol_jit(state.obs["state"])
          if args.getup_action == "incremental":
            # 与 envs/go1_getup.py 的 step 同一个变换
            act = state.data.qpos[7:] + act * args.getup_action_scale
        else:
          act, st_now, rj_g, getup_fsm = getup_jit(
              getup_fsm, state.data.qpos, state.data.qvel, home_jp)
          if bool(rj_g):
            state = gk.apply_joint_reset(state, home_jp)
        state, qpos, qvel, t_step, t_copy = frame(state, d_view, act)
      else:
        state, qpos, qvel, t_step, t_copy = frame(state, d_view)

      scan_feats = None
      if show_scan and scan_on[0]:
        ts = time.perf_counter()
        scan_feats = draw_scan(v.user_scn, qpos, state.data)
        ema_u("scan", (time.perf_counter() - ts) * 1000.0)
      elif show_scan:
        v.user_scn.ngeom = 0

      # ---- 起身收尾 ----
      if getup_on[0] and getup_pol_jit is not None:
        # 学到的策略: 必须"站住"才交还 (只看 up_z 不够, 起身中途会经过直立姿态)
        upz = gk.up_z_from_quat(qpos[3:7])
        zr = z_rel_of(qpos)
        ok = ((upz > args.getup_up_th)
              and (args.getup_zlo <= zr <= args.getup_zhi))
        if (upz > getup_best[0]) or (getup_best[0] == 0.0):
          getup_best[0] = upz
        if (args.getup_zlo <= zr <= args.getup_zhi) and upz > getup_best_ok[0]:
          getup_best_ok[0] = upz
        getup_hold_cnt[0] = getup_hold_cnt[0] + 1 if ok else 0
        if getup_hold_cnt[0] >= args.getup_hold:
          getup_on[0] = False
          cooldown[0] = args.getup_cooldown
          # 交还"去污染": 走路策略 obs 里的 last_act 在起身期间被推到很远,
          # 不清零走路策略会看到没见过的输入 -> 抽搐 (见 实验记录 §29.25)
          state.info["last_act"] = jp.zeros(12)
          print(f"   [起身] 成功 OK 站住 {getup_hold_cnt[0]} 帧 "
                f"(up_z={upz:+.3f} 离地={zr:.3f}m), 交还走路策略 "
                f"[{fi - getup_f0} 帧]", flush=True)
        elif (fi - getup_f0) >= args.getup_timeout:
          getup_on[0] = False
          cooldown[0] = args.getup_cooldown
          print(f"   [起身] 失败 超时 {args.getup_timeout} 帧 "
                f"(当前 up_z={upz:+.3f} 离地={zr:.3f}m; "
                f"最好 up_z={getup_best[0]:+.3f}, 高度带内最好 "
                f"{getup_best_ok[0]:+.3f}, 阈值 {args.getup_up_th})", flush=True)
          if args.fall_action == "getup":
            rng, key = jax.random.split(rng)
            state = reset_jit(key)
            print("   [起身] 回退到重置", flush=True)
      elif getup_on[0] and int(st_now) != 0:
        ok = int(st_now) == gk.STATUS_OK
        getup_on[0] = False
        cooldown[0] = args.getup_cooldown
        print(f"   [起身] {'成功 OK' if ok else '失败'} "
              f"({gk.phase_name(getup_fsm)}, {fi - getup_f0} 帧)", flush=True)
        if not ok and args.fall_action == "getup":
          rng, key = jax.random.split(rng)
          state = reset_jit(key)
          print("   [起身] 回退到重置", flush=True)
      if cooldown[0] > 0:
        cooldown[0] -= 1

      t2 = time.perf_counter()
      v.sync()
      ema_u("sync", (time.perf_counter() - t2) * 1000.0)
      ema_u("step", t_step)
      ema_u("copy", t_copy)
      ema_u("total", (time.perf_counter() - f0) * 1000.0)
      fi += 1
      if fi % 25 == 0:
        print(f"[watch] frame {fi}: fps={1000.0/max(ema['total'],1e-6):.0f}  "
              f"step={ema['step']:.1f} copy={ema['copy']:.1f} "
              f"sync={ema['sync']:.1f} "
              + (f"scan={ema['scan']:.1f} " if show_scan else "")
              + f"total={ema['total']:.1f} ms", flush=True)

      v.overlay = {
          "title": f"Go1 v20 高帧率 ({os.path.basename(pkl)})",
          "command": f"指令 vx={command[0]:+.2f} wz={command[2]:+.2f}",
          "speed": f"实际 vx={qvel[0]:+.2f} vy={qvel[1]:+.2f} "
                   f"wz={qvel[5]:+.2f} z={qpos[2]:.2f}",
          "perf": f"fps={1000.0/max(ema['total'],1e-6):.0f}  "
                  f"step={ema['step']:.1f}ms copy={ema['copy']:.1f}ms "
                  f"sync={ema['sync']:.1f}ms"
                  + (f" scan={ema['scan']:.1f}ms" if show_scan else ""),
          "help": "W/S vx±0.2 | A/D wz±0.2 | R 重置 | G 手动起身"
                  + (" | T 扫描点" if show_scan else ""),
      }
      if getup_on[0]:
        if getup_pol_jit is not None:
          v.overlay["getup"] = (
              f"起身中 (策略): 站住 {getup_hold_cnt[0]}/{args.getup_hold} 帧, "
              f"{fi - getup_f0} 帧")
        else:
          v.overlay["getup"] = (
              f"起身中: {gk.phase_name(getup_fsm)} "
              f"(第 {int(np.asarray(getup_fsm['attempt'])) + 1} 个方案, "
              f"{fi - getup_f0} 帧)")
      if scan_feats is not None:
        v.overlay["scan"] = (
            f"高度特征({n_scan}点): min={scan_feats.min():+.2f} "
            f"max={scan_feats.max():+.2f} mean={scan_feats.mean():+.2f} "
            f"(红=上坡, 蓝=下陷)")

      if not args.fast:
        slack = env.dt - (time.perf_counter() - f0)
        if slack > 0:
          time.sleep(slack)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
