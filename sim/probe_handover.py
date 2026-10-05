#!/usr/bin/env python
"""探针: 离线复现"起身站住 -> 交还走路"的瞬间, 用数据决定交还方案 (不再盲改查看器)。

现场问题 (用户 2026-09-21): 查看器里起身成功、交还走路策略之后
"抽搐 -> 摔倒 -> 再起身 -> 再抽搐"; 查看器上一轮加的"交还时清零
info['last_act']"被怀疑有错。

为什么必须离线: 查看器里一帧只能看到现象, 对照不了方案。这里把交还**瞬间冻结**
(同一批终态、同一批随机密钥), 在上面跑多种交还处理, 用**可比的标量**说话:
    dact  = max_i |act[t,i] - act[t-1,i]|          (ctrl 跳变幅度)
    jerk  = max_i |act[t] - 2act[t-1] + act[t-2]|  (ctrl 的"抽搐"程度)
    fell  = 连续 fall_debounce 帧满足查看器的摔倒判据
    stand = 末态仍满足查看器的交还判据

走路策略的 obs (v23, 91 维) 是**单帧**: base 48 (含 last_act) + height_scan 35 +
相位 8。last_act 占 base 的 [36:48], 语义 = **上一步的 ctrl 目标**(绝对位置目标,
action_scale=1.0)。起身用锚定动作时它能到 default +- 4 rad, 走路时只在 +-1.5 附近。

两段式 (一个进程只建一个 env; 两个 env 同进程会撞 warp 的 CUDA-graph 地址缓存):
  stage=spawn: 建 Go1GetupV2, 跑起身策略到满足查看器判据, 把那一帧状态 dump 成 npz
  stage=walk : 建 Go1Walk (查看器同款 cfg), 从 npz 的终态出发跑走路策略, 对照方案

用法:
  python sim/probe_handover.py --stage spawn
  python sim/probe_handover.py --stage spawn --up_th 0.97 --dump sim/ho097.npz
  python sim/probe_handover.py --stage walk --cmd 0.5
  python sim/probe_handover.py --stage walk --cmd 0.0
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
from mujoco import mjx

from sim.common import load_policy

HERE = os.path.dirname(os.path.abspath(__file__))

# 交还方案表: name -> (last_act 取法, 混合帧数)
#   fresh = 走路策略从它自己的出生点出发 (原生基准, 不是交还)
#   ctrl  = last_act 保留真值 (起身最后的目标) —— "什么都不做"
#   zero  = last_act 清零并**重算 obs** (查看器想做的事, 做对了)
#   warm  = last_act 换成不动点 a* = pi(obs(last_act=a*)) (自洽的首帧动作)
#   hold  = 不交还, 一直保持起身最后的目标 (判断终态本身稳不稳)
VARIANTS = [
    ("ref_stand",     "fresh", 0),
    ("hold_getup",    "hold",  0),
    ("truth",         "ctrl",  0),
    ("zero",          "zero",  0),
    ("warm",          "warm",  0),
    ("blend10",       "ctrl",  10),
    ("blend30",       "ctrl",  30),
    ("zero_blend10",  "zero",  10),
    ("zero_blend30",  "zero",  30),
    ("warm_blend30",  "warm",  30),
]


def up_z(q):
  return 1.0 - 2.0 * (q[..., 4] ** 2 + q[..., 5] ** 2)


def tilt_deg(up):
  return np.degrees(np.arccos(np.clip(up, -1.0, 1.0)))


def st4(v):
  """均值/中位/p10/p90 一行。"""
  v = np.asarray(v, np.float64)
  return ("均 " + format(v.mean(), ".3f") + " 中 " + format(np.median(v), ".3f")
          + " p10 " + format(np.percentile(v, 10), ".3f")
          + " p90 " + format(np.percentile(v, 90), ".3f"))


def jerk_of(A):
  j = np.zeros(A.shape[:2], np.float32)
  j[2:] = np.abs(A[2:] - 2.0 * A[1:-1] + A[:-2]).max(2)
  return j


def dact_of(A):
  d = np.zeros(A.shape[:2], np.float32)
  d[1:] = np.abs(A[1:] - A[:-1]).max(2)
  return d


def fall_stats(U, Z, fall_z, fall_uz, deb):
  F, N = U.shape
  bad = (Z < fall_z) | (U < fall_uz)
  cnt = np.zeros(N, np.int32)
  fell = np.zeros(N, bool)
  tfall = np.full(N, -1, np.int32)
  for t in range(F):
    cnt = np.where(bad[t], cnt + 1, 0)
    new = (cnt >= deb) & (~fell)
    tfall = np.where(new, t, tfall)
    fell |= new
  return fell, tfall


# ------------------------------------------------------------------ stage spawn
def stage_spawn(args):
  from envs.go1_getup_v2 import Go1GetupV2, default_config as cfg_fn
  cfg = cfg_fn()
  cfg.getup2.action_clip = args.getup2_action_clip
  cfg.getup2.hold_steps = 10 ** 9      # 关掉稀疏奖金 (与本探针无关)
  env = Go1GetupV2(cfg)
  pol, ns = load_policy(args.getup_pkl)
  ne = env.observation_size["state"][0]
  print("[spawn] 起身策略 obs=" + str(ns) + "  env obs=" + str(ne)
        + "  action_clip=" + str(cfg.getup2.action_clip)
        + "  scale=" + str(cfg.getup2.action_scale)
        + "  判据 up_z>" + str(args.up_th) + " z_rel in ["
        + str(args.z_lo) + "," + str(args.z_hi) + "] 连续 " + str(args.hold) + " 步")
  if ns != ne:
    print("[错误] 起身策略维度与 getup_v2 环境不符")
    return 1
  N = args.envs
  reset = jax.jit(jax.vmap(env.reset))
  step = jax.jit(jax.vmap(env.step))
  act_fn = jax.jit(jax.vmap(lambda o: pol(o, jp.zeros((2,), jp.uint32))[0]))
  st = reset(jax.random.split(jax.random.PRNGKey(args.seed), N))
  st = jax.block_until_ready(st)
  hfield = bool(env._hfield_ok)

  def ground(xy):
    if hfield:
      return np.asarray(env.terrain_height(jp.asarray(xy)))
    return np.zeros(len(xy), np.float32)

  q0 = np.asarray(st.data.qpos)
  up0 = up_z(q0)
  Bq = np.zeros((N, 19), np.float32)
  Bv = np.zeros((N, 18), np.float32)
  Bc = np.zeros((N, 12), np.float32)
  Baf = np.zeros((N, 12), np.float32)
  Braw = np.zeros((N, 12), np.float32)
  upf = np.zeros(N, np.float32)
  zrf = np.zeros(N, np.float32)
  atf = np.full(N, -1, np.int32)
  frozen = np.zeros(N, bool)
  hold = np.zeros(N, np.int32)
  t0 = time.time()
  never_ok = 0
  for i in range(args.steps):
    st = step(st, act_fn(st.obs["state"]))
    q = np.asarray(st.data.qpos)
    qv = np.asarray(st.data.qvel)
    u = up_z(q)
    zr = q[:, 2] - ground(q[:, :2])
    ok = (u > args.up_th) & (zr >= args.z_lo) & (zr <= args.z_hi)
    hold = np.where(ok, hold + 1, 0)
    new = (hold >= args.hold) & (~frozen)
    if new.any():
      idx = np.where(new)[0]
      Bq[idx] = q[idx]
      Bv[idx] = qv[idx]
      Bc[idx] = np.asarray(st.data.ctrl)[idx]
      Baf[idx] = np.asarray(st.info["prev_a_filt"])[idx]
      Braw[idx] = np.asarray(st.info["last_raw_act"])[idx]
      upf[idx] = u[idx]
      zrf[idx] = zr[idx]
      atf[idx] = i + 1
      frozen[idx] = True
    if frozen.all():
      break
  nf = int(frozen.sum())
  never_ok = N - nf
  print("[spawn] " + str(nf) + "/" + str(N) + " 个 world 在 " + str(args.steps)
        + " 步内满足判据 = " + format(100.0 * nf / N, ".1f") + "%  (耗时 "
        + format(time.time() - t0, ".1f") + "s)")
  if nf:
    afq = np.abs(Baf).max(1)
    cq = np.abs(Bc - Bq[:, 7:]).max(1)
    jv = np.abs(Bv[:, 6:]).max(1)
    bv = np.abs(Bv[:, :6]).max(1)
    print("  --- 交还瞬间的终态诊断 (只看冻结的 " + str(nf) + " 个) ---")
    print("  " + "up_z".ljust(16) + st4(upf[frozen])
          + "   倾角均 " + format(tilt_deg(upf[frozen]).mean(), ".1f") + " 度")
    print("  " + "z_rel".ljust(16) + st4(zrf[frozen]))
    print("  " + "max|a_filt|".ljust(16) + st4(afq[frozen])
          + "   走路侧 |last_act| 量级约 1.5")
    print("  " + "max|ctrl-qpos|".ljust(16) + st4(cq[frozen]))
    print("  " + "max|qvel_joint|".ljust(16) + st4(jv[frozen]))
    print("  " + "max|qvel_base|".ljust(16) + st4(bv[frozen]))
    print("  " + "到站帧数".ljust(16) + st4(atf[frozen]))
    print("  关节自由度超程比例 (max|a_filt|>3): "
          + format(100.0 * (afq[frozen] > 3.0).mean(), ".1f") + "%")
    print("  高度带内 up_z<=0.97 (倾角>14 度) 的比例: "
          + format(100.0 * (upf[frozen] <= 0.97).mean(), ".1f") + "%")
  np.savez(args.dump, qpos=Bq, qvel=Bv, ctrl=Bc, a_filt=Baf, raw=Braw,
           up0=up0, up=upf, zr=zrf, at=atf, frozen=frozen,
           default_pose=np.asarray(env._default_pose),
           action_scale=np.float32(cfg.getup2.action_scale),
           up_th=np.float32(args.up_th), hold=np.int32(args.hold),
           z_lo=np.float32(args.z_lo), z_hi=np.float32(args.z_hi))
  print("[spawn] 已写出 " + args.dump)
  return 0


# ------------------------------------------------------------------- stage walk
def build_walk_cfg(ns, njmax=256):
  """查看器同款 cfg。

  njmax 默认 **256** —— 这是 go1_walk.default_config() 的值, 也是训练与查看器实际
  用的值 (getup_v2 没有覆盖它)。mujoco_warp 的 njmax 是 **per-world 约束行数上限**,
  而 hfield 地形上摔倒时一个 world 实测要 ~1300 行 -> 严重截断。
  """
  from envs.go1_walk import default_config
  cfg = default_config()
  if ns > 48:
    cfg.terrain = True
    cfg.height_scan.enable = True
    cfg.command_config.frame = "body"
    cfg.healthy_roll_range = 17.5
    cfg.healthy_pitch_range = 17.5
  if ns > 83:
    cfg.reward_config.scales.feet_phase = 2.0
  cfg.obs_noise.enable = False
  cfg.push_config.enable = False
  cfg.njmax = njmax
  return cfg


def stage_walk(args):
  from envs.go1_walk import Go1Walk
  d = np.load(args.dump)
  keep = np.where(d["frozen"])[0]
  if len(keep) == 0:
    print("[错误] dump 里没有满足判据的样本")
    return 1
  pol, ns = load_policy(args.walk_pkl)
  cfg = build_walk_cfg(ns, args.njmax)
  env = Go1Walk(cfg)
  ne = env.observation_size["state"][0]
  N = len(keep)
  print("[walk] 走路策略 obs=" + str(ns) + "  env obs=" + str(ne)
        + "  cmd vx=" + str(args.cmd) + "  终态样本 " + str(N) + " 个"
        + "  帧数 " + str(args.frames))
  if ns != ne:
    print("[错误] 走路策略维度与 env 不符")
    return 1
  reset = jax.jit(jax.vmap(env.reset))
  step_jit = jax.jit(jax.vmap(env.step))
  dum = jp.zeros((2,), jp.uint32)
  pol_fn = jax.jit(jax.vmap(lambda o: pol(o, dum)[0]))

  def first_act(st, x):
    s2 = st.replace(info={**st.info, "last_act": x})
    obs = env._get_obs(s2.data, s2.info)
    return pol(obs["state"], dum)[0]

  first_act_jit = jax.jit(jax.vmap(first_act))

  def step_set(st, x, act):
    s2 = st.replace(info={**st.info, "last_act": x})
    return env.step(s2, act)

  step_set_jit = jax.jit(jax.vmap(step_set))

  keys = jax.random.split(jax.random.PRNGKey(args.seed + 1), N)
  st0 = reset(keys)
  qpos = jp.asarray(d["qpos"][keep])
  qvel = jp.asarray(d["qvel"][keep])
  ctrl = jp.asarray(d["ctrl"][keep])
  # time 必须写成 (N,) 的数组: vmap 要求所有入参叶子秩 >= 1
  data = st0.data.replace(qpos=qpos, qvel=qvel, ctrl=ctrl,
                          time=jp.zeros((N,), jp.float32))
  data = mjx.forward(env.mjx_model, data)
  st0 = st0.replace(data=data)
  cmd3 = jp.broadcast_to(jp.asarray([args.cmd, 0.0, 0.0], jp.float32), (N, 3))
  st0.info["command"] = cmd3
  st_ref = reset(jax.random.split(jax.random.PRNGKey(args.seed + 7), N))
  st_ref.info["command"] = cmd3

  # ---- 终态诊断量 (每个 world 一个标量, 后面用于分组) ----
  cq0 = np.asarray(ctrl) - np.asarray(qpos)[:, 7:]
  diag = {}
  diag["up_z"] = np.asarray(d["up"][keep])
  diag["z_rel"] = np.asarray(d["zr"][keep])
  diag["倾角度"] = tilt_deg(diag["up_z"])
  diag["max|a_filt|"] = np.abs(np.asarray(d["a_filt"][keep])).max(1)
  diag["max|ctrl-qpos|"] = np.abs(cq0).max(1)
  diag["max|qvel_j|"] = np.abs(np.asarray(qvel)[:, 6:]).max(1)

  X_ctrl = ctrl
  X_zero = jp.zeros((N, 12), jp.float32)
  # warm 不动点: a <- pi(obs(last_act=a)), 从真值起步 (obs 里 last_act 与自身输出自洽)
  aw = first_act_jit(st0, X_ctrl)
  for _ in range(args.warm_iters):
    aw = first_act_jit(st0, aw)
  X_warm = aw
  a_truth = first_act_jit(st0, X_ctrl)
  a_zero = first_act_jit(st0, X_zero)
  a_warm = first_act_jit(st0, X_warm)
  j0_truth = np.abs(np.asarray(a_truth) - np.asarray(ctrl)).max(1)
  j0_zero = np.abs(np.asarray(a_zero) - np.asarray(ctrl)).max(1)
  j0_warm = np.abs(np.asarray(a_warm) - np.asarray(ctrl)).max(1)
  print("  第一帧 ctrl 跳变 max|act_0 - 起身最后目标|:  "
        "truth " + format(j0_truth.mean(), ".3f")
        + "   zero " + format(j0_zero.mean(), ".3f")
        + "   warm " + format(j0_warm.mean(), ".3f")
        + "   (走路策略自身输出与终态目标的差距)")
  print("  warm 不动点 |a*|: " + st4(np.abs(np.asarray(X_warm)).max(1))
        + "   与 truth 输出的差 max " + format(np.abs(np.asarray(X_warm)
                                                     - np.asarray(a_truth)).max(), ".3f"))

  # 各方案的"首帧动作" (t=0 会应用的 ctrl 目标), 只为打印
  def t0_act(kind):
    if kind == "hold":
      return ctrl
    if kind == "zero":
      return a_zero
    if kind == "warm":
      return a_warm
    if kind == "fresh":
      return pol_fn(st_ref.obs["state"])
    return a_truth

  F = args.frames
  res = {}
  for name, kind, blend_n in VARIANTS:
    if kind == "fresh":
      st = st_ref
      x0 = X_zero
    elif kind == "hold":
      st = st0
      x0 = X_ctrl
    elif kind == "ctrl":
      st = st0
      x0 = X_ctrl
    elif kind == "zero":
      st = st0
      x0 = X_zero
    else:
      st = st0
      x0 = X_warm
    frm = ctrl if blend_n > 0 else None
    left = blend_n
    A = np.zeros((F, N, 12), np.float32)
    U = np.zeros((F, N), np.float32)
    Z = np.zeros((F, N), np.float32)
    VJ = np.zeros((F, N), np.float32)
    DN = np.zeros((F, N), np.float32)
    for t in range(F):
      if kind == "hold":
        act_cmd = np.asarray(ctrl)
      else:
        a_pol = np.asarray(pol_fn(st.obs["state"]))
        if left > 0 and frm is not None:
          # 与查看器同款线性斜坡: 第 1 个走路帧 k=0 -> 完全是起身最后的目标
          k = 1.0 - left / float(blend_n)
          act_cmd = np.asarray(frm) * (1.0 - k) + a_pol * k
          left -= 1
        else:
          act_cmd = a_pol
      A[t] = act_cmd
      if t == 0:
        st = step_set_jit(st, x0, jp.asarray(act_cmd))
      else:
        st = step_jit(st, jp.asarray(act_cmd))
      q = np.asarray(st.data.qpos)
      qv = np.asarray(st.data.qvel)
      u = up_z(q)
      U[t] = u
      Z[t] = q[:, 2] - (np.asarray(env.terrain_height(q[:, :2]))
                        if env._hfield_ok else 0.0)
      VJ[t] = np.abs(qv[:, 6:]).max(1)
      DN[t] = np.asarray(st.done)
    fell, tfall = fall_stats(U, Z, args.fall_z, args.fall_uz, args.fall_debounce)
    J = jerk_of(A)
    D = dact_of(A)
    fin_ok = (U[-1] > args.up_th) & (Z[-1] >= args.z_lo) & (Z[-1] <= args.z_hi)
    j0 = np.abs(A[0] - np.asarray(ctrl)).max(1)
    res[name] = dict(A=A, U=U, Z=Z, VJ=VJ, DN=DN, J=J, D=D,
                     fell=fell, tfall=tfall, fin_ok=fin_ok, j0=j0,
                     fell_early=fall_stats(U[:30], Z[:30], args.fall_z,
                                           args.fall_uz, args.fall_debounce)[0],
                     fell_late=fall_stats(U[30:], Z[30:], args.fall_z,
                                          args.fall_uz, args.fall_debounce)[0])
    print("  [" + name.ljust(13) + "] 摔倒 "
          + format(100.0 * fell.mean(), "5.1f") + "%  "
          + "末态站住 " + format(100.0 * fin_ok.mean(), "5.1f") + "%  "
          + "jerk均 " + format(J[2:].mean(), ".4f") + "  "
          + "dact均 " + format(D[1:].mean(), ".4f") + "  "
          + "|act|均 " + format(np.abs(A).mean(), ".3f") + "  "
          + "首帧跳 " + format(j0.mean(), ".3f") + "  "
          + "max|qvel_j| " + format(VJ.max(), ".2f") + "  "
          + "末20帧done% " + format(100.0 * (DN[-20:].max(0) > 0).mean(), "5.1f"))

  print("")
  print("=== 汇总 (cmd vx=" + str(args.cmd) + ", " + str(N) + " 个终态, "
        + str(F) + " 帧 = " + format(F * env.dt, ".1f") + "s) ===")
  hdr = ("方案".ljust(14) + "摔倒%".rjust(8) + "末态站住%".rjust(11)
         + "jerk均".rjust(10) + "dact均".rjust(10) + "首帧抖".rjust(9)
         + "摔倒中位帧".rjust(12))
  print(hdr)
  print("  (前30帧摔倒 = 交还瞬间的瞬态; 30帧后摔倒 = 后续稳定性)")
  order = sorted(res.keys(), key=lambda k: (res[k]["fell"].mean(),
                                            res[k]["J"][2:].mean()))
  for k in order:
    r = res[k]
    fm = np.median(r["tfall"][r["fell"]]) if r["fell"].any() else -1
    print(k.ljust(14)
          + format(100.0 * r["fell"].mean(), "7.1f") + " "
          + format(100.0 * r["fin_ok"].mean(), "10.1f") + " "
          + format(r["J"][2:].mean(), "9.4f") + " "
          + format(r["D"][1:].mean(), "9.4f") + " "
          + format(r["j0"].mean(), "8.3f") + " "
          + format(fm, "11.0f")
          + "   前30帧 " + format(100.0 * r["fell_early"].mean(), "5.1f") + "%"
          + "   30帧后 " + format(100.0 * r["fell_late"].mean(), "5.1f") + "%")

  # ---- 哪个终态诊断量与 "交还后摔倒" 相关 (用 truth 组) ----
  print("")
  print("=== 终态诊断量 vs 交还后摔倒 (方案 truth, 即不做任何交还处理) ===")
  r = res["truth"]
  y = r["fell"].astype(np.float64)
  for k in ["up_z", "倾角度", "max|a_filt|", "max|ctrl-qpos|", "max|qvel_j|"]:
    v = np.asarray(diag[k], np.float64)
    if v.std() < 1e-9:
      print("  " + k.ljust(16) + " 常数, 跳过")
      continue
    c = float(np.corrcoef(v, y)[0, 1])
    lo = np.percentile(v, 33)
    hi = np.percentile(v, 67)
    m_lo = y[v <= lo]
    m_hi = y[v >= hi]
    print("  " + k.ljust(16) + " corr(与摔倒)=" + format(c, "+.3f")
          + "   低1/3 组摔倒 " + format(100.0 * m_lo.mean(), "5.1f") + "%"
          + "   高1/3 组摔倒 " + format(100.0 * m_hi.mean(), "5.1f") + "%")

  print("")
  print("=== 末态 max|a_filt| (起身最后的动作幅度) 分箱后的摔倒率 (truth 组) ===")
  qs = np.percentile(diag["max|a_filt|"], [20, 40, 60, 80])
  edges = [-1e9] + list(qs) + [1e9]
  for i in range(len(edges) - 1):
    m = (diag["max|a_filt|"] > edges[i]) & (diag["max|a_filt|"] <= edges[i + 1])
    if m.sum():
      print("  [" + format(edges[i], ".2f") + ", " + format(edges[i + 1], ".2f")
            + "]  n=" + str(int(m.sum())).rjust(4)
            + "  摔倒 " + format(100.0 * res["truth"]["fell"][m].mean(), "5.1f") + "%"
            + "  末态站住 " + format(100.0 * res["truth"]["fin_ok"][m].mean(), "5.1f") + "%")

  print("")
  print("=== up_z 分箱后的摔倒率 (truth 组) ===")
  for lo, hi in [(0.95, 0.96), (0.96, 0.97), (0.97, 0.98), (0.98, 1.01)]:
    m = (diag["up_z"] > lo) & (diag["up_z"] <= hi)
    if m.sum():
      print("  (" + format(lo, ".2f") + ", " + format(hi, ".2f") + "]  n="
            + str(int(m.sum())).rjust(4)
            + "  摔倒 " + format(100.0 * res["truth"]["fell"][m].mean(), "5.1f") + "%"
            + "  末态站住 " + format(100.0 * res["truth"]["fin_ok"][m].mean(), "5.1f") + "%")
  print("")
  print("=== 各方案里 jerk 最大的前 2s 是不是也摔倒 (看'抽搐'与摔倒是否同源) ===")
  for k in order:
    r = res[k]
    jm = r["J"][2:].mean(0)
    hi = jm >= np.percentile(jm, 67)
    lo = jm <= np.percentile(jm, 33)
    print("  " + k.ljust(14)
          + "jerk 高1/3 组摔倒 " + format(100.0 * r["fell"][hi].mean(), "5.1f") + "%"
          + "  低1/3 组摔倒 " + format(100.0 * r["fell"][lo].mean(), "5.1f") + "%")
  return 0


# -------------------------------------------------------------- stage realfall
def stage_realfall(args):
  """真摔倒 -> 起身: 用**走路中真的摔倒**的姿态当起点, 量起身策略的成功率。

  为什么需要它: stage spawn 用的是 getup_v2 训练分布 (地面 +0.35m 自由落体 + 随机
  关节/朝向), 而查看器里起身是从"走路策略真的摔了"的姿态开始的。两者是不是同一个
  分布, 决定了 82% 那个数字能不能搬到查看器里。用户第一轮反馈
  "站立效果很不错, 但为什么经常站立成功了却提示没成功" 就是这个问题。

  失败的三种形态必须分开 (查看器只打印 up_z, 看不出区别):
    A. 从来没直起来        -> 高度带内最好 up_z 和总最好 up_z 都低
    B. 直起来了但高度不在带内 -> 总最好 up_z 高, 但带内最好 up_z 很低  <- 最像"看着站起来了"
    C. 都满足但没保持 25 帧  -> 两个都高, 但连续帧数不够

  只有 in-bounds 的样本计入: 走出地形边缘的 world 会**自由落体** (无地面), 实测
  qvel 每帧 +0.196 (= 重力), 几十秒后 z_rel = -2558m, 那种样本不是起身问题。
  """
  from envs.go1_walk import Go1Walk
  pol_w, ns = load_policy(args.walk_pkl)
  cfg = build_walk_cfg(ns, args.njmax)
  env = Go1Walk(cfg)
  pol_g, ng = load_policy(args.getup_pkl)
  print("[realfall] 走路 obs=" + str(ns) + " 起身 obs=" + str(ng)
        + "  env obs=" + str(env.observation_size["state"][0])
        + "  cmd vx=" + str(args.cmd) + "  envs=" + str(args.envs))
  if ns != env.observation_size["state"][0] or ng != 5 * 42:
    print("[错误] 维度不符 (走路需与 env 一致; 起身需 210 维 getup_v2)")
    return 1
  N = args.envs
  dum = jp.zeros((2,), jp.uint32)
  # **全部 jit**: eager 的 vmap 每帧都会重新 trace+compile (实测 500s 都跑不完)
  reset = jax.jit(jax.vmap(env.reset))
  step_vm = jax.jit(jax.vmap(env.step))
  polw_v = jax.jit(jax.vmap(lambda o: pol_w(o, dum)[0]))
  polg_v = jax.jit(jax.vmap(lambda o: pol_g(o, dum)[0]))
  gh_v = jax.jit(jax.vmap(lambda xy: env.terrain_height(xy[None, :])[0]))
  gyro_v = jax.jit(jax.vmap(env.get_gyro))
  grav_v = jax.jit(jax.vmap(env.get_gravity))
  keys = jax.random.split(jax.random.PRNGKey(args.seed), N)
  st = reset(keys)
  st = jax.block_until_ready(st)
  st.info["command"] = jp.broadcast_to(
      jp.asarray([args.cmd, 0.0, 0.0], jp.float32), (N, 3))

  # ---- 阶段 A: 走路直到查看器判据说"倒了" (连续 fall_debounce 帧) ----
  fcnt = jp.zeros(N, jp.int32)
  frozen = np.zeros(N, bool)
  Bq = jp.zeros((N, 19), jp.float32)
  Bv = jp.zeros((N, 18), jp.float32)
  Bc = jp.zeros((N, 12), jp.float32)
  hfield = bool(env._hfield_ok)

  def ground(xy):                       # xy: numpy (N,2)
    if hfield:
      return np.asarray(gh_v(jp.asarray(xy)))
    return np.zeros(len(xy), np.float32)

  at = np.full(N, -1, np.int32)
  n_respawn = 0
  for i in range(args.steps):
    st = step_vm(st, polw_v(st.obs["state"]))
    q = np.asarray(st.data.qpos)
    # 走出地形边缘的 world 会没完没了地自由落体 (没有地面), 永远摔不出"界内的摔倒"。
    # 把没冻结的越界 world 原地重生成 (reset), 让它们继续在界内走 —— 只有界内的摔倒
    # 才是"起身问题"的样本。
    oob = ((np.abs(q[:, 0]) > args.bound_xy)
           | (np.abs(q[:, 1]) > args.bound_xy)) & (~frozen)
    if oob.any():
      n_respawn += int(oob.sum())
      st_re = reset(jax.random.split(
          jax.random.PRNGKey(args.seed + 1000 + i), N))
      # 只换 qpos/qvel (整棵 state 树里有些叶子形状是 (0,) 的变长接触数组, 不能
      # jp.where 广播)。mjx.step 每步都会从 qpos 重算接触, 所以不用 forward。
      m2 = jp.asarray(oob)[:, None]
      st = st.replace(data=st.data.replace(
          qpos=jp.where(m2, st_re.data.qpos, st.data.qpos),
          qvel=jp.where(m2, st_re.data.qvel, st.data.qvel)))
      fcnt = jp.where(jp.asarray(oob), jp.zeros(N, jp.int32), fcnt)
      q = np.asarray(st.data.qpos)
    upz = 1.0 - 2.0 * (q[:, 4] ** 2 + q[:, 5] ** 2)
    zr = q[:, 2] - ground(q[:, :2])
    fallen = (zr < args.fall_z) | (upz < args.fall_uz)
    fcnt = jp.where(fallen, fcnt + 1, jp.zeros(N, jp.int32))
    trig = np.asarray(fcnt) >= args.fall_debounce
    new = trig & (~frozen)
    if new.any():
      idx = np.where(new)[0]
      Bq = Bq.at[idx].set(q[idx])
      Bv = Bv.at[idx].set(np.asarray(st.data.qvel)[idx])
      Bc = Bc.at[idx].set(np.asarray(st.data.ctrl)[idx])
      at[idx] = i
      frozen[idx] = True
    if frozen.all():
      break
    if (i + 1) % 200 == 0:
      print("    阶段A 步 " + str(i + 1) + ": 已摔倒 "
            + str(int(frozen.sum())) + "/" + str(N), flush=True)
  qq = np.asarray(Bq)
  zrf = qq[:, 2] - ground(qq[:, :2])
  upf = 1.0 - 2.0 * (qq[:, 4] ** 2 + qq[:, 5] ** 2)
  # in-bounds 过滤: 地形范围内的摔倒才是"摔倒", 出界的是掉下悬崖的自由落体
  inb = (np.abs(qq[:, 0]) < args.bound_xy) & (np.abs(qq[:, 1]) < args.bound_xy)
  print("  阶段A: 越界重生 " + str(n_respawn) + " 次 (把走出地形边缘的 world 拉回界内)")
  print("  阶段A: " + str(int(frozen.sum())) + "/" + str(N) + " 个 world 在 "
        + str(args.steps) + " 步内摔倒;  其中界内 (|xy|<"
        + str(args.bound_xy) + ") 的 " + str(int(inb.sum()))
        + "   到摔帧数 均 " + format(at[inb].mean(), ".0f")
        + "  被排除(出界/自由落体) " + str(int((~inb).sum())))
  if int(inb.sum()) == 0:
    print("[错误] 没有界内的摔倒样本")
    return 1

  # ---- 阶段 B: 从冻结的摔倒姿态跑起身策略 (查看器同款锚定动作 + 5 帧历史) ----
  st2 = reset(jax.random.split(jax.random.PRNGKey(args.seed + 3), N))
  data = st2.data.replace(qpos=jp.asarray(Bq), qvel=jp.asarray(Bv),
                          ctrl=jp.asarray(Bc),
                          time=jp.zeros((N,), jp.float32))
  st2 = st2.replace(data=mjx.forward(env.mjx_model, data))
  st2.info["command"] = jp.broadcast_to(
      jp.asarray([0.0, 0.0, 0.0], jp.float32), (N, 3))
  hist = None
  aprev = jp.zeros((N, 12), jp.float32)
  rawp = jp.zeros((N, 12), jp.float32)
  Z12 = jp.zeros((N, 12), jp.float32)
  hcnt = np.zeros(N, np.int32)
  tol = np.zeros(N, np.int32)          # 容错计数: 满足 +1 / 不满足 -1 (不归零)
  tol_at = np.full(N, -1, np.int32)
  best_any = np.zeros(N, np.float32)
  best_ok = np.zeros(N, np.float32)
  first_at = np.full(N, -1, np.int32)
  for i in range(args.getup_steps):
    fr = jp.concatenate([gyro_v(st2.data), grav_v(st2.data),
                         st2.data.qpos[:, 7:19] - env._default_pose,
                         st2.data.qvel[:, 6:18], rawp], axis=1)
    hist = (jp.broadcast_to(fr[:, None, :], (N, 5, 42)) if hist is None
            else jp.concatenate([hist[:, 1:], fr[:, None, :]], axis=1))
    a = polg_v(hist.reshape(N, -1))
    a_c = jp.clip(a, -args.getup2_action_clip, args.getup2_action_clip)
    a_f = args.getup_action_alpha * a_c + (1.0 - args.getup_action_alpha) * aprev
    aprev = a_f
    rawp = a_c
    st2 = step_vm(st2, env._default_pose + args.getup_action_scale * a_f)
    q = np.asarray(st2.data.qpos)
    upz = 1.0 - 2.0 * (q[:, 4] ** 2 + q[:, 5] ** 2)
    zr = q[:, 2] - ground(q[:, :2])
    ok = (upz > args.up_th) & (zr >= args.z_lo) & (zr <= args.z_hi)
    best_any = np.maximum(best_any, upz)
    best_ok = np.maximum(best_ok, np.where(ok, upz, 0.0))
    hcnt = np.where(ok, hcnt + 1, 0)
    newly = (hcnt >= args.hold) & (first_at < 0)
    first_at = np.where(newly, i + 1, first_at)
    tol = np.where(ok, tol + 1, np.maximum(tol - 1, 0))
    tnew = (tol >= args.hold) & (tol_at < 0)
    tol_at = np.where(tnew, i + 1, tol_at)
    if (i + 1) % 200 == 0:
      print("    阶段B 步 " + str(i + 1) + ": 成功 "
            + str(int(((first_at >= 0) & inb).sum())) + "/" + str(int(inb.sum())),
            flush=True)
    if (first_at[inb] >= 0).all():
      break
  m = inb
  succ = first_at >= 0
  A_never = best_any < args.up_th
  B_off = (~A_never) & (best_ok == 0.0)
  C_nothold = (~A_never) & (best_ok > 0.0) & (~succ)
  print("")
  print("=== 真摔倒 -> 起身 (n=" + str(int(m.sum())) + ", 起身最多 "
        + str(args.getup_steps) + " 帧) ===")
  print("  起身成功 (查看器判据 up_z>" + str(args.up_th) + " 且 z_rel∈["
        + str(args.z_lo) + "," + str(args.z_hi) + "] 连续 " + str(args.hold)
        + " 帧): " + str(int((succ & m).sum())) + "/" + str(int(m.sum()))
        + " = " + format(100.0 * (succ & m).sum() / m.sum(), ".1f") + "%")
  ts = (tol_at >= 0) & m
  print("  若把 '连续 " + str(args.hold) + " 帧' 改成**容错计数** (满足+1, 不满足-1, 不归零): "
        + str(int(ts.sum())) + "/" + str(int(m.sum())) + " = "
        + format(100.0 * ts.sum() / m.sum(), ".1f") + "%  (严格连续是 "
        + format(100.0 * (succ & m).sum() / m.sum(), ".1f") + "%)")
  if ts.any() and (succ & m).any():
    print("    到站时间: 严格 均 " + format(first_at[succ & m].mean() * 0.02, ".2f")
          + "s  容错 均 " + format(tol_at[ts].mean() * 0.02, ".2f") + "s")
  print("  失败形态 A 从来没直起来 (总最好 up_z<" + str(args.up_th) + "): "
        + str(int((A_never & m).sum())))
  print("  失败形态 B 直起来了但**高度不在带内**: " + str(int((B_off & m).sum()))
        + "   <- 用户说的'看着站起来了却提示没成功'")
  print("  失败形态 C 满足过但没连续 " + str(args.hold) + " 帧: "
        + str(int((C_nothold & m).sum())))
  print("  高度带内最好 up_z: 均 " + format(best_ok[m].mean(), ".3f")
        + "  中 " + format(np.median(best_ok[m]), ".3f")
        + "  |  总最好 up_z: 均 " + format(best_any[m].mean(), ".3f")
        + "  中 " + format(np.median(best_any[m]), ".3f"))
  print("  起始姿态: up_z 均 " + format(upf[m].mean(), ".3f")
        + " (仰卧<-0.5: " + str(int((upf[m] < -0.5).sum()))
        + "  侧躺|upz|<=0.5: " + str(int((np.abs(upf[m]) <= 0.5).sum()))
        + "  俯卧0.5~0.9: " + str(int(((upf[m] > 0.5) & (upf[m] <= 0.9)).sum()))
        + ")  起始 z_rel 均 " + format(zrf[m].mean(), ".3f"))
  print("  对比 stage spawn (训练分布): 88.7% 成功 —— 两者差距 = 训推分布差")
  return 0


# ----------------------------------------------------------------- stage cycle
def stage_cycle(args):
  """端到端: 走路 -> 摔倒 -> 起身 -> 交还 -> 再走路, 跑很久, 对照交还混合帧数。

  这是查看器调度器的**无渲染完整复刻** (同一个 Go1Walk env、同一份 getup_v2 历史
  obs 缓冲、同一套摔倒/交还判据、同一个 cooldown), 所以它的数字可以直接当查看器
  的验证。核心指标:
    交还次数 n_hand
    交还后复发 n_post = 交还在 relapse_window 帧内又摔倒的次数 <- "抽搐摔倒"的直接度量
    原生摔倒 n_pre   = 与交还无关的摔倒 (走路策略自己就不稳)
    relapse = n_post / n_hand
  """
  from envs.go1_walk import Go1Walk
  pol_w, ns = load_policy(args.walk_pkl)
  cfg = build_walk_cfg(ns, args.njmax)
  env = Go1Walk(cfg)
  pol_g, ng = load_policy(args.getup_pkl)
  ne = env.observation_size["state"][0]
  print("[cycle] 走路 obs=" + str(ns) + "  起身 obs=" + str(ng)
        + "  env obs=" + str(ne) + "  cmd vx=" + str(args.cmd))
  if ns != ne:
    print("[错误] 走路策略维度与 env 不符")
    return 1
  if ng != 5 * 42:
    print("[错误] 起身策略不是 getup_v2 的 210 维历史 obs")
    return 1
  N = args.envs
  F = args.frames
  T = args.relapse_window
  dum = jp.zeros((2,), jp.uint32)
  reset = jax.jit(jax.vmap(env.reset))
  step_vm = jax.vmap(env.step)
  polg_v = jax.vmap(lambda o: pol_g(o, dum)[0])
  polw_v = jax.vmap(lambda o: pol_w(o, dum)[0])
  gh_v = jax.vmap(lambda xy: env.terrain_height(xy[None, :])[0])
  # 传感器 helper 是按**单 world** 写的 (site_xmat[..., id].T @ g), 批量 data 上会
  # 退化成 (N,9) -> 矩阵乘维度错; 必须 vmap 到每个 world。
  gyro_v = jax.vmap(env.get_gyro)
  grav_v = jax.vmap(env.get_gravity)
  keys = jax.random.split(jax.random.PRNGKey(args.seed), N)
  st0 = reset(keys)
  st0 = jax.block_until_ready(st0)
  st0.info["command"] = jp.broadcast_to(
      jp.asarray([args.cmd, 0.0, 0.0], jp.float32), (N, 3))
  Z12 = jp.zeros((N, 12), jp.float32)

  def init_carry():
    return {
        "mode": jp.zeros(N, bool),
        "fcnt": jp.zeros(N, jp.int32),
        "hcnt": jp.zeros(N, jp.int32),
        "cool": jp.zeros(N, jp.int32),
        "g0": jp.zeros(N, jp.int32),
        "hist": jp.zeros((N, 5, 42), jp.float32),
        "aprev": Z12,
        "raw": Z12,
        "frm": Z12,
        "left": jp.zeros(N, jp.int32),
        "aapp": Z12,
        "n_hand": jp.zeros(N, jp.int32),
        "n_ok": jp.zeros(N, jp.int32),
        "n_post": jp.zeros(N, jp.int32),
        "n_pre": jp.zeros(N, jp.int32),
        "n_to": jp.zeros(N, jp.int32),
        "gbest": jp.zeros(N, jp.float32),
        "gzr": jp.zeros(N, jp.float32),
        "gup": jp.zeros(N, jp.float32),
        "gok": jp.zeros(N, bool),
        "gstand_f": jp.zeros(N, jp.float32),
        "last_h": jp.full(N, -10 ** 6, jp.int32),
        "dist": jp.zeros(N, jp.float32),
        "walk_f": jp.zeros(N, jp.float32),
    }

  def make_one(blend_n):
    @jax.jit
    def one(st, c, t):
      q = st.data.qpos
      qv = st.data.qvel
      gh = gh_v(q[:, :2])
      upz = 1.0 - 2.0 * (q[:, 4] ** 2 + q[:, 5] ** 2)
      zr = q[:, 2] - gh
      mode = c["mode"]
      armable = (~mode) & (c["cool"] <= 0)
      fallen = (zr < args.fall_z) | (upz < args.fall_uz)
      fcnt = jp.where(armable & fallen, c["fcnt"] + 1,
                      jp.where(armable, jp.zeros(N, jp.int32), c["fcnt"]))
      trig = armable & (fcnt >= args.fall_debounce)
      stand = (upz > args.up_th) & (zr >= args.z_lo) & (zr <= args.z_hi)
      hcnt = jp.where(trig, jp.zeros(N, jp.int32),
                      jp.where(mode, jp.where(stand, c["hcnt"] + 1,
                                              jp.zeros(N, jp.int32)),
                               jp.zeros(N, jp.int32)))
      to = mode & ((t - c["g0"]) >= args.getup_timeout)
      ok_g = mode & (hcnt >= args.hold)
      hand = mode & (ok_g | to)
      active = mode | trig
      fr = jp.concatenate([gyro_v(st.data), grav_v(st.data),
                           q[:, 7:19] - env._default_pose, qv[:, 6:18],
                           c["raw"]], axis=1)
      hist = jp.where(trig[:, None, None],
                      jp.broadcast_to(fr[:, None, :], (N, 5, 42)),
                      jp.concatenate([c["hist"][:, 1:], fr[:, None, :]], axis=1))
      a_raw = polg_v(hist.reshape(N, -1))
      a_c = jp.clip(a_raw, -args.getup2_action_clip, args.getup2_action_clip)
      a_f = (args.getup_action_alpha * a_c
             + (1.0 - args.getup_action_alpha) * c["aprev"])
      act_g = env._default_pose + args.getup_action_scale * a_f
      a_w = polw_v(st.obs["state"])
      if blend_n > 0:
        kk = 1.0 - c["left"].astype(jp.float32) / float(blend_n)
        act_w = jp.where((c["left"] > 0)[:, None],
                         c["frm"] * (1.0 - kk[:, None]) + a_w * kk[:, None],
                         a_w)
      else:
        act_w = a_w
      act = jp.where(active[:, None], act_g, act_w)
      st2 = step_vm(st, act)
      nc = dict(c)
      nc["mode"] = jp.where(trig, True, jp.where(hand, False, mode))
      nc["fcnt"] = fcnt
      nc["hcnt"] = hcnt
      nc["cool"] = jp.where(hand, jp.int32(args.getup_cooldown),
                            jp.maximum(c["cool"] - 1, 0))
      nc["g0"] = jp.where(trig, t, c["g0"])
      nc["hist"] = hist
      nc["aprev"] = jp.where(trig[:, None], Z12,
                             jp.where(mode[:, None], a_f, Z12))
      nc["raw"] = jp.where(trig[:, None], Z12,
                           jp.where(mode[:, None], a_c, Z12))
      nc["frm"] = jp.where(hand[:, None], act_g, c["frm"])
      nc["left"] = jp.where(hand, jp.int32(blend_n),
                            jp.maximum(c["left"] - 1, 0))
      nc["aapp"] = act
      nc["n_hand"] = c["n_hand"] + hand.astype(jp.int32)
      nc["n_ok"] = c["n_ok"] + ok_g.astype(jp.int32)
      # "交还后复发" 只以**成功交还**为参照: 超时交还时机器人本来就还躺着,
      # 下一帧必然又满足摔倒判据 -> 拿超时当参照会把指标污染成 90%+ (实测)。
      near = (t - c["last_h"]) < T
      nc["n_post"] = c["n_post"] + (trig & near).astype(jp.int32)
      nc["n_pre"] = c["n_pre"] + (trig & (~near)).astype(jp.int32)
      nc["n_to"] = c["n_to"] + to.astype(jp.int32)
      # 记录"最直立那一帧"的 up_z 与相对高度 —— 用来区分两种失败:
      #   从来没直起来 (gbest 低)  vs  直起来了但高度不在带内 (gzr 出界)
      better = upz > c["gbest"]
      nc["gzr"] = jp.where(trig, 0.0,
                           jp.where(mode & better, zr, c["gzr"]))
      nc["gup"] = jp.where(trig, 0.0,
                           jp.where(mode & better, upz, c["gup"]))
      nc["gbest"] = jp.where(trig, 0.0,
                             jp.where(mode, jp.maximum(c["gbest"], upz),
                                      c["gbest"]))
      nc["gok"] = jp.where(trig, False,
                           jp.where(mode, c["gok"] | stand, c["gok"]))
      nc["gstand_f"] = (c["gstand_f"]
                        + jp.where(mode, stand.astype(jp.float32), 0.0))
      nc["last_h"] = jp.where(ok_g, t, c["last_h"])
      nc["dist"] = c["dist"] + jp.where(~mode,
                                        jp.linalg.norm(qv[:, :2], axis=1),
                                        0.0) * env.dt
      nc["walk_f"] = c["walk_f"] + (~mode).astype(jp.float32)
      y = {"trig": trig, "hand": hand, "to": to, "upz": upz, "zr": zr,
           "dact": jp.abs(act - c["aapp"]).max(1), "in_walk": ~mode}
      return st2, nc, y
    return one

  out = {}
  for blend_n in args.cycle_blends:
    one = make_one(blend_n)
    st = st0
    c = init_carry()
    ys = []
    t0 = time.time()
    for t in range(F):
      st, c, y = one(st, c, jp.int32(t))
      ys.append(y)
      if args.verbose and (t % 50 == 0 or t == F - 1):
        print("    t=" + str(t).rjust(4)
              + " 起身中 " + str(int(np.asarray(c["mode"]).sum())).rjust(3)
              + " upz均 " + format(float(np.asarray(y["upz"]).mean()), ".3f")
              + " 起身中upz均 "
              + format(float(np.asarray(jp.where(c["mode"], y["upz"], 0.0)
                                        ).sum() / max(int(np.asarray(
                                            c["mode"]).sum()), 1)), ".3f")
              + " 起身期最好upz均 "
              + format(float(np.asarray(c["gbest"]).mean()), ".3f")
              + " |qvel|max "
              + format(float(np.abs(np.asarray(st.data.qvel)).max()), ".1f"))
    nh = int(np.asarray(c["n_ok"]).sum())
    nhtot = int(np.asarray(c["n_hand"]).sum())
    npost = int(np.asarray(c["n_post"]).sum())
    npre = int(np.asarray(c["n_pre"]).sum())
    nto = int(np.asarray(c["n_to"]).sum())
    dist = np.asarray(c["dist"])
    walkf = np.asarray(c["walk_f"]) / float(F)
    upz = np.stack([np.asarray(y["upz"]) for y in ys])
    dact = np.stack([np.asarray(y["dact"]) for y in ys])
    hand = np.stack([np.asarray(y["hand"]) for y in ys])
    rec = np.zeros_like(hand)
    for k in range(1, 11):
      rec[k:] |= hand[:-k]
    pd = float(dact[rec].mean()) if rec.any() else 0.0
    fin = (upz[-1] > args.up_th)
    gb = np.asarray(c["gbest"])
    gok = np.asarray(c["gok"])
    gsf = np.asarray(c["gstand_f"])
    gu = np.asarray(c["gup"]); gz = np.asarray(c["gzr"])
    mm = gu > 0.95
    if mm.any():
      inband = ((gz[mm] >= 0.20) & (gz[mm] <= 0.34)).sum()
      print("  [blend " + str(blend_n).rjust(2) + "] 直起来过 (up_z>0.95) 的 world: "
            + str(int(mm.sum())).rjust(3) + "/" + str(N)
            + "; 其中最直立那一帧高度落在 [0.20,0.34] 的: " + str(int(inband))
            + "/" + str(int(mm.sum()))
            + "  高度分布 p10 " + format(np.percentile(gz[mm], 10), ".3f")
            + " 中 " + format(np.median(gz[mm]), ".3f")
            + " p90 " + format(np.percentile(gz[mm], 90), ".3f"))
    else:
      print("  [blend " + str(blend_n).rjust(2) + "] 直起来过 (up_z>0.95) 的 world: 0/"
            + str(N))
    print("  [blend " + str(blend_n).rjust(2) + "] 起身期: 站姿帧数均 "
          + format(gsf.mean(), ".1f") + "  '曾满足判据' 的 world "
          + str(int(gok.sum())).rjust(4) + "/" + str(N)
          + "  起身期最好 up_z: 均 " + format(gb.mean(), ".3f")
          + " 中 " + format(np.median(gb), ".3f")
          + " p90 " + format(np.percentile(gb, 90), ".3f"))
    out[blend_n] = dict(nh=nh, npost=npost, npre=npre, nto=nto, pd=pd,
                        dist=float(dist.mean()), walkf=float(walkf.mean()),
                        fin=float(fin.mean()), dt=time.time() - t0,
                        upz=float(upz[-1].mean()))
    print("  [blend " + str(blend_n).rjust(2) + "] 成功交还 " + str(nh).rjust(4)
          + " (总交还 " + str(nhtot) + ")"
          + "  交还后复发 " + str(npost).rjust(4)
          + "  原生摔倒 " + str(npre).rjust(4)
          + "  relapse " + format(100.0 * npost / max(nh, 1), "5.1f") + "%"
          + "  超时 " + str(nto).rjust(3)
          + "  交还后10帧内 ctrl 抖动均 " + format(pd, ".3f")
          + "  |  走路帧占比 " + format(100.0 * walkf.mean(), "4.1f") + "%"
          + "  走了 " + format(dist.mean(), ".1f") + "m"
          + "  末态 up_z " + format(upz[-1].mean(), ".3f")
          + "  (" + format(out[blend_n]["dt"], ".0f") + "s)")

  print("")
  print("=== 汇总 (端到端 " + str(F) + " 帧 = " + format(F * env.dt, ".1f")
        + "s, " + str(N) + " 个 world, 交还窗口 " + str(T) + " 帧) ===")
  print("  混合帧数  成功交还  交还后复发   relapse%   原生摔倒  交还后10帧抖动")
  for blend_n in args.cycle_blends:
    r = out[blend_n]
    print("  " + str(blend_n).rjust(6) + "   " + str(r["nh"]).rjust(7) + "   "
          + str(r["npost"]).rjust(9) + "   "
          + format(100.0 * r["npost"] / max(r["nh"], 1), "7.1f") + "   "
          + str(r["npre"]).rjust(8) + "   " + format(r["pd"], "12.3f"))
  best = min(args.cycle_blends,
             key=lambda b: (out[b]["npost"] / max(out[b]["nh"], 1),
                            out[b]["pd"]))
  print("  -> 交还后复发率最低且抖动最小的是 blend=" + str(best))
  return 0


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--stage", choices=["spawn", "walk", "cycle", "realfall"],
                  required=True)
  ap.add_argument("--dump", default=os.path.join(HERE, "handover_states.npz"))
  ap.add_argument("--envs", type=int, default=256)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--steps", type=int, default=400,
                  help="[spawn] 起身阶段最多跑多少控制步")
  ap.add_argument("--frames", type=int, default=150,
                  help="[walk] 交还后跑多少控制步 (150 步 = 3.0s)")
  ap.add_argument("--warm_iters", type=int, default=6,
                  help="[walk] warm 不动点迭代次数")
  ap.add_argument("--getup_pkl",
                  default=os.path.join(HERE, "..", "policies", "go1_getup_policy.pkl"))
  ap.add_argument("--walk_pkl",
                  default=os.path.join(HERE, "..", "policies", "go1_walk_policy.pkl"))
  ap.add_argument("--getup2_action_clip", type=float, default=8.0,
                  help="必须与训练一致, 否则动作被夹掉 = 假成功率")
  ap.add_argument("--up_th", type=float, default=0.95,
                  help="交还判据的 up_z 阈值 (查看器 --getup_up_th)")
  ap.add_argument("--z_lo", type=float, default=0.20)
  ap.add_argument("--z_hi", type=float, default=0.34)
  ap.add_argument("--hold", type=int, default=25,
                  help="交还判据需要连续满足多少帧 (查看器 --getup_hold)")
  ap.add_argument("--cmd", type=float, default=0.5,
                  help="[walk] 交还后给的 vx 指令 (查看器默认 0.5; 试 0.0 对照)")
  ap.add_argument("--fall_uz", type=float, default=0.30)
  ap.add_argument("--fall_z", type=float, default=0.16)
  ap.add_argument("--fall_debounce", type=int, default=18)
  ap.add_argument("--getup_action_scale", type=float, default=0.5)
  ap.add_argument("--getup_action_alpha", type=float, default=0.8)
  ap.add_argument("--getup_cooldown", type=int, default=50,
                  help="[cycle] 交还/超时后多少帧内不再触发起身 (查看器同款)")
  ap.add_argument("--getup_timeout", type=int, default=400)
  ap.add_argument("--getup_steps", type=int, default=600,
                  help="[realfall] 起身阶段最多跑多少步")
  ap.add_argument("--bound_xy", type=float, default=5.0,
                  help="[realfall] |xy| 超过它就当作走出地形边缘 (自由落体), 排除")
  ap.add_argument("--njmax", type=int, default=256,
                  help="per-world 约束行数上限。默认 256 = go1_walk 训练档; "
                       "hfield 地形上摔倒时实测需要 ~1300")
  ap.add_argument("--relapse_window", type=int, default=150,
                  help="[cycle] 交还后多少帧内又摔倒算'交还复发'")
  ap.add_argument("--verbose", action="store_true",
                  help="[cycle] 每 50 帧打一行状态")
  ap.add_argument("--cycle_blends", type=str, default="0,10,30",
                  help="[cycle] 要对照的交还混合帧数, 逗号分隔")
  args = ap.parse_args()
  args.cycle_blends = [int(v) for v in str(args.cycle_blends).split(",") if v != ""]
  t0 = time.time()
  if args.stage == "spawn":
    rc = stage_spawn(args)
  elif args.stage == "walk":
    rc = stage_walk(args)
  elif args.stage == "cycle":
    rc = stage_cycle(args)
  else:
    rc = stage_realfall(args)
  print("[完成] " + format(time.time() - t0, ".1f") + "s")
  return rc


if __name__ == "__main__":
  raise SystemExit(main())
