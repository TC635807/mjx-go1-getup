#!/usr/bin/env python
"""probe_getup_standability.py — 判据本身可不可行?

三个问题 (都是"成功率上不去"最可能的系统性原因):
  Q1 get_gravity 在**名义站姿**下是不是 [0,0,-1]? (IMU site 有旋转变换时会偏)
  Q2 地形坡度分布如何? 判据要求 up_z>0.99 (即躯干相对**世界竖直** <8.1 度),
     而四足站在**斜坡**上躯干必然随坡面倾斜 -> 坡度 >8 度的地方该判据**物理上不可达**。
  Q3 站在坡上/名义姿态不动 (action=0), 判据能拿到多少成功率? 这是"平凡策略"的分数,
     任何学习策略的及格线。

用法:
  python sim/probe_standability.py
"""
import os
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.30")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import jax
import jax.numpy as jp
import numpy as np

from envs.go1_getup import Go1Getup, default_config


def report(name, v, fmt="{:+.4f}"):
  v = np.asarray(v).ravel()
  print(f"  {name:<34} 均值 {fmt.format(v.mean())}  中位 {fmt.format(np.median(v))}  "
        f"p10 {fmt.format(np.percentile(v,10))}  p90 {fmt.format(np.percentile(v,90))}")


def main() -> int:
  # ---------------- Q1: 平地 + 名义站姿下的 gravity ----------------
  print("=" * 78)
  print("Q1  名义站姿 (home keyframe) 的机体系重力向量")
  cfg = default_config()
  cfg.terrain = False
  cfg.height_scan.enable = False
  env_flat = Go1Getup(cfg)
  kf = env_flat._mj_model.keyframe("home")
  print(f"  home keyframe qpos[2] = {float(kf.qpos[2]):.4f}  "
        f"quat = {np.round(np.asarray(kf.qpos[3:7]), 5).tolist()}")
  print(f"  default_pose = {np.round(np.asarray(env_flat._default_pose), 4).tolist()}")
  data = env_flat.mjx_model  # noqa: F841  (仅占位)
  # 直接前向 home keyframe
  import mujoco
  m = env_flat.mj_model
  d = mujoco.MjData(m)
  mujoco.mj_resetDataKeyframe(m, d, 0)
  import mujoco.mjx as mjx
  dx = mjx.put_data(m, d)
  g = env_flat.get_gravity(dx)
  print(f"  get_gravity(home) = {np.round(np.asarray(g), 5).tolist()}   "
        f"err vs [0,0,-1] = {float(jp.sum(jp.square(env_flat._up_vec - g))):.5f}")
  iid = env_flat._imu_site_id
  print(f"  IMU site xmat = \n{np.round(np.asarray(d.site_xmat[iid]).reshape(3,3), 5)}")

  # ---------------- Q2: 地形坡度分布 ----------------
  print("=" * 78)
  print("Q2  地形坡度分布 (随机 20000 点, 有限差分)")
  cfg2 = default_config()
  env = Go1Getup(cfg2)
  k = jax.random.PRNGKey(0)
  xy = jax.random.uniform(k, (20000, 2), minval=-5.0, maxval=5.0)
  eps = jp.float32(0.05)
  h = env.terrain_height(xy)
  hx = env.terrain_height(xy + jp.array([eps, 0.0]))
  hy = env.terrain_height(xy + jp.array([0.0, eps]))
  gx = (hx - h) / eps
  gy = (hy - h) / eps
  slope = np.degrees(np.arctan(np.sqrt(np.asarray(gx) ** 2 + np.asarray(gy) ** 2)))
  report("坡度 (度)", slope, "{:.2f}")
  for th in (2.0, 5.0, 8.1, 10.0, 15.0):
    print(f"    坡度 < {th:4.1f} 度的比例: {100*float((slope < th).mean()):5.1f}%")
  report("地面高度 (m)", np.asarray(h), "{:+.3f}")

  # ---------------- Q3: 名义姿态不动, 判据成功率 ----------------
  print("=" * 78)
  print("Q3  各出生档下 '零动作(保持关节角)' 的判据成功率")
  for (alpha, beta, tag) in ((0.0, 0.0, "alpha=0 beta=0 (竖直+站姿关节)"),
                             (1.0, 0.0, "alpha=1 beta=0 (竖直+随机关节)"),
                             (1.0, 1.0, "alpha=1 beta=1 (真任务)")):
    c = default_config()
    c.getup.difficulty = alpha
    c.getup.orient_rand = beta
    c.getup.drop_prob = 1.0
    c.njmax = 768
    c.naconmax = 8 * 8192
    e = Go1Getup(c)
    N = 1024
    reset = jax.jit(jax.vmap(e.reset))
    step = jax.jit(jax.vmap(e.step))
    st = jax.block_until_ready(reset(jax.random.split(jax.random.PRNGKey(1), N)))

    def zref(q):
      return np.asarray(e.terrain_height(jp.asarray(q[:, :2])))

    q = np.asarray(st.data.qpos)
    up = 1.0 - 2.0 * (q[:, 4] ** 2 + q[:, 5] ** 2)
    zr0 = q[:, 2] - zref(q)
    xy0 = jp.asarray(q[:, :2])
    hh = e.terrain_height(xy0)
    gxx = (e.terrain_height(xy0 + jp.array([0.05, 0.0])) - hh) / 0.05
    gyy = (e.terrain_height(xy0 + jp.array([0.0, 0.05])) - hh) / 0.05
    sl0 = np.degrees(np.arctan(np.sqrt(np.asarray(gxx) ** 2 + np.asarray(gyy) ** 2)))
    hold = np.zeros(N, np.int32)
    best = np.zeros(N, np.int32)
    zero = jp.zeros((N, e.mjx_model.nu))
    for i in range(300):
      st = step(st, zero)
      if (i + 1) % 5 == 0:
        q = np.asarray(st.data.qpos)
        gh = zref(q)
        up = 1.0 - 2.0 * (q[:, 4] ** 2 + q[:, 5] ** 2)
        zr = q[:, 2] - gh
        ok = (up > 0.99) & (zr >= 0.24) & (zr <= 0.32)
        hold = np.where(ok, hold + 5, 0)
        best = np.maximum(best, hold)
    succ = best >= 25
    print(f"  --- {tag}")
    print(f"      零动作成功率: {int(succ.sum())}/{N} = {100*succ.mean():.1f}%")
    print(f"      出生时已达判据: {100*float(((up>0.99)&(zr0>=0.24)&(zr0<=0.32)).mean()):.1f}%")
    with np.errstate(invalid="ignore"):
      print(f"      成功样本的出生坡度: 中位 {np.nanmedian(sl0[succ]):.1f}°   "
            f"失败样本: 中位 {np.nanmedian(sl0[~succ]):.1f}°")
  print("=" * 78)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())