#!/usr/bin/env python
"""probe_getup_v2.py — 自检移植版环境 (envs/go1_getup_v2.py)。"""
import argparse
import os
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.40")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import jax
import jax.numpy as jp
import numpy as np

from envs.go1_getup_v2 import Go1GetupV2, default_config

HOME = np.array([0.0, 0.8, -1.5, 0.0, 0.8, -1.5, 0.0, 1.0, -1.5, 0.0, 1.0, -1.5],
                np.float32)


def terms_of(st, steps, env, n_term_ok):
  """把 st.metrics 里 reward/* 的平均每步值取出来。"""
  out = {}
  for k, v in st.metrics.items():
    if k.startswith("reward/"):
      out[k.replace("reward/", "")] = float(np.mean(np.asarray(v)))
  return out


def report(name, st, env, zref, acc, nan, dones, steps, cfg, extra=""):
  q = np.asarray(st.data.qpos)
  zr = q[:, 2] - zref(q)
  upw = 1.0 - 2.0 * (q[:, 4] ** 2 + q[:, 5] ** 2)
  stand = (np.abs(zr - cfg.getup2.desired_base_height) < 0.06) & (upw > 0.90)
  print("  --- " + name)
  print("      NaN 帧数 %d  出界率 %.1f%%  末态 z_rel 中位 %+.3f  up_z 中位 %+.3f  "
        "站住(世界) %.1f%% %s" % (nan, 100 * dones.mean(), np.median(zr),
                                  np.median(upw), 100 * stand.mean(), extra))
  keys = sorted(acc, key=lambda k: -abs(acc[k]))
  print("      平均每步逐项: " + "  ".join(
      "%s=%+.3f" % (k, acc[k] / steps) for k in keys))
  tot = sum(acc[k] for k in keys) / steps
  print("      平均每步合计 %+.3f   (~%+.0f / episode)" % (tot, tot * steps))


def rollout(step, st0, act, N, steps, env, zref, cfg):
  st = st0
  acc = {}
  nan = 0
  dones = np.zeros(N, bool)
  for i in range(steps):
    st = step(st, act if not callable(act) else act(i))
    for k, v in st.metrics.items():
      if k.startswith("reward/"):
        acc[k] = acc.get(k, 0.0) + float(np.mean(np.asarray(v)))
    nan += int(np.sum(~np.isfinite(np.asarray(st.data.qpos)).all(-1)))
    dones |= np.asarray(st.done) != 0
  return st, acc, nan, dones


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--envs", type=int, default=256)
  ap.add_argument("--steps", type=int, default=300)
  ap.add_argument("--seed", type=int, default=0)
  a = ap.parse_args()
  cfg = default_config()
  env = Go1GetupV2(cfg)
  N, S = a.envs, a.steps
  g = cfg.getup2
  print("[配置] obs state=%d priv=%d  history=%d  clip_a=%.1f scale=%.2f alpha=%.2f" % (
      env.observation_size["state"][0],
      env.observation_size["privileged_state"][0], g.history,
      g.action_clip, g.action_scale, g.action_filter_alpha), flush=True)
  print("        期望站立高度 %.2f m  门控 %.1f 度  出生 z=terrain+%.2f  xy±%.1f" % (
      g.desired_base_height, g.ori_gate_deg, g.spawn_height, g.spawn_xy_range))

  reset = jax.jit(jax.vmap(env.reset))
  step = jax.jit(jax.vmap(env.step))
  st0 = jax.block_until_ready(reset(jax.random.split(jax.random.PRNGKey(a.seed), N)))

  def zref(q):
    return np.asarray(env.terrain_height(jp.asarray(q[:, :2])))

  q0 = np.asarray(st0.data.qpos)
  print("  出生: 世界 up_z 中位 %+.3f  z_rel 中位 %+.3f" % (
      np.median(1.0 - 2.0 * (q0[:, 4] ** 2 + q0[:, 5] ** 2)),
      np.median(q0[:, 2] - zref(q0))), flush=True)

  # 1) a = 0 (target = 名义站姿)
  z = np.zeros((N, 12), np.float32)
  st, acc, nan, dones = rollout(step, st0, jp.asarray(z), N, S, env, zref, cfg)
  report("a=0 (target = 名义站姿)", st, env, zref, acc, nan, dones, S, cfg)

  # 2) 定值站立偏置 (standpose 探针扫到的 thigh+0.04 / calf+0.15)
  bias = np.zeros(12, np.float32)
  for i in (1, 4, 7, 10):
    bias[i] = 0.04
  for i in (2, 5, 8, 11):
    bias[i] = 0.15
  act = jp.asarray(np.tile(bias / g.action_scale, (N, 1)).astype(np.float32))
  st, acc, nan, dones = rollout(step, st0, act, N, S, env, zref, cfg)
  report("定值站立偏置 thigh+0.04 calf+0.15", st, env, zref, acc, nan, dones, S, cfg)

  # 3) 随机动作 (PPO 早期)
  k = jax.random.PRNGKey(7)
  kk = jax.random.split(k, S)
  def rand_act(i):
    return jax.random.normal(kk[i], (N, 12))
  st, acc, nan, dones = rollout(step, st0, rand_act, N, S, env, zref, cfg)
  report("a ~ N(0,1) 随机", st, env, zref, acc, nan, dones, S, cfg)

  # 4) 奖励面排序: 把状态**强行摆成站立**再走一步, 看两个正项是否接近满分
  q = np.asarray(st0.data.qpos).copy()
  q[:, 2] = zref(q) + g.desired_base_height
  q[:, 3:7] = np.array([1.0, 0.0, 0.0, 0.0], np.float32)
  q[:, 7:19] = HOME
  d = st0.data.replace(qpos=jp.asarray(q), qvel=jp.zeros_like(st0.data.qvel))
  st_stand = st0.replace(data=d)
  st_stand = jax.block_until_ready(step(st_stand, jp.zeros((N, 12))))
  t = terms_of(st_stand, 1, env, True)
  qq = np.asarray(st_stand.data.qpos)
  upw = 1.0 - 2.0 * (qq[:, 4] ** 2 + qq[:, 5] ** 2)
  print("  --- 强行摆成站立态 (z=terrain+%.2f, 朝向竖直, 关节=home) 再走一步" %
        g.desired_base_height)
  print("      末态 z_rel 中位 %+.3f  up_z 中位 %+.3f" % (
      np.median(qq[:, 2] - zref(qq)), np.median(upw)))
  print("      逐项: " + "  ".join("%s=%+.3f" % (k, t[k])
                                   for k in sorted(t, key=lambda x: -abs(t[x]))))
  print("      两个正项合计 %+.3f  (满分 2.0)" %
        (t.get("track_orientation", 0) + t.get("track_height", 0)))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
