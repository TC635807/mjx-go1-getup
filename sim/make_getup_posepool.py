#!/usr/bin/env python
"""make_getup_posepool.py — 预生成"已沉降的摔倒姿态"池 (qpos/qvel)。

为什么需要它 (两个已知缺陷的交汇点):
  1. brax 的 AutoResetWrapper **不重置 env 的 info, 也不调用 env.reset**
     (brax/envs/wrappers/training.py:138-158 只替换 pipeline_state 和 obs),
     所以 num_resets_per_eval=0 时整轮训练只有 768 个**冻结**出生姿态
     -> 策略在 eval 的 256 个新姿态上泛化差。
  2. 想"每 episode 重新采"就得在 step 里跑 reset, 而 reset 里的 0.6s 自由沉降
     是 ~950ms/次(vmap 768 env), 直接卡死训练; 用 lax.cond 分支在 vmap 下会被
     摊成 select -> 两个分支都算 -> 更慢。
  折中: 把沉降**离线**做进一个姿态池, 训练时"重采"退化成一次 gather (微秒级),
  于是可以在 step 里无分支地做 jp.where(restart, pool[idx], current)。

用法:
  python sim/make_getup_posepool.py --envs 8192 --batches 4 \
      --difficulty 1.0 --orient_rand 1.0 --out models/getup_posepool.npz
"""
import argparse
import os
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.40")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import jax
import jax.numpy as jp
import numpy as np

from envs.go1_getup import Go1Getup, default_config


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--envs", type=int, default=2048,
                  help="每批 env 数。**上限受 naconmax 约束**: 起身躺地时约 33 个接触/env,"
                       "而 naconmax 是全体 world 共享的, 8192 env 会越界 ->"
                       "CUDA illegal memory access (实测)")
  ap.add_argument("--batches", type=int, default=8)
  ap.add_argument("--difficulty", type=float, default=1.0)
  ap.add_argument("--orient_rand", type=float, default=1.0)
  ap.add_argument("--drop_height", type=float, default=None,
                  help="相对地面的抛下高度 (默认用 env 的 0.5m)。'真实摔倒'档给 0.15")
  ap.add_argument("--out", type=str, required=True)
  args = ap.parse_args()

  cfg = default_config()
  cfg.njmax = 768
  cfg.naconmax = max(8 * 8192, 64 * args.envs)
  cfg.getup.difficulty = args.difficulty
  cfg.getup.orient_rand = args.orient_rand
  cfg.getup.drop_prob = 1.0
  if args.drop_height is not None:
    cfg.getup.drop_height = args.drop_height
  env = Go1Getup(cfg)
  reset = jax.jit(jax.vmap(env.reset))

  Q, V = [], []
  t0 = time.time()
  N = args.envs
  for b in range(args.batches):
    st = jax.block_until_ready(reset(jax.random.split(jax.random.PRNGKey(1000 + b), N)))
    Q.append(np.asarray(st.data.qpos, np.float32))
    V.append(np.asarray(st.data.qvel, np.float32))
    q = Q[-1]
    up = 1.0 - 2.0 * (q[:, 4] ** 2 + q[:, 5] ** 2)
    zr = q[:, 2] - np.asarray(env.terrain_height(jp.asarray(q[:, :2])))
    print("  batch %d: up_z 中位 %+.3f  z_rel 中位 %+.3f  仰卧 %.0f%%  侧躺 %.0f%%  俯卧 %.0f%%  竖直 %.0f%%" % (
        b, np.median(up), np.median(zr), 100*(up < -0.5).mean(),
        100*(np.abs(up) <= 0.5).mean(), 100*((up > 0.5) & (up <= 0.9)).mean(),
        100*(up > 0.9).mean()), flush=True)
  qpos = np.concatenate(Q, 0)
  qvel = np.concatenate(V, 0)
  np.savez_compressed(args.out, qpos=qpos, qvel=qvel,
                      difficulty=args.difficulty, orient_rand=args.orient_rand)
  print("[%.1fs] 已保存 %s  qpos%s qvel%s" % (time.time()-t0, args.out, qpos.shape, qvel.shape))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
