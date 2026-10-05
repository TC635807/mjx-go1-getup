#!/usr/bin/env python
"""probe_getup_nefc.py — njmax=256 在起身任务里会不会溢出、溢出会不会变 NaN?

背景: mujoco_warp 的 njmax 是 **per-world 的约束行数上限**。v10 把走路档从 64 提到 256,
但**起身**比走路多得多: 摔倒在地时躯干/大腿/小腿互相挤压, 自碰撞约束暴涨。
本探针用 768 env (训练档) 跑 300 步, 直接数 qpos/qvel/qacc 里的非有限值。

若 256 会溢出 -> 训练时 episode 会被 NaN 护栏杀掉, 而"腿收起来用力"恰恰是接触最密的动作
-> PPO 学到"别制造接触" = **翻正后趴平**的退化吸引子。这条要是成立, 就是 0% 的真凶之一。

用法: python -u sim/probe_nefc.py --njmax 256
"""
import argparse
import os
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.45")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import jax
import jax.numpy as jp
import numpy as np

from envs.go1_getup import Go1Getup, default_config


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--njmax", type=int, default=256)
  ap.add_argument("--naconmax", type=int, default=8 * 8192)
  ap.add_argument("--envs", type=int, default=768)
  ap.add_argument("--steps", type=int, default=300)
  ap.add_argument("--policy", type=str, default=None, help="可选 pkl; 不给就随机动作")
  ap.add_argument("--action_std", type=float, default=0.8)
  ap.add_argument("--seed", type=int, default=0)
  a = ap.parse_args()

  cfg = default_config()
  cfg.njmax = a.njmax
  cfg.naconmax = a.naconmax
  cfg.getup.difficulty = 1.0
  cfg.getup.orient_rand = 1.0
  cfg.getup.drop_prob = 1.0
  env = Go1Getup(cfg)
  N = a.envs
  print(f"njmax={a.njmax} naconmax={a.naconmax} envs={N} steps={a.steps} "
        f"policy={a.policy or 'random'} std={a.action_std}", flush=True)

  reset = jax.jit(jax.vmap(env.reset))
  step = jax.jit(jax.vmap(env.step))
  st = jax.block_until_ready(reset(jax.random.split(jax.random.PRNGKey(a.seed), N)))

  if a.policy:
    from sim.common import load_policy
    pol, _ = load_policy(a.policy)
    dummy = jp.zeros((2,), jp.uint32)
    act_fn = jax.jit(jax.vmap(lambda o: pol(o, dummy)[0]))

  def bad(data):
    q = data.qpos
    return (jp.sum(~jp.isfinite(q), axis=-1) > 0,
            jp.sum(~jp.isfinite(data.qvel), axis=-1) > 0,
            jp.sum(~jp.isfinite(data.qacc), axis=-1) > 0)

  nan_q = np.zeros(N, bool)
  nan_v = np.zeros(N, bool)
  nan_a = np.zeros(N, bool)
  nan_r = np.zeros(N, bool)
  nan_steps = np.zeros(a.steps, np.int32)
  done_cnt = np.zeros(N, np.int32)
  k = jax.random.PRNGKey(123)
  for i in range(a.steps):
    if a.policy:
      act = act_fn(st.obs["state"])
    else:
      k, kk = jax.random.split(k)
      act = a.action_std * jax.random.normal(kk, (N, env.action_size))
    st = step(st, act)
    bq, bv, ba = bad(st.data)
    br = ~jp.isfinite(st.reward)
    bq, bv, ba, br, dn = [np.asarray(x) for x in (bq, bv, ba, br, st.done)]
    nan_q |= bq; nan_v |= bv; nan_a |= ba; nan_r |= br
    done_cnt += dn.astype(np.int32)
    nan_steps[i] = int((bq | bv | ba | br).sum())
    if (i + 1) % 50 == 0:
      print(f"    step {i+1:4d}: 本步 NaN env 数 {nan_steps[i]:4d}  "
            f"累计 q{int(nan_q.sum()):4d} v{int(nan_v.sum()):4d} "
            f"a{int(nan_a.sum()):4d} r{int(nan_r.sum()):4d}  "
            f"done 累计 {int(done_cnt.sum())}", flush=True)
  print(f"  结果 njmax={a.njmax}: 出现过 NaN/非有限的 env: "
        f"qpos {int(nan_q.sum())}/{N} ({100*nan_q.mean():.1f}%)  "
        f"qvel {int(nan_v.sum())}  qacc {int(nan_a.sum())}  reward {int(nan_r.sum())}", flush=True)
  print(f"  非有限 env-step 总数: {int(nan_steps.sum())}/{a.steps*N} = "
        f"{100*nan_steps.sum()/(a.steps*N):.4f}%", flush=True)
  print(f"  env 终止 (出界/NaN护栏) 累计 {int(done_cnt.sum())}/{N}", flush=True)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
