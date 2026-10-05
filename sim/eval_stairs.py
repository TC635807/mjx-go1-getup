#!/usr/bin/env python
"""上下台阶成功率评估 (v23 新增)。

为什么必须固定摆放 (§24.3 的教训): "随机出生 + 爬升>10cm 占比" 这个指标主要
在测"随机 yaw 下有没有正好遇到楼梯" (几何天花板只有 10.5%), 无法区分"没学会"
与"没遇到"。所以把机器人**直接放到楼梯口正对楼梯**, 量成功率。

地形里两条楼梯的位置是**固定**的 (sim/gen_terrain.py: flights=(ext*0.05,-ext*0.80,'x')
和 (-ext*0.80, ext*0.05, 'y'), ext=6.0):
  A: 沿 +x, x0=0.3, y0=-4.8, 宽~1.96m, 5 级 x 0.06m = 总高 0.30m, 进深 0.55m
  B: 沿 +y, x0=-4.8, y0=0.3
判据: 15s 内足下地形高度上升(下降) >= 0.24m (即 4/5 级) 且**从未摔倒终止**。

用法:
  python sim/eval_stairs.py --pkl policies/go1_walk_policy.pkl --label v22
"""
import argparse
import os
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.25")
os.environ.setdefault("GALLIUM_DRIVER", "d3d12")
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + "/sim")

import jax                      # noqa: E402
import jax.numpy as jp          # noqa: E402
import numpy as np              # noqa: E402

from envs.go1_walk import Go1Walk, default_config   # noqa: E402
from watch_v20_fast import load_policy              # noqa: E402

EXT = 6.0
X0A, Y0A = EXT * 0.05, -EXT * 0.80          # 楼梯 A: 沿 +x
X0B, Y0B = -EXT * 0.80, EXT * 0.05          # 楼梯 B: 沿 +y
SPAN = 5 * 0.55                              # 5 级 x 0.55m
WIDTH = max(1.8, 2 * (1.5 * 0.30 / np.tan(np.radians(30.0))) + 0.4)
CLIMB = 0.24                                 # 成功判据: 4/5 级


def build():
    cfg = default_config()
    cfg.foot_soft_contact = False
    cfg.terrain = True
    cfg.height_scan.enable = True
    cfg.random_init.enable = False           # 手动摆位
    cfg.command_config.frame = "body"
    cfg.healthy_roll_range = 17.5
    cfg.healthy_pitch_range = 17.5
    cfg.reward_config.scales.feet_phase = 2.0
    cfg.reward_config.max_foot_height = cfg.reward_config.gait_swing_height
    cfg.graph_mode = "WARP_STAGED"
    return Go1Walk(cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--n_envs", type=int, default=32)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--cmd", type=float, default=0.6)
    args = ap.parse_args()

    env = build()
    policy, dims = load_policy(args.pkl if os.path.isabs(args.pkl)
                               else os.path.join(ROOT, args.pkl))
    label = args.label or os.path.basename(args.pkl)
    print("=" * 74)
    print("上下台阶成功率 | %s | obs=%s | n_envs=%d | %d 步 (%.1fs) | 指令 vx=%.2f"
          % (label, dims, args.n_envs, args.steps, args.steps * env.dt, args.cmd))
    print("  判据: 地形高度变化 >= %.2fm 且从未终止  (楼梯总高 0.30m, 宽 %.2fm)"
          % (CLIMB, WIDTH))

    rng0 = jp.zeros((2,), dtype=jp.uint32)
    act_v = jax.jit(jax.vmap(lambda o: policy(o, rng0)[0]))
    reset_v = jax.jit(jax.vmap(env.reset))
    step_v = jax.jit(jax.vmap(env.step), donate_argnums=0)

    def place(st, xy, yaw, cmd):
        z = env.terrain_height(xy[None, :])[0] + 0.35
        half = yaw * 0.5
        quat = jp.stack([jp.cos(half), jp.zeros(()), jp.zeros(()), jp.sin(half)])
        qpos = jp.concatenate([jp.array([xy[0], xy[1], z]), quat, st.data.qpos[7:]])
        data = st.data.replace(qpos=qpos, qvel=jp.zeros_like(st.data.qvel))
        info = dict(st.info)
        info["command"] = cmd
        info["h_spawn"] = env.terrain_height(xy[None, :])[0]
        obs = env._get_obs(data, info)
        return st.tree_replace({"data": data, "obs": obs, "info": info})

    place_v = jax.jit(jax.vmap(place))

    cases = [
        ("A 上台阶 (+x)", np.array([X0A - 0.45, Y0A + WIDTH / 2]), 0.0, +1),
        ("A 下台阶 (-x)", np.array([X0A + SPAN + 0.45, Y0A + WIDTH / 2]), np.pi, -1),
        ("B 上台阶 (+y)", np.array([X0B + WIDTH / 2, Y0B - 0.45]), np.pi / 2, +1),
        ("B 下台阶 (-y)", np.array([X0B + WIDTH / 2, Y0B + SPAN + 0.45]), -np.pi / 2, -1),
    ]

    n = args.n_envs
    print("  %-14s %10s %12s %10s %10s" %
          ("情形", "成功率", "中位|Δh|(m)", "摔倒率", "最大Δh(m)"))
    all_succ = []
    for name, base_xy, base_yaw, sign in cases:
        keys = jax.random.split(jax.random.PRNGKey(0), n)
        st = reset_v(keys)
        jitter = jax.random.uniform(jax.random.PRNGKey(7), (n, 2),
                                    minval=-0.12, maxval=0.12)
        xy = jp.asarray(base_xy)[None, :] + jitter
        yaw = jp.asarray(base_yaw) + jax.random.uniform(
            jax.random.PRNGKey(8), (n,), minval=-0.1, maxval=0.1)
        cmd = jp.tile(jp.asarray(np.array([args.cmd, 0.0, 0.0], np.float32)), (n, 1))
        st = place_v(st, xy, yaw, cmd)
        h0 = np.asarray(env.terrain_height(xy))

        ever_done = np.zeros(n, dtype=bool)
        best = np.full(n, -np.inf)
        for i in range(args.steps):
            st.info["command"] = jp.asarray(
                np.array([[args.cmd, 0.0, 0.0]] * n, np.float32))
            st = step_v(st, act_v(st.obs["state"]))
            q = np.asarray(st.data.qpos)
            h = np.asarray(env.terrain_height(jp.asarray(q[:, :2])))
            gain = (h - h0) * sign
            best = np.maximum(best, gain)
            ever_done |= np.asarray(st.done) > 0.5
        succ = (best >= CLIMB) & (~ever_done)
        r = 100.0 * succ.mean()
        all_succ.append(r)
        print("  %-14s %9.1f%% %12.3f %9.1f%% %10.3f" %
              (name, r, float(np.median(np.abs(best))), 100 * ever_done.mean(),
               float(np.max(best))))
    print("  %-14s %9.1f%%" % ("**平均**", float(np.mean(all_succ))))
    print("=" * 74)


if __name__ == "__main__":
    main()
