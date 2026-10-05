#!/usr/bin/env python
"""v19 定期测试: 把某个 checkpoint 放进**测试环境**跑标准化评估。

与 brax 内置 eval 的区别 (为什么还要单独做)
------------------------------------------
brax 的 `eval/episode_reward` 用的是**训练同款**环境 (含随机出生/推挤/噪声),
所以那个数字混了两件事: 策略变好 + 出生点运气。而且每个 checkpoint 用的
出生点不同, 5M vs 10M 的数字**不可直接比较**。

本脚本用**固定 seed + 可复现的出生点集**, 让不同 checkpoint 面对完全相同的地形
与初始状态, 于是数字可比。同时给出分项诊断 (走得远不远 / 摔没摔 / 爬没爬高)。

用法:
  python sim/eval_walk.py --ckpt logs/walk/checkpoints/<step>
  python sim/eval_walk.py --pkl policies/go1_walk_policy.pkl
"""
import argparse
import json
import os
import sys

import jax
import jax.numpy as jp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.3")

import mujoco
from mujoco import mjx

from envs.go1_walk import Go1Walk, default_config

# 测试用固定 seed (不同 checkpoint 共用 -> 出生点/指令完全一致)
TEST_SEED = 20260915
N_ENVS = 512


def infer_layers_and_dims(params):
  """从参数形状反推隐层与 obs 维度 (v19 有 83/154, v18 是 48/119)。"""
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
  layers = tuple(found[i] for i in sorted(found)) if found else (128, 128)

  def first_dim(node):
    best = {}

    def w(n):
      if isinstance(n, (tuple, list)):
        for v in n:
          w(v)
        return
      if not hasattr(n, "items"):
        return
      for k, v in n.items():
        if (k == "hidden_0" and hasattr(v, "get") and "kernel" in v
            and hasattr(v["kernel"], "shape")):
          best.setdefault("dim", int(v["kernel"].shape[0]))
        w(v)

    w(node)
    return best.get("dim")

  if isinstance(params, (tuple, list)) and len(params) >= 3:
    ns, nv = first_dim(params[1]) or 48, first_dim(params[2]) or 119
  else:
    ns, nv = (first_dim(params) or 48), 119
  return layers, ns, nv


def load_policy(path):
  """path 可以是 pkl 或 orbax checkpoint 目录。"""
  from brax.training.agents.ppo import networks as ppo_networks
  if os.path.isdir(path):
    from brax.training.agents.ppo import checkpoint as ppo_ckpt
    cfg = json.load(open(os.path.join(path, "ppo_network_config.json")))
    kw = cfg["network_factory_kwargs"]
    params = ppo_ckpt.load(path)
    layers, ns, nv = infer_layers_and_dims(params)
    net = ppo_networks.make_ppo_networks(
        observation_size={"state": (ns,), "privileged_state": (nv,)},
        action_size=cfg["action_size"],
        policy_hidden_layer_sizes=tuple(kw["policy_hidden_layer_sizes"]),
        value_hidden_layer_sizes=tuple(kw["value_hidden_layer_sizes"]),
        policy_obs_key=kw["policy_obs_key"],
        value_obs_key=kw["value_obs_key"],
        distribution_type=kw.get("distribution_type", "tanh_normal"),
        activation=jax.nn.tanh)
  else:
    import pickle
    with open(path, "rb") as f:
      params = pickle.load(f)
    layers, ns, nv = infer_layers_and_dims(params)
    net = ppo_networks.make_ppo_networks(
        observation_size={"state": (ns,), "privileged_state": (nv,)},
        action_size=12,
        policy_hidden_layer_sizes=layers,
        value_hidden_layer_sizes=layers,
        policy_obs_key="state",
        value_obs_key="privileged_state",
        distribution_type="normal",
        activation=jax.nn.tanh)
  return ppo_networks.make_inference_fn(net)(params, deterministic=True), ns, nv


def make_test_env(note, n_state_hint=None):
  """测试环境: 地形 + 感知 (与训练一致), 但关掉推挤/观测噪声, 便于看真实能力。

  `n_state_hint`: 策略的 actor obs 维度。用来决定是否打开相位奖励 ——
  相位奖励会在 obs 里加 8 维 (cos,sin x 4足), 关了维度就不匹配 (91 vs 83)。
  这样 v19 (83) 与 v20 (91) 的策略都能用同一个脚本评估。
  """
  cfg = default_config()
  cfg.terrain = True
  cfg.height_scan.enable = True
  cfg.random_init.enable = True       # 随机出生覆盖全地形, 但不带额外扰动
  cfg.push_config.enable = False
  cfg.obs_noise.enable = False
  cfg.command_config.sample = True
  cfg.command_config.frame = "body"
  cfg.foot_soft_contact = False
  cfg.healthy_roll_range = 17.5
  cfg.healthy_pitch_range = 17.5
  cfg.reward_config.scales.feet_slip = -0.5
  cfg.reward_config.scales.feet_height = -1.0
  cfg.reward_config.scales.feet_dangle = -0.5
  cfg.reward_config.air_time_threshold = 0.2
  cfg.reward_config.air_time_max = 0.5
  cfg.reward_config.scales.base_height = -200.0
  cfg.reward_config.scales.feet_impact = -2.0
  # 83 = 48 + 35 (无相位); 91 = 48 + 35 + 8 (相位)
  if n_state_hint is not None and n_state_hint > 83:
    cfg.reward_config.scales.feet_phase = 2.0
    print(f"  [v20] 策略 {n_state_hint} 维 -> 打开相位奖励 (obs +8 维)")
  print(f"  测试环境: {note}")
  return Go1Walk(cfg)


def evaluate(env, policy, n_envs=N_ENVS, seed=TEST_SEED):
  reset_j = jax.jit(jax.vmap(env.reset))
  step_j = jax.jit(jax.vmap(env.step))
  keys = jax.random.split(jax.random.PRNGKey(seed), n_envs)
  st = reset_j(keys)

  # 记录初始状态 (用于"是否爬升"判断)
  q0 = np.asarray(st.data.qpos)
  h0 = np.asarray(jax.vmap(env.terrain_height)(
      jp.asarray(q0[:, :2], dtype=jp.float32)))
  z_rel0 = q0[:, 2] - h0

  T = env._config.episode_length
  rew, step_i = np.zeros(n_envs), np.zeros(n_envs, dtype=int)
  alive = np.ones(n_envs, dtype=bool)
  # 速度统计: **必须用机体系速度**。
  # 测 qvel[:,0] (全局 x) 是错的 —— 随机出生带随机 yaw, 即使完美行走,
  # 全局 x 速度对随机的朝向求平均也≈0, 会假象成"原地不动"。
  # 训练用的指令也是机体系 (command_config.frame="body"), 所以要用
  # R^T @ v_global 与指令同空间比较。
  vx_body_sum = np.zeros(n_envs)      # 机体系前向速度
  vxy_norm_sum = np.zeros(n_envs)     # 机体系水平速度模长 (不看方向)
  track_err_sum = np.zeros(n_envs)    # ||v_cmd[:2] - v_body[:2]||
  dist_sum = np.zeros(n_envs)         # 世界系实际位移 (标量路程)
  prev_xy = None
  zrel_min = np.full(n_envs, np.inf)
  zrel_max = np.full(n_envs, -np.inf)
  ground_max = h0.copy()
  fell = np.zeros(n_envs, dtype=bool)

  for i in range(T):
    act, _ = policy(st.obs["state"], jax.random.PRNGKey(i))
    st = step_j(st, act)
    r = np.asarray(st.reward)
    d = np.asarray(st.done)
    q = np.asarray(st.data.qpos)
    qv = np.asarray(st.data.qvel)
    # 机体系速度: 用四元数取 yaw, 把世界系水平速度转回机体系
    quat = q[:, 3:7]
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    c, s = np.cos(yaw), np.sin(yaw)
    vbx = c * qv[:, 0] + s * qv[:, 1]
    vby = -s * qv[:, 0] + c * qv[:, 1]
    cmd = np.asarray(st.info["command"])
    h = np.asarray(jax.vmap(env.terrain_height)(
        jp.asarray(q[:, :2], dtype=jp.float32)))
    zr = q[:, 2] - h
    live = alive & (d < 0.5)
    rew[live] += r[live]
    vx_body_sum[live] += vbx[live]
    vxy_norm_sum[live] += np.hypot(vbx, vby)[live]
    track_err_sum[live] += np.linalg.norm(cmd[live, :2] - np.stack(
        [vbx, vby], axis=-1)[live], axis=-1)
    if prev_xy is not None:
      dist_sum[live] += np.linalg.norm(
          q[live, :2] - prev_xy[live], axis=-1)
    prev_xy = q[:, :2].copy()
    zrel_min[live] = np.minimum(zrel_min[live], zr[live])
    zrel_max[live] = np.maximum(zrel_max[live], zr[live])
    ground_max[live] = np.maximum(ground_max[live], h[live])
    step_i[live] += 1
    fell |= (d > 0.5) & alive
    alive &= (d < 0.5)

  nz = np.maximum(step_i, 1)
  ep_len = step_i.astype(float)
  return {
      "n": n_envs,
      "reward_mean": float(rew.mean()),
      "reward_per_step": float((rew / nz).mean()),
      "ep_len_mean": float(ep_len.mean()),
      "ep_len_p50": float(np.percentile(ep_len, 50)),
      "survive_rate": float((~fell).mean()),
      "fell_rate": float(fell.mean()),
      # 机体系速度 (与指令同空间)
      "avg_vx_body": float((vx_body_sum / nz).mean()),
      "avg_speed_xy": float((vxy_norm_sum / nz).mean()),
      "avg_track_err": float((track_err_sum / nz).mean()),
      # dist_sum 累计的是每步位移长度 (米/步), 除以步数后是**平均位移速度**
      "avg_path_len": float((dist_sum / nz).mean()),
      "ground_climb_mean": float((ground_max - h0).mean()),
      "ground_climb_max": float((ground_max - h0).max()),
      "frac_ground_above_02": float(((ground_max - h0) > 0.02).mean()),
      "frac_ground_above_10": float(((ground_max - h0) > 0.10).mean()),
      "zrel_min_mean": float(zrel_min[np.isfinite(zrel_min)].mean()),
  }


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--ckpt", default=None, help="orbax checkpoint 目录")
  ap.add_argument("--pkl", default=None, help="策略 pkl")
  ap.add_argument("--label", default=None, help="打印用标签 (默认取路径名)")
  ap.add_argument("--n_envs", type=int, default=N_ENVS)
  args = ap.parse_args()
  path = args.ckpt or args.pkl
  if not path:
    raise SystemExit("必须给 --ckpt 或 --pkl")
  label = args.label or os.path.basename(path.rstrip("/"))

  if not os.path.exists(path):
    print(f"[跳过] 不存在: {path}")
    return 2

  print(f"=== 测试 {label} ===")
  policy, ns, nv = load_policy(path)
  print(f"  策略 obs: state={ns} privileged={nv}")
  env = make_test_env(f"terrain+scan, 随机出生, 无推挤/无观测噪声, "
                      f"固定 seed={TEST_SEED}, {args.n_envs} 环境",
                      n_state_hint=ns)

  # 维度一致性
  env_ns = env.observation_size["state"][0]
  if env_ns != ns:
    print(f"  [错误] 策略 {ns} 维 vs 环境 {env_ns} 维, 不匹配")
    return 3

  m = evaluate(env, policy, n_envs=args.n_envs)
  print()
  print(f"  {'指标':<26} {'值':>12}")
  print("  " + "-" * 42)
  print(f"  {'每 episode 奖励':<26} {m['reward_mean']:>12.2f}")
  print(f"  {'每步奖励':<26} {m['reward_per_step']:>12.4f}")
  print(f"  {'平均 episode 长度':<26} {m['ep_len_mean']:>12.1f} 步 "
        f"({m['ep_len_mean']*env.dt:.1f}s / {env._config.episode_length*env.dt:.0f}s)")
  print(f"  {'episode 长度中位数':<26} {m['ep_len_p50']:>12.1f} 步")
  print(f"  {'存活率 (未终止)':<26} {100*m['survive_rate']:>11.1f}%")
  print()
  print("  --- 速度 (机体系, 与指令同空间) ---")
  print(f"  {'平均前向速度 vx_body':<26} {m['avg_vx_body']:>12.3f} m/s")
  print(f"  {'平均水平速度模长':<26} {m['avg_speed_xy']:>12.3f} m/s")
  print(f"  {'速度跟踪误差 (越小越好)':<26} {m['avg_track_err']:>12.3f} m/s")
  print(f"  {'平均位移速度 (世界系)':<26} {m['avg_path_len']:>12.3f} m/s")
  print()
  print("  --- 地形能力 (关键) ---")
  print(f"  {'足下地面最大爬升 (均值)':<26} {m['ground_climb_mean']:>12.4f} m")
  print(f"  {'足下地面最大爬升 (最好)':<26} {m['ground_climb_max']:>12.4f} m")
  print(f"  {'爬升 >2cm 的环境占比':<26} {100*m['frac_ground_above_02']:>11.1f}%")
  print(f"  {'爬升 >10cm 的环境占比':<26} {100*m['frac_ground_above_10']:>11.1f}%")
  print(f"  {'躯干相对站高最低 (均值)':<26} {m['zrel_min_mean']:>12.4f} m")
  print()
  print("  JSON: " + json.dumps(m, ensure_ascii=False))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
