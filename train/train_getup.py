#!/usr/bin/env python
"""训练 Go1 起身策略 (地形版, sim/getup 的调度器方案)。

为什么单独一个训练入口 (而不是给 train_go1.py 加开关):
  train_go1.py 里 400~700 行全是**走路专属**的配置接线 (奖励开关/地形/指令/
  restore 维度自检...), 给起身任务复用会把那条已经被 v19~v23 验证过的链路搅乱。
  起身是**单任务**, PPO 超参只需要一组, 所以这里写成独立入口, 超参照抄 v23 那一档
  (sb3_full 的账目), 保证两个策略的优化制度一致。

与 train_go1.py 的对应关系:
  num_envs=768, unroll=32      -> rollout 24576 (= SB3 的 n_envs×n_steps)
  batch_size=2 序列 × 32 步    -> minibatch 64 条 (= SB3 的 batch=64)
  num_minibatches=384          -> 768 = 384×2 恰好铺满
  updates_per_batch=10         -> 10 epochs (= SB3 n_epochs)
  lr=3e-4, entropy=0, gamma=0.99, clip=0.2, grad_clip=0.5, 网络 (128,128) tanh

训练进度的"真信号"不是 eval_reward, 而是 sim/eval_getup.py 的**成功率/up_z** ——
奖励里 orientation + torso_height 占主导, 与"站起来了没有"高度相关, 但只当参考。

用法:
  # 只做静态自检 (建 env + 建网络, **不训练**)
  python -m train.train_getup --dry_run
  # 1M 步 smoke
  python -u -m train.train_getup --num_timesteps 1000000 \
      --num_evals 2 --save_name go1_getup_policy_smoke.pkl --logdir logs/getup_v2
  # 正式 50M
  python -u -m train.train_getup --num_timesteps 50000000 \
      --num_evals 20 --save_name go1_getup_policy.pkl --logdir logs/getup_v2
"""
import argparse
import functools
import os
import pickle
import shutil
import sys
import time

# 必须在 import jax **之前** 设:XLA 会按这个比例一次性预留显存。
# 踩过的坑 (§29.15): 不设时默认预留 75%% (8GiB 卡 -> 6.1GiB), 剩给 warp 的 CUDA graph
# 池只有 ~2GiB; 加上 --num_resets_per_eval>0 会反复调 reset, 跑到 1.5M 步时
# "Failed to allocate 83MB on device 'cuda:0'" 直接崩。给 XLA 留 60%% 就没事了。
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.45")

import jax
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from ml_collections import config_dict
from mujoco_playground import locomotion, wrapper

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from envs.go1_getup import Go1Getup, default_config  # noqa: E402

locomotion.register_environment("Go1Getup", Go1Getup, default_config)

POLICY_DIR = os.path.join(_ROOT, "policies")
WALK_PKL = os.path.join(POLICY_DIR, "go1_walk_policy.pkl")


def _first_kernel_dim(node):
  """从 brax PPO 参数里反推第一层输入维 (与 sim/view_go1.py 的 infer_obs_sizes 同法)。"""
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


def check_walk_obs_match(env, path=WALK_PKL):
  """自检: 起身策略与走路策略必须**吃同一个 obs 向量** (调度器合并的前提)。

  返回 (ok, 说明字符串)。这是本脚本最重要的一条断言 —— 两个策略 obs 不同形
  也能各自训, 但查看器里就没法用"同一个 obs 换 act 来源"来接。
  """
  if not os.path.exists(path):
    return True, f"(跳过: 找不到走路权重 {path})"
  with open(path, "rb") as f:
    params = pickle.load(f)
  w_state = _first_kernel_dim(params[1])
  w_priv = _first_kernel_dim(params[2])
  e_state = env.observation_size["state"][0]
  e_priv = env.observation_size["privileged_state"][0]
  ok = (w_state == e_state) and (w_priv == e_priv)
  return ok, (f"走路 {w_state}/{w_priv}  vs  起身 {e_state}/{e_priv}  "
              f"{'一致 ✅' if ok else '不一致 ❌ (合并前必须对齐)'}")


def _load_init_pkl(path):
  """读 BC 出来的 (normalizer, policy, value) 三元组 (相对路径按项目根解析)。"""
  p = path if os.path.isabs(path) else os.path.join(_ROOT, path)
  with open(p, "rb") as f:
    params = pickle.load(f)
  print(f"  [init] warm-start 参数: {p}  ({len(params)} 元组)")
  return params


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--num_timesteps", type=int, default=50_000_000)
  ap.add_argument("--num_evals", type=int, default=20)
  ap.add_argument("--num_eval_envs", type=int, default=128)
  ap.add_argument("--episode_length", type=int, default=300,
                  help="控制步; 6s (playground getup 同值)")
  ap.add_argument("--num_envs", type=int, default=768)
  ap.add_argument("--unroll_length", type=int, default=32)
  ap.add_argument("--num_minibatches", type=int, default=384)
  ap.add_argument("--updates_per_batch", type=int, default=10)
  ap.add_argument("--lr", type=float, default=3e-4)
  ap.add_argument("--entropy", type=float, default=0.0)
  ap.add_argument("--init_noise_std", type=float, default=1.0,
                  help="策略动作噪声的初值 (brax 的 init_noise_std, std_param 直接取它)。"
                       "**§29.16 根因**: 默认 1.0 意味着每个采样动作带 ±1.0 噪声 = "
                       "目标关节每步抖 ±0.5 rad (action_scale=0.5) -> 任何需要精确计时的"
                       "起身动作都做不出来, PPO 只能收敛到对噪声最鲁棒的策略: 翻正+趴平不动。"
                       "微调/BC warm-start 时应调到 0.1~0.2")
  ap.add_argument("--discounting", type=float, default=0.99)
  ap.add_argument("--clipping_epsilon", type=float, default=0.2)
  ap.add_argument("--max_grad_norm", type=float, default=0.5)
  ap.add_argument("--layers", type=str, default="128,128")
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--logdir", type=str, default="logs/getup_v2")
  ap.add_argument("--save_name", type=str, default="go1_getup_policy.pkl")
  ap.add_argument("--graph_mode", type=str, default="WARP",
                  choices=["WARP", "WARP_STAGED", "WARP_STAGED_EX", "NONE"],
                  help="训练侧用 WARP (地址稳定, 图缓存命中); 查看器才需要 WARP_STAGED")
  ap.add_argument("--walk_pkl", type=str, default=WALK_PKL,
                  help="用来做 obs 同形自检的走路权重")
  ap.add_argument("--settle_time", type=float, default=None,
                  help="摔倒姿态的自由沉降时长 (秒)。**性能敏感**: brax 的 "
                       "AutoResetWrapper 是无条件 reset+select (vmap 下分支不省), "
                       "reset 里的物理会被每一步都跑一遍 -> 沉降越久训练越慢")
  ap.add_argument("--drop_prob", type=float, default=None,
                  help="摔倒姿态的占比 (默认 0.75)")
  ap.add_argument("--torso_height_w", type=float, default=None,
                  help="覆盖 reward_config.scales.torso_height (把'站得更高'变成"
                       "直接单调的收益)")
  ap.add_argument("--orientation_w", type=float, default=None,
                  help="覆盖 scales.orientation。§29.14: v25 用 torso_height=6 而 "
                       "orientation=1, 策略就'牺牲倾角换高度'(up_z>0.99 的比例从 "
                       "69%% 掉到 2%%) -> 两个都得提上去")
  ap.add_argument("--posture_w", type=float, default=None,
                  help="覆盖 scales.posture (关节贴近名义站姿的稠密项)")
  ap.add_argument("--stand_still_w", type=float, default=None,
                  help="覆盖 scales.stand_still")
  ap.add_argument("--num_resets_per_eval", type=int, default=0,
                  help="§29.15 关键旋钮: brax 在**每个 eval 段内**重置环境几次。"
                       "=0 时 (full_reset=False 缓存 first_data) 出生姿态整轮只算一次, "
                       "全程只有 768 个固定摔倒姿态; >0 则每段重置 K 次 -> K*(num_evals-1) "
                       "组全新姿态, 而且**没有每步成本** (reset 只在段间调用)")
  ap.add_argument("--difficulty", type=float, default=1.0,
                  help="§29.15 出生难度 α (每次运行一个**固定值**)。α=0 -> 朝向随机但 "
                       "关节=名义站姿 (翻过去就站着, 易); α=1 -> 关节也全随机 (真任务)。"
                       "课程用**多次运行**手动推进 (配 --restore 续训): 因为 difficulty "
                       "是 trace 期常量, jax.jit 缓存命中时改它不会重编 "
                       "(sim/probe_reset_cost.py 实测 0.8s 命中), 运行中改难度无效")
  ap.add_argument("--orient_rand", type=float, default=1.0,
                  help="§29.15 第二根课程轴 β: 出生朝向随机程度。β=0 -> 身体竖直出生"
                       "(只考'把腿撑起来'= v24/v25 唯一没学会的技能); β=1 -> 朝向全随机"
                       "(还要先翻正, v24 已会)。课程 = 先 β=0 再 β=1 两次运行")
  ap.add_argument("--init_pkl", type=str, default=None,
                  help="warm-start: 直接喂一份 (normalizer, policy, value) 的 pkl 当初始"
                       "参数 (brax 的 restore_params)。§29.15: BC 出来的权重走这里, "
                       "让 PPO 从'会起身'开始微调, 而不是从零探索")
  ap.add_argument("--restore", type=str, default=None,
                  help="从 brax checkpoint 目录续训 (多阶段课程用)。注意 brax 的 "
                       "checkpoint **不含优化器状态**, Adam 会从零重建")
  ap.add_argument("--naconmax", type=int, default=0,
                  help="覆盖 cfg.naconmax。语义是**全体 world 共享的接触槽总数** "
                       "(mujoco_warp types.py: shared across all worlds), 不是 per-env。"
                       "本项目 8*8192=65536; 768 env 实测峰值 3700 -> 有余量, 但调小"
                       "实测不掉时间 (只省内存, 见 §29.12)")
  ap.add_argument("--task", choices=["getup", "getup_res", "getup_v2", "walk"],
                  default="getup",
                  help="getup = 增量动作的纯起身 (旧); getup_res = 关键帧先验+学习残差 (v30); "
                       "**getup_v2 = 移植公开配方** (动作锚定名义站姿 + 宽高斯 + 地形相对朝向 "
                       "+ 观测历史, envs/go1_getup_v2.py); walk = 诊断用对照")
  ap.add_argument("--action_clip", type=float, default=None,
                  help="[task=getup_v2] 动作幅度上限 C: target = default + 0.5*clip(a,+-C)。"
                       "开源给 Go2 用的是 3.0 (= default+-1.5 rad), 但 **Go1 的 thigh 行程 "
                       "是 5.2 rad**, 关键帧起身要 thigh=4.501 —— sim/probe_v2_range.py 实测: "
                       "C=3 时关键帧只有 11.3%%, C=8 (= default+-4.0) 时 37.1%%")
  ap.add_argument("--joint_alpha", type=float, default=None,
                  help="[task=getup_v2] 出生关节难度 alpha: joints=(1-a)*default+a*U(限位)。"
                       "1.0=公开配方原样; 课程用多次运行推进 (a 是 trace 期常量)")
  ap.add_argument("--desired_base_height", type=float, default=None,
                  help="[task=getup_v2] 站立时躯干离地目标高度 (默认 0.28)")
  ap.add_argument("--residual_scale", type=float, default=None,
                  help="[task=getup_res] 残差幅度: ctrl = ctrl_fsm + a*residual_scale "
                       "(默认 0.30)。有效探索噪声 = init_noise_std * residual_scale")
  ap.add_argument("--dry_run", action="store_true",
                  help="只建 env + 网络 + 打印配置, **不调用 ppo.train**")
  args = ap.parse_args()

  print("backend:", jax.default_backend())
  t0 = time.time()

  # ---------------- 环境 ----------------
  if args.task == "walk":
    from envs.go1_walk import Go1Walk, default_config as walk_cfg_fn
    locomotion.register_environment("Go1WalkDbg", Go1Walk, walk_cfg_fn)
    env_cfg = walk_cfg_fn()
    env_cfg.terrain = True
    env_cfg.height_scan.enable = True
    env_cfg.random_init.enable = True
    env_cfg.reward_config.scales.feet_phase = 2.0
    env_cfg.reward_config.max_foot_height = \
        env_cfg.reward_config.gait_swing_height
    env_cfg.command_config.frame = "body"
    env_cfg.healthy_roll_range = 17.5
    env_cfg.healthy_pitch_range = 17.5
    env_name = "Go1WalkDbg"
  elif args.task == "getup_res":
    from envs.go1_getup_residual import (  # noqa: E402
        Go1GetupResidual, default_config as res_cfg_fn)
    locomotion.register_environment("Go1GetupRes", Go1GetupResidual, res_cfg_fn)
    env_cfg = res_cfg_fn()
    env_name = "Go1GetupRes"
  elif args.task == "getup_v2":
    from envs.go1_getup_v2 import (  # noqa: E402
        Go1GetupV2, default_config as v2_cfg_fn)
    locomotion.register_environment("Go1GetupV2", Go1GetupV2, v2_cfg_fn)
    env_cfg = v2_cfg_fn()
    env_name = "Go1GetupV2"
  else:
    env_cfg = default_config()
    env_name = "Go1Getup"
  env_cfg.episode_length = args.episode_length
  env_cfg.graph_mode = args.graph_mode
  if args.task in ("getup", "getup_res"):
    if args.settle_time is not None:
      env_cfg.getup.settle_time = args.settle_time
    if args.drop_prob is not None:
      env_cfg.getup.drop_prob = args.drop_prob
    if args.task == "getup":
      for _n, _v in (("orientation", args.orientation_w),
                     ("torso_height", args.torso_height_w),
                     ("posture", args.posture_w),
                     ("stand_still", args.stand_still_w)):
        if _v is not None:
          env_cfg.reward_config.scales[_n] = _v
    env_cfg.getup.difficulty = args.difficulty     # §29.15 固定难度 (课程=多次运行)
    env_cfg.getup.orient_rand = args.orient_rand
    if args.task == "getup_res" and args.residual_scale is not None:
      env_cfg.getup.residual_scale = args.residual_scale
  if args.task == "getup_v2":
    if args.desired_base_height is not None:
      env_cfg.getup2.desired_base_height = args.desired_base_height
    if args.joint_alpha is not None:
      env_cfg.getup2.joint_alpha = args.joint_alpha
    if args.action_clip is not None:
      env_cfg.getup2.action_clip = args.action_clip
  if args.naconmax:
    env_cfg.naconmax = args.naconmax
  env = locomotion.load(env_name, config=env_cfg)

  n_state = env.observation_size["state"][0]
  n_priv = env.observation_size["privileged_state"][0]
  print(f"env: {type(env).__name__}  action_size={env.action_size}  "
        f"obs={n_state}/{n_priv}  dt={env.dt}s "
        f"({1/env.dt:.0f}Hz)  episode={args.episode_length} 步 "
        f"({args.episode_length*env.dt:.1f}s)")

  # ---- 起身任务的三条护栏 (漏一条都会训出一坨没用的东西, 见 §29.9) ----
  if args.task == "walk":
    gu = None
    print("  [task=walk] 诊断模式: 用同一个 trainer 跑走路环境")
  else:
    gu = env_cfg.get("getup", None)
  if args.task == "getup_v2":
    g2 = env_cfg.getup2
    print(f"  [guard] getup_v2: 动作 target = default_pose + {g2.action_scale}*"
          f"clip(a,±{g2.action_clip}) 低通 alpha={g2.action_filter_alpha}  "
          f"历史={g2.history} 帧  站立目标高度={g2.desired_base_height} m  "
          f"出生 z=terrain+{g2.spawn_height}  xy±{g2.spawn_xy_range}")
    print("  [guard] getup_v2: **不做姿态/高度终止** (只留 NaN/出界护栏), "
          "奖励 = 2 个宽高斯 + 1 个站姿项 + 5 个小正则")
    _sc = dict(env_cfg.reward_config.scales)
    print("  [guard] 奖励权重: " + "  ".join(
        f"{k}={v}" for k, v in _sc.items()))
  if gu is not None:
    print(f"  [guard] 终止: 只挡 NaN/出界 (走路那份 tilt/z 判据在 getup 任务里会秒死)")
  if gu is not None:
    z_ref_is_rel = bool(getattr(env, "_hfield_ok", False))
    print(f"  [guard] 高度基准: 相对地形={z_ref_is_rel} "
          f"(hfield={getattr(env, '_hfield_ok', None)})  "
          f"z_des={gu.z_des}  期望站立≈0.277")
    print(f"  [guard] 摔倒采样: drop_prob={gu.drop_prob} "
          f"drop_height={gu.drop_height} settle={gu.settle_time}s "
          f"({env._settle_substeps} substeps)  naconmax={env_cfg.naconmax}")
    _sc = dict(env_cfg.reward_config.scales)
    print("  [guard] 奖励权重: " + "  ".join(
        f"{k}={v}" for k, v in _sc.items()))
    if args.task == "getup":
      print(f"  [guard] 高度门控 = 纯斜坡 clip(h/{gu.z_des}, 0, 1)")
    else:
      print(f"  [guard] getup_res: 残差幅度={gu.residual_scale}  "
            f"成功奖金={gu.success_bonus}  判据 up>{gu.up_th} & "
            f"z_rel∈[{gu.z_lo},{gu.z_hi}] 连续 {gu.hold_steps} 步")
    print(f"  [guard] 课程: difficulty={gu.difficulty:.2f} (关节随机度) "
          f"orient_rand={gu.orient_rand:.2f} (朝向随机度)  "
          f"num_resets_per_eval={args.num_resets_per_eval}  "
          f"restore={args.restore}")
  if args.task != "getup_v2" and (n_state != 91 or n_priv != 162):
    print(f"  [警告] obs 不是 91/162 —— 检查 terrain/height_scan/feet_phase "
          f"是否都开着, 否则与走路策略不同形")

  if args.task == "getup_v2":
    print("  [guard] obs 同形自检: getup_v2 用 42x5=210 维历史 obs, "
          "**按方案 A 不与走路同形** (查看器用 --getup_obs history 单独维护缓冲)")
  else:
    ok_obs, msg = check_walk_obs_match(env, args.walk_pkl)
    print(f"  [guard] obs 同形自检: {msg}")
    if not ok_obs:
      print("  [警告] 调度器方案要求两个策略同形; 继续训练前请先对齐。")

  layers = tuple(int(x) for x in args.layers.split(","))
  batch_size = args.num_envs // args.num_minibatches
  assert args.num_envs % args.num_minibatches == 0, (
      f"num_envs={args.num_envs} 必须能被 num_minibatches={args.num_minibatches} 整除")
  rollout = args.num_envs * args.unroll_length
  print(f"  [ppo] rollout={args.num_envs}×{args.unroll_length}={rollout} 条/次  "
        f"minibatch={batch_size} 序列×{args.unroll_length}步="
        f"{batch_size*args.unroll_length} 条  "
        f"每批梯度更新={args.updates_per_batch*args.num_minibatches} 次  "
        f"网络={layers} tanh")

  # ---------------- PPO (配置与 v23 那一档一致, 见文件头) ----------------
  ppo_params = config_dict.create(
      num_timesteps=args.num_timesteps,
      num_evals=args.num_evals,
      reward_scaling=1.0,
      episode_length=args.episode_length,
      normalize_observations=False,
      action_repeat=1,
      unroll_length=args.unroll_length,
      num_minibatches=args.num_minibatches,
      num_updates_per_batch=args.updates_per_batch,
      discounting=args.discounting,
      learning_rate=args.lr,
      entropy_cost=args.entropy,
      num_envs=args.num_envs,
      batch_size=batch_size,
      max_grad_norm=args.max_grad_norm,
      clipping_epsilon=args.clipping_epsilon,
      network_factory=config_dict.create(
          policy_hidden_layer_sizes=layers,
          value_hidden_layer_sizes=layers,
          policy_obs_key="state",
          value_obs_key="privileged_state",
          distribution_type="normal",
          activation=jax.nn.tanh,
      ),
  )

  logdir = args.logdir if os.path.isabs(args.logdir) else os.path.join(
      _ROOT, args.logdir)
  ckpt_path = os.path.join(logdir, "checkpoints")
  os.makedirs(ckpt_path, exist_ok=True)

  ppo_params.network_factory.init_noise_std = args.init_noise_std
  training_params = dict(ppo_params)
  network_factory = functools.partial(
      ppo_networks.make_ppo_networks, **training_params.pop("network_factory"))
  train_fn = functools.partial(
      ppo.train,
      **training_params,
      network_factory=network_factory,
      seed=args.seed,
      # 每个 eval 存一次 -> best/ 与中途 pkl 都靠它 (sim/eval_getup.py 可以直接吃目录)
      save_checkpoint_path=ckpt_path,
      restore_checkpoint_path=args.restore,
      restore_params=(_load_init_pkl(args.init_pkl) if args.init_pkl else None),
      wrap_env_fn=wrapper.wrap_for_brax_training,
      num_eval_envs=args.num_eval_envs,
      # §29.15: >0 才会在每个 eval 段内多次 reset -> 出生姿态不再固定为 768 个
      num_resets_per_eval=args.num_resets_per_eval,
  )

  if args.dry_run:
    # 只把网络建出来 (形状自检), 不调用 train_fn
    net = network_factory(
        observation_size={"state": (n_state,),
                          "privileged_state": (n_priv,)},
        action_size=env.action_size)
    p_params = net.policy_network.init(jax.random.PRNGKey(0))
    print(f"  [dry_run] 策略网络 init 成功: "
          f"{jax.tree.map(lambda x: x.shape, p_params)}")
    print(f"  [dry_run] 未调用 ppo.train; 用时 {time.time()-t0:.1f}s")
    return 0

  # ---------------- 训练 ----------------
  best_ckpt_dir = os.path.join(ckpt_path, "best")

  best_reward, best_step = -float("inf"), 0
  times = [time.monotonic()]
  # 分段计时 (§29.12): 原来的累计 fps = num_steps/(now-start) 把"首次编译"整块摊到
  # 全程, 短跑时严重失真 —— 10 次迭代的累计 fps 基本等于 1/编译时间。这里额外打印
  # **本段**耗时和每迭代耗时, 把编译项隔离掉, 才能判断两个任务到底差多少。
  step_per_iter = args.num_envs * args.unroll_length
  seg = {"n": 0, "t": times[0]}
  def progress(num_steps, metrics):
    nonlocal best_reward, best_step
    now = time.monotonic()
    d_n = (num_steps - seg["n"]) / step_per_iter
    d_t = now - seg["t"]
    seg["n"], seg["t"] = num_steps, now
    seg_str = (f" seg={d_t:6.1f}s/{d_n:5.1f}it"
               f"={1000*d_t/max(d_n,1e-9):6.0f}ms/it"
               f" seg_fps={step_per_iter*d_n/max(d_t,1e-9):7.0f}")
    times.append(now)
    fps = num_steps / max(times[-1] - times[0], 1e-9)
    loss = metrics.get("losses/total_loss", float("nan"))
    ent = metrics.get("losses/entropy", float("nan"))
    if "eval/episode_reward" in metrics:
      r = float(metrics["eval/episode_reward"])
      if r > best_reward:
        best_reward, best_step = r, num_steps
        src = os.path.join(ckpt_path, f"{num_steps:012d}")
        if os.path.isdir(src):
          if os.path.isdir(best_ckpt_dir):
            shutil.rmtree(best_ckpt_dir)
          shutil.copytree(src, best_ckpt_dir)
          print(f"  [best] eval_reward={r:.3f} @ {num_steps} -> best/",
                flush=True)
      # 起身任务的真信号: getup/is_success 是"本步刚满足判据(连续25步)"的标志,
      # EvalWrapper 把它聚合成 episode_getup/is_success = **每个 episode 的成功次数**,
      # 也就是**成功率**。ep_len 下降 + 它上升 = 真的站起来了。
      _succ = metrics.get("eval/episode_getup/is_success",
                          metrics.get("eval/episode_getup/stand", float("nan")))
      _upz = metrics.get("eval/episode_getup/up_z", float("nan"))
      _zr = metrics.get("eval/episode_getup/z_rel", float("nan"))
      print(f"{num_steps}: eval_reward={r:.3f} "
            f"ep_len={metrics.get('eval/avg_episode_length', float('nan')):.1f} "
            f"succ={_succ:.3f} up_z={_upz:.3f} z_rel={_zr:.3f} "
            f"| loss={loss:.4f} ent={ent:.3f}{seg_str} "
            f"cum_fps={fps:.0f}", flush=True)
    else:
      print(f"{num_steps}: loss={loss:.4f} ent={ent:.3f}{seg_str} "
            f"cum_fps={fps:.0f}", flush=True)

  print(f"开始训练 (num_timesteps={args.num_timesteps}, "
        f"评估 {args.num_evals} 次)... 用时 {time.time()-t0:.1f}s", flush=True)
  make_inference_fn, params, _ = train_fn(environment=env,
                                          progress_fn=progress)
  print("训练完成", flush=True)

  save_path = os.path.join(POLICY_DIR, args.save_name)
  with open(save_path, "wb") as f:
    pickle.dump(params, f)
  print(f"已保存策略: {save_path}")
  print(f"best eval_reward={best_reward:.3f} @ {best_step} 步")
  print(f"下一步: python sim/eval_getup.py --pkl {save_path}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
