#!/usr/bin/env python
"""训练 Go1 行走策略 (复刻 quadruped-rl-locomotion 任务), brax PPO + GPU (warp)。

环境: envs/go1_walk.py (参考仓库奖励/终止/obs 1:1)。
PPO 配置: 对齐 mujoco_playground 官方 locomotion (100M 步, 8192 envs,
(512,256,128) 网络, value 用 privileged_state)。参考仓库用 SB3 默认超参
(12 envs CPU), 我们保持 GPU 并行配置 —— 任务和环境已 1:1, 训练框架差异
本身就是本实验要验证的变量之一。

用法 (WSL, 从 RLcontroller_go1 目录):
  python -u -m train.train_go1 --num_timesteps 100000000
"""
import argparse
import functools
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from ml_collections import config_dict

from mujoco_playground import locomotion, wrapper

from envs.go1_walk import Go1Walk, default_config

# 注册自定义环境
locomotion.register_environment("Go1Walk", Go1Walk, default_config)

POLICY_DIR = os.path.join(os.path.dirname(__file__), "..", "policies")
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")


def _check_ckpt_overwrite(ckpt_path, allow_overwrite):
  """拒绝往已有检查点的目录写 (brax 的 save 是 force 覆盖)。

  续训时步数从 0 重新计数, 新检查点会沿用 5/10/15/20M 这套名字; 若 logdir
  指向上一次训练, 就会静默覆盖掉它的证据。v22 之前没有任何闸门。
  """
  if not (os.path.isdir(ckpt_path) and os.listdir(ckpt_path)):
    return
  existing = sorted(os.listdir(ckpt_path))
  if not allow_overwrite:
    raise SystemExit(
        f"拒绝启动: {ckpt_path} 已有 {len(existing)} 个检查点 "
        f"({', '.join(existing[:6])}{'...' if len(existing) > 6 else ''})。\n"
        f"brax 的 checkpoint.save 会 force 覆盖同名步数目录 -> 续训会毁掉"
        f"已有证据。请换 --logdir (推荐), 或确认无价值后加 --allow_overwrite。\n"
        f"上一版训练的备份: models/go1_v21_logs_frozen_backup/")
  print(f"警告: --allow_overwrite 已开, 将覆盖 {ckpt_path} 下 {len(existing)} "
        f"个同名检查点目录: {', '.join(existing[:6])}")


def _check_restore(restore, step_offset, obs_state, obs_priv):
  """续训前自检: --restore 指向可读检查点, 且 obs 维度与本次一致。

  brax 对不存在的路径会抛 ValueError, 但"半个目录"或维度不匹配的失败信息
  不直观 —— 这里提前显式报出来。
  """
  if restore is None:
    return
  import json
  r = os.path.abspath(restore)
  if not os.path.isdir(r):
    raise SystemExit(f"--restore 路径不存在: {r}")
  cfg = os.path.join(r, "ppo_network_config.json")
  if not os.path.isfile(cfg):
    raise SystemExit(
        f"--restore 不像一个 brax 检查点 (缺 ppo_network_config.json): {r}")
  with open(cfg) as f:
    rc = json.load(f)
  obs = rc.get("observation_size", {})
  got_state = (obs.get("state") or {}).get("shape")
  got_priv = (obs.get("privileged_state") or {}).get("shape")
  nf = rc.get("network_factory_kwargs", {})
  print(f"续训自检: restore={r}")
  print(f"  检查点 obs: state={got_state} privileged_state={got_priv}  "
        f"layers={nf.get('policy_hidden_layer_sizes')}/"
        f"{nf.get('value_hidden_layer_sizes')}")
  print(f"  本次 obs:   state=[{obs_state}] privileged_state=[{obs_priv}]")
  mism = []
  if got_state is not None and list(got_state) != [obs_state]:
    mism.append(f"state {got_state} != [{obs_state}]")
  if got_priv is not None and list(got_priv) != [obs_priv]:
    mism.append(f"privileged_state {got_priv} != [{obs_priv}]")
  if mism:
    raise SystemExit(
        "--restore 的 obs 维度与本次训练不一致, 无法恢复:\n  "
        + "\n  ".join(mism)
        + "\n(维度不一致说明环境配置与产出该检查点的训练不同)")
  print("  维度一致 ✓")
  print("  注意: brax checkpoint **不含优化器状态与步数** -> Adam 动量从零"
        "重建, num_steps 从 0 记。这是续训必须降 lr 的原因 (§9.2/§22)")
  if step_offset:
    print(f"  step_offset={step_offset} (仅日志显示累计步数, 不改训练语义)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_timesteps", type=int, default=100_000_000)
    ap.add_argument("--num_envs", type=int, default=8192)
    ap.add_argument("--episode_length", type=int, default=750)
    ap.add_argument("--num_evals", type=int, default=20)
    ap.add_argument("--restore", type=str, default=None,
                    help="从 checkpoint 目录恢复继续训练")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--logdir", type=str, default=None)
    ap.add_argument("--allow_overwrite", action="store_true",
                    help="允许往非空 logdir 写检查点 (默认拒绝)。brax 的 "
                         "checkpoint.save 用 force=True 覆盖同名步数目录, 续训"
                         "时新检查点会沿用 5/10/15/20M 这套相对步数名 -> 若不换 "
                         "logdir 会**静默覆盖**上一版训练的证据 (v22 起加此闸门)")
    ap.add_argument("--step_offset", type=int, default=0,
                    help="续训起点步数 (仅**打印**用): 日志里的 num_steps 是本次"
                         "训练的步数, 加上它才是累计步数。不改训练语义")
    ap.add_argument("--save_name", type=str, default="go1_walk_policy.pkl")
    ap.add_argument("--sb3_opt", action="store_true",
                    help="v2: 对齐 SB3 优化制度 (minibatch 256→640 样本, "
                         "updates 4→8/批, entropy 0, discount 0.99)。"
                         "梯度密度 128/批 → 2048/批 (16×), 100M 步 ≈ "
                         "125 万次梯度更新 ≈ 参考仓库 5M 步 (SB3, 78 万次) 的 1.6×")
    ap.add_argument("--sb3_net", action="store_true",
                    help="v3: 网络缩到 (64,64) + value (64,64) (SB3 默认 "
                         "MlpPolicy 结构, ~1.5 万参数, 与参考仓库对齐)。"
                         "需配合 --sb3_opt (v2 教训: 优化制度是主变量)")
    ap.add_argument("--sb3_full", action="store_true",
                    help="v4: 完全仿照参考仓库 (SB3 默认) 的算法参数 —— "
                         "rollout 24576 (768envs×unroll32), minibatch 64, "
                         "10 epochs (每批 3840 次梯度更新, 密度 0.156/样本 "
                         "= SB3 精确值), entropy 0, discount 0.99, clip 0.2, "
                         "max_grad_norm 0.5, 无 obs 归一化, 无压缩高斯分布, "
                         "tanh 激活, 网络 (64,64)。总步数用 5M (参考默认)。"
                         "唯一残余差异: GAE 窗口 32 vs SB3 2048 (显存所限)")
    ap.add_argument("--hard_contact", action="store_true",
                    help="v5: 物理对齐参考仓库 — 足端硬接触 (XML 原样 "
                         "solimp 0.015/1/0.031 condim=6) + Newton 100 迭代。"
                         "实测 warp 硬接触与 CPU 参照逐位一致 (diag_hard_contact.py); "
                         "软接触静平衡残差 324 vs 0.08 且会弹飞 (v1-v4 站立根因)")
    ap.add_argument("--norm_obs", action="store_true",
                    help="打开 brax 观测运行归一化 (v6 加过; v10 证明对 NaN "
                         "既非必要也非充分 —— NaN 真因是 njmax 溢出, 见实验"
                         "记录 §4.3。保留开关仅供对照)")
    # --- v11: 指令条件化 (每 episode 从 [cmd_low, cmd_high] 采样指令) ---
    ap.add_argument("--sample_cmd", action="store_true",
                    help="v11: 打开指令采样 (实现一个策略覆盖多方向)。"
                         "默认关 = 恒定指令 [0.5,0,0], 与 v1-v10 一致")
    ap.add_argument("--cmd_low", type=str, default=None,
                    help="指令下界, 逗号分隔 3 值 (vx,vy,wz), 如 '0,0,-1'")
    ap.add_argument("--cmd_high", type=str, default=None,
                    help="指令上界, 逗号分隔 3 值 (vx,vy,wz), 如 '0.5,0,1'")
    ap.add_argument("--fwd_turn", action="store_true",
                    help="v11 预设: 只做前进 + 转向 (vy=0, vx∈[0,0.5], "
                         "wz∈[-1,1])。等价于 --sample_cmd "
                         "--cmd_low 0,0,-1 --cmd_high 0.5,0,1")
    # --- 微调稳定性旋钮 (v11 教训: 直接热启动会被梯度打崩, 见实验记录) ---
    ap.add_argument("--lr", type=float, default=None,
                    help="覆盖学习率 (默认 SB3 档 3e-4)。热启动微调时 Adam 的"
                         "归一化步长会把收敛好的策略推走, 需显著调低 (如 1e-4/3e-5)")
    ap.add_argument("--entropy", type=float, default=None,
                    help="覆盖 entropy_cost (默认 0)。收敛后的策略 log_std 往往"
                         "缩得很小 → 微调时没有探索, 加一点熵可恢复探索")
    ap.add_argument("--layers", type=str, default=None,
                    help="覆盖 policy/value 隐层, 逗号分隔, 如 '128,128'。"
                         "SB3 档 (64,64) 是单指令容量; 指令条件化(多方向)建议加大")
    ap.add_argument("--body_frame", action="store_true",
                    help="v14: 指令解释为机体系(朝向)速度 (obs 的 base linvel 也"
                         "换成机体系)。转向必须用: global 语义下'边走边转'要求"
                         "机器人保持全局速度同时自转 = 持续横行, 不可实现")
    ap.add_argument("--tilt_deg", type=float, default=None,
                    help="覆盖健康终止的 roll/pitch 阈值 (默认 10 → 等效 ±20°)。"
                         "参考仓库的 ±20° 对直行够用, 但转弯时躯干正常俯仰就会"
                         "越界 (v14 实测: 所有终止都卡在 20° 边界, up_z 仍 0.99)")
    # --- v16: 防打滑/迈步奖励 (playground go1/joystick 同款配方) ---
    ap.add_argument("--natural_gait", action="store_true",
                    help="v16: 打开'真迈步'奖励 —— 接触相足端滑移惩罚 "
                         "(feet_slip) + 摆动抬脚高度塑形 (feet_height) + "
                         "空距阈值 1.0→0.2 并封顶 0.5s。参考仓库配方下"
                         "'贴地滑行'能拿满速度跟踪奖励, 是 v10-v15 滑行步态的"
                         "直接原因 (实验记录 §10); 这三项把解推向抬腿迈步")
    ap.add_argument("--dangle_penalty", action="store_true",
                    help="v17: 在 --natural_gait 基础上加'悬空过久'逐步惩罚。"
                         "feet_slip/feet_height 只在触地帧结算, 永不落地的腿"
                         "躲得掉 → v16 出现 FL 吊挂 14.7s (duty 0.01)。"
                         "这一项按 clip(air_time-0.8s) 每步扣分, 封顶 2.0")
    ap.add_argument("--slip_penalty", type=float, default=None,
                    help="覆盖 feet_slip 权重 (默认 -0.5)")
    ap.add_argument("--foot_height", type=float, default=None,
                    help="覆盖 feet_height 权重 (默认 -1.0)")
    ap.add_argument("--air_thresh", type=float, default=None,
                    help="覆盖 air_time_threshold (默认 0.2)")
    ap.add_argument("--dangle_w", type=float, default=None,
                    help="覆盖 feet_dangle 权重 (v17 默认 -0.5)")
    # --- v19: 地形感知 + 鲁棒性 ---
    ap.add_argument("--terrain", action="store_true",
                    help="v19: 载入地形场景 (凹凸 + 2 条 5 级阶梯, 总高差 0.30m)。"
                         "同时自动打开随机出生 —— 不随机 xy 的话 8192 环境全"
                         "挤在原点压平区, 永远见不到台阶 (见 §17.2)")
    ap.add_argument("--height_scan", action="store_true",
                    help="v19: 35 点 (7x5) 高度扫描进 obs (actor+critic 都拿)。"
                         "需配合 --terrain (平地档特征恒为 0, 无信息)")
    ap.add_argument("--no_random_init", action="store_true",
                    help="v19: 关掉随机出生 (调试用: 全部从原点出生, 便于与"
                         "v18 基线对照)")
    ap.add_argument("--random_init_range", type=float, default=None,
                    help="覆盖出生点 xy 范围 (默认 5.0m; 地形半边长 6m)")
    ap.add_argument("--push", action="store_true",
                    help="v19: 打开周期性外力推挤 (模拟真实扰动, 提升鲁棒性)")
    ap.add_argument("--push_vel", type=float, default=None,
                    help="覆盖推挤强度 (默认 0.5 m/s)")
    ap.add_argument("--push_interval", type=int, default=None,
                    help="覆盖推挤间隔 (默认 250 步 = 5s)")
    ap.add_argument("--obs_noise", action="store_true",
                    help="v19: 打开观测传感器噪声 (小幅值, 提升鲁棒性/sim2real)")
    ap.add_argument("--scan_noise", type=float, default=None,
                    help="覆盖高度扫描噪声 (默认 0.02m, 仅 actor)")
    # --- v19: 显式爬升激励 + 下降引导 ---
    ap.add_argument("--climb_reward", action="store_true",
                    help="v19: 打开相对地形的躯干高度奖励 (legged_gym 的 "
                         "base_height)。地形上防止躯干下沉/拖地 —— 爬台阶的"
                         "必要条件 (躯干压低就上不去)。注意它本身不是高度奖励,"
                         "而是站高维护项")
    ap.add_argument("--base_height_w", type=float, default=None,
                    help="覆盖 base_height 权重 (默认 -200.0)。**平方**误差语义, "
                         "量级与线性直觉差很多: 偏差 2cm -> -0.08, 5cm -> -0.5, "
                         "10cm -> -2.0 (实测 z_rel 中位数 0.276 vs 目标 0.28, "
                         "正常站立时该项 ≈ -0.0005, 几乎免费)")
    ap.add_argument("--soft_landing", action="store_true",
                    help="v19: 打开落地冲击惩罚 (下降引导)。按落地帧足端向下"
                         "速度超过阈值的部分平方惩罚, 抑制下台阶/下坡时砸地")
    ap.add_argument("--impact_w", type=float, default=None,
                    help="覆盖 feet_impact 权重 (默认 -2.0)。实测该信号很稀疏"
                         "(95%% 的时刻为 0, 只 0.39%% 的(足,步)对触发) —— "
                         "阈值 0.5m/s 不会误伤正常步态; 单次硬落地约 -0.7~-5")
    ap.add_argument("--impact_vel", type=float, default=None,
                    help="覆盖落地冲击速度阈值 (默认 0.5 m/s, 低于它不罚)")
    # --- v20: 对角步态引导 + 落地惩罚改按时间结算 ---
    ap.add_argument("--gait_phase", action="store_true",
                    help="v20: 打开对角步态相位奖励 (playground feet_phase 同款)。"
                         "v19 退化成'前/后成对'步态 (前腿占空比 13%%、"
                         "对角同步 13%%), 纯惩罚推不出步态, 需要显式相位引导")
    ap.add_argument("--gait_w", type=float, default=None,
                    help="覆盖 feet_phase 权重 (默认 2.0, playground spot 同值)")
    ap.add_argument("--gait_freq", type=float, default=None,
                    help="覆盖步频 Hz (默认 1.8; v19 实测主周期 0.52-0.56s≈1.9Hz)")
    ap.add_argument("--gait_freq_scaled", action="store_true",
                    help="v20: 步频随指令速度缩放 (站着不动不踏步)。默认关 = "
                         "恒定步频 (playground 做法)")
    ap.add_argument("--keep_foot_height", action="store_true",
                    help="v20: 不把 max_foot_height 联动到 gait_swing_height。"
                         "默认联动 (两者应一致, 否则 feet_height 与 feet_phase "
                         "对'抬多高'的要求打架)")
    ap.add_argument("--body_contact_w", type=float, default=None,
                    help="v21: 非足端部件 (小腿/大腿/躯干) 撞地惩罚权重。"
                         "不加 = 0 (关闭)。修 §20 '后腿关节着地': 原 collision "
                         "项在 warp 后端是空实现 (无 data.contact), 膝盖撑地免费。"
                         "建议 -150 (实测标定: 坏步态平均贡献 -0.39, 与速度跟踪 "
                         "+1.0 同量级; 新增零梯度步最小)")
    ap.add_argument("--body_contact", action="store_true",
                    help="v21: 用推荐权重 -150 打开 body_contact 惩罚")
    ap.add_argument("--body_contact_max", type=float, default=None,
                    help="覆盖累计穿透深度上限 (m, 默认 0.02)。防止深穿透时"
                         "惩罚压过正奖励 -> 总奖励 clip 到 0 -> PPO 无梯度")
    # --- v22: 显式爬升激励 (修 §24 的激励缺口) ---
    ap.add_argument("--ascent", action="store_true",
                    help="v22: 打开「相对出生点的地形升高」奖励。原奖励面里没有"
                         "任何一项随「站得更高」增加 (base_height 是相对量, 速度"
                         "跟踪/相位/成本都与地形高度无关) -> 绕开楼梯与爬上去收益"
                         "完全相同。实测把 v21 策略放到楼梯脚下, 它走 2m 却横向"
                         "漂移 1.4m 绕开了楼梯 (最大爬升 4.7cm) = 激励问题")
    ap.add_argument("--ascent_w", type=float, default=None,
                    help="覆盖 ascent 权重 (默认 2.0)。**上界必须守住**: 这是"
                         "状态型奖励 (每步按当前高度给分), 在台顶站着不动可持续"
                         "收钱。实测套利算术: 走路每步 +1.776, 站着不动 +0.347, "
                         "站台顶 = 0.347 + W*0.30 -> W=4 给 +1.547 (88%% 走路, 险), "
                         "W>=8 给 +2.75 **超过走路 -> 策略会主动退化成站桩**。"
                         "(注意: argparse 的 help 里 %% 必须转义成 %%)")
    ap.add_argument("--ascent_cap", type=float, default=None,
                    help="覆盖升高上限 (m, 默认 0.35)")
    ap.add_argument("--ascent_gate", type=float, default=None,
                    help="v23: ascent 的行进门控速度 (m/s, 默认 0.2)。ascent 乘上 "
                         "clip(|v_body|/gate,0,1), 静止时归零 -> 不再有'站台顶不动"
                         "也收钱'的套利, 于是 W 可以提上去。设 0 关闭 (=v22 行为)")
    # --- v23: 左右镜像对称正则 ---
    ap.add_argument("--symmetry", action="store_true",
                    help="v23: 打开左右镜像对称正则 (修 §27 的永久不对称姿态: "
                         "后腿 hip 共模 -22 度, 左后膝多弯 0.5rad -> 小腿蹭地 66%%)。"
                         "形式 -W*EMA(镜像偏差); 用时间平均以避免对抗 trot 的左右反相")
    ap.add_argument("--symmetry_w", type=float, default=None,
                    help="覆盖对称权重 (默认 -0.4, 负值=惩罚)。按实测标定: "
                         "v22 的 EMA≈1.19 -> 0.48/步; v18(对称) EMA≈0.045 -> 0.018/步")
    ap.add_argument("--symmetry_beta", type=float, default=None,
                    help="覆盖 EMA 步长 (默认 0.01; 1/beta=100 步 ≈ 3.6 个步态周期)")
    ap.add_argument("--symmetry_cap", type=float, default=None,
                    help="覆盖对称项上限 (默认 2.0, 防负项压过正奖励把梯度打 0)")
    ap.add_argument("--trunk_height_ref", action="store_true",
                    help="v22 实验性对照 (**非默认**): base_height 参考点改成只用"
                         "躯干正下方 (默认是躯干+4足 5 点均值, 与 v21 一致)。"
                         "实测 mean5 的额外惩罚是随足端落点的振荡噪声 "
                         "(corr=+0.13, 均值 -0.034/步), 不是定向对抗爬升的力量 -> "
                         "本轮默认不改, 保持单变量")
    args = ap.parse_args()

    print("backend:", jax.default_backend())

    # v2 (--sb3_opt): 对齐参考仓库 (SB3 默认) 的优化制度。
    # v1 诊断 (实验记录 §4): 梯度密度 v1=128 次/批 vs SB3=3840 次/rollout,
    # 折算每样本更新次数差 200×。v2 只改优化制度四个参数, 网络结构/环境不动
    # (一次一个变量)。
    sb3_opt = args.sb3_opt
    sb3_net = getattr(args, "sb3_net", False)
    sb3_full = getattr(args, "sb3_full", False)

    # --- v2/v3 分层开关 (保留对照): sb3_opt = 优化制度, sb3_net = 网络 ---
    if sb3_full:
        # v4: 完全仿照参考仓库 (SB3 默认) 的算法参数。
        # SB3 默认: n_envs=12, n_steps=2048 → rollout 24576; batch=64;
        # n_epochs=10 → 每 rollout 3840 次梯度更新 (密度 0.156/样本);
        # gamma=0.99, clip_range=0.2, ent_coef=0, max_grad_norm=0.5,
        # lr=3e-4, MlpPolicy(64,64) tanh, 无 obs 归一化, 高斯无 squash。
        # brax 映射: num_envs=768 × unroll 32 = 24576 (batch_size 按 brax
        # 语义 = 每 minibatch 的 env 数 64; num_minibatches = 24576/64=384);
        # num_updates_per_batch=10 (epochs)。
        # 唯一残余差异: GAE 窗口 32 vs SB3 2048 (8GB 显存装不下 768×2048)。
        #
        # v9 关键修正 (brax 账目): brax 的 batch_size 是"每个 minibatch 的
        # 序列数", 每条序列 unroll_length 步 → minibatch 转移数 =
        # batch_size × unroll_length。v8 用 batch_size=64 → minibatch 2048
        # 条 (SB3 的 32 倍) → 每样本更新密度只有 SB3 的 1/32, 学得慢。
        # 正确映射: batch_size=2 (2 序列 × 32 步 = 64 条 = SB3 的 minibatch
        # 64), num_minibatches=384, updates=10 → 每训练步样本数
        # 24576、梯度更新 3840 次、密度 6.4 样本/更新 —— 与 SB3 完全一致,
        # 且保留 GAE 窗口 32. 总预算 203 训练步 = 5M 环境步 (= 参考仓库)。
        num_envs = 768
        unroll_length = 32
        batch_size = 2
        minibatches = 384
        updates_per_batch = 10
        entropy_cost = 0.0
        discounting = 0.99
        max_grad_norm = 0.5
        clipping_epsilon = 0.2
        normalize_observations = False
        policy_layers = (64, 64)
        value_layers = (64, 64)
        distribution_type = "normal"
        # brax network_factory 的 activation 是函数不是字符串 (SB3 "tanh")
        activation = __import__("jax").nn.tanh
        print(f"v9 sb3_full: num_envs={num_envs}, unroll={unroll_length}, "
              f"minibatch={batch_size}序列×{unroll_length}步="
              f"{batch_size*unroll_length}条 (=SB3 的 64), "
              f"updates/步={updates_per_batch*minibatches} 次, "
              f"密度 {unroll_length*batch_size/updates_per_batch:.1f} 样本/更新 "
              f"(=SB3 的 6.4)")
    else:
        if sb3_opt:
            minibatches = 256
            updates_per_batch = 8
            entropy_cost = 0.0
            discounting = 0.99
            print("v2 sb3_opt: minibatch=640样本, updates/批=2048, "
                  "entropy=0, discount=0.99")
        else:
            minibatches = 32
            updates_per_batch = 4
            entropy_cost = 1e-2
            discounting = 0.97

        # v3 (--sb3_net): 网络 (64,64) 对齐 SB3 默认 MlpPolicy (~1.5 万参数)。
        # 注意: brax 的 tanh_normal policy 头是 mean/log_std 双头, 输出 24 维
        # (vs SB3 同款), 参数量 ~2 万, 仍与参考仓库同量级。
        if sb3_net:
            policy_layers = (64, 64)
            value_layers = (64, 64)
        else:
            policy_layers = (512, 256, 128)
            value_layers = (512, 256, 128)
        num_envs = args.num_envs
        unroll_length = 20
        # brax 要求 batch_size × num_minibatches 能被 num_envs 整除
        # (train.py:339), 且 minibatch 转移数 = batch_size × unroll_length。
        # 取 batch_size = num_envs // num_minibatches, 使 rollout 恰好等于
        # num_envs × unroll_length, 与官方 locomotion 档语义一致
        # (v1: num_envs=8192, minibatches=32 → batch_size=256, 密度 1280 样本/更新)。
        # 注意默认 num_envs=8192 配 batch_size=64 会让 64×32=2048 不被 8192 整除,
        # brax 直接 assert 失败 —— 这是 v9 加 batch_size 变量时引入的回归。
        assert num_envs % minibatches == 0, (
            f"num_envs={num_envs} 必须能被 num_minibatches={minibatches} 整除")
        batch_size = num_envs // minibatches
        max_grad_norm = 1.0
        clipping_epsilon = 0.3
        normalize_observations = True
        distribution_type = "tanh_normal"
        # brax 的 activation 是函数不是字符串, 传 "silu" 会 TypeError
        activation = jax.nn.silu

    if args.layers:
        policy_layers = tuple(int(x) for x in args.layers.split(","))
        value_layers = policy_layers

    env_cfg = default_config()
    env_cfg.episode_length = args.episode_length
    # v5: 物理对齐参考仓库 (硬接触足端 + Newton 100 迭代)
    if getattr(args, "hard_contact", False):
        env_cfg.foot_soft_contact = False
        print(f"v5 hard_contact: 足端硬接触 (XML 原样) + Newton "
              f"{env_cfg.hard_contact_iters} 迭代 / ls "
              f"{env_cfg.hard_contact_ls_iters} (参考仓库物理)")
    # v6: 硬接触 + obs 归一化 (SB3 VecNormalize 缺失件)
    if getattr(args, "norm_obs", False):
        normalize_observations = True
        print("v6 norm_obs: 打开观测运行归一化 (SB3 语义)")

    # v11: 指令条件化 —— 每 episode 采样指令, 一个策略覆盖多方向
    if args.fwd_turn:
        args.sample_cmd = True
        args.cmd_low = args.cmd_low or "0,0,-1"
        args.cmd_high = args.cmd_high or "0.5,0,1"
    if args.sample_cmd:
        env_cfg.command_config.sample = True
        if args.cmd_low:
            env_cfg.command_config.low = [
                float(x) for x in args.cmd_low.split(",")]
        if args.cmd_high:
            env_cfg.command_config.high = [
                float(x) for x in args.cmd_high.split(",")]
        print(f"v11 指令条件化: 每 episode 采样 cmd ~ U("
              f"{list(env_cfg.command_config.low)}, "
              f"{list(env_cfg.command_config.high)})  "
              f"(fixed={list(env_cfg.command_config.fixed)} 仅用于评估/可视化)")
    if args.body_frame:
        env_cfg.command_config.frame = "body"
        print("v14 body_frame: 指令=机体系(朝向)速度, obs 的 base linevel 同步换"
              "机体系 (转向必需)")
    if args.tilt_deg is not None:
        env_cfg.healthy_roll_range = args.tilt_deg
        env_cfg.healthy_pitch_range = args.tilt_deg
        print(f"v15 tilt_deg={args.tilt_deg}: 终止阈值放宽到等效 ±"
              f"{2*args.tilt_deg:.0f}° (参考档 10 → ±20°)")
    if args.dangle_penalty:
        args.natural_gait = True
    # v16: 防打滑/迈步奖励 (playground go1/joystick 配方)
    if args.natural_gait:
        env_cfg.reward_config.scales.feet_slip = -0.5
        env_cfg.reward_config.scales.feet_height = -1.0
        env_cfg.reward_config.air_time_threshold = 0.2
        env_cfg.reward_config.air_time_max = 0.5
        print(f"v16 natural_gait: feet_slip={env_cfg.reward_config.scales.feet_slip}"
              f" (接触相足端速度² 惩罚), "
              f"feet_height={env_cfg.reward_config.scales.feet_height}"
              f" (摆动峰值抬到 {env_cfg.reward_config.max_foot_height}m), "
              f"air_time_threshold={env_cfg.reward_config.air_time_threshold}"
              f" 封顶 {env_cfg.reward_config.air_time_max}s")
    if args.dangle_penalty:
        env_cfg.reward_config.scales.feet_dangle = -0.5
        print(f"v17 dangle_penalty: feet_dangle="
              f"{env_cfg.reward_config.scales.feet_dangle} "
              f"(悬空超过 {env_cfg.reward_config.air_time_limit}s 逐步扣分, "
              f"封顶 2.0)")
    if args.slip_penalty is not None:
        env_cfg.reward_config.scales.feet_slip = args.slip_penalty
        print(f"覆盖 feet_slip={args.slip_penalty}")
    if args.foot_height is not None:
        env_cfg.reward_config.scales.feet_height = args.foot_height
        print(f"覆盖 feet_height={args.foot_height}")
    if args.air_thresh is not None:
        env_cfg.reward_config.air_time_threshold = args.air_thresh
        print(f"覆盖 air_time_threshold={args.air_thresh}")
    if args.dangle_w is not None:
        env_cfg.reward_config.scales.feet_dangle = args.dangle_w
        print(f"覆盖 feet_dangle={args.dangle_w}")

    # --- v19: 地形感知 + 鲁棒性 ---
    if args.height_scan and not args.terrain:
        # 平地档没有 hfield, 采样器不初始化; 高度特征会是常数 0 -> 白占 35 维。
        # 直接报错而不是静默降级, 避免训出一个"以为有地形输入"的假对照。
        raise SystemExit("--height_scan 需要 --terrain (平地上高度特征恒为 0)")
    if args.terrain:
        env_cfg.terrain = True
        # 地形训练必须随机出生: reset 默认把机器人放原点, 而地形在
        # flatten_origin 里把原点周围 0.5m 压平了 -> 8192 环境全在平地,
        # 见不到台阶, 学不会爬梯 (§17.2)。故 --terrain 隐式打开随机出生。
        if not args.no_random_init:
            env_cfg.random_init.enable = True
        if args.random_init_range is not None:
            env_cfg.random_init.xy_range = args.random_init_range
        print(f"v19 terrain: 地形场景 + 随机出生 "
              f"(xy±{env_cfg.random_init.xy_range}m, yaw 随机, "
              f"z=地面+{env_cfg.random_init.spawn_z_offset}m, "
              f"出生速度±{env_cfg.random_init.random_vel})")
    if args.no_random_init:
        env_cfg.random_init.enable = False
        print("v19 no_random_init: 全部从原点出生 (调试/基线对照)")
    if args.height_scan:
        env_cfg.height_scan.enable = True
        hs = env_cfg.height_scan
        print(f"v19 height_scan: {hs.nx}x{hs.ny}={hs.nx*hs.ny} 点, "
              f"x∈[{hs.x_min},{hs.x_max}] y∈[{hs.y_min},{hs.y_max}], "
              f"base={hs.base} scale={hs.scale} clip_m={hs.clip_m} "
              f"noise={hs.noise}m (仅 actor)  -> obs 48+{hs.nx*hs.ny}="
              f"{48+hs.nx*hs.ny} (actor) / {119+hs.nx*hs.ny} (critic)")
    if args.scan_noise is not None:
        env_cfg.height_scan.noise = args.scan_noise
        print(f"覆盖 height_scan.noise={args.scan_noise}")
    if args.push:
        env_cfg.push_config.enable = True
        if args.push_vel is not None:
            env_cfg.push_config.max_vel = args.push_vel
        if args.push_interval is not None:
            env_cfg.push_config.interval = args.push_interval
        print(f"v19 push: 每 {env_cfg.push_config.interval} 步推一次, "
              f"水平速度冲量 ±{env_cfg.push_config.max_vel} m/s")
    if args.obs_noise:
        env_cfg.obs_noise.enable = True
        on = env_cfg.obs_noise
        print(f"v19 obs_noise: lin_vel±{on.lin_vel} ang_vel±{on.ang_vel} "
              f"gravity±{on.gravity} dof_pos±{on.dof_pos} "
              f"dof_vel±{on.dof_vel} (仅 actor; critic 无噪)")

    # --- v19: 显式爬升激励 + 下降引导 ---
    # 权重来自实测标定 (sim/probe_v19_reward_scale.py(已删除, 结论见实验记录)), 不是猜的:
    #   base_height 是**平方**误差 -> 线性直觉会差一个数量级。
    #   实测正常站立时平方误差 p50=1.9e-5 (≈免费), 摔倒/拖地时才大。
    #   -200 下: 偏差 2cm=-0.08, 5cm=-0.5, 10cm=-2.0; 每步均值 -0.88
    #   (-2000 会给到每步 -8.8, 是 v18 每步奖励 2.46 的 3.6 倍 -> 会把总奖励
    #    打到 0, PPO 无梯度, 正是 v1-v7 的失败模式, 所以不用)
    if args.climb_reward:
        env_cfg.reward_config.scales.base_height = -200.0
    if args.base_height_w is not None:
        env_cfg.reward_config.scales.base_height = args.base_height_w
        args.climb_reward = True
    if env_cfg.reward_config.scales.get("base_height", 0.0) != 0.0:
        print(f"v19 climb_reward: base_height="
              f"{env_cfg.reward_config.scales.base_height} "
              f"(相对足下地形, 目标站高 "
              f"{env_cfg.reward_config.base_height_target}m; **平方**误差: "
              f"偏差 2cm=-0.08, 5cm=-0.5, 10cm=-2.0; "
              f"正常站立实测每步≈-0.0005)")
    if args.soft_landing:
        env_cfg.reward_config.scales.feet_impact = -2.0
    if args.impact_w is not None:
        env_cfg.reward_config.scales.feet_impact = args.impact_w
    if args.impact_vel is not None:
        env_cfg.reward_config.impact_vel_limit = args.impact_vel
    if env_cfg.reward_config.scales.get("feet_impact", 0.0) != 0.0:
        _iw = env_cfg.reward_config.scales.feet_impact
        _lim = env_cfg.reward_config.impact_vel_limit
        print(f"v19 soft_landing: feet_impact={_iw}, 阈值 {_lim} m/s "
              f"(落地 1.0m/s={_iw*(1.0-_lim)**2:.2f}, "
              f"1.5m/s={_iw*(1.5-_lim)**2:.2f}, "
              f"2.0m/s={_iw*(2.0-_lim)**2:.2f} /足; "
              f"实测正常步态 95% 的时刻不触发)")
        print(f"v20 注: feet_impact 已改为**按时间结算** (净空 < "
              f"{env_cfg.reward_config.impact_height}m 时每步考核向下速度), "
              f"不再是'只在落地帧' —— 修 §18.10.3 的激励漏洞")
        print("v20 注: feet_height 已改为**摆动相均值**结算 (同源漏洞)")

    # --- v20: 对角步态相位引导 ---
    if args.gait_phase:
        env_cfg.reward_config.scales.feet_phase = 2.0
    if args.gait_w is not None:
        env_cfg.reward_config.scales.feet_phase = args.gait_w
    if args.gait_freq is not None:
        env_cfg.reward_config.gait_freq = args.gait_freq
    if args.gait_freq_scaled:
        env_cfg.reward_config.gait_freq_scaled_by_cmd = True
    if env_cfg.reward_config.scales.get("feet_phase", 0.0) != 0.0:
        rc = env_cfg.reward_config
        # **必须联动**: feet_phase 要求抬到 gait_swing_height, feet_height 罚
        # 偏离 max_foot_height。两者不一致时策略被两个目标撕扯, 谁都不满足。
        if not args.keep_foot_height:
            rc.max_foot_height = rc.gait_swing_height
        print(f"v20 gait_phase: feet_phase={rc.scales.feet_phase} "
              f"(对角 trot 相位 [0,pi,pi,0], 步频 {rc.gait_freq}Hz"
              f"{', 随指令缩放' if rc.gait_freq_scaled_by_cmd else ''}, "
              f"swing_height={rc.gait_swing_height}, sigma={rc.gait_phase_sigma})")
        print(f"     奖励 = exp(-Σ(足端净空 - rz(相位))²/{rc.gait_phase_sigma})")
        print(f"     联动: max_foot_height={rc.max_foot_height} "
              f"(feet_height 的抬脚目标, 与 swing_height 一致"
              f"{'' if not args.keep_foot_height else ' —— 被 --keep_foot_height 关掉'})")
        print(f"     区分度自检: 理想trot=1.0000, 全贴地≈0.34, 全悬≈0.19 "
              f"(gap≈0.66; 若 gap<0.15 奖励面太平, 推不动步态)")

    # --- v22: 显式爬升激励 (修 §24 的激励缺口) ---
    if args.ascent:
        env_cfg.reward_config.scales.ascent = 2.0
    if args.ascent_w is not None:
        env_cfg.reward_config.scales.ascent = args.ascent_w
    if args.ascent_cap is not None:
        env_cfg.reward_config.ascent_cap = args.ascent_cap
    if args.ascent_gate is not None:
        env_cfg.reward_config.ascent_gate = args.ascent_gate
    if env_cfg.reward_config.scales.get("ascent", 0.0) != 0.0:
        rc = env_cfg.reward_config
        if not args.terrain:
            raise SystemExit(
                "--ascent 需要 --terrain (平地上地形高度恒为 0, 该项无信息)")
        print(f"v22 ascent: ascent={rc.scales.ascent} "
              f"(每步 += W * clip(h_now - h_spawn, 0, {rc.ascent_cap}))")
        print(f"     为什么需要它: 原奖励面**没有任何一项**随「站得更高」增加 —— "
              f"base_height 是相对量 (躯干离足下地面多高), 地形上爬升不改变它; "
              f"速度跟踪/相位/各项 cost 都与地形高度无关。")
        print(f"     -> 绕开楼梯与爬上去收益**完全相同**, 而绕行更容易 -> 策略学会绕行。")
        print(f"     实测依据: 把 v21 策略放在楼梯脚下正对楼梯, 走 2.0m 却横向漂移"
              f" 1.4m 绕开, 最大爬升 4.7cm (sim/diag_v22_climb_test.py)")
        print(f"     标定 (sim/probe_v22_reward_terms.py 实测每步均值): "
              f"linear_vel_tracking=+1.76, feet_phase=+1.59, base_height=-0.30")
        _gate = env_cfg.reward_config.ascent_gate
        if _gate > 0.0:
            print(f"     v23 行进门控: ascent *= clip(|v_body|/{_gate},0,1) -> "
                  f"静止/站台顶不动 = 0 (站桩套利消失), 故 W 可提到 "
                  f"{env_cfg.reward_config.scales.ascent}; "
                  f"无门控时 W>=4 就有站桩风险 "
                  f"(走路 +1.776 vs 站台顶 0.347+W*0.30)")
        else:
            print(f"     反套利上界 (无门控 = v22 行为): 走路每步 +1.776, "
                  f"站着不动 +0.347, 站台顶=0.347+W*0.30 -> W=4 给 +1.547 "
                  f"(88%, 险); W>=8 超过走路 -> 会退化成站桩")
        print(f"     信号稀疏性: 升高 >2cm 只占 1.8% 的步 (>10cm 占 0.0%) -> "
              f"每步均值小, 但梯度方向明确")
    # base_height 参考点: 默认 mean5 = 与 v21 完全一致 -> v22 是单变量实验。
    # trunk 仅供单独对照 (实测它**不是**爬升失败的原因, 见 _cost_base_height)。
    if args.trunk_height_ref:
        env_cfg.reward_config.base_height_ref = "trunk"
        print("v22 base_height_ref=trunk (**实验性对照, 非默认**): 参考面 = 躯干"
              "正下方一点。实测量化: mean5 的额外惩罚是随足端落点的振荡噪声 "
              "(corr=+0.13, 均值 -0.034/步, 峰值 -0.083 = 走路 4.7%), 不是定向"
              "对抗爬升的力量 —— 故本轮默认不改它, 只加 ascent (单变量)")
    else:
        print("v22 base_height_ref=mean5 (默认): 参考面 = 躯干+4足 5 点均值, "
              "**与 v21 一致** -> 本轮唯一变量是 ascent")

    # --- v21: 非足端部件撞地惩罚 (修 §20 "后腿关节着地") ---
    # --- v23: 左右镜像对称正则 ---
    if args.symmetry:
        env_cfg.reward_config.scales.symmetry = -0.4
    if args.symmetry_w is not None:
        env_cfg.reward_config.scales.symmetry = args.symmetry_w
    if args.symmetry_beta is not None:
        env_cfg.reward_config.symmetry_beta = args.symmetry_beta
    if args.symmetry_cap is not None:
        env_cfg.reward_config.symmetry_cap = args.symmetry_cap
    if env_cfg.reward_config.scales.get("symmetry", 0.0) != 0.0:
        print(f"v23 symmetry: symmetry={env_cfg.reward_config.scales.symmetry} "
              f"beta={env_cfg.reward_config.symmetry_beta} "
              f"cap={env_cfg.reward_config.symmetry_cap} "
              f"(每步 -= W * EMA(镜像偏差), EMA 时间常数 "
              f"{1.0/env_cfg.reward_config.symmetry_beta:.0f} 步)")
    if args.body_contact:
        env_cfg.reward_config.scales.body_contact = -150.0
    if args.body_contact_w is not None:
        env_cfg.reward_config.scales.body_contact = args.body_contact_w
    if args.body_contact_max is not None:
        env_cfg.reward_config.body_contact_max = args.body_contact_max
    if env_cfg.reward_config.scales.get("body_contact", 0.0) != 0.0:
        _bw = env_cfg.reward_config.scales.body_contact
        _bm = env_cfg.reward_config.body_contact_margin
        _bx = env_cfg.reward_config.body_contact_max
        print(f"v21 body_contact: body_contact={_bw} "
              f"(非足端碰撞体 = 小腿+大腿+躯干, 共 28 个几何)")
        print(f"     形式 = -{abs(_bw)} * min(Σ relu(margin−净空), {_bx}m), "
              f"margin={_bm}m")
        print(f"     margin=0 的含义 (**实测定标**): 平地沉降后的标称站姿, "
              f"28 个几何净空全为正 (最小 +2.2mm)")
        print(f"       -> 正常姿态惩罚严格=0, 不会推歪姿势; 只有真穿透才扣分")
        print(f"     标定依据 (sim/calib_penalty_form.py(已删除, 结论见实验记录), MuJoCo 接触判定):")
        print(f"       大腿区触地率  v18(好)= 0.0%  v20(坏)= 6.6%  <- 干净判据")
        print(f"       后腿穿透深度  v18=0.00055m  v20=0.00305m  (5.6x)")
        print(f"       选线性不选平方: 区分度 4.07x vs 2.22x, 且浅穿透有梯度")
        print(f"     上限 {_bx}m: 深穿透封顶, 避免惩罚压过正奖励把总奖励打 0")
        print(f"       (总奖励 clip(·,0,·) -> 梯度归零, 是 v1-v7 的失败模式)")

    # 地形/高度扫描会改 obs 维度 -> 网络输入层形状必须随之改变。brax 从
    # env.observation_size 自动推断 (mjx_env.py 用 jax.eval_shape(reset)),
    # 这里只做一次显式核对, 避免"以为改了其实没生效"。
    _probe = env_cfg.height_scan.enable
    _n_scan = env_cfg.height_scan.nx * env_cfg.height_scan.ny if _probe else 0
    # v20: 相位 (cos,sin) x 4 足 = 8 维, 仅在开相位奖励时进 obs
    _n_phase = 8 if env_cfg.reward_config.scales.get("feet_phase", 0.0) != 0.0 else 0
    print(f"预期 obs 维度: state={48 + _n_scan + _n_phase}  "
          f"privileged_state={119 + _n_scan + _n_phase}"
          f"{'  (含相位 8 维)' if _n_phase else ''}")

    _lr = args.lr if args.lr is not None else 3e-4
    _ent = args.entropy if args.entropy is not None else entropy_cost
    print(f"优化: lr={_lr} entropy_cost={_ent} clip={clipping_epsilon} "
          f"grad_clip={max_grad_norm} layers={policy_layers} "
          f"restore={args.restore}")

    logdir = args.logdir or os.path.join(LOG_DIR, "walk")
    # orbax 保存要求绝对路径 (相对路径在首次 checkpoint 时抛
    # "Checkpoint path should be absolute")
    logdir = os.path.abspath(logdir)
    ckpt_path = os.path.join(logdir, "checkpoints")

    # --- 启动前安全检查 (放在 env/locomotion.load 之前: 拒绝时不该已占 GPU) ---
    _check_ckpt_overwrite(ckpt_path, args.allow_overwrite)
    _check_restore(args.restore, args.step_offset,
                   48 + _n_scan + _n_phase, 119 + _n_scan + _n_phase)

    os.makedirs(logdir, exist_ok=True)
    os.makedirs(ckpt_path, exist_ok=True)

    env = locomotion.load("Go1Walk", config=env_cfg)
    print(f"env: {type(env).__name__} action_size={env.action_size}")

    ppo_params = config_dict.create(
        num_timesteps=args.num_timesteps,
        num_evals=args.num_evals,
        reward_scaling=1.0,
        episode_length=args.episode_length,
        normalize_observations=normalize_observations,
        action_repeat=1,
        unroll_length=unroll_length,
        num_minibatches=minibatches,
        num_updates_per_batch=updates_per_batch,
        discounting=discounting,
        learning_rate=args.lr if args.lr is not None else 3e-4,
        entropy_cost=(args.entropy if args.entropy is not None
                      else entropy_cost),
        num_envs=num_envs,
        batch_size=batch_size,
        max_grad_norm=max_grad_norm,
        clipping_epsilon=clipping_epsilon,
        network_factory=config_dict.create(
            policy_hidden_layer_sizes=policy_layers,
            value_hidden_layer_sizes=value_layers,
            policy_obs_key="state",
            value_obs_key="privileged_state",
            distribution_type=distribution_type,
            activation=activation,
        ),
    )

    training_params = dict(ppo_params)
    if "network_factory" in training_params:
        del training_params["network_factory"]
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks, **ppo_params.network_factory
    )

    train_fn = functools.partial(
        ppo.train,
        **training_params,
        network_factory=network_factory,
        seed=args.seed,
        save_checkpoint_path=ckpt_path,
        restore_checkpoint_path=args.restore,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        num_eval_envs=128,
    )

    times = [time.monotonic()]

    # eval_reward 最高的 checkpoint → checkpoints/best/
    best_reward = -float("inf")
    best_step = 0
    best_ckpt_dir = os.path.join(ckpt_path, "best")

    def _m(metrics, *names):
        for n in names:
            if n in metrics:
                return float(metrics[n])
        return float("nan")

    # 续训时 num_steps 从 0 重新计数, 日志里同时打印累计步数 (step_offset + 
    # num_steps) 和相对步数, 避免把 25M 读成 5M (§22 的坑)。
    _off = args.step_offset

    def progress(num_steps, metrics):
        nonlocal best_reward, best_step
        times.append(time.monotonic())
        fps = num_steps / (times[-1] - times[0])
        loss = _m(metrics, "losses/total_loss", "total_loss")
        ent = _m(metrics, "losses/entropy", "entropy")
        pl = _m(metrics, "losses/policy_loss", "policy_loss")
        vl = _m(metrics, "losses/value_loss", "value_loss")
        tag = f"{num_steps}(累计{_off + num_steps})" if _off else f"{num_steps}"
        if "eval/episode_reward" in metrics:
            r = float(metrics["eval/episode_reward"])
            if r > best_reward:
                best_reward = r
                best_step = num_steps
                src = os.path.join(ckpt_path, f"{num_steps:012d}")
                if os.path.isdir(src):
                    if os.path.isdir(best_ckpt_dir):
                        shutil.rmtree(best_ckpt_dir)
                    shutil.copytree(src, best_ckpt_dir)
                    print(f"  [best] eval_reward={r:.3f} @ {tag} → best/",
                          flush=True)
            print(f"{tag}: eval_reward={r:.3f} "
                  f"ep_len={metrics.get('eval/avg_episode_length', float('nan')):.1f} "
                  f"| loss={loss:.4f} pi={pl:.4f} vf={vl:.4f} ent={ent:.2f} "
                  f"fps={fps:.0f}", flush=True)
        else:
            print(f"{num_steps}: loss={loss:.4f} pi={pl:.4f} vf={vl:.4f} "
                  f"ent={ent:.2f} fps={fps:.0f}", flush=True)

    print("开始训练...")
    make_inference_fn, params, _ = train_fn(
        environment=env, progress_fn=progress
    )
    print("训练完成")

    import pickle
    save_path = os.path.join(POLICY_DIR, args.save_name)
    with open(save_path, "wb") as f:
        pickle.dump(params, f)
    print(f"已保存策略: {save_path}")
    print(f"best eval_reward={best_reward:.3f} @ {best_step}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())