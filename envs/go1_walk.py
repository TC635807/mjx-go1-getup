"""Go1 行走环境: 逐条复刻 quadruped-rl-locomotion/go1_mujoco_env.py 的任务,
跑在 MJX warp 后端 (GPU) 上, brax PPO 训练。

复刻对照 (SB3 gymnasium 语义 1:1 转 JAX):
  - 奖励 (线性配方, 每步总值 clip ≥0, 参考仓库 max(0, rewards - costs)):
      +2.0*exp(-||v_cmd - v||²/0.25)      linear_vel_tracking
      +1.0*exp(-(w_cmd - wz)²/0.25)       angular_vel_tracking
      +1.0*feet_air_time (接触时刻 Σ(t-1.0)*first_contact, 指令>0.1 才生效)
      -0.0002*Στ²                          torque
      -0.01*Σ(a_t - a_{t-1})²              action_rate
      -2.0*vz²                             vertical_vel
      -0.05*(ωx²+ωy²)                      xy_angular_vel
      -10.0*soft_joint_range 越界量         joint_limit
      -0.01*Σω_joint²                      joint_velocity
      -2.5e-7*Σqacc²                       joint_acceleration
      -1.0*Σprojected_gravity[:2]²         orientation
      -1.0*非足端碰撞数                     collision
      -0.1*Σ(qpos-default)²                default_joint_position
  - 终止 (is_healthy): 有限值 + z∈(0.22,0.65) + |roll|<10° + |pitch|<10°。
    这是参考仓库"效果好"的关键差异之一 (我们 ANYmal 用 25.8°)。
  - 观测 48 维 (与参考仓库 1:1 同序同缩放):
      v*2.0 (3) + ω*0.25 (3) + projected_gravity (3) + v_cmd*2.0 (3)
      + (qpos-qpos0)*1.0 (12) + qvel_j*0.05 (12) + last_action (12), clip ±100。
  - 指令: 恒定 [0.5, 0, 0] (参考仓库 min=max=[0.5,0,0] —— 任务窄是成功主因)。
  - reset: 关节位置 U(-0.1,0.1) 噪声 + ctrl N(0,0.1) 打破对称 (参考同款)。

与参考仓库的必要差异 (GPU 训练工程层面, 语义不变):
  - 动作 = 12 个绝对位置目标 (ctrlrange 内)。SB3 tanh_normal 输出 [-1,1],
    环境内线性映射到 ctrlrange (等效于他们 position 控制 + 无界 action)。
  - episode 15s (750 步 ctrl) 同参考; 每步奖励不乘 dt (SB3 每步累加语义,
    brax 默认乘 dt; 保持与参考仓库 episode 总量可比)。
  - 足端软接触 (solimp 0.9/0.95, condim=3): 参考仓库硬接触
    (solimp 0.015/1/0.031, condim=6) 在 MJX 迭代式求解器下不收敛
    (我们 ANYmal v23b/25/26/27b 同款教训); foot_soft_contact=False 可切回。
  - obs 的 projected_gravity 用机体系重力向量 (gravity sensor), 参考仓库
    的欧拉角点乘版本在小角度下等价且无奇异性。

用法: train/train_go1.py
"""
import os
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np

from mujoco_playground._src import mjx_env

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.join(_THIS_DIR, "..")
XML_PATH = os.path.join(_ROOT_DIR, "models", "go1", "scene_mjx_position.xml")
# 地形场景 (小幅凹凸 + 小台阶)。训练与 view 都通过 config.terrain 选它,
# 两条路径共用同一 XML -> 严格一致 (见 sim/gen_terrain.py 的确定性生成)。
TERRAIN_XML_PATH = os.path.join(
    _ROOT_DIR, "models", "go1", "scene_mjx_terrain.xml")

FEET = ["FR", "FL", "RR", "RL"]  # 参考仓库足序


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.02,          # 50Hz 控制 = 参考仓库 frame_skip=10 × 0.002
      sim_dt=0.002,          # 参考仓库 XML timestep=0.002
      episode_length=750,    # 15s (参考 _max_episode_time_sec=15.0)
      # 奖励**下限**. SB3 语义原本是 max(0, r-c) -> clip(total, 0, ...)。
      # 对起身任务这是有害的: 早期随机策略的"两个宽高斯 - 小正则"总和接近 0,
      # 一旦为负就被夹成 0 -> **奖励恒 0, PPO 完全没有梯度** (v2 实测随机动作
      # 平均 +0.008/步)。get-up-isaaclab 用 only_positive_rewards=False (允许负奖励)。
      # 默认 0.0 保持走路/旧起身任务逐位不变; 只有 getup_v2 把它设成 -1e6。
      reward_clip_min=0.0,
      action_scale=1.0,      # 动作=绝对位置目标
      # --- 物理开关 (MJX 训练必需) ---
      solver_iterations=2,   # warp 后端软接触收敛实测值 (ANYmal v28 诊断)
      solver_ls_iterations=5,
      foot_soft_contact=True,   # 足端软接触 (MJX 迭代求解器下硬接触不收敛)
      # v5: 物理完全对齐参考仓库 (CPU Newton 语义)。foot_soft_contact 必须为
      # False 搭配使用。warp 后端硬接触是否收敛/NaN 由 smoke 验证决定。
      # diag_hard_contact.py 实测: iterations=20 与 100 的沉降 z/残差/6s 随机
      # 动作结果逐位一致 (0.2796/0.083/ok) → 用 20 省 2-3 倍时间。
      hard_contact_iters=20,
      hard_contact_ls_iters=10,
      # naconmax 是**全体 world 共享的接触槽总数**, 不是 per-env —— mujoco_warp
      # types.py: "naconmax: maximum number of contacts (shared across all worlds)",
      # io.py 断言 naconmax >= mjd.ncon * nworld。8*8192 = 65536 在 num_envs=8192 时
      # 只有 8 个接触/world (playground go1 用 4*8192)。
      # 实测 (sim/diag_speed.py, 768 env): 走路峰值 2369 / 起身躺地峰值 3700 (全局),
      # 即 4.8/world -> 65536 对 768 env 有 ~85/world 余量; 调到 32768/49152 无速度差异
      # (§29.12)。调小只省内存/带宽, 不掉时间。
      naconmax=8 * 8192,
      # v10: njmax 64→256。硬接触 + 摔倒在地时自碰撞约束数会超过 64,
      # mujoco_warp 报 "nefc overflow" 后写入越界 → qacc/sensordata 变 NaN
      # → 奖励 NaN → PPO 参数被打爆 (v5/v7/v9 的 NaN 崩溃根因之一)。
      njmax=256,
      noise_config=config_dict.create(
          level=1.0,
          scales=config_dict.create(
              gyro=0.0,      # 参考仓库无传感器噪声 (仅 reset 噪声)
              gravity=0.0,
              joint_pos=0.0,
              joint_vel=0.0,
              linvel=0.0,
          ),
      ),
      reward_config=config_dict.create(
          scales=config_dict.create(
              # 正奖励 (参考 reward_weights)
              linear_vel_tracking=2.0,
              angular_vel_tracking=1.0,
              feet_airtime=1.0,
              # 负成本 (参考 cost_weights)
              torque=-0.0002,
              vertical_vel=-2.0,
              xy_angular_vel=-0.05,
              action_rate=-0.01,
              joint_limit=-10.0,
              # v10: 参考仓库 _calc_reward 里 joint_velocity_cost 和
              # collision_cost 都算出来了, 但 costs 求和时漏加 (见
              # go1_mujoco_env.py:362-371)。为与参考"实际生效"的奖励面
              # 完全一致, 这里把权重置 0 (项仍计算, 仅作诊断)。
              joint_velocity=0.0,
              joint_acceleration=-2.5e-7,
              orientation=-1.0,
              collision=0.0,
              default_joint_position=-0.1,
              # v16 新增 (默认 0 关闭, 由 train_go1.py --natural_gait 打开):
              # 接触相足端滑移惩罚 (playground go1/joystick.py:512 同款结构)。
              # 参考仓库没有这一项 —— 没有它时"贴地滑行"能拿满速度跟踪奖励,
              # 是 v10-v15 滑行步态的直接原因 (实验记录 §10)。
              feet_slip=0.0,
              # v16 新增 (默认 0 关闭): 摆动相抬脚高度塑形 (playground
              # joystick.py:530 _cost_feet_height)。正面激励抬腿。
              feet_height=0.0,
              # v17 新增 (默认 0 关闭): 悬空过久惩罚。feet_slip 和 feet_height
              # 都只在触地帧生效 → "永远不落地"的腿同时躲开两项惩罚, 是 v16
              # 出现单腿吊挂 (FL duty 0.01, 吊 14.7s) 的直接原因。这一项按
              # 每步惩罚 clip(air_time - air_time_limit), 吊着就持续扣分。
              feet_dangle=0.0,
              # 参考仓库没有的 (默认 0 关闭, 保留代码路径对照)
              termination=0.0,
              pose=0.0,
              stand_still=0.0,
              # v19 新增 (默认 0 关闭, 由 --climb_reward / --soft_landing 打开):
              # 相对地形的躯干高度奖励 (显式爬升激励) + 落地冲击惩罚 (下降引导)。
              # 见 §17.16。
              base_height=0.0,
              feet_impact=0.0,
              # v20 新增 (默认 0 关闭, 由 --gait_phase 打开): 对角步态相位引导。
              # 参考 playground 的 feet_phase (spot 用 2.0)。
              feet_phase=0.0,
              # v21 新增 (默认 0 关闭, 由 --body_contact_w 打开): 非足端部件
              # (小腿/大腿/躯干) 撞地惩罚。修 §20 "后腿关节着地"的漏洞 ——
              # 原 collision 项是空实现且权重 0, 膝盖撑地完全免费。
              body_contact=0.0,
              # v22 新增 (默认 0 关闭, 由 --ascent 打开): 相对出生点的地形
              # 升高奖励 (显式爬升激励)。修 §24 的激励缺口 —— 原奖励面**没有任何
              # 一项随"站得更高"增加** (base_height 是相对量, 速度跟踪/相位/成本
              # 都与地形高度无关), 于是绕开楼梯与爬上去收益完全相同。
              # 权重 2.0 由反套利算术定 (±见 _reward_ascent 的注释: W>=8 会让
              # "站台顶不动"超过"走路", 策略会退化)。
              ascent=0.0,
              # v23 新增 (默认 0 关闭, 由 --symmetry 打开): 左右镜像对称正则。
              # 动机 (§27 实测): v20 起策略收敛到**永久不对称姿态** —— 后腿 hip
              # 共模 -22° (RR 外扩 -0.53 / RL 内收 -0.23), 左后膝 -2.01 vs 其余
              # -1.46~-1.53 -> RL 小腿蹭地 66% (其余 5.7/6.3/7.7%); v18 (平地那代)
              # 后腿共模只有 +1.7°。**平地上同样存在** -> 不是地形/出生点问题,
              # 也**不能靠继续加步数修好** (已是收敛吸引子)。
              # 形式: -W * EMA(镜像偏差)。**必须用时间平均**: trot 的左右腿本来
              # 反相 (相位差 π), 瞬时镜像差恒非零, 用瞬时值会去对抗步态本身。
              symmetry=0.0,
          ),
          tracking_velocity_sigma=0.25,  # 参考 _tracking_velocity_sigma
          # air_time_threshold: 参考仓库是 1.0 (对正常摆动 0.2-0.4s 是倒扣,
          # 见 §10.3)。v16 的 --natural_gait 会把它覆盖成 0.2 并启用封顶;
          # 默认保持 1.0 以便 v10-v15 的回归对照逐位复现。
          air_time_threshold=1.0,
          # 空距奖励封顶 (v16): 不封顶时"吊腿 10s 再落地"能拿 +9.8, 与正常
          # 步态的每秒收益相当 → 三足解。默认不封顶, 保持旧行为。
          air_time_max=float("inf"),
          # v16 摆动相目标抬脚高度 (playground max_foot_height=0.1)。
          # 球足半径 0.023, 静息足心 z≈0.023; 抬到 0.08 即净空 ~0.06。
          max_foot_height=0.08,
          # v17: 单足连续悬空上限 (s)。超过即按超出量持续惩罚。
          # 正常摆动 0.2-0.4s; 给 2 倍余量 → 0.8s。playground 没有这一项,
          # 它的 feet_air_time 不封顶本身就压着"别吊太久", 但那只在落地时
          # 结算 —— 不落地的腿躲得掉, 所以这里用逐步惩罚。
          air_time_limit=0.8,
          cmd_threshold=0.1,             # 指令<0.1 无 airtime 奖励 (参考同款)
          # ---- v19: 相对地形躯干高度 (显式爬升激励) ----
          # legged_gym 原式: (mean(root_z - measured_heights) - target)²
          # 注意它本身就减了 measured_heights, **已经是相对地形量** —— 平地上
          # 等价于绝对高度, 地形上才是对的。这正是 §17 指出的激励缺口:
          # 原奖励里没有任何一项关心"站在地形多高", 爬楼梯只能靠"为了维持
          # 速度指令"间接驱动, 而绕开楼梯同样能满足指令。
          #
          # 目标值: Go1 沉降后躯干 z≈0.2779 (relative to 足下地面)。
          # 取 0.28。**必须与 height_scan.base 一致**, 否则两处对"标称站高"
          # 的定义打架 (obs 说 0.28, 奖励要 0.3)。
          base_height_target=0.28,
          # v22: ascent 项的上限 (m)。只按 clip(升高, 0, cap) 结算, 防止某个
          # 环境爬上天花板后靠一项吃满奖励 (地形最高 0.30m, 取 0.35 留余量)。
          ascent_cap=0.35,
          # v23: ascent 的**行进门控**。ascent 是状态型奖励, 站在台顶不动也
          # 持续收钱 —— §24.6 的反套利算术因此把 W 压到 2 (W=4 就占走路的 88%)。
          # 乘一个 clip(|v_body|/gate, 0, 1) 之后, 静止时该项归零, 于是可以在
          # 不诱发"站桩退化"的前提下把 W 提上去。gate=0 关闭门控 (复现 v22)。
          ascent_gate=0.2,
          # v23: 对称偏差 EMA 的步长系数。1/beta ≈ 时间常数 (步);
          # 0.01 -> 100 步 = 2s ≈ 3.6 个步态周期, 足以滤掉左右相位差。
          symmetry_beta=0.01,
          # v23: 对称偏差的上限。总奖励在 step() 里 clip(·,0,·), 负项压过正项会
          # 让梯度归零 (§20.6 的教训) -> 给这一项封顶, 与 body_contact 同处理。
          symmetry_cap=2.0,
          # v22: base_height 的地面参考点。
          #   "mean5" = 躯干 + 4 足的 5 点均值 (**默认, 保持 v21 行为**)
          #   "trunk" = 只用躯干正下方 (可选, 见 _cost_base_height 的实测说明)
          # 默认保持 mean5 是为了让 v22 成为**单变量实验** (只加 ascent):
          # 实测量化后, mean5 的额外惩罚**不是**定向对抗爬升的力量 (斜率
          # corr=+0.13), 而是随足端落点的振荡噪声 (均值 -0.034/步, 峰值
          # -0.083 = 走路的 4.7%)。既然它不是爬升失败的原因, 就不该在本轮
          # 一起改 —— 否则 40M 的改善无法归因 (§20.7 的"一次只改一个变量")。
          base_height_ref="mean5",
          # 参考仓库 max_contact_force=100 (但它标注 "TODO: Not used");
          # 这里用于 feet_impact 的冲击阈值 (N·s 等效, 见 _cost_feet_impact)。
          max_contact_force=100.0,
          # v19: 落地冲击的"无惩罚"下界 (m/s)。足端向下速度低于它不罚。
          # 正常步态落地 |vz| ≈ 0.2-0.5 m/s; 从台阶跳下可达 1.5+。
          impact_vel_limit=0.5,
          # v20: feet_impact 的"接近地面"过渡高度 (m)。净空 >= 它时权重为 0
          # (高处的下落速度不算砸地, 是正常摆动); 净空 0 (触地) 时权重为 1。
          impact_height=0.09,
          # ---- v20: 对角步态引导 (相位奖励) ----
          # 参考实现: mujoco_playground/_src/gait.py 的 get_rz + GAIT_PHASES,
          # 以及 spot/joystick_gait_tracking.py 的 _reward_feet_phase。
          #
          # 动机 (§18.10): v19 20M 退化成了"前/后成对"步态 (前腿占空比 13%,
          # 后腿 78%, 对角同步仅 13%), 前腿基本吊着不落地。要把它推回对角 trot,
          # 需要**显式的相位引导** —— 仅靠"惩罚"是推不出步态的。
          #
          # 机制: 每足有目标相位 phi_i, trot 为 [0, π, π, 0] (FR,FL,RR,RL)。
          # 相位循环推进, 由 get_rz 给出该相位下应有的目标足高:
          #   x=(phi+π)/2π <0.5 -> 支撑相 (从 0 升到 swing_height),
          #   x>0.5            -> 摆动相 (从 swing_height 落回 0)
          # 奖励 = exp(-Σ(z_foot - rz)²/σ), σ=0.1 (playground 同款)。
          gait_freq=1.8,          # Hz, 步频。v19 实测主周期 0.52-0.56s ≈ 1.9Hz
          # v20 修正 (实测标定, 见 §19.8): 原用 playground 的 sigma=0.1 +
          # swing_height=0.08 -> **奖励面几乎是平的**:
          #   "理想 trot" 1.0000 vs "四足全贴地不动" 0.9583, 只差 0.04。
          # 后果 (实测): v20 @5M 反而退化成"四足贴地不动"(前腿占空比 8.7%,
          # 后腿 96-99%, 对角同步 10-20%, vx 0.052)。
          # 机制: sigma=0.1 对 4 足求和 -> 容差极大; 且 rz 均值 0.04 与静息
          # 净空 0.023 很接近 -> "脚一直贴地"的误差天然就小, 反而被**偏好**。
          #
          # 修正: 提高抬脚目标 + 收紧 sigma, 把区分度从 0.04 拉到 0.66。
          # 扫描结果 (sim/probe_v20_phase_flatness.py(已删除, 结论见实验记录)):
          #   sh=0.15, sigma=0.02 -> A(理想)=1.0000 B(贴地)=0.3371 C(全悬)=0.1880
          #   gap = A - max(B,C) = 0.6629  (原配置只有 0.0417)
          gait_phase_sigma=0.02,  # was 0.1 (playground 值, 对我们的量纲太松)
          gait_swing_height=0.15, # was 0.08; 同时是"抬脚目标高度"
          # 注意: swing_height 必须与 max_foot_height 一致, 否则 feet_height
          # (罚偏离 0.08) 与 feet_phase (要抬到 0.15) 打架 —— 已在 train_go1.py
          # 里联动设置。
          # 相位推进的速率随指令速度缩放: 站着不动时不该继续"踏步"。
          # False = 恒定步频 (playground 做法); True = 步频 ∝ |v_cmd|。
          gait_freq_scaled_by_cmd=False,
          gait_freq_min=0.6,      # 缩放时的下限 (Hz), 避免接近 0 导致相位停滞
          # v21: 非足端部件撞地惩罚的"免罚裕度"(m)。净空 >= margin 不罚。
          # **实测定标为 0** (见 sim/calib_penalty_form.py(已删除, 结论见实验记录)): 平地上把 home
          # keyframe 物理沉降 1s 后, 全部 28 个非足端碰撞几何的净空**全为正**
          # (最小 +2.2mm) —— 所以 margin=0 时该惩罚对标称静息姿态严格为 0,
          # 不会把正常姿态推歪。而一旦真的穿透 (关节着地), 立刻按深度扣分。
          #
          # 为什么不给正裕度: 实测标称站姿的小腿净空只有 +2.2~+5mm (模型里
          # 小腿碰撞胶囊本来就贴边: calf2 伸到 z=-0.20, 足端球最低 -0.236,
          # 只差 2.6cm), margin 超过 ~3mm 就会把正常站姿也罚上。
          body_contact_margin=0.0,
          # 累计穿透深度的上限 (m)。防止深穿透时惩罚无限增长把总奖励压到 0
          # (本项目 total reward clip(·,0,·) -> 梯度归零, v1-v7 的失败模式)。
          # 取 0.02m: 实测坏步态累计穿透 p95 = 0.0154m, 所以只在极端情况
          # (摔倒) 才封顶; 配合权重 -150 时最坏贡献被限在 -3.0 (与正奖励
          # 预算 ~+2.3~3.0 同量级, 不会彻底吃掉信号)。
          body_contact_max=0.02,
      ),
      # 指令 (参考 _desired_velocity_min/max + 每 episode _sample_desired_vel)
      # fixed: 评估/可视化用的固定指令 (v1-v10 一直只用这一档)
      # sample=True 时 reset 里从 [low, high] 均匀采样, 实现指令条件化策略
      #
      # frame 决定"指令的速度是哪个坐标系":
      #   "global": 参考仓库原语义 —— 直接和 qvel[:2] (全局速度) 比。
      #             参考仓库指令只有 [0.5,0,0] 且机器人 yaw≈0, 全局≈机体系,
      #             所以这个写法在他们的任务里完全没问题。
      #   "body":   指令解释为机体系(朝向)速度, 和 Rᵀ·qvel[:3] 比。
      #             **边走边转必须用这个**: 用 global 的话, 机器人一边自转
      #             一边还得维持全局 +x 速度 → 得持续"漂移/横行", 不可实现,
      #             策略只能选择"不转" (v12/v13 实测: 纯转向能转, 转+走全摔)。
      command_config=config_dict.create(
          fixed=[0.5, 0.0, 0.0],
          sample=False,
          low=[0.0, 0.0, -1.0],
          high=[0.5, 0.0, 1.0],
          frame="global",
      ),
      # 健康终止阈值 (参考 is_healthy)
      healthy_z_range=(0.22, 0.65),
      healthy_roll_range=10.0,   # deg
      healthy_pitch_range=10.0,  # deg
      # reset 噪声 (参考 _reset_noise_scale=0.1)
      reset_noise_scale=0.1,
      # obs 缩放 (参考 _obs_scale)
      obs_scale=config_dict.create(
          linear_velocity=2.0,
          angular_velocity=0.25,
          dofs_position=1.0,
          dofs_velocity=0.05,
      ),
      obs_clip=100.0,
      impl="warp",
      # warp 后端的 CUDA graph 捕获模式 (仅 impl="warp" 时有效)。
      # 默认 None = 用 MJX 默认 (WARP): "capture the graph and replay it for
      # matching buffer addresses"。**这个默认对训练是对的, 对交互式查看器是错的**:
      # 查看器每帧新建 command 数组、并按 R 重置 state, buffer 地址反复变动 ->
      # WARP 模式每次都要重新 capture -> 跑几千帧后崩在
      #   wp.capture_begin(...) -> RuntimeError: Warp error: unknown stream
      # (用户实测 frame 2125 崩溃)。
      #
      # 实测对比 (sim/probe_graph_mode.py(已删除, 结论见实验记录), 模拟查看器循环: 每帧新建 command +
      # 每 300 帧 reset, 各跑 3000 帧):
      #   NONE           139.0 ms/帧  不崩 (但慢)
      #   JAX            预热即崩 (CUDA_ERROR_NOT_SUPPORTED: 无法建子图节点)
      #   WARP (默认)    185.5 ms/帧  不崩*  <- 真实查看器下会崩, 见上
      #   WARP_STAGED     16.3 ms/帧  不崩  <- 用 staging buffer + 图内 memcpy,
      #                                        专为"地址会变"的场景设计
      #   WARP_STAGED_EX  57.3 ms/帧  不崩
      # 所以**交互式查看器应设 graph_mode="WARP_STAGED"** (快 11 倍且稳定)。
      # 训练是固定 buffer 地址 (jax.jit(donate_argnums=0), 地址稳定), 保持默认
      # WARP —— 那里图缓存能命中, 是 25.9ms -> 14.8ms 的大头 (见文件头)。
      graph_mode=None,
      # 地形: False = 平地 (原 plane 地板, v1-v18 全部结果); True = 小幅凹凸
      # + 小台阶 (models/go1/scene_mjx_terrain.xml, 由 sim/gen_terrain.py 生成)。
      # 注意: 换地形后策略必须重训 (v18 是在平地上训的), 不能从 v18 热启动 ——
      # 地形改变了足端可达性与接触面, 属换任务 (见实验记录 §9.2 热启动教训)。
      terrain=False,
      # ---- v19: 地形感知 (height scan) ----
      # 35 点 (7x5) 高度网格进 obs, actor + critic 都拿。这是**单阶段非对称
      # actor-critic**(一个策略 + 一个看了更多信息的价值函数), 不是教师/学生
      # 两阶段蒸馏 —— 见 §17.1。必须 terrain=True 才有意义: 平地上采样恒为 0。
      height_scan=config_dict.create(
          enable=False,
          nx=7,                  # 前向分辨率 (前进方向给更多, 与 legged_gym x>y 一致)
          ny=5,                  # 侧向分辨率
          x_min=-0.3,            # 机体系前向范围 (m)
          x_max=0.9,             # 前视偏置: 台阶需预判, 向后看不提供额外信息
          y_min=-0.4,
          y_max=0.4,
          # 基准高度 (m): 特征 = clip(躯干z - 地面h - base, ±clip_m) * scale。
          # legged_gym 写 root_z-0.5-h (0.5 是 ANYmal 标称站高); Go1 实测
          # 沉降后躯干 z≈0.2779 → 取 0.28。**必须按机器人标定**, 否则特征
          # 整体偏置, 地形维在 obs 里"不可见" (§16.6.3)。
          base=0.28,
          # clip_m: **米制**差值的截断 (legged_gym 的 clip(root_z-0.5-h,-1,1))。
          # 取 0.35 > 地形总高差 0.30 -> 正常站立/行走时**不饱和**, 梯度全程保留;
          # 只在摔倒等异常姿态下截断, 保数值稳定。
          # (曾经的错误写法: 对已乘 scale 的特征 clip ±0.25 -> 高差 >6.25cm
          #  就饱和, 30cm 楼梯全程顶格, 梯度消失。)
          clip_m=0.35,
          scale=5.0,             # ×5: 米制差变特征, 值域 ±1.75, 与关节角/重力可比
          # 噪声 (仅 actor, critic 无噪): 训练时扰动地面高度, 避免过拟合精确
          # 地形, 也是 sim2real 关键 (legged_gym 0.1 / IsaacLab ray_cast_drift)。
          noise=0.02,            # m, 均匀分布半宽
      ),
      # ---- v19: 鲁棒性 (小幅度噪声 + 扰动) ----
      # 随机出生: 地形训练的前提 —— 不随机 xy 的话 8192 环境全挤在原点那块
      # 半径 0.5m 的压平区, 永远见不到台阶/楼梯, 学不会爬梯 (§17.2)。
      random_init=config_dict.create(
          enable=False,          # 开 terrain 时由 train_go1.py 自动打开
          xy_range=5.0,          # 出生点 xy ~ U(-5, 5), 避开 ±6m 边界
          random_yaw=True,       # 随机朝向 (与机体系指令自洽)
          spawn_z_offset=0.35,   # 躯干出生高度 = 地面h + 0.35 (home keyframe 值)
          random_vel=0.3,        # 出生速度 ~ U(-0.3, 0.3) m/s / rad/s
      ),
      # 外力推挤 (模拟真实扰动): 每 interval 步给基座叠加随机水平速度
      push_config=config_dict.create(
          enable=False,
          interval=250,          # 步 (50Hz 下 = 5s)
          max_vel=0.5,           # 冲量等效速度 (m/s) 加到基座水平速度
      ),
      # 观测噪声 (传感器噪声): 原 noise_config 是占位(全 0)/从未接线, 这里按
      # §17.5 的量级打开小幅值 —— 参考仓库本来就是 0, 属 v19 新增鲁棒性。
      obs_noise=config_dict.create(
          enable=False,
          lin_vel=0.05,          # 乘 obs_scale 前 (m/s)
          ang_vel=0.05,          # rad/s
          gravity=0.02,          # 单位向量分量
          dof_pos=0.01,          # rad
          dof_vel=0.15,          # rad/s
      ),
  )


class Go1Walk(mjx_env.MjxEnv):
  """Go1 前进 0.5 m/s (复刻 quadruped-rl-locomotion 训练任务)。"""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(config, config_overrides)

    # 地形选择 (训练与 view 共用): config.terrain 决定加载哪个场景 XML
    xml_path = TERRAIN_XML_PATH if self._config.terrain else XML_PATH

    assets = {}
    xml_dir = os.path.dirname(xml_path)
    mjx_env.update_assets(assets, xml_dir, "*.xml")
    mjx_env.update_assets(assets, os.path.join(xml_dir, "assets"))
    self._model_assets = assets
    self._mj_model = mujoco.MjModel.from_xml_path(xml_path)
    self._mj_model.opt.timestep = self._config.sim_dt
    self._mj_model.opt.ccd_iterations = 20

    # 求解器配置 (warp 后端软接触收敛)
    self._mj_model.opt.iterations = self._config.solver_iterations
    self._mj_model.opt.ls_iterations = self._config.solver_ls_iterations

    # v5: 物理对齐参考仓库 → 保留 XML 硬接触足端原样 (solimp 0.015/1/0.031,
    # condim=6) + Newton 完整迭代 (iterations=100, ls=50, 与 CPU MuJoCo 默认
    # 一致)。软接触模式 (默认) 保持 MJX 快速物理。
    if not self._config.foot_soft_contact:
      self._mj_model.opt.iterations = self._config.hard_contact_iters
      self._mj_model.opt.ls_iterations = self._config.hard_contact_ls_iters
    if self._config.foot_soft_contact:
      for name in FEET:
        gid = self._mj_model.geom(name).id
        self._mj_model.geom_solimp[gid] = np.array([0.9, 0.95, 0.023, 0.5, 2.0])
        self._mj_model.geom_condim[gid] = 3

    # graph_mode: 交互式查看器要传 "WARP_STAGED" (见 config 注释); None = MJX 默认
    _gm = None
    if self._config.impl == "warp" and self._config.graph_mode is not None:
      from mujoco.mjx.warp import types as _mjxw_types
      _gm = getattr(_mjxw_types.GraphMode, str(self._config.graph_mode))
    self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl,
                                    graph_mode=_gm)
    self._xml_path = xml_path
    self._imu_site_id = self._mj_model.site("imu").id

    # ---- v19: 地形高度采样器 ----
    # 纯 JAX 双线性插值, 与 C++ mj_rayHfield 逐点对齐 (p99=0.35mm, 见
    # sim/probe_terrain_scan_jax.py(已删除, 结论见实验记录))。mjx 的 JAX 后端不支持 hfield 碰撞,
    # mjx.ray 的 _RAY_FUNC 分发表里也没有 HFIELD (issue #2155 官方明确不做),
    # 所以训练循环里只能自己解析采样; warp 后端把 hfield 数据暴露成 JAX 数组。
    #
    # 索引映射 (§16.8.4 已用引擎射线确认, 无需再翻转):
    #   col = (x/sx + 1)/2 * (ncol-1)   col 沿 +x
    #   row = (y/sy + 1)/2 * (nrow-1)   row 沿 +y
    #   h   = 双线性(data[row, col]) * elevation_z
    # 注意 hfield size[3] = base_z 只撑几何包围盒, **不影响采样高度** (实测:
    # 当前 base_z=0.096 时 mj_ray 与 data*elevation_z 仍逐点一致)。
    #
    # 本场景 floor geom 是 pos=0/quat=单位四元数 (scene_mjx_terrain.xml), 所以
    # 世界坐标 == geom 局部坐标, 采样退化为纯查表; 代码里仍保留完整旋转形式,
    # 并对非平凡位姿给出警告, 避免以后挪动 floor 时静默采错 (§16.8 的教训)。
    self._hfield_ok = False
    self._scan_grid_body = None
    if self._config.height_scan.enable or self._config.terrain:
      self._setup_height_sampler()


    self._init_q = jp.array(self._mj_model.keyframe("home").qpos)
    self._default_pose = jp.array(self._mj_model.keyframe("home").qpos[7:])

    # 软关节范围 (参考仓库: ctrlrange 内缩 5%)
    ctrl_range_offset = (
        0.5 * (1 - 0.9) * (self.mj_model.actuator_ctrlrange[:, 1]
                            - self.mj_model.actuator_ctrlrange[:, 0])
    )
    self._soft_lowers = (self.mj_model.actuator_ctrlrange[:, 0]
                         + ctrl_range_offset)
    self._soft_uppers = (self.mj_model.actuator_ctrlrange[:, 1]
                         - ctrl_range_offset)

    self._torso_body_id = self._mj_model.body("trunk").id
    self._torso_mass = self._mj_model.body_subtreemass[self._torso_body_id]

    self._feet_site_id = np.array(
        [self._mj_model.site(name).id for name in FEET])
    self._feet_geom_id = np.array(
        [self._mj_model.geom(name).id for name in FEET])
    # 碰撞检测体 (参考 _cfrc_ext_contact_indices 语义: 非足端的碰撞 geom)
    self._contact_geom_id = np.array([
        gid for gid in range(self._mj_model.ngeom)
        if gid not in self._feet_geom_id
        and gid != self._mj_model.geom("floor").id
    ])

    # ---- v21: 非足端碰撞体的几何表 (用于 _cost_body_contact) ----
    # 动机 (§20): warp 后端**没有** data.contact (DataWarp 无该属性, 实测
    # AttributeError), 所以参考仓库的 collision_cost 一直是个空实现, 权重也
    # 是 0 —— 结果小腿/大腿/躯干拖地完全免费, 策略把膝盖当第五条腿用
    # (实测 RR/RL_calf 有 30% 的帧与地面接触, 前腿只有 4-5%)。
    #
    # 这里改成**纯几何**判定: 用 data.geom_xpos/geom_xmat 复算每个碰撞体的
    # 世界系最低点, 减去该点脚下的地形高度 = 净空。不需要接触数组, 在 warp
    # 后端可用, 且对平地/hfield 都成立。
    #
    # 只取"会碰到地"的部件: 小腿 + 大腿 + 躯干。髋部圆柱与躯干上的小盒子在
    # 正常姿态下离地 >15cm, 纳入只是徒增计算 (真要碰到时 z 范围终止已触发)。
    #
    # size 语义 (autolimits 编译后 fromto 已折成 pos+size):
    #   球     size[0]=半径
    #   胶囊/圆柱 size[0]=半径, size[1]=半长 (沿 geom 局部 z)
    #   盒     size[:3]=半边长
    _want_body = ("calf", "thigh", "trunk")
    _bcf_gid, _bcf_type, _bcf_r, _bcf_half, _bcf_box = [], [], [], [], []
    for gid in self._contact_geom_id:
      # 只看**参与碰撞**的 geom: _contact_geom_id 里还混着视觉 mesh
      # (contype=conaffinity=0), 它们的尺寸是整块网格, 拿来算净空没有意义。
      if (self._mj_model.geom_contype[gid] == 0
          and self._mj_model.geom_conaffinity[gid] == 0):
        continue
      body = self._mj_model.body(self._mj_model.geom_bodyid[gid]).name
      if not any(body.endswith(w) for w in _want_body):
        continue
      gtype = int(self._mj_model.geom_type[gid])
      size = np.asarray(self._mj_model.geom_size[gid], dtype=np.float32)
      is_sphere = gtype == mujoco.mjtGeom.mjGEOM_SPHERE
      is_box = gtype == mujoco.mjtGeom.mjGEOM_BOX
      _bcf_gid.append(gid)
      _bcf_type.append(gtype)
      _bcf_r.append(0.0 if is_box else float(size[0]))
      _bcf_half.append(0.0 if (is_sphere or is_box) else float(size[1]))
      _bcf_box.append([float(v) for v in size[:3]])
    self._bcf_gid = jp.array(_bcf_gid, dtype=jp.int32)
    self._bcf_type = jp.array(_bcf_type, dtype=jp.int32)
    self._bcf_r = jp.array(_bcf_r, dtype=jp.float32)
    self._bcf_half = jp.array(_bcf_half, dtype=jp.float32)
    self._bcf_box = jp.array(_bcf_box, dtype=jp.float32)
    # 每条腿的碰撞几何在 _bcf_* 里的索引 (诊断用)
    self._bcf_leg_idx = {f: [] for f in FEET}
    for i, gid in enumerate(_bcf_gid):
      leg = self._mj_model.body(
          self._mj_model.geom_bodyid[gid]).name.split("_")[0]
      if leg in self._bcf_leg_idx:
        self._bcf_leg_idx[leg].append(i)

    foot_linvel_sensor_adr = []
    for f in FEET:
      sensor_id = self._mj_model.sensor(f"{f}_global_linvel").id
      adr = self._mj_model.sensor_adr[sensor_id]
      dim = self._mj_model.sensor_dim[sensor_id]
      foot_linvel_sensor_adr.append(list(range(adr, adr + dim)))
    self._foot_linvel_sensor_adr = jp.array(foot_linvel_sensor_adr)

    self._feet_floor_found_sensor = [
        self._mj_model.sensor(f"{f}_floor_found").id for f in FEET
    ]

    # 固定指令 (评估/可视化); 训练时若 command_config.sample=True 则由 reset
    # 每 episode 重新采样, 实际生效的指令放在 info["command"] 里
    self._cmd = jp.array(self._config.command_config.fixed)
    self._cmd_low = jp.array(self._config.command_config.low)
    self._cmd_high = jp.array(self._config.command_config.high)

    # v20: 对角步态 (trot) 的每足目标相位, 顺序与 FEET 一致 = FR, FL, RR, RL。
    # 取自 mujoco_playground/_src/gait.py 的 GAIT_PHASES[0] (trot)。
    # 对角对 (FR,RL) 同相, (FL,RR) 反相 —— 这是"对角"的定义。
    self._trot_phase0 = jp.array([0.0, jp.pi, jp.pi, 0.0])

  # ---------- 访问器 ----------

  @property
  def xml_path(self) -> str:
    return self._xml_path

  @property
  def action_size(self) -> int:
    return self._mjx_model.nu

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self._mj_model

  @property
  def mjx_model(self) -> mjx.Model:
    return self._mjx_model

  # ---------- 传感器读取 ----------

  def get_upvector(self, data: mjx.Data) -> jax.Array:
    return mjx_env.get_sensor_data(self.mj_model, data, "upvector")

  def get_local_linvel(self, data: mjx.Data) -> jax.Array:
    return mjx_env.get_sensor_data(self.mj_model, data, "local_linvel")

  def get_global_linvel(self, data: mjx.Data) -> jax.Array:
    return mjx_env.get_sensor_data(self.mj_model, data, "global_linvel")

  def get_global_angvel(self, data: mjx.Data) -> jax.Array:
    return mjx_env.get_sensor_data(self.mj_model, data, "global_angvel")

  def get_accelerometer(self, data: mjx.Data) -> jax.Array:
    return mjx_env.get_sensor_data(self.mj_model, data, "accelerometer")

  def get_gyro(self, data: mjx.Data) -> jax.Array:
    return mjx_env.get_sensor_data(self.mj_model, data, "gyro")

  def get_gravity(self, data: mjx.Data) -> jax.Array:
    """机体系重力向量 (R^T @ g)。直立时 ≈ [0,0,-1]。"""
    return data.site_xmat[self._imu_site_id].T @ jp.array([0, 0, -1])

  def get_body_linvel(self, data: mjx.Data) -> jax.Array:
    """机体系线速度 (R^T @ v_global)。前进时 x 为正, 竖直 z 仍朝上。"""
    return data.site_xmat[self._imu_site_id].T @ data.qvel[0:3]

  # ---------- 地形高度采样 (v19) ----------

  def _setup_height_sampler(self) -> None:
    """把 hfield 提成常量 + 构造采样闭包。

    地形是静态的且所有环境共享同一份高度场, 所以 hfield_data 不参与 vmap,
    直接闭包捕获即可 (无 pytree/广播问题)。
    """
    if self._mj_model.nhfield < 1:
      raise ValueError(
          "height_scan/terrain 需要 hfield 地形, 但当前模型没有 hfield "
          "(scene_mjx_position.xml 是 plane)。请用 terrain=True。")
    hid = 0
    hf_name = self._mj_model.hfield(hid).name if self._mj_model.nhfield else ""
    # floor geom 是承载地形的那个 geom
    gid = self._mj_model.geom("floor").id
    if self._mj_model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_HFIELD:
      raise ValueError(
          f"geom 'floor' 不是 hfield (type={self._mj_model.geom_type[gid]}), "
          "高度采样会失效。地形场景应把 floor 设为 hfield。")

    self._hf_nrow = int(self._mj_model.hfield_nrow[hid])
    self._hf_ncol = int(self._mj_model.hfield_ncol[hid])
    self._hf_size = np.array(self._mj_model.hfield_size[hid])   # [sx,sy,elev,base]
    self._hf_gpos = np.array(self._mj_model.geom_pos[gid])
    self._hf_gquat = np.array(self._mj_model.geom_quat[gid])
    adr = int(self._mj_model.hfield_adr[hid])
    n = self._hf_nrow * self._hf_ncol
    # 二维表, 便于双线性索引 (row 沿 +y, col 沿 +x)
    self._hf_table = jp.array(
        np.asarray(self._mj_model.hfield_data[adr:adr + n]).reshape(
            self._hf_nrow, self._hf_ncol))

    # 位姿断言: 非平凡位姿会让"世界坐标->geom 局部"的映射出错, 采样位置静默
    # 偏移。当前 XML 是 pos=0/quat=id, 保留完整旋转实现以防有人挪动 floor。
    if not (np.allclose(self._hf_gpos, 0.0, atol=1e-9)
            and np.allclose(self._hf_gquat, np.array([1.0, 0, 0, 0]),
                            atol=1e-9)):
      print(f"[warn] floor geom 位姿非平凡 pos={self._hf_gpos} "
            f"quat={self._hf_gquat} —— 采样器会走完整旋转路径 "
            f"(正确但更贵); 若地形布局异常请复查 §16.8 的坐标约定。")

    # 采样网格 (机体系局部 xy), 一次性算好。顺序: x 外层, y 内层
    hs = self._config.height_scan
    gx = np.linspace(hs.x_min, hs.x_max, hs.nx)
    gy = np.linspace(hs.y_min, hs.y_max, hs.ny)
    GX, GY = np.meshgrid(gx, gy, indexing="ij")   # (nx, ny)
    self._scan_xy = jp.array(
        np.stack([GX.ravel(), GY.ravel()], axis=-1).astype(np.float32))  # (N,2)
    self._scan_n = int(hs.nx * hs.ny)
    self._hfield_ok = True

  def terrain_height(self, xy_world: jax.Array) -> jax.Array:
    """世界系 (..., 2) 平面坐标 -> 地面高度 (...,) (米)。纯 JAX, 可 jit/vmap。

    双线性插值。超出地形范围的点按边界 clamp (与 legged_gym 的 clip 语义一致)。
    """
    if not self._hfield_ok:
      raise RuntimeError("height sampler 未初始化 (需 terrain=True 或 height_scan.enable)")
    sz = jp.array(self._hf_size, dtype=jp.float32)
    lead = xy_world.shape[:-1]
    p = xy_world.reshape(-1, 2)

    # 世界 -> geom 局部 (本场景 geom_quat=id/pos=0, 下面 R 退化为单位阵)
    if np.allclose(self._hf_gquat, np.array([1.0, 0, 0, 0]), atol=1e-9):
      local = p - jp.array(self._hf_gpos[:2], dtype=jp.float32)
    else:
      # 兜底路径 (本场景不触发, 只保证挪动 floor 后不静默采错)。
      # 查询点取世界 z = gpos_z (平面近似, 无法知道真实 z), 故 local z=0。
      w, x, y, z = [jp.float32(v) for v in self._hf_gquat]
      R = jp.array([
          [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
          [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
          [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
      ])
      dif = p - jp.array(self._hf_gpos[:2], dtype=jp.float32)   # (M,2)
      dif3 = jp.concatenate([dif, jp.zeros_like(dif[:, :1])], axis=-1)
      local = (dif3 @ R)[:, :2]        # R^T @ dif  ==  dif @ R

    u = local[:, 0] / sz[0]
    v = local[:, 1] / sz[1]
    col = (u + 1.0) * 0.5 * (self._hf_ncol - 1)
    row = (v + 1.0) * 0.5 * (self._hf_nrow - 1)
    # 边界 clamp 到 [0, n-2], 使 r0+1/c0+1 始终合法
    col = jp.clip(col, 0.0, self._hf_ncol - 2.0)
    row = jp.clip(row, 0.0, self._hf_nrow - 2.0)

    r0 = jp.floor(row).astype(jp.int32)
    c0 = jp.floor(col).astype(jp.int32)
    fr = row - r0
    fc = col - c0
    t = self._hf_table
    h = ((1 - fr) * ((1 - fc) * t[r0, c0] + fc * t[r0, c0 + 1])
         + fr * ((1 - fc) * t[r0 + 1, c0] + fc * t[r0 + 1, c0 + 1]))
    return (h * sz[2]).reshape(lead)

  def height_scan_features(
      self, data: mjx.Data, rng: Optional[jax.Array] = None
  ) -> jax.Array:
    """35 维高度扫描特征 (机体系, yaw 对齐, 相对躯干高度)。

    特征 = clip(躯干z - 地面h - base, ±clip_m) * scale
      负 = 前方地面更高 (障碍/上坡)   正 = 前方下陷
    clip 作用在**米制差值**上 (legged_gym 同款), 所以 clip_m 必须大于地形
    总高差, 否则爬梯时特征全程饱和、梯度消失。
    网格只按 yaw 旋转 (不加 roll/pitch): 上坡时网格不跟着躯干倾斜, 否则
    特征会随姿态抖动 (IsaacLab 的 ray_alignment="yaw" 同理)。
    rng 非 None 时给地面高度加均匀噪声 (仅 actor 用; critic 传 None)。
    """
    hs = self._config.height_scan
    base_pos = data.qpos[:3]
    # yaw-only 旋转: 从四元数取水平朝向
    q = data.qpos[3:7]
    q = q / jp.maximum(jp.linalg.norm(q), 1e-8)
    w, x, y, z = q
    yaw = jp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    c, s = jp.cos(yaw), jp.sin(yaw)
    xy_local = self._scan_xy                                   # (N,2)
    xy_world = jp.stack([
        c * xy_local[:, 0] - s * xy_local[:, 1],
        s * xy_local[:, 0] + c * xy_local[:, 1],
    ], axis=-1) + base_pos[None, :2]
    h = self.terrain_height(xy_world)                          # (N,)
    if rng is not None and hs.noise > 0.0:
      h = h + jax.random.uniform(
          rng, h.shape, minval=-hs.noise, maxval=hs.noise)
    d_m = base_pos[2] - h - hs.base                            # (N,) 米制相对高差
    return jp.clip(d_m, -hs.clip_m, hs.clip_m) * hs.scale

  def terrain_height_under_feet(self, data: mjx.Data) -> jax.Array:
    """4 只脚各自足下地形高度 (世界 z, 米)。用于相对抬脚高度奖励。"""
    return self.terrain_height(data.site_xpos[self._feet_site_id][:, :2])

  # ---------- 终止 ----------

  def _is_healthy(self, data: mjx.Data) -> jax.Array:
    """复刻参考仓库 is_healthy 的原语义 (含其"四元数分量当欧拉角"的写法)。

    参考仓库:
      state = concat(qpos, qvel)
      healthy = 有限值 and 0.22 <= z <= 0.65
                and -10° <= state[4] <= 10°   # 实为 qpos[4] = 四元数 x
                and -10° <= state[5] <= 10°   # 实为 qpos[5] = 四元数 y
    四元数 x≈roll/2, y≈pitch/2 → 等效 |roll|,|pitch| < 20° (比真欧拉 10°
    宽一倍)。这是关键: 我们 v1-v6 用真欧拉 ±10° (gravity 分量阈值) 严了
    一倍, reset 噪声后落地即终止 → episode 秒死 → 奖励下限 0 下 PPO
    几乎没有正信号 (eval_reward 恒 0.000 的根因)。
    """
    min_z, max_z = self._config.healthy_z_range
    healthy = jp.isfinite(data.qpos).all() & jp.isfinite(data.qvel).all()
    # v19: z 判据改成**相对足下地形**的高度。
    # 原版直接用绝对 qpos[2] —— 平地上没问题 (地面 z=0), 但地形总高差 0.30m
    # 时, 站在最高台阶 z≈0.58 已贴近上界 0.65; 随出生点随机化后, 在 0.30m
    # 台顶出生 z=0.65 会**出生即终止** (§16.6.5 记的耦合隐患)。
    # 用躯干下方地面高度做基准后, 阈值语义恢复为"相对站立高度", 与地形幅度
    # 解耦 —— 以后加大地形不必再回来调阈值。
    z_ref = self.terrain_height(data.qpos[None, :2])[0] if self._hfield_ok else 0.0
    z_rel = data.qpos[2] - z_ref
    healthy &= (z_rel > min_z) & (z_rel < max_z)
    # 参考仓库语义: 直接卡四元数 x/y 分量 (≈ ±20° 欧拉)
    q = data.qpos[3:7]
    q = q / jp.linalg.norm(q)  # MuJoCo 内部归一化, 这里显式一致
    rad = jp.deg2rad(jp.float32(self._config.healthy_roll_range))
    healthy &= (jp.abs(q[1]) < rad) & (jp.abs(q[2]) < rad)
    return healthy

  def _get_termination(self, data: mjx.Data) -> jax.Array:
    return ~self._is_healthy(data)

  # ---------- reset ----------

  def reset(self, rng: jax.Array) -> mjx_env.State:
    qpos = jp.array(self._mj_model.keyframe("home").qpos)
    ri = self._config.random_init

    # 关节位置 U(-0.1, 0.1) 噪声 (参考 reset_model: 整个 qpos 加噪声)
    rng, key = jax.random.split(rng)
    s = self._config.reset_noise_scale
    qpos = qpos + jax.random.uniform(
        key, (self.mjx_model.nq,), minval=-s, maxval=s)
    # ctrl = key_ctrl + N(0, 0.1) (参考 reset_model)
    rng, key = jax.random.split(rng)
    ctrl = jp.array(self._mj_model.key_ctrl[0]) + s * jax.random.normal(
        key, (self.mjx_model.nu,))
    qvel = jp.zeros(self.mjx_model.nv)

    # ---- v19: 随机出生 (地形训练的前提) ----
    # 不随机 xy 的话 8192 个环境全落在原点那块半径 0.5m 的压平区
    # (flatten_origin(radius=0.5)), 永远见不到 0.06m 台阶和 30cm 楼梯。
    # 随机 xy + yaw 后, 一部分环境直接出生在斜坡/阶梯上, 另一部分在附近,
    # 走几步就能遇到地形 —— 这是"学会爬梯"的必要条件。
    if ri.enable:
      rng, kxy, kyaw, kv = jax.random.split(rng, 4)
      xy = jax.random.uniform(
          kxy, (2,), minval=-ri.xy_range, maxval=ri.xy_range)
      # z 取出生点地面高度 + 名义站高: 高于地面 → 自由落体沉降, 不会穿地
      if self._hfield_ok:
        h0 = self.terrain_height(xy[None, :])[0]
      else:
        h0 = jp.zeros(())
      z = h0 + ri.spawn_z_offset
      if ri.random_yaw:
        yaw = jax.random.uniform(kyaw, (), minval=-jp.pi, maxval=jp.pi)
      else:
        yaw = jp.zeros(())
      half = yaw * 0.5
      quat = jp.stack([jp.cos(half), jp.zeros(()), jp.zeros(()), jp.sin(half)])
      qpos = jp.concatenate([
          jp.stack([xy[0], xy[1], z]), quat, qpos[7:]])
      # 出生速度扰动 (小幅度): 让策略不是每次都从静止开始收拾
      if ri.random_vel > 0.0:
        qvel = jax.random.uniform(
            kv, (6,), minval=-ri.random_vel, maxval=ri.random_vel)
        qvel = jp.concatenate([qvel, jp.zeros(self.mjx_model.nv - 6)])
      # 注意: 上面的关节噪声作用在 qpos[7:] 前, 这里整体替换 xy/z/quat,
      # 关节部分仍保留噪声 —— 正是想要的语义。

    data = mjx_env.make_data(
        self.mj_model,
        qpos=qpos,
        qvel=qvel,
        ctrl=ctrl,
        impl=self.mjx_model.impl.value,
        naconmax=self._config.naconmax,
        njmax=self._config.njmax,
    )
    data = mjx.forward(self.mjx_model, data)
    # 参考仓库无沉降段: reset 即开始 (qvel=0, 噪声让 RL 自己收拾)。
    # ANYmal 工程的沉降是给 kp=100 45kg 大机器用的; Go1 12kg 初始瞬态温和。
    data = data.replace(time=0.0)

    # 指令: 训练时每 episode 从 [low, high] 采样 (参考仓库 _sample_desired_vel
    # 也是每个 reset 重采样, 只是他们把 min/max 设成了同一个点)
    if self._config.command_config.sample:
      rng, key = jax.random.split(rng)
      cmd = jax.random.uniform(
          key, (3,), minval=self._cmd_low, maxval=self._cmd_high)
    else:
      cmd = self._cmd

    info = {
        "rng": rng,
        # v19: 每步观测噪声用的 key (step 里原地推进, 保持 JAX 纯函数性)
        "noise_key": rng,
        "command": cmd,
        "last_act": jp.zeros(self.mjx_model.nu),
        "feet_air_time": jp.zeros(4),
        "last_contact": jp.zeros(4, dtype=bool),
        # v16: 摆动相足端最高点 (playground joystick 同款状态)
        "swing_peak": jp.zeros(4),
        # v20: 对角步态相位。每足有自己的 phi_i (trot = [0, π, π, 0]),
        # 每次 reset 随机一个整体相位偏移 —— 避免策略依赖固定的"起始相位",
        # 否则它可能学会只对齐 episode 开头的那一小段 (playground 也是随机的)。
        # 注意 phase_dt 在这里先给 0, 由 step 里按指令速度每步算 (见 step)。
        "gait_phase": self._trot_phase0 + jax.random.uniform(
            rng, (), minval=0.0, maxval=2 * jp.pi),
        "swing_clear_sum": jp.zeros(4),   # v20: 摆动相净空累计
        "swing_steps": jp.zeros(4),       # v20: 摆动相步数累计
        # v19: 推挤倒计时 (每 push.interval 步触发一次)
        "push_countdown": jp.array(self._config.push_config.interval,
                                   dtype=jp.int32),
        # v22: 出生点的地形高度 —— ascent 项 (相对出生点升高) 的参考点。
        # 存在 info 里而不是每步重算: 出生点固定, 只算一次。
        "h_spawn": self._spawn_ground(data),
        # v23: 6 维镜像偏差的滑动平均 (**带符号**, 见 _cost_symmetry)
        "sym_ema": jp.zeros(6),
    }

    metrics = {}
    for k in self._config.reward_config.scales.keys():
      metrics[f"reward/{k}"] = jp.zeros(())

    obs = self._get_obs(data, info)
    reward, done = jp.zeros(2)
    return mjx_env.State(data, obs, reward, done, metrics, info)

  # ---------- step ----------

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    # 动作语义 1:1 复刻参考仓库: 策略输出即 ctrl 目标值 (原始, 无缩放),
    # MuJoCo 按 actuator ctrlrange 内部 clamp (v8 修正: 此前用 [-1,1] 线性
    # 映射到 ctrlrange, 零动作 = ctrlrange 中点 → hip 目标 1.9 rad 远离
    # home 0.8, 初始策略甩腿 → 关节速度 7 rad/s → 成本压过奖励, 奖励被
    # clip 到 0 → PPO 全程无梯度, 是 v1-v7 学不会的直接原因之一)。
    lo = self.mj_model.actuator_ctrlrange[:, 0]
    hi = self.mj_model.actuator_ctrlrange[:, 1]
    motor_targets = jp.clip(action, lo, hi)

    # v19: 每步从 info 的 rng 流派生两个 key (观测噪声 + 推挤), 再推进流。
    # 用 info 里存 key 而不是依赖全局状态, 保持 reset/step 的函数式语义
    # (与原有 info 字典原地写入的约定一致)。
    rng, noise_key, push_key = jax.random.split(state.info["rng"], 3)
    state.info["rng"] = rng
    state.info["noise_key"] = noise_key

    # ---- v19: 外力推挤扰动 ----
    # 在物理积分**之前**给基座叠加水平速度, 所以本步奖励看到的是"真被推了"
    # 之后的真实速度 (物理自洽)。只在水平两维: 不动 vz (不伪造垂直速度),
    # 否则 vertical_vel 成本会被误触发。
    pc = self._config.push_config
    data_in = state.data
    if pc.enable:
      kick = jax.random.uniform(
          push_key, (2,), minval=-pc.max_vel, maxval=pc.max_vel)
      do_push = state.info["push_countdown"] <= 0
      data_in = data_in.replace(qvel=data_in.qvel.at[:2].add(
          jp.where(do_push, kick, jp.zeros(2))))
      state.info["push_countdown"] = jp.where(
          do_push, jp.array(pc.interval, dtype=jp.int32),
          state.info["push_countdown"] - 1)
    else:
      state.info["push_countdown"] = jp.zeros((), dtype=jp.int32)

    data = mjx_env.step(
        self.mjx_model, data_in, motor_targets, self.n_substeps
    )
    # NaN 护栏 (v8 起, v10 加固): warp 求解器在极端自碰撞/摔倒下可能发散。
    # v8 只回填了 qpos/qvel, 但奖励还读 qacc / qfrc_actuator / sensordata,
    # obs 还读 site_xmat / actuator_force —— 这些仍是非有限值时奖励变 NaN,
    # GAE/优势变 NaN, 整个参数被打爆 (v9 在 147K 步 NaN 的直接原因)。
    # 现在把本步用到的所有 data 字段一并回填成有限值, 并让 done 立即置位。
    finite = jp.isfinite(data.qpos).all() & jp.isfinite(data.qvel).all()
    data = data.replace(
        qpos=jp.where(finite, data.qpos, self._init_q),
        qvel=jp.where(finite, data.qvel,
                      jp.zeros(self.mjx_model.nv)),
        qacc=jp.where(finite, data.qacc,
                      jp.zeros(self.mjx_model.nv)),
        qfrc_actuator=jp.where(finite, data.qfrc_actuator,
                               jp.zeros(self.mjx_model.nv)),
        actuator_force=jp.where(finite, data.actuator_force,
                                jp.zeros(self.mjx_model.nu)),
        sensordata=jp.where(finite, data.sensordata,
                            jp.zeros(self.mjx_model.nsensordata)),
        site_xmat=jp.where(finite, data.site_xmat,
                           jp.zeros_like(data.site_xmat)),
    )

    contact = jp.array([
        data.sensordata[self._mj_model.sensor_adr[sid]] > 0
        for sid in self._feet_floor_found_sensor
    ])
    contact_filt = contact | state.info["last_contact"]
    first_contact = (state.info["feet_air_time"] > 0.0) * contact_filt
    state.info["feet_air_time"] += self.dt
    # v16: 摆动相足端最高点 (在清零之前先更新, 与 playground 同序)。
    # v19: 改成**离地净空** (足端 z − 足下地形高度), 不再是绝对世界 z。
    # 原因 (实测, §16.6.4): 绝对 z 在 0.06m 台阶上把拖地惩罚从 0.891 削到
    # 0.038, 地形 >=0.08m 处几乎归零 → 策略会学"专门在高处拖地"; 0.30m 处
    # 还会反向狂罚正常站立。改成净空后, "抬脚 0.08m" 的语义在任何地面高度上
    # 都成立。平地档 (无 hfield) 净空 == 绝对 z, 与原行为逐位一致。
    if self._hfield_ok:
      foot_z = data.site_xpos[self._feet_site_id][..., -1]
      clearance = foot_z - self.terrain_height_under_feet(data)
    else:
      clearance = data.site_xpos[self._feet_site_id][..., -1]
    state.info["swing_peak"] = jp.maximum(state.info["swing_peak"], clearance)
    # v20: 摆动相净空累计 (feet_height 改成"整段均值"结算, 见 _cost_feet_height)。
    # 只累计"本步离地"的腿; 仍在摆动中的腿每步都累加, 落地帧结算后清零。
    in_swing = ~contact
    state.info["swing_clear_sum"] += clearance * in_swing
    state.info["swing_steps"] += in_swing.astype(clearance.dtype)

    # v20: 相位推进。step_dt = 2π·f·dt; f 可选随指令速度缩放 (站着不踏步)。
    # playground 用固定 f (不随指令), 但那会让"站着不动也踏步" —— 我们的指令
    # 含 vx=0 档 (cmd 采样 low=[0,0,-1]), 所以提供缩放开关。
    gf = self._config.reward_config.gait_freq
    if self._config.reward_config.gait_freq_scaled_by_cmd:
      spd = jp.linalg.norm(state.info["command"][:2])
      f_eff = jp.maximum(gf * spd / 0.5,
                         self._config.reward_config.gait_freq_min)
    else:
      f_eff = jp.array(gf)
    phase_next = state.info["gait_phase"] + 2 * jp.pi * f_eff * self.dt
    # 归一化到 [-π, π) (playground 同款), 避免长时间累加丢精度
    state.info["gait_phase"] = (
        jp.fmod(phase_next + jp.pi, 2 * jp.pi) - jp.pi)

    # ---- v23: 左右镜像对称偏差 (带符号的 DC 估计) ----
    # **关键**: 必须对**带符号的偏差**做滑动平均, 之后再平方 —— 不能对"平方"
    # 做平均。原因 (实测标定暴露): trot 的左右腿反相, 瞬时镜像差恒非零,
    # E[(x_R - x_L)²] 收敛到**步态振幅的平方**, 与平均无关 -> 那样罚的是
    # "腿摆了多少", 会把 trot 压平。而 E[x_R - x_L] 对反相信号一阶抵消,
    # 只留下**静偏置**(后腿 hip 共模、单侧膝多弯), 正是要修的。
    # 实测量级: v22@40M = 1.19, v18(对称那代) = 0.045 -> 26x 区分度。
    qj = data.qpos[..., 7:].reshape(*data.qpos.shape[:-1], 4, 3)   # FR,FL,RR,RL
    _hip, _thi, _cal = qj[..., 0], qj[..., 1], qj[..., 2]
    # 镜像要求: hip_R = -hip_L (轴保持, 角变号); thigh/calf_R = thigh/calf_L
    _mvec = jp.stack([
        _hip[..., 0] + _hip[..., 1],      # FR + FL   (对称应为 0)
        _hip[..., 2] + _hip[..., 3],      # RR + RL
        _thi[..., 0] - _thi[..., 1],      # FR - FL
        _thi[..., 2] - _thi[..., 3],      # RR - RL
        _cal[..., 0] - _cal[..., 1],      # FR - FL
        _cal[..., 2] - _cal[..., 3],      # RR - RL
    ], axis=-1)                            # (..., 6)
    _beta = self._config.reward_config.symmetry_beta
    state.info["sym_ema"] = ((1.0 - _beta) * state.info["sym_ema"]
                             + _beta * _mvec)

    obs = self._get_obs(data, state.info)
    done = self._get_termination(data) | (~finite)

    rewards = self._get_reward(
        data, action, state.info, done, first_contact, contact
    )
    rewards = {
        k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
    }
    # SB3 语义: reward 是每 step 值 (不乘 dt), 下限 0 (参考 max(0, r-c))
    total = sum(rewards.values())
    total = jp.where(jp.isfinite(total), total, 0.0)
    reward = jp.clip(total, self._config.reward_clip_min, 10000.0)

    state.info["last_act"] = action
    # 参考仓库用 contact_filter (= 本步接触 | 上步接触) 清零 air_time
    state.info["feet_air_time"] *= ~contact_filt
    # v16: 落地后清零摆动峰值 (与 playground 同序: 奖励读的是"本步落地"的峰值)
    state.info["swing_peak"] *= ~contact
    # v20: 落地帧结算完摆动相统计后清零 (下个摆动相重新累计)
    state.info["swing_clear_sum"] *= ~contact
    state.info["swing_steps"] *= ~contact
    state.info["last_contact"] = contact
    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v

    done = done.astype(reward.dtype)
    state = state.replace(data=data, obs=obs, reward=reward, done=done)
    return state

  # ---------- obs ----------

  def _obs_noise_vec(self) -> jax.Array:
    """观测噪声幅度向量 (48 维, 与基座 obs 同布局)。

    与 legged_gym 的 _get_noise_scale_vec 同构: 噪声按**已缩放**的 obs 量级
    给出 (noise_scale × obs_scale), 均匀分布 ±该值。指令与上一步动作不加噪
    (指令是自己下的, 上一步动作是自己输出的, 没有传感器误差)。
    """
    sc = self._config.obs_scale
    on = self._config.obs_noise
    z3 = jp.zeros(3)
    return jp.concatenate([
        jp.full((3,), on.lin_vel * sc.linear_velocity),   # base lin vel
        jp.full((3,), on.ang_vel * sc.angular_velocity),  # base ang vel
        jp.full((3,), on.gravity),                        # projected gravity
        z3,                                               # command (无噪)
        jp.full((12,), on.dof_pos * sc.dofs_position),    # dof pos
        jp.full((12,), on.dof_vel * sc.dofs_velocity),    # dof vel
        jp.zeros(12),                                     # last action (无噪)
    ])

  def _get_obs(
      self, data: mjx.Data, info: dict[str, Any]
  ) -> Dict[str, jax.Array]:
    sc = self._config.obs_scale
    # 参考仓库直接用全局 qvel[:3]。frame="body" 时换成机体系 (朝向) 速度 ——
    # 指令条件化做转向时, obs 里必须有机体系速度, 否则策略看不到自己的朝向,
    # 无法把"朝向速度指令"对上 (gravity 对 yaw 不变, obs 里没有别的 yaw 信息)。
    if self._config.command_config.frame == "body":
      base_linear_velocity = self.get_body_linvel(data) * sc.linear_velocity
    else:
      base_linear_velocity = data.qvel[0:3] * sc.linear_velocity
    base_angular_velocity = data.qvel[3:6] * sc.angular_velocity
    projected_gravity = self.get_gravity(data)
    dofs_position = data.qpos[7:] - jp.array(
        self._mj_model.key_ctrl[0])
    dofs_velocity = data.qvel[6:]

    # v19: base (48 维, **无噪**) —— 这是"真值", critic 用它。
    # 原实现把 state 直接当 critic 的前缀; 现在 actor 侧要加传感器噪声,
    # 若沿用会让噪声灌进 critic, 特权的意义就没了。故显式拆成三段:
    #   clean base (48)  ->  actor 加噪 / critic 原样
    #   height scan (35) ->  actor 带噪 / critic 无噪
    #   privileged extras (71) -> 仅 critic
    base = jp.hstack([
        base_linear_velocity,            # 3
        base_angular_velocity,           # 3
        projected_gravity,               # 3
        info["command"] * sc.linear_velocity,  # 3 (指令条件化的入口)
        dofs_position,                   # 12
        dofs_velocity,                   # 12
        info["last_act"],                # 12
    ])

    hs = self._config.height_scan
    if hs.enable:
      if not self._hfield_ok:
        raise RuntimeError("height_scan.enable=True 但采样器未初始化 "
                           "(需要 terrain=True)")
      scan_clean = self.height_scan_features(data, rng=None)
      scan_actor = self.height_scan_features(data, rng=info["noise_key"])
    else:
      scan_clean = scan_actor = None

    # v20: 步态相位进 obs (cos, sin) —— **这是相位奖励能起作用的必要条件**。
    # playground 的 joystick_gait_tracking 也是这么做的 (它把 cos/sin 拼进 obs)。
    # 为什么必须给: 相位奖励的目标 rz(phi) 是**随相位周期变化**的, 策略若看不到
    # 自己处在哪个相位, rz 对它就是"无法预测的时变目标" = 噪声, 学不会。
    # 实测证据 (§19.11): 没给相位时 corr(实际足高, 目标rz) ≈ 0.00, feet_phase
    # 恒为 0; 给了之后才有可学性。
    # 用 cos/sin 而不是裸相位: 后者在 ±π 处不连续, 网络难以拟合周期函数。
    if self._config.reward_config.scales.get("feet_phase", 0.0) != 0.0:
      ph = info["gait_phase"]
      phase_enc = jp.concatenate([jp.cos(ph), jp.sin(ph)])   # (8,) 4足 x 2
    else:
      phase_enc = None

    # ---- actor obs ----
    if self._config.obs_noise.enable:
      no = (2.0 * jax.random.uniform(info["noise_key"], base.shape) - 1.0)
      base_actor = base + no * self._obs_noise_vec()
    else:
      base_actor = base
    actor_parts = [base_actor] if scan_actor is None else [base_actor, scan_actor]
    if phase_enc is not None:
      actor_parts.append(phase_enc)          # v20: 相位 (cos,sin) 进 actor
    state = jp.clip(jp.hstack(actor_parts),
                    -self._config.obs_clip, self._config.obs_clip)

    # ---- critic obs (无噪, 含全部特权信息) ----
    priv_parts = [base]
    if scan_clean is not None:
      priv_parts.append(scan_clean)
    if phase_enc is not None:
      priv_parts.append(phase_enc)           # critic 也拿 (无噪, 与 actor 同源)
    priv_parts += [
        self.get_gyro(data),                  # 3
        self.get_accelerometer(data),         # 3
        projected_gravity,                    # 3
        self.get_global_linvel(data),         # 3
        self.get_global_angvel(data),         # 3
        dofs_position,                        # 12
        dofs_velocity,                        # 12
        data.actuator_force,                  # 12
        info["last_contact"].astype(jp.float32),  # 4
        data.sensordata[self._foot_linvel_sensor_adr].ravel(),  # 12
        info["feet_air_time"],                # 4
    ]
    privileged_state = jp.hstack(priv_parts)

    return {
        "state": state,
        "privileged_state": privileged_state,
    }

  # ---------- 奖励 (参考 _calc_reward 1:1) ----------

  def _get_reward(
      self,
      data: mjx.Data,
      action: jax.Array,
      info: dict[str, Any],
      done: jax.Array,
      first_contact: jax.Array,
      contact: jax.Array,
  ) -> Dict[str, jax.Array]:
    cmd = info["command"]
    return {
        "linear_vel_tracking": self._reward_linear_vel_tracking(data, cmd),
        "angular_vel_tracking": self._reward_angular_vel_tracking(data, cmd),
        "feet_airtime": self._reward_feet_air_time(
            info["feet_air_time"], first_contact, cmd),
        "feet_slip": self._cost_feet_slip(data, contact, cmd),
        "feet_height": self._cost_feet_height(
            info["swing_clear_sum"], info["swing_steps"], first_contact, cmd),
        "feet_dangle": self._cost_feet_dangle(
            info["feet_air_time"], contact, cmd),
        # v19: 显式爬升激励 + 下降引导
        "base_height": self._cost_base_height(data),
        "feet_impact": self._cost_feet_impact(data, contact),
        # v20: 对角步态相位引导
        "feet_phase": self._reward_feet_phase(data, info["gait_phase"]),
        "torque": self._cost_torque(data),
        "action_rate": self._cost_action_rate(action, info["last_act"]),
        "vertical_vel": self._cost_vertical_vel(data),
        "xy_angular_vel": self._cost_xy_angular_vel(data),
        "joint_limit": self._cost_joint_limit(data),
        "joint_velocity": self._cost_joint_velocity(data),
        "joint_acceleration": self._cost_joint_acceleration(data),
        "orientation": self._cost_orientation(data),
        "collision": self._cost_collision(data),
        # v21: 非足端部件撞地 (膝盖/小腿/大腿/躯干拖地)。修 §20 的漏洞。
        "body_contact": self._cost_body_contact(data),
        # v22: 相对出生点的地形升高 (显式爬升激励)。修 §24 的激励缺口:
        # 原奖励面没有任何一项随"站得更高"增加, 绕开楼梯与爬上去等价。
        "ascent": self._reward_ascent(data, info["h_spawn"]),
        # v23: 左右镜像对称 (时间平均偏差), 权重为负
        "symmetry": self._cost_symmetry(info),
        "default_joint_position": self._cost_default_joint_position(data),
        "termination": done,
        "pose": 0.0,
        "stand_still": 0.0,
    }

  def _reward_linear_vel_tracking(
      self, data: mjx.Data, cmd: jax.Array) -> jax.Array:
    # 参考 linear_velocity_tracking_reward: exp(-||v_cmd - qvel[:2]||²/σ)
    # frame="body" 时改用机体系速度 (边走边转的必要条件)
    if self._config.command_config.frame == "body":
      v = self.get_body_linvel(data)[:2]
    else:
      v = data.qvel[0:2]
    vel_sqr_error = jp.sum(jp.square(cmd[:2] - v))
    return jp.exp(-vel_sqr_error
                  / self._config.reward_config.tracking_velocity_sigma)

  def _reward_angular_vel_tracking(
      self, data: mjx.Data, cmd: jax.Array) -> jax.Array:
    ang_vel_error = jp.square(cmd[2] - data.qvel[5])
    return jp.exp(-ang_vel_error
                  / self._config.reward_config.tracking_velocity_sigma)

  def _reward_feet_air_time(
      self, air_time: jax.Array, first_contact: jax.Array, cmd: jax.Array
  ) -> jax.Array:
    # 参考 feet_air_time_reward 结构: air_time 以秒累计 (每步 += dt), 首次
    # 接触时奖励 Σ(air_time - threshold)*first_contact。
    # v16 两处修正 (参考值下这一项在鼓励"不抬腿", 见 §10.3):
    #   - threshold 1.0 → 0.2 (playground 用 0.1): 正常摆动 0.2-0.4s 落地
    #     得 +0.0~+0.2, 而不是参考的 -0.6~-0.8;
    #   - 封顶 air_time_max: 吊腿 10s 落地从 +9.8 降到 +0.3, 与正常步态的
    #     每秒收益相比不再有优势。
    # 纯转向 (vx=vy=0) 时整项关掉 (参考语义)。
    capped = jp.minimum(air_time, self._config.reward_config.air_time_max)
    rew = jp.sum((capped - self._config.reward_config.air_time_threshold)
                 * first_contact)
    rew *= jp.linalg.norm(cmd[:2]) > self._config.reward_config.cmd_threshold
    return rew

  def _cost_feet_slip(
      self, data: mjx.Data, contact: jax.Array, cmd: jax.Array
  ) -> jax.Array:
    # v16: 接触相足端水平速度平方和 (playground go1/joystick.py:512 同款)。
    # 足端 framelinvel 传感器给的是世界系速度, 真支撑时 ≈0, 打滑时 ≈ 躯干
    # 速度 (v10-v15 实测 0.46 m/s) → 这一项直接惩罚"贴地滑行"。
    # 只在有速度指令时生效 (站立时脚动一点不算打滑)。
    feet_vel = data.sensordata[self._foot_linvel_sensor_adr]  # (4, 3)
    vel_xy_norm_sq = jp.sum(jp.square(feet_vel[..., :2]), axis=-1)  # (4,)
    rew = jp.sum(vel_xy_norm_sq * contact)
    return rew * (jp.linalg.norm(cmd[:2]) > self._config.reward_config.cmd_threshold)

  def _cost_feet_dangle(
      self, air_time: jax.Array, contact: jax.Array, cmd: jax.Array
  ) -> jax.Array:
    # v17: 单足连续悬空超限惩罚 (每步生效, 不等落地)。
    # feet_slip / feet_height 都只在触地帧结算 → 永不落地的腿躲得掉,
    # v16 因此出现 FL 吊挂 14.7s (duty 0.01)。这里按超出量逐步扣分。
    # 封顶 2.0s: 不封顶时吊 14s 会到 -13/步, 把总奖励压到 0 (PPO 无梯度)。
    # 封顶后吊腿代价 ≈ -2/步 ≈ 整个速度跟踪奖励, 足够强但不会打死信号。
    over = jp.clip(air_time - self._config.reward_config.air_time_limit,
                   min=0.0, max=2.0)
    rew = jp.sum(over * (~contact))
    return rew * (jp.linalg.norm(cmd[:2]) > self._config.reward_config.cmd_threshold)

  def _cost_feet_height(
      self, swing_sum: jax.Array, swing_steps: jax.Array,
      first_contact: jax.Array, cmd: jax.Array
  ) -> jax.Array:
    """摆动相抬脚高度塑形 (**按摆动相均值结算**, v20 改)。

    v16 原实现: 落地帧用 swing_peak 算 (peak/target - 1)²。
    **v20 改动原因 (§18.10.3)**: 与 feet_impact 同源的漏洞 —— 只在落地帧结算,
    于是"不落地"完全免罚, 策略学会高频轻点地绕过它。改成"整个摆动相内
    净空的平均值"后, 抬腿高度在摆动全程都被考核, 点地绕不过去。

    语义: target = max_foot_height。摆动相平均净空接近 target -> 0 罚;
    贴地拖行 (均值≈0.02) -> error≈-0.75, 平方 0.56/足。
    无摆动 (swing_steps=0) 时返回 0 (那一步没有可考核的摆动相)。
    """
    if self._config.reward_config.scales.get("feet_height", 0.0) == 0.0:
      return jp.zeros(())
    target = self._config.reward_config.max_foot_height
    mean_clear = swing_sum / jp.maximum(swing_steps, 1.0)
    error = mean_clear / target - 1.0
    # 只在"本步有摆动在进行"的腿上考核 (swing_steps>0), 且该段已结束或进行中
    # 都算 —— 用 first_contact 标记"这一段摆动刚结束"的时刻结算, 但用的是
    # **整段均值**而不是峰值, 所以点地无法绕开。
    active = swing_steps > 0.0
    rew = jp.sum(jp.square(error) * first_contact * active)
    return rew * (jp.linalg.norm(cmd[:2]) > self._config.reward_config.cmd_threshold)

  # ---------- v19 新增奖励: 爬升激励 + 下降引导 ----------

  def _spawn_ground(self, data: mjx.Data) -> jax.Array:
    """出生点处的地形高度 (ascent 项的参考点)。平地档恒为 0。"""
    if not self._hfield_ok:
      return jp.zeros(())
    return self.terrain_height(data.qpos[None, :2])[0]

  def _reward_ascent(self, data: mjx.Data, h_spawn: jax.Array) -> jax.Array:
    """相对**出生点**的地形升高奖励 (v22, 显式爬升激励)。

    为什么需要它 (§24 的激励缺口): 原奖励面里**没有任何一项随"站得更高"增加**:
      - `linear_vel_tracking` / `angular_vel_tracking`: 只与速度有关, 与高度无关;
      - `base_height`: 是**相对**量 (躯干离足下地面多高), 地形上爬升不改变它;
      - `feet_phase` / `feet_height` / `feet_slip`: 都是足端**净空**或滑移, 与地形高度无关;
      - 各项 cost: 与高度无关。
    于是"绕开楼梯在平地跑"与"爬上楼梯"拿到的奖励**完全相同**(而绕行更容易),
    策略自然学会绕行。实测 (§24.3): 把 v21 策略直接放在楼梯脚下正对楼梯,
    它走 2.0m 却横向漂移 1.4m 绕开了楼梯, 最大爬升仅 4.7cm —— 证明是
    **激励**问题而非能力问题。

    形式: `h_now - h_spawn`, clip 到 [0, ascent_cap]。
      - **必须用绝对地形高度差**, 不能用"躯干相对地面"的量 —— 后者恰恰是
        base_height 那种地形不变量的写法, 正是本项要补的缺口。
      - clip 下限 0: 下坡不罚 (本项目目标是能上能下, 下坡由 feet_impact 管);
      - clip 上限: 防某个环境爬到顶后靠一项吃满奖励。
    """
    if self._config.reward_config.scales.get("ascent", 0.0) == 0.0:
      return jp.zeros(())
    if not self._hfield_ok:
      return jp.zeros(())        # 平地档: h 恒为 0, 该项无信息
    h_now = self.terrain_height(data.qpos[None, :2])[0]
    rise = h_now - h_spawn
    # v23: 行进门控 —— 静止时 |v|->0, 该项归零, 于是 W 不再被"站桩套利"卡在 2。
    # 只对**上升**部分门控 (下降时 rise<=0 本来就被 clip 掉)。
    _g = self._config.reward_config.ascent_gate
    if _g > 0.0:
      _v = jp.linalg.norm(self.get_body_linvel(data)[:2])
      rise = rise * jp.clip(_v / _g, 0.0, 1.0)
    return jp.clip(rise, 0.0, self._config.reward_config.ascent_cap)

    # 权重上界 (**必须守住, 否则退化成"爬上顶站着不动"**):
    # 这是**状态型**奖励 (每步按当前高度给分), 所以在台顶站着不动可以持续收钱。
    # 用实测分项均值做的套利算术 (sim/probe_v22_reward_terms.py 口径):
    #   走路（平地, v21 实测） 每步 ≈ +1.776
    #   站着不动（不高）       每步 ≈ +0.347  (速度/相位/冲击项全部塌掉)
    #   站台顶 = 0.347 + W*0.30
    #     W=2 -> +0.947  (走路 1.776 的 53%, 无套利)
    #     W=4 -> +1.547  (88%, 余量仅 13%, 偏险)
    #     W=8 -> +2.747  **超过走路 -> 站着比走着好, 会主动退化**
    # 故默认 W=2.0 (留 47% 余量), 并把 W=8 记为禁区。
    # 另一条更强的路 (未采用, 留记录): 用"只奖励新高"的势能差形式
    # (potential-based shaping, Ng 1999) 可证明不改变最优策略、也躲得掉
    # 站桩套利, 但信号更稀疏 —— 当前策略爬升极少时几乎全程为 0。

  def _cost_base_height(self, data: mjx.Data) -> jax.Array:
    """相对地形的躯干高度跟踪 (显式爬升激励)。

    legged_gym 原式: (mean(root_z - measured_heights) - target)²
    减去 mean(measured_heights) 这一步使它对**地形高度不变** —— 平地上等于
    绝对高度, 地形上则衡量"躯干离足下地面多高"。

    为什么需要它 (这是 §17 指出的激励缺口): 原奖励里没有一项关心地形进程,
    爬楼梯的唯一动机是"为了继续跟踪速度指令", 而楼梯只占 10.8% 面积 ——
    **绕开楼梯在平地上跑, 与爬上去, 拿到的速度跟踪奖励相同**。加上本项后,
    在楼梯上维持正常站高本身有正收益, 绕行不再等价。

    与 feet_height 的区别: feet_height 管"摆动相抬脚多高"(离地净空), 本项管
    "躯干离地多高"(站立高度)。两者互补, 不重复。

    实现要点: 用躯干正下方 + 4 只脚共 5 点取地面高度并取均值。legged_gym 用
    整张 height scan 求均值; 我们只用脚下 5 点 —— 更便宜, 且更贴近"当前支撑面"
    (上坡时躯干下方的地面比"前方 0.9m 处"的地面更能代表站高基准)。

    **v22 备注 (实测量化后撤回了一个错误猜想)**: 曾怀疑 5 点均值会在"抬腿迈上
    高处"时抬高参考面 -> 加重惩罚 -> 对抗爬升。三个探针 (sim/check_v22_bh_ref.py,
    sim/check_v22_bh_gradient.py) 的结论是**该猜想不成立**:
      - 参考点之差 h_mean5 - h_trunk 是**混合符号** (51.7% 为正, 43.3% 为负,
        均值 -0.46mm), 不是稳定抬高;
      - 沿楼梯上升时, "额外惩罚 vs 地面高度"的斜率 corr = **+0.13** (无系统
        方向) —— 即它是随足端落点的**振荡噪声**, 不是定向阻力;
      - 但量级不小: 均值 -0.034/步, 峰值 -0.083 (= 走路每步 +1.776 的 4.7%)。
    故 v22 **默认保留 mean5** (保持与 v21 一致, 让 ascent 成为唯一变量);
    想要更干净的站高信号可用 base_height_ref="trunk" 做单独对照。
    这段留给后人: 直觉上很合理的"artifact", 实测可能是噪声 —— 先量化再改。
    """
    if self._config.reward_config.scales.get("base_height", 0.0) == 0.0:
      return jp.zeros(())
    if not self._hfield_ok:
      return jp.zeros(())        # 平地档: 地面恒为 0, 该项退化为常数, 无信息
    # 参考点选择 (默认 mean5 = 与 v21 一致; "trunk" 为实验性对照)。
    # 两者的实测量化差异见本函数 docstring 的 v22 备注。
    rc = self._config.reward_config
    if rc.get("base_height_ref", "mean5") == "trunk":
      h_ground = self.terrain_height(data.qpos[None, :2])[0]
    else:
      pts = jp.concatenate([
          data.qpos[None, :2],
          data.site_xpos[self._feet_site_id][:, :2],
      ], axis=0)
      h_ground = jp.mean(self.terrain_height(pts))
    z_rel = data.qpos[2] - h_ground
    target = self._config.reward_config.base_height_target
    return jp.square(z_rel - target)

  def _cost_feet_impact(
      self, data: mjx.Data, contact: jax.Array
  ) -> jax.Array:
    """落地冲击惩罚 (**按时间结算**, 不再只在落地那一帧)。

    动机 (下楼梯/下台阶): 从高处落下时腿若以高速垂直触地, 冲击大、易弹跳、
    也更容易摔。原奖励里 `vertical_vel` 只惩罚**躯干**垂直速度, 对"脚先砸下去"
    没有约束; `feet_slip` 只罚水平滑移。

    **v20 关键改动 (修 §18.10.3 的激励漏洞)**: v19 用 `first_contact` 掩码,
    只在落地的**那一帧**结算 —— 于是"不落地 = 完全免罚"。策略因此学会高频
    轻点地 (摆动中位 0.02s) 来绕开这一项, 代价是前腿几乎不承重 (占空比 13%)。
    改成按时间结算后: 只要足端处于**接近地面**(或已接触)的状态, 其向下速度
    每步都被检查 —— "悬着不落"不再免费, 唯一免罚的方式是"接近地面时向下
    速度小", 也就是真正的轻柔落地。

    实现: 惩罚量按 `(1 - 归一化净空)` 加权 —— 离地越高权重越小 (高处的下落
    速度不算"砸地", 那是正常摆动), 越接近地面权重越大 (要落下了还很快 = 砸)。
    权重 w = clamp(1 - clearance / impact_height, 0, 1), 线性过渡。
    """
    if self._config.reward_config.scales.get("feet_impact", 0.0) == 0.0:
      return jp.zeros(())
    feet_vel = data.sensordata[self._foot_linvel_sensor_adr]   # (4,3) 世界系
    vz = feet_vel[..., 2]                                       # 向下为负
    excess = jp.clip(-vz - self._config.reward_config.impact_vel_limit, min=0.0)
    if self._hfield_ok:
      clearance = (data.site_xpos[self._feet_site_id][..., -1]
                   - self.terrain_height_under_feet(data))
    else:
      clearance = data.site_xpos[self._feet_site_id][..., -1]
    # 接近地面的程度: 净空 0 -> w=1, 净空 >= impact_height -> w=0
    h_scale = self._config.reward_config.impact_height
    w = jp.clip(1.0 - clearance / h_scale, 0.0, 1.0)
    # 已接触的帧 w 必然 ≈1 (净空≈0), 所以无需再用 contact 掩码 —— 保留参数
    # 只为接口清晰 (也便于以后想改成"仅触地"+时间加权时切换)。
    del contact
    return jp.sum(jp.square(excess) * w)

  # ---------- v20: 对角步态引导 (相位奖励) ----------

  def _get_rz(self, phi: jax.Array, swing_height: float) -> jax.Array:
    """目标足高轨迹 (相位 -> 应有足端高度)。

    逐行复刻 mujoco_playground/_src/gait.py 的 get_rz:
      x = (phi + π) / 2π
      x <= 0.5 -> 支撑相: cubic bezier 从 0 升到 swing_height (前半个周期)
      x >  0.5 -> 摆动相: cubic bezier 从 swing_height 落回 0 (后半个周期)
      bezier(t) = t³ + 3t²(1-t)   (三次贝塞尔, 两端导数为 0 -> 落地轻)
    """
    def bezier(y0, y1, x):
      y_diff = y1 - y0
      b = x ** 3 + 3 * (x ** 2 * (1 - x))
      return y0 + y_diff * b

    x = (phi + jp.pi) / (2 * jp.pi)
    stance = bezier(0.0, swing_height, 2 * x)
    swing = bezier(swing_height, 0.0, 2 * x - 1)
    return jp.where(x <= 0.5, stance, swing)

  def _reward_feet_phase(
      self, data: mjx.Data, phase: jax.Array
  ) -> jax.Array:
    """对角步态相位奖励 (playground spot joystick_gait_tracking 同款结构)。

    为什么需要它 (§18.10): v19 20M 退化成"前/后成对"步态 —— 前腿占空比 13%、
    后腿 78%、对角同步仅 13%。纯靠成本项惩罚不出步态, 必须有**显式的相位引导**。

    机制: 每足有目标相位 phi_i (trot = [0, π, π, 0]), 相位随时间循环推进,
    `_get_rz` 给出该相位下应有的足端高度, 奖励 = exp(-Σ(z - rz)²/σ)。
    对角对的相位相反 -> 奖励面直接鼓励"对角同起同落"。

    **关键设计: 用"离地净空"而非绝对世界 z** (与 feet_height 同理, §16.6.4)。
    地形上地面高度非零, 若用绝对 z 则目标轨迹 rz 必须跟着地形变, 而 rz 是
    相位的函数 —— 两者无法调和。改用净空 (足端 z − 足下地形高度) 后, rz 的
    语义在任何地面高度上都成立。
    """
    if self._config.reward_config.scales.get("feet_phase", 0.0) == 0.0:
      return jp.zeros(())
    foot_z = data.site_xpos[self._feet_site_id][..., -1]
    if self._hfield_ok:
      clearance = foot_z - self.terrain_height_under_feet(data)
    else:
      clearance = foot_z
    rz = self._get_rz(phase, self._config.reward_config.gait_swing_height)
    error = jp.sum(jp.square(clearance - rz))
    return jp.exp(-error / self._config.reward_config.gait_phase_sigma)

  def _cost_torque(self, data: mjx.Data) -> jax.Array:
    return jp.sum(jp.square(data.qfrc_actuator[-12:]))

  def _cost_action_rate(
      self, act: jax.Array, last_act: jax.Array
  ) -> jax.Array:
    return jp.sum(jp.square(last_act - act))

  def _cost_vertical_vel(self, data: mjx.Data) -> jax.Array:
    return jp.square(data.qvel[2])

  def _cost_xy_angular_vel(self, data: mjx.Data) -> jax.Array:
    return jp.sum(jp.square(data.qvel[3:5]))

  def _cost_joint_limit(self, data: mjx.Data) -> jax.Array:
    out_of_range = (
        jp.clip(self._soft_lowers - data.qpos[7:], min=0.0)
        + jp.clip(data.qpos[7:] - self._soft_uppers, min=0.0))
    return jp.sum(out_of_range)

  def _cost_joint_velocity(self, data: mjx.Data) -> jax.Array:
    return jp.sum(jp.square(data.qvel[6:]))

  def _cost_joint_acceleration(self, data: mjx.Data) -> jax.Array:
    return jp.sum(jp.square(data.qacc[6:]))

  def _cost_orientation(self, data: mjx.Data) -> jax.Array:
    # 参考 non_flat_base_cost: Σ projected_gravity[:2]² (机体系重力 xy)
    g = self.get_gravity(data)
    return jp.sum(jp.square(g[:2]))

  def _body_clearance(self, data: mjx.Data) -> jax.Array:
    """每个非足端碰撞体 (小腿/大腿/躯干) 相对地面的净空 (米), 形状 (G,)。

    净空 = 世界系最低点 z − 该点 xy 处的地形高度。负 = 已接触/穿透。

    为什么不直接用接触数组: warp 后端的 DataWarp **没有** contact 字段
    (实测 AttributeError: 'Data' object has no attribute 'contact'), 这正是
    参考仓库的 collision_cost 在本项目一直是空实现的原因 (§20.1)。纯几何
    复算不依赖后端, 且天然给出**连续**的净空 (比 0/1 接触计数更利于梯度)。

    形状公式 (geom 局部 z 轴在世界的方向 = geom_xmat 的第 3 列):
      球:       z_center − r
      胶囊/圆柱: min(两端点 z) − r
      盒:       z_center − Σ|R_z|·size
    """
    # geom_xmat 在 MJX 里已经是 (..., ngeom, 3, 3) (不是压平的 9 维), 直接索引
    # 即可; 不用 reshape —— vmap 后的批量 data 下 reshape 的目标形状容易写错
    # (实测踩过: reshape 出 (28,56) 之类, 广播报错)。
    gx = data.geom_xpos[..., self._bcf_gid, :]                # (...,G,3)
    R = data.geom_xmat[..., self._bcf_gid, :, :]              # (...,G,3,3)
    # 平地档没有 hfield 采样器 —— 地面恒为 z=0 (与 _cost_base_height 同处理)
    # 用 [..., :2] / [..., 2] 而非 [:, :2]: 对 vmap 后的批量 data (N,G,3) 也成立
    ground = (self.terrain_height(gx[..., :2]) if self._hfield_ok
              else jp.zeros(gx.shape[:-1]))                   # (...,G)
    axis_z = R[..., 2, 2]                                     # 局部 z 的世界 z 分量
    zc = gx[..., 2]
    is_sphere = self._bcf_type == mujoco.mjtGeom.mjGEOM_SPHERE
    is_box = self._bcf_type == mujoco.mjtGeom.mjGEOM_BOX
    cap_low = jp.minimum(zc + self._bcf_half * axis_z,
                         zc - self._bcf_half * axis_z) - self._bcf_r
    box_low = zc - jp.sum(jp.abs(R[..., 2, :]) * self._bcf_box, axis=-1)
    low = jp.where(is_sphere, zc - self._bcf_r, cap_low)
    low = jp.where(is_box, box_low, low)
    return low - ground

  def _cost_body_contact(self, data: mjx.Data) -> jax.Array:
    """非足端部件撞地惩罚 (v21 新增, 修 §20 的激励漏洞)。

    为什么需要 (§20): 训练里**没有任何一项**惩罚"小腿/大腿/躯干贴地":
      * `_cost_collision` 是空实现 (warp 无 contact 数组), 权重也是 0;
      * z 范围终止 (0.22) 只看躯干高度 —— 后腿膝盖撑地时躯干反而更高
        (实测 z_rel 中位 0.309 > 目标 0.28), 离终止线很远, 照样活着;
      * `base_height` 只管"躯干离足下地面多高", 对"小腿是否压在地上"完全
        不敏感 (它只会因为 0.309 略高于 0.28 target 而轻微扣分, 与接触无关)。
    结果: 策略把后腿膝盖当第五条腿用, RR/RL_calf 有 30% 的帧在接触地面,
    前腿只有 4-5%。表现为"全程后腿关节着地"。

    形式: 对每个陷入地面的碰撞体按**穿透深度**线性累计
        pen = min( Σ_l relu(margin − clearance_l), body_contact_max )
    margin=0 (实测定标) -> 只有在**真的穿透**时才非零, 标称站姿严格为 0。

    为什么用线性而不是平方 (§20.6 实测对比):
      * 区分度更好: 坏步态/好步态 = 4.07x (平方只有 2.22x);
      * 梯度不随穿透深度衰减 —— 浅穿透(刚碰到)也有恒定梯度把腿推出去,
        平方形式在浅穿透处梯度 ~0, 反而"陷进去才管", 容易卡在临界状态。
    代价是 0 处导数不连续, 但配合 clip 后对 PPO 无影响 (同 relu 类惩罚的惯例)。

    **上限 body_contact_max 是必需的** (§20.6 的教训): 本项目的总奖励在
    `step()` 里 `clip(total, 0, ...)`, 一旦负项压过正项, 总奖励变 0 ->
    PPO 梯度归零 (这是 v1-v7 失败模式的成因)。深穿透时若不设上限, 惩罚会
    轻松超过正奖励上限 (~4/步), 把信号打没。故对"累计穿透深度"设上限,
    使最大贡献可控 (见 train_go1.py 的打印)。
    """
    if self._config.reward_config.scales.get("body_contact", 0.0) == 0.0:
      return jp.zeros(())
    margin = self._config.reward_config.body_contact_margin
    clr = self._body_clearance(data)
    pen = jp.sum(jp.clip(margin - clr, min=0.0))
    return jp.minimum(pen, self._config.reward_config.body_contact_max)

  def _cost_collision(self, data: mjx.Data) -> jax.Array:
    # 参考仓库 collision_cost: 非足端体 cfrc_ext 范数 > 0.1 计 1 (8 个体)。
    # warp 后端的 Data 里没有 contact 数组 (cfrc_ext 也不可用), 无法复刻其
    # 语义 —— 真正生效的替代品是 v21 的 `_cost_body_contact` (纯几何净空,
    # 连续可微)。本函数保留为空实现以维持旧配置的逐位复现 (§20.1)。
    return jp.zeros(())

  def _cost_symmetry(self, info: dict[str, Any]) -> jax.Array:
    """左右镜像对称偏差 (v23, 时间平均量)。返回 >= 0, 权重为负。

    镜像映射 (四个 hip 关节轴同为 (1,0,0), 大腿/膝同为 (0,1,0)):
        hip:  hip_R = -hip_L      (轴保持, 旋转角变号)
        thigh/calf: q_R = q_L     (轴方向被镜像翻转 + 角变号 -> 净不变)
    所以镜像偏差 = (hip_F + hip_L)² + (hip_R + hip_L)²
                + (thigh_FR - thigh_FL)² + (thigh_RR - thigh_RL)²
                + (calf_FR - calf_FL)² + (calf_RR - calf_RL)²
    (前后腿之间不做比较: home 姿态本来就是前 thigh 0.8 / 后 1.0。)

    量级 (实测, 700 步 rollout 的 DC 估计): v22@40M ≈ 1.19, v18 (对称那代) ≈ 0.045。
    权重按这个标定 (见 sim/probe_sym_scale.py), 不是拍脑袋。
    """
    if self._config.reward_config.scales.get("symmetry", 0.0) == 0.0:
      return jp.zeros(())
    return jp.minimum(jp.sum(info["sym_ema"] ** 2),
                      self._config.reward_config.symmetry_cap)

  def _cost_default_joint_position(self, data: mjx.Data) -> jax.Array:
    return jp.sum(jp.square(
        data.qpos[7:] - jp.array(self._mj_model.key_ctrl[0])))