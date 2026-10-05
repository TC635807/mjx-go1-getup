# 起身（get-up）配方调研

两份调研笔记的合并：一份逐条对照公开的开源实现（get-up-isaaclab、HoST、HumanUP、AFR 等），
一份汇总各家的成功率、起点分布和课程设计。数字都来自仓库源码或论文摘要，附了出处。

调研的目的是先弄清公开工作怎么做，再对照本项目改。envs/go1_getup_v2.py 就是按
get-up-isaaclab 的配方重写的，起身成功率从 0% 做到 77~81%（见 status.md）。

---

## 一、开源实现配方对照（get-up-isaaclab / HoST / HumanUP / AFR）

# 别人怎么做"四足/人形起身" —— 开源实现与论文配方汇总

> 调研日期: 2026-09-20。目的: 停止在本项目里"盲试", 先把**公开可复现的配方**抄清楚,
> 再对照本项目的差异, 得出改动清单。所有数字都来自仓库源码 / 论文摘要, 附出处。

## 0. 一句话结论

**没有任何一个公开工作用"从零单阶段 PPO + 增量动作 + 窄判据奖励"做出起身。**
成功的配方有四个共同点:
1. **动作以"名义站立姿态"为锚**(`target = default_pose + scale*a`, a 有界), 不是加在当前关节角上无界漂;
2. **奖励只有 1~2 个宽高斯**(`exp(-err²/σ)`, σ 很大), 没有窄门/悬崖/稀疏奖金;
3. **几乎不做姿态终止**(只有 timeout), 但**都有课程**: 外力上拉/拖拽、地形难度、正则强度;
4. **训练量比本项目大 1~2 个数量级**(单任务 10⁸ 步量级) + 观测历史帧 + 非对称 actor-critic。

---

## 1. `iit-DLSLab/get-up-isaaclab` —— 最接近本项目的一个 (四足 Go2, sim-to-real)

出处: <https://github.com/iit-DLSLab/get-up-isaaclab>
关键文件: `tasks/get_up/go2_env_cfg.py`, `tasks/get_up/getup_env.py`, `tasks/get_up/agents/rsl_rl_ppo_cfg.py`

### 1.1 任务与终止

| 项 | 值 |
|---|---|
| episode | `6.0 s`, `decimation=4`, `sim dt=1/200` -> 50 Hz, **300 控制步** |
| `_get_dones` | `died` **恒为 False**, 只有 `time_out` —— **完全没有姿态/高度/能量终止** |
| 终止 | 无 |

### 1.2 出生 (reset)

```python
joint_pos = default_joint_pos + U(-pi, pi)         # 全域随机, 再 clamp 到关节限位
root_xy  += U(-2.0, 2.0)                            # 平面 ±2 m
root_z    = default_root_state[:, 2]                # **不抛**, 就是默认站立高度
root_quat = random_orientation(...)                 # 朝向全随机 (env 0..500 保持竖直, 用于评估/课程)
root_vel  = 0
self.episode_length_buf[:] = randint(0, max_len)    # 打散 reset 时刻
```

**注意: 不是"从 0.5 m 抛下"** —— 就是在站立高度把姿态/关节打乱, 让物理自己倒。

### 1.3 动作 (与本项目差异最大的一条)

```python
a = clip(a, -3.0, 3.0)                               # desired_clip_actions = 3.0
a_f = 0.8 * a + 0.2 * a_prev                         # use_filter_actions, alpha = 0.8
target = action_scale(0.5) * a_f + default_joint_pos # **绝对, 以名义站姿为锚**
```

-> 可达目标 = 名义站姿 ± 1.5 rad。策略只需要学"偏离站姿多少", 不是"关节角本身"。

### 1.4 观测

`obs_1 = [ang_vel_b(3), projected_gravity_b(3), (joint_pos - default)(12), joint_vel(12), prev_action(12)] = 42`

**`use_observation_history=True, history_length=5` -> actor obs = 210**。critic = actor + 特权(PD 增益, 足接触, 高度误差, terrain pitch, 4x4 高度图)。

### 1.5 奖励 (只有 2 个宽高斯 + 1 个姿态项 + 4 个小惩罚)

```python
# 姿态误差用 **ray scanner 估的局部地形朝向** 做基准, 不是世界竖直!
terrain_pitch = -atan2(mean_h_front - mean_h_back, ds)
ori_err = (terrain_pitch - root_pitch)^2 + (terrain_roll - root_roll)^2
track_orientation = exp(-ori_err / 0.1)              # sigma^2 = 0.1 -> sigma ~ 0.316 rad(很宽)

# 高度误差同样是"局部地形"基准, 目标 0.30 m
h_err = (0.30 + mean_height_ray - root_z)^2
track_height = exp(-h_err / 0.1)

# 门控: 只有躯干与局部地形朝向差 < 0.5 rad(~28.6 度) 时才给高度/站姿分
gate = (|terrain_pitch - root_pitch| < 0.5) * (|terrain_roll - root_roll| < 0.5)

feet_to_hip_distance = -mean(sqrt(dx^2 + dy^2)) * 1.5  # 让足落在髋正下方 (yaw 系)
# 惩罚: action_rate -0.001, action_smoothness -0.001,
#       joints_acc -2.5e-7, joints_torques -2.5e-6, joints_energy -1e-4
```

**没有 posture / stand_still / dof_pos_limits / 稀疏成功奖金**。全部乘以 `step_dt`。

### 1.6 PPO (rsl_rl)

`lr 1e-3` + **`schedule="adaptive"`, `desired_kl=0.01`**; `5` epochs, `4` minibatches,
**`entropy_coef=0.005`**, `gamma 0.99`, `lam 0.95`, `clip 0.2`, `max_grad_norm 1.0`, `init_std 1.0`;
网络 `[128,128,128]` **ELU**; `num_steps_per_env=24`。

### 1.7 训练规模

`num_envs=4096`。flat: `max_iterations=1000` -> 1000 x 24 x 4096 = **~9.8e7 步**;
rough: `max_iterations=8000` -> **~7.9e8 步**。

---

## 2. `InternRobotics/HoST` —— 人形, RSS 2025 (2502.08378)

出处: <https://github.com/InternRobotics/HoST>, 配置 `envs/g1/g1_config_ground.py`, `envs/base/host_ground.py`

| 项 | 值 |
|---|---|
| 出生 | `pos=[0,0,0.5]`, `rot=[0,-1,0,1]`(**上下颠倒**), 然后 **`unactuated_timesteps=30`**(前 30 个 sim step 不驱动) |
| episode | `10 s` -> 500 控制步 (decimation 4, dt 0.005) |
| 动作 | `action_scale=1`, 语义 **`target = a + cur_dof_pos`(增量)**, `init_noise_std=0.8` |
| 观测 | **`num_actor_history = 6`**, 单帧 76 -> 456 |
| 奖励结构 | **多 critic: 4 个组 `[task, regu, style, target]`, 权重 `[2.5, 0.1, 1, 1]`** |
| task 组 | 只有 2 项: `task_orientation`(Gaussian, σ=1) + `task_head_height` |
| post-task 组 | 权重 **10**(不是 1): `target_ang_vel_xy`, `target_lin_vel_xy`, `target_target_orientation`, `target_target_base_height` —— 站起来之后要"站住不动" |
| **课程** | **`pull_force = True, force = 100` (实际 200 N)** —— 给躯干一个**向上拉力**, 到 `threshold_height=0.9` 后退火 |
| 其他 | `only_positive_rewards=False`; 域随机化: 摩擦 0.1~1、restitution 0~1、kp/kd x0.85~1.15、**延迟 up to 5 步**、初始关节 x0.9~1.1 + offset ±0.1 |

**要点**: HoST 自己的论文明说"摔倒 -> 跪姿"这一段**靠随机动作噪声探索不出来**, 所以他们加向上拉力 + 多 critic + 地形课程。

---

## 3. `RunpeiDong/HumanUP` —— 人形, RSS 2025 (2502.12152)

出处: <https://github.com/RunpeiDong/HumanUP>, 配置 `envs/g1waist/g1waist_up_config.py`, `envs/base/humanoid_config.py`

**两阶段课程 (论文摘要原文)**: Stage 1 "discovering a good getting-up trajectory under **minimal constraints on smoothness or speed/torque limits**";
Stage 2 "refines the discovered motions into **deployable (smooth and slow)** motions that are robust to variations in initial configuration and terrains"。

| 项 | 值 |
|---|---|
| 出生 | 站立高度 `pos=[0,0,0.8]`; Stage 2 用**姿态池**(论文: 2 万个 "0.5 m 掉落 + 沉降 10 s" 的姿态) |
| 终止 | `terminate_after_contacts_on = ["torso_link"]  # NOTE: now there is no termination after contacts`; `termination = -500`(终止事件罚款); `termination_height = 0.0` |
| **课程 1** | **`drag_robot_by_force = True`, `drag_force = 1500 N`, 拖 `head`, 正弦退火**, `drag_interval=50` |
| 课程 2 | `regularization_scale` 0.8~2.0 按 step_height 升(先弱正则发现动作, 后强正则变平滑) |
| 课程 3 | `standing_scale` 按 cos 升 |
| 主要正奖励 | `base_height_exp=5`, `head_height_exp=5`, `stand_on_feet=2.5`, `feet_height=2.5`; 目标高度 0.728 m |
| 正则 | `dof_error -0.03`, `action_rate -0.1`, `dof_vel -1e-4`, `dof_pos_limits -5`, `torques -1e-6` |
| 模仿项 | `target_joint_pos_scale = 0.17`(Stage 2 用参考动作做慢速模仿) |
| PPO | `action_scale=0.5`, `decimation=20`, `dt=0.001`, `init_noise_std=1.0` |

---

## 4. 其它可参考的成功率数字

| 工作 | 平台 | 报告的成功率 | 出处 |
|---|---|---|---|
| **Learning to Recover** (2506.05516) | 轮腿四足 x2 | **99.1% / 97.8%**(无平台专属调参) | Episode-based Dynamic Reward Shaping + 课程 + **非对称 AC** + 观测注噪 |
| AFR (2412.16924) | Go1 -> Spot/ANYmal | 论文给了"成功率与恢复速度均优于基线"(具体 % 未抽到) | Isaac Gym; **350 步上限, 连续站立 100 步即成功终止** |
| HoST (2502.08378) | G1 人形 | sim-to-real, 具体 % 未抽到 | 见 §2 |
| HumanUP (2502.12152) | G1 人形 | 真机 face-up / face-down, 平地+可变形+滑+坡 | 见 §3 |

额外一条: **"Efficient and Robust Self-Recovery of Quadruped Robots Using Asymmetric PPO"**(ITU) —— 四足自恢复 + 非对称 PPO, 见 <https://research.itu.edu.tr/en/publications/efficient-and-robust-self-recovery-of-quadruped-robots-using-asym/>

---

## 5. 对照: 本项目 vs `get-up-isaaclab` (同为四足, 差异最可比)

| 项 | 本项目 getup | get-up-isaaclab | 差距性质 |
|---|---|---|---|
| **动作** | `target = qpos + 0.5*a`(**增量, 无界**) | `target = default_pose + 0.5*filt(a)`, **a clip ±3**, 低通 0.8 | **结构性**: 我这边没有"名义站姿"锚, 关节可以漂到任意位置 |
| **观测** | 91 维**单帧** | 42 x **5 帧历史** = 210 | **结构性**: 策略看不到关节速度历史/相位 |
| **出生** | 相对地面 **+0.5 m 抛下** + 沉降 0.6 s | **站立高度直接落位**, 不抛 | 我这边更难; 且"抛下"制造了额外的不确定性 |
| **终止** | NaN / 出界 / z_rel<-0.35 | **完全没有** | 我的终止会在"还在挣扎但有希望"时把 episode 掐死 |
| **奖励** | 9 项 + 窄尖峰(σ=0.05) + 稀疏奖金 | **2 个宽高斯(σ≈0.316) + 地形相对朝向 + 0.5 rad 门控** | 我的奖励窄、有台阶、可套利(已实测两个套利点) |
| **朝向基准** | 世界竖直 | **局部地形朝向**(ray scanner) | 我已独立发现同一问题(§29.18) —— 他们用高度扫描解决 |
| **PPO** | brax, lr 3e-4 固定, entropy 0, 10 epoch | rsl_rl, lr 1e-3 **adaptive (KL 0.01)**, entropy **0.005**, 5 epoch, 4 minibatch | 我这边**没有** KL 自适应, 也没有熵奖励 |
| **规模** | 768 env, **6e6 步** | 4096 env, **9.8e7 (flat) / 7.9e8 (rough) 步** | **差 16~130 倍** |

---

## 6. 建议的改动清单 (按 收益/成本 排序, **尚未执行**)

| # | 改动 | 文件 | 依据 | 成本 |
|---|---|---|---|---|
| 1 | **动作换成"以名义站姿为锚 + 有界 + 低通"**: `target = default_pose + 0.5*clip(a,±3)`, `a_f = 0.8a+0.2a_prev` | `envs/go1_getup.py` 的 `step` | get-up-isaaclab §1.3; 这是唯一"不改奖励就能立刻测"的一条 | 10 行 |
| 2 | **奖励换成 2 个宽高斯**: `exp(-h_err²/0.1) + exp(-ori_err/0.1)`, `ori_err` 相对**局部地形法向**, 门控 `|Δ| < 0.5 rad` | `envs/go1_getup.py` 的 `_get_reward` | §1.5; 同时解决我已发现的"世界竖直在斜坡上不可达" | 40 行 |
| 3 | **出生改成"站立高度直接落位 + 关节 default+U(-π,π) 夹限位"**, 不再抛 0.5 m | `envs/go1_getup.py` 的 `_reset_fallen` | §1.2 | 15 行 |
| 4 | **去掉姿态/高度终止**, 只留 NaN + timeout | `envs/go1_getup.py` 的 `_is_healthy` | §1.1 | 5 行 |
| 5 | **观测历史 x4~5 帧** —— **与"91 维调度器合并"直接冲突**, 需取舍 | `envs/go1_getup.py` + `sim/view_go1.py` | §1.4 | 中 |
| 6 | **PPO 加 KL 自适应 + 熵奖励** (brax 没有 adaptive KL, 只能手动分段降 lr; 或改用 rsl_rl 风格) | `train/train_getup.py` | §1.6 | 中 |
| 7 | **训练量提到 2~5e7 步, env 提到 2048** | 命令行 | §1.7 | 4 小时 |
| 8 | **(可选) 外力课程**: 给躯干加向上拉力并按高度退火 | `envs/go1_getup.py` | HoST §2 / HumanUP §3; 专治"摔倒->跪姿"探索不出来 | 中 |

### 6.1 关于本项目已有的"关键帧 + 残差"路线

§29.20 的实测结论是: **关键帧先验是混沌的**(有效 ±0.09 rad 扰动就让成功率 29.3% -> 10.9%), 所以**任何**在它上面加噪声的 RL 都只会退化。
公开工作里**没有**人用"开环关键帧 + 残差"做起身 —— 他们都是**从零用 RL 学整个动作**, 靠的是上面那 8 条(动作锚定 + 宽奖励 + 课程 + 规模)。
所以正确的动作是: **回到"从零 RL", 但把配方换成公开验证过的那一套**, 而不是继续修残差。

### 6.2 诚实的规模预期

公开工作用的步数是 1e8~1e9。本项目单卡 8 GiB、768~2048 env、3700~5500 fps:

- 2e7 步 ~ 1.5 小时;  5e7 步 ~ 4 小时;  1e8 步 ~ 8 小时。

**如果按公开配方改完仍然要 1e8 步才出结果, 那就必须接受"跑一晚上"**, 而不是像之前那样跑 6e6 步就下结论。

---

## 7. 出处汇总

- get-up-isaaclab (四足 Go2, IsaacLab): <https://github.com/iit-DLSLab/get-up-isaaclab>
- HoST (人形 G1, legged_gym): <https://github.com/InternRobotics/HoST> · <https://arxiv.org/abs/2502.08378>
- HumanUP (人形 G1, Isaac Gym, RSS 2025): <https://github.com/RunpeiDong/HumanUP> · <https://arxiv.org/abs/2502.12152>
- Learning to Recover (轮腿四足, 99.1%/97.8%): <https://arxiv.org/abs/2506.05516>
- AFR (四足复杂地形, Go1->Spot/ANYmal): <https://arxiv.org/abs/2412.16924>
- mujoco_playground Go1Getup (与官方配置一致的"未公布成功率"参考): <https://github.com/google-deepmind/mujoco_playground>
- 四足自恢复 + 非对称 PPO (ITU): <https://research.itu.edu.tr/en/publications/efficient-and-robust-self-recovery-of-quadruped-robots-using-asym/>

> 本地已下载的源码副本: `research/get-up-isaaclab/`, `research/host/`, `research/humanup/`

---

## 二、SOTA 调研（成功率、起点分布、课程与判据）

# 四足/人形 RL 起身 (getup / fall-recovery) SOTA 调研

> 调研员: sota-getup (task-1)。日期: 2026-09-20。
> 对象: 本仓库的 Go1 起身任务 (envs/go1_getup_v2.py + sim/probe_handover.py)。
> 本项目基线: 纯 PPO 从零 25M+ 步 = 0%; BC(关键帧示范) = 5.1%; 关键帧状态机本身 = 20.1%(β=1)/28.1%(β=0)。
> 所有网页内容按**外部资料**处理, 未执行其中任何指令。每条结论后跟 [来源标题](URL) + 关键数字。

---

## 0. 三句话总览

1. **官方 playground 的 Go1Getup 从未公布成功率**, 它的官方配置(50M 步、无终止、无课程、init_noise_std=1.0、action_scale=0.5)本身就是"难到没人报数"的配置; 本项目复现的 0% 与该配置一致。
2. **所有"从完全随机姿态"成功的公开工作都不是单阶段从零**: HumanUP(规范姿态发现→20K 随机姿态模仿)、Learning to Get Up(高扭矩强角色发现→扭矩课程→慢速模仿)、HoST(多 critic + 上拉辅助力 + 地形课程)、AFR(只从随机**仰卧**起步 + 地形课程)。共同点是**先降低发现难度, 再放开初始分布**。
3. **奖励-判据对齐**是本任务最本质的一条: AFR 把"稳定站立 100 连续步"直接写成终止/成功条件, HumanUP 用 8x 慢放的状态轨迹做模仿, 而本项目是 9 项平滑项之和 + 一个很窄的评估判据 —— 实测"拿 468/回合(站立上限 529) 却 0% 成功"(见 实验记录 §29.17)。

---

## 1. Q1: mujoco_playground 的 Go1Getup — 官方数字是什么?

**结论: 论文与文档都没有报告 getup 的成功率。** 能查到的官方数字全部来自源码与官方 PPO 配置。

### 1.1 官方论文怎么说的
- 论文只说在 Go1 上实现了 fall recovery 并做了 sim-to-real: "On the Unitree Go1, we additionally implement fall recovery and handstand environments." / "First, on the Unitree Go1, we deploy joystick, fall recovery, and handstand policies." / "For fall recovery, we follow [30, 58], enabling the robot to return to a stable 'home' posture from arbitrary fallen configurations."
- 全文的 success-rate 数字属于 **Franka 重定向 (sim2real 成功率)** 与**抓取任务 (100% in 12 trials)**, **与 getup 无关**。
  [MuJoCo Playground (ar5iv 2502.08844)](https://ar5iv.labs.arxiv.org/html/2502.08844)

### 1.2 官方 env 源码的全部关键数字 ([getup.py 源码](https://raw.githubusercontent.com/google-deepmind/mujoco_playground/main/mujoco_playground/_src/locomotion/go1/getup.py))
| 量 | 官方值 |
|---|---|
| ctrl_dt / sim_dt | 0.02s / 0.004s (50Hz 控制) |
| episode_length | **300** (6s) |
| action_scale | **0.5**, 增量语义 target = qpos[7:] + action*0.5 |
| drop_from_height_prob | **0.6** (40% 从 home 站立出生) |
| 出生 | qpos[2]=**0.5m**; 四元数 = 归一化 N(0,I) 采样(**任意朝向**); 12 关节在硬限位内均匀采样; 根速度 qvel[0:6] 属于 [-0.5, 0.5]; 自由沉降 settle_time=0.5s |
| 终止 | energy_termination_threshold = **np.inf** → **永不因姿态/高度终止** (代码只有 NaN/能量一条) |
| 奖励权重 | orientation 1.0, torso_height 1.0, posture 1.0, stand_still 1.0, action_rate -0.001, dof_pos_limits -0.1, torques -1e-5, dof_acc -2.5e-7, dof_vel -0.1 |
| 门控 | posture / stand_still 乘 gate = is_upright(ori_tol=0.01) 且 is_at_desired_height(z_des=0.275, pos_tol=0.005) |

官方 docstring 自己写明了"防躺平"动机: "Torso height: The torso should be at a desired height. **This is to prevent the robot from flipping over and just lying on the ground.**" —— 也就是说官方知道这个退化吸引子, 但**没有**给成功率的证据。官方文档还明确写了动作语义选择的实验结论: "We tried using the same action space used in the joystick task ... but it didn't work as well as adding to the current joint configuration."

### 1.3 官方 PPO 配置 ([locomotion_params.py](https://raw.githubusercontent.com/google-deepmind/mujoco_playground/main/mujoco_playground/config/locomotion_params.py))
- Go1Getup 专属: **num_timesteps = 50,000,000**, num_evals = 5, policy (512,256,128), value (512,256,128), **policy_obs_key="state", value_obs_key="privileged_state"** (asymmetric actor-critic)。
- 其余沿用全局默认: normalize_observations=True, unroll_length=20, num_minibatches=32, num_updates_per_batch=4, **discounting=0.97**, learning_rate=3e-4, entropy_cost=1e-2, num_envs=8192, batch_size=256, max_grad_norm=1.0。
- **没有覆盖 init_noise_std** → brax 默认 **1.0**(见 Q5)。
- 数字含义: 官方认为 getup 需要 **50M 步**(和 joystick 的 200M、backflip 的 200M 同档), 是整套 locomotion 里最贵的任务之一。

---

## 2. Q2: 公开 getup / fall-recovery 的成功率与随机化范围

**结论: 没有任何公开工作报告"从完全随机朝向 + 随机关节 + 高掉落"的单阶段从零成功率; 凡是做到"任意姿态"的, 都是两阶段或带课程的。** 下表按"起点分布"分类。

| 工作 | 起点分布 | 随机化范围 | 终止 / 成功判据 | 报告的成功率 |
|---|---|---|---|---|
| **mujoco_playground Go1Getup** | 完全随机 (任意四元数 + 关节全域均匀 + 0.5m 掉落) | 无 DR; 60% 掉落 / 40% 站立出生 | 只有 NaN (energy=inf) | **未报告** |
| **AFR** ([2412.16924](https://arxiv.org/abs/2412.16924)) | **随机仰卧** (random supine), 不是任意朝向 | payload [-2.5,2.5]kg; Kp/Kd/motor strength x[0.9,1.1]; COM shift ±50mm; trunk mass [4.0,28.0]kg; 地形: 坡度 0-45 度, 离散障碍高 0.10-0.30m, 石块 0.15-0.40m / 缝 0.10-0.30m | **350 步上限; 或保持稳定站立 100 连续步即终止** | 每地形 **50 次试验**, Table III 给成功率 + 恢复时间 (优于 PPO baseline); 具体百分比未能从 HTML 抽取(见 §6 备注) |
| **HoST** ([2502.08378](https://arxiv.org/abs/2502.08378), [代码](https://github.com/InternRobotics/HoST)) | Unitree G1, **多种姿态**(仰卧/俯卧/坐/靠树/坡地), 多种地形 | Table II: trunk mass U(-2,5)kg; COM offset ±0.03m; link mass xU(0.8,1.2); friction U(0.1,1); restitution U(0,1); P/D gain xU(0.85,1.15); torque RFI ±0.05 倍 limit; motor strength U(0.9,1.1); **control delay U(0,100)ms**; 初始关节偏置 U(-0.1,0.1)rad、scale U(0.9,1.1) | 多 critic + 上拉辅助力 + terrain 课程 | 论文实验段给仿真/真机曲线 (HTML 被截断未能取到百分比, 见 §6); 论文/项目页以"diverse postures 上平滑稳定"为主 |
| **HumanUP** ([2502.12152](https://arxiv.org/abs/2502.12152)) | Stage I: **规范姿态**(并混入站立姿态); Stage II: **任意初始姿态** | Stage II 用 **20K 仰卧姿态数据集** (从 **0.5m 掉落 + 仿真 10s** 让自碰撞收敛), 10K 训练 / 10K 评估; 完整碰撞网格; 强正则 | Stage II 模仿 **8 倍慢放的 Stage I 状态轨迹** | 论文 V/VI 节有 supine 起身、prone 到 supine 翻滚两个 task; 表格数值未能抽取 |
| **Learning to Get Up** ([Xie et al., SIGGRAPH 2022](https://ar5iv.labs.arxiv.org/html/2205.00307)) | rag-doll 从 **1.5m 掉落 + 随机姿态**, 掉落期动作 a ~ N(0, 0.1), 固定 **80 控制步** | 落地后进入 lying pose | 首阶段 "每个 episode 到 250 步结束, **没有任何早停判据**"; 第三阶段"偏离参考运动就早停" | 报告的是**运动多样性与质量** (weak-and-slow), 不是单一成功率 |
| **Playground manipulation 对照** ([open_cabinet.py](https://raw.githubusercontent.com/google-deepmind/mujoco_playground/main/mujoco_playground/_src/manipulation/franka_emika_panda/open_cabinet.py)) | — | action_scale = **0.04** (getup 是 0.5) | — | (仅用于 Q5 的噪声对照) |

**要点 (对本项目直接有用)**
1. **"完全随机"必须靠课程/两阶段**: HumanUP 明确写 "Stage I learns to get up (and roll over) from a canonical pose, accelerating learning, while Stage II starts from arbitrary initial poses, enhancing generalization. To further speed up Stage I, we mix in standing poses."
2. **单阶段能做到的是"随机仰卧"档** (AFR), 且它用了 4096 并行 env + 地形课程 + 100 连续步成功终止。
3. **"任意姿态"的落地成本被 HumanUP 明码标价**: 20K 个"掉落 0.5m + 沉降 10s"的预生成姿态池, 一半训练一半评估 —— 这正是本项目 §29.14 建议 3(预生成沉降姿态池) 的公开实现。

---

## 3. Q3: 防"翻正即趴平"退化吸引子的已知配方

**结论: 公开工作里没有靠"再加步数"解决的; 有效配方是 4 类 —— (a) 成功终止/稀疏成功信号、(b) 探索辅助(强角色高扭矩 / 上拉外力 / 弱正则发现阶段)、(c) 课程(姿态/地形/扭矩)、(d) 奖励分组/多 critic。动作噪声 std 与 asymmetric AC 是次要旋钮。**

### (a) 成功终止 + 稀疏成功信号 (直接对标本项目的"468/529 但 0%")
- **AFR**: "Episodes begin with a random supine position and **terminate after 350 timesteps or upon maintaining a stable standing posture for 100 consecutive timesteps**." ([2412.16924](https://arxiv.org/abs/2412.16924))
- **HoST**: 奖励分 4 组 —— task / style / regu / **post (post-task reward: 描述"站起来成功之后应该保持站立"的行为)**, 用 multi-critic 各自优化。([2502.08378](https://arxiv.org/abs/2502.08378))
- **Learning to Get Up**: 早停只用在**模仿阶段** ("early termination to the episode if the current state diverges too much from the reference motion"), 发现阶段**不早停**。这条对本项目很关键: 本项目现在"只挡 NaN"是发现阶段的正确做法, 但**缺"成功终止 + 成功奖金"**, 导致策略可以在奖励里 farm 到 468 而不触发判据。

### (b) 探索辅助 —— 公开工作一致指出"随机动作噪声发现不了起身"
- **Learning to Get Up**: 核心是先训 **强角色(高扭矩上限)** 以"探索更大的状态-动作空间", 因为 "characters can become trapped in local minima, which results in **variants of a kneeling motion**" —— 与本项目的"翻正后趴/跪"是同一个局部极小。三步: (i) 强角色发现模式; (ii) **降低扭矩上限的课程**适配弱角色; (iii) 学"更慢速度"的模仿策略(ε-RSI + 偏离参考就早停)。([2205.00307](https://ar5iv.labs.arxiv.org/html/2205.00307))
- **HoST**: "The primary exploration challenges emerge during the transition from falling to stable kneeling, **a stage that proves difficult to explore effectively through random action noise alone.**" 对策是**上拉力课程**: "we apply an upward force F on the robot base ... This force takes effect only when the robot's trunk achieves a near-vertical orientation ... The force magnitude decreases progressively"。代码值: curriculum.pull_force=True, **force=100 (注释: 因额外 keyframe torso link 实为 200)**, threshold_height=0.9, dof_vel_limit=300, base_vel_limit=20; 另有 **action rescaler** 逐步收紧动作幅度。([HoST g1_config_ground.py](https://raw.githubusercontent.com/InternRobotics/HoST/main/legged_gym/legged_gym/envs/g1/g1_config_ground.py))
- **HumanUP**: Stage I 用**简化碰撞网格 + 极弱正则**去发现动作 ("very weak regularization"), Stage II 才上强正则(smoothness / DoF 速度惩罚)。([2502.12152](https://arxiv.org/abs/2502.12152))

### (c) 课程 (公开值)
| 工作 | 课程轴 | 数值 |
|---|---|---|
| HumanUP | 姿态 | Stage I 规范姿态 → Stage II 20K 随机姿态(0.5m 掉落 + 10s 沉降) |
| HoST | 地形 + 上拉力 | terrain curriculum=True, max_init_terrain_level=**5**, num_cols=20; pull force 100→0; action rescaler 逐步收紧 |
| AFR | 地形难度 | 坡度 0→45 度, 障碍高 0→0.30m, 石块/缝逐步加大 |
| Learning to Get Up | 角色强度(扭矩上限) + 速度 | 3 阶段 (strong → weak → slow) |

### (d) 奖励分组 / 权重
- HoST: task/style/regu/post 四组 + multi-critic("multi-critic RL to optimize distinct reward groups independently for a better reward balance"); algorithm 侧 value_smoothness_coef=0.1, smoothness 界 [0.1,1.0]; orientation_threshold=0.99; target_base_height 分三档 0.45/0.45/0.65。
- 本项目已有的方向一致(§29.13/§29.15): 硬门 → 斜坡、orientation 与 torso_height 权重一起提。

### (e) 次要旋钮 (有公开数字, 但不是本问题的解)
- **init_noise_std**: HoST = **0.8** (rsl_rl 基类 1.0); 见 Q5。
- **asymmetric actor-critic**: playground 官方 Go1Getup 就是 value_obs_key="privileged_state"; 本项目也在用 → **不是解药**。([locomotion_params.py](https://raw.githubusercontent.com/google-deepmind/mujoco_playground/main/mujoco_playground/config/locomotion_params.py))

---

## 4. Q4: DAgger / 交互式模仿把 BC 提升到专家水平

**结论: DAgger 是当前"补 BC 分布漂移"的标准做法, 有多个公开的腿足机器人实例; 关键超参是 聚合策略(每轮把新轨迹并入全集 D)、轮数(通常 3-10)、在 student 自己访问到的状态上标注。**

| 工作 | DAgger 用法 | 公开超参 / 细节 |
|---|---|---|
| [Learning Perceptive Humanoid Locomotion (2503.00692)](https://ar5iv.labs.arxiv.org/html/2503.00692) | "We employ Dataset Aggregation (DAgger) for behavior cloning through **iterative data relabeling**" | 每轮 k: 用 student 在并行仿真里 rollout, 记录 **teacher 的动作**; 损失是 MSE 到 teacher 动作; **D = 全部历史轨迹的并集 (i=1..k)**; 总损失 L_student = L_imitation + **λ=0.5** · L_ELBO |
| [Robot Parkour Learning (2309.05665)](https://ar5iv.labs.arxiv.org/html/2309.05665) | 用 DAgger 把 5 个技能策略蒸馏成单个视觉策略 | "we use DAgger [44,45] to distill them into a single vision-based parkour policy"; 只在仿真里 query 特权教师 |
| [Saving the Limping (2210.00474)](https://ar5iv.labs.arxiv.org/html/2210.00474) | teacher-student 在线聚合 (DAgger 思路) | "previous works focus on imitating mu's behaviors only by using supervised learning **inspired by DAgger**"; 在 student 轨迹上最小化 latent 标签的 MSE |
| [HumanUP (2502.12152)](https://arxiv.org/abs/2502.12152) | 两阶段模仿(非纯 DAgger, 但机制相似) | Stage II 直接**模仿 Stage I 状态轨迹的 8 倍慢放版本**, 再上强正则 |

**对本项目的落地要点**
1. 关键帧状态机是**反馈式**的(不是一条固定轨迹), 所以它在 student 任意访问到的状态上都能给出动作 —— 这正是 DAgger 需要的 expert 形式(本项目 sim/probe_getup_keyframe.py 里已有"在当前 state 上反解动作"的机制可用)。
2. 只把**成功 episode** 收进示范会重新引入分布漂移(本项目 BC v1 就是用"只留成功"的 20.9 万条, 结果 5.1% vs 专家 20.1%, 见 实验记录 §29.16/§29.17); DAgger 的正解是**在 student 真实访问到的状态分布上标注**(含它开始失败的偏离状态)。
3. 每轮把新数据并入全集 D(不要只留最新一轮), 这是 DAgger 的 mistake-bound 保证所在。

---

## 5. Q5: brax PPO 的 init_noise_std / std_param 语义与推荐值

**结论: init_noise_std 是"可学习的动作标准差参数 std_param 的初值", 只影响早期探索; 官方默认 1.0。公开的起身工作(HoST)用 0.8; 没有找到"精细任务用远小于 1.0"的公开证据 —— 更常见的等价做法是调小 action_scale。有效探索噪声约等于 init_noise_std 乘 action_scale。本项目 getup 是 1.0 x 0.5 = 0.5 rad/步, 是 MuJoCo Playground 全套任务里最大的一档。**

### 5.1 语义 (源码原文)
- [brax/training/networks.py](https://raw.githubusercontent.com/google/brax/main/brax/training/networks.py):
  - noise_std_type='scalar' (默认) → std_module = Param(**init_noise_std**, size=param_size, name='std_param') → std_params 直接取 init_noise_std;
  - noise_std_type='log' → LogParam(init_value) → std = exp(logparam);
  - state_dependent_std=True 时 std 由网络输出, 与 init_noise_std 无关。
- [brax/training/agents/ppo/networks.py](https://raw.githubusercontent.com/google/brax/main/brax/training/agents/ppo/networks.py): make_ppo_networks(..., **init_noise_std: float = 1.0**, distribution_type 默认 'tanh_normal')。
- 本项目用的是 distribution_type='normal' → **均值不过 tanh**, 动作可以"瞬间甩到行程端点"(这正是关键帧 kip 需要的), 采样 a = mean + std 乘 ε。

### 5.2 公开取值
| 来源 | init_noise_std | action_scale | 有效每步噪声 |
|---|---|---|---|
| brax PPO 默认 | 1.0 | — | — |
| rsl_rl / legged_gym 默认 ([HoST base cfg](https://raw.githubusercontent.com/InternRobotics/HoST/main/legged_gym/legged_gym/envs/base/legged_robot_config.py)) | 1.0 | — | — |
| **HoST (人形起身)** | **0.8** ([g1_config_ground.py](https://raw.githubusercontent.com/InternRobotics/HoST/main/legged_gym/legged_gym/envs/g1/g1_config_ground.py)) | 1, decimation 4 (dt 0.02s) | 0.8 |
| playground Go1Getup | 1.0 (未覆盖) | **0.5** | **0.5 rad/步 约 28.6 度/步** |
| playground Franka open_cabinet (精细操作) | 1.0 (未覆盖) | **0.04** | 0.04 |

→ 精细操作任务"看起来噪声小"不是把 std 调到 0.04, 而是 action_scale=0.04; **两者乘积才是探索幅度**。

### 5.3 实践含义 (与本项目 §29.16/§29.17 一致)
- std_param 是**可学习**的, 训练中会自己收缩, 所以初值主要决定**早期探索的幅度**。从零训练 vs BC 微调应该用不同初值。
- brax 的 restore_params **不恢复优化器状态**(本项目已记 §29.16), Adam 会从零重建 → 微调应用小 lr, 否则第一次更新就把策略推歪。

---

## 6. 备注: 取证边界 (诚实声明)

- AFR 的 Table III 成功率百分比与 HoST 的仿真/真机成功率百分比: 论文 HTML 被 web_fetch 截断(100k 字符), 表格数值单元格未能可靠抽取; 为避免编造, 本文只写了可核实的量(试验次数 50/地形、350 步、100 连续步终止、4096 envs、DR 范围)。需要精确百分比时请直接看 PDF: [AFR PDF](https://arxiv.org/pdf/2412.16924v1) Table III, [HoST PDF](https://arxiv.org/pdf/2502.08378) §V/§VI。
- 未找到任何公开工作报告"playground Go1Getup + 随机姿态"的成功率数字(官方论文、README、搜索均无)。本项目的 0% 是当前已知的、与该配置一致的公开经验。

---

## 7. 本项目的 5 条具体改动建议 (按 预期收益/成本 排序)

> 前提事实(来自 实验记录 §29.13 至 §29.17): 纯 PPO 从零 25M+ = 0%; BC 单独 = 5.1%(β=1)/7.4%(β=0); 关键帧 = 20.1%/28.1%; BC+PPO 三种微调账目全部退化到 0%; 一次运行拿到 468/回合(站立上限 529)却仍 0% 成功。

### 建议 1 (最高收益/成本比): 用关键帧专家做 3 轮 DAgger, 直接补 BC 的分布漂移
- 新文件: 建议新增 sim/dagger_getup.py; 复用 sim/collect_getup_demos.py 的专家与 [train/bc_getup.py](../train/bc_getup.py) 的训练循环。
- 做法: 第 k 轮用当前 BC 策略(而非专家)在 Go1Getup 里 rollout(建议 1024 env x 300 步, drop_prob=1.0), 对**每一个访问到的状态**用关键帧专家反解动作打标签, 把该轮轨迹并入全集 D; 用 D 重新训 BC(policy MSE, 沿用 [bc_getup.py](../train/bc_getup.py) 的 mode()+MSE 设计, 别用 NLL)。
- 具体值: 3 轮; 每轮训练到验证 MSE 接近专家示范的 0.0526; 保留全部历史数据(D 为各轮并集); init_noise_std 用 0.3(见建议 5)。
- 依据: [2503.00692](https://ar5iv.labs.arxiv.org/html/2503.00692) 的"历史全集并集 + λ=0.5 多任务损失"; [2309.05665](https://ar5iv.labs.arxiv.org/html/2309.05665) 的 DAgger 蒸馏; 以及本项目 §29.17 的结论 4。
- 验收: [sim/eval_getup.py](../sim/eval_getup.py) 的**实用成功率**从 5.1% 往上走(目标 >=15%); 若 3 轮后不涨, 说明专家在偏离状态上不可用, 转建议 4。

### 建议 2 (高收益/低成本): 稀疏成功奖金 + "成功终止", 让奖励与判据对齐
- 文件: [envs/go1_getup.py](../envs/go1_getup.py) 的 _get_reward (L247-288) 与 _is_healthy (L140-151); 在 info 里加计数器。
- 值: 判据用本项目已有的成功定义(up_z > 0.99 且 z_rel 属于 [0.24, 0.32] 连续 **25** 步) → 触发时给 **+60** 稀疏终端奖金(约为站立 300 步总奖励的 1/9, 足够大但不压过稠密项)并置 done=1(成功终止); 另加 **no-progress 失败早停**: 连续 **100** 步 z_rel 低于 0.20 且关节速度低于 2.0 rad/s 则 done(不给奖金)。
- 依据: AFR "保持稳定站立 100 连续步即终止" + 350 步上限 ([2412.16924](https://arxiv.org/abs/2412.16924)); HoST 的 post-task reward ([2502.08378](https://arxiv.org/abs/2502.08378)); 注意**失败早停不能**用"姿态不对"判据(本项目 §29.2 已经踩过, 起身全程在健康区外)。
- 验收: 训练日志里出现"成功终止"计数; eval_getup.py 严格成功率 > 0。

### 建议 3 (极高性价比): 修两个已实测坏掉的奖励项
- 文件: [envs/go1_getup.py](../envs/go1_getup.py) 的 _get_reward。
- (i) **stand_still**: 现在收到的是 Go1Walk.step 传来的**绝对 ctrl 目标**(0.8~1.5 rad), exp(-0.5 乘 ||a||^2) 恒约等于 0(实测站立时 +0.0003, 上限 0.5)。改成用**归一化策略动作**: 在 step() 里把原始 action 存进 info(如 info["last_policy_act"]), 奖励用 gate 乘 exp(-0.5 乘 ||last_policy_act||^2)。
- (ii) **orientation**: 实测站立时只有 0.887/3.0(get_gravity() 即便竖直也偏 up_vec 约 46 度, §29.17), 于是"顶 orientation"比"站起来"更划算(可 farm 到 +2.1/步)。改成用**根四元数显式算 torso up 向量**与 world up 的夹角误差, 让真竖直时饱和到 1.0 x 权重 3.0。
- 验收: [sim/probe_getup_reward_terms.py](../sim/probe_getup_reward_terms.py) 复测: 站立档 orientation 应约 3.0、stand_still 应约 0.4~0.5, 且"趴平"档合计明显低于"站直"(现在 0.717 vs 1.764)。

### 建议 4 (高收益/中高成本): 两阶段"发现 → 可部署", 并用预生成姿态池把 reset 变廉价
- 文件: [train/train_getup.py](../train/train_getup.py)(已有 --difficulty / --orient_rand / --init_pkl / --restore) 与 [envs/go1_getup.py](../envs/go1_getup.py); 另需新增姿态池脚本。
- Stage A(发现, 弱正则): --difficulty 0 --orient_rand 0, 把 dof_vel(-0.1)/dof_acc(-2.5e-7)/action_rate(-0.001) 权重 **乘 0.1**, 10M 步, 直到 eval 出现非零成功率。依据: HumanUP Stage I "simplified collision mesh + very weak regularization", Learning to Get Up 首阶段高扭矩/无早停。
- Stage B(可部署): 预生成 **20K 个"从 0.5m 掉落 + 沉降 10s"** 的已沉降姿态池(npz), reset 改成 gather(不跑物理沉降; 项目已实测 settle=0.6s 时 reset 947ms/次、settle=0 时 6ms); 放开 --difficulty 1 --orient_rand 1; 用 Stage A 的轨迹做 **8 倍慢放**的模仿项 + 恢复强正则。
- 依据: HumanUP 20K 姿态(10K 训练/10K 评估)、Stage II 模仿 8 倍慢放 ([2502.12152](https://arxiv.org/abs/2502.12152)); 本项目 §29.14 建议 3。
- 注意: 本项目实测 num_resets_per_eval>0 会让 warp mempool 崩(§29.16), 姿态池 + gather 正好绕开它。

### 建议 5 (低成本, 已验证必要但不充分): 微调的信任域与"有效探索噪声"
- 文件: [train/train_getup.py](../train/train_getup.py#L124-L129) 与 [train/bc_getup.py](../train/bc_getup.py#L45-L47)。
- 值: init_noise_std **0.3**(上限参照 HoST 的 0.8); **action_scale 0.5 → 0.25**(仅微调阶段, 把有效噪声从 0.15 降到 0.075); num_minibatches **384 → 32**、updates_per_batch **10 → 2**、lr **1e-5**、entropy 0、加 desired_kl 约 0.01; 第一段只更新策略、冻结 value(BC 的 value R^2 约 0.36, 优势估计不可靠)。
- 依据: 有效噪声 = init_noise_std 乘 action_scale(playground 操作任务 1.0 x 0.04 对照 [open_cabinet.py](https://raw.githubusercontent.com/google-deepmind/mujoco_playground/main/mujoco_playground/_src/manipulation/franka_emika_panda/open_cabinet.py)); HoST 0.8 ([g1_config_ground.py](https://raw.githubusercontent.com/InternRobotics/HoST/main/legged_gym/legged_gym/envs/g1/g1_config_ground.py)); 本项目 §29.17 三次微调全退化。
- 诚实预期: 本条**单用已被本项目否证三次**(std 1.0/0.15/0.3 都退化), 所以它是建议 1/2/4 的配套, 不是主攻方向; 它的价值是把"探索噪声"变量固定住, 让建议 1/2 的效果可归因。

### 一句话优先级
**建议 3(改两行奖励) → 建议 1(DAgger 3 轮) → 建议 2(成功奖金+终止) → 建议 4(两阶段+姿态池) → 建议 5(微调参数)**;
如果只能做一件事: 做**建议 1**, 因为它是唯一被公开工作反复验证"能把模仿策略提到专家水平"的机制, 而本项目已经有可用专家(20.1%)和可用示范管线。
