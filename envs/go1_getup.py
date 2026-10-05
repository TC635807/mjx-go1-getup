#!/usr/bin/env python
"""Go1 起身任务 (地形版): 复用 Go1Walk 的模型/地形/物理/obs, 只换 reset 分布与奖励。

为什么不直接用 `mujoco_playground` 的 Go1Getup (见 实验记录 §29.9):
  1. 它的 XML 依赖外部 `mujoco_menagerie` 仓库, 本机没有 -> 连模型都建不起来;
  2. 它的物理与本项目不是同一个 (playground 把 Kp=35/Kd=0.5 写进 gainprm/biasprm,
     本项目 `kp=20/kv=1` + 关节 damping=1.0), 且用的是 fullcollisions 模型;
  3. 它的地面是 **flat**, 而我们要的恰恰是 hfield 地形。
所以这里只**抄它的奖励设计**, 模型/地形/obs 全部沿用本项目。

与 Go1Walk 的差异 (只有三处):

  1. **reset**: 以 `drop_prob` 的概率撒"摔倒姿态" —— 随机 xy (同一套随机出生
     分布) + `terrain_height(xy) + drop_height` 的**相对**高度 + 随机四元数 +
     关节范围内均匀采样 + 随机根速度, 然后自由沉降 `settle_time` 秒;
     否则沿用走路的站立出生。
     (`getup.py:140` 用的是固定高度 0.5m —— 平地语义; 地形上会出生即嵌进土里。)

  2. **奖励**: 换成 playground getup 的 9 项。**高度项一律改成相对足下地形**:
     `torso_height = qpos[2] - terrain_height(qpos[:2])`。
     playground 用的是 `site_xpos[imu][2]` 与固定 `_z_des=0.275` (绝对值),
     在 0.30m 高台上会把"站在台顶"判成"不在目标高度", gate 永远 0 -> 学不会。
     这是本项目已经踩过两次的同一类坑 (§16.6.4 的 feet_height / §29.2 的摔倒判据)。

  3. **终止**: 只保留 NaN 护栏。走路那份 `_is_healthy` 在 |roll|,|pitch|>17.5 度
     或相对高度不在 (0.22,0.65) 时终止 —— 起身任务**全程都在这个区域外**, 不关掉就是
     "episode 秒死、奖励恒 0" (§9.2 记过同款教训)。

obs 与走路**完全同形** (91 / 162): 沿用 `_get_obs`, 并保留 `feet_phase` 尺度
(只为让相位仍拼进 obs)。这样查看器里两个策略吃同一个 obs 向量, 合并只需切换 act 的来源;
将来要做 §29.10 的 task-conditioned 单策略时也只要再加一个 one-hot 维度。

已知与 playground 的两处**有意**偏差:
  * `action_rate` 只保留一阶项 (本项目 step 不维护 `last_last_act`); 二阶项是
    轻微平滑项, 影响可忽略。
  * 指令不做特殊处理: 训练时照走路档随机采样, 查看器里就是用户的实时指令 —— 起身策略
    会忽略它, 但训练/推理的输入分布一致 (省掉查看器一个特例)。
"""
import jax
import jax.numpy as jp
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env

from envs.go1_walk import Go1Walk, default_config as walk_default_config

# playground getup 的 9 项权重 (原样抄)
GETUP_SCALES = {
    "orientation": 1.0,
    "torso_height": 1.0,
    "posture": 1.0,
    "stand_still": 1.0,
    "action_rate": -0.001,
    "dof_pos_limits": -0.1,
    "torques": -1e-5,
    "dof_acc": -2.5e-7,
    "dof_vel": -0.1,
}


def default_config() -> config_dict.ConfigDict:
  cfg = walk_default_config()
  # 地形档: 与 v19+ 走路策略同一套物理/观测 (hfield + 35 点扫描 + 随机出生)
  cfg.terrain = True
  cfg.height_scan.enable = True
  cfg.random_init.enable = True
  sc = dict(GETUP_SCALES)
  # feet_phase = 0 会让 _get_obs 不拼相位 -> obs 变 83 维, 与走路策略不同形。
  # 这里给个非 0 权重只为把相位留在 obs 里 (奖励里并不返回这一项, 所以权重无实际作用)。
  sc["feet_phase"] = 2.0
  cfg.reward_config.scales = config_dict.create(**sc)
  # 起身是单任务: 6s (300 控制步) 足够; settle 之后还有 ~5.4s 学动作。
  cfg.episode_length = 300
  cfg.getup = config_dict.create(
      drop_prob=0.75,       # 摔倒姿态占比 (其余是站立出生, 学"别乱动")
      drop_height=0.5,      # 相对地面的抛下高度 (playground 同值)
      settle_time=0.6,      # 自由沉降时长 (秒); 0.5m 自由落体 0.32s, 留余量
      z_des=0.26,           # 目标躯干**相对**高度 (站直约 0.277)
      z_tol=0.01,           # 高度判据容差
      # 动作语义: **增量**。playground getup.py:200 的动作就是
      #   motor_targets = qpos[7:] + action*0.5
      # 它的 docstring 明说"加成绝对 home 姿态不如加在当前关节角上"。
      # 本项目走路档是绝对 ctrl; 但起身**必须**用增量 —— 实测 1M 步 smoke 用绝对
      # 语义时, 随机初始策略输出 ±1 级的目标直接把狗甩飞 (末态相对高度 -2m, 已经
      # 出了 hfield), 成功率 0%。增量式让早期探索是"小幅调整当前关节", 温和得多。
      action_scale=0.5,
      # 飞出去/陷下去 -> 终止。没有这条时被甩出地形 (|xy|>6 之外 hfield 无地面)
      # 的 episode 会白跑: 狗在虚空里自由落体, 采样全浪费。
      out_of_bounds_xy=6.2,   # hfield 半径 6m
      min_z_rel=-0.35,        # 相对地面高度下限 (正常躺地约 0.05, 站立 0.277)
      ori_tol=0.05,         # "竖直"判据: |up - gravity|^2 阈值 (地形有坡度, 比 playground 0.01 宽)
      soft_limit=0.95,      # 软关节限位系数
      # §29.15 课程: 出生难度 α ∈ [0,1]。只插值**关节** (朝向始终随机):
      #   α=0 -> 朝向随机 + 关节=名义站姿 (翻过去就站着, 易)
      #   α=1 -> 关节也全随机 (真任务)
      # 它是**每次运行固定的** (多阶段课程靠多次运行 + --restore 推进) —— 因为它是
      # trace 期常量, jax.jit 缓存命中时改它不会重编 (probe_reset_cost.py 实测),
      # 所以"运行中改难度"是无效的。
      difficulty=1.0,
      # §29.15 第二根课程轴: 朝向随机程度 β ∈ [0,1]。
      #   β=0 -> 出生时**身体竖直** (只需要把腿撑起来 -> 直击 v24/v25 唯一没学会的技能)
      #   β=1 -> 朝向全随机 (还要先翻正, 这一项 v24 已经会了)
      # 用**伯努利二选一**而不是四元数插值: 插值出来的"半倒"姿态一落地就被重力塌回地面,
      # 实测 α=0.5 与 α=1 的沉降后分布几乎重合 (probe_getup_curriculum.py)。
      orient_rand=1.0,
  )
  return cfg


class Go1Getup(Go1Walk):
  """从任意摔倒姿态站起来的策略 (地形版)。"""

  def __init__(self, config: config_dict.ConfigDict = None, **kwargs):
    super().__init__(config or default_config(), **kwargs)  # type: ignore[arg-type]

    cfg = self._config.getup
    # 关节硬限位 / 软限位 (照 playground: 软 = 中心 +/- 0.5*range*factor)
    lo, hi = self.mj_model.jnt_range[1:].T   # [0] 是 free joint, 跳过
    self._lowers = jp.asarray(lo, jp.float32)
    self._uppers = jp.asarray(hi, jp.float32)
    c = 0.5 * (lo + hi)
    r = hi - lo
    self._soft_lowers = jp.asarray(c - 0.5 * r * cfg.soft_limit, jp.float32)
    self._soft_uppers = jp.asarray(c + 0.5 * r * cfg.soft_limit, jp.float32)
    self._up_vec = jp.array([0.0, 0.0, -1.0])
    self._settle_substeps = max(
        1, int(round(cfg.settle_time / self.mjx_model.opt.timestep)))
    self._ctrl_lo = jp.asarray(self.mj_model.actuator_ctrlrange[:, 0], jp.float32)
    self._ctrl_hi = jp.asarray(self.mj_model.actuator_ctrlrange[:, 1], jp.float32)

  # ---------------------------------------------------------------- 地形基准
  def _z_ref(self, data: mjx.Data) -> jax.Array:
    """躯干正下方的地面高度 (平地 = 0)。"""
    if self._hfield_ok:
      return self.terrain_height(data.qpos[None, :2])[0]
    return jp.zeros(())

  # ---------------------------------------------------------------- 终止
  def _is_healthy(self, data: mjx.Data) -> jax.Array:
    """起身任务**不做姿态/高度终止**, 只挡 NaN。

    走路那份判据 (|roll|,|pitch|>17.5 度 或 相对高度不在 (0.22,0.65)) 会把
    "躺在地上"直接判死 —— 而那正是起身任务的起点。漏掉这条 = episode 秒死,
    奖励恒 0, PPO 没有梯度 (§9.2 的同款教训)。
    """
    finite = jp.isfinite(data.qpos).all() & jp.isfinite(data.qvel).all()
    gu = self._config.getup
    z_rel = data.qpos[2] - self._z_ref(data)
    in_bounds = jp.all(jp.abs(data.qpos[:2]) < gu.out_of_bounds_xy)
    return finite & in_bounds & (z_rel > gu.min_z_rel)

  # ---------------------------------------------------------------- reset
  def _random_fallen(self, rng: jax.Array):
    """随机摔倒姿态: 随机地形点 + 相对地面 drop_height + 随机朝向 + 随机关节 + 随机根速度。"""
    cfg = self._config.getup
    ri = self._config.random_init
    rng, kxy, kq, kj, kv = jax.random.split(rng, 5)
    xy = jax.random.uniform(
        kxy, (2,), minval=-ri.xy_range, maxval=ri.xy_range)
    h0 = self.terrain_height(xy[None, :])[0] if self._hfield_ok else jp.zeros(())
    quat = jax.random.normal(kq, (4,))
    quat = quat / (jp.linalg.norm(quat) + 1e-6)
    joints = jax.random.uniform(
        kj, (self.mjx_model.nu,), minval=self._lowers, maxval=self._uppers)
    qpos = jp.concatenate([jp.stack([xy[0], xy[1], h0 + cfg.drop_height]),
                           quat, joints])
    qvel = jp.zeros(self.mjx_model.nv).at[0:6].set(
        jax.random.uniform(kv, (6,), minval=-0.5, maxval=0.5))
    return qpos, qvel

  def reset(self, rng: jax.Array) -> mjx_env.State:
    # 先按走路档 reset: 站立出生 + 完整 info / metrics (指令采样、相位、推挤倒计时...)
    state = super().reset(rng)
    rng, kdrop, kfall, korient = jax.random.split(rng, 4)

    qpos_fall, qvel_fall = self._random_fallen(kfall)

    # §29.15 课程: difficulty α **只插值关节**, 朝向始终随机。
    # 为什么只插关节 (v26 的第一版教训, sim/probe_getup_curriculum.py 实测):
    # 一开始把**整个 qpos**(含朝向) 向站立姿态插值, 但沉降后 α=0.5 与 α=1 的出生分布
    # 几乎一样 (up_z 中位 +0.057 vs -0.100; 相对高度 0.142 vs 0.141) —— "半站半倒"
    # 的姿态一落地就被重力塌回地面。课程轴选错了。
    # 真正缺的技能是"翻正之后把腿撑起来": v24 末态 up_z 中位 0.997 却只有 0.128m,
    # v25 撑到 0.173m 但丢了倾角。所以课程轴 = **腿部构型**:
    #   α=0 -> 朝向随机 + 关节=名义站姿 -> 翻过去就是站立 (易)
    #   α=1 -> 关节也全随机 -> 翻过去还得把腿重新摆到站姿 (真任务)
    # 朝向恢复本身已经学会 (v24), 保持它随机即可, 不参与插值。
    alpha = jp.float32(self._config.getup.difficulty)
    beta = jp.float32(self._config.getup.orient_rand)
    q_stand = state.data.qpos
    # 轴 1 — 腿部构型: 线性插值 (蹲得多深是连续的难度)
    joints = (1.0 - alpha) * q_stand[7:] + alpha * qpos_fall[7:]
    # 轴 2 — 朝向: **二选一** (伯努利), 因为它不是连续难度而是"要不要先翻正"
    use_rand_quat = jax.random.bernoulli(korient, beta)
    quat = jp.where(use_rand_quat, qpos_fall[3:7], q_stand[3:7])
    # 出生高度: **两个档分别算** (踩过的坑: 竖直档若也用 h0+0.05, 躯干会被埋进地里
    # 0.23m, 沉降后变成"翻正但塌着的一坨"而不是站立 —— 出生分布会整档错掉)。
    xy = qpos_fall[:2]
    h0 = (self.terrain_height(xy[None, :])[0] if self._hfield_ok
          else jp.zeros(()))
    h0_stand = (self.terrain_height(q_stand[:2][None, :])[0]
                if self._hfield_ok else jp.zeros(()))
    stand_h = q_stand[2] - h0_stand           # 名义站立时躯干离地高度 (~0.277)
    z_rand = h0 + self._config.getup.drop_height
    z_up = h0 + stand_h + jp.float32(0.02)    # 竖直档: 直接放在站立高度上
    z = jp.where(use_rand_quat, z_rand, z_up)
    qpos_fall = jp.concatenate([xy, z[None], quat, joints])

    do_drop = jax.random.bernoulli(kdrop, self._config.getup.drop_prob)
    qpos = jp.where(do_drop, qpos_fall, state.data.qpos)
    qvel = jp.where(do_drop, qvel_fall, state.data.qvel)

    # 自由沉降: ctrl 先保持关节角 (与 playground 同), 让它在重力下自然落地
    ctrl = jp.clip(qpos[7:], self._ctrl_lo, self._ctrl_hi)
    data = state.data.replace(qpos=qpos, qvel=qvel, ctrl=ctrl)
    data = mjx_env.step(self.mjx_model, data, ctrl, self._settle_substeps)
    data = data.replace(time=0.0)

    info = state.info
    info["h_spawn"] = self._spawn_ground(data)   # 保持与 data 一致 (getup 奖励不用它)
    info["last_act"] = jp.zeros(self.mjx_model.nu)
    info["feet_air_time"] = jp.zeros(4)
    info["last_contact"] = jp.zeros(4, dtype=bool)
    info["swing_peak"] = jp.zeros(4)
    info["swing_clear_sum"] = jp.zeros(4)
    info["swing_steps"] = jp.zeros(4)
    info["sym_ema"] = jp.zeros(6)

    obs = self._get_obs(data, info)
    return mjx_env.State(data, obs, jp.zeros(()), jp.zeros(()),
                         state.metrics, info)

  # ---------------------------------------------------------------- step
  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    """增量动作 -> 绝对 ctrl 目标, 再交给 Go1Walk.step。

    envs/go1_walk.py 的动作语义是"策略输出即 ctrl 目标"(绝对)。起身档改成
    playground 的增量式: target = 当前关节角 + action*action_scale。
    查看器里要**做同样的变换** (--getup_action incremental)。
    """
    target = (state.data.qpos[7:]
              + action * self._config.getup.action_scale)
    return super().step(state, jp.clip(target, self._ctrl_lo, self._ctrl_hi))

  # ---------------------------------------------------------------- 奖励
  def _get_reward(self, data, action, info, done, first_contact, contact):
    """playground getup 的 9 项; 高度全部相对足下地形。"""
    cfg = self._config.getup
    up = self.get_gravity(data)                  # 机体系重力方向 (与 playground 同源)
    ori_err = jp.sum(jp.square(self._up_vec - up))
    is_upright = ori_err < cfg.ori_tol

    z_rel = data.qpos[2] - self._z_ref(data)
    h = jp.minimum(z_rel, jp.float32(cfg.z_des))
    # §29.13: 硬门 -> 线性斜坡。
    # 原版 is_at_h = (z_des - h) < z_tol 等价于硬门 h > 0.25; 门内 posture+stand_still
    # 合计 ~2.0, 门外 = 0 -> 在 h=0.25 处一个 **2.0 的奖励悬崖, 悬崖以下没有任何梯度**。
    # 10.8M 步实测: 成功率 0, 末态高度卡在 0.128m (= 悬崖下), 且随步数递减增长
    # (0.084@1.1M -> 0.128@10.8M) -> 就是被这个悬崖挡住 (§29.13)。
    # 改成斜坡后全程单调: h 越高分数越高, 不给"趴平不动"留套利 (斜坡上沿才给分)。
    # §29.15: 下沿归零 -> 纯斜坡 ramp = clip(h/z_des, 0, 1), 不再有任何平台/悬崖。
    # 依据: §29.13 把硬门(0.25)换成斜坡[0.20,0.26] 后, v25 末态高度确实 0.128->0.173m
    # (+35%), 但**正好停在新下沿 0.20 之下** (gate_h = 0) —— 悬崖只是被搬了个位置。
    # 现在 h 从 0 到 z_des 全程线性淡入。
    ramp = jp.clip(h / jp.float32(cfg.z_des), 0.0, 1.0)
    is_at_h = (jp.float32(cfg.z_des) - h) < cfg.z_tol      # 保留作诊断口径
    gate = is_upright * ramp

    joints = data.qpos[7:]
    torques = data.actuator_force
    return {
        "orientation": jp.exp(-2.0 * ori_err),
        "torso_height": jp.exp(h) - 1.0,
        "posture": gate * jp.exp(
            -0.5 * jp.sum(jp.square(joints - self._default_pose))),
        "stand_still": gate * jp.exp(-0.5 * jp.sum(jp.square(action))),
        "action_rate": jp.sum(jp.square(action - info["last_act"])),
        "torques": (jp.sqrt(jp.sum(jp.square(torques)))
                    + jp.sum(jp.abs(torques))),
        "dof_pos_limits": (
            jp.sum(-jp.clip(joints - self._soft_lowers, None, 0.0))
            + jp.sum(jp.clip(joints - self._soft_uppers, 0.0, None))),
        "dof_acc": jp.sum(jp.square(data.qacc[6:])),
        "dof_vel": jp.sum(jp.square(jp.maximum(
            jp.abs(data.qvel[6:]) - 2.0 * jp.pi, 0.0))),
    }

