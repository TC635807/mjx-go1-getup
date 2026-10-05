#!/usr/bin/env python
"""Go1 起身 v2 —— **按公开可复现配方重写** (2026-09-20, 见 notes/getup_others.md)。

为什么另起一个 env 而不是改 Go1Getup: 旧的那份被 BC/DAgger/关键帧那条线用着,
而且 §29.18~§29.20 的全部实测结论都挂在它的行为上, 不能动。

抄的是 `iit-DLSLab/get-up-isaaclab` (四足 Go2, 已 sim-to-real) 的配置, 逐条对应:

  1. **动作锚在名义站姿上, 有界 + 低通** (对应 getup_env.py::_pre_physics_step):
        a   = clip(a, ±3.0)
        a_f = 0.8*a + 0.2*a_prev
        target = default_pose + 0.5 * a_f
     可达范围 = 名义站姿 ±1.5 rad。旧档是 `target = qpos + 0.5a` (增量/无界),
     策略得从零学关节角本身 —— 这是两边最大的结构差异。

  2. **奖励只有 2 个宽高斯** (对应 getup_env.py::_get_rewards):
        ori_err = 2*(1 - dot(u_body, n_terrain))     # 相对**局部地形法向**, 不是世界竖直
        track_orientation = exp(-ori_err / 0.1)
        h_err = (0.28 + terrain_h - root_z)^2
        track_height = exp(-h_err / 0.1)             # sigma = sqrt(0.1) = 0.316 m, 很宽
        门控 = dot(u_body, n_terrain) > cos(0.5 rad)
     世界竖直那个基准在坡度 >8.1 度的地方物理上不可达 (本项目 §29.18 实测),
     而四足站在坡上躯干本来就随坡倾 —— 用局部法向才是自洽的。

  3. **不做姿态/高度终止** (对应 _get_dones: died 恒 False), 只留 NaN 与出界护栏。

  4. **出生 = 站立高度直接落位** (对应 _reset_idx):
        关节 = default + U(-pi, pi) 再 clamp 到关节限位; 朝向均匀随机; 零速;
        z = terrain_h + 0.35;  xy ~ U(-2, 2)。
     不是"从 0.5 m 抛下 + 沉降 0.6 s"。副作用: reset 里**不需要跑物理沉降**, 便宜很多。

  5. **观测带 5 帧历史** (对应 use_observation_history/history_length):
        单帧 = [ang_vel_b(3), projected_gravity_b(3), qpos[7:]-default(12),
                qvel[6:](12), 上一步原始动作(12)] = 42;  obs = 42 x 5 = 210。
     **这打破了"与走路策略 91 维同形"的调度器合并前提** —— 用户已选方案 A:
     起身策略自带历史缓冲, 查看器里单独维护 (见 sim/view_go1.py 的 getup_v2 模式)。

  6. **PPO 侧**: entropy 0.005 / 5 epoch / lr 1e-3 (rsl_rl 的 adaptive KL 在 brax 里没有,
     用分段降 lr 近似) —— 见 train/train_getup.py --task getup_v2。

与开源的**有意差异**:
  * 不开 action noise / obs noise (先拿到干净的基线, 后面再加域随机化);
  * 没有 RMA / 高度图特权量, critic 只拿 obs + 少量特权 (本项目不追求 sim2real);
  * 高度扫描仍是本项目的 hfield 双线性采样器 (不是 ray caster)。
"""
import os
import sys

import jax
import jax.numpy as jp
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env

from envs.go1_walk import Go1Walk, default_config as walk_default_config

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
NU = 12


def default_config() -> config_dict.ConfigDict:
  cfg = walk_default_config()
  cfg.terrain = True
  cfg.height_scan.enable = True
  cfg.random_init.enable = False     # 出生由本 env 自己给 (站立高度 + 随机关节/朝向)
  cfg.push_config.enable = False
  cfg.obs_noise.enable = False
  # **不夹下界**: 允许负奖励, 否则早期总分被夹到 0 -> PPO 没有梯度
  cfg.reward_clip_min = -1.0e6
  # **成功终止 + 稀疏奖金**: v2 的失败模式很特别 ——
  # "末态近站立 30.5%, 但保持 25 步只有 1.6%" (到站太晚/站不稳)。
  # 公开配方没有这一项, 但 AFR (2412.16924) 就是"连续站立 N 步即成功终止";
  # 加这一项把"站住"从"每步 1.5 分的稠密信号"变成一个有明确终点的目标。
  # 8.0s @50Hz。300 步时策略"到达站姿"太晚 (末态近站立 30.5% 但"保持 25 步"只有 1.6%),
  # 留出时间学"站住"
  cfg.episode_length = 400
  cfg.getup2 = config_dict.create(
      history=5,                 # 观测历史帧数
      spawn_height=0.35,         # 相对地面的出生高度 (= home keyframe 的 z)
      spawn_xy_range=2.0,
      action_clip=3.0,           # desired_clip_actions
      action_scale=0.5,
      action_filter_alpha=0.8,   # use_filter_actions
      desired_base_height=0.28,  # 站立时躯干离地高度 (本项目实测稳定站姿 ~0.277)
      gauss_sigma2=0.1,          # 高斯的 "sigma^2", 与开源同值 -> sigma=0.316
      # 朝向奖励的基准: True = **世界竖直**(与判据/查看器一致); False = 局部地形法向(开源原版)。
      # 为什么必须可切: 开源那边没有"世界竖直"的评估判据, 所以他们用**局部地形法向**是对的;
      # 但本项目判据是 `up_world > 0.99` (倾角 < 8.1 度) + 查看器调度器同一套。
      # v2f 实测: 用地形法向训出来的策略, 末态 up_z 中位 0.963 (倾角 13.7 度)、92% 落在
      # 高度带内 —— **站是站起来了, 就是不水平**, 于是世界判据下成功率假 0。
      ori_world=True,
      ori_gate_deg=28.6,         # 门控: 躯干与局部法向夹角 < 该值才给高度/站姿分
      hip_y_offset=0.095,        # feet_to_hip 的目标 y 偏置 (足略微外张)
      out_of_bounds_xy=5.8,      # 纯护栏 (hfield 半径 6m)
      min_z_rel=-0.30,           # 纯护栏, 不做姿态终止
      # 出生关节难度 alpha: joints = (1-alpha)*default + alpha*U(限位)。
      # 1.0 = 公开配方原样 (全随机); 小值 = 课程 (HumanUP 的 Stage I "规范姿态+弱正则")。
      # 用它做多阶段课程: 每次运行一个固定值 + --init_pkl/--restore 续训。
      joint_alpha=1.0,
      hold_steps=25,             # 连续满足站姿判据多少步算成功
      success_bonus=600.0,       # 成功奖金 (约等于 400 步站立的稠密分, 保证"早上早拿"比"晚站"更划算)
  )
  # 奖励项名 -> 权重 (全部乘 step_dt; 只有 2 个宽高斯是正项)
  cfg.reward_config.scales = config_dict.create(
      track_orientation=1.0,
      track_height=1.0,
      feet_to_hip=1.5,
      action_rate=-0.001,
      action_smoothness=-0.001,
      joints_acc=-2.5e-7,
      joints_torques=-2.5e-6,
      joints_energy=-1e-4,
  )
  return cfg


class Go1GetupV2(Go1Walk):
  """公开配方版起身环境 (动作锚定 + 宽高斯 + 地形相对朝向 + 观测历史)。"""

  def __init__(self, config: config_dict.ConfigDict = None, **kwargs):
    super().__init__(config or default_config(), **kwargs)  # type: ignore[arg-type]
    g = self._config.getup2
    self._H = int(g.history)
    self._clip_a = float(g.action_clip)
    self._act_scale = float(g.action_scale)
    self._alpha = float(g.action_filter_alpha)
    self._des_bh = float(g.desired_base_height)
    self._sig2 = float(g.gauss_sigma2)
    self._gate_cos = float(np.cos(np.deg2rad(float(g.ori_gate_deg))))
    self._spawn_h = float(g.spawn_height)
    self._spawn_xy = float(g.spawn_xy_range)
    self._oob = float(g.out_of_bounds_xy)
    self._min_zr = float(g.min_z_rel)
    self._hold_steps = int(g.hold_steps)
    self._success_bonus = float(g.success_bonus)
    # 关节硬限位 (出生采样用)
    lo, hi = self.mj_model.jnt_range[1:].T
    self._lowers = jp.asarray(lo, jp.float32)
    self._uppers = jp.asarray(hi, jp.float32)
    self._ctrl_lo = jp.asarray(self.mj_model.actuator_ctrlrange[:, 0], jp.float32)
    self._ctrl_hi = jp.asarray(self.mj_model.actuator_ctrlrange[:, 1], jp.float32)
    # feet_to_hip 的目标 y 偏置; FEET 顺序 = ["FR","FL","RR","RL"] -> FR/RL 在 -y 侧
    self._hip_y_off = jp.asarray(
        [-float(g.hip_y_offset), float(g.hip_y_offset),
         -float(g.hip_y_offset), float(g.hip_y_offset)], jp.float32)
    self._torso_id = self._mj_model.body("trunk").id
    self._feet_site = np.array([self._mj_model.site(n).id
                                for n in ("FR", "FL", "RR", "RL")])
    # 髋体 (feet_to_hip 的参考点)。**不能拿躯干原点代替**: 髋在躯干系里 x=±0.1881 / y=±0.04675,
    # 用躯干原点会把"足落在髋正下方"错判成"足要收到躯干中线" (v2a~v2c 的实际错误)。
    self._hip_body = np.array([self._mj_model.body(n).id
                               for n in ("FR_hip", "FL_hip", "RR_hip", "RL_hip")])
    self._eps = jp.float32(0.05)

  # ---------------------------------------------------------------- 地形法向
  def _terrain_normal(self, xy: jax.Array) -> jax.Array:
    """局部地形单位法向 (世界系)。平地返回 [0,0,1]。"""
    if not self._hfield_ok:
      return jp.array([0.0, 0.0, 1.0])
    e = self._eps
    h = self.terrain_height(xy[None, :])[0]
    gx = (self.terrain_height((xy + jp.array([e, 0.0]))[None, :])[0] - h) / e
    gy = (self.terrain_height((xy + jp.array([0.0, e]))[None, :])[0] - h) / e
    n = jp.array([-gx, -gy, 1.0])
    return n / (jp.linalg.norm(n) + 1e-9)

  @staticmethod
  def _up_from_quat(q: jax.Array) -> jax.Array:
    """根四元数 (w,x,y,z) -> 躯干 up 轴在世界系。"""
    w, x, y, z = q[0], q[1], q[2], q[3]
    return jp.array([2.0 * (x * z + w * y),
                     2.0 * (y * z - w * x),
                     1.0 - 2.0 * (x * x + y * y)])

  def _z_ref(self, data: mjx.Data) -> jax.Array:
    return self.terrain_height(data.qpos[None, :2])[0] if self._hfield_ok else jp.zeros(())

  # ---------------------------------------------------------------- 终止
  def _is_healthy(self, data: mjx.Data) -> jax.Array:
    """**不做姿态/高度终止** (get-up-isaaclab 的 _get_dones: died 恒 False)。

    只留三条纯护栏: NaN、跑出 hfield、相对高度低于 -0.30 m (陷进地里)。
    走路的 17.5 度 tilt 判据在起身任务里会把"躺在地上"直接判死 (§9.2 教训)。
    """
    finite = jp.isfinite(data.qpos).all() & jp.isfinite(data.qvel).all()
    in_bounds = jp.all(jp.abs(data.qpos[:2]) < self._oob)
    z_rel = data.qpos[2] - self._z_ref(data)
    return finite & in_bounds & (z_rel > self._min_zr)

  # ---------------------------------------------------------------- reset
  def _spawn(self, rng: jax.Array):
    """站立高度直接落位 + 随机关节/朝向 (get-up-isaaclab::_reset_idx)。"""
    rng, kxy, kq, kj = jax.random.split(rng, 4)
    xy = jax.random.uniform(kxy, (2,), minval=-self._spawn_xy,
                            maxval=self._spawn_xy)
    h0 = self.terrain_height(xy[None, :])[0] if self._hfield_ok else jp.zeros(())
    quat = jax.random.normal(kq, (4,))
    quat = quat / (jp.linalg.norm(quat) + 1e-6)
    joints_r = jax.random.uniform(kj, (NU,), minval=self._lowers,
                                  maxval=self._uppers)
    a = jp.float32(self._config.getup2.joint_alpha)
    joints = (1.0 - a) * self._default_pose + a * joints_r
    qpos = jp.concatenate([xy, (h0 + self._spawn_h)[None], quat, joints])
    qvel = jp.zeros(self.mjx_model.nv)
    return qpos, qvel, joints

  def reset(self, rng: jax.Array) -> mjx_env.State:
    state = super().reset(rng)            # 拿到合法的 data/info/metrics
    qpos, qvel, joints = self._spawn(rng)
    data = state.data.replace(
        qpos=qpos, qvel=qvel, time=jp.zeros(()),   # 必须是 jnp 标量, Python float 在
        ctrl=jp.clip(joints, self._ctrl_lo, self._ctrl_hi))  # vmap 广播时没有 .shape
    data = mjx.forward(self.mjx_model, data)
    info = state.info
    info["obs_hist"] = None
    info["prev_a_filt"] = jp.zeros(NU)
    info["prev_a_filt2"] = jp.zeros(NU)
    info["last_raw_act"] = jp.zeros(NU)
    info["_restart"] = jp.zeros((), bool)
    obs = self._get_obs(data, info)
    # 训练日志里的真信号 (brax 的 EvalWrapper 会把 metrics 聚合成 episode_xxx)
    state.metrics["getup/up_z"] = jp.zeros(())
    state.metrics["getup/z_rel"] = jp.zeros(())
    state.metrics["getup/stand"] = jp.zeros(())
    state.metrics["getup/success"] = jp.zeros(())
    info["up_hold"] = jp.int32(0)
    info["allow_respawn"] = jp.int32(0)
    return state.replace(data=data, obs=obs,
                         reward=jp.zeros(()), done=jp.zeros(()))

  # ---------------------------------------------------------------- step
  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    info = state.info
    # brax 的 AutoResetWrapper 只换 data/obs, **不动 info** (training.py:138-158),
    # 所以历史缓冲/滤波器要在"本 episode 第一步"自己重置。判据用 data.time==0
    # (reset 里显式置 0, 每步 +0.02)。
    restart = state.data.time <= 0.0
    info["_restart"] = restart
    # ---- 每 episode 重新采出生姿态 ----
    # brax 的 AutoResetWrapper 只把 pipeline_state/obs 换回 first_*, 所以**不重采**的话
    # 整轮训练 768 个 env 只有一个冻结出生姿态 (这是 v2a/v2b 的隐藏缺陷)。
    # 这里在 restart 那一帧用 jp.where 无分支地换掉 qpos/qvel, 代价只有几次采样。
    respawn = restart & (info["allow_respawn"] > 0)
    info["allow_respawn"] = jp.where(restart, jp.int32(1), info["allow_respawn"])
    info["up_hold"] = jp.where(restart, jp.int32(0), info["up_hold"])
    rng, ks = jax.random.split(info["rng"])
    info["rng"] = rng
    qp_new, qv_new, _ = self._spawn(ks)
    d = state.data
    state = state.replace(data=d.replace(
        qpos=jp.where(respawn, qp_new, d.qpos),
        qvel=jp.where(respawn, qv_new, d.qvel),
        time=jp.where(respawn, jp.zeros(()), d.time)))
    info["prev_a_filt"] = jp.where(restart, jp.zeros(NU), info["prev_a_filt"])
    info["prev_a_filt2"] = jp.where(restart, jp.zeros(NU),
                                    info["prev_a_filt2"])
    a = jp.clip(action, -self._clip_a, self._clip_a)
    a_f = self._alpha * a + (1.0 - self._alpha) * info["prev_a_filt"]
    info["prev_a_filt2"] = info["prev_a_filt"]
    info["prev_a_filt"] = a_f
    info["last_raw_act"] = a
    target = self._default_pose + self._act_scale * a_f
    state = super().step(state, target)   # Go1Walk.step 就是"绝对 ctrl"语义
    # 站住判据 (与奖励同源: 地形相对朝向 + 目标高度带), 只作日志
    q = state.data.qpos
    n = self._terrain_normal(q[:2])
    u = self._up_from_quat(q[3:7])
    cos_t = (u[2] if self._config.getup2.ori_world else jp.dot(u, n))
    z_rel = q[2] - self._z_ref(state.data)
    state.metrics["getup/up_z"] = 1.0 - 2.0 * (q[4] ** 2 + q[5] ** 2)
    state.metrics["getup/z_rel"] = z_rel
    stand = ((cos_t > self._gate_cos)
             & (jp.abs(z_rel - self._des_bh) < 0.08))
    state.metrics["getup/stand"] = stand.astype(jp.float32)
    # 连续 hold_steps 步站住 -> 在**恰好达到那一步**给一次稀疏奖金。
    # **不终止 episode**: 我们有稠密奖励, 一旦"成功即终止", 策略会宁可晚点站起来
    # (早终止 = 丢掉剩下的稠密分)。改成一次性奖金后, "早上早拿 + 一直站住" 同时最优。
    hold = jp.where(stand, info["up_hold"] + 1, 0)
    info["up_hold"] = hold
    just = (hold == self._hold_steps).astype(state.reward.dtype)
    state = state.replace(reward=state.reward + just * self._success_bonus)
    state.metrics["getup/success"] = just
    return state

  # ---------------------------------------------------------------- 观测
  def _frame(self, data: mjx.Data, info: dict) -> jax.Array:
    """单帧 42 维 (get-up-isaaclab 的 actor obs 单帧, 顺序一致)。"""
    raw = info.get("last_raw_act")
    if raw is None:
      raw = jp.zeros(NU)
    return jp.concatenate([
        self.get_gyro(data),                       # 3  ang_vel_b
        self.get_gravity(data),                    # 3  projected_gravity_b
        data.qpos[7:] - self._default_pose,        # 12 joint_pos - default
        data.qvel[6:],                             # 12 joint_vel
        raw,                                       # 12 prev raw action
    ])

  def _get_obs(self, data: mjx.Data, info: dict):
    frame = self._frame(data, info)                # (42,)
    hist = info.get("obs_hist")
    if hist is None:
      hist = jp.broadcast_to(frame, (self._H, frame.shape[0]))
    else:
      hist = jp.concatenate([hist[1:], frame[None, :]], axis=0)
    restart = info.get("_restart")
    if restart is None:
      restart = jp.zeros((), bool)
    hist = jp.where(restart[..., None, None],
                    jp.broadcast_to(frame, (self._H, frame.shape[0])), hist)
    info["obs_hist"] = hist
    obs = hist.reshape(-1)
    # critic 特权量: obs + [相对高度, 地面高, 世界 up_z, 四足接触]
    z_rel = data.qpos[2] - self._z_ref(data)
    up_z = self._up_from_quat(data.qpos[3:7])[2]
    contact = info.get("last_contact")
    if contact is None:
      contact = jp.zeros(4, dtype=bool)
    extra = jp.concatenate([jp.reshape(z_rel, (1,)),
                            jp.reshape(self._z_ref(data), (1,)),
                            jp.reshape(up_z, (1,)),
                            contact.astype(jp.float32)])
    return {"state": obs, "privileged_state": jp.concatenate([obs, extra])}

  # ---------------------------------------------------------------- 奖励
  def _get_reward(self, data, action, info, done, first_contact, contact):
    """2 个宽高斯 + 1 个站姿项 + 5 个小正则 (get-up-isaaclab::_get_rewards)。"""
    q = data.qpos
    z_ref = self._z_ref(data)
    z_rel = q[2] - z_ref
    n = self._terrain_normal(q[:2])
    u = self._up_from_quat(q[3:7])
    if self._config.getup2.ori_world:
      cos_t = jp.clip(u[2], -1.0, 1.0)          # 与判据 up_z 同一个量
    else:
      cos_t = jp.clip(jp.dot(u, n), -1.0, 1.0)
    gate = (cos_t > self._gate_cos).astype(jp.float32)

    ori_err = 2.0 * (1.0 - cos_t)
    track_orientation = jp.exp(-ori_err / self._sig2)
    h_err = jp.square(self._des_bh + z_ref - q[2])
    track_height = jp.exp(-h_err / self._sig2) * gate

    # feet_to_hip: 足端相对躯干水平距离 (yaw 系), 目标 y 偏置 -> 鼓励足落在髋下
    # 开源的 ROT_W2H = yaw_quat(root): 去掉 roll/pitch, 只留 yaw。
    # 世界系直接减会在机器人朝向 +y 时把 y 偏置用反 -> 必须转进 yaw 系。
    w, x, y, z = q[3], q[4], q[5], q[6]
    yaw = jp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    cy, sy = jp.cos(yaw), jp.sin(yaw)
    foot = data.site_xpos[self._feet_site] - q[:3]
    hip = data.xpos[self._hip_body] - q[:3]
    def _yaw(v):
      return jp.stack([cy * v[:, 0] + sy * v[:, 1],
                       -sy * v[:, 0] + cy * v[:, 1]], axis=-1)
    fh, hh = _yaw(foot), _yaw(hip)
    dx = fh[:, 0] - hh[:, 0]
    dy = fh[:, 1] + self._hip_y_off - hh[:, 1]
    feet_to_hip = -jp.mean(jp.sqrt(jp.square(dx) + jp.square(dy))) * gate

    a = info["last_raw_act"]
    a_prev = info["prev_a_filt"]
    a_prev2 = info.get("prev_a_filt2")
    if a_prev2 is None:
      a_prev2 = jp.zeros(NU)
    torque = data.actuator_force
    reward = {
        "track_orientation": track_orientation,
        "track_height": track_height,
        "feet_to_hip": feet_to_hip,
        "action_rate": jp.sum(jp.square(a - a_prev)),
        "action_smoothness": jp.sum(jp.square(a - 2.0 * a_prev + a_prev2)),
        "joints_acc": jp.sum(jp.square(data.qacc[6:])),
        "joints_torques": jp.sum(jp.square(torque)),
        "joints_energy": jp.sum(jp.abs(torque * data.qvel[6:])),
    }
    return reward
