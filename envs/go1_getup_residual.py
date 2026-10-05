#!/usr/bin/env python
"""Go1 起身 —— **关键帧先验 + 学习残差** (v30 路线)。

为什么走这条路 (来自 §29.18 的诊断链, 见 实验记录_Go1复现.md):
  * 纯 PPO 从零 25M+ 步 = 0%: 收敛到"翻正 + 趴平"(末态 up_z 0.997 / z_rel 0.128m)。
    公开工作 (HumanUP 2502.12152 / Learning to Get Up 2205.00307 / HoST 2502.08378)
    也都**不是单阶段从零**; HoST 原文说"摔倒->跪姿"这段靠随机动作噪声无法有效探索。
  * 关键帧状态机在**同一个动作接口/同一套出生分布**下能到 18.9%~28.1% —— 任务可行,
    缺的不是奖励而是"看着脚下构型逐帧改力矩"这种**闭环精细修正**; 状态机是开环的
    (hfield 上腿会被格子卡住, 单方案约 1/3)。
  * 所以把状态机的输出当**先验动作**, 让 RL 只学残差:
        ctrl = clip(ctrl_fsm + a * residual_scale)
    策略要学的东西从"从零发现起身机动"降级成"在已知机动上做小幅修正", 探索难度天差地别。
  * obs 仍与走路**同形** (91/162) -> 调度器"换 act 来源"的前提不变。

实现要点 (三个坑):
  1. brax 的 AutoResetWrapper **不重置 env 的 info** (brax/envs/wrappers/training.py:138-158
     只替换 pipeline_state 和 obs), 所以 FSM 相位在 episode 自动重置后不会回到 P_WAIT。
     这里用 `data.time == 0` 当"本 episode 第一步"的判据, 在那一步把 FSM/计数器全部重置
     (Go1Getup.reset 里显式 `data.replace(time=0.0)`, 之后每步 time += 0.02)。
  2. 同一个原因导致出生姿态整轮**冻结**在 768 个。这里用离线预沉降姿态池
     (sim/make_getup_posepool.py) 在 restart 那一步做一次 gather 重采 -> 每个 episode
     都是新的摔倒姿态, 成本只有一次索引 (而不是 reset 的 ~950ms 物理沉降)。
  3. 动作语义: 本 env 是**绝对** ctrl (直接调 Go1Walk.step), 不是 Go1Getup 的增量语义。
     增量语义的 Go1Getup 保留给 BC/DAgger 那条线。

奖励与判据对齐 (判据 = sim/eval_getup.py 的同一个量):
    up = 1 - 2*(qpos[4]^2 + qpos[5]^2);  z_rel = qpos[2] - terrain_height(xy)
    站立 = up > 0.99 且 z_rel in [0.24, 0.32] 连续 25 步
  * up 用**根四元数**直接算, 不再用 get_gravity (后者与判据不是同一个量);
  * 高度用**高斯尖峰**对齐"落在带内"的判据 (斜坡在带内仍有斜率, 会把策略拉到带外);
  * 成功 -> 稀疏奖金 + **成功终止** (AFR 2412.16924: 保持稳定站立即成功终止)。
"""
import os
import sys

import jax
import jax.numpy as jp
import numpy as np
from ml_collections import config_dict
from mujoco_playground._src import mjx_env

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
  sys.path.insert(0, _ROOT)

from envs.go1_getup import Go1Getup, default_config as getup_default_config
from envs.go1_walk import Go1Walk
from sim.getup_keyframe import fsm_step, init_fsm

FSM_KEYS = ("stand_target", "phase", "timer", "stable", "attempt", "max_attempts",
            "kip", "dur_kip", "ctrl_prev")
POOL_PATH = os.path.join(_ROOT, "models", "getup_posepool.npz")


def default_config() -> config_dict.ConfigDict:
  cfg = getup_default_config()
  cfg.getup.residual_scale = 0.30   # 残差幅度: ctrl = ctrl_fsm + a*0.30
  cfg.getup.band_sigma = 0.05       # 高度尖峰宽度 (m)
  cfg.getup.z_ramp_lo = 0.16        # 斜坡下沿: 低于它就完全不给奖励
  # P_STAND/P_FINISH 的保持目标相对 HOME 的偏置 (hip, thigh, calf), 各腿同值。
  # 默认全 0 = 与旧行为逐位一致; 扫参见 sim/probe_getup_standbias.py。
  cfg.getup.stand_bias = (0.0, 0.0, 0.0)
  cfg.getup.success_bonus = 300.0   # 稀疏成功奖金 (站立稠密项约 6.3/步)
  # **关键**: 高度斜坡的下沿。v30a (下沿=0) 实测: 趴着不动 400 步能拿满 ~1.8/步
  cfg.getup.hold_steps = 25         # 与 eval_getup.py 的 --hold 一致
  cfg.getup.use_pose_pool = True
  cfg.getup.pose_pool_path = POOL_PATH   # 可换成别的池 (例如'真实摔倒'档)
  # 判据用到的两个量, 直接从这里取, 保证训练/评估同源
  cfg.getup.z_lo = 0.24
  cfg.getup.z_hi = 0.32
  cfg.getup.up_th = 0.99
  # 姿态池是 8192 env 一起沉降出来的, 约束行数比走路多 (实测 needs>=260)
  cfg.njmax = 768
  # 奖励项**全部重建**: 旧 9 项里 posture 把关节拉向名义站姿(而名义站姿静置会翻倒,
  # 见 sim/probe_getup_standpose.py), stand_still 收到的是绝对 ctrl 恒等于 0。
  cfg.reward_config.scales = config_dict.create(
      feet_phase=2.0,      # 只为把相位留在 obs 里 (91 维); 奖励里不返回这一项
      upright=1.0,         # up_gate: up<=0.80 -> 0, up>=0.99 -> 1 (与判据同一个 up)
      height_ramp=1.0,     # up_gate * clip((z_rel-0.16)/0.10, 0, 1)
      height_band=4.0,     # up_gate * 高斯尖峰 (峰值在 z_des=0.26)
      still=0.3,           # up_gate * ramp * exp(-0.5*||qvel[6:]||^2)
      action_rate=-0.01,
      res_mag=-0.002,      # 惩罚大残差 (鼓励靠近先验)
      dof_vel=-0.02,
  )
  # 400 步 = 8s: 关键帧一个完整周期是 WAIT+KIP(50~125)+CROUCH(60)+STAND(125) ~= 385 步,
  # 300 步时状态机连自己的 P_DONE 都到不了 (probe_keyframe_horizon 实测"自称成功 0%")。
  cfg.episode_length = 400
  return cfg


class Go1GetupResidual(Go1Getup):
  """ctrl = clip(ctrl_fsm + a*residual_scale); obs 与走路同形。"""

  def __init__(self, config: config_dict.ConfigDict = None, **kwargs):
    super().__init__(config or default_config(), **kwargs)  # type: ignore[arg-type]
    gu = self._config.getup
    self._res_scale = float(gu.residual_scale)
    self._z_lo = float(gu.z_lo)
    self._z_hi = float(gu.z_hi)
    self._up_th = float(gu.up_th)
    self._hold_steps = int(gu.hold_steps)
    self._success_bonus = float(gu.success_bonus)
    self._use_pool = False
    pool_path = getattr(gu, "pose_pool_path", POOL_PATH) or POOL_PATH
    if gu.use_pose_pool and os.path.exists(pool_path):
      z = np.load(pool_path)
      self._pool_qpos = jp.asarray(z["qpos"], jp.float32)
      self._pool_qvel = jp.asarray(z["qvel"], jp.float32)
      self._use_pool = True
      print(f"[getup_res] 姿态池 {self._pool_qpos.shape} <- {pool_path}")
    else:
      print(f"[getup_res] 无姿态池 ({pool_path} 不存在), 出生姿态整轮冻结")

  def _stand_target(self):
    """HOME + cfg.getup.stand_bias (hip/thigh/calf 各腿同偏置; 默认 0 = 与旧行为一致)。"""
    b = np.asarray(self._config.getup.get("stand_bias", (0.0, 0.0, 0.0)),
                   dtype=np.float32)
    t = np.asarray(self._default_pose, dtype=np.float32).copy()
    for i in (0, 3, 6, 9):
      t[i] += b[0]
    for i in (1, 4, 7, 10):
      t[i] += b[1]
    for i in (2, 5, 8, 11):
      t[i] += b[2]
    return t

  # ---------------- FSM 在 info 里的扁平读写 ----------------
  def _fsm_put(self, info, fsm):
    for k in FSM_KEYS:
      info["fsm_" + k] = fsm[k]

  def _fsm_get(self, info):
    return {k: info["fsm_" + k] for k in FSM_KEYS}

  def reset(self, rng: jax.Array):
    state = super().reset(rng)
    info = state.info
    self._fsm_put(info, init_fsm(hfield=bool(self._hfield_ok),
                                 stand_target=self._stand_target()))
    info["up_hold"] = jp.int32(0)
    info["last_res"] = jp.zeros(self.mjx_model.nu)
    info["last_res_prev"] = jp.zeros(self.mjx_model.nu)
    info["n_success"] = jp.zeros(())
    info["fsm_status"] = jp.int32(0)   # 只在 step 里写 -> 必须在 reset 里先建好
    # 0 = "reset 自己那一次出生已经够新, 不要在第一步重采"; 之后一直 1。
    # 这样 sim/eval_getup.py 按 reset 后的姿态分桶仍然是对的。
    info["allow_respawn"] = jp.int32(0)
    # metric 的 key 集合必须在 reset 时定下来 (brax 的 EvalWrapper 会 tree_map 结构)
    state.metrics["getup/is_success"] = jp.zeros(())
    state.metrics["getup/up_z"] = jp.zeros(())
    state.metrics["getup/z_rel"] = jp.zeros(())
    return state

  # ---------------- step ----------------
  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    info = state.info
    gu = self._config.getup
    restart = state.data.time <= 0.0

    # (1) episode 第一步: FSM 归零 + 计数器清零 + (可选) 从姿态池重采出生姿态
    fresh = init_fsm(hfield=bool(self._hfield_ok),
                     stand_target=self._stand_target())
    fsm = {k: jp.where(restart, fresh[k], info["fsm_" + k]) for k in FSM_KEYS}
    info["up_hold"] = jp.where(restart, jp.int32(0), info["up_hold"])
    info["last_res"] = jp.where(restart, jp.zeros(self.mjx_model.nu),
                                info["last_res"])
    respawn = restart & (info["allow_respawn"] > 0) if self._use_pool else restart
    info["allow_respawn"] = jp.where(restart, jp.int32(1), info["allow_respawn"])
    if self._use_pool:
      rng, kp = jax.random.split(info["rng"])
      info["rng"] = rng
      idx = jax.random.randint(kp, (), 0, self._pool_qpos.shape[0])
      d = state.data
      state = state.replace(data=d.replace(
          qpos=jp.where(respawn, self._pool_qpos[idx], d.qpos),
          qvel=jp.where(respawn, self._pool_qvel[idx], d.qvel)))

    # (2) 先验动作 -> 加残差 -> 绝对 ctrl
    ctrl_fsm, status, reset_joints, fsm_new = fsm_step(
        fsm, state.data.qpos, state.data.qvel, self._default_pose)
    target = jp.clip(ctrl_fsm + action * self._res_scale,
                     self._ctrl_lo, self._ctrl_hi)
    info["last_res_prev"] = info["last_res"]
    info["last_res"] = action
    self._fsm_put(info, fsm_new)
    info["fsm_status"] = status

    # (3) 物理 + 奖励走走路档的绝对语义
    state = Go1Walk.step(self, state, target)

    # (4) 关键帧的"关节手术": 进 P4 时把残留角拉回 home (否则伺服拉不回, pitch 卡住)
    q = state.data.qpos
    newj = jp.where(jp.asarray(reset_joints)[..., None],
                    self._default_pose, q[7:19])
    state = state.tree_replace(
        {"data": state.data.replace(qpos=q.at[7:19].set(newj))})

    # (5) 判据 (与 sim/eval_getup.py 逐字一致) + 稀疏成功奖金 + 成功终止
    q = state.data.qpos
    up = 1.0 - 2.0 * (q[4] ** 2 + q[5] ** 2)
    z_rel = q[2] - self._z_ref(state.data)
    ok = ((up > self._up_th) & (z_rel >= self._z_lo) & (z_rel <= self._z_hi))
    hold = jp.where(ok, info["up_hold"] + 1, 0)
    info["up_hold"] = hold
    success = hold >= self._hold_steps
    s_f = success.astype(state.reward.dtype)
    state = state.replace(
        done=jp.maximum(state.done, s_f),
        reward=state.reward + s_f * self._success_bonus)
    info["n_success"] = info["n_success"] + s_f
    state.metrics["getup/is_success"] = s_f
    state.metrics["getup/up_z"] = up
    state.metrics["getup/z_rel"] = z_rel
    return state

  # ---------------- 奖励 (重建, 与判据同源) ----------------
  def _get_reward(self, data, action, info, done, first_contact, contact):
    gu = self._config.getup
    q = data.qpos
    up = 1.0 - 2.0 * (q[4] ** 2 + q[5] ** 2)
    z_rel = q[2] - self._z_ref(data)
    z_lo_r = jp.float32(gu.z_ramp_lo)
    ramp = jp.clip((z_rel - z_lo_r) / (jp.float32(gu.z_des) - z_lo_r),
                   0.0, 1.0)
    band = jp.exp(-jp.square((z_rel - jp.float32(gu.z_des))
                             / jp.float32(gu.band_sigma)))
    # v30c **关键修正**: 每一项都乘 up_gate。
    # v30b 实测 (best checkpoint 只有 12.1% 成功, 却把 eval_reward 从 244 涨到 485):
    # height_band 权重 4.0 只要求"高", 不要求"直" —— "后腿站起来/头朝下顶高"这种
    # 姿态能白拿 4.0/步, 是新的套利点。up_gate 把"高但不直"的收益清零。
    up_gate = jp.clip((up - 0.80) / 0.19, 0.0, 1.0)
    res = info["last_res"]
    res_prev = info["last_res_prev"]
    return {
        "upright": up_gate,
        "height_ramp": up_gate * ramp,
        "height_band": up_gate * band,
        "still": up_gate * ramp * jp.exp(-0.5 * jp.sum(jp.square(data.qvel[6:]))),
        "action_rate": jp.sum(jp.square(res - res_prev)),
        "res_mag": jp.sum(jp.square(res)),
        "dof_vel": jp.sum(jp.square(jp.maximum(
            jp.abs(data.qvel[6:]) - 2.0 * jp.pi, 0.0))),
    }
