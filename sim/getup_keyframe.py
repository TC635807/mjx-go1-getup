#!/usr/bin/env python
"""关键帧起身控制 (把 §15 的结论落成可复用模块, 并修掉它在 hfield 地面上的失效)。

来源与结论
----------
§15 (`RLcontroller_go1/sim/probe_getup_e2e.py --phase 6`) 给出的序列:
  W 等静止 (闭环) -> P2 kip (四腿全伸) -> P3 蹲姿收脚 + 站立 -> P4 收尾

机制: 翻面**不是绕 roll 的翻滚, 而是绕俯仰轴 (thigh) 的"仰卧起坐"** —— 四腿
同时全伸 (thi=4.501 / cal=-2.818, 都是 ctrlrange 的行程端点) 时, 躯干被反作用
力矩抬离地面并旋转, 越过临界点后重力把身体带到俯卧 (§15.3)。

**本模块新增的关键修正 (2026-09-19)**: §15 的 kip **只在 plane 地面上成立**。
在查看器真正用的地形场景 (floor 是 `hfield`, 见 models/go1/scene_mjx_terrain.xml)
上, 同一个 kip 成功率只有 ~10%: 实测 kip 期间 plane 有 4~8 个接触、机器人会**离地**,
而 hfield 有 20~40 个接触、腿一直贴着格子被"拖住", 躯干最高只抬 9cm (plane 是 35cm)。
把 hfield 压成绝对平面 (elevation_z=1e-6) 后**仍然失败** —— 所以是 hfield 的接触
生成方式, 不是地形起伏。修法是换一套 kip 幅度 + **失败重试**: 见 VARIANTS_HFIELD。

实测 (sim/probe_getup_mjx.py, MJX/warp, 自然随机摔落):
  plane floor : 100%  (与 §15 的 CPU 结果一致)
  hfield floor: 单方案 ~50%, 多方案重试后见探针输出 (仍显著低于 plane)

因此查看器里**失败会回退到重置** (可用 --fall_action 选), 不会卡死。

与查看器的接口
--------------
* 动作语义: `envs/go1_walk.py` 的 `motor_targets = clip(action, ctrlrange)` ——
  动作就是 ctrl 目标, 无缩放。本模块输出的 12 维向量直接当 action 喂 `env.step`。
* 时钟: 所有时长以**控制步**计 (env.dt == ctrl_dt == 0.02s, 50Hz)。查看器 1 帧 = 1 步。
* P4 收尾有**状态手术**: 进入 P4 那一帧把 qpos[7:19] 复位到 home keyframe
  (§15.5 E: kip 会留下非 home 的关节残角 calf≈-1.71 vs home -1.5, 位置伺服拉不
  回来, pitch 卡在 -3.4°)。`fsm_step` 用 `reset_joints` 标志通知调用方。
"""
import numpy as np
import jax.numpy as jp
import jax.numpy as jnp       # 同 jp (本文件两种写法混用, 保留别名避免误用)

# ---------------------------------------------------------------- 常量
CTRL_DT = 0.02                  # envs/go1_walk.py ctrl_dt (50Hz 控制)

HOME = np.array([0.0, 0.8, -1.5,
                 0.0, 0.8, -1.5,
                 0.0, 1.0, -1.5,
                 0.0, 1.0, -1.5], dtype=np.float32)

CROUCH = HOME.copy()
CROUCH[[1, 4, 7, 10]] = 1.5      # TUCK_THI
CROUCH[[2, 5, 8, 11]] = -2.5     # TUCK_CAL

CTRL_LO = np.array([-0.863, -0.686, -2.818] * 4, dtype=np.float32)
CTRL_HI = np.array([0.863, 4.501, -0.888] * 4, dtype=np.float32)

# ---- kip 方案: (hip, thigh, calf, 时长[控制步]) --------------------------
# plane floor: §15 原版, 100%
VARIANTS_PLANE = [
    (0.0, 4.501, -2.818, 50),        # §15 P2, 1.0s
]
# hfield floor: 实测扫参得到; 按"单次成功率"排序, 失败就换下一个。
# 关键不是某一个方案多好 (单方案上限也就 ~1/3), 而是**让每次重试从一个不同的
# 接触构型出发** —— hfield 的失败是"腿被格子卡住"这种构型依赖的失败, 换个幅度
# 就换了一条接触历史, 所以 8 个差异够大的方案比 4 个同族方案强得多。
VARIANTS_HFIELD = [
    (0.0, 4.501, -2.818, 50),        # §15 平面 kip (在 hfield 上也常能离地)
    (0.3, 3.8, -2.4, 100),
    (-0.3, 3.5, -2.2, 75),
    (0.3, 3.2, -2.0, 50),
    (-0.3, 4.0, -2.0, 75),
    (0.0, 3.5, -2.6, 100),
    (0.3, 4.2, -2.8, 60),
    (-0.3, 3.8, -2.4, 125),
]

KIP = np.array([0.0, 4.501, -2.818] * 4, dtype=np.float32)   # 兼容旧引用

# 相位
P_WAIT, P_KIP, P_CROUCH, P_STAND, P_FINISH, P_DONE, P_FAIL = 0, 1, 2, 3, 4, 5, 6
PHASE_NAMES = {
    P_WAIT: "W 等静止",
    P_KIP: "P2 kip 翻面",
    P_CROUCH: "P3a 蹲姿收脚",
    P_STAND: "P3b 站立",
    P_FINISH: "P4 收尾",
    P_DONE: "完成 (交还策略)",
    P_FAIL: "起身失败",
}
N_PHASE = 7

# 时长 (控制步; 1 步 = 0.02s)
DUR_CROUCH = 60                 # 1.2s
DUR_STAND = 125                 # 2.5s
DUR_FINISH = 150                # 3.0s

WAIT_MAX = 300                  # 等静止最多 6s
REST_VTH = 0.15
REST_HOLD = 15                  # 0.3s

RAMP = 4.0                      # P3a 关节限速 rad/s

PRONE_UZ = 0.85                 # kip 后 up_z 高于此值才算翻面成功
STATUS_RUN, STATUS_OK, STATUS_FAIL = 0, 1, 2

DEFAULT_MAX_ATTEMPTS = 8


def variants_for(hfield: bool):
    """地面类型 -> kip 方案表。hfield = 地形场景 (查看器里 v19+ 策略用)。"""
    return VARIANTS_HFIELD if hfield else VARIANTS_PLANE


def up_z_from_quat(quat) -> float:
    """躯干 up 轴在世界 z 的投影 (= R[2,2])。1=站正, 0=侧躺, -1=仰卧。"""
    w, x, y, z = [float(v) for v in quat]
    return 1.0 - 2.0 * (x * x + y * y)


# ---------------------------------------------------------------- 状态机
def _variant_arrays(variants, idx):
    hip, thi, cal, dur = variants[int(idx) % len(variants)]
    kip = jp.asarray(np.array([hip, thi, cal] * 4, dtype=np.float32))
    return kip, jp.int32(dur)


def init_fsm(hfield=False, max_attempts=DEFAULT_MAX_ATTEMPTS, stand_target=None):
    """单环境状态机初态 (可被 jax.vmap 批量展开)。

    hfield=True 时用 hfield 方案表 (地形场景); 返回值全部是 jax 数组。

    stand_target: P_STAND/P_FINISH 持有的**关节目标** (12 维绝对 ctrl), 默认 HOME。
      **为什么要可配** (sim/probe_getup_standpose.py 实测): 这套 kp=20/kv=1 的位置伺服
      下"命令名义关节角"**不是**站立平衡点 —— 名义姿态静置会翻到背上 (up_z -0.997),
      能站住的是一条很窄的偏置带 (thigh ~ +0.04, calf ~ +0.15)。所以最后"站住"那段
      用 HOME 是次优的; 做成参数就能直接扫出更好的保持目标。
    """
    kip, dur = _variant_arrays(variants_for(hfield), 0)
    st = (jp.asarray(HOME) if stand_target is None
          else jp.asarray(np.asarray(stand_target, dtype=np.float32)))
    return {
        "stand_target": st,
        "phase": jp.int32(P_WAIT),
        "timer": jp.int32(0),
        "stable": jp.int32(0),
        "attempt": jp.int32(0),
        "max_attempts": jp.int32(max_attempts),
        "kip": kip,
        "dur_kip": dur,
        "ctrl_prev": jp.asarray(HOME),
    }


def fsm_step(fsm, qpos, qvel, home_joint):
    """走**一个控制步**。返回 (ctrl, status, reset_joints, fsm_new)。

    ctrl         : (12,) float32 —— 直接当 action 用
    status       : int32 标量, 0=进行中, 1=成功(已站好), 2=放弃
    reset_joints : bool 标量 —— 这一帧要把 qpos[7:19] 复位到 home_joint
    """
    del home_joint                # 只用于调用方做状态手术
    p = fsm["phase"]
    t = fsm["timer"]
    stable = fsm["stable"]
    ctrl_prev = fsm["ctrl_prev"]

    v = jp.linalg.norm(qvel[:6])
    stable2 = jp.where(v < REST_VTH, stable + 1, 0)
    t2 = t + 1

    c_crouch = jp.clip(jp.asarray(CROUCH), ctrl_prev - RAMP * CTRL_DT,
                       ctrl_prev + RAMP * CTRL_DT)
    ctrl = jp.where(p == P_KIP, fsm["kip"],
                    jp.where(p == P_CROUCH, c_crouch,
                             fsm["stand_target"]))
    ctrl = jp.clip(ctrl, jp.asarray(CTRL_LO), jp.asarray(CTRL_HI))

    up_z = 1.0 - 2.0 * (qpos[4] ** 2 + qpos[5] ** 2)
    prone_ok = up_z > PRONE_UZ

    # ---- 相位推进
    adv_wait = (p == P_WAIT) & ((stable2 >= REST_HOLD) | (t2 >= WAIT_MAX))
    adv_kip = (p == P_KIP) & (t2 >= fsm["dur_kip"])
    adv_crouch = (p == P_CROUCH) & (t2 >= DUR_CROUCH)
    adv_stand = (p == P_STAND) & (t2 >= DUR_STAND)
    adv_finish = (p == P_FINISH) & (t2 >= DUR_FINISH)
    adv = adv_wait | adv_kip | adv_crouch | adv_stand | adv_finish

    # kip 结束: 翻面成功 -> 站起来; 失败 -> 换下一个方案重试, 用完则放弃
    kip_fail = adv_kip & (~prone_ok)
    attempt2 = jp.where(kip_fail, fsm["attempt"] + 1, fsm["attempt"])
    retry = kip_fail & (attempt2 < fsm["max_attempts"])

    nxt = p
    nxt = jnp.where(adv_wait, P_KIP, nxt)
    nxt = jnp.where(adv_kip & prone_ok, P_CROUCH, nxt)
    nxt = jnp.where(kip_fail & retry, P_WAIT, nxt)
    nxt = jnp.where(kip_fail & (~retry), P_FAIL, nxt)
    nxt = jnp.where(adv_crouch, P_STAND, nxt)
    nxt = jnp.where(adv_stand, P_FINISH, nxt)
    # P4 跑完时必须**真的站住**才算成功: 否则查看器会以为"起身成功"而放弃回退,
    # 把一条还躺着的狗留在场上 (实测地形档有约 1/4 的情况是"状态机说成功, 其实躺着")。
    upright = up_z > 0.90
    nxt = jnp.where(adv_finish & upright, P_DONE, nxt)
    nxt = jnp.where(adv_finish & (~upright), P_FAIL, nxt)

    # 换方案: 按新的 attempt 在方案表上取 (表是常量, 逐元素 stack 后索引)
    kips = jnp.stack([_variant_arrays(_VAR_TABLE, i)[0]
                      for i in range(len(_VAR_TABLE))])
    durs = jnp.stack([_variant_arrays(_VAR_TABLE, i)[1]
                      for i in range(len(_VAR_TABLE))])
    sel = jnp.mod(attempt2, len(_VAR_TABLE))
    new_kip = kips[sel]
    new_dur = durs[sel]
    kip2 = jnp.where(kip_fail, new_kip, fsm["kip"])
    dur2 = jnp.where(kip_fail, new_dur, fsm["dur_kip"])

    timer2 = jnp.where(adv, jnp.int32(0), t2)
    reset_joints = adv_stand
    status = jnp.where(nxt == P_DONE, jp.int32(STATUS_OK),
              jnp.where(nxt == P_FAIL, jp.int32(STATUS_FAIL),
                        jp.int32(STATUS_RUN)))

    fsm_new = {
        "stand_target": fsm["stand_target"],
        "phase": nxt,
        "timer": timer2,
        "stable": stable2,
        "attempt": attempt2,
        "max_attempts": fsm["max_attempts"],
        "kip": kip2,
        "dur_kip": dur2,
        "ctrl_prev": ctrl,
    }
    return ctrl, status, reset_joints, fsm_new


# hfield/plane 两张表合并成一张, 供 vmap 里的 jnp.where 取用
_VAR_TABLE = VARIANTS_HFIELD + [v for v in VARIANTS_PLANE
                                if v not in VARIANTS_HFIELD]


def apply_joint_reset(state, home_joint):
    """进入 P4 时的状态手术: qpos[7:19] <- home keyframe (mjx_env.State)。"""
    return state.tree_replace({
        "data": state.data.replace(
            qpos=state.data.qpos.at[7:19].set(jp.asarray(home_joint))),
    })


def phase_name(fsm) -> str:
    return PHASE_NAMES.get(int(np.asarray(fsm["phase"])), "?")
