#!/usr/bin/env python
"""Go1 训练环境原封不动当仿真环境: warp 后端跑物理 + 策略推理, 渲染到 viewer。

与训练完全一致 (同一个 Go1Walk env 类、同一个 warp 后端)。

帧率优化 (2026-09-11, 实测见 sim/diag_viewer_perf.py(已删除, 结论见实验记录)):
  - `step_jit` 用 `donate_argnums=0`: 让输出复用输入 buffer, 地址稳定后 warp 的
    CUDA graph 缓存命中。**这是最大头**: 单步 25.9ms → 14.8ms; 端到端 (mimic)
    每帧 43.3ms → 13.4ms (23fps → 75fps 上限; 实时节流下稳定 50fps)。
    注意不能改成 graph_mode=NONE: 不 capture 反而 131ms/步 (8fps)。
  - 渲染搬运只拷 qpos/qvel + CPU `mj_forward`, 不再 `mjx.get_data`+`mj_copyData`
    (后者要搬整个 Data, 含 naconmax=65536 的接触数组; 1.13ms → 0.02ms)。
  - 原来的 `time.sleep(env.dt)` 是无条件叠加的, 与帧耗时无关地再加 20ms;
    现在改成"距下一帧还差多少补多少"的实时节流 —— 快时不超频, 慢时不叠加。
  - overlay 显示 fps 与 step/copy/render 分项耗时, 便于继续定位瓶颈。

用法:
  python sim/view_go1.py --pkl policies/go1_walk_policy.pkl
  # v14/v15 必须带 --body_frame 与训练时的 --tilt_deg, 否则 obs 与训练不一致
  python sim/view_go1.py --pkl policies/go1_walk_policy.pkl \
      --body_frame --tilt_deg 17.5

getup_v2 (方案 A, 见 envs/go1_getup_v2.py): 起身策略 obs 是 42x5=210 维**历史帧**,
与走路策略 (91/83/48) 本来就不同形, 所以查看器**自己维护历史缓冲**, 两个策略各吃
各的 obs (不再共用一份 state):
  python sim/view_go1.py --pkl models/go1_walk_policy_v20.pkl \
      --getup_pkl models/go1_getup_v2.pkl --getup_obs history --getup_action anchored
不加 --getup_obs history 时行为与旧版逐位一致 (起身策略吃走路那份 state)。

物理默认硬接触 (v5+ 训练档); 只有看 v1–v4 老策略才加 --soft_contact。
每 25 帧向终端打一行 fps/分项耗时, overlay 里也有。

键盘:
  W/S  提高/降低 vx 指令
  A/D  左转(wz+)/右转(wz−) 指令 (每帧 0.02 rad/s)
  R    重置   Q/Esc 退出
"""
import argparse
import os
import sys
import time

# --- llvmpipe 线程限流 (必须在 import mujoco 之前设置!) ---
# 背景 (2026-09-14 实测): WSL 无 /dev/dri -> OpenGL 走 Mesa llvmpipe 软件
# 渲染, 默认会用满全部 CPU 核 (本机 16 核)。而 jax/warp 的 kernel 分发是
# CPU 侧的 —— llvmpipe 吃满核后饿死分发线程, 表现为 viewer 里 "step" 暴涨:
#   地形 + quality=low, LP_NUM_THREADS 未设 -> step=460ms, 2-3 fps
#   地形 + quality=low, LP_NUM_THREADS=4  -> step=22-25ms, 37-40 fps
#   地形 + quality=low, LP_NUM_THREADS=2  -> 卡爆 (渲染本身跟不上, 反而更慢)
# 4 是本机的最优值 (留出 12 核给物理分发)。
# 用户若已显式设置该变量则不覆盖 (setdefault 语义)。
os.environ.setdefault("LP_NUM_THREADS", "4")
# --- 显存池上限 (必须在 import jax 之前设置!) ---
# XLA 默认预占 ~75% 显存 -> 两个查看器同时开时打满 8G, GPU 掉到 180MHz,
# 单步 15ms 暴涨到 300-550ms (2-3fps)。实测见 watch_v20_fast.py 同处注释。
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.25")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import jax
import jax.numpy as jp
import mujoco
import mujoco.viewer
import numpy as np
from pynput import keyboard

from envs.go1_walk import Go1Walk, default_config
from sim import getup_keyframe as gk

_ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CKPT_DIR = os.path.join(_ROOT_DIR, "logs", "walk", "checkpoints")

# getup_v2 的历史观测规格 (envs/go1_getup_v2.py::_frame / cfg.getup2.history):
# 单帧 42 维 x 5 帧 = 210 维, 拼平顺序 hist.reshape(-1) (最新帧在最后)。
GETUP_FRAME_DIM = 42
GETUP_HIST = 5


def find_latest_ckpt():
    steps = []
    for name in os.listdir(CKPT_DIR):
        p = os.path.join(CKPT_DIR, name)
        if os.path.isdir(p) and name.isdigit():
            steps.append((int(name), p))
    if not steps:
        return None
    steps.sort()
    return steps[-1][1]


def infer_hidden_layers(params):
    """从参数形状反推隐层大小 (pkl 不存网络配置: v10=(64,64), v14/v15=(128,128))。

    pkl 是 brax 的 (normalizer, policy_params, value_params) 三元组;
    policy params 形如
      {'MLP_0': {'hidden_0': {'kernel': (in, out), 'bias': (out,)}, 'hidden_1': ...},
       'Dense_0': {...}, 'std_param': ...}
    hidden 层嵌在 MLP_0 下, 所以递归找所有以 hidden_ 开头、含 kernel 的项。
    """
    roots = [params]
    if isinstance(params, (tuple, list)) and len(params) > 1:
        roots = [params[1]]  # (normalizer, policy, value) -> 只看 policy
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
    if not found:
        return (64, 64)
    return tuple(found[i] for i in sorted(found))


def infer_obs_sizes(params):
    """从参数形状反推 obs 维度, 返回 (state_dim, privileged_dim)。

    v19 起 obs 维度不再是固定的 48/119 (开 height_scan 后变 83/154), 加载器
    再写死就会 ScopeParamShapeError。这里直接读第一层 kernel 的输入维:
      policy:  MLP_0.hidden_0.kernel shape = (obs_in, hidden)
      value:   MLP_0.hidden_0.kernel shape = (obs_in, hidden)
    params 是 brax 的 (normalizer, policy_params, value_params) 三元组。
    """
    def first_kernel_dim(node):
        # 递归找最浅的 hidden_0 kernel (MLP_0 下), 返回其输入维
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

    pol = val = None
    if isinstance(params, (tuple, list)) and len(params) >= 3:
        pol, val = first_kernel_dim(params[1]), first_kernel_dim(params[2])
    else:
        pol = first_kernel_dim(params)
    # 兜底: 拿不到就退回 v18 的已知尺寸 (平地档)
    pol = pol or 48
    val = val or 119
    return pol, val


def build_policy(params, obs_key="state"):
    # 网络结构从参数形状反推 —— v10 是 (64,64), v14/v15 是 (128,128),
    # 必须与训练时一致, 否则参数形状对不上 (ScopeParamShapeError)。
    # v19: obs 维度也一并反推 (height_scan 会改维度, 不能写死)。
    from brax.training.agents.ppo import networks as ppo_networks
    layers = infer_hidden_layers(params)
    n_state, n_priv = infer_obs_sizes(params)
    network = ppo_networks.make_ppo_networks(
        observation_size={"state": (n_state,), "privileged_state": (n_priv,)},
        action_size=12,
        policy_hidden_layer_sizes=layers,
        value_hidden_layer_sizes=layers,
        policy_obs_key=obs_key,
        value_obs_key="privileged_state",
        distribution_type="normal",
        activation=jax.nn.tanh,
    )
    make_inference_fn = ppo_networks.make_inference_fn(network)
    return make_inference_fn(params, deterministic=True)


def load_policy_ckpt(ckpt_path):
    # 不用 brax ppo_ckpt.load_policy (其 load_config 对 json 里的
    # mean_kernel_init_fn=null 会 KeyError) —— 手动读 json 建网络。
    from brax.training.agents.ppo import checkpoint as ppo_ckpt
    from brax.training.agents.ppo import networks as ppo_networks
    import json as _json
    cfg = _json.load(open(os.path.join(ckpt_path, "ppo_network_config.json")))
    kw = cfg["network_factory_kwargs"]
    params = ppo_ckpt.load(ckpt_path)
    # v19: 维度从参数反推 (json 里的 observation_size 可能是 dict 也可能缺)
    n_state, n_priv = infer_obs_sizes(params)
    network = ppo_networks.make_ppo_networks(
        observation_size={"state": (n_state,), "privileged_state": (n_priv,)},
        action_size=cfg["action_size"],
        policy_hidden_layer_sizes=tuple(kw["policy_hidden_layer_sizes"]),
        value_hidden_layer_sizes=tuple(kw["value_hidden_layer_sizes"]),
        policy_obs_key=kw["policy_obs_key"],
        value_obs_key=kw["value_obs_key"],
        distribution_type=kw.get("distribution_type", "tanh_normal"),
        activation=jax.nn.tanh,
    )
    make_inference_fn = ppo_networks.make_inference_fn(network)
    return make_inference_fn(params, deterministic=True), (n_state, n_priv)


def load_policy_pkl(pkl_path):
    import pickle
    with open(pkl_path, "rb") as f:
        params = pickle.load(f)
    return build_policy(params), infer_obs_sizes(params)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--pkl", type=str, default=None)
    ap.add_argument("--cmd", type=float, default=0.5, help="vx 指令 (W/S 键调)")
    ap.add_argument("--wz", type=float, default=0.0,
                    help="wz 转向指令 rad/s, 正=逆时针 (A/D 键调)")
    ap.add_argument("--seed", type=int, default=-1,
                  help="随机出生的 seed。-1 (默认) = 每次启动随机取 (否则每次都"
                       "落在同一个点; 实测 seed=0 会固定落在 (-4.17,+2.81) 的陡坡,"
                       " 左右脚地面高差 3.5cm); 给具体整数可复现同一起点")
    ap.add_argument("--soft_contact", dest="hard_contact", action="store_false",
                    help="足端软接触 (只有 v1–v4 老策略才用; 默认硬接触 = v5+ 训练物理)")
    ap.set_defaults(hard_contact=True)
    ap.add_argument("--body_frame", action="store_true",
                    help="指令/观测用机体系速度 (v14+ 训练用)。看 v14/v15 必须加,"
                         "否则策略收到的 obs 与训练时不一致, 行为会不对")
    ap.add_argument("--terrain", action="store_true",
                    help="加载地形场景 (凹凸 + 2 条 5 级阶梯, 总高差 0.30m, "
                         "models/go1/scene_mjx_terrain.xml)")
    ap.add_argument("--height_scan", action="store_true",
                    help="v19: 打开 35 点高度扫描 (看 v19 策略必须加, 否则 obs "
                         "维度是 48 而策略期望 83 -> 形状不符)。自动开启 --terrain")
    ap.add_argument("--no_random_init", action="store_true",
                    help="v19: 关闭随机出生 (默认开; 关掉便于固定起点复现)")
    ap.add_argument("--push", action="store_true",
                    help="v19: 打开周期性推挤扰动 (观察鲁棒性)")
    ap.add_argument("--obs_noise", action="store_true",
                    help="v19: 打开观测噪声 (观察鲁棒性)")
    ap.add_argument("--scan_points", action="store_true",
                    help="把 35 个高度扫描点画成小球 (位置=地面高度, 颜色=特征值: "
                         "红=前方更高/上坡, 蓝=下陷), 并画\"机器人->采样点\"射线。"
                         "需要 --height_scan (v19/v20/v21 策略自动满足)。"
                         "按 T 键可实时开关")
    ap.add_argument("--warmup", type=int, default=0,
                    help="开窗口前的额外预热帧数 (默认 0 = 关)。曾试过 250 想跑掉"
                         "启动暂态, 但实测反而更慢 (step 340ms 不恢复), 故默认关。"
                         "保留开关供后续实验")
    ap.add_argument("--graph_mode", choices=["WARP", "WARP_STAGED",
                                             "WARP_STAGED_EX", "NONE"],
                    default="WARP_STAGED",
                    help="warp 的 CUDA graph 捕获模式。**默认 WARP_STAGED**: "
                         "MJX 默认的 WARP 模式按 buffer 地址缓存图, 而查看器每帧"
                         "新建 command 数组 + R 键重置都会换地址 -> 反复重新捕获 "
                         "-> 跑几千帧后崩 (Warp error: unknown stream, 用户实测 "
                         "frame 2125)。WARP_STAGED 用 staging buffer, 实测 "
                         "16.3ms/帧 且 3000 帧无崩溃 (WARP 是 185ms/帧)。"
                         "传 WARP 可复现旧行为, NONE 最稳但慢 (139ms/帧)")
    ap.add_argument("--tilt_deg", type=float, default=None,
                    help="健康终止阈值 (v15 训练用 17.5; 默认 10 = 参考档 ±20°)")
    # --- 摔倒处理 (2026-09-19): 默认**不再自动重置**, 改为 §15 的关键帧起身 ---
    # 地面类型决定用哪套 kip: plane floor (平地场景) 用 §15 原版 (实测 100%);
    # hfield floor (地形场景) 用扫参版 + 重试 (实测显著更低, 见 sim/probe_getup_mjx.py)。
    ap.add_argument("--fall_action", choices=["getup", "getup_hold", "reset"],
                    default="getup",
                    help="摔倒后怎么办: getup=尝试关键帧起身, 失败则重置 (默认); "
                         "getup_hold=尝试起身但失败就躺那不动; reset=旧行为 (直接重置)")
    ap.add_argument("--oob_reset_xy", type=float, default=5.8,
                    help="|xy| 超过它就当作掉出地形边缘并**直接重置**。\n"
                         "地形 (scene_mjx_terrain.xml) 半边长 6.0m, 界外没有地面 ->\n"
                         "自由落体 (实测 |qvel| 每帧 +0.196 = 重力), 此时触发起身只会\n"
                         "让策略在半空挥腿 (视觉上就是抽搐), 还要等 400 帧超时才重置。\n"
                         "默认 5.8 与 envs/go1_getup_v2.py 的 out_of_bounds_xy 一致。")
    ap.add_argument("--fall_z", type=float, default=0.16,
                    help="摔倒判据 1: 躯干**离地**高度低于此值 (米)。必须用相对高度"
                         " —— 地形最高台 0.30m, 绝对 z 判据在高台上永远不触发")
    ap.add_argument("--handover_blend", type=int, default=0,
                    help="交还给走路策略时, 在多少帧内从起身最后的目标线性过渡到走路\n"
                         "策略的目标。**默认 0 = 不混合**。\n"
                         "为什么默认关 (2026-09-21 实测, sim/probe_handover.py): 起身成功\n"
                         "那一帧它的 ctrl 目标是 default+0.5*±8 = 偏离 2.7 rad、而且\n"
                         "**顶在动作裁剪上** (median |a_f|=8.0) —— 那不是站立目标, 是\n"
                         "发力状态。把它当混合起点 = 头 10/30 帧先按这个目标驱动:\n"
                         "  227 个终态 × 150 帧, 摔倒率: 不混合 6.6%% / blend10 37.0%% /\n"
                         "  blend30 73.1%%; 一直保持起身最后目标不交还 = 90.3%%。\n"
                         "  (argparse 对 help 做百分号格式化: 字面百分号要写两遍)\n"
                         "所以混合帧数越大越像抽搐。只有做对照实验时才开。")
    ap.add_argument("--fall_debounce", type=int, default=18,
                    help="摔倒判据要连续满足这么多帧才触发起身 (0.02s/帧; "
                         "滤掉行进中的瞬时踉跄, 真摔倒会一直满足)")
    ap.add_argument("--fall_uz", type=float, default=0.30,
                    help="摔倒判据 2: 躯干 up 轴在世界 z 的投影低于此值。\n"
                         "0.30 ≈ 倾斜 72.5 度。**旧默认 0.45 (63 度) 太松**: 走路时一次\n"
                         "深度踉跄就能连续满足 8 帧而误触发起身 (用户反馈\"明明还能走路\"+ )。")
    ap.add_argument("--getup_attempts", type=int, default=gk.DEFAULT_MAX_ATTEMPTS,
                    help="关键帧起身最多试几个 kip 方案 (见 sim/getup_keyframe.py)")
    # --- 调度器方案: 学到的起身策略 (两个网络, 一份物理, 状态机路由) ---
    ap.add_argument("--getup_pkl", type=str, default=None,
                    help="学到的起身策略权重 (train/train_getup.py 的产物)。给了它, "
                         "摔倒后就**用策略起身**; 不给则退回关键帧序列。"
                         "默认要求 obs 与走路策略同形 (91/162), 否则拒绝启动; "
                         "getup_v2 的 210 维历史 obs 要配 --getup_obs history")
    ap.add_argument("--getup_obs", choices=["flat", "history"], default="flat",
                    help="起身策略吃哪种 obs。**默认 flat** = 旧行为, 直接吃与环境"
                         "(走路策略)同形的那份 state; history = getup_v2 的 "
                         "42x5=210 维**历史帧**, 由查看器自己维护 (与走路不同形, "
                         "故跳过同形检查)")
    ap.add_argument("--getup_action",
                    choices=["incremental", "absolute", "residual", "anchored"],
                    default="incremental",
                    help="起身策略的动作语义。**默认增量**: 训练侧 "
                         "envs/go1_getup.py 把动作解释成 qpos[7:]+a*scale "
                         "(playground getup 同款); 查看器必须做同样变换。"
                         "anchored = getup_v2 语义 (绝对 ctrl 目标): "
                         "act = default_pose + scale*lowpass(clip(a, ±clip))")
    ap.add_argument("--getup_action_scale", type=float, default=0.5,
                    help="动作缩放。incremental: qpos[7:]+a*scale (训练侧 "
                         "cfg.getup.action_scale); anchored: default_pose+scale*a_f "
                         "(envs/go1_getup_v2.py 的 getup2.action_scale)")
    ap.add_argument("--getup_action_clip", type=float, default=8.0,
                    help="[--getup_action anchored] 动作幅度上限 C: "
                         "target = default_pose + scale*clip(a, +-C)。与训练时的 "
                         "getup2.action_clip 必须一致。默认 8.0 是**本项目实测值**: "
                         "Go1 的 thigh 行程 5.2 rad, 关键帧起身要 thigh=4.501, "
                         "而 C=3.0 (开源给 Go2 的值) 时关键帧只有 11.3%% 成功率、C=8.0 时 37.1%% "
                         "(sim/probe_v2_range.py)。用错值会静默把策略废掉")
    ap.add_argument("--getup_action_alpha", type=float, default=0.8,
                    help="[--getup_action anchored] 低通系数 "
                         "a_f = alpha*a + (1-alpha)*a_prev (与 "
                         "getup2.action_filter_alpha 一致)")
    ap.add_argument("--getup_residual_scale", type=float, default=0.15,
                    help="[--getup_action residual] ctrl = 关键帧 ctrl + a*scale "
                         "(与 envs/go1_getup_residual.py 的 residual_scale 一致)")
    ap.add_argument("--getup_up_th", type=float, default=0.95,
                    help="交还走路策略的**竖直度阈值**: up_z > 该值。\n"
                         "0.99 = 倾角 < 8.1 度 (旧硬编码, 实测根本达不到 -> 永远超时失败);\n"
                         "0.95 = 倾角 < 18.2 度 (默认, 与 sim/eval_getup.py --up_th 同口径)。\n"
                         "实测: 学习到的起身策略稳定站在 ~13 度倾角上, 用 0.99 会让\n"
                         "\"视觉上已经站起来了\"却一直判失败、拿不回走路权。")
    ap.add_argument("--getup_zlo", type=float, default=0.20,
                    help="交还走路策略的躯干**相对地面**高度下界 (站直约 0.277)")
    ap.add_argument("--getup_zhi", type=float, default=0.34,
                    help="交还走路策略的躯干相对地面高度上界")
    ap.add_argument("--getup_hold", type=int, default=25,
                    help="起身成功后还要**保持**多少帧才交还 (25 帧 = 0.5s, "
                         "防'刚站起来又被判摔倒'的乒乓)")
    ap.add_argument("--getup_timeout", type=int, default=400,
                    help="起身最多等多少帧 (400 帧 = 8s), 超时按失败处理")
    ap.add_argument("--getup_cooldown", type=int, default=50,
                    help="交还/重置后多少帧内不再触发起身 (25 帧 = 0.5s)")
    ap.add_argument("--fast", action="store_true",
                    help="不做实时节流, 全速跑 (机器人看起来会比真实速度快);"
                         " 默认按 env.dt 节流, 能跑满就是 50fps 实时")
    # --- 渲染质量 (WSL 无 /dev/dri, OpenGL 走 Mesa llvmpipe 软件渲染) ---
    ap.add_argument("--quality", choices=["low", "medium", "high"],
                    default=None,
                    help="渲染质量档。实测 (sim/diag_render_opt.py(已删除, 结论见实验记录)): "
                         "high(全开)=393ms/帧 2.5fps, medium(关阴影)=113ms 8.8fps, "
                         "low(关阴影+反射+天空盒)=70ms 14fps。"
                         "软件渲染下阴影贴图 (2 光源 × 4096²) 是最大头。"
                         "默认: 地形模式=low, 平地=medium (地形三角形多, low 收益最大)")
    ap.add_argument("--shadowsize", type=int, default=None,
                    help="覆盖阴影贴图尺寸 (默认 4096; 缩到 1024 省显存但实测"
                         "对 llvmpipe 帮助有限, 因为瓶颈是逐光源的深度渲染次数)")
    ap.add_argument("--overview", type=float, default=None, metavar="DIST",
                    help="俯瞰地形全貌: 相机固定在地形上方俯视 (DIST=相机距离, "
                         "米; 建议 7~9 可看全 ±6m 地形)。默认相机跟随机器人。"
                         "地形加大到 0.30m 高差后, 跟随视角看不到整体起伏与楼梯")
    args = ap.parse_args()

    print(f"[渲染] LP_NUM_THREADS={os.environ.get('LP_NUM_THREADS')} "
          f"(llvmpipe 线程限流, 见文件头; 4 为本机最优)", flush=True)

    t0 = time.time()
    policy_dims = None
    if args.pkl:
        print(f"加载策略 pickle: {args.pkl}", flush=True)
        policy, policy_dims = load_policy_pkl(args.pkl)
        src = args.pkl
    else:
        ckpt = args.ckpt or find_latest_ckpt()
        if ckpt is None:
            print("没有 checkpoint, 请先训练或指定 --pkl", flush=True)
            return 1
        ckpt = os.path.abspath(ckpt)  # orbax 要求绝对路径
        print(f"加载 checkpoint: {ckpt}", flush=True)
        policy, policy_dims = load_policy_ckpt(ckpt)
        src = ckpt
    if policy_dims:
        print(f"  策略 obs: state={policy_dims[0]} privileged={policy_dims[1]}",
              flush=True)

    # ---- 调度器方案: 第二个网络 (起身策略) ----
    # flat (默认): obs 必须与走路同形; history (方案 A): getup_v2 的 210 维历史帧,
    # 由查看器自己维护 -> 与走路不同形是**正确**的, 跳过同形检查。
    getup_policy = None
    if args.getup_pkl:
        if args.getup_obs == "history" and args.getup_action == "residual":
            print("[错误] --getup_obs history 与 --getup_action residual 不兼容 "
                  "(residual = 关键帧先验 + 同形 obs 残差)。", flush=True)
            return 1
        print(f"加载起身策略 pickle: {args.getup_pkl}", flush=True)
        getup_policy, getup_dims = load_policy_pkl(args.getup_pkl)
        if args.getup_obs == "history":
            # getup_v2: state=42x5=210 维历史帧, 与走路策略本来就不同形 ——
            # 查看器自己维护历史缓冲, 所以这里**跳过**同形检查 (不是错误)。
            need = GETUP_HIST * GETUP_FRAME_DIM
            print(f"  起身策略 obs={getup_dims[0]} (历史 {GETUP_HIST} 帧) "
                  f"privileged={getup_dims[1]} (与走路不同形, 跳过同形检查)",
                  flush=True)
            if getup_dims[0] != need:
                print(f"[警告] --getup_obs history 期望 state={need} "
                      f"({GETUP_FRAME_DIM}x{GETUP_HIST}), 但该策略是 "
                      f"{getup_dims[0]} 维; 请确认 --getup_pkl 是 getup_v2 的产物",
                      flush=True)
        else:
            if policy_dims is not None and tuple(getup_dims) != tuple(policy_dims):
                print(f"[错误] 起身策略 obs {tuple(getup_dims)} != 走路策略 "
                      f"{tuple(policy_dims)}。调度器方案要求两个策略吃**同一个** "
                      f"obs 向量 (见 实验记录 §29.10/29.11); 若这是 getup_v2 的 "
                      f"210 维历史策略, 请加 --getup_obs history。", flush=True)
                return 1
            print(f"  起身策略 obs: state={getup_dims[0]} "
                  f"privileged={getup_dims[1]} (与走路同形 ✅)", flush=True)

    if args.height_scan:
        args.terrain = True          # 高度扫描必须有 hfield
    # 按策略维度自动补齐环境开关 —— 目标是"训练环境原封不动", 不该要求用户
    # 记住每个版本要加哪些 flag:
    #   48 = 平地 v18;  83 = +高度扫描 (v19);  91 = +相位 (v20)
    # v17 起 (含 v19/v20) 训练用的是机体系指令 + 放宽的终止倾角, 这两项也必须
    # 与训练一致, 否则 obs 与终止判据都对不上, 行为会怪。
    if policy_dims:
        ns = policy_dims[0]
        if ns > 48 and not args.height_scan:
            print("  [自动] 策略 >48 维 -> 开启 --height_scan --terrain", flush=True)
            args.height_scan = True
            args.terrain = True
        if ns > 83:
            print("  [自动] 策略 >83 维 -> 开启相位奖励 (v20, obs +8 维)",
                  flush=True)
        if ns > 48 and not args.body_frame:
            print("  [自动] v19/v20 训练用机体系指令 -> 开启 --body_frame",
                  flush=True)
            args.body_frame = True
        if ns > 48 and args.tilt_deg is None:
            print("  [自动] v17+ 训练用 tilt_deg=17.5 -> 与训练一致", flush=True)
            args.tilt_deg = 17.5

    # 画质默认值必须在**自动检测之后**再定: 自动检测可能把 args.terrain 打开,
    # 而地形模式应默认 low 档。放在前面会导致 v19/v20/v21 策略静默用 medium
    # (实测: 只有 ~21 fps, 而 low 档 24-31 fps)。用户显式给 --quality 时不覆盖。
    if args.quality is None:
        args.quality = "low" if args.terrain else "medium"

    cfg = default_config()
    if args.hard_contact:
        cfg.foot_soft_contact = False
    if args.body_frame:
        cfg.command_config.frame = "body"
    if args.tilt_deg is not None:
        cfg.healthy_roll_range = args.tilt_deg
        cfg.healthy_pitch_range = args.tilt_deg
    if args.terrain:
        cfg.terrain = True
        # 地形模式默认随机出生 (与训练一致); --no_random_init 固定起点便于复现
        cfg.random_init.enable = not args.no_random_init
        if args.no_random_init:
            # 原点出生会落在 flatten_origin 的压平区, 与 v18 基线可直接对比
            pass
    if args.height_scan:
        cfg.height_scan.enable = True
    if args.push:
        cfg.push_config.enable = True
    if args.obs_noise:
        cfg.obs_noise.enable = True
        cfg.height_scan.noise = 0.0 if not args.height_scan else cfg.height_scan.noise
    # v20: 与训练一致 —— 开相位奖励时 obs 多 8 维 (cos,sin x 4足)。
    # 不设这个, 91 维策略会因 obs 只有 83 维而报 ScopeParamShapeError。
    if policy_dims and policy_dims[0] > 83:
        cfg.reward_config.scales.feet_phase = 2.0
    # 交互式查看器必须用 WARP_STAGED: MJX 默认的 WARP 模式按 buffer 地址缓存
    # CUDA graph, 而本循环每帧新建 command 数组、R 键还会重置 state (换地址)
    # -> 反复重新捕获 -> 跑几千帧后 "Warp error: unknown stream" (实测 frame
    # 2125)。实测 (sim/probe_graph_mode.py(已删除, 结论见实验记录)): WARP_STAGED 16.3ms/帧且 3000 帧
    # 不崩, WARP 是 185ms/帧。训练侧不受影响 (固定地址, 仍用默认 WARP)。
    cfg.graph_mode = args.graph_mode
    if args.graph_mode != "WARP":
        print(f"[warp] graph_mode={args.graph_mode} "
              f"(查看器用; 训练默认 WARP 在地址变动下会崩)", flush=True)
    env = Go1Walk(cfg)
    m = env.mj_model

    # 扫描点可视化的几何准备 (35 点)
    n_scan = env._scan_n if (env._hfield_ok and cfg.height_scan.enable) else 0
    show_scan = bool(args.scan_points) and n_scan > 0
    if args.scan_points and n_scan == 0:
        print("[提示] --scan_points 需要 --height_scan (v19/v20/v21 策略会自动开启); "
              "本次未启用扫描显示", flush=True)

    # 维度一致性检查 (不匹配会 ScopeParamShapeError, 但报错晦涩)
    if policy_dims is not None:
        env_dims = (env.observation_size["state"][0],
                    env.observation_size["privileged_state"][0])
        if tuple(policy_dims) != tuple(env_dims):
            print(f"[警告] 策略输入维度 {tuple(policy_dims)} != 环境 obs 维度 "
                  f"{tuple(env_dims)}。\n"
                  f"        v20(91)/v19(83) 需要 --height_scan --terrain；"
                  f"v18(48) 不需要。",
                  flush=True)
        else:
            print(f"[维度] 策略与环境一致: state={env_dims[0]} "
                  f"privileged={env_dims[1]}"
                  f"{'  (含相位 8 维)' if env_dims[0] > 83 else ''}", flush=True)

    # --- 渲染质量 (WSL 无 /dev/dri → OpenGL 落到 Mesa llvmpipe 软件渲染) ---
    # 实测 (sim/diag_render_opt.py(已删除, 结论见实验记录), 640x480 离屏): Go1 场景 393ms/帧 = 2.5fps,
    # 与分辨率无关 (320x240 也是 453ms) → 是逐光源阴影贴图的固定开销
    # (2 光源 × shadowsize 4096²), 不是像素填充。最小场景只要 9.6ms, 说明
    # llvmpipe 本身不慢。
    #   high   = 全开                     393ms → 2.5fps
    #   medium = 关阴影                   113ms → 8.8fps
    #   low    = 关阴影 + 反射 + 天空盒     70ms → 14fps
    # 阴影在 MjvScene.flags 上 (不在 model 上), 所以两条路径:
    #   - model 级: light_castshadow[:]=0 (启动前, 查看器共享同一 model 对象)
    #   - 场景级: v.user_scn.flags[...]=0 (启动后, 覆盖渲染线程用的场景)
    # 这里先用 model 级; 若查看器重建了场景, 循环里还会再兜一次 (见下)。
    _flags = mujoco.mjtRndFlag
    _off = []
    if args.quality in ("low", "medium"):
        m.light_castshadow[:] = 0
        _off.append("光源阴影")
    if args.shadowsize is not None:
        m.vis.quality.shadowsize = args.shadowsize
    print(f"[渲染] quality={args.quality}  关闭: "
          f"{'+'.join(_off) if _off else '无 (全开)'}  "
          f"shadowsize={m.vis.quality.shadowsize}", flush=True)

    reset_jit = jax.jit(env.reset)
    # donate_argnums=0: 输出复用输入 buffer → 地址稳定 → warp CUDA graph 缓存命中,
    # 单步 25.9ms → 14.8ms (见 sim/diag_viewer_perf.py(已删除, 结论见实验记录))。state 用完即弃, 捐赠安全。
    step_jit = jax.jit(env.step, donate_argnums=0)

    # ---- 关键帧起身 (2026-09-19, sim/getup_keyframe.py) ----
    # 摔倒不再自动重置: 交给 §15 的关键帧序列。动作语义 = ctrl 目标, 所以
    # 起身状态机输出的 12 维向量可以直接当 action 喂 step_jit。
    # 地面类型决定用哪套 kip 幅度 (plane 100%; hfield 明显更低, 见探针)。
    home_jp = jp.asarray(np.asarray(env._init_q)[7:], jp.float32)
    getup_jit = jax.jit(gk.fsm_step)
    hfield_floor = bool(getattr(env, "_hfield_ok", False))

    def ground_h(q):
        """躯干正下方的地面高度 (米); 平地 = 0。

        摔倒判据必须用它 —— 地形最高台 0.30m, 用绝对 z<0.15 在高台上永远不触发
        (旧代码就是这个问题: 地形模式下摔倒后从来自动重置不了)。
        """
        if not hfield_floor:
            return 0.0
        return float(env.terrain_height(jp.asarray(q[None, :2]))[0])

    def z_rel_of(q):
        """躯干相对地面的高度 (米) —— 摔倒判据与起身完成判据共用。"""
        return q[2] - ground_h(q)

    # 起身策略的推理 (obs 与走路同一个 key -> 调度器只需换 act 来源)
    getup_pol_jit = None
    if getup_policy is not None:
        getup_pol_jit = jax.jit(
            lambda obs: getup_policy(obs, jp.zeros((2,), jp.uint32))[0])

    # ---- getup_v2 (方案 A): 查看器自己维护 (5,42) 历史缓冲 ----
    # 单帧 42 维的**顺序必须**与 envs/go1_getup_v2.py::_frame 逐位一致 (否则策略
    # 看到的量与训练不一致):
    #   [ gyro(3), gravity(3), qpos[7:19]-default_pose(12), qvel[6:18](12),
    #     上一步原始动作(12) ] = 42
    # 拼平顺序同 env: hist.reshape(-1), 最新帧在最后。
    # 两个策略 obs 不同形, 所以这份历史缓冲与走路策略的 state 各走各的, 不共用。
    getup_hist_mode = bool(args.getup_obs == "history")
    # "新模式" 标志: 只用来给日志/overlay 加后缀, 保证旧路径 (flat +
    # incremental/absolute/residual) 的输出与改动前逐字一致。
    getup_v2_mode = bool(getup_hist_mode or args.getup_action == "anchored")
    getup_frame_jit = None
    if getup_hist_mode:
        _getup_def_pose = env._default_pose

        def _getup_frame(data, raw_act):
            """getup_v2 单帧 42 维 (与 Go1GetupV2._frame 同序)。"""
            return jp.concatenate([
                env.get_gyro(data),                 # 3  ang_vel_b (sensor "gyro")
                env.get_gravity(data),              # 3  projected_gravity_b
                data.qpos[7:19] - _getup_def_pose,  # 12 joint_pos - default
                data.qvel[6:18],                    # 12 joint_vel
                raw_act,                            # 12 上一步原始动作
            ])

        # jit 成一个 kernel: 逐条 eager 调用每帧 5+ 次 kernel launch (见本文件
        # scan 可视化的教训: eager 100 次调用 51.6ms -> jit 0.32ms)。
        getup_frame_jit = jax.jit(_getup_frame)

    print(f"  摔倒处理: {args.fall_action} | 起身方式="
          f"{'学到的策略' if getup_pol_jit is not None else '关键帧序列'}"
          + (f" (obs={args.getup_obs}"
             + ("=210 历史 5 帧" if getup_hist_mode else "")
             + f", action={args.getup_action})"
             if (getup_pol_jit is not None and getup_v2_mode) else "")
          + f" | 交还判据 up_z>{args.getup_up_th} 且 相对高度∈[{args.getup_zlo},"
          f"{args.getup_zhi}] 保持 {args.getup_hold} 帧", flush=True)

    print(f"[{time.time()-t0:5.1f}s] 环境就绪 (warp)  "
          f"frame={env._config.command_config.frame} "
          f"tilt=±{2*(args.tilt_deg if args.tilt_deg is not None else 10):.0f}°  "
          f"节流={'关(全速)' if args.fast else '实时'}", flush=True)

    d_view = mujoco.MjData(env.mj_model)

    command = np.array([args.cmd, 0.0, args.wz])
    reset_requested = False
    key_states = {}
    # 单发按键: 按一下跳一档 (W/S 调 vx, A/D 调 wz) + T 切扫描点显示。
    # on_press 里只认"松->按"的跳变 (滤 OS autorepeat), update_command 消费后清掉。
    tap_requests = set()
    TAP_KEYS = frozenset("wsad tg".replace(" ", ""))
    force_getup = [False]       # G 键: 手动触发一次起身
    scan_toggle = [True]         # T 键切换 (list 便于闭包内改); 给了 --scan_points 默认开

    def _norm(key):
        # 统一用小写字符串做键 (不用 KeyCode 实例: pynput 每次 press 事件
        # 可能给新实例, 不保证按值等价, autorepeat 判重会失效)。
        if isinstance(key, keyboard.KeyCode) and key.char is not None:
            return key.char.lower()
        return key

    def update_command():
        # W/S 调 vx; A/D 调 wz (A=左转/逆时针 wz>0, D=右转/顺时针 wz<0)
        # 按一下跳一档 0.2 (原来是按住每帧 +0.02 —— 帧率决定斜坡速度, 慢)。
        # on_press 只认"松->按"跳变, 所以按住不动不会连跳; autorepeat 的
        # 连发 press 也被同一跳变判据滤掉 (key_states 里已是 True)。
        step = 0.2
        if "w" in tap_requests:
            command[0] = min(command[0] + step, 1.5)
        if "s" in tap_requests:
            command[0] = max(command[0] - step, 0.0)
        if "a" in tap_requests:
            command[2] = min(command[2] + step, 1.5)
        if "d" in tap_requests:
            command[2] = max(command[2] - step, -1.5)
        if "t" in tap_requests:
            scan_toggle[0] = not scan_toggle[0]
            print(f"  扫描点显示: {'开' if scan_toggle[0] else '关'}", flush=True)
        if "g" in tap_requests:
            force_getup[0] = True      # 下一帧强制起身 (调试用)
        tap_requests.clear()

    def on_press(key):
        nonlocal reset_requested
        k = _norm(key)
        was_down = key_states.get(k, False)
        key_states[k] = True
        # 边沿触发: 只认"松->按"的跳变。OS autorepeat 会连发 press 事件,
        # 不滤掉的话"按一下"会变成连跳好几档 (速率取决于系统重复率)。
        if not was_down and k in TAP_KEYS:
            tap_requests.add(k)
        if k == "r":
            reset_requested = True

    def on_release(key):
        key_states[_norm(key)] = False

    keyboard.Listener(on_press=on_press, on_release=on_release).start()
    print("键盘: W/S 每按一下 vx±0.2 | A/D 每按一下 wz±0.2 (A=逆时针, D=顺时针)"
          " | R 重置 | G 手动起身"
          f" | 摔倒 -> {args.fall_action}"
          + (" | T 切换扫描点" if show_scan else ""), flush=True)

    # ---- 35 点高度扫描的可视化 ----
    # 与 env.height_scan_features 用**同一套变换** (否则画出来的点和策略看到的
    # 不一致): 网格只按 yaw 旋转 (不加 roll/pitch), 原点在躯干投影, 地面高度由
    # env.terrain_height 双线性插值取。
    #
    # **性能关键**: 采样点 xy / 地面高度 / 特征这三步必须一起 jit 成**一个**
    # kernel。第一版我按 viz_v19_perception.py 的写法在循环里逐条 eager 调用,
    # 每帧 100+ 次 kernel launch -> 实测 51.6 ms/帧 (jit 后 0.32ms, 差 160 倍),
    # 直接把查看器从 22ms 拖到 365ms (2 fps)。数值上两者一致 (最大差 4.8e-07)。
    _hs = cfg.height_scan
    _scan_xy = np.asarray(env._scan_xy)                 # (35,2) 机体系
    _scan_lim = max(_hs.clip_m * _hs.scale, 1e-6)
    _scan_xy_j = jp.asarray(_scan_xy, dtype=jp.float32)

    def scan_feat_rgba(f):
        """特征 -> 颜色。负(前方地面更高/上坡)=红, 0=绿, 正(下陷)=蓝。

        注: 原 viz_v19_perception.py 的写法 r=0.5(1-t), g=0.35, b=0.5(1+t) 在
        t=0 时给 (0.5,0.35,0.5) 灰紫, 与它自己"t=0 -> 绿"的注释不符 (已在此
        修正为真正的绿色中点)。红蓝两端的行为两者一致。
        """
        t = float(np.clip(f / _scan_lim, -1.0, 1.0))
        r = 0.15 + 0.75 * max(-t, 0.0)
        g = 0.25 + 0.55 * (1.0 - abs(t))
        b = 0.15 + 0.75 * max(t, 0.0)
        return np.array([r, g, b, 0.9], dtype=np.float32)

    @jax.jit
    def _scan_viz(data, qpos):
        """一次 kernel 算出 (采样点世界 xy, 地面高度, 高度特征)。"""
        feat = env.height_scan_features(data, rng=None)
        q = qpos[3:7] / jp.maximum(jp.linalg.norm(qpos[3:7]), 1e-8)
        w, x, y, z = q
        yaw = jp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        c, s = jp.cos(yaw), jp.sin(yaw)
        xy = jp.stack([
            c * _scan_xy_j[:, 0] - s * _scan_xy_j[:, 1] + qpos[0],
            s * _scan_xy_j[:, 0] + c * _scan_xy_j[:, 1] + qpos[1],
        ], axis=-1)
        return xy, env.terrain_height(xy), feat

    def draw_scan_points(scn, qpos, data):
        """把 35 个采样点画成小球 + 从躯干连射线。返回特征数组 (供 overlay)。"""
        xy, gz, feats = _scan_viz(data, qpos)
        pts_xy = np.asarray(xy)
        gz = np.asarray(gz)
        feats = np.asarray(feats)
        base = np.asarray(qpos[:3], dtype=float)
        ng = scn.maxgeom
        scn.ngeom = 0
        mat = np.eye(3).reshape(9)
        for i in range(n_scan):
            if scn.ngeom >= ng:
                break
            pos = np.array([pts_xy[i, 0], pts_xy[i, 1], gz[i]])
            mujoco.mjv_initGeom(
                scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([0.012, 0.0, 0.0]), pos, mat, scan_feat_rgba(feats[i]))
            scn.ngeom += 1
            if scn.ngeom < ng:      # 射线: 躯干 -> 采样点
                mujoco.mjv_initGeom(
                    scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_LINE,
                    np.zeros(3), np.zeros(3), mat,
                    np.array([0.7, 0.7, 0.2, 0.30], dtype=np.float32))
                mujoco.mjv_connector(
                    scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_LINE, 1.0,
                    np.array([base[0], base[1], base[2]]), pos)
                scn.ngeom += 1
        return feats

    # ---- JIT 预热: 在开窗口之前编译完 ----
    # 目的只有一个: 让窗口一出现就是可用的, 而不是在窗口里卡在首次编译上。
    #
    # 已知现象 (实测, 尚未完全消除): 窗口刚出现的头 ~100 帧偏慢 (step≈340ms),
    # 之后稳定 ~26ms/30fps。4575 帧/170s 的均值 26.8fps 与"前 100 帧慢 + 其后
    # 31fps"吻合。怀疑与 §15.4 同源: 真正 GL 绘制在查看器的 C++ 渲染线程里,
    # 它的 CPU 开销在帧计时里不可见 (render 只测 v.sync 的交接), 但会和 jax 的
    # kernel 分发线程抢核。`--warmup N` 可额外跑 N 帧 (默认 0 = 关; 实测开大
    # 反而更慢, 故不做默认)。
    print(f"[{time.time()-t0:5.1f}s] JIT 编译中 (完成后才开窗口)...", flush=True)
    seed = args.seed if args.seed >= 0 else int(time.time() * 1000) % (2**31 - 1)
    _rng0 = jax.random.PRNGKey(seed)
    _state0 = reset_jit(_rng0)
    _state0 = step_jit(_state0, jp.zeros(env.action_size))
    _state0 = jax.block_until_ready(_state0)
    if show_scan:
        jax.block_until_ready(_scan_viz(_state0.data, _state0.data.qpos))
    for _i in range(args.warmup):
        _state0.info["command"] = jp.array(command)
        _act, _ = policy(_state0.obs["state"], jp.zeros((2,), dtype=jp.uint32))
        _state0 = step_jit(_state0, _act)
    _state0 = jax.block_until_ready(_state0)
    print(f"[{time.time()-t0:5.1f}s] JIT 完成, 开窗口", flush=True)

    with mujoco.viewer.launch_passive(env.mj_model, d_view) as v:
        # low 档额外关反射/天空盒: 这两项只在运行时场景 (user_scn) 上可改,
        # model 侧没有对应字段。放在这里而不是循环里, 设一次即可。
        if args.quality == "low":
            v.user_scn.flags[_flags.mjRND_REFLECTION] = 0
            v.user_scn.flags[_flags.mjRND_SKYBOX] = 0
            v.user_scn.flags[_flags.mjRND_SHADOW] = 0
            print("[渲染] low: 已关场景级 阴影+反射+天空盒", flush=True)
        if args.overview is not None:
            # 俯瞰地形: 关掉跟随 (type=FREE 且固定 lookat), 相机拉到高处俯视。
            # 默认相机是 TRACKING (跟随机器人), 在 30cm 高差 + ±6m 地形下看不到
            # 整体起伏与楼梯位置。
            cam = v.cam
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = [0.0, 0.0, 0.25]
            cam.distance = float(args.overview)
            cam.azimuth = 135.0
            cam.elevation = -35.0
            print(f"[相机] 俯瞰模式: 距离={args.overview} 方位=135° 俯角=-35° "
                  f"(看地形全貌; 默认跟随机器人)", flush=True)
        # 预热已在开窗口前完成 (见上), 这里只建初态
        rng = jax.random.PRNGKey(seed)
        state = reset_jit(rng)
        state = jax.block_until_ready(state)
        state.info["command"] = jp.array(command)
        getup_active = [False]
        getup_fsm = None
        getup_frame0 = 0
        # getup_v2 (--getup_obs history) 的历史状态:
        #   getup_hist      = (5,42) 缓冲, None 表示下一帧用 5 份当前帧重建
        #   getup_a_prev    = 低通状态 a_prev (每次触发清零)
        #   getup_raw_prev  = 上一帧喂进 frame 的原始动作 (触发时清零)
        getup_hist = None
        getup_a_prev = jp.zeros(12)
        getup_raw_prev = jp.zeros(12)
        fall_count = [0]          # 摔倒判据连续满足帧数 (去抖)
        oob_seen = [False]        # 出界提示只打一次
        getup_hold_cnt = [0]      # 起身后"站住"连续帧数 (判交还)
        getup_best = [0.0]        # 本次起身到过的最大 up_z (诊断)
        getup_best_ok = [0.0]     # 高度带内到过的最大 up_z (诊断)
        last_getup_act = [None]   # 起身最后一帧的 ctrl 目标 (交还混合用)
        handover_from = [None]    # 交还混合的起点
        handover_left = [0]       # 交还混合剩余帧数
        cooldown = [0]            # 交接冷却, 防乒乓
        # 打印出生点与原地坡度: 固定 seed 会让每次启动落在同一个点, 必须看得见
        # (实测 seed=0 -> (-4.17,+2.81), 地面高 0.215m, 左右脚地面高差 3.5cm)。
        try:
            _q0 = np.asarray(state.data.qpos)
            _w, _x, _y, _z = _q0[3:7]
            _yaw = float(np.arctan2(2*(_w*_z + _x*_y), 1 - 2*(_y*_y + _z*_z)))
            _h0 = float(env.terrain_height(jp.asarray(_q0[None, :2]))[0])
            _py = np.array([-np.sin(_yaw), np.cos(_yaw)])
            _hl = float(env.terrain_height(jp.asarray((_q0[:2] + 0.25*_py)[None, :]))[0])
            _hr = float(env.terrain_height(jp.asarray((_q0[:2] - 0.25*_py)[None, :]))[0])
            print(f"[出生] seed={seed}  xy=({_q0[0]:+.2f}, {_q0[1]:+.2f})  "
                  f"地面高={_h0:.4f}m  横向坡度={100*(_hl-_hr)/0.5:+.1f}%  "
                  f"左右地面高差={_hl-_hr:+.4f}m", flush=True)
        except Exception as _e:
            print(f"[出生] 位置信息取不到: {type(_e).__name__}", flush=True)
        print(f"[{time.time()-t0:5.1f}s] 就绪, 开始", flush=True)

        ema = {"step": 0.0, "copy": 0.0, "render": 0.0, "total": 0.0,
               "scan": 0.0}
        alpha = 0.1
        frame_i = 0

        def _ema(k, x):
            ema[k] = x if ema[k] == 0.0 else (1.0 - alpha) * ema[k] + alpha * x

        while v.is_running():
            sim = v._get_sim()
            if sim is not None and not sim.run:
                v.sync()
                time.sleep(0.01)
                continue

            frame_t0 = time.perf_counter()
            update_command()
            if reset_requested:
                reset_requested = False
                rng, key = jax.random.split(rng)
                state = reset_jit(key)
                getup_active[0] = False

            # ---- 摔倒判定 (默认不再自动重置, 交给关键帧起身) ----
            q_now = np.asarray(state.data.qpos)
            if not getup_active[0] and cooldown[0] <= 0:
                if (abs(q_now[0]) > args.oob_reset_xy
                        or abs(q_now[1]) > args.oob_reset_xy):
                    # ---- 出界 = 走出地形边缘 -> **直接重置, 不要试图起身** (2026-09-22) ----
                    # 地形半边长 6.0m, 界外没有地面 -> 一直自由落体 (实测 |qvel| 每帧
                    # +0.196 ≈ 重力, 几十秒后 z_rel = -2558m)。若按摔倒去触发起身,
                    # 起身策略会在**半空中**挥腿 —— 这正是"抽搐"的一大来源; 而且还要
                    # 等 400 帧超时才重置。默认 --cmd 0.5 时机器人 8~16s 就走到边缘。
                    rng, key = jax.random.split(rng)
                    state = reset_jit(key)
                    q_now = np.asarray(state.data.qpos)
                    fall_count[0] = 0
                    cooldown[0] = args.getup_cooldown
                    if not oob_seen[0]:
                        oob_seen[0] = True
                        print(f"   [出界] |xy| > {args.oob_reset_xy:.1f}m "
                              f"(地形半边长 6.0m, 界外无地面) -> 直接重置, "
                              f"不试图在半空起身", flush=True)
                upz_now = gk.up_z_from_quat(q_now[3:7])
                z_rel_now = z_rel_of(q_now)
                fallen = (z_rel_now < args.fall_z) or (upz_now < args.fall_uz)
                fall_count[0] = fall_count[0] + 1 if fallen else 0
                armed = fall_count[0] >= args.fall_debounce
                if force_getup[0] or (armed and args.fall_action != "reset"):
                    fall_count[0] = 0
                    getup_active[0] = True
                    getup_hold_cnt[0] = 0
                    getup_best[0] = 0.0
                    getup_best_ok[0] = 0.0
                    getup_frame0 = frame_i
                    # getup_v2: 历史缓冲下一帧重建; 低通 a_prev 与上一帧原始动作
                    # 每次触发都清零 (与 env reset/restart 时清零同义)
                    getup_hist = None
                    getup_a_prev = jp.zeros(12)
                    getup_raw_prev = jp.zeros(12)
                    # residual 模式要**同时**跑关键帧与策略, 所以 FSM 一律初始化
                    getup_fsm = gk.init_fsm(
                        hfield=hfield_floor,
                        max_attempts=args.getup_attempts)
                    if getup_pol_jit is None:
                        how = (f"关键帧 方案表="
                               f"{'hfield' if hfield_floor else 'plane'} "
                               f"x{args.getup_attempts}")
                    elif args.getup_action == "residual":
                        how = (f"关键帧先验 + 学习残差 "
                               f"(scale={args.getup_residual_scale})")
                    else:
                        how = "学到的策略"
                    print(f"   [起身] 触发 (up_z={upz_now:+.2f} "
                          f"离地={z_rel_now:.3f}m) -> {how}"
                          + (f" [obs={args.getup_obs}"
                             f", action={args.getup_action}]"
                             if (getup_pol_jit is not None and getup_v2_mode)
                             else ""), flush=True)
                elif armed:
                    fall_count[0] = 0
                    rng, key = jax.random.split(rng)
                    state = reset_jit(key)
                    cooldown[0] = args.getup_cooldown
            force_getup[0] = False

            state.info["command"] = jp.array(command)
            st_now = jp.int32(0)
            if getup_active[0]:
                last_getup_act[0] = None
                if getup_pol_jit is not None and args.getup_action == "residual":
                    # 与 envs/go1_getup_residual.py 的 step 同一个变换:
                    #   ctrl = clip(ctrl_fsm + a*residual_scale); 进 P4 时做关节手术
                    # env 是 Go1Walk (绝对 ctrl 语义), 所以这里不需要再变换。
                    res = getup_pol_jit(state.obs["state"])
                    ctrl_fsm, st_now, rj_j, getup_fsm = getup_jit(
                        getup_fsm, state.data.qpos, state.data.qvel, home_jp)
                    act = ctrl_fsm + res * args.getup_residual_scale
                    if bool(rj_j):
                        state = gk.apply_joint_reset(state, home_jp)
                elif getup_pol_jit is not None:
                    # getup_v2: 查看器自己维护历史缓冲 -> 拼出 210 维喂策略,
                    # 不复用走路策略那份 state (两者本来就不同形)。
                    if getup_hist_mode:
                        frame = getup_frame_jit(state.data, getup_raw_prev)
                        if getup_hist is None:
                            # 触发那一帧: 5 份当前帧的拷贝 (同 env reset 的
                            # jp.broadcast_to(frame, (H, 42)))
                            getup_hist = jp.broadcast_to(
                                frame, (GETUP_HIST, GETUP_FRAME_DIM))
                        else:
                            # 之后每帧: 丢最旧, 最新帧追加在**最后**
                            getup_hist = jp.concatenate(
                                [getup_hist[1:], frame[None, :]], axis=0)
                        getup_obs = getup_hist.reshape(-1)   # = hist.reshape(-1)
                    else:
                        getup_obs = state.obs["state"]
                    a_raw = getup_pol_jit(getup_obs)
                    if args.getup_action == "anchored":
                        # 与 envs/go1_getup_v2.py::step 逐行同款, **绝对 ctrl 语义**:
                        #   a   = clip(a_raw, ±action_clip)
                        #   a_f = alpha*a + (1-alpha)*a_prev        (低通)
                        #   act = default_pose + action_scale*a_f
                        # 注意: 这里**不做**旧 incremental 的 qpos[7:]+act*scale 变换。
                        a = jp.clip(a_raw, -args.getup_action_clip,
                                    args.getup_action_clip)
                        a_f = (args.getup_action_alpha * a
                               + (1.0 - args.getup_action_alpha) * getup_a_prev)
                        getup_a_prev = a_f
                        getup_raw_prev = a          # env 存的是裁剪后的 raw
                        act = (env._default_pose
                               + args.getup_action_scale * a_f)
                    elif args.getup_action == "incremental":
                        # 与 envs/go1_getup.py 的 step 完全同一个变换, 否则
                        # 查看器里的行为与训练不一致 (动作被放大/缩小一个量级)
                        act = (state.data.qpos[7:]
                               + a_raw * args.getup_action_scale)
                        getup_raw_prev = a_raw
                    else:                           # absolute: 策略输出即 ctrl
                        act = a_raw
                        getup_raw_prev = a_raw
                else:
                    act, st_now, rj_j, getup_fsm = getup_jit(
                        getup_fsm, state.data.qpos, state.data.qvel, home_jp)
                    if bool(rj_j):
                        state = gk.apply_joint_reset(state, home_jp)
                last_getup_act[0] = np.asarray(act).copy()
            else:
                act, _ = policy(state.obs["state"],
                                jp.zeros((2,), dtype=jp.uint32))
                # 默认关 (blend=0): 直接交还。开的时候是"从起身最后的目标线性过渡到
                # 走路策略的目标", 实测会显著加重抽搐/摔倒 (见 --handover_blend 的 help)。
                if handover_left[0] > 0 and handover_from[0] is not None:
                    k = 1.0 - handover_left[0] / float(args.handover_blend)
                    act = handover_from[0] * (1.0 - k) + act * k
                    handover_left[0] -= 1
            state = step_jit(state, act)

            # 只搬渲染需要的 qpos/qvel: mjx.get_data 会搬整个 Data (含 naconmax
            # =65536 的接触数组), 实测 1.13ms vs 0.02ms。np.asarray 同时完成
            # 设备同步, 所以上面的 step 耗时在这里才真正结算。
            qpos = np.asarray(state.data.qpos)
            qvel = np.asarray(state.data.qvel)
            t_step = (time.perf_counter() - frame_t0) * 1000.0

            t1 = time.perf_counter()
            d_view.qpos[:] = qpos
            d_view.qvel[:] = qvel
            mujoco.mj_forward(env.mj_model, d_view)
            t_copy = (time.perf_counter() - t1) * 1000.0

            # ---- 35 点扫描可视化 (T 键开关) ----
            # 放在 mj_forward 之后: 采样点位置需要地面高度, 特征需要 state.data
            scan_feats = None
            if show_scan and scan_toggle[0]:
                t_s = time.perf_counter()
                scan_feats = draw_scan_points(v.user_scn, qpos, state.data)
                _ema("scan", (time.perf_counter() - t_s) * 1000.0)
            elif show_scan:
                v.user_scn.ngeom = 0      # 关掉时清空自绘几何

            # ---- 起身收尾 ----
            if getup_active[0] and getup_pol_jit is not None:
                # 学到的策略: 必须"站住"才交还 —— up_z 且**相对高度**且保持 N 帧。
                # 只看 up_z 不够: 起身过程中会经过完全直立的中间姿态 (实测关键帧
                # 方案在 P3a 阶段 up_z=1.0000 而离地只有 5.9cm, 见 §29.10)。
                upz = gk.up_z_from_quat(qpos[3:7])
                zr = z_rel_of(qpos)
                ok = ((upz > args.getup_up_th)
                      and (args.getup_zlo <= zr <= args.getup_zhi))
                # 记录本次起身"最接近成功"的一帧, 失败时打出来 —— 否则用户只看到\n                # "失败", 完全不知道差在哪 (这正是"看着站起来了却报失败"的现场)。
                if (upz > getup_best[0]) or (getup_best[0] == 0.0):
                    getup_best[0] = upz
                if (args.getup_zlo <= zr <= args.getup_zhi) and upz > getup_best_ok[0]:
                    getup_best_ok[0] = upz
                # 容错计数: 满足 +1, 不满足 **-1 (不归零)** (2026-09-22, 有实测)
                # 实测 (sim/probe_handover.py --stage realfall, 从走路真摔倒的 128 个
                # 样本): 严格"连续 25 帧" 77.3%; 容错计数 81.2%, 且到站 5.71s -> 4.89s。
                # 失败形态里 26/29 是"其实已经站进高度带、但中间掉出一两帧就归零" ——
                # 这正是用户说的"视觉上站起来了却提示没成功"。放宽的代价很小: 交还后的
                # 稳定性几乎不受终态质量影响 (§29.27: 7.9% vs 走路原生基线 8.4%)。
                if ok:
                    getup_hold_cnt[0] += 1
                else:
                    getup_hold_cnt[0] = max(getup_hold_cnt[0] - 1, 0)
                if getup_hold_cnt[0] >= args.getup_hold:
                    getup_active[0] = False
                    cooldown[0] = args.getup_cooldown
                    # ---- 交还 (2026-09-21 用 sim/probe_handover.py 的实测把这里定下来) ----
                    # 上一轮的两条"修复"都被数据否掉了, 这里**回退**:
                    #  1) 曾经把 info["last_act"] 清零。清零 = 告诉走路策略"我上一步
                    #     把 12 个关节都命令到了 0" —— 这同样是假话 (真值是 default±4),
                    #     实测 摔倒率 8.4% vs 保留真值 6.6% (227 终态, 噪声内但更差)。
                    #     保留真值 = "什么都不做", 也就是回退到这轮改动之前。
                    #  2) 曾经默认 --handover_blend 10。起身最后的目标偏离 2.7 rad 且
                    #     顶在裁剪上, 从它开始混合会把机器人往前推: blend10 -> 37.0%、
                    #     blend30 -> 73.1% 摔倒。所以默认改成 0 (直接交还)。
                    if args.handover_blend > 0 and last_getup_act[0] is not None:
                        handover_from[0] = jp.asarray(last_getup_act[0])
                        handover_left[0] = args.handover_blend
                    print(f"   [起身] 成功 ✅ 站住 {getup_hold_cnt[0]} 帧 "
                          f"(up_z={upz:+.3f} 离地={zr:.3f}m), 交还走路策略 "
                          f"[{frame_i - getup_frame0} 帧]", flush=True)
                elif (frame_i - getup_frame0) >= args.getup_timeout:
                    getup_active[0] = False
                    cooldown[0] = args.getup_cooldown
                    print(f"   [起身] 失败 ❌ 超时 {args.getup_timeout} 帧 "
                          f"(当前 up_z={upz:+.3f} 离地={zr:.3f}m; "
                          f"最好 up_z={getup_best[0]:+.3f}, "
                          f"高度带内最好 up_z={getup_best_ok[0]:+.3f} "
                          f"阈值 {args.getup_up_th})", flush=True)
                    if args.fall_action == "getup":
                        rng, key = jax.random.split(rng)
                        state = reset_jit(key)
                        print("   [起身] 回退到重置", flush=True)
            elif getup_active[0] and int(st_now) != 0:
                ok = int(st_now) == gk.STATUS_OK
                getup_active[0] = False
                cooldown[0] = args.getup_cooldown
                print(f"   [起身] {'成功 ✅' if ok else '失败 ❌'} "
                      f"({gk.phase_name(getup_fsm)}, 用了 "
                      f"{frame_i - getup_frame0} 帧)", flush=True)
                if not ok and args.fall_action == "getup":
                    rng, key = jax.random.split(rng)
                    state = reset_jit(key)
                    print("   [起身] 回退到重置", flush=True)
            if cooldown[0] > 0:
                cooldown[0] -= 1

            t2 = time.perf_counter()
            v.sync()
            t_render = (time.perf_counter() - t2) * 1000.0
            _ema("step", t_step)
            _ema("copy", t_copy)
            _ema("render", t_render)
            _ema("total", (time.perf_counter() - frame_t0) * 1000.0)
            frame_i += 1
            if frame_i % 25 == 0:
                print(f"[viewer] frame {frame_i}: "
                      f"fps={1000.0/max(ema['total'], 1e-6):.0f}  "
                      f"step={ema['step']:.1f} copy={ema['copy']:.1f} "
                      f"render={ema['render']:.1f} "
                      f"scan={ema['scan']:.1f} "
                      f"total={ema['total']:.1f} ms", flush=True)

            v.overlay = {
                "title": f"Go1 复现 ({os.path.basename(src.rstrip('/'))})",
                "command": f"指令 vx={command[0]:+.2f} vy={command[1]:+.2f} "
                           f"wz={command[2]:+.2f}",
                "speed": f"实际 vx={qvel[0]:+.2f} vy={qvel[1]:+.2f} "
                         f"wz={qvel[5]:+.2f} z={qpos[2]:.2f}",
                "perf": f"fps={1000.0/max(ema['total'], 1e-6):.0f}  "
                        f"step={ema['step']:.1f}ms copy={ema['copy']:.1f}ms "
                        f"render={ema['render']:.1f}ms"
                        + (f" scan={ema['scan']:.1f}ms" if show_scan else ""),
                "help": ("W/S vx±0.2 | A/D wz±0.2 (按一下跳一档) | R 重置"
                         " | G 手动起身"
                         + (" | T 扫描点" if show_scan else "")),
            }
            if getup_active[0]:
                if getup_pol_jit is not None:
                    v.overlay["getup"] = (
                        f"起身中 (策略"
                        + (f" {args.getup_action}"
                           f"{'/' + args.getup_obs if getup_hist_mode else ''}"
                           if getup_v2_mode else "")
                        + f"): 站住 {getup_hold_cnt[0]}"
                        f"/{args.getup_hold} 帧, {frame_i - getup_frame0} 帧")
                else:
                    v.overlay["getup"] = (
                        f"起身中: {gk.phase_name(getup_fsm)} "
                        f"(第 {int(np.asarray(getup_fsm['attempt'])) + 1} 个方案, "
                        f"{frame_i - getup_frame0} 帧)")
            if scan_feats is not None:
                v.overlay["scan"] = (
                    f"高度特征({n_scan}点): min={scan_feats.min():+.2f} "
                    f"max={scan_feats.max():+.2f} mean={scan_feats.mean():+.2f} "
                    f"(红=前方更高/上坡, 蓝=下陷)")

            # 实时节流: 只补足到 env.dt, 慢帧不额外叠加 sleep
            if not args.fast:
                slack = env.dt - (time.perf_counter() - frame_t0)
                if slack > 0:
                    time.sleep(slack)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())