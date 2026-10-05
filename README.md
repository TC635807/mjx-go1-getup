[English](README.en.md) | **简体中文**

# MJX Go1 Get-Up

**基于 MJX（MuJoCo Warp）+ Brax PPO 的 Unitree Go1 地形行走与学习式摔倒起身** —— 在**单张 8 GB
笔记本 GPU** 上端到端训练。不用 Isaac，不用集群，不用动捕参考。

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![JAX](https://img.shields.io/badge/JAX-0.7.2-9B30FF)
![MuJoCo MJX](https://img.shields.io/badge/MuJoCo%20MJX-3.12-1f6feb)
![Brax](https://img.shields.io/badge/Brax-0.14.2-orange)
![License](https://img.shields.io/badge/license-MIT-green)

![走路策略在程序化地形上](docs/assets/walk-terrain.png)

---

## 仓库内容

三个部分，跑的是**同一个** MuJoCo 模型、50 Hz、全部在 MJX 上：

| 组件 | 做什么 | 观测 | 动作语义 |
|---|---|---|---|
| **走路策略** — [`envs/go1_walk.py`](envs/go1_walk.py) | 速度跟踪 trot，跑在程序化高度场地形上（起伏、台地、两条楼梯） | 91 维 | 12 个绝对关节位置目标 |
| **起身策略** — [`envs/go1_getup_v2.py`](envs/go1_getup_v2.py) | 从**任意摔倒姿态**站起，包括走路策略"真摔倒"后实际落在的那些姿态 | 5 × 42 = 210 维（历史帧） | 锚定式：`target = home_pose + 0.5·clip(a, ±8)` |
| **调度器 / 查看器** — [`sim/view_go1.py`](sim/view_go1.py) | 去抖摔倒检测 → 交给起身策略 → 站住后交还走路策略 | — | — |

重点不是"策略能站起来"，而是**整条链路都被离线探针量化过**：摔倒多久被检出、起身成功几次、
花了多久，以及控制权交还给走路策略之后的**那两秒**会发生什么。

## 结果

下表中每个数字都能用本仓库的脚本复现（命令见"快速开始"），原始开发日志在
[`docs/experiment-log.md`](docs/experiment-log.md)。

| 指标 | 数值 | 测法 |
|---|---|---|
| **走路真摔倒 → 起身成功率** | **77.3 %**（严格）/ **81.2 %**（容错），n = 128 | [`sim/probe_handover.py`](sim/probe_handover.py) `--stage realfall` |
| 到站耗时 | 5.71 s → **4.89 s**（容错判据） | 同上 |
| 交还瞬间的摔倒率 | **7.9 %**；10 帧动作混合 38.3 %、30 帧 73.6 % | `--stage cycle`，227 个起身终态 |
| 走路策略（60 M 步） | best `eval_reward` 2339.7，平均 episode 670 / 750 步（13.4 s），30 s 长测 0 摔 | [`train/train_go1.py`](train/train_go1.py) eval |
| 台阶 | 下台阶 62–87 %，**上台阶 0 %** —— 未解决 | [`sim/eval_stairs.py`](sim/eval_stairs.py) |
| 训练吞吐 | 起身 **22–25 k 步/s**（40 M ≈ 30 分钟）；走路 ≈ 4.4 k 步/s（60 M ≈ 4 小时） | RTX 5060 Laptop 8 GB, WSL2 |

### 三条花了最久才接受的结论

1. **瓶颈是判据，不是姿态。** 起身策略其实相当可靠地到达了站姿（高度带内最好的 `up_z` 中位数
   0.999），真正卡住的是旧判据"必须**连续** 25 帧满足"。把计数器改成容错（满足 +1 / 不满足 −1，
   而不是清零）后，成功率 77.3 % → 81.2 %，到站时间少 0.8 s。**`up_z` 阈值取多少会让结论差一个
   数量级**（0.99 → 倾角 8.1° ≈ 0 %；0.95 → 18.2° ≈ 77 %），定义见
   [`docs/status.md`](docs/status.md) §3。
2. **把两个策略的动作做混合是有害的。** 起身策略是靠**顶着动作裁剪**站住的（median |a| = 8.0 =
   clip）。把它的末态目标线性混合进走路策略的目标、持续 10~30 帧，等于先把人推倒：摔倒率
   7.9 % → 38.3 %（10 帧）→ 73.6 %（30 帧）。**硬切换**才对。
3. **动作空间锚定比奖励塑形更关键。** 起身任务最大的一次跃迁，是把 `target = qpos + 0.5·a`
   （无界，策略要从零学关节角本身）换成 `target = home_pose + 0.5·clip(a, ±8)` —— 也就是开源项目
   [get-up-isaaclab](https://github.com/iit-DLSLab/get-up-isaaclab) 的配方；再配上**只有两个宽高斯**
   的奖励和**完全不做姿态终止**，任务才从 0 % 走到 ~80 %。与公开实现（HoST、HumanUP、AFR…）的
   逐条对照见 [`docs/getup-recipes.md`](docs/getup-recipes.md)。

## 安装

```bash
git clone https://github.com/TC635807/mjx-go1-getup.git
cd mjx-go1-getup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

验证环境：Python 3.14、JAX 0.7.2、MuJoCo 3.12、Brax 0.14.2、`warp-lang` 1.16，RTX 5060 Laptop
（8 GB）、WSL2。训练需要 CUDA-12 GPU；查看器强烈建议也用 GPU。几个必须注意的环境变量
（`XLA_PYTHON_CLIENT_MEM_FRACTION`、`LP_NUM_THREADS`、`GALLIUM_DRIVER`）和我们踩过的所有
WSL 渲染坑，都写在 [`docs/viewer-guide.md`](docs/viewer-guide.md) 里。

## 快速开始

**1. 看它走路、摔倒、再爬起来**（交互式 MuJoCo 查看器）：

```bash
python sim/view_go1.py \
    --pkl policies/go1_walk_policy.pkl \
    --getup_pkl policies/go1_getup_policy.pkl \
    --getup_obs history --getup_action anchored --cmd 0
```

`W/S` 调前进速度指令，`A/D` 转向，`R` 重置，`Q` 退出。`--cmd 0` 时机器人原地不动，是看起身
最干净的方式；默认的 `--cmd 0.5` 会在 8~16 s 内走出 6 m 地形边缘（出界会被重置，而不是在半空
乱蹬）。

**2. 离线评估起身成功率**（不渲染，256 个并行环境）：

```bash
python sim/eval_getup.py --task getup_v2 --getup2_action_clip 8.0 \
    --up_th 0.95 --z_lo 0.20 --z_hi 0.34 \
    --pkl policies/go1_getup_policy.pkl --envs 256 --steps 400
```

> ⚠️ `--getup2_action_clip` **必须与训练一致**（8.0）。用配置默认值（3.0）去评估发布策略会把动作
> 夹掉，评出假的 0 %。

**3. 复现头条数字** —— 一直走到真摔倒，再交给起身策略：

```bash
python sim/probe_handover.py --stage realfall --envs 128 --steps 1500 \
    --getup_steps 600 --cmd 0.5 --bound_xy 5.0
```

**4. 从零训练**：

```bash
# 起身策略：40 M 步，RTX 5060 Laptop 上约 30 分钟
python -u -m train.train_getup --task getup_v2 --action_clip 8.0 \
    --num_timesteps 40000000 --num_evals 16 --episode_length 400 \
    --num_minibatches 4 --updates_per_batch 5 --lr 5e-4 --entropy 0.005 --init_noise_std 1.0

# 走路策略：60 M 步
python -u -m train.train_go1 --num_timesteps 60000000
```

检查点写到 `logs/`，策略写到 `policies/`。两个不花 GPU 时间的自检：
`python -m train.train_getup --dry_run`（只建环境 + 建网络）和 `python sim/probe_getup_v2.py`
（起身环境自检 —— 改起身环境之前先跑它）。

## 目录结构

```
mjx-go1-getup/
├── envs/                     # MJX 环境（物理 + 任务定义）
│   ├── go1_walk.py           #   走路：地形、奖励、终止、91 维 obs
│   ├── go1_getup.py          #   起身 v1：增量动作（保留作对照）
│   ├── go1_getup_residual.py #   起身：关键帧状态机 + 学习残差
│   └── go1_getup_v2.py       #   * 起身 v2：锚定动作 + 宽高斯 + 历史帧
├── train/
│   ├── train_go1.py          # 走路训练（brax PPO）
│   └── train_getup.py        # 起身训练：--task getup | getup_res | getup_v2
├── sim/
│   ├── view_go1.py           # * 交互式查看器 + 走路/起身调度器
│   ├── watch_v20_fast.py     #   高帧率观测脚本（见 docs/viewer-guide.md）
│   ├── eval_getup.py         #   起身成功率 / 分姿态桶 / 到站时间
│   ├── eval_walk.py          #   走路评估（固定出生点集，可比）
│   ├── eval_stairs.py        #   上下台阶成功率（固定摆放）
│   ├── probe_handover.py     # * 交还瞬间的定量探针
│   ├── probe_getup_v2.py     #   起身 v2 环境自检
│   ├── probe_standability.py #   判据本身到底可不可达？
│   ├── probe_nefc.py         #   接触约束溢出探针（见"已知问题"）
│   ├── gen_terrain.py        #   确定性地形生成器（hfield PNG + json）
│   ├── getup_keyframe.py     #   脚本化关键帧起身（基线 + 查看器兜底）
│   └── common.py             #   共用的策略加载（pkl / orbax checkpoint）
├── models/go1/               # MuJoCo 模型（MJCF + 网格 + 地形）
├── policies/                 # 发布策略（走路 v23@60M、起身 v2f）
└── docs/                     # 实验记录、状态、查看器指南、起身配方调研
```

## 文档

| 文档 | 内容 |
|---|---|
| [`docs/status.md`](docs/status.md) | **先读这个** —— 当前结果，以及所有判据（摔倒检测、站住、交还）的精确定义 |
| [`docs/experiment-log.md`](docs/experiment-log.md) | 完整开发日志（§1–§29.28）：每一次失败尝试，和否掉它的那个实测数字 |
| [`docs/getup-recipes.md`](docs/getup-recipes.md) | 与开源起身实现逐条对照（get-up-isaaclab、HoST、HumanUP、AFR）+ SOTA 调研 |
| [`docs/viewer-guide.md`](docs/viewer-guide.md) | 怎么写一个快的 MJX 查看器：逐帧耗时拆解、JIT 陷阱、WSL 渲染 |

代码、docstring 与本文档均以中文为主，英文版概览见 [README.en.md](README.en.md)。

## 已知问题与路线图

* **`njmax=256` 对起身任务太小。** 在高度场地形上摔倒时单个 world 实测需要 `nefc ≈ 1300`，
  训练中会打 `nefc overflow` 告警（719 次 / 384 k world-帧）。用 `njmax=2048` 重训一版是待办
  —— 它不解释观察到的抽搐，但会影响"躺在地上"时的接触精度。可用
  [`sim/probe_nefc.py`](sim/probe_nefc.py) 复现。
* **起身策略是"顶着动作裁剪"站住的**（median |action| = 8.0 = clip，关节还在 11.5 rad/s 上动）。
  奖励里加一项"站住时要安静"、或者让成功奖金要求姿态静止，是下一步的质量提升。
* **训推分布差 ~11 个点**：训练出生姿态 88.7 % vs 走路真摔倒 77.3 %。计划用真摔倒姿态池做微调
  （[`sim/make_getup_posepool.py`](sim/make_getup_posepool.py)）。
* **上台阶未解决**（固定楼梯上 0 % 成功率），走路策略的镜像对称性也不完美（hip RR/RL 偏差 ≈ 16°）。

## 致谢

* [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) 与
  [Brax](https://github.com/google/brax) —— 本项目的训练栈（环境 API、PPO、MJX/warp 后端）。
* [get-up-isaaclab](https://github.com/iit-DLSLab/get-up-isaaclab)（IIT-DLSLab）—— `go1_getup_v2.py`
  逐条照抄的起身配方；[HoST](https://github.com/InternRobotics/HoST) 与
  [HumanUP](https://github.com/RunpeiDong/HumanUP) —— 多 critic / 两阶段课程，影响了本项目的路线图。
* [quadruped-rl-locomotion](https://github.com/nimazareian/quadruped-rl-locomotion) —— 走路任务复刻自此
  仓库，之后移植到 MJX。
* Unitree Go1 的 MJCF 来自 [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)。

## License

[MIT](LICENSE)
