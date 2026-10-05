[English](README.en.md) | 简体中文

# MJX Go1 Get-Up

Unitree Go1 的行走与起身策略。物理用 MJX（MuJoCo Warp），训练用 Brax PPO，全流程在一张
RTX 5060 Laptop（8 GB）上跑通。

![走路策略在程序化地形上](docs/assets/walk-terrain.png)

## 仓库内容

三个部分跑的是同一个 MuJoCo 模型，控制频率 50 Hz：

- **走路策略** `envs/go1_walk.py`：速度跟踪 trot。地形是程序生成的 hfield，包含起伏、台地和
  两条楼梯。观测 91 维，动作是 12 个绝对关节位置目标。
- **起身策略** `envs/go1_getup_v2.py`：从摔倒姿态站起。观测是 5 帧 × 42 维的历史，
  动作是 `target = home_pose + 0.5 * clip(a, ±8)`。
- **调度器** `sim/view_go1.py`：检测到摔倒后切到起身策略，站起来保持 0.5 s 再把控制权交回
  走路策略。

## 结果

| 指标 | 数值 | 复现脚本 |
|---|---|---|
| 走路真摔倒后的起身成功率 | 77.3%（严格）/ 81.2%（容错），n = 128 | `sim/probe_handover.py --stage realfall` |
| 到站时间 | 5.71 s → 4.89 s（容错判据） | 同上 |
| 交还后的摔倒率 | 7.9%（10 帧动作混合 38.3%，30 帧 73.6%） | `sim/probe_handover.py --stage cycle` |
| 走路策略（60M 步） | eval_reward 2339.7，平均 episode 670/750 步 | `train/train_go1.py` 的训练日志 |
| 台阶 | 下台阶 62~87%，上台阶 0% | `sim/eval_stairs.py` |
| 训练速度 | 起身 22k~25k 步/s（40M 步约 30 分钟）；走路约 4.4k 步/s | RTX 5060 Laptop 8 GB |

这些数字都是在上面几个脚本里跑出来的，命令见「使用方法」。起身的判据定义（`up_z` 阈值、
连续多少帧算站住）在 `docs/status.md`，换个阈值数字会差很多，看数之前先确认口径。

## 几点经验

- **起身成功率主要取决于判据，不是策略。** 策略基本都能站到位（高度带内最好的 `up_z` 中位数
  0.999），卡住的是「必须连续 25 帧满足」。把计数器改成满足加一、不满足减一之后，成功率
  77.3% → 81.2%，到站时间少 0.8 s。另外 `up_z` 取 0.99（倾角 8.1°）还是 0.95（18.2°），
  成功率差一个数量级。
- **交还时不要混合两个策略的动作。** 起身策略是靠顶住动作上限站住的，`|a|` 的中位数正好等于
  裁剪值 8.0。把它的末态目标和走路策略的目标按比例混合，前 10 帧摔倒率就升到 38.3%，30 帧
  升到 73.6%；直接切换是 7.9%。
- **动作空间用锚定式比增量式好训。** 增量式 `target = qpos + 0.5 * a` 要求策略自己学关节角，
  换成 `target = home_pose + 0.5 * clip(a, ±8)`（get-up-isaaclab 的写法），配上两个宽高斯奖励、
  不做姿态终止，这个任务才从 0% 做到 80% 左右。

## 安装

```bash
git clone https://github.com/TC635807/mjx-go1-getup.git
cd mjx-go1-getup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

开发环境是 Python 3.14 + JAX 0.7.2 + MuJoCo 3.12 + Brax 0.14.2 + warp-lang 1.16，训练需要
CUDA 12 的 GPU。几个环境变量（`XLA_PYTHON_CLIENT_MEM_FRACTION`、`LP_NUM_THREADS`、
`GALLIUM_DRIVER`）和 WSL 下的渲染问题写在 `docs/viewer-guide.md`。

## 使用方法

### 查看器

```bash
python sim/view_go1.py \
    --pkl policies/go1_walk_policy.pkl \
    --getup_pkl policies/go1_getup_policy.pkl \
    --getup_obs history --getup_action anchored --cmd 0
```

W/S 调前进速度，A/D 转向，R 重置，Q 退出。`--cmd 0` 时机器人原地走，看起身最清楚；默认的
`--cmd 0.5` 会在 8~16 s 内走到地形边缘（出界会重置，不会在半空触发起身）。

### 评估起身

```bash
python sim/eval_getup.py --task getup_v2 --getup2_action_clip 8.0 \
    --up_th 0.95 --z_lo 0.20 --z_hi 0.34 \
    --pkl policies/go1_getup_policy.pkl --envs 256 --steps 400
```

> [!WARNING]
> `--getup2_action_clip` 必须和训练时一致（8.0）。用配置默认值 3.0 会把动作夹掉，
> 评出来的成功率是假的 0%。

### 复现起身成功率

```bash
python sim/probe_handover.py --stage realfall --envs 128 --steps 1500 \
    --getup_steps 600 --cmd 0.5 --bound_xy 5.0
```

### 训练

```bash
# 起身策略：40M 步，RTX 5060 Laptop 上约 30 分钟
python -u -m train.train_getup --task getup_v2 --action_clip 8.0 \
    --num_timesteps 40000000 --num_evals 16 --episode_length 400 \
    --num_minibatches 4 --updates_per_batch 5 --lr 5e-4 --entropy 0.005 --init_noise_std 1.0

# 走路策略：60M 步
python -u -m train.train_go1 --num_timesteps 60000000
```

检查点存在 `logs/`，策略存在 `policies/`。改代码之前可以先跑两个自检：
`python -m train.train_getup --dry_run`（只建环境和网络）和 `python sim/probe_getup_v2.py`
（起身环境自检）。

## 目录

```
mjx-go1-getup/
├── envs/                     # MJX 环境（物理 + 任务定义）
│   ├── go1_walk.py           #   走路：地形、奖励、终止、91 维观测
│   ├── go1_getup.py          #   起身 v1：增量动作（保留作对照）
│   ├── go1_getup_residual.py #   起身：关键帧状态机 + 学习残差
│   └── go1_getup_v2.py       #   起身 v2：锚定动作 + 宽高斯 + 历史帧
├── train/
│   ├── train_go1.py          # 走路训练（brax PPO）
│   └── train_getup.py        # 起身训练：--task getup | getup_res | getup_v2
├── sim/
│   ├── view_go1.py           # 交互式查看器 + 走路/起身调度器
│   ├── watch_v20_fast.py     # 高帧率观测脚本
│   ├── eval_getup.py         # 起身成功率 / 分姿态桶 / 到站时间
│   ├── eval_walk.py          # 走路评估（固定出生点集）
│   ├── eval_stairs.py        # 上下台阶成功率
│   ├── probe_handover.py     # 交还瞬间的定量探针
│   ├── probe_getup_v2.py     # 起身环境自检
│   ├── probe_standability.py # 判据本身可不可达
│   ├── probe_nefc.py         # 接触约束溢出探针
│   ├── gen_terrain.py        # 地形生成器（hfield PNG + json）
│   ├── getup_keyframe.py     # 脚本化关键帧起身（基线 + 查看器兜底）
│   └── common.py             # 策略加载（pkl / orbax checkpoint）
├── models/go1/               # MuJoCo 模型（MJCF + 网格 + 地形）
├── policies/                 # 两个策略：走路 v23@60M、起身 v2f
└── docs/                     # 实验记录、状态、查看器指南、起身配方调研
```

## 文档

- `docs/status.md`：当前结果，以及各判据的精确定义（建议先看这个）
- `docs/experiment-log.md`：开发记录，按时间顺序写，包含失败的尝试
- `docs/getup-recipes.md`：公开起身实现的配方对照和 SOTA 调研
- `docs/viewer-guide.md`：MJX 查看器的逐帧耗时拆解和常见坑

英文版概览见 [README.en.md](README.en.md)。

## 已知问题

- `njmax=256` 对起身任务偏小。摔倒时单个 world 实测需要约 1300 个约束，训练日志里会出现
  `nefc overflow` 告警（719 次 / 384k world-帧）。用 `njmax=2048` 重训一版是待办，可以用
  `sim/probe_nefc.py` 复现。这不解释观察到的抽搐，但会影响躺地时的接触精度。
- 起身策略是靠顶住动作裁剪站住的（`|action|` 中位数 8.0 = clip，关节速度还有 11.5 rad/s）。
  奖励里加一项「站住时要安静」，或者让成功奖金要求姿态静止，是下一步。
- 训练出生姿态 88.7% 对走路真摔倒 77.3%，差 11 个点。计划用真摔倒姿态池微调
  （`sim/make_getup_posepool.py`）。
- 上台阶没做出来（固定楼梯上 0%），走路策略的镜像对称性也不完美（hip RR/RL 偏差约 16°）。

## 致谢

- [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground)、
  [Brax](https://github.com/google/brax)：本项目的训练栈
- [get-up-isaaclab](https://github.com/iit-DLSLab/get-up-isaaclab)（IIT-DLSLab）：
  `envs/go1_getup_v2.py` 的配方来自这里
- [HoST](https://github.com/InternRobotics/HoST)、
  [HumanUP](https://github.com/RunpeiDong/HumanUP)：多 critic 和两阶段课程
- [quadruped-rl-locomotion](https://github.com/nimazareian/quadruped-rl-locomotion)：
  走路任务是从这个仓库复刻过来的
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)：Go1 的 MJCF

## License

[MIT](LICENSE)
