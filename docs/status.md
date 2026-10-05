# 状态与判据口径 (截至 2026-09-22)

> 一页纸总览：当前策略的成功率、**所有判据的精确定义**（差一个数就会得出相反结论）、
> 已知坑与下一步待办。第一次读本项目建议先看这里，再看
> [实验记录](experiment-log.md) 的 §29.27 / §29.28。

---


> 这份文件是**给下一个接手的人 (或下一个会话的自己) 的一页纸总览**。
> 完整实验记录在 [`experiment-log.md`](experiment-log.md)
> (本轮内容 = §29.27 / §29.28)。

---

## 1. 一句话现状

**起身策略 77~81% (从"走路真摔倒"的姿态), 交还抽搐的根因已定位并修复 (3 处改动 + 1 处回退),
全部有 `sim/probe_handover.py` 的实测数字支撑。**

| 环节 | 状态 |
|---|---|
| 走路策略 v23@60M | ✅ 正常 (重跑确认逐位未变) |
| 起身策略 (学习型) | ✅ **77.3%** 严格 / **81.2%** 容错, 从**走路真摔倒**的 128 个样本 (§29.28) |
| 交还抽搐 | ✅ **已解决**: 根因是我上一轮加的 `--handover_blend` (§29.27) |
| 掉出地形边缘 | ✅ **已修**: 界外无地面会一直自由落体, 查看器里会在半空触发起身 (§29.28) |

---

## 2. 怎么跑 (查看器)

```bash
cd mjx-go1-getup
python sim/view_go1.py \
  --pkl policies/go1_walk_policy.pkl \
  --getup_pkl policies/go1_getup_policy.pkl \
  --getup_obs history --getup_action anchored
```

**测起身的时候把 vx 指令按成 0** (按 S 两下), 或者直接 `--cmd 0`: 默认 `--cmd 0.5` 会让
机器人一直往前走, 8~16s 就走到地形边缘。现在出界会自动重置 (不会在半空乱蹬), 但指令为 0
时观察起身最干净。

离线探针 (本轮新工具, 建议先看它再改任何东西):

```bash
# 交还瞬间的 10 组对照 (冻结 227 个起身终态, 不需要渲染)
python sim/probe_handover.py --stage spawn --envs 256 --dump sim/ho095.npz
python sim/probe_handover.py --stage walk  --cmd 0.5 --frames 500 --dump sim/ho095.npz

# "走路真摔倒 -> 起身"的成功率与失败形态
python sim/probe_handover.py --stage realfall --envs 128 --steps 1500 \
    --getup_steps 600 --cmd 0.5 --bound_xy 5.0
```

离线评估 / 重训命令见 §4。

---

## 3. 关键口径 (差一个数就得出完全不同的结论)

### 3.1 站立判据 `up_z`

`up_z = 1 - 2*(qy² + qz²)` (根四元数的**世界**竖直投影) = **躯干倾角的 cos**。

| up_z | 倾角 | 含义 |
|---|---|---|
| 1.00 | 0° | 完全竖直 |
| **0.99** | **8.1°** | 旧硬编码判据 —— 这台机器基本做不到 |
| **0.95** | **18.2°** | **当前默认口径** (查看器 `--getup_up_th`、eval `--up_th`) |
| 0.93 | 21.6° | 更宽 |
| 0.45 | 63° | 旧"摔倒"阈值 (太松, 已改) |
| 0.00 | 90° | 侧躺 |

实测: 起身成功时**高度带内最好 up_z 中位数 = 0.999**(严格满足), 但**只有 77%** 能保持连续
25 帧 —— 所以判据的"连续"是主要瓶颈, 不是姿态。

### 3.2 查看器摔倒判据 (逐帧算, 连续帧计数)

```python
upz_now  = 1 - 2*(q[4]² + q[5]²)                       # 世界竖直投影 = cos(倾角)
z_rel_now = q[2] - terrain_height(q[:2])                # 躯干**离它正下方地面**的高度
fallen = (z_rel_now < fall_z) or (upz_now < fall_uz)
fall_count = fall_count + 1 if fallen else 0            # 不满足就清零
armed = fall_count >= fall_debounce                     # 连续满足才"武装"
触发条件 = armed and cooldown <= 0 and not getup_active and |xy| <= oob_reset_xy
```

| 参数 | 默认 | 含义 |
|---|---|---|
| `--fall_uz` | **0.30** | 倾角 > 72.5° 就算倒 (旧 0.45 = 63°, 太松会误触发) |
| `--fall_z` | 0.16 m | 躯干离地 < 16cm (站直 0.277m, 躺地 ~0.13m) |
| `--fall_debounce` | **18 帧 = 0.36s** | 条件要**连续**满足 18 帧 (1 帧 = ctrl_dt = 0.02s) |
| `--getup_cooldown` | **50 帧 = 1.0s** | 交还/超时后不再触发起身 |
| `--oob_reset_xy` | **5.8 m** | NEW: `|xy|` 超过它就**直接重置**, 不试图起身 |

**为什么要"连续帧计数"**: 走路时一次深度踉跄也能瞬间满足 `up_z<0.45`, 单帧判据会误触发;
真摔倒会一直满足。这就是用户报的"明明还能走路就触发起身"的原因 (旧值 0.45/8 帧太松)。

### 3.3 起身成功判据 (交还条件)

`up_z > 0.95` 且 `z_rel ∈ [0.20, 0.34]`, **容错计数** 达到 25:

```python
if ok: getup_hold_cnt += 1          # ok = (upz > getup_up_th) and (zlo <= zr <= zhi)
else:  getup_hold_cnt = max(getup_hold_cnt - 1, 0)   # NEW: 容错, 不归零
```

旧版是"任何一帧不满足就归零"。实测 (128 个真摔倒样本): 严格 77.3% -> 容错 **81.2%**,
到站 5.71s -> **4.89s**。失败形态里 26/29 是"其实已经站进高度带、中间掉出 1~2 帧就归零"
—— 正是用户说的"看着站起来了却提示没成功"。

### 3.4 交还: `--handover_blend` **默认 0 (不要动它)**

上一轮我把它默认成 10, 结果**加重**了抽搐。实测 (227 个起身终态, 500 帧):

| 方案 | 摔倒% | 末态站住% |
|---|---|---|
| **不混合 (默认)** | **7.9** | 85.9 |
| 走路策略原生基线 (对照) | 8.4 | 92.1 |
| blend 10 (上一轮的默认) | **38.3** | 59.5 |
| blend 30 | **73.6** | 35.2 |
| 一直保持起身最后目标不交还 | **90.7** | 0.0 |

原因: 起身"站住"那一帧的动作是**顶在裁剪上的发力状态** (median `|a_f|` = 8.0 = clip,
目标偏离关节实际位置 2.7 rad)。从它开始混合 = 头 10/30 帧先按这个目标驱动 = 把人推倒。

---

## 4. 关键产物

| 文件 | 说明 |
|---|---|
| `policies/go1_getup_policy.pkl` | **当前最优起身策略** (真摔倒口径 77.3% / 容错 81.2%) |
| `policies/go1_walk_policy.pkl` | 走路策略 (未改动) |
| `sim/probe_handover.py` | **NEW**: 交还瞬间 / 真摔倒起身 的定量探针 (4 个 stage) |
| `envs/go1_getup_v2.py` | 公开配方版起身环境 (锚定动作 + 宽高斯 + 地形朝向 + 5 帧历史) |
| `sim/eval_getup.py` | 评估 (`--task getup_v2` / `--up_th` / `--getup2_action_clip`) |
| `sim/view_go1.py` | 查看器 + 调度器 (**唯一**支持 getup_v2 的查看器) |
| `docs/getup-recipes.md` | 公开开源实现的配方对照与出处 |

离线评估 / 重训:

```bash
python sim/eval_getup.py --task getup_v2 --getup2_action_clip 8.0 \
  --up_th 0.95 --z_lo 0.20 --z_hi 0.34 \
  --pkl policies/go1_getup_policy.pkl --envs 256 --steps 400

python -u -m train.train_getup --task getup_v2 \
  --action_clip 8.0 --num_timesteps 40000000 --num_evals 16 \
  --episode_length 400 --num_minibatches 4 --updates_per_batch 5 \
  --lr 5e-4 --entropy 0.005 --init_noise_std 1.0
```

---

## 5. 本轮修复清单 (全部有实测依据)

| # | 改动 (都在 `sim/view_go1.py`) | 依据 |
|---|---|---|
| 1 | `--handover_blend` 默认 **10 -> 0** | 摔倒率 6.6% vs 37.0% (blend10) vs 73.1% (blend30) |
| 2 | **删掉** `info["last_act"] = jp.zeros(12)` | 清零 8.4% vs 保留真值 6.6% (我上一轮改错了方向) |
| 3 | 新增 `--oob_reset_xy` (默认 5.8) | 界外无地面 -> 自由落体, 在半空触发起身 = 抽搐 |
| 4 | "站住"计数改**容错** (满足+1 / 不满足-1) | 77.3% -> 81.2%, 5.71s -> 4.89s |

---

## 6. ⚠️ 训练侧待办 (已定位, 本轮没做)

1. **`njmax` 太小**: `getup_v2` 继承 `go1_walk` 的 `njmax=256`, 而 hfield 地形上摔倒时
   **单 world 实测 `nefc` 要 ~1300** (`nefc overflow` 告警 719 次 / 384k world-帧);
   其他起身工具 (`go1_getup_residual`/`dagger_getup`/`probe_getup_standability`) 都给 768。
   -> 给起身环境设 `njmax=2048` 重训一版 (注意: 256 与 2048 的发散曲线一致, 所以这不解释
   抽搐; 它只影响"摔倒时的接触精度")。
2. **终态动作顶在裁剪上**: 起身"站住"是靠发力顶住的 (median `|a_f|` = 8.0 = clip, 关节还
   在 11.5 rad/s 上动)。-> 奖励加"站住时动作/关节速度要小", 或把成功奖金判据改成"安静
   站立", 让终态变成真·静止站姿。这两条都是"提升起身质量", 与修抽搐解耦。
3. 训推分布差 11 个点 (训练出生 88.7% -> 走路真摔倒 77.3%): 可以用**走路真摔倒**的姿态
   池做微调 (`sim/make_getup_posepool.py` 已有姿态池工具)。

---

## 7. 记录位置

* 主记录: `experiment-log.md`
  * **§29.27** = 交还抽搐的根因 + 10 组对照
  * **§29.28** = 真摔倒起身成功率 + 出界/容错计数两个新发现 + 修复汇总
* 配方调研: `getup-recipes.md`
* 下载的公开源码: `research/` (get-up-isaaclab / HoST / HumanUP)

---

## 8. 环境与性能事实 (避免重复摸底)

* 单卡 RTX 5060 Laptop 8 GiB, WSL2; **一个进程独占一个 env** (两个 env 同进程会撞
  warp CUDA-graph 地址缓存)。
* Python 3.14 venv（开发机）; 依赖与版本见根目录 `requirements.txt`。
* 训练速度: 起身 v2 实测 **22k~25k fps**, 40M 步 ≈ 30 分钟。
* `XLA_PYTHON_CLIENT_MEM_FRACTION` 必须在 `import jax` **之前**设 (脚本里已 setdefault)。
* **评估/探针必须复现训练的动作范围** (`--getup2_action_clip 8.0`, 默认配置是 3.0),
  否则会评出假 0%。
* **eager 的 `jax.vmap(env.step)` 每帧都会重新 trace+compile** (实测 500s 跑不完一个探针);
  循环里一律 `jax.jit(jax.vmap(...))`。
* 地形是 `hfield size="6 6 0.3 0.096"` -> **半边长 6.0m**, 出生 xy ~ U(-5,5) -> 界外没有
  地面, 会无限自由落体。
* **`argparse` 的 help 里不能有裸百分号** (已踩 4 次): argparse 会做 `help % params`,
  遇到 `73.1%` 就 `ValueError: badly formed help string`, **启动即崩**。要写 `73.1%%`。
  改完 help 一定要跑一次 `--help` 烟测 (不需要窗口, 秒级)。
