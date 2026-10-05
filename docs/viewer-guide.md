# 观测器（Viewer）撰写指南

两份文档的合并：《MuJoCo + JAX 观测器脚本撰写指南与注意事项》和《MuJoCo 查看器指南》。

文中的数字都是在本机测的（WSL2 / 16 核 / RTX 5060 Laptop / jax 0.7.2 + warp 1.16.0 +
mujoco 3.12.0），换机器后量级会变，但坑的机理和排查方法是一样的。对应的脚本是
sim/view_go1.py 和 sim/watch_v20_fast.py。

---

## 一、观测器脚本撰写指南（JAX/MJX 逐帧耗时拆解）

# MuJoCo + JAX 观测器脚本撰写指南与注意事项

> 适用场景: 用 MJX(warpx) 跑仿真、JAX/brax 加载策略、`mujoco.viewer` 开窗口做
> **交互式观测**的脚本 (本项目即 `sim/view_go1.py`、`sim/watch_v20_fast.py`)。
>
> 本文全部数字都是**本机实测** (WSL2, 16 核, RTX 5060 Laptop, jax 0.7.2 +
> warp 1.16.0, mujoco 3.12.0)。换机器后量级可能不同, **但每条"陷阱"的机理
> 与判别方法仍然适用**。
>
> 与训练脚本的根本区别: 训练侧 batch=768、buffer 地址稳定、不看画面; 观测侧
> batch=1、每帧新建数组、有人盯着画面。**同一份代码, 两种用法的最优解不同** ——
> 这是本文所有注意事项的总纲。

---

## 0. 一张图: 观测脚本每帧的耗时构成

```
每帧:
  ① 组 command (3 维)           ~0 ms    但会改 buffer 地址 (陷阱 2)
  ② 策略前向  policy(obs)       eager 11.0ms / jit 0.08ms   (陷阱 1)
  ③ 物理一步  step(state,act)    13-15 ms (WARP_STAGED)      (陷阱 2)
  ④ 搬 qpos/qvel 回 CPU         0.2 ms    (别搬整个 Data, 陷阱 4)
  ⑤ CPU mj_forward 灌 d_view    0.2 ms
  ⑥ 自绘扫描点 (若开)            3.5 ms    (必须 jit, 陷阱 3)
  ⑦ v.sync() 交给渲染线程        2.4 ms    (真正 GL 绘制不在这里!)
  ⑧ 节流 sleep                  补足到 env.dt
```

对照实测: 优化前 **373 ms/帧 (3 fps)**, 优化后 **19.5 ms/帧 (50 fps)**。
差值几乎全部来自 ②③⑥ 三处, 与"物理太慢"无关。

> 上表的 ③ 是**单进程独占 GPU** 的实测值。若显存被占满 (例如有残留的查看器
> 进程没退), ③ 会暴涨到 **300-550 ms/帧 (2-3 fps)** —— 见第 7 节。这类"假慢"
> 必须在读代码之前先排除。

---

## 1. 陷阱一: 策略前向没 jit —— 每帧白扔 ~11ms

**最容易被忽略、收益最大的一条。**

```python
# ✗ 错: eager 调用。小网络的 Python dispatch 开销 >> 计算本身
act, _ = policy(state.obs["state"], rng)

# ✓ 对: jit 成一个 kernel
act_jit = jax.jit(lambda obs: policy(obs, rng_const)[0])
act = act_jit(state.obs["state"])
```

实测 (128×128 网络, obs 91 维):

| 写法 | 耗时 |
|---|---|
| eager | **10.97 ms** |
| jit | **0.08 ms** (137x) |

同进程冷/热两轮 A/B (排除进程早期效应): jit 稳定省 ~11ms, **与冷热无关**。

**为什么容易漏**: 训练时策略在 `jax.lax.scan`/`vmap` 里被整体 jit, 从没暴露
过 dispatch 开销; 只有观测脚本这种"每帧裸调一次"的写法才踩到。

**注意 rng**: jit 时把 rng 固定成常量 (确定性推理用不到真随机), 否则每次传
不同 key 会导致重新编译。本项目策略是 `deterministic=True`, 直接给常量即可。

**等价性验证**: jit 前后应比**动作**的偏差, 不要比 qpos —— 一步物理求解会把
1e-7 的动作差混沌放大成米级, 拿 qpos 比会把"等价"误判成"不一致"。

---

## 2. 陷阱二: `graph_mode` 对训练最优, 对观测可能是灾难

MJX 在 warp 后端下把 kernel 包成 CUDA graph。默认 `graph_mode=WARP` 的语义是
**"按 buffer 地址缓存图"** (见 `warp/_src/jax/ffi.py` 的 `JaxCallableGraphMode`)。

* **训练**: `jax.jit(donate_argnums=0)` + 固定地址 -> 图缓存命中 -> 快。
* **观测**: 每帧 `state.info["command"] = jp.array(cmd)` 新建数组, 按 R 键 reset
  换 state -> **地址每帧变 -> 反复重新 capture** -> 跑几千帧后崩:
  `wp.capture_begin(...) -> RuntimeError: Warp error: unknown stream`。

实测 (模拟观测循环, 各跑 3000 帧):

| graph_mode | 步时 | 结果 |
|---|---|---|
| NONE | 139.0 ms | 不崩但慢 |
| JAX | 预热即崩 | `CUDA_ERROR_NOT_SUPPORTED` |
| WARP (默认) | 185.5 ms | 探针下不崩, **真实窗口几千帧后崩** |
| **WARP_STAGED** | **16.3 ms** | **不崩 (staging buffer + 图内 memcpy, 专为地址会变设计)** |
| WARP_STAGED_EX | 57.3 ms | 不崩 |

**结论**: 观测脚本用 `WARP_STAGED`; 训练保持默认。所以 env config 里要有
`graph_mode` 字段, 让两种用法各取所需 —— 不要为了观测去改训练的配置。

**顺带一条**: `donate_argnums=0` 下, 循环里复用的固定数组会被 jax 判为
`Array has been deleted`。要么每帧重建 (推荐, 3 维数组开销可忽略), 要么不捐赠。

**这一项会静默丢失 (本项目真实发生过)**: `watch_v20_fast.py` 的最终交付版就漏了
这行 —— 文档写了、代码没有, 表现为"每帧重新 capture"+"跑几千帧后崩"。
**交付前 grep 一遍, 别信文档**:

```bash
grep -n 'graph_mode' sim/*.py     # 每个查看器都应有一处设置 WARP_STAGED
```

---

## 3. 陷阱三: 循环里逐个写 JAX 算子 = 每帧上百次 kernel launch

观测脚本常有"按帧算一点东西给可视化用"的需求 (本项目: 35 个地形采样点的
位置 + 地面高 + 特征)。**必须把这一整串 jit 成一个 kernel。**

| 写法 | 每帧耗时 (两次独立测量) |
|---|---|
| eager 逐条 (照抄分析脚本) | **51.6 ms** / **315.8 ms** |
| jit 成一个 kernel | **0.32 ms** / **6.28 ms** |

**慢 50~160 倍** (比值随机器状态波动, 关键是量级), 数值一致 (最大差 <5e-07)。
原因: eager 写法每帧要在 Python 侧按顺序发起一百多个 tiny kernel, 每个都有
launch 开销与同步。

**判别方法**: 观测脚本里凡是"每帧调用一次、返回若干数组给渲染用"的 JAX 函数,
一律 `@jax.jit`。若输入含非常量结构 (如 env 对象), 把它排除出参数列表或用闭包。

---

## 4. 陷阱四: 别把整个 Data 搬回 CPU

渲染只需要 `qpos`/`qvel`, 不需要接触数组等巨大字段。

```python
# ✗ 慢: mjx.get_data + mj_copyData 会搬整个 Data (含 naconmax=65536 的接触数组)
# ✓ 快: 只搬 qpos/qvel (1.13 ms -> 0.02 ms)
qpos = np.asarray(state.data.qpos)
qvel = np.asarray(state.data.qvel)
d_view.qpos[:] = qpos
d_view.qvel[:] = qvel
mujoco.mj_forward(env.mj_model, d_view)
```

**附带好处**: `np.asarray` 会**强制设备同步**, 所以策略+step 的真实耗时在这里
才结算 —— 计时点应放在这一行之后, 否则测到的是异步派发速度, 不是帧耗时。

---

## 5. 陷阱五: 渲染后端 —— "没有 /dev/dri" 不等于"只能软件渲染"

WSL2 下 `ls /dev/dri` 确实是空的, 于是很容易得出"只能 llvmpipe 软件渲染"的
结论并去调 `LP_NUM_THREADS` 限流 (治标)。**先问"能不能换条路"。**

本机有 `/dev/dxg` + Mesa 的 `d3d12_dri.so`, 即能走 D3D12 硬件后端:

| GL 后端 | 离屏渲染 640x480 |
|---|---|
| 默认 (llvmpipe) | **491.2 ms/帧** |
| `MESA_LOADER_DRIVER_OVERRIDE=d3d12` | 458.9 ms/帧 (无效) |
| **`GALLIUM_DRIVER=d3d12`** | **15.3 ms/帧 (32x)** |

```python
# 必须在 import mujoco 之前设!
os.environ.setdefault("GALLIUM_DRIVER", "d3d12")   # 用 setdefault, 允许用户覆盖
```

* 关键变量是 **`GALLIUM_DRIVER`**; `MESA_LOADER_DRIVER_OVERRIDE` 实测无效。
* 窗口路径 (GLFW) 下同样生效, 不只是离屏 EGL。
* **价值不只是渲染快**: 硬件渲染几乎不占 CPU, 于是"软件渲染吃满 16 核 ->
  饿死 jax 的 kernel 分发线程 -> `step` 暴涨"这条链路从根上消失。这才是关键 ——
  jax/warp 的 kernel 分发是**在 CPU 侧**的, 与 GL 抢核会直接表现为 step 变慢。
* 回退要留: `GALLIUM_DRIVER=` (空串) 可强制 llvmpipe, 此时才需要
  `LP_NUM_THREADS=4` 限流 (实测 llvmpipe + 限流也有 50-61fps, 因为策略已 jit)。

**验证后端真的换了** (别靠猜, 也别信 `MESA_*`) —— 自己建临时上下文再查:

```python
def active_gl_renderer():
    ctx = mujoco.GLContext(64, 64)
    ctx.make_current()          # 主线程直接 glGetString 会因无 current context 返回 None
    gl = ctypes.CDLL("libGL.so.1")
    gl.glGetString.restype = ctypes.c_char_p
    return gl.glGetString(0x1F01).decode()   # GL_RENDERER
# 期望: "D3D12 (AMD Radeon 780M Graphics)" 而不是 "llvmpipe (LLVM ...)"
```

**阴影是软件渲染的大头**: MuJoCo 默认对每个光源渲染 `shadowsize=4096` 阴影图,
开销是"逐光源的深度渲染次数"而非分辨率 (640x480 与 320x240 都是 ~490ms)。
关法有两处 (model 级 + 运行时场景级, 后者在 `v.user_scn.flags` 上):

```python
m.light_castshadow[:] = 0                       # model 级 (开窗口前)
v.user_scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0   # 场景级, 运行时可改
v.user_scn.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0
v.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
```

---

## 6. 陷阱六: 开窗口后的早期帧是暂态, 用"窗口前预热"消除

现象: 窗口刚出现的头 ~400 帧 `step=373ms`, 之后**自行恢复**到 52ms; 期间还
可能再爆一次。同一份配置在进程里早期测 63ms、稍后测 14.9ms。

**解法**: 在 `launch_passive` **之前**, 用与主循环**完全相同**的调用序列跑
`--warmup N` 帧 (默认 150), 让编译与首帧开销在"还没有窗口"时结清:

```python
for _ in range(args.warmup):
    state, qpos, qvel, _, _ = frame(state, d_view)   # 与主循环同一个 frame()
if show_scan:
    jax.block_until_ready(scan_viz(state.data, state.data.qpos))   # 扫描也预热
```

实测: 预热后第一行日志 (frame 25) 就是 28fps, frame 100 已 50fps, 无 373ms 段。

**注意**: 预热必须走**同一个函数**, 不能手写一份"简化版" —— 否则漏掉某个
kernel 就白预热了 (本项目第一版就漏了扫描点的预热)。

---

## 7. 陷阱七: 显存池预占 —— 第二个 GPU 进程会让帧率塌方 10~35 倍

**症状**: 脚本一个字没改, 昨天 50fps, 今天 `step=302ms` / **2-3 fps**。分项里
`copy=0.2 / sync=2.4 / scan=3.4` **全部正常, 只有 `step` 暴涨**; 而且
`nvidia-smi` 看利用率不高, WSL 内 `load average` 也只有 0.11。

**根因**: XLA 默认**预占约 75% 显存**。8GiB 卡上单进程就要 ~6GiB, 一旦出现第二个
GPU 进程 (上一个查看器窗口没关, 并发的训练/评估/看护), 显存直接打满 -> GPU 无法
前进 (SM 时钟掉到 ~180MHz) -> 单步从 15ms 变成 300-550ms。

| 场景 | 显存 | `clocks.sm` | 帧耗时 | fps |
|---|---|---|---|---|
| 单进程 | 未打满 | 1500+ MHz | **15.7 ms** | **63.7** |
| 双进程, **默认**预占 | **7674 / 8192 MiB** | **180 MHz** | A **306.9** / B 32.4 ms | **3.3** / 30.8 |
| 双进程, `MEM_FRACTION=0.25` | 未打满 | 正常 | **27.1 / 27.1 ms** | **36.9 / 36.9** |

注意中间那行: **两个进程通常只有一个被饿死**, 另一个还有 30fps —— 所以症状是
"我这个查看器莫名其妙只有 3fps", 而不是"两个都慢", 极易误判成代码问题。

**修复 (每个观测脚本必备, 必须在 `import jax` 之前)**:

```python
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.25")  # 8G 卡 -> ~2G
```

实测: 单进程 15.7ms/帧; 双进程各 27ms/帧 —— **并发也不再塌方**。其余 sim 脚本
(评估/诊断) 用 0.3~0.55, 训练侧用 0.80。

**判别顺序**: 症状异常 -> ① 先看 GPU 进程与显存:

```bash
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
nvidia-smi --query-gpu=memory.used,pstate,clocks.sm --format=csv
```

`memory.used` 接近上限、`clocks.sm` < 200MHz 是**显存打满的指纹** -> ② 再怀疑代码。
做 A/B 前先 `pkill -f 你的脚本` 确认退干净。

**为什么这条很难自己发现 (2026-09-18 回归案例)**:

1. 本项目 21 个 sim 脚本里 **19 个都设了** `XLA_PYTHON_CLIENT_MEM_FRACTION`,
   **只有 `view_go1.py` 和 `watch_v20_fast.py` 漏了** —— 少数派漏项, 不逐个
   grep 根本看不出来;
2. 出问题时文档 §22 的"50fps"结论**仍然是对的**: **文档对、脚本错**;
3. 第一轮误判成 `graph_mode` (补上 `WARP_STAGED` 后 302 -> 550ms, 因为真因还在);
   第二轮误判成"物理变慢 / 地形变重";
4. 真正定位只靠两步:
   **(a) 无窗口 headless 基准** —— 复刻 `frame()` 全流程 (策略 jit + step + 搬 qpos),
   单进程只有 **15.7ms**, 证明**软件栈没问题**;
   **(b) 双进程并发 A/B** 复现出 306.9ms, 再用 `MEM_FRACTION=0.25` 反证回 27ms。
   复现脚本: `sim/probe_viewer_slow.py`。

**教训**: "慢"分"进程内"与"进程外"两类。**先在同机跑一个能出数的 headless 基准,
把进程内因素全部排除, 再去找进程外**; 别一上来就读代码猜配置。

---

## 8. 观察正确性: 观测脚本必须与训练严格同档

观测的意义是"看策略真实表现", 所以环境档位、obs 组装、命令坐标系必须与训练
**逐项一致**。本项目踩过的具体坑:

| 项 | 说明 |
|---|---|
| **obs 维度** | v19/v20/v21 因 height_scan / 相位而不同 (48 / 83 / **91**)。必须从权重反推维度, 不能写死 |
| **网络隐层** | pkl 不存网络配置, 需从参数形状反推 (v10 是 64,64; v14+ 是 128,128) |
| **奖励开关影响 obs** | v20 开 `feet_phase` 会让 obs 多 8 维 (cos,sin × 4 足)。**不设这个, 91 维策略会因 obs 只有 83 维报 ScopeParamShapeError** |
| **命令坐标系** | v17+ 训练用机体系 (`command_config.frame="body"`)。用世界系比指令会得出"走反了"的错误结论 |
| **终止倾角** | v17+ 用 ±17.5°, 写默认 10° 会让策略频繁误重置 |
| **联动参数** | v20 要求 `max_foot_height == gait_swing_height`, 否则策略被两个目标撕扯 |
| **随机出生** | 训练开随机 xy + 随机 yaw。观测若关了, 行为会与训练不同 |

**最佳实践**: 让观测脚本**按策略维度自动补齐开关**, 而不是要求用户记住每个
版本加哪些 flag。本项目 `view_go1.py` 就是这么做的 (检测到 >83 维就自动开
height_scan/terrain/body_frame/tilt_17.5), 并打印"自动开启 X"。

### 8.3 摔倒处理: 判据要用**相对地面**高度, 动作要先在目标物理里验

观测脚本常带"摔倒自动重置"。两个坑:

1. **判据不能用世界 z 的绝对值**: 地形最高台 0.30m 时, 狗躺在高台上躯干 z 约 0.39m,
   `qpos[2] < 0.15` 永远不成立 —— 本项目地形模式下**自动重置从来没生效过**。
   正确写法: `qpos[2] - terrain_height(qpos[:2]) < 阈值`, 或再加一条
   `up_z = 1-2*(qx^2+qy^2) < 0.45` (倾斜 > 63 度); 并**连续满足 N 帧**
   (本项目 8 帧) 去抖。
2. **"能站起来的动作"必须先在实际物理里跑过**: §15 的关键帧起身在 CPU + `plane`
   地面上 100%, 搬到查看器的 MJX + `hfield` 地形地面上只有 6~23% (机理见实验
   记录 §29.3: hfield 按格子生成接触, 腿被拖住)。**floor geom 类型 (plane vs hfield)
   本身就是会改变结果的自变量**, 不能假设等价。

现成实现: `sim/getup_keyframe.py` + 两个查看器的
`--fall_action {getup,getup_hold,reset}`。

---

## 9. 自绘几何 (扫描点等) 的正确姿势

```python
# 查看器把自绘几何放在**独立的** v.user_scn, 由渲染循环与主场景合成。
# 所以: 只重置 user_scn 的 ngeom, 不要碰主 scene。
v.user_scn.ngeom = 0
for i in range(n_scan):
    mujoco.mjv_initGeom(scn.geoms[n], mjGEOM_SPHERE, size, pos, mat, rgba)
    scn.ngeom += 1
```

**两个坑**:

1. **离屏渲染时不要照抄 `scn.ngeom = 0`**: 离屏没有 `user_scn`, 若对**同一个**
   scene 重置 ngeom, 会把 `mjv_updateScene` 刚放进去的机器人 geom 一起抹掉 ——
   渲染出"一堆绿球但没有机器人"。离屏要**追加**, 或维护独立 scene。
2. **绘制内容必须与环境特征同源**: 画出来的点和策略看到的不一致, 观测就失去
   意义。本项目把"采样点 xy / 地面高 / 特征"用与 `env.height_scan_features`
   **完全相同**的变换 (只按 yaw 旋转, 原点在躯干投影) jit 在一起, 并用
   `verify_watch_v20_fast.py` 断言两者**逐位一致 (偏差 0.000e+00)**。

**颜色映射要自检**: 本项目发现原分析脚本的 `r=0.5(1-t), g=0.35, b=0.5(1+t)`
在 t=0 时给 (0.5,0.35,0.5) 灰紫, 与它自己"t=0 -> 绿"的注释不符。中点、两端都
应验算一遍。

---

## 10. 交互与节流

```python
# 实时节流: 只补足到 env.dt, 慢帧不额外叠加 sleep
if not args.fast:
    slack = env.dt - (time.perf_counter() - f0)
    if slack > 0:
        time.sleep(slack)
```

* **不要**无条件 `time.sleep(env.dt)`: 那会在慢帧上再叠加 20ms。
* 键盘用 `pynput` 时, **单发按键要认"松->按"边沿** (用 key_states 记状态),
  否则 OS autorepeat 会把"按一下"变成连跳好几档。且键名统一用小写字符串
  (不存 KeyCode 实例 —— 每次 press 事件的实例不保证按值等价)。
* overlay 里同时显示 **fps + step/copy/sync/scan 分项**。分项是定位瓶颈的唯一
  依据; 只有 fps 的话, 你无法知道该优化哪一项。

---

## 11. 验证: 加速不能改行为

**任何"提速"改动都要过闸门, 而不是看 fps 涨了就收工。** 本项目闸门
(`sim/verify_watch_v20_fast.py`, 四道全过):

| 检查 | 判据 |
|---|---|
| 策略 obs == 环境 obs | 维度一致 (91/162) |
| 自绘几何 vs env 特征 | 偏差 ≈ 0 (实测 0.000e+00) |
| 策略真的在走 | 机体系 vx 跟踪指令; 躯干 z 保持标称 (0.28-0.34) 不坠 |
| jit vs eager | **动作**偏差 < 1e-5 (实测 1.19e-06) |

**核验脚本自己也会错** (本项目三处误判, 记录下来):

* 用**世界系** `qvel[0]` 比机体系指令 -> 误判"走反了"。随机出生带随机 yaw,
  两者本就不该相等; 要用 `env.get_body_linvel`。
* 跑 400 步 (8s) 会**走出 ±6m 地形边界**坠落。不是策略问题, 是核验窗口太长。
* 用 **qpos** 判 jit 等价性 -> 混沌放大导致误判。判据应是动作本身。

**教训**: 指标报异常时, 先怀疑指标 (§18.7 同源)。

### 11.1 别被"进程外的干扰"和"误导性指标"骗了

同一脚本在不同时刻跑出 24fps —— 甚至 **3.3fps** (平时 50fps), 查了半天,
结论都是**我自己的错** (进程外的干扰):

* **残留进程抢 GPU**: 上一轮后台的查看器没退干净, 两个进程共享同一块 GPU。
  **实测后果远比"帧率腰斩"严重**: 其中一个进程塌到 **3.3 fps (306.9 ms/帧)**,
  另一个仍有 30.8 fps —— 因为 XLA 默认预占 ~75% 显存, 两个进程打满 8G 后
  GPU 掉到 180MHz。**完整机理 / 实测表 / 修复见第 7 节。**
  此时 **WSL 内 `load average` 只有 0.11**, 看不出异常; 要看 **GPU 进程与显存**:
  `nvidia-smi --query-compute-apps=pid,used_memory --format=csv`
  `nvidia-smi --query-gpu=memory.used,clocks.sm --format=csv`。
  做 A/B 前先 `pkill -f 你的脚本` 并确认已退干净。
* **不要打印不能代表目标场景的指标**: 脚本曾打印"无窗口稳态 step", 它比窗口内
  偏高约 2 倍 (24ms vs 12.5ms), 我据此怀疑"窗口外反而更慢", 还做了"让出 CPU"
  对照实验 —— 四种写法都 18-19ms, 与窗口内 **total** 一致, 说明那个数只是受
  开窗口前驱动/排队状态影响的偏置量。**去掉它, 只留主循环 fps 分项。**

**判别顺序**: 症状异常 -> ①先确认没有其他进程抢 GPU -> ②再确认指标本身是否
代表目标场景 -> ③最后才怀疑代码。

---

## 12. 被否决的"优化" (留作记录, 别重蹈)

**把 `naconmax` 从 65536 降到 512**: 单环境步耗时 14.94 → 12.28 ms (**-18%**),
看着是白捡的。但用固定动作序列跑 500 步对比轨迹:

| naconmax | 与默认的最大偏差 | **step1** 偏差 |
|---|---|---|
| 2048 | 8.88e-01 | 1.67e-03 |
| 512 | 5.19e-01 | 7.81e-03 |

**从第 1 步就分叉** (1.7e-3 远超 float32 舍入), 随后混沌放大到 0.5m。
机理: 接触数组容量改变求解器里的求和/排序顺序 -> 物理不再等价。
**观测脚本必须复现训练物理, 不能为了 18% 改动力学。**

**判据要点**: 看**第 1 步**是否分叉。后期差异可能是混沌放大, 不能当证据。

---

## 13. 交付清单 (写完观测脚本对着过一遍)

- [ ] 策略前向 **jit** 了 (陷阱 1)
- [ ] `graph_mode="WARP_STAGED"` (陷阱 2)
- [ ] `XLA_PYTHON_CLIENT_MEM_FRACTION=0.25` (陷阱 7)
- [ ] 循环里每个按帧调用的 JAX 函数都 jit 成一个 kernel (陷阱 3)
- [ ] 只搬 qpos/qvel; 计时点在 `np.asarray` **之后** (陷阱 4)
- [ ] `GALLIUM_DRIVER=d3d12` 用 `setdefault`, 且打印**实际** GL_RENDERER (陷阱 5)
- [ ] 开窗口**之前**预热, 且走同一个 `frame()` (陷阱 6)
- [ ] 环境档位/obs 维度/命令坐标系与训练逐项对齐, 并**断言**维度 (第 8 节)
- [ ] 自绘几何用 `user_scn`; 与环境特征同源并断言 (第 9 节)
- [ ] 节流是"补足到 dt"而非无条件 sleep; 按键认边沿 (第 10 节)
- [ ] 通过正确性闸门 (第 11 节), 且核验脚本自身的坐标系/边界无误
- [ ] 任何省时间的改动都先证明**物理等价** (第 12 节)
- [ ] 交付前 grep 四个关键配置: `graph_mode` / `XLA_PYTHON_CLIENT_MEM_FRACTION` /
      `GALLIUM_DRIVER` / `LP_NUM_THREADS` —— 本项目两次回归都是"文档写了、代码没有"
- [ ] 摔倒/终止判据用**相对地面**高度, 不用世界 z 绝对值 (第 8.3 节)
- [ ] 任何"能站起来/能翻过去"的动作, 先在**训练同款物理**里量成功率 (第 8.3 节)
- [ ] 对照实验**一个进程只建一个环境** (第 15 节)

---

## 14. 一页速查: 典型症状 -> 病因

| 症状 | 首选怀疑 |
|---|---|
| `step` 几百 ms, 但 render 只有 2ms | 显存打满 (**第 7 节**) / CPU 抢核 (软件渲染) / 策略未 jit |
| `step` 300-550ms, 而 copy/sync/scan 全部正常 | 显存打满 (残留 GPU 进程 + XLA 预占), **第 7 节** |
| `clocks.sm` < 200 MHz 或显存接近上限 | 同上 (显存打满的指纹) |
| 跑几千帧后崩 `unknown stream` | `graph_mode` 该用 WARP_STAGED |
| 早期帧慢、之后自行恢复 | 窗口暂态, 用预热消除 |
| fps 低但各项分项都不高 | 节流逻辑 (无条件 sleep) |
| obs 形状不符 `ScopeParamShapeError` | 环境开关与训练不一致 (第 8 节) |
| 画出来的点和预期不符 | 自绘几何与 env 特征不同源 (第 9 节) |
| 改小缓冲区后"变快但行为变了" | 物理不再等价, 弃用 (第 12 节) |
| 观测说策略摔了, 但看起来正常 | 先怀疑核验指标 (坐标系/边界) |
| 昨天 50fps 今天 3fps, 代码没改 | 先 `nvidia-smi` 查残留进程与显存 (**第 7 节**), 别读代码 |
| 同一段代码换个进程就正常 / 两个环境互相串味 | 一进程两环境, CUDA graph 按地址复用 (**第 15 节**) |
| 地形模式摔倒后"从不重置" | 摔倒判据用了绝对 z (**第 8.3 节**) |

## 15. 陷阱八: 一进程两环境 —— warp 的 CUDA graph 会按地址串味

做 A/B 时在一个进程里先后建两个环境 (nq/nv/接触规模相同), 后一个的 `step` 会命中
前一个的捕获图。现象极具误导性: "地形里 kip 完全不动" / "平地里莫名翻身", 而且
**同一段代码换个进程跑就正常**。

规则: **一个进程只建一个环境**; 对照实验用两个进程 (本项目
`sim/probe_getup_mjx_trace.py` 就是为此写的)。这与陷阱二 (WARP 按 buffer 地址缓存)
同源, 只是这次串的是**模型**而不是 buffer。

---

## 二、查看器使用指南（键位 / 判据 / 调度器）

# MuJoCo 交互式查看器中文指南

> 适用版本：MuJoCo 3.12.0（Python 交互式查看器，与官方 `simulate` 程序同一套界面）
> 对应命令：`mjv <模型.xml>` = `python -m mujoco.viewer --mjcf <模型.xml>`

---

## 一、你在看的是什么

`mjv` 打开的窗口就是 MuJoCo 自带的**交互式仿真查看器**：它一边用 OpenGL 渲染你的模型，一边实时推进物理仿真。
你机器上的界面分成四块：

```
┌────────────────────────────────────────────────────────┐
│ 3D 视图（模型在这里动）                │ 右面板        │
│  左上：实时倍速 %（如有）              │  Joint        │
│  顶部：PAUSE / LOADING 提示            │  Control      │
│  左下：Info 统计（时间/求解器/FPS...）  │  Equality     │
├─────────────┬──────────────────────────┴───────────────┤
│  左面板      │                                          │
│  File/Option/Simulation/Watch/Physics/               │
│  Rendering/Visualization/Group enable/Logging        │
└─────────────┴──────────────────────────────────────────┘
```

- **左面板（宽）**：模型、物理、渲染、视角的全部设置。
- **右面板（窄）**：关节滑条、控制输入、等式约束开关。
- 用 **Tab / Shift+Tab** 可以随时显示或隐藏右/左面板，看模型更清爽。

加载模型的三种方式：
1. `mjv models/mujoco_menagerie/unitree_go2/scene.xml`（就是你现在的用法）
2. 打开空查看器后，**把 XML 文件直接拖进窗口**
   - 例：`mjv` 后拖入 `models/mujoco_menagerie/franka_fr3/fr3.xml`
3. 程序里调用 `mujoco.viewer.launch(model, data)` 或 `launch_passive(...)`（写代码时用）

本地即有大量现成模型（`models/mujoco_menagerie/`），可以试试：

```bash
mjv models/mujoco_menagerie/unitree_go2/scene.xml          # 宇树 Go2 机器狗
mjv models/mujoco_menagerie/franka_fr3/scene.xml           # Franka 机械臂
mjv models/mujoco_menagerie/boston_dynamics_spot/scene.xml # 波士顿动力 Spot
```

---

## 二、鼠标 & 键盘操作总表（按 F1 随时调出）

这是你在窗口里按 **F1** 看到的帮助表，下面逐行翻译：

| 操作 | 英文原文 | 含义 |
|---|---|---|
| `Space`（空格） | Play / Pause | **播放 / 暂停仿真**，暂停时顶部会显示 PAUSE |
| `+` / `-` | Speed Up / Down | 加速 / 减速实时倍速（左上角会显示如 `50%`） |
| `←` / `→` | Step Back / Forward | 暂停时**后退 / 前进一个物理步**（配合 History 回放） |
| `Tab` / `Shift+Tab` | Toggle Right / Left UI | 显示/隐藏**右面板 / 左面板** |
| `[` / `]` | Cycle cameras | 在模型自带相机之间**循环切换** |
| `Esc` | Free camera | 恢复**自由相机** |
| 双击物体 | Select | **选中**该物体（选中后才能拖拽施力） |
| `Page Up` | Select parent | 选中它的**父级物体** |
| 右键双击 | Center camera | 相机中心对准该点 |
| `Ctrl`+右键双击 | Tracking camera | **跟踪跟随**该物体 |
| 滚轮 / 中键拖 | Zoom | 缩放 |
| 左键拖动 | View Orbit | 旋转视角 |
| 右键拖动（Shift 加持变慢） | View Pan | 平移视角 |
| `Ctrl`+左键拖 | Object Rotate | 拖动**被选中物体旋转**（对它施加扰动力矩） |
| `Ctrl`+右键拖 | Object Translate | 拖动**被选中物体平移**（对它施加扰动力） |
| `F1` | Help | 帮助表 |
| `F2` | Info | 左下角统计信息 |
| `F3` | Profiler | 右上角性能分析图 |
| `F4` | Sensors | 右下角传感器波形 |
| `F5` | Full screen | 全屏 |
| 按住**面板标题栏右键** | Show UI shortcuts | 显示该面板所有项的快捷键提示 |
| 双击面板标题栏 | Expand/collapse all | 折叠 / 展开面板全部条目 |

> 💡 施加"外力"的完整操作：**左键双击某部件 → 按住 Ctrl 左键拖 = 旋转它；Ctrl 右键拖 = 平移它**。松手后物体按物理规律继续运动。这是查看器最常用的交互玩法。

---

## 三、左面板逐节详解

点每节标题可折叠/展开。

### 1. File（文件）
| 项 | 说明 |
|---|---|
| `Save xml` | 把当前模型导出为 XML（mjModel→mjSpec→文件） |
| `Save mjb` | 导出为二进制模型 .mjb（加载更快，不可读） |
| `Print model` | 打印模型结构全文（保存为 .txt） |
| `Print data` | 打印当前 mjData 状态 |
| `Quit` | 退出查看器 |
| `Screenshot` | **截图**，保存为启动目录下的 `screenshot.png` |

### 2. Option（查看器外观选项）
只影响查看器本身，不影响物理：字体大小（Font）、UI 配色主题（Color）、控件间距（Spacing）、垂直同步（Vertical Sync，防画面撕裂）等。
拖动面板边缘的小竖条可以调整面板宽度；数字输入框可直接双击修改。

### 3. Simulation（仿真控制）★最常用
| 项 | 说明 |
|---|---|
| `Pause` | 暂停/继续（= 空格） |
| `Reset` | 重置仿真到初始状态（= 退格键） |
| `Reload` | 重新从文件加载模型 |
| `Key: Load key / Save key` | **关键帧**存取：`Save key` 把当前姿势/状态存进缓冲，`Load key` 取回 |
| `Key` 滑条 | 直接跳到模型 XML 里预定义的第 N 个 keyframe（`<keyframe>` 定义） |
| `Real time` 滑条 | 实时倍速（与 `+`/`-` 等效），100% = 1 秒仿真跑 1 秒实际时间 |
| `History` 滑条 | **时间回放**：查看器一直在记录状态，拖回去可以倒放历史（暂停时配合 `←/→` 单步） |

### 4. Watch（观察/追踪）
- `Camera`：选 `Free`（自由）或 `Tracking`（跟踪），以及模型 XML 里定义的所有 `<camera>`（切换会看到预设机位）。
- `Tracking`：选择要跟踪的物体（body/相机模式配合用）。

### 5. Physics（物理参数）★调参重点
全部对应物理引擎 `mjOption`，**改动立即生效**：

**求解设置**
- `Integrator`：积分器。Euler（默认）/ RK4（更准更慢）/ implicit / implicitfast（关节阻尼大时更稳）
- `Cone`：摩擦锥模型。Pyramidal / Elliptic
- `Jacobian`：雅可比矩阵 Dense / Sparse / Auto
- `Solver`：约束求解器。PGS / CG / **Newton**（默认，通常最快）

**数值上限**
- `Timestep`：步长（秒）。改小更准、更慢
- `Iterations` / `Tolerance`：求解器迭代数 / 收敛容差
- `LS Iter` / `LS Tol`：线搜索参数；`Noslip Iter/Tol`：无滑修正；`CCD Iter/Tol`：连续碰撞检测；`Sleep Tol`：休眠阈值；`SDF Iter/Init`：SDF 求解参数

**环境力**
- `Gravity`：重力（默认 0 0 -9.81）
- `Wind`：风；`Magnetic`：磁场；`Density`：空气密度；`Viscosity`：介质粘度（水感）

**物理参数**
- `Imp Ratio`： impedance 摩擦比
- `Margin` / `Sol Imp` / `Sol Ref` / `Friction`：接触求解器覆盖参数
- `Contact Override`：运行时强制覆盖接触参数

**Disable / Enable 复选框**：临时关闭/打开某个机制，如 `Contact`（全部接触失效！）、`Gravity`、`Energy`（能量统计）、`Island`（分岛求解）、`Sleep`（休眠加速）…… 勾选 = 当前会话生效，不写回 XML。

**Actuator Group Enable**：`Act Group 0-5`，勾掉后整组执行器失效。

### 6. Rendering（渲染）
| 项 | 说明 |
|---|---|
| `Camera` | 切换机位：Free / Tracking / 模型相机列表 |
| `Label` | 画面上显示名称标签：None / Body / Joint / Geom / Site / Contact / Force… |
| `Frame` | 显示**坐标系**：None / Body / Geom / World…（调试朝向必用） |
| `Copy camera` | 把当前机位复制成 `<camera .../>` XML 文本 |

**Model Elements**（可视化元素开关，对应 mjVIS 标志，常用的有）：
- `Contact point` 接触点、`Contact force` 接触力红色箭头、`Contact torque` 扭矩、`Contact friction/split/gap` 摩擦/分解/间隙
- `COM` 质心、`Inertia` 惯性椭球、`Joint` 关节轴、`Tendon` 腱、`Actuator` 执行机构
- `Select` 选中指示、`Transparency` 透明度、`Perturb` 扰动手势
- `Tree depth` / `Flex layer`：BVH 包围盒/柔性件层级

**OpenGL Effects**（渲染效果开关，mjRND 标志）：
- `Shadow` 阴影、`Fog` 雾、`Haze` 朦胧、`Reflection` 倒影、`Add noise` 噪点、`Stereo` 立体、`Wireframe` 线框等 —— 每个勾就是一种 OpenGL 效果，画面卡时先关 Shadow/Reflection。

### 7. Visualization（查看参数）
- **头灯**：Active / Ambient / Diffuse / Specular
- `Orthographic` 正交投影（工程制图感）；`Field of view` 视场角
- `Center` / `Azimuth` / `Elevation`：观察点、方位角、仰角
- `Align` 按钮：自动框住整个模型（视角乱了点它）
- `Extent` / `All (meansize)`：场景缩放基准
- **Scale** 组：各种指示器的大小（力箭头 Force、质心 Com、接触宽高 Contact、关节轴 Joint、坐标系 Frame……）
- **Color** 组：这些指示器的颜色

### 8. Group enable（分组可见性）
MuJoCo 里 geom/site/joint/tendon/actuator/flex/skin 各有 `group 0~5` 分组，XML 里常把"碰撞体"和"外观体"分组。
这里每个复选框**临时隐藏/显示对应组**——模型乱七八糟的调试网格瞬间清空就是这么干的（比如勾掉 `Geom 1` 只留视觉 mesh）。

### 9. Logging（日志）
- `Console`：把日志输出到终端；`File`：写入日志文件
- `Info topics`：按主题勾选要记录的日志内容

---

## 四、右面板逐节详解

### 1. Joint（关节）
模型里每个**滑块(slide)/铰链(hinge)**关节一个滑条（滑块关节默认 ±1，铰链默认 ±π；模型给限位则用限位）。
拖动 = 直接设置该关节的**位置 qpos**（暂停时改姿势，播放时则持续生效后松开回弹）。
自由关节(free)/球关节(ball)不显示滑条；在左面板 `Group enable → Joint` 中关掉某组，对应滑条会消失。

### 2. Control（控制）★给机器人下指令
- `Clear all`：所有控制一键归零。
- 模型里的每个**执行器(actuator)**一个滑条，拖动 = 设置它的控制信号 **ctrl**（比如电机的目标位置/力矩）。
- 范围来自模型定义的 `ctrlrange`；模型没定义执行器时此节为空。
- 配合暂停：暂停 → 拖 ctrl → 空格播放，看执行器怎么驱动机器人。

### 3. Equality（等式约束）
模型 XML 里定义的每条 `<equality>`（焊接 weld / 连接连体 connect 等）一行复选框，勾/不勾 = **运行时开/关这条约束**（比如解开某条焊接）。

---

## 五、左下角 Info 统计栏（F2 开关）

| 字段 | 含义 |
|---|---|
| `Time` | 仿真时间（秒），与画面右上/左上无关 |
| `Size` | 约束总数 nefc（括号内为接触点数 ncon） |
| `CPU` | 单步物理耗时（毫秒） |
| `Solver` | 求解器误差（log10）+ 实际迭代次数 |
| `FPS` | 渲染帧率 |
| `Memory` | mjData 内存池使用率（如 `12.3% of 3.6 MiB`） |
| `Energy` | （Physics 里开启 Energy 后出现）总能量 |
| `Islands` | 约束孤岛数（默认出现） |

调试心法：**Size 里 con 数大 → 接触多 → 注意 Solver 是否收敛（F3 看曲线）；CPU 一行是物理耗时，FPS 是渲染耗时。**

---

## 六、覆盖层开关速查

- `F1` 帮助 · `F2` Info · `F3` Profiler（右上 4 张图：CPU 耗时 / 规模 / 求解收敛曲线 Counts/Convergence）· `F4` 传感器波形 · `F5` 全屏
- 暂停时顶部会显示 **PAUSE(n)** = 正在回放第 n 个历史状态。

---

## 七、高频工作流清单

```text
换姿势        暂停(Space) → 左面板 Joint 拖滑条 / Ctrl 拖物体 → 保存 key / Screenshot
施加外力      左键双击选中部件 → Ctrl+左拖(旋转) / Ctrl+右拖(平移)
看接触/力     F2 开 Info + Rendering 勾 Contact point / Contact force
隐藏调试网格  Group enable 里勾掉不需要的分组（或 Scene XML 里改 group）
回放刚才几秒  暂停 → 拖 History 滑条 → ←/→ 逐帧找状态
换机位        [ ] 循环模型相机，Esc 回自由机位；Ctrl+右双击跟踪某物体
截图          左面板 File → Screenshot（存到启动目录 screenshot.png）
慢动作        按 "-" 把倍速调低（左上角显示 %）
看性能        F3：物理每步耗时、约束数量、收敛曲线
```

---

## 八、与命令行的对应（写代码时会用到）

| 你在 GUI 里做的事 | Python 里等价的东西 |
|---|---|
| 拖动 ctrl 滑条 | `data.ctrl[i] = ...` |
| Joint 滑条 | `data.qpos[adr] = ...` |
| Rendering 的勾选 | `opt.flags[mjtVisFlag.mjVIS_CONTACTPOINT] = 1` |
| OpenGL Effects | `scn.flags[mjtRndFlag.mjRND_SHADOW] = 0` |
| Camera 切换 | `cam.type / cam.fixedcamid / cam.trackbodyid` |
| Space 暂停 | 你的循环里自己控制 `mj_step` 调不调 |

被动查看器示例（`launch_passive`）见官方文档：
<https://mujoco.readthedocs.io/en/stable/python.html#interactive-viewer>

---

*本指南基于你本机 3.12.0 的实际界面字符串与 simulate.cc 源码核对生成；个别条目名称随版本略有差异，以 F1 帮助窗口为准。*