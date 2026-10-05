**English** | [简体中文](README.md)

# MJX Go1 Get-Up

Locomotion and fall-recovery policies for the Unitree Go1. Physics runs in MJX (MuJoCo Warp),
training uses Brax PPO, and the whole pipeline fits on one RTX 5060 Laptop GPU (8 GB).

![Walking policy on the procedural terrain](docs/assets/walk-terrain.png)

## Contents

Three parts, all using the same MuJoCo model at 50 Hz:

- **Walk policy** `envs/go1_walk.py` — velocity-tracking trot on a procedurally generated height
  field (bumps, plateaus, two staircases). 91-dim observation, 12 absolute joint-position targets.
- **Get-up policy** `envs/go1_getup_v2.py` — stands up from fallen poses. The observation is a
  5-frame history of 42-dim frames; the action is `target = home_pose + 0.5 * clip(a, ±8)`.
- **Scheduler** `sim/view_go1.py` — detects a fall, switches to the get-up policy, and hands
  control back to the walk policy once the robot has stood still for 0.5 s.

## Results

| metric | value | script |
|---|---|---|
| Get-up success after a real walking fall | 77.3% (strict) / 81.2% (lenient), n = 128 | `sim/probe_handover.py --stage realfall` |
| Time to stand | 5.71 s → 4.89 s (lenient criterion) | same |
| Fall rate after hand-over | 7.9% (38.3% with a 10-frame action blend, 73.6% with 30) | `sim/probe_handover.py --stage cycle` |
| Walk policy (60M steps) | eval_reward 2339.7, mean episode 670/750 steps | `train/train_go1.py` logs |
| Stairs | down 62–87%, up 0% | `sim/eval_stairs.py` |
| Training speed | get-up 22k–25k steps/s (40M in ~30 min); walk ~4.4k steps/s | RTX 5060 Laptop 8 GB |

All of these come from the scripts listed above; the commands are in Usage. The get-up criteria
(the `up_z` threshold, how many consecutive frames count as standing) are defined in
`docs/status.md` — the number changes a lot with the criterion, so check the definition first.

## Notes from development

- **Get-up success depends mostly on the criterion, not the policy.** The policy usually reaches
  the standing pose (median best `up_z` inside the height band: 0.999); what failed was the
  "25 consecutive frames" rule. Changing the counter to +1 when satisfied and −1 when not moved
  success from 77.3% to 81.2% and cut time-to-stand by 0.8 s. Using `up_z` = 0.99 (8.1° tilt)
  instead of 0.95 (18.2°) changes the success rate by an order of magnitude.
- **Don't blend the two policies' actions at hand-over.** The get-up policy stands up by pushing
  against the action clip, so the median `|a|` is exactly the clip value 8.0. Blending its final
  target into the walk policy's target raises the fall rate to 38.3% with a 10-frame blend and
  73.6% with 30; a hard switch gives 7.9%.
- **Anchored actions train much more easily than incremental ones.** With
  `target = qpos + 0.5 * a` the policy has to learn joint angles from scratch. Switching to
  `target = home_pose + 0.5 * clip(a, ±8)` (the recipe from get-up-isaaclab), plus two wide
  Gaussian rewards and no posture termination, is what took the task from 0% to about 80%.

## Installation

```bash
git clone https://github.com/TC635807/mjx-go1-getup.git
cd mjx-go1-getup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Developed with Python 3.14, JAX 0.7.2, MuJoCo 3.12, Brax 0.14.2 and warp-lang 1.16. Training needs
a CUDA 12 GPU. The environment variables that matter (`XLA_PYTHON_CLIENT_MEM_FRACTION`,
`LP_NUM_THREADS`, `GALLIUM_DRIVER`) and the WSL rendering issues are written up in
`docs/viewer-guide.md`.

## Usage

### Viewer

```bash
python sim/view_go1.py \
    --pkl policies/go1_walk_policy.pkl \
    --getup_pkl policies/go1_getup_policy.pkl \
    --getup_obs history --getup_action anchored --cmd 0
```

W/S change the forward velocity, A/D turn, R resets, Q quits. With `--cmd 0` the robot walks in
place, which is the clearest way to watch recoveries. The default `--cmd 0.5` reaches the terrain
edge in 8–16 s (out of bounds resets, it does not trigger a get-up in mid-air).

### Evaluating the get-up policy

```bash
python sim/eval_getup.py --task getup_v2 --getup2_action_clip 8.0 \
    --up_th 0.95 --z_lo 0.20 --z_hi 0.34 \
    --pkl policies/go1_getup_policy.pkl --envs 256 --steps 400
```

> [!WARNING]
> `--getup2_action_clip` must match training (8.0). With the config default of 3.0 the policy's
> actions get clipped and the reported success rate is a fake 0%.

### Reproducing the success rate

```bash
python sim/probe_handover.py --stage realfall --envs 128 --steps 1500 \
    --getup_steps 600 --cmd 0.5 --bound_xy 5.0
```

### Training

```bash
# get-up policy: 40M steps, about 30 minutes on an RTX 5060 Laptop
python -u -m train.train_getup --task getup_v2 --action_clip 8.0 \
    --num_timesteps 40000000 --num_evals 16 --episode_length 400 \
    --num_minibatches 4 --updates_per_batch 5 --lr 5e-4 --entropy 0.005 --init_noise_std 1.0

# walk policy: 60M steps
python -u -m train.train_go1 --num_timesteps 60000000
```

Checkpoints go to `logs/`, policies to `policies/`. Two self-checks worth running before changing
code: `python -m train.train_getup --dry_run` (builds the environment and network only) and
`python sim/probe_getup_v2.py` (environment self-test).

## Repository layout

```
mjx-go1-getup/
├── envs/                     # MJX environments (physics + task definition)
│   ├── go1_walk.py           #   walk: terrain, reward, termination, 91-dim obs
│   ├── go1_getup.py          #   get-up v1: incremental actions (kept for comparison)
│   ├── go1_getup_residual.py #   get-up: keyframe state machine + learned residual
│   └── go1_getup_v2.py       #   get-up v2: anchored actions, wide Gaussians, history
├── train/
│   ├── train_go1.py          # walk training (brax PPO)
│   └── train_getup.py        # get-up training: --task getup | getup_res | getup_v2
├── sim/
│   ├── view_go1.py           # interactive viewer + walk/get-up scheduler
│   ├── watch_v20_fast.py     # high-FPS watcher
│   ├── eval_getup.py         # get-up success / posture buckets / time-to-stand
│   ├── eval_walk.py          # walk evaluation on a fixed spawn set
│   ├── eval_stairs.py        # stair-climbing success
│   ├── probe_handover.py     # quantitative probe of the hand-over moment
│   ├── probe_getup_v2.py     # get-up environment self-test
│   ├── probe_standability.py # is the standing criterion reachable?
│   ├── probe_nefc.py         # contact-constraint overflow probe
│   ├── gen_terrain.py        # terrain generator (hfield PNG + json)
│   ├── getup_keyframe.py     # scripted keyframe get-up (baseline + viewer fallback)
│   └── common.py             # policy loading (pkl / orbax checkpoint)
├── models/go1/               # MuJoCo model (MJCF + meshes + terrain)
├── policies/                 # released policies: walk v23@60M, get-up v2f
└── docs/                     # experiment log, status, viewer guide, get-up recipes
```

## Documentation

- `docs/status.md` — current results and the exact definition of every criterion (start here)
- `docs/experiment-log.md` — the development log in chronological order, including failed attempts
- `docs/getup-recipes.md` — comparison with public get-up implementations and a SOTA survey
- `docs/viewer-guide.md` — per-frame cost breakdown and common traps when writing an MJX viewer

Chinese version: [README.md](README.md).

## Known issues

- `njmax=256` is too small for the get-up task. A fallen robot needs roughly 1300 constraints per
  world, and the training logs show `nefc overflow` warnings (719 events / 384k world-steps).
  Retraining with `njmax=2048` is an open item; `sim/probe_nefc.py` reproduces it. It does not
  explain the observed instability, but it does affect contact accuracy while the robot is down.
- The get-up policy stands by saturating the action clip (median `|action|` 8.0 = clip, joint
  velocity still 11.5 rad/s). Adding a "stay quiet while standing" reward term, or requiring a
  still posture for the success bonus, is the next step.
- Training spawns give 88.7% versus 77.3% from real walking falls, an 11-point gap. The plan is to
  fine-tune on a pose pool collected from real falls (`sim/make_getup_posepool.py`).
- Up-stairs does not work (0% on the fixed staircase), and the walk policy is not perfectly
  mirror-symmetric (hip RR/RL drift about 16°).

## Acknowledgements

- [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) and
  [Brax](https://github.com/google/brax) — the training stack used here
- [get-up-isaaclab](https://github.com/iit-DLSLab/get-up-isaaclab) (IIT-DLSLab) — the recipe behind
  `envs/go1_getup_v2.py`
- [HoST](https://github.com/InternRobotics/HoST) and
  [HumanUP](https://github.com/RunpeiDong/HumanUP) — multi-critic and two-stage curricula
- [quadruped-rl-locomotion](https://github.com/nimazareian/quadruped-rl-locomotion) — the walking
  task was reproduced from this repository
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) — the Go1 MJCF

## License

[MIT](LICENSE)
