# Reading the TensorBoard logs of the navigation training runs

This page applies to every navigation run trained with `Navigation/scripts/train.py` (2D LiDAR, MLP policy) or
`Navigation/scripts/train_3d.py` (3D LiDAR, CNN policy). Both write the same scalars.

```bash
tensorboard --logdir Navigation/runs_3d/NAV04_static_lidar3d-r4_cnn     # one experiment (3D)
tensorboard --logdir Navigation/runs/NAV01_static_initial              # one experiment (2D)
tensorboard --logdir_spec nav01:Navigation/runs/NAV01_static_initial,nav04:Navigation/runs_3d/NAV04_static_lidar3d-r4_cnn
```

On a remote server, add `--port 6006` and forward the port (`ssh -L 6006:localhost:6006 <server>`).

## Basics

* **x-axis:** the training iteration. Each iteration collects 32 control steps (6.4 s of simulated time) in every
  environment (4,096 × 32 ≈ 131k transitions), then runs the PPO update.
  * Exception: tags ending in `/time` use wall-clock seconds since the start of training.
* **Two groups of statistics:**
  * `Episode*` (written by the environment) = mean over **all episodes that ended during the iteration**, typically
    hundreds to thousands with 4,096 environments;
  * `Train/mean_*` (written by RSL-RL) = mean over the **last 100 finished episodes**, a small, noisier sample.
* **Early iterations are biased.** Training starts with random episode lengths (`init_at_random_ep_len=True`), so the
  first ~350 steps (1–2 iterations) include episodes cut short artificially.
* **Smoothing:** the TensorBoard smoothing slider only affects display; values around 0.6–0.9 make trends easier to see.

## Episode statistics (from the environment)

### `Episode_Termination/*`: how episodes end

Each value is the fraction of the episodes that ended in that iteration, so the four add up to 1.

| Tag | Meaning | What you want |
|---|---|---|
| `goal` | robot within `goal_tolerance` (0.20 m) of the goal | rises toward 1 |
| `safety_violation` | footprint clearance to a wall, baffle or obstacle < `safety_margin` (0.05 m); episode terminated | falls toward 0 |
| `physical_collision` | clearance ≤ 0, i.e. actual contact (a stricter subset of a safety violation, reported separately) | stays near 0 |
| `timeout` | 70 s (350 steps) without reaching the goal or a violation | falls once the policy reaches the goal |

Typical course:
1. At the start, mostly `safety_violation`: the random policy drives into walls.
2. Then mostly `timeout`: the policy has learned to avoid walls but not yet to reach the goal.
3. Then `goal` grows.

Being stuck in phase 2 means the policy is cautious but lacks direction (look at `Episode_Reward/progress`).

### `Episode/length_steps`, `Episode/min_clearance_m`

| Tag | Meaning |
|---|---|
| `length_steps` | mean episode length in control steps (5 Hz; 350 = the 70 s timeout). High while the policy only survives (timeouts); it then decreases as goals are reached faster. |
| `min_clearance_m` | mean, over episodes, of the smallest footprint clearance reached during the episode (m). Below 0.05 means safety violations; a healthy navigating policy stays above it. |

### `Episode_Reward/*`: the reward split into its terms

Each value is the **sum of that reward term over an episode**, averaged over the episodes that ended in the
iteration. The total episode return is the sum of the six terms. The terms (`rewards.*` in the experiment config,
`CorridorNavEnv._get_rewards`), with the NAV01 / NAV04 values:

| Tag | Per control step | Per-episode range and reading |
|---|---|---|
| `progress` | `10 × (previous goal distance − goal distance)` | about 10 × (net reduction of the straight-line goal distance). Start–goal distances are 5.9–6.8 m (mean 6.3 m), so a successful episode gets about +60. Rises as the policy learns to move toward the goal. |
| `proximity` | `−1 × max(0, 1 − clearance / 0.15)²`, active only when the clearance is below 0.15 m | ≤ 0. Penalizes driving close to walls and obstacles; should shrink in magnitude over training. |
| `time` | `−0.1 /s × 0.2 s = −0.02` | equals −0.02 × episode length: −7 for a full 70 s timeout. Smaller in magnitude = shorter episodes. |
| `goal` | `+30` on the step the goal is reached | = 30 × fraction of successful episodes. |
| `collision` | `−30` on a safety violation (and therefore also on contact) | = −30 × fraction of violations. Its magnitude should go to 0. |
| `timeout` | `0` (penalty disabled in NAV01–NAV04) | always 0. |

## Training progress (from RSL-RL)

| Tag | Meaning |
|---|---|
| `Train/mean_reward` | mean **total return** (sum of all reward terms) of the last 100 finished episodes. The main "is it learning" curve. For a reference: a successful episode scores about +60 (progress) + 30 (goal) − a few (time, proximity), i.e. roughly +85. A wall crash after little progress scores about −30. |
| `Train/mean_episode_length` | mean length (control steps) of the last 100 finished episodes. |
| `Train/mean_reward/time`, `Train/mean_episode_length/time` | the same two curves with wall-clock seconds on the x-axis (useful to compare runs of different speed, e.g. 2D vs 3D). |

## PPO internals (`Loss/`, `Policy/`)

| Tag | Meaning |
|---|---|
| `Loss/value` | mean squared error of the critic (value function) against the returns, averaged over the update. Large and noisy early (returns change as the policy changes), then lower. Spikes when outcomes change abruptly (e.g. goals start to be reached). Not expected to reach 0. |
| `Loss/surrogate` | PPO clipped surrogate (policy-gradient) loss. Small (about ±0.01) and centred near 0 by construction (normalized advantages). Its value says little about learning, but persistent large values or NaN indicate a problem. |
| `Loss/entropy` | entropy of the Gaussian action distribution: for the two actions (v, w), 2 × (1.419 + ln σ) with σ = `Policy/mean_std`. Starts at 2.84 (σ = 1) and decreases as the policy becomes more deterministic. A fast collapse (σ → 0.1 within a few dozen iterations) can mean premature convergence. The 0.005 entropy bonus slows the decrease. |
| `Loss/learning_rate` | current learning rate. Adaptive schedule: from 3e-4, multiplied by 1.5 when the policy change per update is small (KL < 0.005) and divided by 1.5 when it is large (KL > 0.02). Bounds 1e-5 to 1e-2. Rising = updates are conservative; falling = updates are too large. |
| `Policy/mean_std` | mean standard deviation of the action noise, in normalized action units (actions are clipped to [−1, 1] and scaled to v_max = 0.22 m/s, w_max = 2 rad/s). Starts at 1.0. It measures exploration; evaluations use the mean action (σ ignored). |

## Speed (`Perf/`)

| Tag | Meaning |
|---|---|
| `Perf/total_fps` | environment steps per second over the whole iteration: 4,096 × 32 / (collection + learning time). About 10.6k for NAV04 (3D LiDAR + CNN) and about 21k for NAV01 (2D). |
| `Perf/collection_time` | seconds per iteration spent simulating and running the policy (physics, ray casting, observations). About 7.9 s for NAV04. |
| `Perf/learning_time` | seconds per iteration of PPO updates (5 epochs × 4 mini-batches). About 4.5 s for NAV04; larger than for 2D because of the CNNs. |

## Reading the curves together

| Pattern | Likely meaning |
|---|---|
| `goal` ↑, `safety_violation` ↓, `Train/mean_reward` ↑, `length_steps` ↓ (after an initial rise) | healthy learning |
| `timeout` dominates for a long time, `progress` flat | the policy avoids walls but does not head for the goal |
| `safety_violation` stays high, `min_clearance_m` low | collisions are not being avoided (check observations, the reward scale) |
| `mean_std` collapses early while success stays low | premature convergence; consider a higher entropy coefficient |
| `learning_rate` at its 1e-5 floor, `value` loss exploding | unstable updates |
| `total_fps` drops during a run | another process on the GPU, or thermal throttling |

Success rates in TensorBoard are measured on **training** episodes: stochastic actions, training goals and
randomization. The reference numbers come from `evaluate.py` / `evaluate_3d.py` (protocol `ref_static_c10`, held-out
obstacle layouts and goal, deterministic actions).
