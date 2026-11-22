# Phase 2: RL-Navigation repository analysis and Isaac Lab migration feasibility

**Date:** 2026-11-21
**Subject:** `~/isaac_lab/nav_reference/RL-Navigation` (`sahars93/RL-Navigation`, HEAD `01fc8f8`, last commit 2025-05-02)
**Target:** Isaac Lab 2.3.2 (fork `c2277524`) / Isaac Sim 5.1.0, conda env `env_isaaclab`
> **Update (Phase 2b):** the faithful-port strategy (§§5–8) is superseded by `PHASE2B_SCENARIOS_AND_REVISED_PLAN.md` (native Isaac Lab environment). §§1–4, 9, 10 remain valid background.

**Scope:** analysis and planning only. The original repository, Isaac Lab and the conda env were not modified (`git status` is clean for RL-Navigation; the fork is unchanged since phase 1).

### How claims are labelled

| Tag | Meaning |
|---|---|
| **[V]** | Verified: read in the source code, or measured/inspected on this server |
| **[P]** | Stated in the paper (arXiv 2501.02902v1, read in full from the PDF) or its appendix PDF in the repo |
| **[A]** | Assumption or inference: not directly verified; the reasoning is given |

Supporting scripts and raw outputs are in `evidence/`.

---

## 0. Executive summary

1. **The repository cannot be run as committed [V].**
   * `utils/task_util.py` imports 12 task modules (`ant.py`, `cartpole.py`, …) that are not in the repo.
   * All three navigation tasks define the same class `JetbotTask`, and only `tasks/jetbot.py` is registered.
   * The obstacle scenes the tasks load (`obstacles_dynamic.usd`, `obstacles4.usd`) are **missing**. The committed `obstacles.usd` is a different 4 × 4 m arena that is incompatible with the hard-coded start and goal positions.

   The original training scene therefore cannot be recovered exactly. A faithful *methodology* reproduction is feasible; an exact *number-for-number* reproduction is not.
2. **The code, configs and paper disagree on several points.** Examples: MLP [64, 64] in the YAML vs [256, 128, 64] in the paper and the shipped models; LSTM with 64 units (ONNX) vs 128 (paper); 200 vs 1500 training epochs; a stepwise vs an exponential proximity penalty.

   One shipped model, `static_120_73reward.onnx`, has exactly the 133-dim observation produced by `tasks/jetbot.py`, a [256, 128, 64] MLP, and a name matching the paper's "maximum of 73 rewards" for the static MLP. That makes **`jetbot.py` + [256, 128, 64] + rl_games PPO** the best-supported reference configuration [V/A].
3. The legacy implementation is **CPU-bound by design** [V]. It trains with `pipeline=cpu`, reads the PhysX LiDAR in a per-env Python loop, runs a per-env NumPy differential-drive controller with `.item()` calls, and does per-env Euler conversions and per-env USD attribute writes for moving obstacles. All of these map to batched GPU operations in Isaac Lab.
4. **The LiDAR, including moving obstacles, reproduces well with Isaac Lab's `MultiMeshRayCaster` [V, measured].**
   * Moving obstacles: **0 range errors in 1,191,843 checks** with 4096 envs and obstacles moving every step.
   * Throughput: 538k env-steps/s, including a 360-ray scan each step.
   * Negative control: disabling transform tracking produced 88.5% errors, so the check is meaningful.
   * **Caveat found:** moving *kinematic* bodies by writing poses every step crashes PhysX (CUDA illegal memory access). Gravity-free *dynamic* bodies driven by velocity commands (the legacy approach) or pose writes work.
5. **Recommendation:**
   * a **Direct workflow** (`DirectRLEnv`), a near one-to-one structural port of the OIGE `RLTask`;
   * **rl_games** (installed 1.6.1 vs the original 1.6.0) for the reproduction stage, so the PPO hyperparameters carry over unchanged;
   * RSL-RL (built-in LSTM and ONNX export) as the later default;
   * an in-house GPU-batched evaluation harness.

   **Estimated effort: about 22–35 person-days (5–8 calendar weeks)** to a validated static + dynamic + LSTM reproduction with evaluation and ONNX export.
6. **Long-term risk (§9):** a faithful port would carry legacy assumptions into phase 3. The most serious is that **collision and proximity penalties are computed from the policy's own LiDAR**. Once the policy controls where it looks, it could learn to avoid penalties by not looking. §9 lists 13 such items and the low-cost design choices that prevent them, adding about 4–6 person-days.

---

## 1. Repository architecture and execution flow

### 1.1 What's in the repo [V]
The repo is a fork of NVIDIA OmniIsaacGymEnvs `2022.2.1` (Isaac Sim 2022.2.1, Python 3.7) with navigation additions. Most of the 244 files are upstream OIGE leftovers. The navigation-relevant parts:

| Path | Responsibility |
|---|---|
| `omniisaacgymenvs/tasks/jetbot.py` (349 lines) | **Static-obstacle navigation task** (the main task; the only registered one) |
| `omniisaacgymenvs/tasks/jetbot_dynamic.py` (420) | Same task + 2 moving obstacles (class also named `JetbotTask`, **not registered**) |
| `omniisaacgymenvs/tasks/dynamic_target.py` (322) | Variant with a moving target, a different obs/reward (**not registered**) |
| `tasks/base/rl_task.py` | OIGE `RLTask` base: buffers, GridCloner, collision filtering, post-physics flow |
| `robots/articulations/jetbot.py` | Loads Nucleus `Isaac/Robots/Jetbot/jetbot.usd` |
| `robots/controllers/differential_controller.py` | Unicycle → wheel speeds (NumPy, one env at a time) |
| `envs/vec_env_rlgames.py` | OIGE vec-env: action noise, clamp, `control_frequency_inv` physics substeps, obs noise |
| `utils/domain_randomization/randomize.py` | OIGE DR (obs/action noise, replicator physics DR) |
| `utils/task_util.py` | Task registry (imports 12 missing modules → ImportError) |
| `cfg/config.yaml`, `cfg/task/Jetbot.yaml`, `cfg/train/JetbotPPO.yaml` | Hydra configs |
| `scripts/rlgames_train.py` | Train/test entry point (rl_games `Runner`) |
| `scripts/rlgames_onnx_normalized.py`, `scripts/rlgames_onnx_lstm` | ONNX export (+ rollout check) for MLP / LSTM |
| `ROS/real_test.py` (ROS 1 `rospy`), `ROS/gazebo_test.py`, `ROS/gazebo_test_lastm.py`, `ROS/gazebo_test_ros2.py` (ROS 2 `rclpy`), `ROS/moving_obsctacle.py` | Deployment / Gazebo inference nodes, Gazebo obstacle mover |
| `onnx_models/*.onnx` (16), `saved_checkpoints/*.pth` (6) | Trained policies (see `evidence/onnx_inventory.txt`) |
| `assets/jetbot/*.usd`, `assets/ridgeback_franka/*` | Scene/robot USDs (none referenced by the tasks; see §1.8) |
| `Appendix__RL_Sim2Real_Isaac_Sim.pdf` | Tutorial appendix (code listings are images; no hyperparameters in text) |

Dependencies: `rl-games==1.6.0`, `hydra-core==1.1.1`, `omegaconf==2.1.1`, Isaac Sim 2022.2.1 (`omni.isaac.core`, `omni.isaac.range_sensor`, `omni.isaac.cloner`, `omni.replicator.isaac`), gym, scipy, cv2 [V].

### 1.2 Execution flow (training) [V]
```
python scripts/rlgames_train.py task=Jetbot pipeline=cpu        (README; cwd must be omniisaacgymenvs/)
 └─ hydra → VecEnvRLGames(headless, sim_device)                 (starts Isaac Sim)
 └─ initialize_task → SimConfig + JetbotTask(...)               (task_util.py)
      JetbotTask.__init__ → RLTask.__init__ (buffers, GridCloner) → action_space override [0.1,0.5]x[-0.5,0.5]
      env.set_task → set_up_scene: Jetbot + PhysX LiDAR prim, target cube, obstacle USD (env_0) → clone N envs
      post_reset: LiDAR interface, initial poses, potentials, reset_idx(all), DR setup
 └─ rl_games Runner(a2c_continuous) loop:
      policy → mu,σ → sample → clamp[-1,1] → rescale to action_space bounds        (rl_games preprocess_actions)
      VecEnvRLGames.step: (+action bias noise) → clamp(±clipActions=1) →
         task.pre_physics_step: reset_idx(envs flagged last step) → per-env diff-drive → wheel velocity targets
         world.step() × controlFrequencyInv (3)  (dt = 1/120 → policy at 40 Hz)
         task.post_physics_step: progress+1 → get_observations → calculate_metrics → is_done
      (+ per-episode obs bias noise) → rl_games
 └─ checkpoints: runs/Jetbot/nn/{Jetbot.pth (best), last_Jetbot_ep_<n>_rew_<r>.pth}
```
Note the OIGE convention: **resets are applied at the start of the next `pre_physics_step`**. The observation returned on a terminal step is the pre-reset state, and the first post-reset observation arrives one step later [V].

### 1.3 Simulation environment [V]
* Physics: `dt = 0.0083` (~1/120 s), TGS solver, 4 position / 0 velocity iterations, scene queries enabled (needed by the PhysX LiDAR), default ground plane, friction 1.0. Default 64 envs, `envSpacing 5.0`. `use_gpu_pipeline` follows `pipeline`; the README trains with **`pipeline=cpu`**.
* Scene per env:
  * Jetbot at (−2, 0, 0) with identity orientation;
  * a 0.1 m green target cube at (3.5, −2.5, 0), with its collision disabled;
  * an obstacle USD referenced at `/World/envs/env_0/Home/home`, loaded from `Path.cwd().parent/assets/jetbot/obstacles_dynamic.usd`, **which doesn't exist**.
* The commented-out code also references `imgs/new_cost2024.png` (an occupancy map for random start poses), which is absent too.

### 1.4 Robot [V]
* Nucleus `Isaac/Robots/Jetbot/jetbot.usd`. In Isaac Sim 5.1 the asset lives at `Isaac/Robots/NVIDIA/Jetbot/jetbot.usd`, with `left_wheel_joint`/`right_wheel_joint` velocity drives (stiffness 0, damping 174.5). Inspected on this server, see `evidence/usd_inventory.txt`.
* Differential drive: `ω_L = (2v − ωL)/(2r)`, `ω_R = (2v + ωL)/(2r)`, with r = 0.0325 m and L = 0.1125 m. The result is applied with `set_joint_velocity_targets`.
* The 5.1 USD places the wheel joints at y = ±0.05 m (a 0.10 m track). Whether the effective track is 0.10 m or 0.1125 m must be **calibrated [A]** (see M1).

### 1.5 LiDAR [V]
* Created at runtime with `RangeSensorCreateLidar` (PhysX scene-query LiDAR, `omni.isaac.range_sensor`) on `chassis/Lidar/Lidar` at +0.015 m.
* Parameters: hFOV 360°, resolution 1° (360 rays), vFOV 30°, vertical resolution 4°, **`high_lod=False`**, range [0.15, 2.0] m, rotation rate 0 (full scan every frame).
* `high_lod=False` restricts the PhysX LiDAR to a single horizontal scan line, so the sensor is effectively 2D **[A]**. This follows from the reshape: `get_linear_depth_data(...).squeeze().reshape(120, 3)` only works with 360 values.
* Read **per env in a Python loop** into NumPy, then **min-pooled in windows of 3 → 120 beams** (3° sectors).
* Assumed behaviour [A], to verify against Isaac Sim's PhysX LiDAR in M3:
  * no-hit returns read as the max range;
  * hits closer than 0.15 m;
  * the index-0 angle.

  The ROS nodes `np.roll` the real scan by 180° before pooling, which suggests that Isaac's index 0 points backwards (−180°) **[A]**.
* Paper: "120 scans … in the range of [0.15, 3] m" and "6-degree resolution" [P]. Both contradict the code, which uses 3° sectors and a 2 m max range.

### 1.6 Observations (`jetbot.py`, 133 dims) [V]
| Slice | Content |
|---|---|
| 0–119 | min-pooled LiDAR ranges (m), unnormalized |
| 120–124 | heading to goal (rad, wrapped to [−π, π]), **repeated 5×** |
| 125–129 | 3D Euclidean distance to goal (m), **repeated 5×** |
| 130–131 | chassis linear velocity x, y in the **world frame** |
| 132 | angular velocity z |

* The 5× repetition is an undocumented way of up-weighting the goal features.
* The paper describes the observation as `L_t, d_t, θ_t, v_{t−1}, ω_{t−1}` [P], i.e. previous commands. The code uses *measured world-frame* velocities instead.
* The policy also applies rl_games running mean/std normalization (`normalize_input: True`). There is no clipping (`clipObservations` is commented out → ∞).

### 1.7 Actions [V]
* 2D action `[v, ω]`. rl_games clamps the Gaussian sample to [−1, 1] and **linearly rescales** it to the task's `action_space`, **v ∈ [0.1, 0.5] m/s, ω ∈ [−0.5, 0.5] rad/s** (verified in installed rl_games `a2c_common.py:1181–1184`; the original 1.6.0 behaves the same [A]).
* The robot can never stop or reverse.
* Paper: same ranges [P].
* Differences in the other tasks: `dynamic_target.py` uses [0, 1] × [−1, 1] plus a +0.2 m/s offset.

### 1.8 Reward (`jetbot.py`) [V]
With `d` = 3D goal distance and `m` = min over the 120 beams:
```
r =  10·(d_prev − d)                       # progress
   + 0.1·(Φ − Φ_prev),  Φ = −d_xy / (1/60) # potential progress = 6·Δd_xy (dt hard-coded 1/60; control dt is 1/40)
   − 1·[m<0.40] − 2·[m<0.35] − 2·[m<0.30]  # stepwise proximity penalty: −1 / −3 / −5
   − 30·[m<0.25]                           # collision (terminal)
   + 30·[d<0.30]                           # goal (terminal)
   + 0.02·(1200 − step)·[d<0.30]           # time bonus on success
   − 30·[step ≥ 1199]                      # timeout (terminal)
```
Paper [P]: `r_distance = d_{t−1} − d_t`; an **exponential** collision penalty `−e^{−min_range}` below `Min`; a time reward of "remaining steps"; Goal / Collision / Max-length fixed rewards of 30; `Min` = 25 cm.

`dynamic_target.py` uses a different, sign-based reward (±0.1 for getting closer, a heading bonus, ±20) [V].

### 1.9 Terminations and resets [V]
* Terminations: collision (min LiDAR range < 0.25 m), goal (d < 0.30 m), timeout (1200 policy steps = **30 s** at 40 Hz). All are treated as terminal by rl_games; no time-out bootstrapping.
* `reset_idx`: the robot returns to its **fixed initial pose and yaw** with zero velocity, and the target to its fixed initial position. Potentials are re-initialised.
* The yaw, occupancy-map start and ±1 m start randomizations are all commented out in `jetbot.py`. The ±1 m randomization is active only in `dynamic_target.py`.
* `Jetbot.yaml` configures `articulation_views.jetbot_view.position` uniform [−2, 2] on reset. However, **no task calls `dr.physics_view.step_randomization(...)`**, which OIGE needs in order to trigger per-reset physics-view DR (only `in_hand_manipulation.py` calls it) [V]. The start position is therefore **almost certainly not randomized** [A]. This is consistent with the paper: "the robot's starting position was carefully chosen to align with the training setup, where it consistently began in relatively free space" [P].
* Noise [V]:
  * observations: a per-env, per-episode **constant Gaussian bias**, σ = 0.2 on all 133 dims;
  * actions: a per-episode bias, σ = 0.05 on the rescaled [v, ω];
  * both are applied in the vec-env, so rewards are computed on clean values.

### 1.10 Moving obstacles (`jetbot_dynamic.py`) [V]
* Two 0.25 × 0.7 × 0.2 m red `DynamicCuboid`s (collision on) at (1.7, 1.6) and (0.0, 2.2).
* Every step, per env, the code writes the USD `physics:velocity` attribute to ±0.8 m/s along y. This only takes effect with `use_flatcache: False` and the CPU path.
* The direction flips when the obstacle's displacement over a step is ≈ 0, i.e. when it is blocked by a wall. The motion pattern is identical in every env, and obstacles can be pushed by the robot.
* The reward, observation and action are identical to `jetbot.py` (133 dims), so static and dynamic checkpoints are interchangeable, which the curriculum relies on.

### 1.11 RL algorithm, network, hyperparameters [V unless noted]
* rl_games `a2c_continuous` PPO, `continuous_a2c_logstd`, shared actor-critic trunk (`separate: False`), **fixed** σ (`fixed_sigma`, init 0 → σ = 1), ELU.
* `JetbotPPO.yaml` settings:
  * network and normalization: MLP **[64, 64]**, input/value normalization;
  * returns: `reward_shaper.scale_value 0.1`, γ = 0.99, λ = 0.95, normalized advantages;
  * learning rate: 3e-4, adaptive schedule with KL threshold 0.008;
  * rollout and updates: horizon 128, minibatch 4096 (= 64 envs × 128 / 2), 4 mini-epochs;
  * losses: ε_clip 0.2, value clipping, critic coefficient 2, entropy 0, bounds-loss 1e-4, gradient norm 1.0;
  * length and checkpoints: `max_epochs` 200, `save_best_after` 50, `save_frequency` 25, `seq_len` 4 (only used by RNNs).
* **The paper and the shipped models differ** [P/V]:

| Item | `JetbotPPO.yaml` | Paper | Shipped ONNX (paper-era) |
|---|---|---|---|
| MLP | [64, 64] | [256, 128, 64] | [256, 128, 64] (some [256, 128, 64, 32]) |
| LSTM | none | 128 units, after input, before MLP | **64 units**, before MLP (3 models) |
| Iterations | 200 | 1500 | n/a |
| Envs | 64 | 64 | n/a |

  No LSTM train config is in the repo; its `rnn` block (`before_mlp`, `seq_len`, layers) is unknown [A].
* Observation sizes of the shipped models: 133, 124, 73, 70, 46, 30, 22, 12 and 362. They correspond to several unpublished task variants, e.g. 124 = 120 + θ + d + v_x + ω (the ROS node layout), 70 = 60 beams + 10, 362 = 360 + 2. Only **133** matches the committed `jetbot.py` [V].

### 1.12 Curriculum [P, no code]
* Paper: static training first, then "after 300 initial steps, we added dynamic objects and continued the training" [P]. Figure 3 compares "Static MLP", "Dynamic MLP", "Dynamic LSTM" and "Dynamic Curriculum".
* **No curriculum code exists** [V]. It was evidently done by hand: train `Jetbot` on the static scene, then restart from the checkpoint with the dynamic task, which is possible because their observations and actions are identical [A]. "300 steps" most likely means 300 rl_games epochs [A].

### 1.13 Saving, export, evaluation, deployment [V]
* **Saving:** rl_games `.pth` files (model, optimizer, running mean/std) under `runs/<name>/nn/`. The shipped `.pth` files need full (unsafe) unpickling, so they weren't loaded; the ONNX files were used as evidence instead.
* **Export:** `rlgames_onnx_normalized.py`
  * wraps `norm_obs` + `a2c_network`, traces it, and exports ONNX with outputs `mu`, `log_std`, `value`, so the observation normalization is baked in;
  * then rolls out one env using `mu` clipped to [−1, 1];
  * the output filename is hard-coded to `onnx_models/final360.onnx`.

  The LSTM exporter adds `out_state`/`hidden_state` I/O, but its rollout samples **stochastic** actions.
* **Evaluation in Isaac:** none quantitative in the repo. `test=True` play only. The paper's Isaac results are trajectory plots and distance-vs-time curves over 30 trials [P].
* **Deployment** (`ROS/real_test.py`, `gazebo_test_ros2.py`) [V]:
  * observation: 124 dims = 120 min-pooled beams (after a 180° `np.roll`, ranges clipped to ≤ 2, inf → 2 or 1.5) + θ + d + odom body-frame v_x + ω_z;
  * output: `mu` is **clipped** to [0.1, 0.4] / [−0.5, 0.5] (**not rescaled** from [−1, 1] as in training), then `v·0.8` and `ω + 0.1` are applied as hand-tuned corrections;
  * rates: 5 Hz (`real_test`) or 10 Hz (`gazebo_test_ros2`), versus **40 Hz in training**;
  * goal threshold 0.22 m;
  * `real_test.py` is ROS 1 and loads a non-existent `onnx_models/static.onnx` with `lstm=True`.
* **Paper:** ROS 2 Galactic on a TurtleBot 4 Lite (RPLIDAR A1M8, 2 m range, 3°), OptiTrack poses. Table 1: success 10/10, 10/10, 8/10, 7/10 [P].

### 1.14 Inconsistencies, bugs and undocumented behaviour (consolidated)
| # | Finding | Type |
|---|---|---|
| 1 | Repo not runnable: missing task modules in `task_util.py`; duplicate `JetbotTask` class names; dynamic tasks unregistered | [V] |
| 2 | Scene USDs `obstacles_dynamic.usd`, `obstacles4.usd` and occupancy map `imgs/new_cost2024.png` missing. The committed `obstacles.usd` (4 × 4 m walls at ±2 m, 4 cylinders r = 0.3 m) conflicts with the start (−2, 0), which is on a wall, and the target (3.5, −2.5), which is outside the arena | [V] |
| 3 | `prev_goal_distance` starts at 0 and is **never reset per env** → a large spurious progress reward at the first step of every episode: ≈ 10·(0 − 6.04) = −60 at the very first step, and 10·(d_end_prev − d_start) after each reset | [V] |
| 4 | Progress is counted twice (`10·Δd` + `0.1·ΔΦ` = `6·Δd_xy` with a hard-coded dt of 1/60, while the control dt is 1/40) | [V] |
| 5 | Proximity penalty is stepwise (−1/−3/−5), while the paper describes an exponential one | [V]/[P] |
| 6 | Goal test uses the 3D distance (robot base vs cube centre); potentials use 2D; `target_margin = 0.35` is unused | [V] |
| 7 | Heading and distance repeated 5× in the observation (undocumented); velocities in the world frame, while the paper and the ROS nodes use body-frame/previous commands | [V]/[P] |
| 8 | Configs disagree with the paper and models (MLP size, LSTM size, epochs, LiDAR range, "6-degree resolution") | [V]/[P] |
| 9 | Start-position DR is configured but never triggered (no `step_randomization` call) | [V] code, [A] effect |
| 10 | Train/deploy mismatch: clip vs rescale of `mu`, ×0.8 / +0.1 corrections, 5–10 Hz vs 40 Hz, 124- vs 133-dim observations, body- vs world-frame velocities | [V] |
| 11 | CPU-only paths (`pipeline=cpu`): per-env Python loops, NumPy controller with `.item()`, SciPy on pose tensors, USD velocity writes | [V] |
| 12 | Commented yaw-randomization code mixes Isaac's wxyz quaternions with SciPy's xyzw | [V] |
| 13 | LSTM export script samples stochastic actions; MLP export is deterministic | [V] |
| 14 | Resets are applied in `pre_physics_step`, so the reset state lags by one step (an OIGE convention that differs from Isaac Lab) | [V] |
| 15 | Collision is detected only via the LiDAR plane (no contact sensing); obstacles below or above the scan plane are invisible | [V] |
| 16 | README refers to `ROS/gazebo_test_lstm.py`; the file is `gazebo_test_lastm.py` | [V] |

---

## 2. Mapping: OmniIsaacGymEnvs → Isaac Lab 2.3.2

Official guide in the fork: `IsaacLab_Fork/docs/source/migration/migrating_from_omniisaacgymenvs.rst`. Isaac Lab's own Direct tasks (ant, cartpole, humanoid, …) are ports of the same OIGE tasks, which is a useful precedent.

| Legacy (OIGE 2022.2.1) | Isaac Lab 2.3.2 equivalent | Reuse level |
|---|---|---|
| `RLTask` / `BaseTask` | `isaaclab.envs.DirectRLEnv` + `DirectRLEnvCfg` | Re-implement (1:1 method map) |
| `set_up_scene` | `_setup_scene` (+ `scene.clone_environments`) | Adapt |
| `pre_physics_step` (reset + action) | `_pre_physics_step` (process actions) + `_apply_action` (each substep); resets are done by the framework after the step | Adapt |
| `get_observations` | `_get_observations` → `{"policy": tensor}` | Adapt (pure torch) |
| `calculate_metrics` | `_get_rewards` | Adapt (fix or flag bugs) |
| `is_done` | `_get_dones` → `(terminated, truncated)` | Adapt |
| `reset_idx`, `post_reset` | `_reset_idx(env_ids)` | Adapt |
| `controlFrequencyInv: 3`, `dt 1/120` | `decimation = 3`, `sim.dt = 1/120` | Direct reuse of the values |
| Hydra `task.yaml` | `@configclass` env cfg (Python) | Re-implement |
| `train/JetbotPPO.yaml` | `agents/rl_games_ppo_cfg.yaml` (same rl_games schema) | **Near-direct reuse** |
| `VecEnvRLGames` | `isaaclab_rl.rl_games.RlGamesVecEnvWrapper` + `scripts/reinforcement_learning/rl_games/train.py` | Framework-provided |
| `ArticulationView` (Jetbot) | `isaaclab.assets.Articulation` + `ArticulationCfg` (tutorial's `JETBOT_CONFIG` as a start) | Adapt |
| `DifferentialController` (NumPy, per env) | 2 vectorized torch lines → `set_joint_velocity_target` | Re-implement (trivial) |
| Target `DynamicCuboid` (no collision) | `VisualizationMarkers` or a kinematic `RigidObject`; goal stored as a tensor | Re-implement |
| Obstacle USD reference | Static colliders: `AssetBaseCfg`/`UsdFileCfg` (reuse `obstacles.usd`), or procedural `CuboidCfg`/`CylinderCfg` per env | Adapt / re-create (originals missing) |
| `DynamicCuboid` + USD velocity writes | `RigidObject` (or `RigidObjectCollection`), `disable_gravity=True`, **`write_root_velocity_to_sim`** (verified) | Re-implement |
| PhysX `RangeSensorCreateLidar` + per-env read | **`MultiMeshRayCaster`** + `LidarPatternCfg(channels=1, h=(−180,180), res=1°)`, batched GPU | Re-implement (verified feasible) |
| `quat_to_euler_angles` (per env) | `isaaclab.utils.math.euler_xyz_from_quat` (batched) | Direct |
| OIGE `Randomizer` on_reset obs/action noise | `DirectRLEnvCfg.observation_noise_model` / `action_noise_model` = `NoiseModelWithAdditiveBiasCfg` (bias resampled at each reset, verified in source) | Config |
| Replicator physics-view DR | `EventTermCfg`-style randomization in `_reset_idx` (torch) | Re-implement |
| Ground plane | `TerrainImporterCfg(terrain_type="plane")` (local, no network) | Direct |
| Camera / viewport setup | `ViewerCfg` | Direct (irrelevant headless) |
| `rlgames_onnx_normalized.py` | **No rl_games exporter in Isaac Lab** [V]; port this script (≈ 60 lines) | Adapt |
| ROS nodes | Out of scope now; later ROS 2 SIL in Isaac Sim | Re-implement later |

**Important behaviour change [V]:** Isaac Lab's rl_games wrapper exposes `Box(−clip_actions, clip_actions)` to rl_games, so rl_games' automatic rescale becomes identity. The legacy mapping [−1, 1] → v ∈ [0.1, 0.5], ω ∈ [−0.5, 0.5] **must be implemented in `_pre_physics_step`**, after clamping to [−1, 1].

### 2.1 Reusability summary
* **Reusable as-is:** physics timing values, reward coefficients, thresholds, PPO hyperparameters (rl_games YAML schema), network sizes, the noise magnitudes, `obstacles.usd` (as a scene option).
* **Needs adaptation:** robot config (drive damping, wheel-base calibration), observation and reward functions (vectorized; reproduce or fix the bugs behind flags), resets, the ONNX exporter.
* **Must be re-implemented:** the LiDAR sensor, moving-obstacle mechanics, scene construction (original scene missing), the task registry, the curriculum (as an explicit script/flag), quantitative evaluation (it never existed).

### 2.2 RL framework compatibility (installed versions)
| Library | Status for this project |
|---|---|
| **rl_games 1.6.1** | [V] installed; the original used 1.6.0 → the same PPO, normalization, bounds loss, adaptive-KL LR and `reward_shaper` semantics, and the YAML transfers almost verbatim. Isaac Lab ships train/play scripts but **no ONNX export**. LSTM is supported by rl_games itself (`network.rnn`); running it through Isaac Lab's wrapper is **not yet tested** [A]. `value_bootstrap` defaults to off, so time-outs are terminal, as in the legacy code [V]. **Recommended for the reproduction.** |
| **RSL-RL 5.0.1** | [V] `RslRlRNNModelCfg` (LSTM/GRU), built-in JIT + ONNX export (including RNN state I/O) in `play.py`. Different PPO details: no bounds loss, no reward shaper, different normalization and std parameterization, so results won't match rl_games exactly. **Recommended as the long-term default** after parity is shown (tighter Isaac Lab integration, better maintained). |
| **SKRL 2.1.0** | [V] installed; supports RNNs, but would need the most new config work. No advantage here. |
| **SB3** | Not suited (CPU-centric, no recurrent PPO in the core). |

---

## 3. LiDAR reproduction with GPU ray casting

### 3.1 Design
* **Sensor:** `MultiMeshRayCaster` attached to the Jetbot chassis at z ≈ 0.015 m above the chassis frame (legacy mount).
  * Pattern: `LidarPatternCfg(channels=1, vertical_fov_range=(0,0), horizontal_fov_range=(−180,180), horizontal_res=1.0)` → **360 rays** [V]. The pattern excludes the duplicate 180° ray.
  * `ray_alignment="yaw"` keeps the scan horizontal if the chassis pitches; `max_distance=2.0`.
* **Targets:**
  * ground plane (string entry, static);
  * static obstacles (`track_mesh_transforms=False`);
  * moving obstacles (`track_mesh_transforms=True`, **required**: 88.5% errors without it, see §3.2).

  USD primitives (Cube, Cylinder, Sphere, Capsule, Cone) are supported directly [V], so there's no need to convert to meshes. The robot itself isn't a target, so no self-hits occur, unlike the PhysX LiDAR.
* **Post-processing (torch, on the GPU):**
  * range = `‖hit − origin‖`;
  * inf/no-hit → 2.0 (legacy max);
  * values < 0.15 → legacy semantics (to be confirmed in M3);
  * min-pool `view(N, 120, 3).min(-1)`;
  * fix the ray ordering to match legacy index 0 (−180°).
* **Update rate:** once per policy step (`update_period = decimation·dt`), as the legacy code read it once per step.

### 3.2 Measured feasibility (this server, `evidence/probe_results.txt`)
| Test | Result |
|---|---|
| Static obstacles, 64 envs | 0 / 6,400 range errors |
| Moving obstacle (dynamic body, gravity off, velocity commands, the legacy mechanism), 4096 envs × 300 steps | **0 / 1,191,843 errors**; 537.7k env-steps/s, 194 Mrays/s (each step = 1 physics step + 360-ray scan) |
| Same, pose writes | 0 / 12,524 errors (64 envs) |
| Moving obstacle **without** transform tracking (negative control) | 88.5% errors → the checker is valid; tracking is required |
| **Kinematic** obstacle moved by `write_root_pose_to_sim` every step | **Crash:** CUDA illegal memory access in `GpuRigidBodyView` on the first step, even without the ray caster |

**Conclusion:** the original LiDAR behaviour, including moving obstacles, can be reproduced fully on the GPU. Use gravity-free dynamic rigid bodies with velocity (or pose) commands. Don't drive kinematic bodies with per-step pose writes; root-causing that crash (or a kinematic-target API) is optional.

### 3.3 Known limitations vs the legacy sensor
* The ray caster is purely geometric. It has no PhysX scene-query filtering and no beam width; neither did the legacy PhysX LiDAR in practice [A].
* Ray hits are exact to the mesh. The legacy LiDAR's quantization and precision are unknown [A]; noise is added at the observation level anyway.
* The LiDAR will not see other robots unless they are added as targets (irrelevant for single-robot envs).
* **Mirrorcle readiness:** patterns are plain `(ray_starts, ray_directions)` tensors, so per-env, time-varying (adaptive) scan patterns are a natural extension. It needs a custom pattern or sensor subclass in phase 3 [A].

---

## 4. Workflow recommendation: Direct vs Manager-based

**Recommendation: Direct (`DirectRLEnv`).**

1. **Fidelity:** the legacy `RLTask` is a single imperative class. Its methods map 1:1 onto `DirectRLEnv` hooks (§2), which keeps the port reviewable line by line against the original.
2. **Stateful, coupled terms:** progress terms need the previous distance and potentials; collision uses the pooled LiDAR minimum, which is shared with the observation; the time bonus depends on the step counter. These are simple in one class but awkward to split into independent manager terms without duplicated computation.
3. **Performance:** one fused torch path per step, with no per-term overhead. Easy to `torch.jit` the reward/observation kernels later.
4. **Adaptive sensing (phase 3):** the policy will control the sensor (scan pattern) as well as the robot, a coupled action → sensor → observation loop. Direct gives full control of the ordering within a step.
5. **What Manager-based would offer:** per-term reward logging, `CurriculumManager`, `EventManager` for DR, easier ablations. These are reproduced cheaply in Direct with a reward-term dict logged to `extras["log"]` and a small curriculum hook.

   Revisit Manager-based in phase 3 if large ablation matrices become the main activity.

---

## 5. Risks, challenges and mitigations

| # | Risk | Likelihood / impact | Mitigation |
|---|---|---|---|
| R1 | Original obstacle scene missing → paper numbers not reproducible | Certain / High | Define a documented *reference arena* (reuse `obstacles.usd` with start and goal adapted, or rebuild a layout approximating paper Fig. 6). Compare learning behaviour and metrics qualitatively, not absolute returns. |
| R2 | Config ambiguity (code vs paper vs models) | Certain / Medium | Primary reference = `jetbot.py` + [256, 128, 64] (matches `static_120_73reward.onnx`). Record every choice in `docs/decisions.md`. Paper values as alternative flags. |
| R3 | Legacy reward bugs (#3, #4) change learning dynamics | High / Medium | `legacy_exact` mode (reproduces the bugs) and `fixed` mode; train both and report. |
| R4 | LiDAR semantic mismatches (index-0 angle, < min-range and no-hit values, mount height vs obstacle heights) | Medium / Medium | M3 validation against Isaac Sim 5.1's PhysX LiDAR in the same scene (if available) plus analytic tests. |
| R5 | Kinematic pose-write crash (verified) | Certain (if used) / Medium | Use dynamic gravity-free bodies with velocity commands (verified at 4096 envs). |
| R6 | rl_games LSTM through the Isaac Lab wrapper untested; no rl_games ONNX exporter | Medium / Low | Early smoke test in M6; port the legacy exporter; fall back to RSL-RL `RslRlRNNModelCfg` (built-in export). |
| R7 | Scaling beyond 64 envs changes PPO statistics (batch 8,192 → 524,288 at 4096 envs) | High / Medium | Parity runs at 64 envs and the original horizon/minibatch. Scale afterwards with retuned horizon/minibatch, reported separately. |
| R8 | Robot dynamics: Jetbot 5.1 asset drive/wheel geometry differs (0.10 vs 0.1125 m track) | Medium / Low | Open-loop v/ω tracking test in M1; calibrate. |
| R9 | Rendering unavailable on the server → no videos | Certain (until admin fix) / Low | Numeric evaluation + offline trajectory/LiDAR plots (npz → matplotlib). |
| R10 | Sim-to-real timing (40 Hz train vs 5–10 Hz deploy) and observation mismatches inherited from the original | Certain / High (for phase 4) | Not part of reproduction; flagged for the ROS 2 SIL phase (match control rate and observation definitions explicitly). |
| R11 | Collision only via the LiDAR plane | Medium / Low | Keep for parity; add a `ContactSensor` ground-truth collision metric in evaluation. |
| R12 | Isaac Lab main-branch fork drift | Low / Medium | Pin fork commit `c2277524`; record it in run metadata (`isaacrun` already logs it). |

---

## 6. Recommended migration strategy and validation gates

The guiding principle is to **port faithfully first, then fix and scale**. Each milestone has a gate that must pass before the next starts. All runs go through `isaacrun -g <GPU>` with the CUDA shim, headless, without rendering.

| Milestone | Content | Validation gate |
|---|---|---|
| **M0** Scaffold | External Isaac Lab project (template generator, as for `isaac_lab_tutorial`), `pip install -e`, gym registration (`STNav-Legacy-Static-v0`, `-Dynamic-v0`), git repo | `list_envs` shows the tasks; a zero-action env steps headless |
| **M1** Robot + drive | Jetbot `ArticulationCfg` (explicit damping), vectorized diff drive, action rescale in env | Open-loop: commanded (v, ω) vs measured body twist within 5% (straight line, constant turn); track width calibrated |
| **M2** Scene | Reference arena (static colliders) + target marker; per-env replication | Offline top-down plot of colliders/start/goal from the stage; no spawn overlap |
| **M3** LiDAR | `MultiMeshRayCaster` 360 → min-pool 120, legacy semantics | Analytic geometry tests (as in the probe) ≤ 1 cm; if feasible, ray-by-ray comparison against the PhysX LiDAR in the same scene |
| **M4** MDP | 133-dim obs, rewards (`legacy_exact` / `fixed`), terminations, resets, bias noise models | Unit tests: torch functions vs a NumPy re-implementation of the legacy formulas on synthetic batches; random-policy episode statistics plausible |
| **M5** Static training parity | rl_games, MLP [256, 128, 64], 64 envs, legacy PPO config, 1500 epochs | Return curve plateaus; eval success ≥ 90% on the reference arena; 3 seeds |
| **M6** Dynamic + curriculum + LSTM | Moving obstacles (dynamic bodies, velocity commands, bounce), curriculum (static → dynamic at epoch 300), LSTM variant | Curriculum beats from-scratch dynamic training (as in paper Fig. 3); LSTM smoke test, then full run |
| **M7** Evaluation harness | Deterministic (`mu`) batched evaluation over fixed seeded scenario suites; per-episode CSV/JSON | Same seed → identical metrics; metrics listed below |
| **M8** Export | ONNX (normalizer baked in, LSTM state I/O); torch vs onnxruntime parity | max \|Δμ\| < 1e-5 over 10k recorded observations |
| **M9** Scale | 1024–4096 envs, retuned batch sizes; wall-clock comparison | Same final eval metrics with lower wall-clock time |

**Evaluation metrics (M7):**
* success / collision / timeout rates;
* time-to-goal and path length;
* SPL (success weighted by path length);
* minimum and mean clearance (LiDAR and ground-truth contact);
* mean speed and command smoothness (Δv, Δω);
* per-scenario breakdown (static, dynamic, obstacle speeds 0.1–0.8 m/s as in the paper and the legacy code).

These are computed in batch on the GPU over N parallel envs with fixed start/goal sets, and written to `logs/eval/<run>/metrics.{csv,json}`.

**Later phases (for planning only):**
* **Phase 3:** Mirrorcle 3D LiDAR as a configurable `MultiMeshRayCaster` pattern, then adaptive-sensing actions.
* **Phase 4:** ROS 2 software-in-the-loop in Isaac Sim (OmniGraph ROS 2 bridge; needs rendering only for visualization, not for ray-cast sensors [A]), then the physical robot.

**Gazebo:** there's no compelling technical reason to reproduce it. Isaac Sim SIL covers the same purpose (ROS 2 interface testing) with the same physics and sensor models as training.

---

## 7. Proposed project directory structure

```
~/isaac_lab/spatiotemporal_sensing/
├── README.md, env.sh, bin/ (isaacrun, check_physx_gpu), validation/      # existing (phase 1)
├── phase2_analysis/                                                      # this report + evidence/
├── source/st_nav/                         # Isaac Lab external extension (pip install -e)
│   ├── pyproject.toml / setup.py, config/extension.toml
│   └── st_nav/
│       ├── assets/jetbot.py               # ArticulationCfg (+ calibrated drive params)
│       ├── scenes/arenas.py               # reference arena(s), procedural obstacle layouts
│       ├── sensors/lidar2d.py             # MultiMeshRayCaster cfg + legacy post-processing
│       ├── sensors/mirrorcle/             # (phase 3)
│       ├── control/diff_drive.py          # vectorized unicycle → wheel velocities
│       ├── tasks/nav_legacy/
│       │   ├── __init__.py                # gym.register(...) static / dynamic / play variants
│       │   ├── nav_env_cfg.py             # DirectRLEnvCfg (+ legacy_exact / fixed flags)
│       │   ├── nav_env.py                 # DirectRLEnv implementation
│       │   ├── mdp.py                     # pure-torch obs / reward / termination functions
│       │   └── agents/                    # rl_games_ppo_mlp.yaml, rl_games_ppo_lstm.yaml, rsl_rl_ppo_cfg.py
│       └── utils/metrics.py               # batched evaluation metrics
├── scripts/
│   ├── train_curriculum.sh                # static → dynamic checkpoint hand-off
│   ├── evaluate.py                        # deterministic batched evaluation → CSV/JSON
│   ├── export_onnx_rlgames.py             # port of legacy exporter + parity check
│   └── record_rollouts.py                 # npz logs for offline plots (no rendering needed)
├── configs/eval_scenarios/*.yaml          # fixed start/goal/obstacle suites
├── tests/                                 # unit tests (diff drive, LiDAR pooling, reward parity)
├── docs/decisions.md                      # every deviation from the original, with rationale
└── logs/                                  # git-ignored (launcher, training, eval)
```

---

## 8. Estimated development effort

Assumes one developer fluent in Python/PyTorch and RL, new to Isaac Lab, using the existing server tooling. GPU training time is listed separately because it runs unattended.

| Milestone | Person-days | GPU wall-clock (est.) [A] |
|---|---|---|
| M0 Scaffold | 1–2 | n/a |
| M1 Robot + drive | 1–2 | n/a |
| M2 Scene | 2–3 | n/a |
| M3 LiDAR + validation | 2–4 | n/a |
| M4 MDP port + unit tests | 3–5 | n/a |
| M5 Static parity training (3 seeds) | 2–4 | hours (64 envs, 1500 epochs) |
| M6 Dynamic, curriculum, LSTM | 4–6 | hours to a day |
| M7 Evaluation harness | 3–4 | minutes per evaluation |
| M8 ONNX export + parity | 1–2 | n/a |
| M9 Scaling study | 2–3 | hours |
| Buffer (debugging, docs) | ~20% | n/a |
| **Total (faithful reproduction)** | **≈ 22–35 person-days (5–8 weeks calendar)** | |
| + Long-term safeguards (§9) | +4–6 | n/a |
| **Total (recommended)** | **≈ 26–41 person-days (6–9 weeks calendar)** | |

The largest uncertainties are R1 (scene reconstruction choices), R3 (reward-bug effects) and R6 (rl_games LSTM path).

---

## 9. Migration choices that affect the long-term goal (adaptive 3D LiDAR sensing)

A faithful port of the legacy method contains assumptions that are harmless for reproduction but would **silently undermine** the spatiotemporal adaptive-sensing research if they were carried forward. Ordered by severity:

| # | Legacy assumption | Why it matters for adaptive sensing | What to do during the migration (low cost now, high cost later) |
|---|---|---|---|
| L1 | **Collision and proximity penalties are computed from the policy's own LiDAR reading** (min range < 0.25 m, the −1/−3/−5 tiers) [V] | Once the policy controls *where it looks*, it can reduce penalties by **not looking at obstacles**, a direct reward-hacking path. Collisions it doesn't sense would go unpunished and unterminated. | Compute collision, clearance and termination from **privileged ground truth** (a `ContactSensor` and/or a fixed dense "oracle" ray cast that the policy can't see), separate from the policy's sensor. Keep a `legacy_exact` flag for parity runs only. |
| L2 | **Fixed 133-dim observation** with a fixed 120-beam vector, MLP input [V] | Adaptive patterns produce variable, sparse, time-staggered measurements; a fixed beam vector can't represent "what was sensed where and when". | Build the observation from a **sensor-module output** (ranges + directions + validity mask + timestamps) and keep the "120 pooled beams" as one encoder option. Plan for memory/aggregation (LSTM, or an ego-centric polar/occupancy grid updated over time). |
| L3 | **Whole 360° scan captured instantly** every policy step (`rotation_rate 0`) [V] | Real MEMS scanning is sequential: each point has its own timestamp and the robot moves during a scan. The *temporal* dimension is the research topic. | Cast at the physics rate (120 Hz) with **ray subsets per tick** and record per-ray timestamps and poses. Measured throughput (538k env-steps/s with a scan *every physics step*) shows this is affordable. |
| L4 | **360° horizontal FOV, 2D plane** [V] | Adaptive sensing matters most when the sensor's FOV, resolution and rate are *limited*. A 360° oracle-like 2D LiDAR baseline may leave little room for improvement. A 2D plane also can't see overhangs or low obstacles, so the scenes give 3D sensing no advantage. | Treat the 360° 2D policy as **baseline 0 only**. Early on, add a limited-FOV / limited-budget 3D ablation using the actual Mirrorcle SyMPL FOV, rate and point budget **[A: datasheet values still to be collected]**. Make the scene generator support **3D geometry** from the start (varied heights, overhangs, thin objects). |
| L5 | **Ray count fixed at sensor initialization** [V: `num_rays` set in `_initialize_rays_impl`] | Point budgets and adaptive patterns change over time. | `ray_directions` is a per-env `(N, R, 3)` tensor read on every update [V], so patterns can be **rewritten per env per step**. Allocate a fixed maximum budget R_max and mask unused rays. Not runtime-tested yet [A]; add it to the M3 tests. |
| L6 | **Simple dynamic obstacles** (two boxes, constant ±0.8 m/s, same pattern in every env) [V] | The value of *temporal* sensing comes from motion, occlusion and surprise; trivially predictable obstacles understate it. | Make obstacle trajectories a randomized, parameterized distribution (speed, heading changes, appearance from occlusion), driven by velocity commands on gravity-free dynamic bodies (verified at 4096 envs). Keep the legacy pattern as one preset. |
| L7 | **Metrics don't exist** in the legacy code [V] | Adaptive-sensing claims need sensing-specific metrics. | Build the M7 harness with hooks for: rays used per step, energy/scan cost, time-to-first-detection of each obstacle, coverage of the free/occupied space, detection latency vs obstacle speed, and nav metrics vs ray budget. Log scan patterns in `record_rollouts.py` (no rendering needed). |
| L8 | **Generic observation-level noise** (constant σ = 0.2 bias on all dims) [V] | Sensor research needs physically motivated, per-ray noise (range-dependent noise, dropouts, angle jitter of the MEMS mirror, max-range behaviour). | Put noise **inside the sensor module**, per ray. Keep the bias model only for non-LiDAR state. |
| L9 | **Motion-only action space** [V] | Sensing actions (pattern parameters, region of interest, budget) are added later, possibly at a different rate and possibly discrete. | Design the action as named heads (`motion`, `sensing`) from the start; use the Direct workflow (§4) so sensing can be applied before casting within a step. Library support matters: rl_games handles continuous or (multi-)discrete actions; RSL-RL 5 is Gaussian/continuous **[A: verify before choosing discrete sensing actions]**. Prefer continuous sensing parameters. |
| L10 | **Single shared observation for actor and critic** [V] | With partial, actively chosen observations, PPO learns much better if the critic sees privileged state (ground-truth clearances, obstacle states). | Define observation groups `policy` and `critic` now (asymmetric actor-critic). Supported by Isaac Lab's rl_games wrapper (`states` ← `critic` group) [V] and by RSL-RL `obs_groups` [A]. |
| L11 | **rl_games-specific tooling** (exporter, checkpoints) | rl_games is for parity only; long term the project should move to RSL-RL (built-in RNN and ONNX export). | Keep the evaluation harness and ONNX parity tests **library-agnostic** (they consume a policy callable or ONNX file), so switching libraries doesn't mean rewriting evaluation. |
| L12 | **Jetbot as the training robot**; 40 Hz control | The real platform for the Mirrorcle sensor is not yet chosen [A]. Mount height and pose, footprint, dynamics and real control/sensing rates determine what the sensor sees and how sim-to-real transfers. | Keep the robot config and sensor mount pluggable. Decide the target platform and real control/sensing rates before phase 3. Plan to train at the deployment rate (the original trained at 40 Hz but deployed at 5–10 Hz). |
| L13 | **MultiMeshRayCaster is purely geometric** [V] | No intensity, reflectivity, multi-return or beam divergence; dark/specular/thin targets behave unlike reality. Ray-cast cost also scales with mesh count in rich 3D scenes. | Sufficient for geometric adaptive sampling; document it as a limitation. Benchmark ray-cast cost with realistic 3D scene complexity before committing to large scenes. Keep RTX LiDAR (after the rendering fix) for fidelity spot-checks only. |

**Net effect on the plan:** these items add about **4–6 person-days** to M2–M7:
* privileged collision checking;
* the sensor-module interface;
* per-ray timestamps;
* critic observation groups;
* sensing-metric hooks;
* 3D-capable scene generation.

Doing them now avoids re-implementing the environment at the start of phase 3. Parity with the original stays available through the `legacy_exact` flags.

---

## 10. What was verified on this server for this report

* **Source reading:** all three navigation tasks, the base class, vec-env, DR, controller, robot, training and export scripts, ROS nodes and configs.
* **Installed packages read:** rl_games action handling, and Isaac Lab's ray casters, noise models, rl_games wrapper and RSL-RL configs.
* **Inspected:** the ONNX graphs of all 16 shipped models (`evidence/onnx_inventory.txt`), and all repo USDs plus the Isaac Sim 5.1 Jetbot asset in Kit (`evidence/usd_inventory.txt`).
* **Measured:** the moving-obstacle ray-cast probe (`evidence/probe_moving_obstacles.py`, `evidence/probe_results.txt`).
* **Not verified (assumptions):**
  * the PhysX LiDAR's exact semantics (no-hit, < min-range, index 0);
  * the rl_games 1.6.0 rescale behaviour (inferred from 1.6.1);
  * the curriculum epoch meaning;
  * the LSTM `rnn` config of the paper;
  * the `.pth` checkpoint contents (not unpickled for safety);
  * rl_games LSTM through the Isaac Lab wrapper.
