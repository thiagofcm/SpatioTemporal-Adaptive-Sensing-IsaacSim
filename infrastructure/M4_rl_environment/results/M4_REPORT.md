# M4 report: RL environment (`STNav-TB4-Corridor-Static-v0`)

Status: **implemented and validated; 31 of 31 checks pass (rerun after the follow-up changes in §7). No PPO training has been run (M5 is waiting for review).**

> Follow-up (after the first review): §7 covers the requested diagnostics. Changes made: `goal_distance_norm` 5.9397 → 7.0, and observation [45:47] now uses base_link velocity instead of CoM velocity. §4's outcome explanation and §5 items 2–3 are resolved there.

## 1. Summary

| Item | Result |
|---|---|
| Environment | `CorridorNavEnv(DirectRLEnv)`, registered as `STNav-TB4-Corridor-Static-v0` |
| Simulation layer | M3 `CorridorScene` reused unchanged in behaviour (split into phases so it can be built inside DirectRLEnv; M3 smoke output identical) |
| Timing | physics 100 Hz, solver 16/2, decimation 20, control 5 Hz, episode 70 s = 350 steps |
| Observation / action | 47-dim / 2-dim continuous, in [−1, 1], scaled to ±0.22 m/s and ±2.0 rad/s |
| Validation | `scripts/m4/m4_validation.py`, 1024 envs: **31/31 pass** (`m4_validation.json`) |
| RSL-RL | 5.0.1, through `RslRlVecEnvWrapper` + `OnPolicyRunner`: API smoke test passes |
| Throughput (full RL loop with policy inference) | **55.6k control steps/s at 16,384 envs** (8.6 GiB); 22.0k at 4,096 envs (4.7 GiB) |
| Stability | 0 non-finite values in 200 × 16,384 steps |

## 2. Implementation

Files (under `source/st_nav/st_nav/`):

* `tasks/nav2d/corridor_env_cfg.py`: `CorridorNavEnvCfg`, `RewardCfg`. Every reward weight, tolerance and margin can be configured.
* `tasks/nav2d/corridor_env.py`: `CorridorNavEnv`.
* `tasks/nav2d/agents/rsl_rl_ppo_cfg.py`: `CorridorNavPPORunnerCfg`. Prepared for M5; not used for training.
* `tasks/nav2d/__init__.py`: Gym registration.
* `envs/corridor_scene.py`: M3 scene, now with a deferred mode (`deferred` → `build_scene_cfg` → `prepare_stage` → `attach`). The standalone mode is unchanged.

Step order (DirectRLEnv): actions → 20 × (command model, wheel targets, physics) → dones → rewards → reset finished envs → observations. Dones and rewards use the state after this step. Reset envs return the observation of their new initial state.

### Observation (47)

| Index | Content | Normalization |
|---|---|---|
| 0–39 | 40 LiDAR ranges (latest scan, with noise) | / 3.5 m, in [0, 1] |
| 40 | distance to goal | / 7.0 m (`goal_distance_norm`), clipped to [0, 1] |
| 41, 42 | sin, cos of goal bearing in the robot frame | — |
| 43, 44 | previous (clipped) action v, ω | [−1, 1] |
| 45, 46 | measured base_link v, ω (body frame) | / v_max, / ω_max |

The LiDAR runs on its own schedule (`lidar_period_s`, default 0.2 s, so one scan per control step). Between scans the observation holds the most recent scan. Every reset forces a fresh scan. The sensing rate is therefore a separate parameter from the control rate. An optional privileged critic group (`privileged_critic=True`: clean scan, true clearance, goal vector) is available and off by default.

### Reward (per control step, every term logged)

| Term | Definition | Default |
|---|---|---|
| progress | 10 · (d_prev − d) | — |
| proximity | −1 · max(0, 1 − c / 0.15)² | c = true footprint clearance |
| time | −0.1 /s · 0.2 s | −0.02 |
| goal | +30, on the terminating step only | — |
| collision | −30, on the terminating step only (physical collision or safety violation) | — |
| timeout | −0 (truncation) | configurable |

Rewards and terminations use true geometry only (analytic clearance of the r = 0.177 m footprint, true goal distance). They never use the LiDAR.

### Terminations

* **goal:** d < 0.20 m. Takes priority on the same step.
* **safety violation:** clearance < 0.05 m. **Physical collision:** clearance ≤ 0. Both terminate with −30. The outcome code records which one happened.
* **timeout:** step 350, returned as `truncated` (time_out), not `terminated`. Evaluation counts it as a failure.
* Per-env outcome codes are in `extras["outcome"]`: 1 goal, 2 physical, 3 safety, 4 timeout. Episode means go in `extras["log"]`: `Episode_Reward/*`, `Episode_Termination/*`, length, minimum clearance.

### Resets (partial, per env)

Each reset:

* resamples the Z1–Z4 layout (bank of 1024), the start pose and the goal;
* resets the robot and the actuator delay/lag state;
* sets the previous action to 0 and `prev_goal_dist` to the new distance, so the first step has no artificial progress;
* takes a fresh scan;
* resets the episode reward sums.

Start pose and goals: see `figures/m4_start_goal_poses.png` and §5.1.

## 3. Validation (`scripts/m4/m4_validation.py`, 1024 envs, all pass)

| Test | What is checked | Result |
|---|---|---|
| T1 observations | shape 47; LiDAR block vs analytic oracle / 3.5; goal distance, sin/cos bearing, previous action and velocity at reset; LiDAR in [0, 1] | max error 7.5e-6 (normalized) |
| T2 actions | clipped action stored as previous action; resulting velocity vs scaled command | v 0.1107 / 0.2204 (expected 0.11 / 0.22); ω 0.901 / −1.801 (expected 0.9 / −1.8) |
| T3 rewards | goal distance, clearance, progress, proximity, time vs independent computation; total = sum of terms; flags vs terminated | all errors 0.0 |
| T3 terminal once | goal/collision rewards appear only on terminating steps | pass (1,237 episodes ended) |
| T4 goal | robots placed inside tolerance → terminate with +30 once | 1024/1024 |
| T4 safety | terminates with −30 at clearance 0.030 m (margin 0.05) | 1024/1024 |
| T4 physical | margin 0: terminates at clearance −1e-5…−4e-5 m | 1024/1024 |
| T4 timeout | truncated exactly at step 350, never terminated | pass |
| T5 resets | previous action 0, previous distance = new distance, actuator state 0, fresh scan (error 6.5e-6), episode length 0 | pass; max first-step progress 0.018 (bound 0.54) |
| T6 stability | no NaN/Inf; observation and reward ranges | reward in [−31.24, 0.42] |
| T7 RSL-RL | wrapper, observation groups, runner construction, one inference call | rsl-rl 5.0.1, policy obs [1024, 47] |

## 4. Integration benchmark (`scripts/m4/m4_benchmark.py`)

GPU 5, no other processes on it. A randomly initialized RSL-RL actor-critic produces the actions (inference only, nothing learned or saved). Each step is a full RL step: policy, command model, 20 physics steps, LiDAR with noise, rewards, dones, partial resets, observations. 200 timed steps follow 10 warm-up steps.

| Envs | Control steps/s | Wall time (200 steps) | Setup | GPU memory | Non-finite |
|---|---|---|---|---|---|
| 4,096 | 22.0k | 37.2 s | 155 s | 4.7 GiB | 0 |
| 16,384 | **55.6k** | 59.0 s | 649 s | 8.6 GiB | 0 |

* **Compared with M2:** the M2 full loop reached 66.0k at 16,384 envs. The complete RL environment costs about 16 % more (policy forward pass, rewards, termination checks, per-env resets with layout resampling).
* **Expected M5 rollout cost:** 32 steps × 16,384 envs per iteration takes about 9.4 s of environment time (4,096 envs: about 6.0 s), plus the PPO update. 1,500 iterations at 16,384 envs means roughly 4 h of environment stepping. These are estimates, not measurements.
* **Outcomes in this run:** every episode ended in a safety violation, exactly 4 per env, about every 50 steps, with mean collision term −0.6 per step = −30/50. Cause, **verified in §7.2**: RSL-RL's inference policy is deterministic (actor mean). An untrained actor gives nearly the same action in every env, and every env starts from the same pose next to the same fixed walls, so all robots follow the same path into a wall. This does not affect the throughput figures. Goal and timeout paths are covered by T4.

## 5. Points for review

1. **Start/goal configuration (reference ST01–03 settings):**
   * fixed start (2.50, 2.50) with yaw −168.69°;
   * start-pose randomization available but off: 5 points, yaw 180° ± 10°;
   * goal drawn uniformly from 9 points around (−1.8, −2.3), tolerance 0.20 m.
2. **[Resolved, §7.1]** **Goal-distance observation is saturated at the start.** Start-to-goal distance from the default start is 6.17–6.79 m (6.59 m for the main goal; 5.89–6.79 m with start-pose randomization), but the normalizer is 5.9397 m (reference `MAX_GOAL_DISTANCE`). So obs[40] reads 1.0 at the start and only changes after the robot has closed up to about 0.85 m of the distance (validation minimum: 0.789). I kept the reference value. Raising `goal_distance_norm` to about 7.0 would remove the saturation.
3. **[Resolved, §7.3]** **Measured velocity exceeds the nominal range.** obs[45] reached 1.28 (≈ 0.28 m/s) in the validation rollout, above v_max. This is probably a short spike from wall contact or the reset. It is bounded and finite, and observation normalization in the PPO configuration handles it. Not investigated further.
4. **Setup time grows with scale:** 649 s at 16,384 envs, against 336 s in M2. The added cost is the per-env baffle meshes (`replicate_physics=False`) and the ray caster's duplicate-mesh check, which grows as O(N · distinct meshes). It is paid once per training run.
5. **PPO configuration prepared, not run:**
   * 32 steps per env;
   * [256, 256, 128] ELU network with observation normalization;
   * lr 3e-4, γ 0.99, λ 0.95, entropy 0.005;
   * 1,500 iterations.

## 6. Deviations and limitations

* `CorridorScene` was split into build phases so DirectRLEnv can own the scene. Behaviour is unchanged: the M3 smoke test gives identical output and the M3 tests still pass (15 passed, 1 skipped).
* The benchmark's outcome distribution comes from an untrained deterministic policy and says nothing about task difficulty.
* No repository reorganization was done, and no validation scripts were deleted (postponed until after M5, as requested).

## Artifacts

* `results/m4/m4_validation.json`: all checks and details.
* `results/m4/benchmark.jsonl`, `results/m4/m4_runlog.txt`: benchmark.
* `results/m4/figures/m4_start_goal_poses.png`: start poses and goals.
* Scripts: `scripts/m4/m4_validation.py`, `scripts/m4/m4_benchmark.py`, `scripts/m4/m4_plot_start_goal.py`.

## 7. Follow-up diagnostics (after the first review)

Scripts:

* `scripts/m4/m4_diagnostics.py`: D1, D2, D3, D4; 1024 envs, seed 11 as in the benchmark.
* `scripts/m4/m4_diag_velocity.py`: D3b; 256 envs.
* `scripts/m4/m4_plot_diagnostics.py`: figure.

Data are in `results/m4/diag/` (`m4_diagnostics.json`, `m4_diag_velocity.json`, traces). Figure: `diag/m4_followup_diagnostics.png`.

### 7.1 Goal-distance normalization and default training distributions (D1)

`goal_distance_norm` is now **7.0 m**, still configurable in `CorridorNavEnvCfg`. At reset obs[40] now ranges 0.882–0.971 (start-to-goal distance 6.17–6.79 m, all ≥ 6.17 for the 9 goals). It no longer saturates.

The defaults below are what training will use. They were measured over 4,989 resets.

| Item | Default |
|---|---|
| Start pose | fixed (2.50, 2.50), yaw −2.9442 rad (−168.69°). One distinct start pose seen in all resets. `start_pose_dr=False` |
| Start clearance | 0.158 m to the top and right outer walls, so 0.108 m before a safety violation (margin 0.05 m) |
| Goal | uniform over the 9 ST01 goals: 532–580 resets each, expected 554 |
| Z1–Z4 sizes | per env from a bank of 1024 reference-sampler entries, fixed for the run (1024 envs use 658 distinct entries; at 4096+ envs all are used) |
| Z1–Z4 offsets | resampled every reset (std 0.14–0.16 m per baffle); d(Z1, Z3) ≥ 0.70 constraint; 0 fallbacks |
| LiDAR | 40 beams, 3.5 m, noise on, one scan per 0.2 s |
| Episode | 350 steps (70 s); goal 0.20 m, safety 0.05 m |

### 7.2 Why every benchmark episode ended in a safety violation (D2)

| | Deterministic `get_inference_policy` (benchmark) | Stochastic `alg.act` (PPO rollouts) |
|---|---|---|
| Actions, step 0 | mean (0.090, −0.104) → v 0.020 m/s, ω −0.21 rad/s | mean (0.04, −0.06) |
| Action std across envs | 0.002–0.009 | 0.71–0.74 |
| Position std across envs | < 1.7 mm for the whole episode | 0.07–0.12 m |
| Episode length | 46 or 47 steps only (46.2 ± 0.4) | 4–120 steps (33.6 ± 24.3) |
| Terminal wall | top outer wall: 2048 / 2048 | top outer wall 1051, right outer wall 866 |
| Outcomes in 120 steps | 2048 safety violations | 1917 safety violations, 0 goals, 0 timeouts |

Confirmed. The inference policy returns the actor mean. An untrained actor gives nearly the same small (v, ω) for every input. With one fixed start pose and fixed walls around it, all envs follow the same path (std < 2 mm). They creep at 0.02 m/s while turning left of the wall, cover the 0.108 m of safety slack, and terminate at the top wall at step 46–47. Each reset brings the robot back to the same state, so the cycle repeats. That gives exactly 4 terminations per env in the 200 timed steps.

Randomization is active: the offsets differ across envs. But the walls near the start are fixed, and Z1 is about 1.3 m away, so randomization cannot desynchronize the envs. With the stochastic actions PPO actually uses, episodes and trajectories are diverse.

### 7.3 Measured linear velocity above v_max (D3, D3b)

Signals recorded per physics substep:

* CoM velocity (the old observation);
* base_link velocity;
* wheel odometry, r · mean(q̇);
* finite difference of the base position;
* command-model output;
* contact force;
* pitch and roll angles and rates;
* height.

Origin:

1. **How the velocity was extracted: main cause.** In Isaac Lab `root_lin_vel_b` is an alias for `root_com_lin_vel_b`, the **centre-of-mass** velocity. base_link's CoM is at (0.015, 0.0003, **0.137**) m in the link frame, so v_com,x = v_link,x + ω_y · 0.137 (checked: error 9e-8; pitch rate explains 86 % of the gap). The largest excursion: CoM velocity 0.284 m/s while the base_link velocity, wheel odometry and position finite difference all held 0.220 m/s = command. Abrupt random action reversals rock the body by up to 4.3° pitch (0.93 rad/s), so the CoM swings up to ±0.13 m/s relative to the base. In straight driving the steady pitch is a constant −0.25° and the two agree within 0.001 m/s.
2. **Reset logic: secondary, small.** Reset writes the robot at z = 0. It rests at z = −4.45 mm, so it falls 4.45 mm in the first 3 substeps. That gives a pitch transient (−0.78°) and up to 0.083 m/s of base velocity with zero command. It dies out within about 0.1 s, before the command arrives (0.14 s delay) and before the next observation. Same in every action group. **Not changed:** it is in the validated M3 scene. A fix is proposed below.
3. **Actuator dynamics: genuine, small.** During abrupt reversals the base_link velocity and the wheel odometry themselves briefly exceed v_max (base up to 0.288 m/s per substep; at observation times up to 0.237 m/s = 1.077 v_max). In these samples the contact force was 0 % of the time above 1 N, and the command-model output never exceeded 0.220. This is the drive and body response, and it is kept as is. Straight start from rest: overshoot 0.225 m/s (+2.4 %).
4. **Contact forces: not involved.** 0 of 3,395 over-threshold samples had contact force above 1 N.

**Change made:** obs[45:47] now use `root_link_lin_vel_b[:, 0]` and `root_link_ang_vel_b[:, 2]`: base_link frame velocity, matching what the real TB4 wheel odometry reports. The angular rate is the same rigid-body quantity as before. No clipping was added.

Validation rerun: 31 / 31 pass. The T6 velocity range went from [−1.016, 1.284] to [−1.016, 1.077]. The T2 velocity tracking is unchanged (0.1107 / 0.2204 m/s, 0.901 / −1.801 rad/s). The previous result is kept in `m4_validation_before_followup.json`.

**Proposed, not applied (needs your OK, since it touches the M3 scene):** reset the robot at its rest height, z = −0.00445 m, instead of 0 to remove the reset drop.

### 7.4 Timeout bootstrapping with RSL-RL 5.0.1 (D4)

Code path:

* The `RslRlVecEnvWrapper` sets `dones = terminated | truncated`.
* Because `cfg.is_finite_horizon = False` (default), it also sets `extras["time_outs"] = truncated`.
* `PPO.process_env_step` then adds γ · V(s_t) · time_out to the stored reward. V(s_t) is the critic value of the observation *before* the final step; the true next state is lost to auto-reset.
* `compute_returns` cuts the bootstrap at every done.

Numerical test: 40-step episodes, half the envs standing still (→ timeout), half driving (→ safety violation from step 30), stepped through the real wrapper, `alg.act` and `alg.process_env_step`.

| Check | Result |
|---|---|
| `time_outs` == `truncated` at every step; `dones` == `terminated` ∣ `truncated` | yes |
| timeouts: 512, all at step 40, all in the standing half | yes |
| terminations: 512, all in the driving half; overlap of terminated and truncated | 0 |
| stored reward on timeout steps − (r + 0.99 · V(s_t)) | max error 0.0 |
| stored reward on all other steps − r (no bootstrap on terminations) | max error 0.0 |

Conclusion: timeouts are bootstrapped and terminations are not, as intended. The one-step approximation, using V(s_t) instead of V(s_{t+1}), is inherent to RSL-RL and Isaac Lab auto-reset. It is standard and small at γ = 0.99. The observation has no time input, so bootstrapping (a time-unaware value) is consistent with the policy's information. In evaluation, timeouts are still counted as failures.
