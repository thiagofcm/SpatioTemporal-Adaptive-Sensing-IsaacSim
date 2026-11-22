# Phase 3 roadmap (revised): articulated TurtleBot4 Standard, 2D LiDAR, PPO

**Date:** 2026-11-21. **Status:** plan for approval; no implementation started.
**Supersedes:** §§3.3, 4–7 of `PHASE3_NATIVE_NAV_PLAN.md`. The investigation sections of that document remain valid.
**Tags:** [V] verified; [M] measured on this server; [A] assumption or estimate.

## 1. Fixed design decisions (from your review)

| Item | Decision |
|---|---|
| Robot | **Mode A only:** articulated TurtleBot4 Standard in PhysX. Pose and velocity come from PhysX. No separate kinematic environment |
| First scenario | `corridor_static_object_random` (static Z1–Z4); direct PPO. Curriculum and geodesic progress are kept as fallbacks |
| Geometric DR | Z1–Z4 lengths/thicknesses sampled per env at startup; obstacle radius via bins. **Per-reset offsets must satisfy the original constraints** (C12, C34 ≥ 0.75 m, d(Z1, Z3) ≥ 0.70 m) |
| LiDAR | 40 beams, uniform 9°, 3.5 m, mount from the TB4 model |
| Control | 5 Hz; v ∈ [−0.22, 0.22] m/s, ω ∈ [−2, 2] rad/s, subject to wheel-speed feasibility (§3) |
| Heights | walls 0.6 m, moving obstacles 0.5 m (both configurable) |
| Observation | 47 dims (40 beams, d_goal, sin/cos θ_goal, a_{t−1}, measured v/ω); privileged critic optional |
| Reward | `r = k_p(d_{t−1} − d_t) − k_o·max(0, 1 − c/c_safe)² − k_t·dt + terminal`, with k_p = 10, k_o = 1, c_safe = 0.15 m, k_t = 0.1, dt = 0.2, +30 goal, −30 collision; all configurable, no other shaping |
| Clearance c | min distance from the **robot footprint** to any wall/obstacle surface (not from the centre), computed from PhysX states |
| Collision vs safety margin | **physical collision:** c ≤ 0 (plus a contact-sensor backstop); **safety termination:** c < m_safe, default 0.05 m (the previous env's margin), configurable; logged as separate outcomes |
| Timeout | truncation in training; failure in evaluation |

## 2. Findings that shape the plan (new since the last report)

* **TB4 Standard description** [V, upstream GitHub, not on this machine]:
  * `turtlebot/turtlebot4` → `turtlebot4_description/urdf/standard/turtlebot4.urdf.xacro`, which includes `irobot_create_description/urdf/create3.urdf.xacro` from `iRobotEducation/create3_sim`;
  * meshes are `.dae`;
  * the xacro needs `xacro` to expand. **No ROS, no `xacro`** on this machine and none in the conda env [V], so expansion needs a separate throw-away venv (or a pre-expanded URDF).
* **LiDAR mount from the model** [V]:
  * `rplidar_link` is at x = −0.040 m, z = **0.193 m** above `base_link` (0.0642 + 0.03 + 0.0987), with yaw +π/2;
  * x matches the reference (−0.04); z differs from the reference's 0.171 m.
* **Fallback asset** [V]: Isaac Sim 5.1's `iRobot/Create3/create_3.usd`:
  * wheels r = 0.03575 m, track 0.24 m, velocity drives with damping 1e7;
  * it contains its own physics scene and ground plane.
* **Real-robot characterization** [V] (`~/Sim2Real/…/motion_dynamic_charac/analysis`):
  * in-motion command delay ≈ 0.12–0.20 s, mostly pure delay (τ ≈ 0–0.1 s), gains ≈ 1.0;
  * linear rise time ≈ 0.065 s; start-from-rest delays much longer (0.3–1.6 s);
  * **wheel-speed saturation ≈ 0.305–0.31 m/s, best fit by proportional "scale" saturation** (`saturation_models.csv`).
* **Velocity feasibility** [V/A]: v = 0.22 m/s with ω = 2 rad/s needs 0.22 + 2·0.12 ≈ 0.46 m/s at the outer wheel, which exceeds the ≈ 0.31 m/s limit. The command is therefore mapped through the measured scale saturation, as on the real robot.

## 3. Actuation and calibration principle

`policy (v, ω)` → **command model in torch** (pure delay θ, residual lag τ_res, gains k, wheel-speed scale saturation at 0.31 m/s) → wheel velocity targets → **PhysX joint velocity drives** → measured base twist.

* **Avoiding double-counted lag:** first measure the lag that the PhysX drives and contacts add by themselves (step response in sim, M1). Then set τ_res so that the *total* simulated response matches the characterization: τ_res ≈ max(0, τ_real − τ_sim). The delay is applied only once, in torch.
* **Delay/lag/gain randomization** (as in the reference DR ranges) is an M5+ option; the baseline uses the mean characterization.

## 4. Milestones

Effort is AI-assisted: **Impl.** = my implementation and debugging hours, **Rev.** = your review hours, **GPU** = compute. Each milestone ends with an approval checkpoint.

### M1: TurtleBot4 Standard articulation and differential drive (highest priority)
* **Objective:** a validated TB4 Standard articulation driven by (v, ω).
* **Tasks:**
  1. Pin upstream `turtlebot4_description` and `irobot_create_description`; expand the xacro in a throw-away venv (not the conda env); strip Gazebo and ros2_control tags; resolve `package://` paths.
  2. Convert with Isaac Lab `UrdfConverterCfg`:
     * floating base, velocity drives on the wheels;
     * merge fixed joints;
     * wheel-drop joints fixed or kept stiff;
     * low-friction caster;
     * simplified shell/tower collisions.
  3. `ArticulationCfg`; differential-drive mapping with scale saturation; command model (§3).
  4. Single-env and 64-env headless tests: straight runs, turns in place, combined arcs, steps matching the real characterization transitions.
* **Deliverables:** `assets/turtlebot4_standard.usd` + `TB4_CFG`, `control/diff_drive.py`, a calibration report (sim vs real step responses, plots), and an import README.
* **Acceptance:**
  * the robot rests stably (no drift or jitter > 1 mm/s at zero command over 60 s);
  * steady-state v and ω track within 3 % below saturation;
  * the scale-saturation behaviour matches the model;
  * simulated delay/rise time match the characterization means within ±0.03 s after calibration;
  * the LiDAR frame is at the URDF pose.
* **Effort:** Impl. 6–10 h · Rev. 1.5 h · GPU minutes.
* **Risks:**
  * xacro expansion tooling;
  * `.dae` meshes or a complex collision shell in the URDF importer;
  * caster and wheel-drop joints causing physics instability;
  * the physics dt needed for stable small wheels (expected 1/100–1/200 s).
* **Fallback:** Create3 USD + TB4 shell/tower as primitives at the URDF dimensions.

### M2: GPU-parallel physics performance gate (early)
* **Objective:** know the real cost before building the MDP.
* **Tasks:**
  * N robots, each in a 6 × 6 m arena of 0.6 m walls (corridor layout);
  * the 40-beam `MultiMeshRayCaster` attached to `rplidar_link`;
  * random wheel commands at 5 Hz;
  * sweep num_envs ∈ {512, 1024, 2048, 4096, 8192} and physics dt ∈ {1/100, 1/200};
  * record GPU memory and VRAM headroom on one A6000.
* **Deliverables:** `scripts/bench_physics.py` and a throughput table (env-steps/s at 5 Hz, physics steps/s, ray-cast share).
* **Acceptance:**
  * stable for 10⁴ control steps at the chosen size;
  * **≥ 15k control steps/s at 4096 envs** [A target];
  * chosen dt/decimation documented.
* **Effort:** Impl. 2–3 h · Rev. 0.5 h · GPU < 1 h.
* **Expected** [A]: the robot has ~5–7 bodies; with dt = 1/100 the decimation is 20, giving roughly 15–40k control steps/s at 4096 envs. The ray-cast cost is negligible (Phase 2: 194 Mrays/s [M]).
* **Risks:** throughput below target, which would force a larger dt, fewer bodies (merge links, fix wheel-drop joints) or fewer envs.

### M3: Static scene, ground-truth geometry and constrained DR
* **Objective:** the `corridor_static_object_random` scene with PhysX-consistent ground truth.
* **Tasks:**
  * port the wall tables and Z sampler (pytest against the reference numpy functions);
  * per-env Z1–Z4 sizes at startup, sampled with the original hard/normal mode mix;
  * per-reset offsets by vectorized rejection sampling under the original constraints (C12/C34 depend only on the fixed lengths; d(Z1, Z3) is checked per candidate, with fallback to the last valid offsets);
  * reset placement of the baffles via pose writes on gravity-free dynamic bodies;
  * footprint clearance c from the PhysX root pose (analytic box/circle SDF minus the footprint radius);
  * start/goal from the reference lists + start-pose DR;
  * contact sensor on the chassis.
* **Deliverables:** `scenes/` module, `geometry.py`, top-down plots of sampled envs from the simulated state.
* **Acceptance:**
  * 10⁵ sampled resets: **0 constraint violations**, offset marginals match the reference sampler (KS test);
  * the analytic clearance agrees with ray-cast distances to the same surfaces within 1 cm;
  * physical contact (contact sensor) occurs only when c ≤ 0.01 m (consistency of the footprint model);
  * the baffle poses read by the ray caster equal those used by the analytic clearance.
* **Effort:** Impl. 5–7 h · Rev. 1.5 h · GPU minutes.
* **Risks:**
  * the footprint is not a perfect circle (bumper/shell): use the URDF collision radius, and check against contacts;
  * the pose-write path at 4096+ envs (verified at 64 [M]; re-verify here).

### M4: Sensor and navigation MDP
* **Objective:** the full `DirectRLEnv`.
* **Tasks:**
  * `Lidar2D` sensor module (uniform 9°, 3.5 m, noise model optional);
  * 47-dim observation; optional critic group;
  * the reward (§1) with per-term logging;
  * outcomes success / physical collision / safety termination / timeout;
  * resets.
* **Deliverables:** `tasks/nav2d/` env + config, registered gym task, pytest for reward/termination terms.
* **Acceptance:**
  * reward terms reproduce hand-computed values on scripted trajectories;
  * a scripted wall-approach yields safety termination at c = m_safe before physical contact, and m_safe = 0 yields physical-collision termination;
  * ray-cast beams match the analytic oracle within 1 cm;
  * a random policy runs 10⁴ steps at 4096 envs without NaNs.
* **Effort:** Impl. 4–6 h · Rev. 1.5 h · GPU minutes.

### M5: PPO training and evaluation on the static scenario
* **Objective:** the first functional navigation policy.
* **Tasks:**
  * RSL-RL PPO (MLP [256, 256, 128]), 4096 envs, γ 0.99–0.995;
  * checkpoints and TensorBoard via `isaacrun`;
  * evaluation harness: deterministic policy, 100 frozen reference-seeded episodes, success / collision / safety / timeout rates, path length, time, min clearance, offline trajectory plots.
* **Deliverables:** trained checkpoints (3 seeds), evaluation tables and plots, run configs.
* **Acceptance:**
  * evaluation success ≥ 90 % (reference point: ~97–98 % for TD3 + sensing in ST01 [V]);
  * physical-collision rate ≤ 5 %;
  * learning curves reproducible across 3 seeds.
* **Effort:** Impl. 3–5 h · Rev. 2–3 h · **GPU ≈ 2–6 h** (e.g. 50–100 M control steps per run at 15–40k steps/s ≈ 0.5–2 h per run; 3 seeds + 1–2 tuning runs).
* **Risks:** convergence on the single S-route. Fallback order: tune γ/entropy → start/goal curriculum → geodesic progress.

### M6 (after baseline): moving obstacles
* **Objective:** `corridor_dynamic_chase_random` (OB1–OB3).
* **Tasks:**
  * vectorized motion types; obstacles as gravity-free dynamic cylinders (0.5 m tall) driven by per-tick velocity commands that land exactly on the reference positions at tick boundaries;
  * presence, speed and path-y DR, radius bins;
  * clearance uses PhysX-read obstacle poses;
  * an evaluation mode with the reference's seeds.
* **Acceptance:** obstacle positions at tick boundaries match the reference process for identical seeds (≤ 1 mm); evaluation vs reference points ST04 (~100 %) and ST05 (~79 %).
* **Effort:** Impl. 5–7 h · Rev. 1.5 h · GPU 2–6 h.

## 5. Totals and sequence

| | Impl. | Review | GPU |
|---|---|---|---|
| M1–M5 (static baseline) | **20–31 h** | **≈ 7–8 h** | **≈ 3–7 h** |
| + M6 (moving obstacles) | +5–7 h | +1.5 h | +2–6 h |

Sequence: **M1 → M2** (stop and review if the robot or throughput gate fails) **→ M3 → M4 → M5 → M6**. Calendar time is about 1.5–2.5 weeks to the static baseline, dominated by the review checkpoints.

## 6. Dependencies and constraints
* No changes to the reference repositories, the Isaac Lab installation or the conda env:
  * the new project lives under `~/isaac_lab/spatiotemporal_sensing/` and is loaded via `PYTHONPATH` from `env.sh`;
  * xacro expansion uses a throw-away venv in the project (to be confirmed at M1).
* Upstream TB4 / Create3 description repos are fetched from public GitHub at a pinned commit (licences checked and recorded at M1).
* All runs go through `isaacrun -g <gpu>` (CUDA shim, single-GPU isolation, headless; no rendering needed).
