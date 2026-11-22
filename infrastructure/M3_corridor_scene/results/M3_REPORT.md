# M3: `corridor_static_object_random` environment, ground-truth geometry and randomized resets

**Date:** 2026-11-22. **Status:** complete, for review. M4 has not been started; no PPO training.

**Configuration (adopted after M2):**
* physics 100 Hz, solver 16/2 iterations;
* 5 Hz control (decimation 20);
* 40-beam LiDAR, 360°, 5 Hz, 3.5 m;
* calibrated actuator model (M1); LiDAR and noise model (M2) unchanged;
* headless; number of envs configurable.

The reference repository was not modified (`git status` unchanged; scene modules imported without writing bytecode).

## 1. Summary

| Requirement | Result |
|---|---|
| Arena, corridor, baffles, start poses, goals reproduced | constants and geometry functions match the reference exactly (unit tests against the reference code) |
| Wall 0.6 m / obstacle 0.5 m | walls and Z1–Z4 baffles 0.6 m (see deviation D3); 0.5 m reserved for moving obstacles (M6) |
| GPU-vectorized resets with reference distributions and constraints | 20,480 randomized layouts: 0 violations (min C12 0.7501, min C34 0.7500, min d(Z1, Z3) 0.715 m); distributions match the reference sampler (KS, §4) |
| Footprint clearance, physical collision, safety margin | contact sensor vs ground-truth clearance: first contact at c = −0.06 mm (median) going forward, −12.7 mm going backward (expected, D5); no contact while c > 1 cm; no c < −3 cm without contact |
| LiDAR in randomized scenes | 491,520 rays vs oracle with per-env walls and baffles: p99 0.56 mm (walls) / 0.66 mm (baffles); 6 rays > 1 cm (grazing corners) |
| Reset logic | exact start poses, zero velocities and command-model state, partial resets leave other envs untouched, start-pose DR and goals as in the reference |
| Throughput / stability (full env with resets) | 16,384 envs: 51k control steps/s, 8.7 GiB, 0 unstable, no NaN; 4,096 envs: 20k control steps/s |

## 2. Implementation (modular)

```
st_nav/config.py                       adopted defaults (100 Hz, 16/2, decimation 20, 40 beams, 5 Hz, 3.5 m, heights)
st_nav/scenes/corridor_static_random.py  scenario: reference constants (with provenance), batched z_walls / rect_distance,
                                       GPU offset sampler with constraint C, reference-sampler size bank, start/goal samplers
st_nav/envs/corridor_scene.py          CorridorScene: build, GPU resets, robot stepping with the command model, ground
                                       truth (footprint clearance, flags), contact sensor, LiDAR read
st_nav/geometry/analytic2d.py          (M2) analytic ray cast + clearance on per-env wall tables
st_nav/sensors/*                       (M2) unchanged; the scene only adds ray-cast targets
```

**One geometry source:**
* Per-env wall tables (10 fixed walls + Z1–Z4 as [cx, cy, hx, hy, yaw]) are written at every reset.
* From these tables:
  * the kinematic baffle bodies are posed (PhysX collision and ray-caster tracking);
  * ground truth is computed (clearance, collisions), independent of the sensor.
* Static walls are one shared primitive group. Baffles are four kinematic rigid bodies per env with per-env USD scale (`replicate_physics=False`).

**Extensibility:**
* Moving obstacles (M6) are additional tracked bodies plus rows in the same tables.
* The Mirrorcle sensor replaces `Lidar2D` behind `SensorOutput`.
* Rewards and terminations (M4) only use `footprint_clearance()`, `goal_distance()` and the flags, never the LiDAR.

## 3. Reference reproduction

| Element | Reference | Isaac Lab |
|---|---|---|
| Arena | 4 outer walls (6.00 m, 0.15 m thick, centres ±2.91 m) | identical |
| Corridor | C1–C6 S-corridor | identical (constants asserted equal) |
| Z1–Z4 | anchored to C1/C2, C5, C6, C3; nominal (0.55, 0.15, 0.1, 0.6) × (0.6, 0.15, 0.6, 0.15) | identical construction (`z_walls` = `static_obstacle_walls`, max diff 1e-12 m over 500 layouts) |
| Length DR | pairs, 50 % "hard" (clearance 0.75–0.80) / 50 % "normal"; L ∈ [0.1, 0.6]; C12, C34 ≥ 0.75 | the reference sampler itself (bank) |
| Thickness DR | U(0.15, 0.6) each | reference sampler (bank) |
| Offset DR | Z1_x, Z2_x ∈ [−0.5, 0]; Z3_y, Z4_y ∈ [−0.55, 0]; d(Z1, Z3) ≥ 0.70 | every reset, GPU rejection with the same ranges and constraint |
| Start pose | (2.5, 2.5), yaw −168.7° (ST01); start-pose DR: 5 points, yaw π ± 10° (ST04–07) | both (`start_pose_dr` flag) |
| Goals | ST01–03 9-goal list (default goal (−1.8, −2.5) also available) | 9-goal list (configurable) |
| Collision | GT centre clearance < 0.22 m (= r 0.17 + 0.05) | GT **footprint** clearance c < 0.05 m safety margin (r = 0.177 m → centre 0.227 m); physical collision c ≤ 0; contact sensor backstop |
| Goal reached | distance < 0.20 m | `goal_distance()` (threshold applied in M4) |

## 4. Validation (scripts/m3/m3_validation.py, 2,048 envs; `results/m3/m3_validation.json`)

**V1: layouts** (20,480 randomized layouts):
* 0 violations of A/B/C;
* all offsets within range;
* 0 offset-sampler fallbacks;
* PhysX baffle poses equal the tables within 11 µm.

KS tests against 20,000 reference samples (p-values):

| Quantity | p |
|---|---|
| offsets Z1–Z4 (per reset) | 0.81, 0.02, 0.10, 0.83 |
| bank lengths Z1–Z4 | 0.79, 0.45, 0.86, 0.88 |
| bank thicknesses Z1–Z4 | 0.57, 0.35, 0.28, 0.17 |
| d(Z1, Z3) | 0.50 (mean 1.385 vs 1.382 m) |

One of 13 p-values at 0.02 is consistent with chance. In an offline simulation with the identical samplers, the offset p-values were 0.2–0.9.

**V2: LiDAR:** 6 rounds × 2,048 random robot poses (noise off), oracle with per-env walls and baffles.

| | Rays | p99 | > 1 cm |
|---|---|---|---|
| fixed walls | 451,517 | 0.56 mm | 4 |
| baffles | 33,088 | 0.66 mm | 1 |
| no-return | 6,915 | 0 | 1 |

The few > 1 cm cases are grazing rays at box corners, as in M2. The p99 is somewhat above M2's 0.03 mm because the robots were teleported and measured 0.2 s later, while still settling (sub-degree tilt of the scan).

**V3: collisions:** each of 2,048 robots is placed at a random free pose and driven straight at 0.15 m/s for 8 s, forward and backward, through the calibrated command model. Clearance is evaluated every physics step against the base_link contact sensor (horizontal force > 1 N; the caster is frictionless, so the ground contributes none).

| | Forward | Backward |
|---|---|---|
| envs with contact | 1,315 | 1,326 |
| clearance at first contact: median (p01 / p99) | −0.06 mm (−0.83 / +0.49 mm) | −12.68 mm (−12.87 / −12.61 mm) |
| contact while c > 1 cm | 0 | 0 |
| c < −3 cm without contact | 0 | 0 |

* The footprint radius 0.177 m reproduces the front bumper to within 1 mm.
* At the rear (base r = 0.164 m) ground truth flags collision 13 mm before contact, which is conservative.
* The safety flag is c < m_safe (default 0.05 m, configurable); physical collision is c ≤ 0; both are logged separately.

**V4: resets:**
* start pose exact (0 error), velocities ≤ 2e-11 m/s, command-model state 0;
* partial reset of 683 envs left the others undisturbed: they moved only their own 2 mm per 10 ms physics step at 0.2 m/s;
* goals always from the list (uniform: 218–238 per goal over 2,048 envs);
* start-pose DR: all starts on the 5 reference points (2,039–2,106 each of 10,240), yaw offsets within ±10.0°;
* minimum footprint clearance at the start: 0.158 m (reference: 0.335 m from the centre = 0.158 m footprint).

**Figures:**
* `figures/m3_randomized_envs_topview.png`: 8 randomized envs with robot, start, goal, LiDAR vs oracle;
* `figures/m3_randomization_distributions.png`.

## 5. Benchmark (scripts/m3/m3_benchmark.py; GPU 5, nothing else running; `results/m3/benchmark.jsonl`)

**Protocol:**
* complete environment, LiDAR noise on;
* random commands through the calibrated command model;
* automatic partial resets on safety violation, goal (< 0.20 m) or timeout (350 steps);
* 150 control steps after warm-up.

| Envs | Control steps/s (incl. resets) | Physics env-steps/s | Resets per control step | GPU memory | Start-up | Unstable / NaN |
|---|---|---|---|---|---|---|
| 4,096 | 20.2k | 0.40 M | 53 | 4.7 GiB | 156 s | 0 / no |
| 16,384 | 51.1k | 1.02 M | 211 | 8.7 GiB | 638 s | 0 / no |

* Random driving ends almost every episode by the safety margin within a few seconds. This is a reset stress test, about 1.3 % of envs reset every step.
* Compared with M2's simpler scene (66k at 16,384), the full environment costs about 23 % more per step: 4 extra tracked meshes per env, contact sensor, noise, and resets every step.

## 6. Deviations from the reference and their implications

| # | Deviation | Justification | Implication |
|---|---|---|---|
| D1 | **Z1–Z4 sizes are fixed per env** (drawn at start-up from a bank of K = 1,024 reference samples); **offsets are resampled every reset** | the mesh ray caster bakes mesh sizes at start-up (M2); per-env sizes keep the joint distribution exact, offsets keep per-reset variation | each env replays one size set across episodes; with thousands of envs the batch covers the distribution (KS above). Larger K improves diversity but increases ray-caster start-up (duplicate check ~ N × K) |
| D2 | baffles are **kinematic rigid bodies** posed at reset; `replicate_physics=False` | per-env geometry and physical robot–baffle contact; kinematic bodies do not interact with the static walls they touch | physical collisions with baffles are simulated. Start-up is slower: 10.6 min at 16,384 envs (USD per-env scaling, per-env PhysX parsing, ray-caster mesh parsing). Phase 2's kinematic-body crash occurred with per-step pose writes; reset-time writes are stable (V1, V3, benchmark) |
| D3 | baffles are **0.6 m** tall | they are wall segments in the reference (`WALLS` list); 0.5 m was agreed for obstacles, applied to the moving obstacles in M6 | configurable (`baffle_height`); no effect on the 2D LiDAR (scan plane 0.188 m) |
| D4 | collision = **footprint** clearance (r = 0.177 m) with a separate safety margin, instead of centre clearance < 0.22 m | requested; the TB4 bumper is 0.177 m (M1) | equivalent threshold 0.227 m from the centre (reference 0.22 m with r = 0.17). Terminal semantics are set in M4 |
| D5 | footprint modelled as a **circle** of the bumper radius | simple, exact at the front (V3) | conservative by 13 mm at the rear (base r = 0.164 m) |
| D6 | no native LiDAR scan timing; robot dynamics from PhysX (not analytic) | approved Mode A design; timing model optional later | as in M1/M2 |
| D7 | evaluation with the reference's own episode seeds is not yet wired (start/goal/offsets are drawn on the GPU) | planned for the M5 evaluation harness (CPU reference samplers → upload) | none for training |

## 7. Limitations

* **Start-up time** is linear in env count: about 2.6 min at 4,096 and 10.6 min at 16,384. It is acceptable for long training runs; for debugging use ≤ 1,024 envs (≈ 30 s). Profiling which part dominates (USD scaling loop, PhysX parsing or ray-caster mesh parsing) is a possible optimization.
* **Rare LiDAR grazing-corner errors:** ≈ 1e-5 of rays.
* **Size diversity** limited to the bank (1,024 configurations by default; configurable).

## 8. Reproduce

```bash
cd ~/isaac_lab/spatiotemporal_sensing
python -m pytest tests/                                   # 15 passed, 1 skipped (Kit-only test)
bin/isaacrun -g auto -- python scripts/m3/m3_validation.py --headless --num_envs 2048
bin/isaacrun -g auto -- python scripts/m3/m3_benchmark.py --headless --num_envs 16384
source env.sh && python scripts/m3/m3_plots.py
```
