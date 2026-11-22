# M1: Articulated TurtleBot4 Standard and differential-drive control

**Date:** 2026-11-21. **Status:** complete, for review. M2 has not been started.
**Scope:** as approved (roadmap `phase3_planning/PHASE3_ROADMAP_MODE_A.md`, M1). Nothing was modified in the reference repositories, the Isaac Lab installation or the conda env. The project code is loaded via `PYTHONPATH` set in `env.sh`.

## 1. Summary

| Acceptance criterion (roadmap) | Result | Status |
|---|---|---|
| Robot rests stably, no drift or jitter > 1 mm/s over 60 s | drift 4.5 mm over 60 s (0.07 mm/s mean), peak 1.2 mm/s, pitch 0.25° | **marginal** (peak 0.2 mm/s above the limit) |
| Steady-state v, ω track within 3 % below saturation | v ≤ 0.45 %, ω ≤ 0.45 % unsaturated; ≤ 0.97 % including saturated cases (35 commands) | pass |
| Scale-saturation behaviour matches the model | saturated commands follow the joint-scaling model (fig. `m1_steady_state_tracking.png`) | pass |
| Delay and rise time match the real characterization within ±0.03 s | in motion: v delay +0.009 s, rise −0.006 s; ω delay +0.017 s, rise −0.015 s | pass |
| LiDAR frame at the URDF pose | `rplidar_link` at (−0.040, 0.000, 0.1929) m in `base_link`, yaw +90.0° (exact) | pass |

Two decisions were taken along the way. Both follow from measurements, and both are documented below:

1. **Wheel colliders:** cylinder → sphere.
2. **Frictionless caster:** the material binding needed a fix because the colliders are instanced.

## 2. Deliverables

| Item | Path |
|---|---|
| TB4 Standard USD (3 bodies, 2 wheel velocity drives) | `assets/turtlebot4_standard/usd/turtlebot4_standard.usd` |
| Build pipeline + provenance + deviations | `assets/turtlebot4_standard/README.md`, `clean_urdf.py`, `convert_urdf.py`, `third_party/` (pinned clones, xacro venv, `ament_index` shim) |
| `TB4_CFG` (ArticulationCfg), geometry constants, **calibrated `TB4_COMMAND_MODEL`** | `source/st_nav/st_nav/assets/turtlebot4.py` |
| Differential drive + command model (batched torch; per-channel delay, lag, gain, scale saturation, acceleration limit) | `source/st_nav/st_nav/control/diff_drive.py` |
| Real-data loader + step metrics; simulation harness | `source/st_nav/st_nav/calibration/` |
| Validation, calibration and plotting scripts; diagnostics | `scripts/m1/` |
| Unit tests (6, CPU) | `tests/test_diff_drive.py` (all pass) |
| Results and figures | `results/m1/*.json`, `results/m1/figures/*.png` |

## 3. Robot model

Source: the official `turtlebot4_description` (Standard; commit `7fd29fb`) and `irobot_create_description` (commit `6c30584`). They are expanded with `xacro` using the `gazebo:=ignition` parameter set and converted with Isaac Lab's `UrdfConverter`. Deviations from upstream are listed with their reasons in `assets/turtlebot4_standard/README.md`:
* Gazebo plugin tags removed.
* `ignition` parameter set instead of `classic`, whose CoM over the axle makes the robot tip backwards.
* Wheel-drop joints fixed.
* Sphere wheel colliders.
* Frictionless caster.
* Solver iterations 16/2 at 200 Hz.

Resulting model:
* 3 rigid bodies: `base_link` 4.99 kg (all fixed links merged) and 2 wheels of 0.2 kg; total **5.39 kg** (unchanged, see §6).
* Wheels: r = 0.03575 m, track 0.233 m.
* Footprint: front bumper r = 0.177 m, base r = 0.164 m.
* CoM 14.7 mm ahead of the axle, 0.137 m above `base_link`.
* At rest: caster 5.6 N, wheels 23.6 N each.

## 4. How the simulated drive was made to track (diagnosis summary)

The first conversion tracked poorly: in-place turns reached 20–60 % of the command and reverse 80 %. Separate tests isolated three causes:

| Cause | Evidence | Fix |
|---|---|---|
| Caster dragging (frictionless material not applied: the importer *instances* the colliders and instance proxies ignore bindings) | material = None on the caster collider; reverse/turn failures only when the caster is pushed or dragged sideways | de-instance the caster collider group at spawn and bind a μ = 0 material (`combine=min`) |
| Contact + drive solve not converged | the robot lifted off the ground tracks exactly; on the ground more effort/damping changes nothing, more solver iterations help | 16 position / 2 velocity iterations (32/4 adds < 1 %) |
| Wide cylinder contacts scrub in turns | wheel kinematics predicted 0.53 rad/s, body turned 0.43 rad/s | sphere wheel colliders (same radius) |

## 5. Actuator calibration (complete step responses, no double counting)

**Data:** 16 real recordings (6 linear, 6 angular, 5 combined; `run_4` aborted after 1.3 s and excluded):
* 5 Hz commands;
* `/odom` at 20 Hz;
* ≈ 3,300 odometry samples per channel.

**Method:** every real command sequence is replayed in simulation with the PhysX drive and contacts in the loop.
* **Candidates:** 702 + 343 command-model parameter sets, each run as its own environment, up to 11,232 parallel robots per run.
* **Scoring:** the complete twist responses are compared with the odometry.
* **No double counting:** the simulated robot by itself responds with no delay and a 17–38 ms rise time (T3), and the fitted parameters only add what is missing.
* **Start-up windows** (2.5 s after a start from rest) are excluded from the fit and reported separately.

**Selected model** (`TB4_COMMAND_MODEL`):

| Channel | Pure delay | Residual lag τ | Gain | Accel. limit |
|---|---|---|---|---|
| linear v | 0.14 s | 0.01 s | 1.00 | 1.0 m/s² |
| angular ω | 0.12 s | 0.05 s | 0.90 | – |

Wheel-speed scale saturation is applied at 0.305 m/s, from your saturation fit.

* **Linear channel:** well identified; the error minimum is sharp in delay and acceleration limit (fig. `m1_calibration_error_surfaces.png`). The 1.0 m/s² limit also removes a start-from-rest overshoot that a pure lag produces.
* **Angular channel:** the error surface is flat (delays 0.08–0.19 s within 2 % of the minimum), because the real angular gain varies between transitions. Delay and lag were therefore chosen from the per-transition timing, within 0.4 % of the minimum error.

**Fit quality:**

| | No command model | Calibrated |
|---|---|---|
| RMS error v (in motion) | 0.0233 m/s | **0.0053 m/s** |
| RMS error ω (in motion) | 0.109 rad/s | **0.086 rad/s** |
| RMS error v / ω (start-up windows) | 0.063 / 0.161 | 0.051 / 0.115 |

**Timing** (same definition applied to real and simulated data: delay = time to 10 % of the step; rise = 10 → 90 %; both relative to the response's own pre-step and settled levels):

| Transitions | Real delay | Sim delay | Real rise | Sim rise |
|---|---|---|---|---|
| v in motion (40) | 0.147 ± 0.040 | 0.156 | 0.100 ± 0.032 | 0.093 |
| ω in motion (71) | 0.113 ± 0.053 | 0.130 | 0.125 ± 0.146 | 0.110 |
| v from rest (25) | **0.531 ± 0.585** | 0.160 | 0.117 ± 0.040 | 0.114 |
| ω from rest (36) | **0.523 ± 0.664** | 0.129 | 0.143 ± 0.139 | 0.113 |

**Figures:**
* `m1_step_responses_all_runs.png` and `m1_step_responses_representative.png`: complete responses, real vs calibrated vs uncalibrated;
* `m1_transition_timing.png`;
* `m1_calibration_error_surfaces.png`;
* `m1_steady_state_tracking.png`.

## 6. Technical limitations and open points

1. **Mass (kept unchanged on your instruction):** 5.39 kg vs ≈ 3.945 kg specified.
   * Tower standoffs 0.26 kg each (implausible for the geometry); 0.22 kg of placeholder sensor-frame masses.
   * The excess raises the CoM, which reduces the margin against tipping back under hard acceleration. The 1.0 m/s² limit keeps pitch ≤ 0.35° from rest.
2. **Firmware start-up delay not modelled:** after ≥ 0.8 s at rest the real robot starts 0.3–1.9 s late (mean 0.53 s), highly variable.
   * The simulation starts after the in-motion delay.
   * In navigation this matters mainly at episode start. It could later be added as a randomized start-up delay in the command model.
3. **Angular gain is command-dependent on the real robot** (0.8–1.3 across transitions; better tracking in arcs than in pure rotation). A single gain of 0.90 is the error-optimal compromise. Gain randomization (as in the reference DR) is the natural next step.
4. **Fitted delays include odometry transport latency.** Robot and PC clocks are not synchronized (≈ 0.42 s offset), so actuation delay and `/odom` latency cannot be separated. The fitted delays are therefore upper bounds of the actuation delay.
5. **Rest jitter:** peak 1.2 mm/s (criterion 1 mm/s); mean drift 0.07 mm/s, i.e. ≈ 5 mm over a 70 s episode. Negligible for navigation; flagged because it is marginally above the stated threshold.
6. **Simulator robustness on this server:**
   * `SimulationApp.close()` occasionally hangs after a run (scripts now `os._exit` after writing results);
   * one diagnostic run hung once during start-up and was stopped by the `isaacrun` timeout; the identical rerun succeeded.
7. **Modelling choices to keep in mind:**
   * sphere wheel colliders (ideal point contact);
   * frictionless caster;
   * fixed wheel-drop suspension;
   * two small collision boxes (LiDAR housing, OAK-D) with odd post-conversion bounds, to re-check in M3 when collisions with walls matter.
8. **Throughput preview** (no ray caster, no MDP, Python step loop): 1.34 M physics env-steps/s with 11,232 robots at 200 Hz, i.e. ≈ 33k control steps/s at 5 Hz.
   * This is an upper-bound indication only. M2 measures the real figure with the arena, the LiDAR and the env loop.

## 7. Reproduce

```bash
cd ~/isaac_lab/spatiotemporal_sensing
bin/isaacrun -g auto -- python scripts/m1/m1_robot_validation.py --headless            # T1-T4
bin/isaacrun -g auto -- python scripts/m1/m1_calibrate.py --headless --mode grid \
    --delays 0.08,0.09,...,0.20 --taus 0,0.01,...,0.08 --kws 0.85,0.875,0.9,0.925,0.95,1.0
bin/isaacrun -g auto -- python scripts/m1/m1_calibrate.py --headless --mode grid --w_fixed 0.12,0.05,0.9 \
    --delays 0.12,...,0.18 --taus 0,...,0.06 --amax 0,0.5,0.75,1.0,1.25,1.5,2.0
bin/isaacrun -g auto -- python scripts/m1/m1_calibrate.py --headless --mode final --final 0.14,0.01,0.12,0.05,0.9,1.0
source env.sh && python scripts/m1/m1_plots.py
python -m pytest tests/test_diff_drive.py
```
