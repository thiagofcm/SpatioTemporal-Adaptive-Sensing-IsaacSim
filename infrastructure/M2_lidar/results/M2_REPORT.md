# M2: 40-beam 2D LiDAR (GPU ray casting) and GPU-parallel performance

**Date:** 2026-11-22. **Status:** complete; approved for continuation to M3 (benchmark finalized after the clean sweep).
**Scope (approved):**
* integrate the 2D LiDAR on Isaac Lab's GPU ray casting;
* review consistency with the reference simplified environment and the physical mounting;
* validate against analytic distances;
* reproduce and validate the noise model;
* benchmark throughput, memory and stability;
* compare 200 Hz with 100 Hz physics.

M1 results and decisions remain as approved: sphere wheels, frictionless caster, `ignition` parameter set, calibrated command model; limitations as documented in `results/m1/M1_REPORT.md`.

## 1. Summary

| Item | Result |
|---|---|
| Ray-cast accuracy vs analytic (noise off), 327,680 rays, random poses and orientations | walls: max 0.66 mm, p99 0.03 mm; cylinders: p99 3.7–4.5 mm (32-facet tessellation); 28 rays (8.5 × 10⁻⁵) off by > 1 cm, all near-tangent rays on cylinder facets |
| Beam ordering / convention | beam i at i × 9° CCW from robot forward; mover placed on beam k is the minimum of beam k in 1,023–1,024 of 1,024 envs (the rest: another object closer, as the oracle confirms) |
| Moving robot + moving obstacle (2.05 M rays) | walls p99 0.03 mm, moving cylinder p99 4.4 mm; the mesh tracking follows the obstacle every step |
| Range limits | no return → exactly 3.5 m with `hit = False`; no reading above 3.5 m |
| Mounting | ray origin = `base_link` + R·(−0.040, 0, 0.1929) within 8 µm; scan plane 0.188 m above ground |
| Noise model (reference datasheet-uniform) | u = e/(a(d)·d): mean −9e-5, var 0.3333 (1/3), bounds ±1.000; KS p = 0.10–0.96 over 10 subsamples; per-band std exact; beam and scan correlations < 0.004 |
| Throughput / memory / stability | 16,384 envs at 100 Hz: 66k control steps/s full loop, 7.4 GiB, 0 unstable; see §5 |
| 200 Hz → 100 Hz physics | all M1 criteria still met; about 2x throughput → **100 Hz adopted** (§6) |

## 2. Sensor implementation (modular)

```
st_nav/sensors/base.py      SensorOutput (ranges, ranges_clean, hit, valid, dirs_b, t_stamp) + RangeSensor interface
st_nav/sensors/patterns.py  BeamListPatternCfg: explicit (azimuth, elevation) ray lists for Isaac Lab ray casters
st_nav/sensors/noise.py     DatasheetUniformRangeNoise (reference model; seeded torch generator)
st_nav/sensors/lidar2d.py   Lidar2DCfg / Lidar2D on MultiMeshRayCaster (40 beams, 9 deg, 3.5 m, TB4 mount)
st_nav/geometry/analytic2d.py  batched analytic ray cast + ground-truth clearance (oracle; GT for M3/M4)
st_nav/scenes/              reference scene tables -> 3D walls (0.6 m) / cylinders (0.5 m); harness
```

* Rewards, terminations and observations will only see `SensorOutput`.
* The `valid`, `dirs_b` and `t_stamp` fields already carry what the Mirrorcle 3D LiDAR and programmable spatial and temporal sensing will need:
  * per-ray directions, which can change per env and per step;
  * masked ray slots;
  * per-ray acquisition times.
* A programmable pattern only needs a different `BeamListPatternCfg` (or a per-step rewrite of the ray directions), with no change to the MDP.
* Ground-truth geometry (clearance, collision) is computed analytically from the same scene tables and is independent of the sensor.

**Isaac Lab detail found and handled:**
* `MultiMeshRayCaster` applies the mounting offset to the ray *start points*, while `data.pos_w` remains the attach-frame (`base_link`) pose.
* Measuring ranges from `pos_w` produced 2–4 cm errors in early tests.
* `Lidar2D` therefore measures from the true ray origin. V4 checks that origin against the ray caster's own ray starts (identical).

## 3. Consistency with the reference sensor and the physical mounting

| Aspect | Reference simplified env | Physical TB4 / real pipeline | Isaac Lab (M2) |
|---|---|---|---|
| Beam count / spacing | 40, `linspace(0, 6.28, 40)` incl. both ends: 9.23° spacing, beam 39 0.18° short of beam 0 | real node picks the native sample nearest each training direction | **40, uniform 9°** (approved) |
| Beam order | CCW from robot forward, beam 0 = forward | same (directions in robot frame) | same |
| Origin x | −0.04 m along heading | `rplidar_link` x = −0.040 m (URDF) | −0.040 m |
| Origin height | 2D (code comment cites 0.171 m from a TurtleBot3 model) | 0.1929 m above `base_link` (URDF), 0.188 m above ground in sim | 0.1929 m (scan plane 0.188 m) |
| Sensor yaw | – (beams defined in robot frame) | `rplidar_link` yaw +90°; the real node subtracts it (`REAL_LIDAR_YAW_IN_BASE`) | rays defined in the base frame; the +90° only matters when mapping real scans |
| Range cap / no return | 3.5 m; no return → 3.5 m | `inf`/`nan` → 3.5 m | 3.5 m; no return → 3.5 m, `hit = False` |
| Minimum range | none | RPLIDAR returns `inf` below ≈ 0.15 m → mapped to 3.5 m | none by default; `below_min_policy="max"` reproduces the real pipeline (off, to match the reference) |
| Noise | U(±a(d)·d), a = 1 % ≤ 3 m, 2 % ≤ 5 m, 2.5 % beyond; from the true distance; per beam, per scan; no-return unchanged; clip [0, 3.5] | – | **identical**, validated (§4) |
| Scan timing | native timing model (7.4–7.8 Hz arrivals, receive delay) | real | not in M2. One scan per 5 Hz control step; `t_stamp` field ready |
| Self-occlusion | none | tower standoffs cross the scan plane | none (the robot is not a ray-cast target). Masking of standoff beams can be added if wanted |

**Changes needed before real-robot deployment:**
1. **Beam table:** the real node's `GAZEBO_BEAM_ANGLES` (the `linspace` quirk) must be replaced by the uniform 9° table so that real beams match the simulated ones.
2. **Minimum range:** decide whether to train with `below_min_policy="max"`, to match the real `inf` → 3.5 m behaviour near obstacles.

## 4. Validation (scripts/m2/m2_lidar_validation.py, 1,024 envs)

Scene: reference `corridor_static_object` walls (14 boxes, 0.6 m), 3 static cylinders (r = 0.17, 0.10, 0.05 m) and 1 tracked moving cylinder (r = 0.17 m), all 0.5 m tall.

**Oracle:** the torch port of the reference ray cast, unit-tested to match the reference `_raycast` within 1e-4 m on 300 random poses (`tests/test_analytic2d.py`). Each oracle evaluation uses the sensor pose reported by PhysX.

* **V1:** 8 rounds of random robot and mover poses with random yaw.
  * Walls (314k rays): max 0.66 mm, p99 0.03 mm.
  * Cylinders (11k rays): median 0.5–0.8 mm, p99 3.7–4.5 mm.
  * The cylinder error is the 32-facet tessellation Isaac Lab uses for USD cylinders. A simulation of a 32-gon predicts p99 4.4 mm for r = 0.17 m and a ≈ 0.2 % chance of a grazing hit/miss flip, which produces the 28 rays above 1 cm.
* **V2:** beam k = env % 40 carries the mover; the expected beam reads 0.3306 m (expected 0.33 m); argmin agrees with the oracle in 1,023–1,024/1,024.
* **V3:** robots driving with the calibrated command model under random commands while a mover circles (50 control steps).
  * Walls p99 0.03 mm; moving cylinder p99 4.4 mm.
  * 287/51,200 scans were excluded where the sensor origin was inside an obstacle or the robot tilted > 2°. Those came from robots driving through the collision-free *test* mover or ramming walls, cases the 2D oracle cannot represent.
* **V4:**
  * no-return rays report exactly 3.5 m;
  * origin = base + URDF offset within 8 µm, identical to the ray caster's ray starts;
  * scan plane 0.188 m above ground; base_link origin 4.4 mm below ground level, as the URDF defines.
* **V5:** see §1. Additional properties:
  * no-return rays are never perturbed;
  * rays within 2 % of the cap can be clipped at 3.5 m (60 of 8 M samples), as in the reference.

**Figures (`results/m2/figures/`):**
* `m2_lidar_scans_topview.png`: 8 random poses, GPU rays vs oracle end points;
* `m2_lidar_ranges_and_ordering.png`;
* `m2_lidar_validation_errors.png`: error histograms and the moving trace;
* `m2_lidar_noise.png`.

## 5. Performance (scripts/m2/m2_benchmark.py, one RTX A6000)

**Protocol:**
* one sequential sweep on GPU 5 with no other process on that GPU at the start of any run (run log `benchmark_clean_runlog.txt`) and no other benchmark of ours running;
* robots get random (v, ω) commands, so they also hit walls (contact stress);
* phases A (physics only), B (+ LiDAR read per control step) and C (full control loop: calibrated command model every physics step, scripted moving obstacle, ground-truth clearance, 47-dim observation);
* 30–100 control steps per phase.

Records: `results/m2/benchmark_clean.jsonl` (sweep) and `benchmark_clean_repeats.jsonl`. Earlier sweeps that overlapped with a second session are kept apart in `results/m2/contaminated/` and **not used**.

Throughput is in control steps per second (5 Hz, 0.2 s simulated each):

| Envs | Physics | Solver | A physics only | B + LiDAR | C full loop | C physics env-steps/s | LiDAR cost | loop cost | GPU mem | setup |
|---|---|---|---|---|---|---|---|---|---|---|
| 512 | 200 Hz | 16/2 | 3.5k | 3.3k | 2.9k | 0.12 M | 7 % | 17 % | 2.9 GiB | 12 s |
| 1,024 | 200 Hz | 16/2 | 6.1k | 5.8k | 4.3k | 0.17 M | 5 % | 30 % | 3.1 GiB | 22 s |
| 2,048 | 200 Hz | 16/2 | 10.9k | 10.1k | 7.5k | 0.30 M | 7 % | 31 % | 3.4 GiB | 43 s |
| 4,096 | 200 Hz | 16/2 | 19.1k | 17.5k | 13.9k | 0.56 M | 9 % | 27 % | 3.9 GiB | 84 s |
| 8,192 | 200 Hz | 16/2 | 30.3k | 28.9k | 24.1k | 0.96 M | 4 % | 20 % | 5.1 GiB | 168 s |
| 16,384 | 200 Hz | 16/2 | 40.8k | 38.2k | 34.2k | 1.37 M | 6 % | 16 % | 7.4 GiB | 333 s |
| 512 | 100 Hz | 16/2 | 6.7k | 5.9k | 5.5k | 0.11 M | 11 % | 18 % | 2.9 GiB | 12 s |
| 1,024 | 100 Hz | 16/2 | 11.9k | 10.6k | 8.2k | 0.16 M | 11 % | 32 % | 3.0 GiB | 22 s |
| 2,048 | 100 Hz | 16/2 | 22.1k | 19.1k | 14.9k | 0.30 M | 14 % | 32 % | 3.4 GiB | 43 s |
| 4,096 | 100 Hz | 16/2 | 38.5k | 33.3k | 27.2k | 0.54 M | 14 % | 29 % | 3.9 GiB | 84 s |
| 8,192 | 100 Hz | 16/2 | 59.9k | 53.8k | 45.9k | 0.92 M | 10 % | 23 % | 5.1 GiB | 167 s |
| 16,384 | 100 Hz | 16/2 | 80.7k | 74.2k | 66.0k | 1.32 M | 8 % | 18 % | 7.4 GiB | 336 s |
| 4,096 | 100 Hz | 32/4 | 29.7k | 26.4k | 22.5k | 0.45 M | 11 % | 24 % | 3.9 GiB | 85 s |
| 16,384 | 100 Hz | 32/4 | 59.0k | 54.6k | 50.3k | 1.01 M | 7 % | 15 % | 7.4 GiB | 335 s |

**Coherence and repeatability** (`scripts/m2/m2_bench_check.py`):
* A ≥ B ≥ C in every run;
* throughput monotonic in env count;
* 100 Hz faster than 200 Hz at every size;
* 0 unstable envs and no NaN in any run;
* repeat runs within 3.5 % of the sweep:

    repeat [4096, 200, (16, 2)] A: 19.1k vs 19.8k (3.5 %) B: 17.5k vs 18.0k (2.9 %) C: 13.9k vs 14.4k (3.3 %)
    repeat [16384, 200, (16, 2)] A: 40.8k vs 40.0k (2.1 %) B: 38.2k vs 37.5k (1.8 %) C: 34.2k vs 33.5k (2.0 %)
    repeat [4096, 100, (16, 2)] A: 38.5k vs 38.9k (0.9 %) B: 33.3k vs 33.3k (0.0 %) C: 27.2k vs 27.0k (0.6 %)
    repeat [16384, 100, (16, 2)] A: 80.7k vs 80.3k (0.5 %) B: 74.2k vs 72.5k (2.3 %) C: 66.0k vs 66.1k (0.2 %)

**Observations:**
* **Best measured:** 16,384 envs at 100 Hz, 66k control steps/s for the full loop (1.3 M physics env-steps/s), 7.4 GiB GPU memory. At 200 Hz the same size gives 34k.
* **LiDAR cost:** 4–14 % of physics-only throughput. The full control loop costs 15–32 %, mostly Python per-step overhead of the command model, which is amortized at large env counts.
* **Python-bound at small sizes:** each physics step has a fixed overhead, so throughput grows almost linearly up to 4k envs and still grows at 16k.
* **Start-up time** grows linearly: about 5.5 min to build 16,384 envs (scene cloning and ray-caster mesh parsing).
* **Memory** grows from 2.9 GiB (512 envs) to 7.4 GiB (16,384 envs), well within the A6000's 48 GiB.

Figure: `figures/m2_benchmark.png`.

## 6. 200 Hz vs 100 Hz physics

M1 criteria re-run at 100 Hz (`m1_robot_validation.py`, `m1_calibrate.py --mode final`, `m2_rate_check.py`):

| | 200 Hz, 16/2 (approved) | 100 Hz, 16/2 | 100 Hz, 32/4 |
|---|---|---|---|
| Steady-state max error v / ω | 0.97 % / 0.45 % | 1.46 % / 0.58 % | 0.89 % / 0.24 % |
| Rest: peak speed / drift (60 s) | 1.2 mm/s / 4.5 mm | 0.46 mm/s / 9.1 mm | 2.9 mm/s / 4.3 mm |
| Drive rise (no command model) | 17–38 ms | 30–35 ms | 12–31 ms |
| Calibrated replay RMS v / ω | 5.30 mm/s / 0.0859 rad/s | 5.55 mm/s / 0.0857 rad/s | 5.26 mm/s / 0.0859 rad/s |
| In-motion timing vs real: v delay / rise | +0.009 / −0.006 s | +0.012 / 0.000 s | +0.006 / −0.004 s |
| In-motion timing vs real: ω delay / rise | +0.017 / −0.015 s | +0.019 / −0.007 s | +0.014 / −0.013 s |
| LiDAR validation | pass | pass (identical) | – |

All three variants meet every M1 criterion (≤ 3 % tracking, ±0.03 s timing). The calibrated command model is unchanged; its delays are multiples of 10 ms, so they are exact at 100 Hz.

**Adopted (your decision, 2026-11-22): 100 Hz physics, 16/2 solver iterations, decimation 20 (5 Hz control).**
It roughly doubles throughput (66k vs 34k control steps/s at 16,384 envs, full loop) while meeting every M1 criterion.

## 7. Limitations and notes

1. **Cylinder tessellation (32 facets):** p99 ≈ 4 mm range excess and rare grazing hit/miss flips (≈ 0.2 % of cylinder rays). Below the noise model (1 % of range ≥ 2 mm at 0.2 m). Finer cylinders (custom mesh) are possible if needed for the Mirrorcle phase.
2. **No self-occlusion by the tower standoffs** and no native scan timing or latency: both are possible additions to the sensor module.
3. **The test mover in M2 has collisions disabled** (sensor test only); collision handling of moving obstacles belongs to M3/M6.
4. **Throughput is Python-loop bound at small env counts:** each physics step costs a fixed few ms of overhead, so throughput scales almost linearly with the number of envs until the GPU saturates (§5).
5. **Simulator robustness:** as in M1, scripts exit with `os._exit` after writing results; no hangs occurred in M2 runs.

## 8. Reproduce

```bash
cd ~/isaac_lab/spatiotemporal_sensing
python -m pytest tests/                                         # all pass (1 skipped outside Kit)
bin/isaacrun -g auto -- python scripts/m2/m2_lidar_validation.py --headless [--physics_hz 100]
bin/isaacrun -g auto -- python scripts/m2/m2_benchmark.py --headless --num_envs 4096 --physics_hz 200
bin/isaacrun -g auto -- python scripts/m1/m1_robot_validation.py --headless --physics_hz 100 --tag _it16-2
bin/isaacrun -g auto -- python scripts/m1/m1_calibrate.py --headless --mode final --final 0.14,0.01,0.12,0.05,0.9,1.0 \
    --physics_hz 100 --tag _100hz_it16-2
source env.sh && python scripts/m2/m2_rate_check.py "" _100hz_it16-2 _100hz_it32-4 && python scripts/m2/m2_plots.py
```
