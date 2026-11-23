# M7 report: fixed 3D LiDAR (Mirrorcle raster approximation)

Status: implemented and validated. No policy has been trained with the 3D LiDAR; the network is unchanged.

## 1. What was implemented

**Sensor model.** One ray-casting LiDAR on Isaac Lab's `MultiMeshRayCaster` (`source/st_nav/st_nav/sensors/lidar.py`,
`RayLidar`). The 2D LiDAR is now its default preset; `lidar2d.py` keeps the old names `Lidar2DCfg` / `Lidar2D`. Select
the sensor with `env.sensor: lidar2d | lidar3d`. Every 3D setting lives in `env.lidar3d.*` (`Lidar3DCfg`):

| Setting | Default | Notes |
|---|---|---|
| `h_fov_deg`, `v_fov_deg` | (−14, 14), (−7, 7) | sensor frame: azimuth CCW from forward (left = +), elevation up |
| `cols`, `rows` | 139, 34 | 4,726 rays |
| `placement` | `cell_center` | one ray per cell centre: azimuth ±13.899° in 0.2014° steps, elevation ±6.794° in 0.4118° steps; or `endpoints` |
| `ray_order` | `row_major_top_left` | **provisional**: ray index = row · 139 + col, row 0 = top, col 0 = left. Also `row_major_bottom_left`, `serpentine_top_left`, `column_major_top_left` |
| `ray_table` | null | measured ray set (.csv / .npz: `azimuth_deg`, `elevation_deg`[, `grid_row`, `grid_col`]); row order = ray index. Replaces the grid; no code change needed. Inline lists `azimuths_deg` / `elevations_deg` also work |
| `mount_pos` | (0, 0, 0.3998) m | base_link; centred on the top platform, 5 cm above its top surface |
| `mount_rpy_deg` | (0, 5, 0) | REP-103, R = Rz(yaw)·Ry(pitch)·Rx(roll); **pitch > 0 = tilted down** |
| `range_max`, `range_min` | 12 m, 0 | no return → `range_max` |
| `noise_type`, `noise_sigma` | `gaussian`, 0.01 m | additive N(0, σ²) on rays with a return, clamped to [0, range_max]; `none` disables it (as does `env.lidar_noise: false`). **Provisional** |
| `period_s` | 0.25 (4 Hz) | existing scheduler: scans at control instants |
| `include_floor` | true | the floor is a ray-cast target |

Object heights are also configurable: `env.wall_height`, `env.baffle_height`, `env.obstacle_height` (0.6 / 0.6 / 0.5 m,
unchanged).

**Mount.** Taken from the TB4 meshes (`infrastructure/robot_viz/results/tb4_geometry.npz`, base_link visual and
collision meshes):
* top platform surface at z = 0.3498 m, rim radius ≈ 0.134 m, centred at (0, 0);
* nothing on the robot is higher.

The sensor sits 5 cm above the platform. Its lowest ray points 6.79° + 5° = 11.79° down and clears the robot's outer
radius (0.178 m) by 1.3 cm. A unit test casts all 4,726 rays against every triangle of both meshes: no hit.

**Rotation convention.** The mount quaternion goes into the ray caster's `OffsetCfg.rot`. A pitch of +5° maps the
forward axis to (0.9962, 0, −0.0872), i.e. down; verified in the unit tests and in the simulator (L2 below).

**Observation.** The policy observation is [4,726 ranges / 12 m | 7 navigation features] = 4,733. `observation_space`
is set by the env from the number of rays. The network, rewards, terminations, physics and the actuator model are
unchanged.

**Timing.** All rays of a scan are acquired at once, at the control instant when the scan is due (existing scheduler).
At 4 Hz with 5 Hz control, scans arrive at 0.4, 0.6, 0.8, 1.0, 1.4, … s: 4 per second, 0.2 or 0.4 s apart, never
more than 0.15 s after the due time.

**Noise.** The noise is keyed per (episode seed, acquisition physics tick, ray) through `st_nav/crng.py` (Box–Muller
on two counter-based uniforms), so it is reproducible across policies, as for the 2D sensor.

**Floor.** The ray caster casts against a finite, invisible, collision-free box whose top face is at z = 0 (`/World/LidarFloor`).
It covers every environment plus the LiDAR range. Physics still uses the ground plane.
* I did not use the ground plane itself: Isaac Lab meshes a USD Plane as a 2,000 km square of two triangles, and
  float32 ray–triangle tests on it erred by up to 1.9 mm (median 0.27 mm). With the box the error is ≤ 0.011 mm.
* The box is added only when `include_floor` is set, so the 2D scene is unchanged.

**Scheduler fix.** The due-time tolerance is now half a physics step (5 ms) instead of 1 µs.
* `ep_time` accumulates 0.2 s in float32, so by t ≈ 6 s it sat a few µs below the due time. The 0.25 s schedule then
  missed the scan at t = 6.0 s and shifted the acquisition phase for the rest of the episode.
* The 2D schedule (period = control period) is unaffected: it is bit-identical, see §2.

**Recording (schema 1.1).** `lidar.npz` has a new *per-scan* layout for 3D. The 2D layout is unchanged.
* Each acquired scan is stored once: observed and noise-free ranges as (S, 4726) float32, acquisition step and time,
  sensor origin, and base quaternion.
* `row_scan_index` maps every control step to the scan in its observation.
* The ray table, the sensor- and base-frame ray directions, `grid_row` / `grid_col`, the mount, the noise and the
  config are also stored, and the metadata gains `geometry_heights_m`.
* A 70 s episode takes 8.8 MB (280 scans), i.e. about 0.9 GB per 100-episode evaluation (git-ignored).

**Tools.** `tools/episode_data.py` reads both layouts and adds `lidar_scan` (3D end points), `range_image` and
`quat_apply`. For 3D episodes:
* `make_video.py` and `plot_episode.py` draw the returns top-down (coloured by height, floor grey);
* single-episode videos and figures add the range image;
* all of these tools still need only numpy and matplotlib.

**Other files.**
* `sensors/ray_tables.py`: grid, ray orders and rotation, with no Kit dependency.
* `geometry/analytic3d.py`: 3D oracle (boxes with height, finite cylinders, floor).
* Config templates `Navigation/configs/_templates/{static,chase}_lidar3d.yaml` list every field.
* `tests/sim/make_untrained_checkpoint.py` and `tests/sim/compare_eval_recordings.py`.

## 2. Validation

**Unit tests:** 40 passed, 1 skipped (needs the simulator). The new tests (`tests/unit/test_lidar3d.py`) cover:
* the grid (count, cell centres, steps, row-major index, the four orders as permutations of the same cells);
* the rotation convention against explicit Rz·Ry·Rx matrices (pitch +5° looks down, the identity is bit-exact);
* the keyed Gaussian noise: reproducible, independent of batch, mean 0 and σ 0.0100, no-return rays untouched,
  clamping, disabling;
* self-occlusion against the TB4 meshes;
* the 3D oracle primitives;
* the per-scan round trip, geometry and range image, and the tools running without torch / st_nav / Isaac Lab.

**Simulator (`tests/sim/lidar3d_validation.py`, 64 envs, robots driving with random actions):** 17/17 checks in both
the static and the dynamic scene (OB1 + OB2 moving).

| Check | Static | Dynamic |
|---|---|---|
| L1 structure | 4,726 rays, observation 4,733, floor in the targets, ray caster offset = mount | same |
| L2 directions cast by the ray caster vs the rotated grid | max 1.2e-7 | same |
| L2 boresight in base_link | (0.9962, 0, −0.0872): down | same |
| L2 optical centre | 0.3998 m above base_link, 0.395 m above the floor; boresight reaches the floor at 4.78 m, lowest row at 1.89 m | same |
| L2 floor returns | 536k, \|z\| ≤ 1.9 µm | 447k, ≤ 2.1 µm |
| L3 vs 3D oracle, walls | p99.9 0.002 mm (16.7 M rays) | p99.9 0.003 mm |
| L3 baffles | p99.9 0.006 mm | – |
| L3 floor | max 0.011 mm | max 0.011 mm |
| L3 obstacles (moving cylinders) | – | p50 0.75 mm, p99 4.5 mm (as M6: 32-facet mesh, motion within one physics step) |
| L3 no return | identical | identical |
| L3 rays off by > 1 cm | 3 | 3,842 (0.28 % of obstacle rays) |
| L3 of these, explained by edge-grazing | all 3 | all 3,842 |
| L4 schedule (10 s) | 40 scans at 0.4, 0.6, 0.8, 1.0, 1.4 … s; observation held between scans, new at each scan | same |
| L5 noise vs independent recomputation of the keyed draws | max 2.4e-7 m | same |
| L5 noise statistics (12 M rays) | mean −0.0006 σ, std 1.0003 σ | −0.0001 σ, 1.0000 σ |
| L6 same actions, 2D vs 3D sensor (50 steps) | poses, rewards, terminations, clearance identical (35 terminations) | identical (42) |

"Explained by edge-grazing" means the ray caster's range lies within the oracle's range for primitives grown or shrunk
by 1 mm (boxes) or 2 mm (cylinders), i.e. the ray grazes an edge.

**2D pipeline unaffected:**
* Golden snapshot `m7` vs `m6` (NAV01 `model_200`): scene generation, reset poses, the clean LiDAR scan, non-LiDAR
  observations and policy actions are bit-identical. The only config differences are the 5 new fields (`sensor`,
  `lidar3d`, `wall_height`, `baffle_height`, `obstacle_height`), at defaults that reproduce the previous behaviour.
* Env validation 31/31; dynamic-scene validation 20/20.
* Re-evaluations compared with the official evaluations (NAV01 `m5_initial` `model_200`, `ref_static_c10`: 98/100;
  NAV03 `model_500`, `ref_chase_c10`: 97/100):
  * every `trajectory.csv`, `steps.csv`, `lidar.npz` and `obstacles.npz` of the 200 episodes is identical, as are the
    `episodes.csv` rows;
  * `metadata.json` differs only in `schema_version`, the file-index text, the new `geometry_heights_m` and the policy
    name (output folder).

**3D recording and visualization:**
* Untrained 4,733-input checkpoints, evaluated with 8 episodes each of `ref_static_c10` and `ref_chase_c10`. All are
  timeouts, as expected.
* The recordings contain 280 scans per 70 s episode (4.0 Hz), with `fresh` matching `scan_step`.
* The recorded noise has std 0.01001 m.
* Points reconstructed from the files lie at the floor (z < 0.1 mm) or at most at the 0.600 m wall height.
* `plot_episode`, `plot_eval` and `make_video` (single episode and grid) ran on the 3D and 2D recordings.
* Figure: `results/scan_example_chase.png` (range image, top-down and side view of one recorded scan).

**Throughput (4,096 envs, full RL step with random-policy inference, back to back on one GPU):**

| Scene | 2D | 3D | GPU memory 2D → 3D |
|---|---|---|---|
| static | 20.7k env steps/s | 18.5k (−11 %) | 4.8 → 10.0 GB |
| dynamic OB1 + OB2 | 27.9k | 20.7k (−26 %) | 3.7 → 9.2 GB |

The 2D and 3D runs use different random policies, so their reset counts differ; this is not an exact like-for-like
comparison.

## 3. Limitations and open points

* **Ray order and directions are provisional** (row-major from the top-left, uniform grid). The measured Mirrorcle ray
  table goes into `lidar3d.ray_table`. The ray index fixes the observation layout, the recording and the noise key.
* **Noise model is provisional:** Gaussian range noise only. No dropouts, no mixed pixels at edges, no beam divergence,
  no intensity, no range-dependent σ.
* **Simultaneous acquisition:** no raster timing within a scan and no motion distortion. Scans are quantized to control
  instants (4 Hz on average, 0.2 / 0.4 s apart).
* **Open-top world:** with 0.6 m walls, the top rows pass over walls beyond about 6.5 m and return nothing (seen in
  the example figure). The heights are configurable.
* **The robot is not a ray-cast target.** Self-occlusion of the default mount is excluded offline against the meshes.
  A different mount pose should be re-checked with the unit test's method.
* **Policy network unchanged:** the MLP's first layer now has 4,733 × 256 weights. No policy has been trained with
  the 3D sensor (out of scope).
* **Memory and speed:** about twice the GPU memory at 4,096 envs; recordings are about 9 MB per episode.
* **Cylinders** keep the 32-facet tessellation, as before.

## 4. Follow-up: mount study

`mount_study/REPORT.md` searches the mount height and pitch (x = y = 0, roll = yaw = 0, FoV unchanged) against four
objectives: no rays escaping above the walls, few floor returns, coverage preserved, no self-occlusion.
* The current mount (0.40 m, 5°) lets 0.04–0.08 % of rays escape (9–11 % of poses).
* Recommended: z = 0.50 m, 7.5° down. No escapes even at the worst observed body tilt, no self-occlusion, obstacle
  coverage unchanged, 8.2 / 12.8 % floor returns (static / dynamic).
* The default configuration has not been changed.
