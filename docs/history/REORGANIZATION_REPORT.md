# Repository reorganization: report (2026-11-22)

Implements the approved proposal ([REORGANIZATION_PROPOSAL_v2.md](REORGANIZATION_PROPOSAL_v2.md)) on branch
`reorg`. Checkpoint: tag `pre-reorg` = commit `487c274`. A tarball of the whole folder is at
`~/isaac_lab/backups/spatiotemporal_sensing_pre_reorg_20261122_1230.tar.gz`.

## Changes

* **Moves (commit `7e797a0`, `git mv`, no content changes, 150 renames):**
  * the M5 pipeline → `Navigation/scripts/`;
  * the M5 run → `Navigation/runs/NAV01_static_initial/`, its evaluation → `Navigation/eval/NAV01_static_initial/ref_static_c10/`;
  * tools → `tools/`;
  * regression scripts → `tests/sim/`, unit tests → `tests/unit/`;
  * M1–M4 scripts and results, robot viz and server diagnostics → `infrastructure/`;
  * analyses → `docs/history/`;
  * `third_party/` → `references/robot_description/`.
  * The full old → new table is in [infrastructure/README.md](../../infrastructure/README.md).
* **Edits (commit `f00a83d`):**
  * default output paths of the moved scripts;
  * URDF and asset-pipeline paths;
  * comments in 3 package files.
  * **No change to the simulation, observations, actions, rewards, sensor model or learning algorithm.**
* **New:**
  * `tools/experiments.py`: IDs NAV/TMP/SPA/STS, paths, YAML overrides on the configclass defaults.
  * `Navigation/scripts/train.py --config <ID>.yaml [--set ...] [--run_tag]`: records `params/{env,agent,config}.yaml`
    and `run_info.json`, including the git state of this repo and of the reference repo.
  * `evaluate.py`: named protocol `ref_static_c10`, output `Navigation/eval/<ID>/<protocol>/<policy>/`,
    `--overwrite` guard.
  * NAV01 config and card; `models/navigation/NAV01_model_200.{pt,json}`; docs (README, EXPERIMENTS, architecture,
    references, card template).
* **`.gitignore`:** TensorBoard events; checkpoints except the listed ones; nested clones; regression outputs.

## Regression results

| Check | Result |
|---|---|
| Unit tests | 15 passed, 1 skipped (as before) |
| Golden snapshot pre vs post (`tests/golden/compare_golden.py`) | **all pass**: resolved default configs identical; scene generation (start poses, goals, walls, Z sizes/offsets) identical; robot pose, noise-free scans, non-LiDAR observations at reset: max diff 0.0; policy actions of `model_200.pt` on a fixed 4,196-observation batch **bit-identical** |
| Env validation (31 checks, 1024 envs) | 31/31 before and after |
| Start-pose DR check | identical to the pre-reorg result (counts 469–526 per point, yaw ±10.0°, placement error 4e-6) |
| Scene smoke (standalone scene) | OK, layouts as expected, 0 sampler fallbacks |
| Moved checkpoint, 100-episode `ref_static_c10` | 99/100 vs 98/100 before. Episode 25 fails identically (same step); episode 73 flipped (min clearance 0.0498 m → 0.0508 m against the 0.05 m margin); 91/98 common successes have identical step counts, the rest ±1 step |
| Stale paths, compilation | none outside history/reports; all scripts compile; launcher, TensorBoard and tools work on the moved folders |
| Throughput A/B (same benchmark, pre-reorg worktree vs reorganized, back to back, 4,096 envs) | 21,478 vs 21,309 env steps/s (−0.8 %, within run-to-run noise) |

## End-to-end validation run (YAML interface)

`Navigation/scripts/train.py --config Navigation/configs/NAV01_static_initial.yaml --set agent.max_iterations=201
--run_tag reorg_validation` → `Navigation/runs/NAV01_static_initial/2026-11-22_12-43-08_ppo_n4096_s42_reorg_validation/`.

Its resolved `params/env.yaml` and `agent.yaml` are identical to the original run's. The only differences are
`log_dir`, `max_iterations` (201 by design), and five evaluation-only fields added after the original run at their
training defaults.

| Metric | Original (M5) | Reorganized |
|---|---|---|
| First iteration with goal rate ≥ 0.9 / ≥ 0.98 | 42 / 46 | 48 / 53 |
| Iterations 150–200 mean: return / goal rate / length | 87.33 / 99.55 % / 147.3 | 87.31 / 99.57 % / 147.6 |
| Iteration 200: return / goal rate / policy std | 87.54 / 99.89 % / 0.335 | 86.95 / 99.89 % / 0.316 |
| `ref_static_c10` evaluation of `model_200` | 98/100 (99/100 on re-evaluation) | **100/100** |
| Successful episodes: duration / mean min clearance | 29.0 s / 0.090 m | 29.2 s / 0.117 m |
| Training throughput (median) | 20.9k env steps/s, 6.26 s/iteration | 20.2k env steps/s, 6.50 s/iteration (−3.7 %) |

Figure and data: `tests/golden/training_comparison.{png,json}`.

**Assessment:**
* The learning curves overlap until about iteration 25, then drift slightly. That is expected: RSL-RL and PhysX on
  the GPU are not bit-reproducible, and the LiDAR noise generator is unseeded.
* Both runs converge to the same plateau. The reorganized run reached 0.9 goal rate 6 iterations later, which is
  within normal run-to-run variance.
* Its policy is slightly more conservative (larger clearance) and scored 100/100.
* The 3.7 % lower training throughput is **not** caused by the code. The back-to-back benchmark of the pre- and
  post-reorganization code differs by 0.8 %. The training runs were made at different times on a shared server.

## Limitations and notes

* **Evaluation is not fully reproducible.** The LiDAR noise generator is unseeded and GPU physics is
  non-deterministic, so borderline episodes (clearance within about 1 mm of the margin) can flip between
  re-evaluations. Seeding the sensor noise from the evaluation seed would fix this, but it changes sensor/eval
  behaviour, so it was left for a separate decision.
* **The size bank depends on `~/adaptive_sensor_policy` at run time.** That tree is on `spatiotemporal` with
  uncommitted changes. Every run now records its commit and dirty state in `params/run_info.json`, and
  `docs/references.md` documents the dependency.
* **The historical reports (M1–M4) still mention the old paths.** `infrastructure/README.md` maps them.
* **The server's inotify watch limit is exhausted.** Isaac Sim logs `errno=28` "No space left on device" for file
  watchers. This is harmless for headless runs; the disk has 395 GB free.
* **Not implemented, as agreed:** `sweep.py` (configs carry an empty `sweep:` section and `train.py --set` is the
  hook), scene registry and scene/sensor selection (M6/M7), evaluation library, scaffolding CLI.
