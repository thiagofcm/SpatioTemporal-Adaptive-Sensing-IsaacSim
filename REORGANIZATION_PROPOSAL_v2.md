# Repository reorganization, v2: research-line architecture

Status: **proposal for review, nothing implemented.**

This version replaces the generic `experiments/` tree of v1 (`REORGANIZATION_PROPOSAL.md`) with top-level research
lines, following the reference `spatiotemporal` branch (`AdaptiveFPS/{scripts,config,runs,eval,evaluation}` +
`tools/experiments.py` + `docs/EXPERIMENTS.md`).

Guiding rules:

* **Conventions now, abstractions later.** M6 (moving obstacles) and M7 (Mirrorcle) add scene and sensor plug
  points when they actually need them.
* **No change to the hot path.** The env step, the GPU ray caster, vectorized resets and RSL-RL are untouched.

---

## 1. Recommended tree

```
spatiotemporal_sensing/
├── README.md                       start here: layout, commands, links
├── env.sh, bin/                    launch tooling                                        (unchanged)
├── assets/turtlebot4_standard/     robot USD + conversion pipeline                       (unchanged)
├── source/st_nav/st_nav/           SHARED SIMULATION LIBRARY (Isaac Lab-native)          (unchanged paths)
│   ├── assets/                     robot configs (TB4_CFG, geometry)
│   ├── control/                    actuator / command dynamics
│   ├── sensors/                    lidar2d, noise, scan patterns      (+ mirrorcle3d, acquisition timing: M7)
│   ├── geometry/                   analytic clearance / ray oracle
│   ├── scenes/                     one file per scene + DR samplers   (+ dynamic scenes, scene registry: M6)
│   ├── envs/                       scene runtime (CorridorScene: spawning, vectorized resets, sensor binding)
│   ├── calibration/                calibration harness and constants
│   └── tasks/
│       ├── nav2d/                  planar navigation task: env, cfg, agents/, Gym registration
│       └── sensing/                adaptive-sensing tasks                                (when AdaptiveSensing starts)
├── Navigation/                     RESEARCH LINE 1
│   ├── scripts/                    train.py, evaluate.py                                 (sweep.py when needed)
│   ├── configs/                    NAV01_static_initial.yaml, _templates/
│   ├── runs/NAV01_static_initial/  <run_name>/: TB events, model_*.pt, params/, config.yaml, train.log
│   └── eval/NAV01_static_initial/  README.md (experiment card) + <scenario>/<policy>/ outputs
├── AdaptiveSensing/                RESEARCH LINE 2 (created with its first experiment; same layout)
│   ├── scripts/  configs/<mode>/  runs/<mode>/<ID>/  eval/<mode>/<ID>/     mode = temporal | spatial | spatiotemporal
│   └── evaluation/                 (only if sensing evaluation outgrows a single script, as in the reference)
├── models/navigation/              frozen navigation policies for AdaptiveSensing: NAV01_model_200.pt + .json
├── tools/                          experiments.py (IDs + paths, both lines), monitor_run.py, plot_eval.py,
│                                   make_video.py, benchmark_env.py
├── tests/
│   ├── unit/                       no simulator (current 4 files)
│   └── sim/                        simulator regression: nav env validation (31 checks), start-pose DR, scene smoke
├── infrastructure/                 M1–M5 development history + server tools (results kept, scripts still runnable)
│   ├── README.md                   index: milestone → question → report → key numbers
│   ├── M1_robot_calibration/  M2_lidar/  M3_corridor_scene/  M4_rl_environment/  M5_initial_navigation/
│   │                               each: REPORT.md, scripts/, results/
│   ├── robot_viz/                  scripts/, results/
│   └── server/                     diag_empty.py, diag_raycaster.py, vkprobe.py, PHASE1_INSTALLATION_REPORT.md
├── docs/                           EXPERIMENTS.md (registry + conventions), architecture.md, research_plan.md,
│                                   server_setup.md, references.md, templates/experiment_card.md, history/
└── references/robot_description/   (was third_party/) TB4 + Create3 description clones, xacro shim
```

Three choices that reduce work compared with v1:

* **The `tasks/nav2d` name stays.** It means planar navigation, which stays true with a 3D LiDAR. Keeping it saves
  every import edit.
* **No shared training or evaluation framework.** `Navigation/scripts/` holds today's M5 scripts. `AdaptiveSensing/`
  gets its own scripts when its requirements are known. Only `tools/experiments.py` (paths and IDs) is common.
* **No experiment-scaffolding CLI or evaluation library yet.** Cards are created from a template by hand, as the
  reference did before its tooling grew.

## 2. Conventions (established now)

| Item | Convention | Example |
|---|---|---|
| Experiment ID | `NAV<NN>_<scene>_<desc>`; AdaptiveSensing: `TMP<NN>` / `SPA<NN>` / `STS<NN>` + `_<scene>_<desc>`. The card records the matching reference experiment (e.g. `ST04`) | `NAV01_static_initial` |
| One ID, four places | `configs/<ID>.yaml`, `runs/<ID>/`, `eval/<ID>/` (card = `eval/<ID>/README.md`), registry row in `docs/EXPERIMENTS.md` | |
| Config | YAML: `exp_id`, `task`, `env:` and `agent:` dotted overrides of the configclass defaults; copied into each run folder; never edited once runs exist | see §4 |
| Run folder | RSL-RL's `<date>_<run_name>`; `run_name` from the config (default `ppo_n<envs>_s<seed>`) | `2026-11-22_00-44-09_m5_initial_4096` (kept as is) |
| Eval scenario | `<protocol>`: a named set of seeds, goal and timeout. Policies in the same scenario folder saw identical episodes | `ref_static_c10` (100 episodes, obstacle seeds 3000+i, start seeds 4000+i, goal (−2, −2), 80 s) |
| Eval policy | `<run_folder>_<checkpoint>` | `2026-11-22_00-44-09_m5_initial_4096_model_200` |
| Frozen policy | `models/navigation/<ID>_<checkpoint>.pt` + `.json` (task ID, obs/action spec, sensor, source run, eval result) | `NAV01_model_200.pt` |

## 3. How the upcoming work fits

| Development | Where it goes | Change it requires |
|---|---|---|
| Dynamic scenes (M6) | `st_nav/scenes/corridor_dynamic_*.py`; moving-obstacle bodies driven in `envs/corridor_scene.py` (same GPU-vectorized reset/step pattern as the static baffles) | a small scene registry and a `scene` field in the env cfg, added **in M6**; then `NAV02_chase-ob1_...` is just a new config |
| Mirrorcle 3D LiDAR (M7) | `st_nav/sensors/mirrorcle3d.py` next to `lidar2d.py` (same `SensorOutput` interface, same `MultiMeshRayCaster` backend) | a `sensor` field in the env cfg; the observation size follows the sensor, added **in M7** |
| Scan patterns, acquisition timing | `st_nav/sensors/patterns.py` (exists), a timing scheduler next to it; the env already decouples sensing from control (`lidar_period_s`) | extend in place |
| Navigation with different sensors | `Navigation/configs/NAV<NN>_...yaml` choosing the sensor; frozen-model `.json` records the sensor and obs spec | none beyond M7 |
| Temporal / spatial / spatiotemporal sensing | `st_nav/tasks/sensing/` (env wraps the nav task + frozen policy); `AdaptiveSensing/{scripts,configs,runs,eval}/<mode>/` | new task and scripts when that work starts |
| Frozen navigation reuse | sensing configs reference `models/navigation/<ID>_<ckpt>.pt`; loaded with RSL-RL's actor (state dict + `.json` spec) | a ~30-line loader, written with the first sensing task |

## 4. Minimum changes for the reorganization

1. **Safety:**
   * a tarball of the folder;
   * `git init` with `.gitignore` excluding the nested clones, intermediate checkpoints, TB events, NPZ, logs and
     `.shim`;
   * commit and tag `pre-reorg`.
2. **Moves (`git mv`, no content changes):**
   * `scripts/m5/{train,m5_evaluate}.py` → `Navigation/scripts/{train,evaluate}.py`.
   * `m5_monitor`, `m5_plot_eval`, `m5_animate_eval` and `m4_benchmark` → `tools/`.
   * `m4_validation`, `m5_verify_start_dr`, `m3_smoke` → `tests/sim/`.
   * Other M1–M4 scripts and results, plus the `scripts/viz` and `results/viz` folders → `infrastructure/`.
   * `tests/*.py` → `tests/unit/`.
   * `validation/` → `infrastructure/server/`.
   * `phase2_analysis/` and `phase3_planning/` → `docs/history/`.
   * `third_party/` → `references/robot_description/`.
   * M5 run → `Navigation/runs/NAV01_static_initial/` (folder name kept); its evaluation →
     `Navigation/eval/NAV01_static_initial/ref_static_c10/<policy>/`.
   * Copies only: the phase-1 report and the research plan.
3. **Path edits:**
   * default output paths in the moved scripts (~25 one-line changes; outputs land next to the moved script's
     milestone);
   * `third_party` paths in 2 asset-pipeline scripts.
   * The asset path, `env.sh`, `bin/` and the package location are unchanged.
4. **Small features:**
   * `tools/experiments.py`: ID check and `paths(line, exp_id)`, ~50 lines.
   * `train.py --config <yaml>`: overrides, output to `runs/<ID>/`, config copied into the run. Also records the
     commit and dirty state of `~/adaptive_sensor_policy`, from which the size bank is built at run time. ~40 lines.
   * `evaluate.py`: the experiment ID is taken from the checkpoint path, `--protocol ref_static_c10` is the current
     default, and output goes to `eval/<ID>/<protocol>/<policy>/`. ~20 lines.
5. **Docs:**
   * new `README.md`;
   * `docs/EXPERIMENTS.md` with the NAV01 row;
   * the NAV01 card from the M5 results;
   * `infrastructure/README.md`;
   * `docs/references.md` with the pinned commits;
   * `models/navigation/NAV01_model_200.pt` + `.json`.

**Deferred to when they're needed:**
* scene registry and scene-agnostic `CorridorScene` (M6);
* sensor selection (M7);
* `sweep.py`;
* evaluation library;
* scaffolding CLI;
* a vendored copy of the reference sampler.

## 5. Preserving functionality and performance

Before and after (`tests/golden/`, ~15 min of GPU time each):

| Check | Pass criterion |
|---|---|
| unit tests | 15 pass, 1 skipped |
| env validation (31 checks, 1024 envs) | 31/31 |
| resolved env + agent configs (`dump_yaml`) | identical except path fields |
| observations at reset, 100 envs, fixed seeds | ≤ 1e-6 |
| `model_200.pt` actions on a fixed observation batch | bit-identical (checkpoints hold only state dicts, so they don't depend on module paths) |
| 100-episode `ref_static_c10` evaluation from the moved checkpoint | 98/100, same failing episodes (25, 73), same per-episode step counts |
| 30-iteration training, 4,096 envs | same learning band; throughput within ±10 % of ~21k steps/s |
| launch + artifacts | `isaacrun` works; scene generation smoke passes; TensorBoard and `make_video.py` work on the moved folders |
| stale paths | `grep` for `results/m`, `scripts/m`, `third_party` finds nothing outside history and reports |

Rollback: one commit per step on a `reorg` branch; `pre-reorg` tag plus the tarball.

## 6. Effort (AI-assisted)

| Step | Time |
|---|---|
| Safety + baseline capture | ~45 min (≈ 15 min GPU) |
| Moves + path edits | ~45 min |
| `experiments.py`, `train.py --config`, `evaluate.py` output layout | ~45 min |
| Docs, NAV01 card, registry, frozen-model metadata | ~45 min |
| Regression run + fixes | ~45 min (≈ 15 min GPU) |
| **Total** | **≈ 4 h in one session**, plus your review of the result |

## 7. Decisions to confirm

1. Directory names `Navigation/` and `AdaptiveSensing/`, with AdaptiveSensing split by
   `<mode>` = temporal / spatial / spatiotemporal, as the reference does.
2. ID prefixes `NAV` and `TMP` / `SPA` / `STS`, and `NAV01_static_initial` for the M5 run.
3. Keep `tasks/nav2d` (no rename).
4. Local git only (no remote) for now.
