# Repository reorganization: analysis and proposal

Status: **proposal for review. Nothing has been moved, deleted or modified.** The reference repository was only
read: `git ls-tree` and `git show` on the `spatiotemporal` branch, plus reading tracked files. Its working tree is on
`spatiotemporal` with uncommitted changes (ST06/ST07 configs and eval folders, `docs/EXPERIMENTS.md`), which were
left untouched.

---

## 1. Reference repository (`~/adaptive_sensor_policy`, branch `spatiotemporal`)

### 1.1 Layout

```
envs/
  turtlebot_var_scan_rate.py        the single env (physics, native LiDAR, sector sensing); all DR flags
  scenes/                           ONE FILE PER SCENE + registry (__init__.py: SCENE_MODULES, get_scene)
    _common.py, _obstacle_motion.py   shared walls / obstacle motion types
    corridor_static_object_random.py, corridor_dynamic_chase_random.py, ... (9 scenes)
models/navigation/td3_stage9_ep7400.pt   the frozen navigation actor reused by every sensing experiment
AdaptiveFPS/                        the adaptive-sensing research line
  scripts/                          entry points: train_adaptive_fps_ppo.py, sweep.py, eval_adaptive_fps.py,
                                    fixed_pareto_search.py
  evaluation/                       evaluator library: config (protocol constants), environment, policies,
                                    episode, results, pareto
  config/spatiotemporal/<ID>.yaml   one YAML per experiment (+ _templates/); config/temporal/ = archive
  runs/spatiotemporal/<ID>/<run>/   training runs (info_settings.txt, model.pt, ckpts/, TB events)
  eval/spatiotemporal/<ID>/         experiment card README.md + <scenario>/<policy>/ outputs
  sweeps/<ID>/sweep_<date>/         generated sweep configs and launcher logs (git-ignored)
tools/                              experiments.py (IDs, paths, scaffolding CLI), make_video*.py, plot_scene.py,
                                    select_best_pareto.py
tests/                              plain-assert scripts (one per feature, mostly DR samplers)
docs/                               EXPERIMENTS.md (registry + conventions), research_context.md,
                                    spatiotemporal_ppo.md (implementation), templates/experiment_card.md
```

### 1.2 Conventions worth adopting

| Aspect | Reference convention | Assessment |
|---|---|---|
| Experiment ID | `ST<NN>_<scene>_<short-desc>`, number never reused; ID is the single name of config, runs folder, eval folder and registry row | **Adopt.** Gives every artifact one traceable name |
| Single source of truth for paths | `tools/experiments.py` defines CONFIG/RUNS/EVAL dirs, ID regex, `new`/`list`/`paths` CLI; trainer, sweep and evaluator all import it | **Adopt.** This is the main reason the layout does not drift |
| Experiment card | `eval/<ID>/README.md` from a template: question, hypothesis, what differs, commands, run and eval tables, results, conclusion; status mirrored in `docs/EXPERIMENTS.md` | **Adopt** |
| Config per experiment | YAML with every trainer argument + a `sweep:` section (Cartesian product); trainer refuses unknown keys and a missing `exp_id` | **Adopt in spirit.** Our configs are Isaac Lab configclasses, so the YAML holds overrides on top of them |
| Immutable experiments | "never edit the config of an experiment that has training runs"; re-evaluation needs `--overwrite` | **Adopt** |
| Run folder | `runs/<ID>/<run_name>/`, run name = date + key hyperparameters; `info_settings.txt` lists every randomized variable | **Adopt** (RSL-RL already writes `params/env.yaml` and `agent.yaml`) |
| Eval folder | `eval/<ID>/<scenario>/<policy>/`; `<scenario>` generated from scene + effective DR flags, so all policies in it saw the same seeds; `summary.csv` per scenario for comparison | **Adopt** the structure. Use shorter scenario names |
| Scene registry | one file per scene with a fixed interface (`NAME`, `WALLS`, `MOVING_OBSTACLES`, `SUPPORTED_DR`), validated at import | **Adopt.** Our scene files need an equivalent registry |
| Frozen policy reuse | `models/navigation/` holds the navigation actor that sensing experiments load | **Adopt** for our PPO navigation checkpoints |
| Archive | temporal (single-rate) work kept read-only in parallel trees (`config/temporal`, `runs/temporal`, `eval/temporal`) with a README | **Adopt the idea** for our M1–M5 infrastructure work |
| Git policy | tracks `model.pt` and selected checkpoints; ignores `ckpts/*`, TB events, `*.npz`, `*.log`, sweeps | **Adopt** a similar policy |

### 1.3 Things not to copy

* The research line is named after one method (`AdaptiveFPS/`). Navigation and every sensing direction then share
  one folder name that fits only one of them. This is the problem you raised.
* Scenario folder names are very long (`corridor_dynamic_chase_random_ob1_ob2_only_obstacle_speed_dr_...`). A short
  scenario name plus a `run_config.json` holding the full flags would be easier to work with.
* The single env file (`turtlebot_var_scan_rate.py`) couples robot, sensor, scene handling and sensing actions. Our
  Isaac Lab code is already more modular, and that should be kept.
* The `temporal` archive lives inside the active research folder, not in a separate archive location.

---

## 2. Current Isaac Lab repository (`~/isaac_lab/spatiotemporal_sensing`)

### 2.1 What is there (≈ 200 MB, **not a git repository**)

| Area | Content | Nature |
|---|---|---|
| `source/st_nav/st_nav/` | the Python package (on `PYTHONPATH` via `env.sh`, not pip-installed) | **production** |
| `assets/turtlebot4_standard/` | converted USD (22 MB) + URDF + conversion scripts | **production** asset + one-time pipeline |
| `bin/`, `env.sh` | launcher, CUDA shim, GPU selection | **production** tooling |
| `scripts/m1`–`m5`, `scripts/viz` | 37 scripts: calibration, diagnostics, validation, benchmarks, plots, **and** the M5 train/eval pipeline | mixed |
| `results/m1`–`m5`, `results/viz` | reports, JSON/NPZ data, figures, benchmark logs, **M5 training run, checkpoints and evaluation** | mixed |
| `tests/` | 4 pure-Python unit tests (15 pass, 1 skipped) | regression |
| `validation/` | server diagnostics (empty scene, ray caster, Vulkan probe) | infrastructure |
| `phase2_analysis/`, `phase3_planning/` | migration analysis and planning reports | documentation / history |
| `third_party/` | `turtlebot4` and `create3_sim` **git clones** (50 MB) + xacro shim, used only to regenerate the USD | reference |
| `logs/launcher/` | isaacrun logs (git-ignored) | runtime |
| `README.md` | still describes only the phase-1 launch tooling (stale) | documentation |

Outside the repo, in `~/isaac_lab/`: `Spatiotemporal_plan.txt` (current research plan), `phase1_diagnostics/`
(installation report and diagnostics), `nav_reference/RL-Navigation` (209 MB clone), `IsaacLab_Fork`
(installation, not ours to reorganize), `isaac_lab_tutorial` (14 GB). Also external: `~/adaptive_sensor_policy`,
`~/Sim2Real/...` (real-robot data for M1 calibration).

### 2.2 Production package: dependencies (measured)

```
tasks/nav2d/corridor_env.py ─▶ envs/corridor_scene.py ─▶ scenes/corridor_static_random.py ─▶ scenes/reference.py ─▶ ~/adaptive_sensor_policy (at run time!)
        │                              ├─▶ sensors/lidar2d.py ─▶ sensors/{base,noise,patterns}.py
        │                              ├─▶ assets/turtlebot4.py (TB4_CFG, USD path = <repo>/assets/...)
        │                              ├─▶ control/diff_drive.py (calibrated command model)
        │                              ├─▶ geometry/analytic2d.py (clearance oracle)
        │                              ├─▶ scenes/spawners.py
        │                              └─▶ calibration/sim_harness.py (GROUND_FRICTION, caster material)
        └─▶ tasks/nav2d/corridor_env_cfg.py, agents/rsl_rl_ppo_cfg.py
Development only: scenes/{builder,harness}.py (M2 scene), calibration/real_data.py (~/Sim2Real data)
```

### 2.3 Assessment

* **The package itself is in good shape:** control, sensors, geometry, scenes, tasks and assets are already
  separated. A large rewrite isn't needed.
* **The coupling you want to remove is real:**
  * `CorridorNavEnv` builds `CorridorScene` directly.
  * `CorridorScene` hardcodes the corridor scene module and the `Lidar2D` sensor.
  * The task ID names the scene (`Corridor-Static`).
  * There is no scene registry and no sensor selection in the config.
* **Navigation pipeline mixed with history:** the train/eval entry points live in `scripts/m5/`, and the only
  trained policy is under `results/m5/runs/`, next to M1–M4 diagnostics.
* **No experiment identity:** run folders are `<date>_<run_name>`. There is no config file, card or registry that
  records why a run exists.
* **Output paths are hardcoded:** about 25 scripts write to `$ST_ROOT/results/m<N>/`.
* **External run-time dependency:** building the obstacle-size bank imports the reference sampler from
  `~/adaptive_sensor_policy`.
  * That working tree has uncommitted changes and can switch branches, which could silently change our training
    distribution.
  * The two corridor unit tests read it too.
* **No version control:** there is no checkpoint to return to.
  * `third_party/` contains two nested git clones, which must be excluded or made submodules before `git init`.

---

## 3. Alternative structures

### A. Mirror the reference (one top-level folder per research line)

```
navigation/{scripts,evaluation,config,runs,eval}/
adaptive_sensing/{temporal,spatial,spatiotemporal}/{scripts,evaluation,config,runs,eval}/
source/st_nav/ (shared library)   infrastructure/   docs/   references/
```

* **Advantages:** closest to the reference, so moving between the two repositories is easy. Each research line is
  self-contained.
* **Disadvantages:**
  * Training and evaluation scripts get duplicated per line. The navigation and sensing pipelines both run RSL-RL
    PPO on Isaac Lab tasks, so they are 90 % identical.
  * Evaluation logic (protocol, seeds, episode recording, metrics) gets copied or cross-imported.
  * Reusing navigation policies from the sensing lines goes "sideways" across folders.

### B. Isaac Lab external-project template

```
source/st_nav/st_nav/tasks/{navigation,sensing}/<task>/{env.py, env_cfg.py, agents/}
scripts/rsl_rl/{train.py, play.py}        # --task <gym id>
logs/rsl_rl/<experiment_name>/<date>_<run>/
```

* **Advantages:**
  * Idiomatic Isaac Lab. Task selection goes through the Gym registry, as Isaac Lab users expect.
  * Minimal custom tooling, and the layout is familiar to Isaac Lab users.
* **Disadvantages:**
  * No experiment IDs, configs, cards or registry, so experiments aren't traceable.
  * Configs are Python-only, which makes sweeps awkward.
  * Evaluation outputs have no home.
  * It ignores the research-workflow philosophy you want to keep.

### C. Hybrid (recommended): Isaac Lab library + reference-style experiment management

```
spatiotemporal_sensing/
├── README.md                       start here: layout, common commands, links
├── env.sh, bin/                    launch tooling (unchanged)
├── source/st_nav/st_nav/           THE reusable library (production code only)
│   ├── assets/                     robot configs (TB4_CFG, geometry)             (unchanged)
│   ├── control/                    calibrated command model                     (unchanged)
│   ├── sensors/                    lidar2d, noise, patterns; later mirrorcle, sector/adaptive sensing
│   ├── geometry/                   analytic 2D clearance / ray oracle           (unchanged)
│   ├── scenes/                     one file per scene + registry (corridor_static_random.py; later corridor_dynamic_*)
│   ├── envs/                       scene runtime (CorridorScene → scene-agnostic, Stage C)
│   ├── tasks/
│   │   ├── navigation/             (was nav2d) nav env, cfg, agents/; Gym registration
│   │   └── sensing/                later: temporal / spatial / spatiotemporal sensing envs (frozen nav policy)
│   ├── evaluation/                 protocols (seeds, goal, timeout), episode recording, results writing (from m5_evaluate.py)
│   ├── experiments.py              IDs, paths, registry (single source of truth, as tools/experiments.py)
│   ├── calibration/                sim_harness (production constants), real_data (dev)
│   └── config.py
├── scripts/                        SHARED entry points, the same for every research line and scene
│   ├── train.py                    --config experiments/<line>/configs/<ID>.yaml   (RSL-RL PPO, any registered task)
│   ├── evaluate.py                 --checkpoint ... --protocol ...                 (writes to the experiment's eval/)
│   └── sweep.py                    Cartesian product of `sweep:` → one train.py per member (isaacrun, GPU list)
├── tools/                          experiments.py CLI (new/list/paths), monitor_run.py, plot_eval.py,
│                                   make_video.py, benchmark_env.py, plot_scene.py
├── experiments/                    ALL experiment artifacts, one subtree per research line, identical layout
│   ├── navigation/
│   │   ├── configs/<ID>.yaml (+ _templates/)
│   │   ├── runs/<ID>/<run_name>/            TB events, model_*.pt, params/, info_settings.txt, train.log
│   │   ├── eval/<ID>/README.md              experiment card
│   │   ├── eval/<ID>/<scenario>/summary.csv  + <policy>/{episodes.csv, summary.json, run_config.json, recording.npz, *.png, *.mp4}
│   │   └── sweeps/<ID>/sweep_<date>/        (git-ignored)
│   ├── temporal/   spatial/   spatiotemporal/    (created when the first experiment exists; same layout)
├── models/navigation/              promoted frozen navigation policies (+ <name>.json metadata) for sensing work
├── tests/
│   ├── unit/                       fast, no simulator (the current 4 tests)
│   ├── sim/                        regression suites on the simulator (nav env validation, start-pose DR, scene smoke)
│   ├── golden/                     reference values for the policy/env equivalence checks (§7)
│   └── run_regression.sh           tiered runner (unit | quick | full)
├── infrastructure/                 development and validation history (read-only archive + server tools)
│   ├── README.md                   index: milestone → question → report → key numbers
│   ├── milestones/M1_robot_calibration/{REPORT.md, scripts/, results/}
│   ├── milestones/M2_lidar/  M3_corridor_scene/  M4_rl_environment/  M5_initial_navigation/
│   ├── robot_viz/{scripts/, results/}
│   └── server/{diag_empty.py, diag_raycaster.py, vkprobe.py, PHASE1_INSTALLATION_REPORT.md}
├── docs/
│   ├── EXPERIMENTS.md              registry (all lines) + conventions
│   ├── architecture.md             package structure, how to add a scene / sensor / task
│   ├── research_plan.md            (copy of ~/isaac_lab/Spatiotemporal_plan.txt)
│   ├── server_setup.md             CUDA shim / launcher (moved out of the current README)
│   ├── references.md               external repositories: path, commit, role
│   ├── templates/experiment_card.md
│   └── history/{phase2_analysis/, phase3_planning/}
├── references/
│   └── robot_description/          (was third_party/) turtlebot4, create3_sim clones + xacro_shim (git-ignored clones, commits pinned in docs/references.md)
└── assets/turtlebot4_standard/     (unchanged location; the USD path is resolved relative to the repo)
```

* **Advantages:**
  * There is one training script, one evaluation script and one evaluation library, shared by navigation and every
    sensing line and every scene. Scene, task, sensor and DR settings come from the experiment config.
  * Research lines differ only by their `experiments/<line>/` subtree and their Gym task. Adding temporal, spatial
    or spatiotemporal work means adding a task and a config, not new pipelines.
  * Experiment IDs, cards, registry and immutable configs follow the reference.
  * The library stays idiomatic Isaac Lab: Gym registry, configclasses, RSL-RL.
  * Infrastructure history is kept intact but sits outside the daily workflow.
* **Disadvantages:**
  * More up-front tooling: the experiments module, a YAML-override loader and the evaluation library.
  * Two config layers: the configclass defaults plus the experiment YAML. The resolved config is always dumped to
    `params/`, which keeps this transparent.
  * The layout differs from the reference's single `AdaptiveFPS/` folder, though the per-line subtrees are the
    same shape.

### Comparison

| Criterion | A: mirror | B: Isaac Lab template | **C: hybrid** |
|---|---|---|---|
| Reference-style IDs, cards, registry | yes | no | **yes** |
| Shared train/eval across lines | no (duplicated) | yes | **yes** |
| Scene/sensor/task selection by config | partial | via task IDs only | **yes** (task + overrides) |
| Navigation policy reuse in sensing | awkward | ad hoc | **models/navigation/** |
| Isaac Lab idiomatic | partial | yes | **yes** |
| Migration effort | medium | low | **medium** |

---

## 4. Recommendation: C (hybrid), introduced in three stages

The stages keep "moving files" separate from "changing interfaces", so every step can be validated against the
same baseline:

* **Stage A: structural move only.** Files are relocated and imports and paths updated. No class, function or
  config changes. The registered task ID stays `STNav-TB4-Corridor-Static-v0`.
* **Stage B: experiment management.**
  * `st_nav/experiments.py`, the `tools/experiments.py` CLI, YAML experiment configs for `scripts/train.py` and the
    `experiments/` output layout.
  * The evaluation library is extracted from `m5_evaluate.py` with identical behaviour.
  * The M5 run is back-filled as **NAV01**.
* **Stage C: decoupling (preparation for M6).**
  * A scene registry, plus a scene-agnostic `CorridorScene` that takes the scene module and the sensor config from
    the env config.
  * A generic task ID (`STNav-TB4-Nav2D-v0`). The old ID stays registered as an alias with identical defaults.
  * Optional: replace the run-time import of the reference sampler with a vendored copy. This is accepted only if
    it produces a bit-identical size bank.
  * This stage changes code paths, so it is validated with the strictest checks (§7.3).

### Naming conventions (proposal)

| Item | Pattern | Example |
|---|---|---|
| Research line | `navigation`, `temporal`, `spatial`, `spatiotemporal` | |
| Experiment ID | `<PREFIX><NN>_<scene>_<short-desc>`; prefix per line: `NAV`, `TMP`, `SPA`, `STS` (not `ST`, to avoid confusion with the reference's ST01–ST07); a card field `Reference equivalent` links to the reference experiment when there is one | `NAV01_static_dr-startpose`, later `NAV02_chase-ob1_dr`, `STS01_static_dr-lidar` |
| Scene token | `static` (corridor_static_random), `chase-<obstacles>` (dynamic), as in the reference | |
| Run name | `<date>_<algo>_n<envs>_s<seed>[_<tag>]` (RSL-RL's `<date>_<run_name>` pattern) | `2026-11-22_00-44-09_ppo_n4096_s42` |
| Eval scenario | `<protocol>_<scene>[_<variant>]`, full flags in `run_config.json` | `ref-static-goalC10_corridor_static_random` |
| Eval policy | `<run_name>_<checkpoint>` or `fixed_<...>` for baselines | `2026-11-22_00-44-09_ppo_n4096_s42_model_200` |
| Promoted model | `models/navigation/<ID>_<checkpoint>.pt` + `.json` (task, obs/action spec, normalizer, training commit, eval summary) | `NAV01_model_200.pt` |

### Git policy (proposal)

* **Tracked:**
  * code, docs, reports, configs and cards;
  * evaluation CSV/JSON and figures;
  * the USD asset;
  * promoted models and the selected checkpoints of each run.
* **Ignored:**
  * intermediate `model_*.pt`, TensorBoard events (kept on disk, not in git), `*.npz` recordings, `*.mp4`
    (except under `eval/`, as in the reference);
  * `sweeps/`, `logs/`, `.shim/`, `__pycache__`, `.pytest_cache`;
  * the robot-description clones.
* **Archived by tarball:** before the migration, a tarball of the whole folder goes outside the repo (§7.1), so
  nothing that git ignores can be lost.

---

## 5. How the workflow looks under C

```bash
# create an experiment (assigns the number, copies the template, creates folders + card + registry row)
python tools/experiments.py new navigation static_dr-startpose --template static_dr \
    --question "Does PPO solve the randomized static corridor with start-pose DR at 4,096 envs?"
#   -> experiments/navigation/configs/NAV02_static_dr-startpose.yaml, eval/NAV02_.../README.md, registry row

# train (the config names the task, the scene and the overrides; the trainer refuses a missing/unknown exp_id)
bin/isaacrun -g 5 -- python scripts/train.py --headless --config experiments/navigation/configs/NAV02_static_dr-startpose.yaml
#   -> experiments/navigation/runs/NAV02_static_dr-startpose/2026-12-01_10-00-00_ppo_n4096_s42/
#        events.out.tfevents.*, model_*.pt, params/{env,agent}.yaml, info_settings.txt, train.log

python tools/monitor_run.py experiments/navigation/runs/NAV02_static_dr-startpose/<run>      # learning + throughput table
tensorboard --logdir experiments/navigation/runs                                             # all navigation runs

# sweep (e.g. seeds) -> one run folder per member
python scripts/sweep.py --config experiments/navigation/configs/NAV02_static_dr-startpose.yaml --gpus 5

# evaluate with a named protocol (seeds, goal, timeout defined once in st_nav/evaluation/protocols.py)
bin/isaacrun -g 5 -- python scripts/evaluate.py --headless \
    --checkpoint experiments/navigation/runs/NAV02_.../<run>/model_200.pt --protocol ref-static-goalC10
#   -> experiments/navigation/eval/NAV02_.../ref-static-goalC10_corridor_static_random/<run>_model_200/
#        episodes.csv, summary.json, run_config.json, recording.npz, eval_trajectories.png
#      and a row in .../ref-static-goalC10_corridor_static_random/summary.csv (all policies, same seeds)

python tools/make_video.py experiments/navigation/eval/NAV02_.../<scenario>/<policy> --episodes 0 1 2

# promote a navigation policy for sensing work
python tools/experiments.py promote NAV02_static_dr-startpose <run>/model_200.pt   # -> models/navigation/NAV02_model_200.{pt,json}
```

Example experiment config (overrides on the configclass defaults; the resolved config is dumped to `params/`):

```yaml
exp_id: NAV01_static_dr-startpose
task: STNav-TB4-Corridor-Static-v0          # Stage C: STNav-TB4-Nav2D-v0 + scene below
scene: corridor_static_random
env:                                         # CorridorNavEnvCfg fields (unknown keys -> error)
  scene.num_envs: 4096
  start_pose_dr: true
agent:                                       # CorridorNavPPORunnerCfg fields
  max_iterations: 1500
  seed: 42
sweep: {}                                    # e.g. {agent.seed: [42, 43, 44]}
```

A future spatiotemporal experiment uses the same scripts:
`experiments/spatiotemporal/configs/STS01_static_dr-lidar.yaml` with `task: STNav-TB4-SectorSensing-v0` and
`nav_policy: models/navigation/NAV01_model_200.pt`. Its outputs go under `experiments/spatiotemporal/`, and
sensing metrics are added through the evaluation library.

---

## 6. File mapping (current → proposed)

Paths relative to the repo root. "Edit" means only paths, imports or default output locations change.

| Current | Proposed | Stage | Edit needed |
|---|---|---|---|
| `README.md` (phase-1 launch doc) | `docs/server_setup.md`; new `README.md` (layout + commands) | A | rewrite |
| `env.sh`, `bin/isaacrun`, `bin/check_physx_gpu` | unchanged | – | – |
| `logs/launcher/` | unchanged (ignored) | – | – |
| `assets/turtlebot4_standard/` | unchanged | – | `clean_urdf.py`, `convert_urdf.py`, `README.md`: `third_party` → `references/robot_description` |
| `third_party/{turtlebot4,create3_sim,xacro_shim}` | `references/robot_description/` | A | (above); clones ignored, commits pinned in `docs/references.md` |
| `source/st_nav/st_nav/{assets,control,geometry,sensors,config.py}` | unchanged | – | – |
| `source/st_nav/st_nav/scenes/*` | unchanged; registry added to `scenes/__init__.py` | C | – |
| `source/st_nav/st_nav/envs/corridor_scene.py` | unchanged path; scene/sensor injected via config | C | code (validated, §7.3) |
| `source/st_nav/st_nav/tasks/nav2d/` | `source/st_nav/st_nav/tasks/navigation/` | A | imports in 8 scripts + registration (uses `__name__`, so automatic) |
| `source/st_nav/st_nav/calibration/` | unchanged (production constants used by the env) | – | – |
| – | `source/st_nav/st_nav/experiments.py`, `evaluation/` | B | new (evaluation logic moved from `m5_evaluate.py`) |
| `scripts/m5/train.py` | `scripts/train.py` | A→B | A: default `--log_root`; B: `--config` |
| `scripts/m5/m5_evaluate.py` | `scripts/evaluate.py` (+ `st_nav/evaluation/`) | A→B | default output dir |
| `scripts/m5/m5_monitor.py` | `tools/monitor_run.py` | A | – |
| `scripts/m5/m5_plot_eval.py`, `m5_animate_eval.py` | `tools/plot_eval.py`, `tools/make_video.py` | A | – |
| `scripts/m5/m5_verify_start_dr.py` | `tests/sim/test_start_pose_dr.py` | A | output path |
| `scripts/m4/m4_validation.py` (31 checks) | `tests/sim/test_nav_env.py` | A | imports, output path |
| `scripts/m3/m3_smoke.py`, `m3_validation.py` | `tests/sim/test_scene_smoke.py`, `tests/sim/test_corridor_scene.py` | A | output path |
| `scripts/m2/m2_lidar_validation.py` | `tests/sim/test_lidar.py` | A | output path |
| `scripts/m4/m4_benchmark.py` | `tools/benchmark_env.py` | A | output path |
| `scripts/m1/*` (calibration, 7 diag scripts, plots, validation) | `infrastructure/milestones/M1_robot_calibration/scripts/` | A | output paths |
| `scripts/m2/{m2_benchmark,m2_bench_check,m2_rate_check,m2_plots}.py` | `infrastructure/milestones/M2_lidar/scripts/` | A | output paths |
| `scripts/m3/{m3_benchmark,m3_plots}.py` | `infrastructure/milestones/M3_corridor_scene/scripts/` | A | output paths |
| `scripts/m4/{m4_diagnostics,m4_diag_velocity,m4_plot_diagnostics,m4_plot_start_goal}.py` | `infrastructure/milestones/M4_rl_environment/scripts/` | A | output paths |
| `scripts/viz/*` | `infrastructure/robot_viz/scripts/` | A | output paths |
| `results/m1`…`results/m4` (incl. `m2/contaminated/`, `m4/diag/`) | `infrastructure/milestones/M<k>_*/results/`; `M<k>_REPORT.md` → `infrastructure/milestones/M<k>_*/REPORT.md` | A | report links |
| `results/viz/` | `infrastructure/robot_viz/results/` | A | – |
| `results/m5/runs/2026-11-22_00-44-09_m5_initial_4096/` | `experiments/navigation/runs/NAV01_static_dr-startpose/2026-11-22_00-44-09_m5_initial_4096/` (folder name kept) | A | – |
| `…/eval_model_200/` | `experiments/navigation/eval/NAV01_static_dr-startpose/ref-static-goalC10_corridor_static_random/2026-11-22_00-44-09_m5_initial_4096_model_200/` | A | `summary.json` keeps the old checkpoint path as provenance |
| `results/m5/m5_initial_4096.stdout.log` | same run folder, `train.log` | A | – |
| `results/m5/runs/2026-11-22_00-40-45_smoke/`, `results/m5/start_dr_check.json` | `infrastructure/milestones/M5_initial_navigation/results/` | A | – |
| `model_200.pt` (copy) | `models/navigation/NAV01_model_200.pt` + `.json` | B | new |
| `tests/test_*.py` (4) | `tests/unit/` | A | – |
| `validation/*.py` | `infrastructure/server/` | A | – |
| `~/isaac_lab/phase1_diagnostics/PHASE1_INSTALLATION_REPORT.md` | copy → `infrastructure/server/` | A | (original left in place) |
| `phase2_analysis/`, `phase3_planning/` | `docs/history/` | A | – |
| `~/isaac_lab/Spatiotemporal_plan.txt` | copy → `docs/research_plan.md` | A | (original left in place) |
| `~/adaptive_sensor_policy`, `~/isaac_lab/nav_reference/RL-Navigation`, `~/Sim2Real/...` | stay where they are; listed in `docs/references.md` (path, commit, role) | A | – |
| `REORGANIZATION_PROPOSAL.md` (this file) | `docs/history/` after the migration | A | – |

Nothing is deleted. The only new copies are the phase-1 report, the research plan and the promoted model.

---

## 7. Migration and regression-validation plan

### 7.1 Stage 0: safety checkpoint and baseline (before anything moves)

1. **Freeze:** no running jobs and no other sessions writing to the folder. Check with `pgrep -f st_nav` and
   `nvidia-smi`.
2. **Full backup:** `tar czf ~/isaac_lab/backups/spatiotemporal_sensing_pre_reorg_<date>.tar.gz spatiotemporal_sensing`.
   This includes everything git will ignore (checkpoints, NPZ, TB events, clones), about 200 MB.
3. **Git:**
   * `git init`.
   * Extend `.gitignore` per §4.
   * Exclude `third_party/turtlebot4` and `third_party/create3_sim` (nested clones) and record their commits
     (`7fd29fb`, `6c30584`).
   * Commit everything else as **`pre-reorg`** and add a tag.
   * Every following stage goes on a branch `reorg`, one commit per stage. Each commit is reviewable with
     `git diff -M --stat`, so renames show up as renames.
4. **Record the baseline** (`tests/golden/`, about 25 min of GPU time on GPU 5):

| ID | Check | What is recorded |
|---|---|---|
| G1 | unit tests | 15 pass, 1 skipped |
| G2 | M4 environment validation (`m4_validation.py`, 1024 envs) | 31/31 + full JSON |
| G3 | start-pose DR check | JSON |
| G4 | **resolved configs** | `dump_yaml` of `CorridorNavEnvCfg()` and `CorridorNavPPORunnerCfg()` defaults, and of the M5 run's `params/` |
| G5 | **reset observations** | env with fixed seed and eval overrides: obs at reset for 100 envs (47-dim), the start poses, the size bank and offsets |
| G6 | **policy function** | load `model_200.pt`, feed a fixed observation batch (G5 + 10k recorded eval observations), store the deterministic actions (CPU, float64 comparison) |
| G7 | **closed-loop evaluation, 2 repeats** | the 100-episode reference protocol twice: per-episode outcome, steps, return; the two repeats give the GPU non-determinism floor (already seen: identical outcomes, return differs in the 5th digit) |
| G8 | **short training** | 30 iterations, 4,096 envs, seed 42: curves of return / goal rate / episode length / fps |

### 7.2 After Stage A (pure move) and Stage B (tooling)

| Check | Pass criterion |
|---|---|
| Import sweep: import every module of `st_nav` and every script with `--help` (outside Kit where possible); `grep` for old paths (`results/m`, `scripts/m`, `tasks.nav2d`, `third_party`) | no errors; no stale references outside `docs/history` and the archived reports |
| G1 unit tests | identical |
| G4 resolved configs | **identical** YAML except path fields (`log_dir`) |
| G5 reset observations | max abs difference ≤ 1e-6 (same seeds, same bank) |
| G6 policy actions on the fixed batch | **bit-identical** |
| G2 env validation | 31/31, numbers within the G2 tolerances |
| G3 start-pose DR | identical counts and ranges |
| G7 evaluation, from the moved checkpoint path | same 98/100 and the same failing episodes (25, 73); per-episode steps equal; mean return within the G7 run-to-run spread |
| G8 short training (30 iterations) | curves inside the band of the baseline run; fps within ±10 % |
| Old artifacts | TensorBoard opens the moved run folder; `tools/make_video.py` renders from the moved eval folder |
| Stage B only | `train.py --config` from the NAV01 YAML resolves to the same `params/` as G4; a new eval lands in the new layout with the same `episodes.csv` content as G7 |

### 7.3 After Stage C (decoupling)

* Every check in §7.2.
* Scene registry: `corridor_static_random` selected by name produces the same G5 observations.
* The old task ID and the new generic ID with the same scene give identical G4 configs and G5 observations.
* If the sampler is vendored: the size bank for seeds 0–1023 and 3000–3099 is equal to the reference bank
  element for element.
* A 200-iteration training run: final goal rate ≥ 0.98 and episode length within ±5 % of the M5 run. This is the
  only check that needs real training, about 20 min.

### 7.4 Rollback

Each stage is one commit on `reorg`. If a check fails, `git reset --hard <previous stage>`, or restore from the
tarball, then investigate. `main` moves to `reorg` only after you approve the stage results.

---

## 8. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Hardcoded `$ST_ROOT/results/m<N>` output paths in ~25 scripts | historical scripts write to old locations, or recreate `results/` | update defaults in Stage A; grep check; archived scripts get an `infra_results(<milestone>)` helper |
| Asset path resolved relative to `assets/turtlebot4.py` and `$ST_ROOT` | robot fails to load if the file or the asset folder moves | neither moves in this plan; G2/G7 would catch it |
| **Run-time dependency on `~/adaptive_sensor_policy`** (size bank, unit tests); that tree has uncommitted changes and can switch branches | silent change of the training and evaluation distribution | record its commit and dirty state in every run's `params/`; Stage C: vendor the sampler with a bit-identical bank test |
| GPU physics is not bit-exact between runs | cannot require identical trajectories | compare exactly where it is possible (configs, reset observations, policy outputs); use tolerances measured from repeated baseline runs for closed loop |
| Nested git clones in `third_party/` | `git add` would record them as broken submodules | exclude before `git init` (§7.1) |
| Large binaries in git (USD 22 MB, GLB 2 × 13 MB, NPZ) | slow repository | track USD and reports; ignore GLB/NPZ/MP4 outside eval; tarball keeps everything |
| Report links and paths inside reports and `summary.json` | stale references in historical documents | keep old paths as provenance; add a path-change note to `infrastructure/README.md`; fix links in the moved reports |
| Renaming `tasks.nav2d` → `tasks.navigation` | old imports fail | updated in the same commit; the registered task ID is unchanged; checkpoints contain only state dicts (`actor_state_dict`, `critic_state_dict`, `optimizer_state_dict`, `iter`, `infos`), so they don't depend on module paths |
| Stale `__pycache__` | phantom imports of old module paths | clear caches before the import sweep |
| Another session or job writing during the move | lost or mixed outputs | freeze step in §7.1 |
| Scope creep (behaviour changes slipping into the refactor) | invalidates the validated pipeline | Stage A/B forbid logic changes; G4/G5/G6 detect any change in config, environment or policy |

---

## 9. Effort estimate

| Stage | Work | Time (incl. validation) |
|---|---|---|
| 0 | backup, git init, ignore policy, baseline commit and tag, golden baselines G1–G8 | 1–1.5 h (≈ 25 min GPU) |
| A | moves, import and path updates, README/docs restructure, infrastructure index | 2–3 h (+ 30 min validation) |
| B | `experiments.py` + CLI, YAML loader for `train.py`, `sweep.py`, evaluation library + protocols, cards/registry, NAV01 back-fill, `models/navigation` | 3–4 h (+ 30 min validation) |
| C | scene registry, scene/sensor injection, generic task ID + alias, optional vendored sampler | 3–5 h (+ 45 min validation incl. 200-iteration training) |
| **Total** | | **≈ 1.5 working days**; A+B alone ≈ 1 day |

Stage C could be done as the first step of M6, since that is when a second scene first exists.

## 10. Decisions needed from you

1. **Structure:** C (recommended), A or B?
2. **Directory names:** `infrastructure/` (or `infra/`, `dev/`, `archive/`), `experiments/` (or `research/`),
   `references/`.
3. **Experiment ID prefixes:** `NAV`/`TMP`/`SPA`/`STS`, or the reference's `ST` numbering continued?
4. **Promotion:** should the M5 run become **NAV01** with `model_200` promoted to `models/navigation/`?
5. **Git:** is local-only git acceptable, or should it be pushed to a remote? (That affects the binary policy.)
6. **Stage C timing:** now, or at the start of M6?
7. **External references:** may I vendor the reference sampler (Stage C), or should the run-time import stay?
