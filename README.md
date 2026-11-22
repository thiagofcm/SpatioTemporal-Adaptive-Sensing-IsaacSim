# Spatiotemporal adaptive sensing: Isaac Lab workspace

A TurtleBot4 navigates a randomized corridor in Isaac Lab 2.3.2 / Isaac Sim 5.1, using a 40-beam 2D LiDAR. The
navigation policy is trained with PPO (RSL-RL) on thousands of GPU-parallel environments. Two research lines build
on it:

* **Navigation:** navigation policies for different scenes and sensors.
* **AdaptiveSensing:** temporal, spatial and spatiotemporal sensing policies. These decide when and where the
  LiDAR (later the programmable Mirrorcle 3D LiDAR) acquires data, driving a frozen navigation policy.

## Start here

| Document | What it covers |
|---|---|
| [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) | **Experiment registry and conventions:** IDs, configs, runs, evaluations |
| [docs/architecture.md](docs/architecture.md) | The shared simulation library and where new scenes, sensors and tasks go |
| [docs/research_plan.md](docs/research_plan.md) | Research direction and roadmap |
| [infrastructure/README.md](infrastructure/README.md) | M1–M5 development and validation history (robot calibration, LiDAR, scene, RL env) |
| [docs/server_setup.md](docs/server_setup.md) | GPU server launch tooling: CUDA workaround, `isaacrun`, GPU selection |
| [docs/references.md](docs/references.md) | External repositories used as references (paths, commits) |

## Repository layout

```
source/st_nav/st_nav/   shared simulation library (Isaac Lab-native): robot assets, actuator model, sensors, geometry,
                        scenes + domain randomization, scene runtime, Gym tasks (tasks/nav2d)
Navigation/             research line: scripts/ (train.py, evaluate.py), configs/<ID>.yaml, runs/<ID>/, eval/<ID>/
AdaptiveSensing/        research line (created with its first experiment): same layout, split by temporal / spatial /
                        spatiotemporal
models/navigation/      frozen navigation policies for AdaptiveSensing (<ID>_<checkpoint>.pt + .json)
tools/                  experiments.py (IDs, paths, YAML configs), monitor_run.py, episode_data.py (evaluation recording
                        format), make_video.py, plot_episode.py, plot_eval.py (standalone: numpy + matplotlib),
                        benchmark_env.py
tests/                  unit/ (no simulator, pytest) · sim/ (simulator regression scripts) · golden/ (refactoring checks)
infrastructure/         M1–M5 development history (reports, scripts, results) + server diagnostics
docs/                   registry, architecture, plan, server setup, references, templates, history/
references/             robot description sources (TurtleBot4, Create3) used to build the robot USD
assets/                 TurtleBot4 USD and its conversion pipeline
env.sh, bin/            environment + launcher (isaacrun: GPU selection, CUDA workaround, logging, timeout)
```

## Common commands

Run from the repository root. `isaacrun` sets up the environment, so no conda activation is needed.

```bash
python tools/experiments.py list                                          # experiments and their runs

# train (defaults from the configclasses, overridden by the YAML, then by --set)
bin/isaacrun -g <gpu> -- python Navigation/scripts/train.py --headless --config Navigation/configs/NAV01_static_initial.yaml \
    [--set agent.seed=43] [--run_tag <tag>]
python tools/monitor_run.py Navigation/runs/<ID>/<run>                     # learning curve + throughput table
tensorboard --logdir Navigation/runs --port 6006

# evaluate (writes Navigation/eval/<ID>/<protocol>/<run>_<checkpoint>/)
bin/isaacrun -g <gpu> -- python Navigation/scripts/evaluate.py --headless --checkpoint Navigation/runs/<ID>/<run>/model_200.pt
python tools/plot_eval.py <eval dir>                                       # whole-evaluation figure
python tools/plot_episode.py <eval dir>/episode_0025                       # one episode: map + time series
python tools/make_video.py <eval dir>/episode_0025                         # one episode MP4
python tools/make_video.py --eval <eval dir> --episodes 0 1 2              # several episodes side by side
```

## Tests

```bash
source env.sh && python -m pytest -q tests                                 # unit tests (no simulator, ~2 s)
bin/isaacrun -g <gpu> -- python tests/sim/nav_env_validation.py --headless --num_envs 1024   # 31 env checks
bin/isaacrun -g <gpu> -- python tests/sim/start_pose_dr_check.py --headless                  # start-pose DR
bin/isaacrun -g <gpu> -- python tests/sim/scene_smoke.py --headless                          # scene generation
```

`tests/golden/` holds the before/after snapshots of the repository reorganization. See
[docs/history/REORGANIZATION_REPORT.md](docs/history/REORGANIZATION_REPORT.md).
