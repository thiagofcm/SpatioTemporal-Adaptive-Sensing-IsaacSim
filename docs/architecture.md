# Architecture

## Shared simulation library: `source/st_nav/st_nav/`

Isaac Lab-native package, on `PYTHONPATH` via `env.sh`. Both research lines use it. It must contain no
research-line-specific code (training loops, experiment paths).

| Module | Content | Grows with |
|---|---|---|
| `assets/turtlebot4.py` | `TB4_CFG` articulation, geometry, `TB4_COMMAND_MODEL` (calibrated) | other robots |
| `control/diff_drive.py` | command model: delay, lag, gains, wheel saturation (GPU-batched) | actuator variants |
| `sensors/` | `lidar2d.py` (MultiMeshRayCaster), `noise.py`, `patterns.py`, `base.py` (`SensorOutput`) | **M7:** `mirrorcle3d.py` (same `SensorOutput`), programmable scan patterns, acquisition timing |
| `geometry/analytic2d.py` | analytic clearance / ray oracle (ground truth for rewards, terminations, validation) | – |
| `scenes/` | registry (`__init__.py`, `get_scene`); one file per scene (`corridor_static_random.py`: Z1–Z4 baffles and their sampler; `corridor_dynamic_chase_random.py`: moving OB1 / OB2, obstacle DR ranges and path-y sampler); `_common.py` (arena, corridor walls, start poses, goals); `_obstacle_motion.py` (GPU-vectorized `random_ping_pong` motion); spawners | more scenes (e.g. OB3 `triggered_chase`) |
| `envs/corridor_scene.py` | scene runtime: the scene chosen by `scene_name`; builds the vectorized scene (walls, baffles, kinematic obstacle cylinders tracked by the ray caster), GPU resets (layout, obstacles, start, goal, robot, actuator), obstacle motion every physics step, ground-truth clearance (walls + active obstacles) | sensor chosen by the env config (M7) |
| `tasks/nav2d/` | navigation DirectRLEnv (`STNav-TB4-Corridor-Static-v0`), its config and the RSL-RL agent config | new navigation tasks/variants |
| `tasks/sensing/` (future) | adaptive-sensing envs wrapping navigation with a frozen policy | AdaptiveSensing |
| `calibration/` | simulation constants used by the env (`sim_harness.py`) and the M1 calibration tools | – |

## Research lines

Each research line owns its scripts, configs, runs and evaluations (`Navigation/`, `AdaptiveSensing/`).
* Shared across lines: the library above, `tools/experiments.py` (IDs, paths, YAML overrides), the visualization
  tools, `models/`.
* Not shared: training and evaluation entry points. AdaptiveSensing gets its own once its requirements are known.
  Code moves into the library or `tools/` only when a second user needs it.

## Adding things

* **A scene:**
  * add `scenes/<name>.py` with `NAME`, `WALLS`, `HAS_BAFFLES`, `MOVING_OBSTACLES` (+ its DR ranges / samplers) and
    register it in `scenes/__init__.py`;
  * extend the scene runtime only for new body or motion types;
  * select it with `env.scene_name` in the experiment YAML (task `STNav-TB4-Corridor-v0`).
* **A sensor:**
  * add `sensors/<name>.py` returning `SensorOutput`;
  * add a config field selecting it (M7);
  * the observation size follows the sensor, and the model `.json` records the sensor used.
* **A navigation experiment:** a new `Navigation/configs/<ID>.yaml` (see `docs/EXPERIMENTS.md`). No code change if
  the setting is already a config field.
* **An adaptive-sensing task:** `tasks/sensing/<task>.py` plus a Gym registration. It loads a frozen policy from
  `models/navigation/` with RSL-RL's actor and the `.json` spec.

Performance-critical paths are the vectorized resets, the ray caster, the command model and RSL-RL rollouts. Keep
them batched on the GPU, and benchmark changes with `tools/benchmark_env.py`.
