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
| `scenes/` | one file per scene: geometry, DR ranges, samplers (`corridor_static_random.py`, incl. the Z1–Z4 sampler ported from the reference); spawners | **M6:** `corridor_dynamic_*.py` + a small scene registry |
| `envs/corridor_scene.py` | scene runtime: builds the vectorized scene, GPU resets (layout, start, goal, robot, actuator), sensor binding, ground-truth queries | **M6:** moving obstacles; scene and sensor chosen by the env config |
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
  * add `scenes/<name>.py` with geometry, DR ranges and samplers;
  * extend the scene runtime if it needs new body types (moving obstacles);
  * select it from the env config (field added in M6), then from the experiment YAML (`env.<field>`).
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
