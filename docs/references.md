# External references

Read-only. Nothing in this repository writes to them, and training, evaluation and scene generation need none of them (checked with `tests/sim/no_reference_guard.py`).

| Name | Location | Commit | Role |
|---|---|---|---|
| Reference simplified environment | `~/adaptive_sensor_policy` (branch `spatiotemporal`) | `d58be3fadb` (2026-11-16) | Source of the corridor geometry, DR ranges, start poses, goals, Z1–Z4 sampler, evaluation protocol and experiment conventions, all **ported** into `st_nav` (`scenes/corridor_static_random.py`) and `Navigation/scripts/evaluate.py`. **Not used at run time.** Optional cross-checks only: `tests/unit/test_corridor_static_random.py`, `tests/unit/test_analytic2d.py` (skipped when absent), the M2/M3 validation scripts via `tools/reference.py`, and `tests/golden/make_reference_bank.py` (golden sampler output, `tests/golden/reference_bank.npz`) |
| RL-Navigation (original OmniIsaacGymEnvs work) | `~/isaac_lab/nav_reference/RL-Navigation` | `01fc8f8` | Phase 2 migration analysis (`docs/history/phase2_analysis`) |
| Real-robot motion recordings | `~/Sim2Real/Adaptive-Frame-Rate-Control-Sim2Real-TurtleBot/RealRobot/motion_dynamic_charac/runs` | – | M1 actuator calibration (`st_nav/calibration/real_data.py`) |
| TurtleBot4 description | `references/robot_description/turtlebot4` (git-ignored clone of github.com/turtlebot/turtlebot4) | `7fd29fb` | URDF/meshes for the robot USD (`assets/turtlebot4_standard/README.md`) |
| iRobot Create3 description | `references/robot_description/create3_sim` (git-ignored clone of github.com/iRobotEducation/create3_sim) | `6c30584` | Create3 base meshes for the robot USD |
| xacro shim + venv | `references/robot_description/xacro_shim` (tracked), `.venv_xacro` (git-ignored) | – | xacro → URDF without ROS (asset pipeline only) |
| Isaac Lab installation | `~/isaac_lab/IsaacLab_Fork` (2.3.2), conda env `env_isaaclab` | – | Not modified |

To rebuild the robot-description clones: `git clone <url> references/robot_description/<name> && git -C ... checkout <commit>`.
