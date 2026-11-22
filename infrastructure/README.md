# Infrastructure: development and validation history (M1–M5)

The work that built and validated the simulation, kept read-only next to the active research lines. Each milestone
folder holds its `scripts/` (still runnable; default outputs go to the milestone's own `results/`) and its
`results/` (report, data, figures). The reports were written before the reorganization, so the paths they mention
are the old ones (table below).

| Milestone | Question | Report | Key results |
|---|---|---|---|
| [M1_robot_calibration](M1_robot_calibration/) | Does the articulated TB4 in PhysX reproduce the real robot's motion? | [M1_REPORT.md](M1_robot_calibration/results/M1_REPORT.md) | calibrated command model (delay 0.14 s, tau, gains, 0.305 m/s wheel saturation); 100 Hz physics, solver 16/2 adopted |
| [M2_lidar](M2_lidar/) | Accurate and fast enough 2D LiDAR on the GPU ray caster? | [M2_REPORT.md](M2_lidar/results/M2_REPORT.md) | walls max error 0.66 mm; 16,384 envs at 100 Hz: 66k control steps/s full loop |
| [M3_corridor_scene](M3_corridor_scene/) | Randomized static corridor with the reference distributions? | [M3_REPORT.md](M3_corridor_scene/results/M3_REPORT.md) | 20,480 layouts, 0 constraint violations, KS-matched to the reference sampler |
| [M4_rl_environment](M4_rl_environment/) | RL environment (47-dim obs, rewards, terminations, resets) correct? | [M4_REPORT.md](M4_rl_environment/results/M4_REPORT.md) | 31/31 checks; 55.6k steps/s at 16,384 envs; follow-up: base-link velocity obs, goal-distance norm 7 m, timeout bootstrapping verified |
| [M5_initial_navigation](M5_initial_navigation/) | First PPO policy | continued as experiment **NAV01** ([card](../Navigation/eval/NAV01_static_initial/README.md)) | 98/100 on the reference protocol at iteration 200 |
| [robot_viz](robot_viz/) | Robot geometry renders / motion video | – | `.glb` exports are regenerable and git-ignored |
| [server](server/) | GPU server diagnostics (PhysX on GPU, ray caster, Vulkan) | [PHASE1_INSTALLATION_REPORT.md](server/PHASE1_INSTALLATION_REPORT.md) | rendering unavailable (`/dev/dri`); headless GPU physics works with the CUDA workaround |

## Path changes (2026-11-22 reorganization)

| Old path (in reports) | New path |
|---|---|
| `scripts/m<k>/*`, `results/m<k>/*` | `infrastructure/M<k>_*/scripts/*`, `infrastructure/M<k>_*/results/*` |
| `scripts/m4/m4_validation.py` | `tests/sim/nav_env_validation.py` (maintained regression; outputs to `tests/sim/results/`) |
| `scripts/m3/m3_smoke.py`, `scripts/m5/m5_verify_start_dr.py` | `tests/sim/scene_smoke.py`, `tests/sim/start_pose_dr_check.py` |
| `scripts/m4/m4_benchmark.py` | `tools/benchmark_env.py` |
| `scripts/m5/train.py`, `scripts/m5/m5_evaluate.py` | `Navigation/scripts/train.py` (now `--config`), `Navigation/scripts/evaluate.py` |
| `scripts/m5/m5_{monitor,plot_eval,animate_eval}.py` | `tools/{monitor_run,plot_eval,make_video}.py` |
| `results/m5/runs/2026-11-22_00-44-09_m5_initial_4096/` | `Navigation/runs/NAV01_static_initial/2026-11-22_00-44-09_m5_initial_4096/` (stdout log → `train.log`) |
| `results/m5/runs/.../eval_model_200/` | `Navigation/eval/NAV01_static_initial/ref_static_c10/2026-11-22_00-44-09_m5_initial_4096_model_200/` |
| `results/m5/start_dr_check.json`, smoke run | `infrastructure/M5_initial_navigation/results/` |
| `scripts/viz`, `results/viz` | `infrastructure/robot_viz/{scripts,results}` |
| `validation/` | `infrastructure/server/` |
| `phase2_analysis/`, `phase3_planning/` | `docs/history/` |
| `third_party/` | `references/robot_description/` |
