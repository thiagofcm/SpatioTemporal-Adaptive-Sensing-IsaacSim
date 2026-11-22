# NAV01_static_initial

**Status:** done

**Question:** Can PPO learn collision-free navigation in the randomized static corridor (TurtleBot4, 40-beam 2D
LiDAR with noise, start-pose DR) with 4,096 parallel environments?

**Hypothesis:** Yes, within the planned 1,500 iterations. The planned check was success on the reference evaluation
protocol with held-out obstacle sizes.

## Setup

| | |
|---|---|
| Created | 2026-11-22 (milestone M5; card and config back-filled during the repository reorganization) |
| Config | [Navigation/configs/NAV01_static_initial.yaml](../../configs/NAV01_static_initial.yaml) (reproduces the original run's settings) |
| Training runs | `Navigation/runs/NAV01_static_initial/` |
| Evaluations | `Navigation/eval/NAV01_static_initial/ref_static_c10/<policy>/` |
| Differs from previous experiment | first experiment |
| Reference equivalent | scene and DR of the reference's corridor_static_object_random with start-pose DR. The reference trains TD3 navigation, not PPO |
| Promoted model | `models/navigation/NAV01_model_200.pt` (+ `.json`) |

## Commands

```bash
bin/isaacrun -g 5 -- python Navigation/scripts/train.py --headless --config Navigation/configs/NAV01_static_initial.yaml
bin/isaacrun -g 5 -- python Navigation/scripts/evaluate.py --headless \
    --checkpoint Navigation/runs/NAV01_static_initial/<run>/model_200.pt          # protocol ref_static_c10
```

## Training runs

| Run folder | Iterations | Seed | Throughput | Notes |
|---|---|---|---|---|
| `2026-11-22_00-44-09_m5_initial_4096` | 201 (stopped by hand; planned 1,500) | 42 | ≈ 21k env steps/s, 6.2 s/iteration (GPU 5) | original M5 run (`scripts/m5/train.py`, before configs existed); converged by iteration ≈ 100 |
| `2026-11-22_12-43-08_ppo_n4096_s42_reorg_validation` | 201 (`--set agent.max_iterations=201`) | 42 | 20.2k env steps/s, 6.5 s/iteration (GPU 5) | reproduction through the YAML interface after the repository reorganization; same plateau (return 87.3, goal 99.6 %); see `docs/history/REORGANIZATION_REPORT.md` |

## Evaluations

| Protocol | Policy | Episodes | Success | Notes |
|---|---|---|---|---|
| `ref_static_c10` (70 s, per-episode seeds) | `2026-11-22_00-44-09_m5_initial_4096_model_200` | 100 | **98 %** | 2 safety violations (episodes 25, 73: the robot clips Z1 at its rightmost offset); 0 collisions or timeouts; 29.0 s mean; min clearance mean 0.090 m |
| `ref_static_c10` (70 s, per-episode seeds) | `2026-11-22_12-43-08_ppo_n4096_s42_reorg_validation_model_200` | 100 | **100 %** | 29.2 s mean; min clearance mean 0.117 m, min 0.065 m |

Evaluations re-run on 2026-11-22 with the 70 s timeout (= training) and per-episode reproducible obstacle / LiDAR-noise randomness; earlier 80 s results were replaced.

## Results

* **Training:** the return rose from −29.6 to 87.7 by iteration 100 and stayed flat (goal rate 99.5–99.9 % on the
  training distribution).
* **Evaluation:** 98 % on the reference protocol, with held-out obstacle sizes and a goal outside the training list.
* **Driving style:** the policy drives near v_max (0.217 m/s mean) and passes close to obstacles (mean minimum
  clearance 0.090 m against the 0.05 m safety margin).
* **Possible memorization:** trajectories are very similar across layouts, which suggests a largely memorized route.
  Figures: `ref_static_c10/<policy>/eval_overview.png`, `episode_0025/episode_0025.png`, `episode_0073/episode_0073.png`.

## Conclusion and next step

The training and evaluation pipeline works. Next: dynamic obstacles (M6) to test reactive behaviour, and possibly a
larger clearance margin.
