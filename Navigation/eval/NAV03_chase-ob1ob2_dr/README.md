# NAV03_chase-ob1ob2_dr

**Status:** evaluating
<!-- planned | training | evaluating | done | superseded by <ID>. Keep the registry row in docs/EXPERIMENTS.md in sync. -->

**Question:** Can PPO learn collision-free navigation in the dynamic corridor with two moving obstacles (OB1 and OB2), using the NAV01
observation, reward and PPO settings?

**Hypothesis:** TODO (to be filled before training).

## Setup

| | |
|---|---|
| Created | 2026-11-22 (milestone M6) |
| Git commit at creation | `07e78a6` (+ uncommitted M6 changes) |
| Config | [Navigation/configs/NAV03_chase-ob1ob2_dr.yaml](../../configs/NAV03_chase-ob1ob2_dr.yaml) |
| Training runs | `Navigation/runs/NAV03_chase-ob1ob2_dr/` |
| Evaluations | `Navigation/eval/NAV03_chase-ob1ob2_dr/ref_chase_c10/<policy>/` |
| Differs from previous experiment | NAV02_chase-ob1_dr: OB2 added (same speed / path-y DR, y-gap constraint between OB1 and OB2) |
| Reference equivalent | ST05_chase-ob1ob2_dr (~/adaptive_sensor_policy; reference trains a sensing policy on a frozen TD3, here the navigation policy itself is trained) |

## Commands

```bash
bin/isaacrun -g <gpu> -- python Navigation/scripts/train.py --headless --config Navigation/configs/NAV03_chase-ob1ob2_dr.yaml
bin/isaacrun -g <gpu> -- python Navigation/scripts/evaluate.py --headless --checkpoint Navigation/runs/NAV03_chase-ob1ob2_dr/<run>/model_<k>.pt   # protocol ref_chase_c10
```

## Training runs

| Run folder | Iterations | Seed | Throughput | Notes |
|---|---|---|---|---|
| `2026-11-22_15-52-39_ppo_n4096_s42` | 526 (stopped by hand after model_500; planned 1,500) | 42 | ≈ 24–26k env steps/s, ≈ 5.2 s/iteration (GPU 5) | training goal rate 91–95 % from iteration ≈ 100; policy std still rising (1.3 at 500) |

## Evaluations

| Protocol | Policy | Episodes | Success | Notes |
|---|---|---|---|---|
| `ref_chase_c10` (70 s, per-episode seeds) | `2026-11-22_15-52-39_ppo_n4096_s42_model_500` | 100 | **97 %** | 3 safety violations (episodes 4, 42, 57: between OB1 and OB2); 34.6 s mean; min clearance mean 0.108 m, min 0.033 m |

## Results

## Conclusion and next step
