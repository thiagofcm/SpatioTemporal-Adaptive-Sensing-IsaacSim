# NAV02_chase-ob1_dr

**Status:** evaluating
<!-- planned | training | evaluating | done | superseded by <ID>. Keep the registry row in docs/EXPERIMENTS.md in sync. -->

**Question:** Can PPO learn collision-free navigation in the dynamic corridor with one moving obstacle (OB1), using the NAV01
observation, reward and PPO settings?

**Hypothesis:** TODO (to be filled before training).

## Setup

| | |
|---|---|
| Created | 2026-11-22 (milestone M6) |
| Git commit at creation | `07e78a6` (+ uncommitted M6 changes) |
| Config | [Navigation/configs/NAV02_chase-ob1_dr.yaml](../../configs/NAV02_chase-ob1_dr.yaml) |
| Training runs | `Navigation/runs/NAV02_chase-ob1_dr/` |
| Evaluations | `Navigation/eval/NAV02_chase-ob1_dr/ref_chase_c10/<policy>/` |
| Differs from previous experiment | NAV01_static_initial: dynamic corridor (no Z1-Z4) with OB1 moving (random_ping_pong, speed and path-y DR), reference ST04 goal list |
| Reference equivalent | ST04_chase-ob1_dr (~/adaptive_sensor_policy; reference trains a sensing policy on a frozen TD3, here the navigation policy itself is trained) |

## Commands

```bash
bin/isaacrun -g <gpu> -- python Navigation/scripts/train.py --headless --config Navigation/configs/NAV02_chase-ob1_dr.yaml
bin/isaacrun -g <gpu> -- python Navigation/scripts/evaluate.py --headless --checkpoint Navigation/runs/NAV02_chase-ob1_dr/<run>/model_<k>.pt   # protocol ref_chase_c10
```

## Training runs

| Run folder | Iterations | Seed | Throughput | Notes |
|---|---|---|---|---|
| `2026-11-22_15-52-17_ppo_n4096_s42` | 571 (stopped by hand after model_500; planned 1,500) | 42 | ≈ 26k env steps/s, ≈ 4.9 s/iteration (GPU 4) | training goal rate 99 % from iteration ≈ 100; policy std rising after ≈ 300 (0.30 → 0.82) |

## Evaluations

| Protocol | Policy | Episodes | Success | Notes |
|---|---|---|---|---|
| `ref_chase_c10` (70 s, per-episode seeds) | `2026-11-22_15-52-17_ppo_n4096_s42_model_500` | 100 | **100 %** | 32.1 s mean; min clearance mean 0.125 m, min 0.064 m |

## Results

## Conclusion and next step
