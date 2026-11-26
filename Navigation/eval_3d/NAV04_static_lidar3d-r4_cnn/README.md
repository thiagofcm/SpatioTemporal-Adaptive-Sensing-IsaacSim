# NAV04_static_lidar3d-r4_cnn

**Status:** prepared (smoke tests passed; full training not run yet)

**Question:** Can PPO with a CNN encoder on the fixed 3D LiDAR range image learn the NAV01 task (randomized static
corridor)?

**Hypothesis:** It reaches NAV01-level success (98 % on `ref_static_c10`), possibly with more iterations than the
2D policy. Learning would be in doubt if the success rate is still near zero after several hundred iterations.

## Setup

| | |
|---|---|
| Created | 2026-11-23 |
| Git commit at creation | `0b4e5ad` (M8 changes not yet committed) |
| Config | [Navigation/configs_3d/NAV04_static_lidar3d-r4_cnn.yaml](../../configs_3d/NAV04_static_lidar3d-r4_cnn.yaml) |
| Training runs | `Navigation/runs_3d/NAV04_static_lidar3d-r4_cnn/` |
| Evaluations | `Navigation/eval_3d/NAV04_static_lidar3d-r4_cnn/<protocol>/<policy>/` |
| Differs from NAV01 | The sensor and the policy encoder. Task, scene, domain randomization, rewards, dynamics, action limits, terminations and PPO hyper-parameters are the same. |
| Reference equivalent | none |

**Sensor:** fixed Mirrorcle 3D LiDAR, R4 mount:
* optical centre (0, 0, 0.40) m in base_link, pitch 6.75° down;
* FoV 28° × 14°, 139 × 34 rays, 12 m range, Gaussian noise σ = 1 cm;
* 4 Hz, with scans at the 5 Hz control instants; the latest complete scan is held between acquisitions.

**Observation:**
* `lidar`: the (1, 34, 139) range image / 12 m;
* `policy`: the 7 navigation features (goal distance, sin / cos of the goal bearing, previous action, v, w),
  empirically normalized.

**Network** (`st_nav/rl/lidar_cnn.py`, `CorridorNavCNNPPORunnerCfg`), for both actor and critic, each with its own
encoder:
* Conv 3×3 stride 2 with 16, 32, 64 filters (ReLU) → flatten (3072) → Linear 128 (ReLU);
* concatenated with the navigation features → MLP 256, 128 (ELU);
* outputs: a Gaussian over (v, w) for the actor, the value for the critic;
* 485k parameters per network.

## Commands

```bash
bin/isaacrun -g <gpu> -- python Navigation/scripts/train_3d.py --headless --config Navigation/configs_3d/NAV04_static_lidar3d-r4_cnn.yaml
bin/isaacrun -g <gpu> -- python Navigation/scripts/evaluate_3d.py --headless --checkpoint Navigation/runs_3d/NAV04_static_lidar3d-r4_cnn/<run>/model_<k>.pt
```

## Smoke tests (2026-11-23, `tests/sim/m8_smoke.py`, 13/13)

* Observation shapes and normalization; the image is the held scan.
* 4 Hz acquisitions at control instants, held between scans.
* Architecture as specified, separate encoders.
* Non-zero CNN gradients; PPO updates change every CNN weight with finite losses.
* Checkpoint round trip identical.
* At 4,096 envs: about 12.4 s per iteration (10.6k env steps/s), 26.9 GB peak GPU memory.
* `evaluate_3d.py` and the visualization tools run on the output.

## Training runs

| Run folder | Iterations | Seed | Throughput | Notes |
|---|---|---|---|---|

## Evaluations

| Protocol | Policy | Episodes | Success | Notes |
|---|---|---|---|---|

## Results
