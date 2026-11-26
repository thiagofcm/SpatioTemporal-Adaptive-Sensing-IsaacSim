# NAV05_chase-ob1ob2_lidar3d-r4_cnn-from-nav04

**Status:** prepared (smoke tests passed; full training not run yet)

**Question:** Starting from the NAV04 static-scene policy, can PPO with the CNN on the fixed 3D LiDAR learn the NAV03
task (dynamic corridor, OB1 + OB2 moving)?

**Hypothesis:** The warm start shortens training compared with NAV03 (1,500 iterations from scratch, 2D LiDAR). The
NAV04 policy does not transfer directly: in the smoke run about 85 % of the first training episodes ended in safety
violations (the scene has no baffles, moving obstacles, other start poses and goals). Measuring the benefit of the warm
start itself would need a 3D from-scratch run on the same task (not planned yet).

## Setup

| | |
|---|---|
| Created | 2026-11-23 |
| Git commit at creation | `0b4e5ad` (M8 changes not yet committed) |
| Config | [Navigation/configs_3d/NAV05_chase-ob1ob2_lidar3d-r4_cnn-from-nav04.yaml](../../configs_3d/NAV05_chase-ob1ob2_lidar3d-r4_cnn-from-nav04.yaml) |
| Training runs | `Navigation/runs_3d/NAV05_chase-ob1ob2_lidar3d-r4_cnn-from-nav04/` |
| Evaluations | `Navigation/eval_3d/NAV05_chase-ob1ob2_lidar3d-r4_cnn-from-nav04/<protocol>/<policy>/` |
| Differs from NAV04 | The task (NAV03 scene: corridor without Z1-Z4, OB1 + OB2 moving with speed / path-y DR, reference start poses and goals) and the initialization. Sensor, network, observation, rewards and PPO hyper-parameters are the same. |
| Differs from NAV03 | The sensor (3D LiDAR, R4 mount, 4 Hz) and the network (CNN), as NAV04. |
| Reference equivalent | none (task as ST05, sensor and network new) |

**Initialization** (`init_from` in the config, `train_3d.py`): NAV04 run `2026-11-23_16-56-10_ppo_cnn_n4096_s42`,
`model_499.pt` (97 % success on `ref_static_c10`). Warm start (`mode: weights`): actor and critic weights, including
the observation-normalizer statistics and the action std (0.49); fresh Adam optimizer, iteration counter from 0,
learning rate 3e-4 (adaptive). The checkpoint's SHA-256 and every loaded tensor's identity are recorded in
`params/run_info.json`.

## Commands

```bash
bin/isaacrun -g <gpu> -- python Navigation/scripts/train_3d.py --headless --config Navigation/configs_3d/NAV05_chase-ob1ob2_lidar3d-r4_cnn-from-nav04.yaml
bin/isaacrun -g <gpu> -- python Navigation/scripts/evaluate_3d.py --headless --checkpoint Navigation/runs_3d/NAV05_chase-ob1ob2_lidar3d-r4_cnn-from-nav04/<run>/model_<k>.pt
```

`evaluate_3d.py` uses `ref_chase_c10` for this dynamic scene by default.

## Smoke tests (2026-11-23)

* 64 envs, 2 iterations: 37 tensors loaded, identical to the checkpoint; iteration 0, learning rate 3e-4; action std
  0.49 inherited (a fresh policy starts at 1.0); `init_from` recorded in `params/run_info.json` and `params/config.yaml`.
* 4,096 envs, 3 iterations: about 11.8 s per iteration (11.1k env steps/s), 25.6 GB peak GPU memory, setup 106 s.
  1,500 iterations take about 5 h.
* `--init_from <NAV04 model_499> --resume` on the NAV04 config: continues at iteration 499 with the learning rate
  reached (1.3e-3), saves `model_500.pt`.
* A 2D checkpoint (NAV01) is rejected (size mismatch) before anything is written.

## Training runs

| Run folder | Iterations | Seed | Throughput | Notes |
|---|---|---|---|---|

## Evaluations

| Protocol | Policy | Episodes | Success | Notes |
|---|---|---|---|---|

## Results

## Conclusion and next step
