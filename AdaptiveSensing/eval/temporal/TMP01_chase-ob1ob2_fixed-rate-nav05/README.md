# TMP01_chase-ob1ob2_fixed-rate-nav05

**Status:** done

**Question:** How does the frozen NAV05 3D-LiDAR navigation policy degrade when its LiDAR observation is refreshed at a
lower fixed rate (4, 2, 1, 0.5, 0.2 Hz) than the native 4 Hz sensor?

**Hypothesis:** Close to the baseline down to about 2 Hz, then a steep drop once the held scan is older than the time
needed to close on a moving obstacle.

## Setup

| | |
|---|---|
| Created | 2026-11-25 |
| Git commit at creation | `0b4e5ad` (plus uncommitted M8 3D-LiDAR work) |
| Config | none: command-line `--fixed <hz>`; scenario and policy from the NAV05 training run |
| Navigation policy (frozen) | `Navigation/runs_3d/NAV05_chase-ob1ob2_lidar3d-r4_cnn-from-nav04/2026-11-23_20-21-14_ppo_cnn_n4096_s42/model_1499.pt` |
| Evaluations | `AdaptiveSensing/eval/temporal/TMP01_chase-ob1ob2_fixed-rate-nav05/ref_chase_c10/<run>_model_1499_fixed<hz>Hz/` |
| Differs from previous experiment | NAV05 evaluation (`ref_chase_c10`, 100 episodes, seed 42) with a scan gate between env and policy |
| Reference equivalent | real-robot fixed-rate evaluation (`~/Sim2Real/Adaptive-Frame-Rate-Control-Sim2Real-TurtleBot`, `AdaptiveFPS/scripts/real_eval_fixed.py`) |

**Sensing.** Unchanged env: native LiDAR at 4 Hz (`lidar3d.period_s = 0.25`; scans taken at the 5 Hz control instants
and held in between, so native intervals are 0.2 / 0.4 s), policy at 5 Hz. The real-robot gate
(`AdaptiveSensing/evaluation/scan_gate.py`) sees every genuinely new native scan: the reset scan is always forwarded
(`next_sample_time = t0 + period`); afterwards a scan is forwarded when `t >= next_sample_time`, then
`next_sample_time += period` until it is past `t`. Forwarded scans replace `last_sampled_obs`; the policy receives
`{"policy": the env's current navigation features, "lidar": last_sampled_obs}`.

**Sensing metrics** (`episodes.csv`): counts include the reset scan; `measured_hz = (accepted_scans - 1) / duration`
excludes it. The terminal state is never observed, so episodes that end between two forwarded scans make the measured
rate lower than requested (strongly at 0.5 and 0.2 Hz, where most episodes end early in a collision).

## Commands

```bash
bin/isaacrun -g <gpu> -- python AdaptiveSensing/scripts/evaluate_fixed_rate.py --headless --fixed 4   # also 2, 1, 0.5, 0.2
python tests/sim/compare_eval_recordings.py <NAV05 eval_3d dir> <fixed4Hz dir>                     # 4 Hz = baseline
```

## Evaluations

Protocol `ref_chase_c10` (100 identical episodes in every condition). Paired comparison with 4 Hz: exact McNemar test
on the per-episode success.

| Requested | Measured (pooled) | Forwarded intervals | Obs. age mean / max | Success | Safety viol. | Timeout | Lost / gained vs 4 Hz | p |
|---|---|---|---|---|---|---|---|---|
| 4 Hz | 3.96 Hz | 0.2, 0.4 s | 0.04 / 0.2 s | **76 %** | 24 | 0 | (baseline) | |
| 2 Hz | 1.98 Hz | 0.4, 0.6 s | 0.16 / 0.4 s | 75 % | 25 | 0 | 4 / 3 | 1.0 |
| 1 Hz | 0.98 Hz | 1.0 s | 0.40 / 0.8 s | 66 % | 33 | 1 | 19 / 9 | 0.087 |
| 0.5 Hz | 0.45 Hz | 2.0 s | 0.87 / 1.8 s | 14 % | 86 | 0 | 62 / 0 | 4e-19 |
| 0.2 Hz | 0.15 Hz | 5.0 s | 2.15 / 4.8 s | 1 % | 99 | 0 | 75 / 0 | 5e-23 |

## Validation

* Simulator repeatability: re-running `evaluate_3d.py` reproduces the stored NAV05 evaluation bit for bit.
* 4 Hz: every native scan forwarded (11,860 / 11,860), gated LiDAR identical to the native one at every step, and the
  recording identical to the NAV05 evaluation (episodes.csv on all shared columns, every trajectory, steps, lidar and
  obstacles file): 76 / 100 reproduced.
* Every rate: native intervals only 0.2 / 0.4 s (native timing unchanged), no forward without a new native scan, held
  scan unchanged between forwards, navigation features = the env's current observation at every step.
* `tests/unit/test_scan_gate.py`: the gate equals a literal transcription of `real_eval_fixed.py` at all five rates.

### Reproducibility note: 2 Hz (added 2026-11-25, during TMP02 validation)

The stored recordings above are kept unchanged. Re-running the unchanged evaluator at 2 Hz (`evaluate_fixed_rate.py
--fixed 2 --out_dir ...`) six times gave two distinct results:

| Runs | Success | vs stored 2 Hz recording |
|---|---|---|
| 1 (GPU 2) | 75 % | identical in every column of `episodes.csv` |
| 5 (GPUs 0, 1, 3, 4, 5) | 76 % | 49 episodes differ (22 in length, 3 in outcome: episodes 27 and 63 goal instead of safety violation, 37 safety violation instead of goal); the 5 runs are identical to each other |

Not tied to a GPU: GPU 2 gave the stored result here and the other result in a TMP02 validation run. The source of the run-to-run variation has not
been identified; 4, 1, 0.5 and 0.2 Hz (TMP02 validation) and the NAV05 `evaluate_3d.py` re-run reproduced exactly. The
2 Hz conclusion is unchanged (75 or 76 % vs 76 % at 4 Hz). Outputs, moved from a temporary folder (their
`run_config.json` command lines still show the temporary `--out_dir`):

* `reproducibility/fixed2Hz_reruns/rerun<k>_gpu<g>/`: the six re-runs (evaluate_fixed_rate.py output + `console.log`);
  `comparison.csv` (per run: success, episodes differing from the stored recording and from rerun 1),
  `differing_episodes.csv` (stored vs rerun 1 length and outcome), `scratch_protocol_summary.csv` (the protocol
  summary rows written next to the re-runs);
* `reproducibility/nav05_evaluate_3d_rerun/`: the `evaluate_3d.py` re-run behind the repeatability bullet above;
* `console/evaluate_fixed_rate_<hz>Hz.log`: console output of the five stored evaluations.

## Results

NAV05 tolerates halving the observation rate (2 Hz: 75 %, not distinguishable from 76 %). At 1 Hz success drops to
66 % (not significant with 100 episodes, p = 0.09), and below 1 Hz the policy collapses (14 % at 0.5 Hz, 1 % at
0.2 Hz). All extra failures are safety violations (clearance < 5 cm), consistent with acting on a scan that no longer
shows the moving obstacles.

## Conclusion and next step

The usable range for a temporal sensing policy over this navigation model is roughly 1–4 Hz; 0.5 Hz and below are
unsafe as constant rates.
