# TMP02_chase-ob1ob2_adaptive-nav05

**Status:** implemented and validated; not trained (frame-cost values to be decided)

**Question:** Can a recurrent sensing policy choose the LiDAR rate (0.2-4 Hz) of the frozen NAV05 navigation policy so
that it acquires fewer scans while keeping navigation performance, in the OB1 + OB2 moving-obstacle scene?

**Hypothesis:** From TMP01 (2 Hz equal to 4 Hz, collapse below 1 Hz as a constant rate), a policy that lowers the rate
when nothing moves nearby and keeps it high near obstacles can save a large share of the frames.

## Setup

| | |
|---|---|
| Created | 2026-11-25 |
| Git commit at creation | `0b4e5ad` (plus uncommitted M8 / TMP01 / TMP02 work) |
| Config | `AdaptiveSensing/configs/temporal/TMP02_chase-ob1ob2_adaptive-nav05.yaml` (scene = NAV05's; env fast path on; 1024 envs x 256 steps = 262,144 transitions per iteration, 8 minibatches of 128 envs, 4 epochs, 200 M steps = 762 iterations; other PPO settings as the reference; sweep over `agent.frame_cost`) |
| Trainer | `AdaptiveSensing/scripts/train_temporal_ppo.py` (`--sweep_index i`: one sweep member per process) |
| Navigation policy (frozen) | NAV05 `model_1499.pt` (as TMP01) |
| Runs | `AdaptiveSensing/runs/temporal/TMP02_chase-ob1ob2_adaptive-nav05/<date>_ppo_lstm_n1024_s1_fc<cost>/` |
| Reference equivalent | adaptive_sensor_policy@dc5b53f `AdaptiveFPS/scripts/train_adaptive_fps_ppo.py` + `envs/turtlebot_var_scan_rate.py` (native-LiDAR mode, OB1 + OB2 config); real robot Sim2Real `real_eval_adaptive.py` |

**Sensing timing** (`AdaptiveSensing/training/temporal_sensing.py`). Env unchanged: native LiDAR 4 Hz (scans at the
5 Hz control instants), control 5 Hz. At every control instant t_k:
1. the sensing policy acts on the held scan *before* the scan of t_k is gated, plus `fps_ratio = fps / 4`,
   `obs_age_ratio = clip(steps_since / 25, 0, 1)` and `frame_ratio = clip(frames / budget, 0, 1)`; the LSTM steps
   every tick;
2. the native scan of t_k is accepted if it is new and t_k >= next_sample_time (real-robot gate, deadline advanced by
   whole periods);
3. on acceptance the selected rate is applied (`set_target_fps`: a changed rate re-anchors next = t_k + 1 / fps),
   steps_since = 0, frames += 1; otherwise the action is discarded (mask 0);
4. NAV05 acts on the most recent accepted scan (as TMP01), then the env steps.

Example (unit test): accepted scan at 4.0 s at 1 Hz -> 4.2-4.8 s actions discarded -> the action selected at 5.0 s
(held 4.0 s scan, age counter 4) gates the 5.0 s scan and sets the next interval.

**Reward:** NAV05 env reward unchanged (progress, proximity, time, goal +30, collision -30, timeout 0)
`- frame_cost` per accepted scan `- budget_penalty` (15) once when the episode's frames first exceed `budget` (500:
unreachable, at most 281 frames in 70 s). Reset scan: frame 1 of the new episode, not charged; reset rate 4 Hz.

**Terminal step.** The last tick of an episode gates the scan of its own instant like any other tick (counted,
action applied, cost charged). The instant at which the episode ends has no decision, so its scan is never gated;
Isaac Lab returns the next episode's reset observation there, which goes only to the next episode. Difference from the
reference native-LiDAR simulator: there the scans of every step's window (t_start, t_end] were gated by the decision
at t_start, so a scan arriving inside the last step was counted in that step; here scans sit on control instants and
each is gated by the decision taken at its own instant. Per-episode frame counts therefore equal the TMP01 convention.

**Sensing policy** (`sensing_agent.py`): NAV05 CNN architecture (Conv 16/32/64 3x3/2 -> 3072 -> Linear 128 ReLU,
random init, trained) + 3 ratios -> Linear 64 Tanh -> LSTM 64 -> actor / critic 64-64 (reference Agent; trunk shared);
475,398 parameters. **Recurrent PPO** (`recurrent_ppo.py`): reference update (minibatches of whole envs over the full
rollout from the rollout-start LSTM state; policy loss, entropy, KL, clipfrac and advantage normalization on accepted
ticks only; value loss on all ticks), minibatches split into env chunks with accumulated gradients (same update).

## Commands

```bash
bin/isaacrun -g <gpu> -- python AdaptiveSensing/scripts/train_temporal_ppo.py --headless \
    --config AdaptiveSensing/configs/temporal/TMP02_chase-ob1ob2_adaptive-nav05.yaml --sweep_index <i>
bin/isaacrun -g <gpu> -- python tests/sim/tmp02_validation.py --headless --fixed <hz>     # vs TMP01 (4, 1, 0.5, 0.2)
bin/isaacrun -g <gpu> -- python tests/sim/tmp02_validation.py --headless --fixed 2 \
    --compare_dir AdaptiveSensing/eval/temporal/TMP01_chase-ob1ob2_fixed-rate-nav05/reproducibility/fixed2Hz_reruns/rerun1_gpu1
bin/isaacrun -g <gpu> -- python tests/sim/tmp02_validation.py --headless --adaptive       # random actions
# results: validation/tmp02_validation_<mode>[_<tag>].json (--tag to keep repeated runs apart)
```

## Validation (2026-11-25)

* Unit tests (`tests/unit/test_scan_gate.py`, `test_temporal_sensing.py`, `test_sensing_agent.py`): rate changes =
  literal `real_eval_adaptive.py` gate on random action sequences; decision timing (example above); masks, frame
  accounting, reward decomposition, budget penalty once per episode, terminal tick, reset attribution, NAV05 input;
  encoder shapes = NAV05 encoder (weights not copied); done-reset of the LSTM; replay reproduces rollout log-probs;
  chunked update = literal reference minibatch update (losses, KL, gradients); gradients reach every CNN layer; GAE.
* Fixed rates in Isaac Lab (`validation/tmp02_validation_fixed*.json`; constant action, gate reset at that rate,
  `ref_chase_c10`): 4, 1, 0.5, 0.2 Hz identical to the TMP01 recordings in every episode (outcome, steps, return,
  accepted / native scans, mean / max observation age). 2 Hz: four runs (GPUs 1, 7, 2, 6), all with the same result
  (76 %), which differs from the stored TMP01 2 Hz recording (`..._fixed2Hz_run1_gpu1.json`, `..._run3_gpu2.json`:
  pass false, 22 lengths / 3 outcomes) and is identical to the TMP01 evaluator's common re-run result
  (`..._fixed2Hz_run4_gpu6_vs_tmp01_rerun1.json`: 0 mismatches); see the TMP01 note below. Run 2 (GPU 7) is kept as
  its console log only.
* Random sensing actions in Isaac Lab (`tmp02_validation_adaptive.json`; 100 envs x 1200 ticks, 2,390 episodes): 0
  violations of the literal accept / re-anchor / discard rule, masks, reward, frame and causal counts, observation
  ratios, reset observations; native intervals only 0.2 / 0.4 s; 138 terminal ticks with an accepted scan, charged.
* Trainer smoke run at the earlier scale (1024 envs x 256 steps, 2 iterations, frame_cost 0;
  `AdaptiveSensing/runs/temporal/TMP02_chase-ob1ob2_adaptive-nav05/2026-11-25_17-03-07_ppo_lstm_n1024_s1_fc0_smoke/`,
  with `console.log`): rollout ~41 s, update ~10-12 s per iteration (~5,000 SPS); peak GPU memory 6.8 GB rollout,
  22.4 GB update (first iteration, 10.9 GB after); image buffer 4.96 GB; NAV05 parameters and normalizer unchanged;
  approx KL ~0 and clipfrac 0 at the first update.
* Trainer smoke run at the training scale (512 envs x 256 steps, 2 iterations, frame_cost 0;
  `.../2026-11-25_18-04-26_ppo_lstm_n512_s1_fc0_smoke/`, with `console.log`): rollout ~38-39 s, update ~5-6 s per
  iteration (~2,950 SPS); peak GPU memory 3.4 GB rollout, 19.4 GB update (first iteration, 7.9 GB after); image buffer
  2.48 GB; NAV05 unchanged. The rollout time barely depends on the number of envs (1024 envs: ~41 s), so halving the
  envs halves the throughput: 1,525 iterations x ~44 s = ~18-19 h for 200 M steps.
* Console output of every validation run: `validation/console/`.

**Note on TMP01 at 2 Hz.** The TMP01 evaluator, re-run 6 times at 2 Hz, reproduced the stored recording (75 %) once
and gave one other result (76 %; 49 episodes differ, 22 in length, 3 in outcome) 5 times; 4, 1, 0.5 and 0.2 Hz
reproduced exactly. The simulator is therefore not bit-repeatable in every condition; the source has not been
identified. Details and outputs: TMP01 card, "Reproducibility note".

## Performance profiling (2026-11-25)

`AdaptiveSensing/scripts/profile_temporal_rollout.py` (trainer's rollout code path, untrained policy, frame_cost 0;
one RTX A6000 per run, runs one after the other unless stated). Results: `profiling/profile_n<N>[_<tag>].json`,
console output in `profiling/console/`.

| Envs | ms / control tick | Rollout (256 ticks) | Transitions / s | PPO update | Update peak memory |
|---|---|---|---|---|---|
| 128 | 145.0 | 37.1 s | 883 | 3.7 s | 9.0 GB |
| 256 | 148.2 | 37.9 s | 1,728 | 4.4 s | 18.0 GB |
| 512 | 146.6 | 37.5 s | 3,493 | 6.8 s | 19.6 GB |
| 1024 | 173.4 | 44.4 s | 5,907 | 11.5 s | 22.7 GB |

Per-tick time is flat up to 512 envs: the rollout is bound by per-step launch / Python / host-synchronization overhead
(process CPU time 1.7-2.2 x wall; CUDA-synchronized instrumentation does not slow it down), not by GPU work.

Breakdown at 512 envs (inclusive ms per tick, instrumented): env.step 140.5 (96 %) = apply_action x20 51.3 (of which
moving-obstacle model + pose writes 33.6), PhysX sim.step x20 38.1, episode resets 14.4, scene.update x20 11.8,
get_observations 11.7 (ray casting 4.5), write_data_to_sim x20 9.6, dones 1.4, rewards 0.3; sensing agent 1.7 (CNN
0.7), NAV05 0.8, gate 0.6, sensing observation 0.2, buffer writes 0.3. About 138 host synchronizations per tick,
almost all in Isaac Lab and env code (sensor_base 55, articulation 27, corridor_env 23, crng 13).

* Env internals (`--detail`, `profile_n512_detail.json`): per control tick ~9,200 GPU kernels, ~650 async copies and
  ~580 stream synchronizations (PhysX included) for 42.7 ms of GPU kernel time: the wall time is ~3.5 x the GPU busy
  time (launch / Python bound). Largest st_nav costs: obstacle pose writes 24.6 ms, diff-drive command model 15.2 ms,
  eager contact-sensor refresh 12.2 ms (sensor unused by the env), obstacle model 11.6 ms, counter-based RNG 11.0 ms
  (LiDAR noise 5.3, resets); torch-visible host syncs by source line are listed in the JSON.
* CNN reuse on unchanged held scans: the held image changes on 14 % of (tick, env); encoding only those saves
  0.3 ms / tick (0.2 %). Not bitwise equal (cuDNN results depend on the batch: features differ by up to 2e-4, log-probs
  by 2e-6, 22 of 131,072 argmax actions flip). Not adopted.
* Concurrent runs: four 512-env profiles on four GPUs at once: 145-161 ms / tick (mean +3.5 %, worst +10 % vs alone).
* Iteration at 512 envs: ~44 s (rollout 37.5 + update 6.8) -> ~18.8 h for 1,525 iterations.
* Data-parallel PPO over W GPUs with the global 512 x 256 batch: every rank still runs 256 ticks at the same per-tick
  time, so only the update shrinks: W = 4 -> ~37.5 + ~3.7 s = ~41 s per iteration (~1.08 x). Not pursued.

## Env fast path (2026-11-25)

Opt-in env setting `fast_path` (`CorridorNavEnvCfg`; items in `st_nav.envs.corridor_scene.FAST_PATH_ITEMS`, or `all`;
default `()` = original code). Each item removes redundant work without changing a computed value:
`lazy_contact` (contact sensor `history_length` 0: refreshed only when read; the env never reads it), `cm_constants`
(command-model smoothing factors / acceleration limits computed once), `rng_key_cache` (hashed constant integer keys of
the counter-based RNG cached), `joint_ids` (wheel joint indices as a device tensor), `reset_constants` (constant
tensors of the reset samplers built once), `obstacle_pose_buffers` (obstacle pose tensors preallocated, x / y written
per physics step, still through Isaac Lab's `write_root_pose_to_sim`). Not used: direct PhysX writes, CUDA graphs,
torch.compile, changes to the delay buffer or the object representation. Selected with `--fast_path` in
`evaluate_3d.py`, `evaluate_fixed_rate.py`, `tests/sim/tmp02_validation.py`, the profiler, and `env.fast_path` in a
config / `--set` for training. Enabled in the TMP02 config (`env.fast_path: [all]`) on 2026-11-25.

**Equivalence** (`fast_path/`):
* `tests/unit/test_fast_path.py`: RNG keys, command model (incl. parameter changes mid-run), reset samplers bitwise.
* Whole-env traces (`tests/sim/fast_path_equivalence.py`; TMP02 env, 512 envs x 800 ticks, NAV05 + random sensing
  actions, 8,430 episode resets): robot pose / velocity, wheel velocities, obstacle bodies and model, LiDAR digests,
  navigation features, rewards, dones, outcomes, masks bitwise identical to the original path for every item alone and
  for `all` (`compare_none_vs_<item>.txt`); the original path repeated is also identical.
* Experiments with `all` (`regressions/`): NAV05 `ref_chase_c10` identical to the stored recording in every file;
  TMP01 4, 2 (vs re-run 1), 1, 0.2 Hz identical; TMP02 random actions 0 violations. NAV04 `ref_static_c10` (2 episodes)
  and TMP01 0.5 Hz (1 episode) differed in the first run; repeated twice with each path (`regressions/repeats/`): the
  original path shows the same deviations in 1 of 2 runs, the fast path reproduced the stored recordings in 2 of 2, and
  the NAV04 alternative result is identical in every file whichever path produced it: the simulator's run-to-run
  variation (TMP01 note), not the fast path.

**Speed** (`fast_path/timing/`; 512 envs, one GPU, runs one after the other):

| | ms / control tick | Host syncs / tick (torch-visible) |
|---|---|---|
| original (2 runs) | 154.6, 153.8 | 138 |
| lazy_contact | 149.6 | 98 |
| cm_constants | 151.2 | 138 |
| rng_key_cache | 149.5 | 125 |
| joint_ids | 155.7 | 113 |
| reset_constants | 149.9 | 133 |
| obstacle_pose_buffers | 150.9 | 138 |
| all (2 runs) | 135.4, 138.0 | 55 |

Single items are within ~1-5 ms (run-to-run noise ~1-3 ms); together -17.5 ms per tick (-11 %). Training, 3
iterations each (`runs/.../*_fastpath-bench-none`, `*_fastpath-bench-all`): rollout 39.0-40.0 s -> 33.7-34.4 s,
update unchanged (4.5 s after the first iteration), iteration ~44 s -> ~38.5 s (1.14 x; SPS 2,930 -> 3,342 at
iteration 3); 1,525 iterations ~18.7 h -> ~16.4 h. The logged training statistics of the two runs (episodes, returns,
success, frames) are identical.

**Training configuration benchmark** (2026-11-25; final TMP02 config: 1024 envs x 256 steps, 8 minibatches, 4 epochs,
chunks of 64 envs, fast path on; 3 iterations, frame_cost 0, one GPU alone;
`runs/.../2026-11-25_19-39-01_ppo_lstm_n1024_s1_fc0_bench/`): rollout 37.8 / 39.3 / 39.1 s, update 10.8 / 8.9 / 8.9 s
(first iteration includes cuDNN autotuning), ~48 s per iteration = ~5,450 transitions / s; peak GPU memory 6.8 GB in the
rollout, 22.4 GB in the first update, 10.9 GB afterwards. Estimated training time (steady iteration ~48 s): 100 M steps
(381 iterations) ~5.1 h, 200 M steps (762 iterations) ~10.2 h per sweep member.

## Frame-cost calibration on the fixed rates (2026-11-25)

`AdaptiveSensing/scripts/frame_cost_calibration.py` on the TMP01 recordings (`ref_chase_c10`, 100 paired episodes per
rate), no simulation: penalized return = NAV05 navigation return - frame_cost x charged acquisitions, charged =
accepted_scans - 1 (TMP02 convention: reset scan free, scan accepted on the last control step charged, episode-end
instant never gated). Output: `frame_cost_calibration/` (`fixed_rate_penalized_returns.csv` for costs 0, 0.025, 0.05,
0.1, 0.2; `dense_grid.csv` 0-4.5 by 0.005; `crossovers.json` with paired-bootstrap 95 % intervals; `*_2hz_rerun1*`: the
same with the common 2 Hz re-run result instead of the stored one).

Best fixed rate by mean penalized return: 4 Hz for c < 0.035, 2 Hz for 0.035-0.38, 1 Hz for 0.385-2.23, 0.5 Hz for
2.235-3.91, 0.2 Hz above 3.92 (2 Hz re-run variant: 2 Hz from 0.015, 1 Hz from 0.425). Crossovers (95 % CI): 2 vs 4 Hz
0.032 (-0.07, 0.12); 1 vs 4 Hz 0.154 (0.04, 0.26); 1 vs 2 Hz 0.382 (0.08, 0.65); 0.5 vs 4 Hz 0.61; 0.2 vs 4 Hz 0.71.
Above ~0.6 the cost makes the crash-prone rates (0.5, 0.2 Hz: 14 % / 1 % success) outrank 4 Hz.

## Results

Not trained yet.
