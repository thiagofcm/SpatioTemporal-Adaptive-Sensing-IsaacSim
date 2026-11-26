# STS01_chase-ob1ob2_fixed-sections-nav05

**Status:** done (factorial 2026-11-26: 125 configurations x 100 episodes, all in-run checks passed)

**Question:** How does the frozen NAV05 navigation policy behave when the left / center / right sections of the 3D
range image are gated independently at constant rates (all 5^3 = 125 combinations of 0.2, 0.5, 1, 2, 4 Hz)?

**Hypothesis:** The center section matters most; side sections can be slowed with a smaller loss than slowing the
whole scan (TMP01), but asynchronous sections (stale columns next to fresh ones) may hurt NAV05 beyond what their
consumption suggests. This sets the baselines (success vs full-scan equivalents) for STS02.

## Setup

| | |
|---|---|
| Created | 2026-11-26 |
| Git commit at creation | `d7abde6` (plus uncommitted TMP01 / TMP02 / STS01 work) |
| Config | none: `AdaptiveSensing/scripts/evaluate_fixed_sections.py --sections <fL> <fC> <fR>`; factorial driver `AdaptiveSensing/scripts/run_sts01_factorial.py` |
| Evaluations | `AdaptiveSensing/eval/spatiotemporal/STS01_chase-ob1ob2_fixed-sections-nav05/ref_chase_c10/<NAV05 run>_model_1499_sections<fL>-<fC>-<fR>Hz/` |
| Navigation policy (frozen) | NAV05 `model_1499.pt` (as TMP01 / TMP02) |
| Differs from previous experiment | TMP01 with one gate per horizontal section instead of one for the whole scan |
| Reference equivalent | ST00_fixed-sector-factorial in `~/adaptive_sensor_policy` (branch `spatiotemporal`) |
| Plan | [docs/spatiotemporal_sensing_plan.md](../../../../docs/spatiotemporal_sensing_plan.md) |

**Sections** (`AdaptiveSensing/evaluation/sections.py`, derived from the env's ray table): columns 0-45 / 46-92 /
93-138 (left / center / right; column 0 = +14 deg) = 1,564 / 1,598 / 1,564 rays of 4,726; weights 0.33094 / 0.33813 /
0.33094.

**Sensing** (`AdaptiveSensing/evaluation/section_scan_gate.py`). Env unchanged (native LiDAR 4 Hz at the 5 Hz
control instants, held in between; control 5 Hz). Every new native scan goes through the real-robot gate of each
section independently: section s accepts it when t >= next_sample_time[s] (then its deadline advances by whole periods
on the lattice t0 + m / f_s); only the accepting sections' columns of the held image are replaced; the others keep
their last accepted scan. Reset: every section accepts the reset scan, starts at its own rate. NAV05 receives the
composite 34 x 139 image plus the current navigation features.

**Consumption:** full-scan equivalents FSE = rays acquired (reset scan included) / 4,726 = sum_s w_s n_s; equal to the
TMP01 frame count when the sections are synchronized. `adaptive_return` = nav return - frame_cost (FSE - 1) (frame_cost
0 here, so adaptive = nav return).

**Protocol:** `ref_chase_c10` exactly as TMP01: 100 envs = 100 episodes (first episode of each env), start poses from
seeds 4000-4099, goal C10 (-2, -2), env seed 42 (counter-based episode seeds), 70 s, NAV05 deterministic. One fresh
process per configuration (identical conditions to the TMP01 per-rate runs).

**Outputs** (no per-episode recordings for the factorial; `--record` adds them): per configuration `episodes.csv`
(reference schema of `AdaptiveSensing/evaluation/metrics.py` + section columns: `hz_/acq_/mean_interval_s_/
mean_obs_age_s_/max_obs_age_s_<section>`, `accepted_scans`, `native_scans`, ray-weighted `mean_obs_age_s`,
`max_obs_age_s`, `start_pose_index`, obstacle speeds / path y; `episode_frame_count` = FSE), `summary.json` (incl. the
in-run checks and timing), `run_config.json`; one row per configuration in `ref_chase_c10/summary.csv`
(`AdaptiveSensing/evaluation/section_metrics.py`).

## Commands

```bash
bin/isaacrun -g 6 -- python AdaptiveSensing/scripts/evaluate_fixed_sections.py --headless --sections 4 1 0.2 [--record]
python AdaptiveSensing/scripts/run_sts01_factorial.py --gpus 6 7          # all 125, resumable (skips finished ones)
python tests/sim/sts01_vs_tmp01.py <synchronized run dir> [<tmp01 run dir>]   # vs the TMP01 recording
python tests/sim/sts01_recording_check.py <run dir made with --record>
```

## Validation (2026-11-26; `validation/`)

* **Unit tests** (CPU): `tests/unit/test_section_scan_gate.py` (11) - layout 46/47/46 -> 1564/1598/1564 rays for
  every ray order; each section = the literal real-robot gate (real_eval_fixed.py) at its own rate for all 125
  combinations; every column of the composite = its section's latest accepted scan at every tick, other sections'
  columns untouched; synchronized sections = `ScanGate` bitwise (fixed and with rate changes); per-section
  `set_rate` = literal real_eval_adaptive.py; counters, rays, FSE exact; masked reset. Three deliberate faults
  (whole image replaced, deadline re-anchored at `now`, rays counted per scan) are each caught.
  `tests/unit/test_section_metrics.py` (3): in-memory collision type = file-based `metrics.collision_type` on all 100
  TMP01 1 Hz episodes; a synchronized row = the reference row; FSE / adaptive-return accounting. Whole unit suite:
  84 passed, 1 skipped.
* **In-run checks** (every evaluation, `summary.json` "checks", all 0 in every validation run): independent literal
  per-section gate (accept / deadline / last acquisition / counters), composite image vs a history of the native
  scans (each section's columns = the native scan of its last acquisition), untouched columns of non-accepting
  sections, native intervals only 0.2 / 0.4 s, reset composite = reset scan, all-4 Hz composite = native scan, NAV05
  navigation features current.
* **Synchronized sections = TMP01** (`compare_tmp01/`): 4, 2, 1, 0.5, 0.2 Hz in all three sections, with and without
  `--record`: `episodes.csv` identical in every reference column for all 100 episodes (outcome, steps, returns, frame
  count = FSE, final pose, collision type, ...), per-episode accepted / native scans and mean / max observation age
  identical; with `--record`, `trajectory.csv`, `steps.csv`, `obstacles.npz` and every TMP01 `lidar.npz` array
  identical, and the composite rebuilt from the section arrays equals the TMP01 observation at every row. 2 Hz is
  compared with the common TMP01 re-run (`reproducibility/fixed2Hz_reruns/rerun1_gpu1`, 76 %; see the TMP01 note on
  the simulator's run-to-run variation at 2 Hz); success 1 / 14 / 66 / 76 / 76 % as TMP01.
* **Mixed rates** (4-1-0.2 and 0.2-2-0.5 with `--record`; 1-4-2 and 0.5-0.5-4 without): all in-run checks 0;
  offline (`recording_checks/`) acquisitions per section = `acq_<s>`, every acquisition on its section's lattice
  (first native scan at / after the deadline), rays: 4,726 + sum of `beams_acquired` = FSE x 4,726, composite minimum
  = `min_lidar_m`, per-section scan indices = latest acquisition of that section.

**Execution time.** One configuration = one process: ~15 s launch / setup + the episodes (until the longest of the
100 first episodes ends: ~24 s when everything collides early, ~32 s at 4 Hz, up to ~66 s with timeouts) + ~5-15 s
shutdown; measured walls 34-35 s (0% success, a 0.2 Hz section) to ~85 s, two GPUs concurrently next to the six TMP02
runs. Estimate for 125 configurations on GPUs 6 + 7: ~60-75 min. Output ~0.3 MB per configuration.

## Results (2026-11-26)

`ref_chase_c10/summary.csv` (one row per configuration), per-configuration `episodes.csv` / `summary.json`;
`ref_chase_c10/factorial_progress.jsonl` (exit code, checks, wall time per configuration), `console/` per-configuration
logs. Run: `run_sts01_factorial.py --gpus 6 7`, 44.9 min for 125 configurations (wall 22-63 s each, mean 43 s; two GPUs
next to the six TMP02 runs), 0 failures, every configuration's in-run checks 0; 40 MB in total.

**Synchronized configurations** (success / mean FSE): 0.2 Hz 0.02 / 2.3, 0.5 Hz 0.14 / 5.8, 1 Hz 0.66 / 30.4,
2 Hz 0.76 / 61.0, 4 Hz 0.76 / 118.6. 1, 2 (vs the common re-run) and 4 Hz identical to TMP01 in every episode. 0.2 and
0.5 Hz differ from TMP01 in a few episodes (0.2 Hz: 1 outcome, 7 lengths; 0.5 Hz: 1 length, final-pose decimals),
although the validation runs of the SAME configurations with the same code were identical to TMP01: the simulator's
run-to-run variation documented in the TMP01 card (source not identified), not the section gate. Differences between
configurations of a few episodes (a few percentage points) are therefore within run-to-run noise.

**Pareto set** (success up, FSE down): 0.2/0.2/0.2 (0.02, 2.3), 0.5/1/0.5 (0.04, 5.5), 0.5/0.5/0.5 (0.14, 5.8),
1/0.5/0.5 (0.19, 9.3), 2/0.5/0.5 (0.20, 14.2), 2/0.5/1 (0.21, 16.4), 1/0.5/2 (0.24, 19.0), 0.5/1/1 (0.41, 19.6),
0.5/2/1 (0.49, 29.5), 1/1/1 (0.66, 30.4), 1/1/2 (0.70, 41.1), 2/1/1 (0.73, 42.0), 1/2/2 (0.74, 52.7), 2/2/2 (0.76,
61.0), 2/4/2 (0.77, 83.5), 4/1/4 (0.78, 88.4), 4/4/2 (0.79, 103.7)  (left/center/right Hz).

Findings:
1. **No fixed spatial split beats uniform sensing at equal consumption.** Against the uniform (TMP01-like) curve,
   linearly interpolated in FSE, 6 of the 120 mixed configurations lie above it, by at most +0.03 (2/1/1 0.73 at
   42.0 FSE, 4/4/2 0.79 at 103.7); none differs significantly from uniform 2 or 4 Hz (paired McNemar p >= 0.38).
   77 lie 0.10 or more below it.
2. **A 0.2 Hz section is fatal:** best success over the 61 configurations containing one is 0.48 (most 0-10 %).
3. **Sections are not equally important, and not left/right symmetric.** Mean success (other sections >= 0.5 Hz) by
   section rate 0.2 / 0.5 / 1 / 2 / 4 Hz: left 0.15 / 0.35 / 0.44 / 0.48 / 0.51, center 0.01 / 0.24 / 0.53 / 0.52 /
   0.49, right 0.00 / 0.09 / 0.49 / 0.58 / 0.63. The right section is the most sensitive, the center needs >= 1 Hz, the
   left tolerates 0.5-1 Hz best. The cause of the left/right asymmetry (route geometry of this protocol, obstacle
   side) has not been analysed yet.
4. **Asynchrony itself costs performance.** With the center at 4 Hz, sides at 1 Hz give 0.35 (38.7 FSE) against 0.66
   for uniform 1 Hz (30.4 FSE; paired p < 0.001): more sensing, worse navigation. 1/2/1 gives 0.55 (p = 0.07). The
   reverse is benign: sides at 4 Hz with the center at 1 Hz give 0.78 (88.4 FSE). Frozen NAV05 is hurt by stale side
   columns next to a fresh center (inconsistent image), consistent with the risk in the plan (section 5.4).

## Conclusion and next step

Fixed spatial allocations offer no consumption saving over uniform sensing for frozen NAV05 in this scene: the
uniform 1 / 2 Hz points are on (or within noise of) the Pareto front, and many mixed allocations, especially a fast
center with slow sides, are worse than uniform sensing at lower cost. Any benefit of STS02 must therefore come from
state-dependent (adaptive) allocation, not from a better constant split, and the policy must learn to avoid
harmful asynchronous combinations (e.g. 1/4/1). Frame-cost decision for STS02 deferred to the review together with
the TMP02 results (as agreed). Caveat: one protocol (C10 goal, 100 episodes); differences of a few points are within
the simulator's run-to-run variation.
