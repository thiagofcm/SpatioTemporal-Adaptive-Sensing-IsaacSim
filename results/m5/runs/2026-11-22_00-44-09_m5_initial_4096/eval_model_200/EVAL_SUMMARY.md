# M5 initial run: evaluation of model_200.pt (reference protocol, 100 episodes)

Protocol (as adaptive_sensor_policy/AdaptiveFPS/evaluation):
* episode i: Z1-Z4 sizes and offsets from the reference sampler, seed 3000+i (held out: training bank used seeds 0-1023);
* start pose from the reference sample_start_pose(4000+i) (verified identical for all 100 seeds);
* fixed goal (-2.0, -2.0) (reference EVAL_GOAL "C10"; not one of the 9 training goals);
* 80 s timeout (400 steps), deterministic policy (actor mean), LiDAR noise on.

Checks: start placement error 1.5e-6 m, offsets equal to the reference values, goal exact.

| Metric | Value |
|---|---|
| Success | **98 / 100** |
| Safety violations | 2 (episodes 25, 73; both clip Z1 at its rightmost offset, at steps 38 and 45) |
| Physical collisions / timeouts | 0 / 0 |
| Return | 84.1 ± 14.3 |
| Success: time / path / mean speed | 29.0 ± 0.7 s / 6.31 ± 0.15 m / 0.217 m/s (v_max 0.22) |
| Minimum clearance (all episodes) | mean 0.090 m, min 0.047 m; 23 of 98 successes came within 0.08 m (limit 0.05 m) |
| By start point (success/episodes) | 0: 23/23, 1: 17/18, 2: 19/20, 3: 16/16, 4: 23/23 |
| By Z pair modes | hard/hard 31/31, hard/normal 21/21, normal/hard 20/21, normal/normal 26/27 |
| Time | 31 s (13.5 s launch and setup, 17.6 s episodes) |

Files: episodes.csv (one row per episode, reference-style fields), summary.json, trajectories.npy, eval_trajectories.png.
