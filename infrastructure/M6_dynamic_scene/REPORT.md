# M6 report: dynamic obstacle scene (OB1, OB2)

Status: implemented and validated; navigation experiments NAV02 (OB1) and NAV03 (OB1 + OB2) are prepared, but their
full training has not been run.

## 1. What was implemented

The scene follows the reference `corridor_dynamic_chase_random` (adaptive_sensor_policy `envs/scenes`, commit
`d58be3fadb`). It is ported, not imported.

| File (`source/st_nav/st_nav/`) | Content |
|---|---|
| `scenes/__init__.py` | scene registry (`SCENE_MODULES`, `get_scene`), with a fixed interface per scene: `NAME`, `WALLS`, `HAS_BAFFLES`, `MOVING_OBSTACLES` |
| `scenes/_common.py` | arena, corridor walls, start poses, goal lists and their samplers (moved from `corridor_static_random.py`, which re-exports them unchanged) |
| `scenes/_obstacle_motion.py` | `RandomPingPongObstacles`: GPU-vectorized port of the reference `random_ping_pong` (details below) |
| `scenes/corridor_dynamic_chase_random.py` | the scene (details below) |
| `envs/corridor_scene.py` | scene runtime (details below) |
| `tasks/nav2d/` | env cfg fields `scene_name`, `active_obstacles`, `obstacle_{presence,speed,y,radius}_dr`; the env advances the obstacles in `_apply_action` (every physics step); generic task ID `STNav-TB4-Corridor-v0` (same classes as `STNav-TB4-Corridor-Static-v0`) |

**`random_ping_pong` motion (`_obstacle_motion.py`).** Each environment and obstacle has its own state and random
draws:
* at reset: s0 ~ U(0, L), a random direction, and a reversal target in the region ahead;
* reversal regions: the outer 25 % of the path at each end;
* at each reversal: a turnaround pause ~ U(4.6, 5.0) s, with the leftover time of the step spent inside the pause;
* presence, speed and path-y randomization per episode.

**The scene (`corridor_dynamic_chase_random.py`).**
* No Z1–Z4 baffles.
* OB1: (1.4, 1.7) → (2.4, 1.7). OB2: (2.4, 1.1) → (1.4, 1.1). Both at 0.06 m/s, radius 0.17 m.
* DR ranges, as in the reference:
  * presence: 0.5 each, never empty;
  * speed: U(0.04, 0.08) m/s;
  * path y: OB1 ∈ [1.20, 1.70], OB2 ∈ [0.60, 1.10], gap y1 − y2 ∈ [0.55, 0.65] m (vectorized rejection sampler).
* The reference ST04/ST05 goal list.

**Scene runtime (`envs/corridor_scene.py`).**
* The scene is chosen by `scene_name`.
* OB1/OB2 are kinematic cylinders (0.5 m tall), tracked by the LiDAR ray caster and posed every physics step.
* Inactive obstacles are parked below the floor.
* The ground-truth clearance covers walls, baffles and *active* obstacles; it drives rewards, safety violations and
  collisions.
* Obstacle draws come after the start/goal draws, so the static scene's random stream is unchanged.

**Unchanged:** observation (47), reward, actuator model, LiDAR configuration, control rate, PPO settings, and the static
scene's behaviour. The static scene's configs, scene generation, observations and policy outputs are identical to
before.

## 2. Validation

### Unit tests (`tests/unit/test_obstacle_motion.py`, CPU)

All pass, 22/22 unit tests in total.

* **Scalar transcription:** the reference `_advance_random_ping_pong` loop, fed the same target and pause draws,
  matches the vectorized trajectories to under 1e-5 m, with identical reversal timing, over 120 s × 8 envs × 2
  obstacles.
* **Over 150 s × 256 envs:**
  * reversal points inside the regions;
  * still periods within 4.6–5.0 s (step-quantized);
  * constant episode speed while moving.
* **Randomization:** speed and path-y ranges and the gap constraint hold; s0 and direction are uniform.
* **Resets:** a partial reset leaves the other envs bit-identical.
* **Presence:** fixed sets (1,), (2,) and (1, 2) work, and inactive obstacles have r = 0 and don't move. Presence DR
  gives {OB1}, {OB2}, {OB1, OB2} each ≈ 1/3, never empty.

### Simulator validation (`tests/sim/dynamic_scene_validation.py`, 1024 envs, presence + speed + y DR)

**20/20 checks pass** (`results/dynamic_scene_validation.json`, figure `results/dynamic_scene_trajectories.png`).

| Check | Result |
|---|---|
| Structure | 10 fixed walls, no baffles, 2 kinematic obstacle bodies |
| Bodies vs motion model (100 s) | max 5.4 µm; inactive bodies at z = −1 m, active at 0.25 m |
| Trajectories (100 s, 474 reversals) | on the episode segment; ‖Δs‖ ≤ v·dt (max ratio 1.00006); forward reversals ≥ 0.750 L, backward ≤ 0.245 L; still periods 4.4–4.8 s at 0.2 s resolution (pauses 4.6–5.0 s) |
| LiDAR vs analytic oracle | walls: p99.9 0.04 mm. Obstacles (42k rays): p50 0.75 mm, p99 4.5 mm; 0.30 % of rays differ by > 1 cm, all near-tangent rays passing within 1.0 mm of the circle edge (32-facet cylinder mesh vs circle: 0.82 mm, plus < 1 physics step of motion). Inactive obstacles: invisible (19k rays that they would block match the active-only oracle, p99 1.7 mm) |
| Independent resets | the other envs' obstacle states and bodies are unchanged; the reset envs are redrawn and their bodies placed at the new state |
| Safety violation by an approaching OB1 | 671/671 envs, all `safety_violation`, 1.4–2.6 s after start; clearance at termination 0.035–0.050 m (one control step of motion below the 0.05 m margin); nearest object = obstacle; envs with OB1 inactive: 0 terminations |
| Physical contact (no termination) | 697/697: the first contact-sensor force > 1 N occurs at an analytic clearance of −12.6 mm in every env (= footprint radius 0.177 m − TB4 base radius 0.164 m, the robot's rear); inactive OB1: no contact |
| Presence DR | {OB1} 0.356, {OB2} 0.320, {OB1, OB2} 0.323, none 0 |

### Static scene preserved

* Unit tests 22/22.
* Golden snapshot vs `pre`: scene generation, observations and policy actions identical. The only config difference
  is the 6 new fields, at their static defaults.
* Env validation 31/31.

### Throughput (4,096 envs, full RL step with policy inference, GPU 5, back to back)

| Scene | Env steps/s | Setup |
|---|---|---|
| `corridor_static_random` | 21.2k (unchanged) | 154 s |
| `corridor_dynamic_chase_random`, OB1 | 27.8k | 106 s |
| `corridor_dynamic_chase_random`, OB1 + OB2 | 27.2k | 106 s |

The dynamic scene is faster: it has 2 tracked cylinders instead of 4 per-env-scaled baffle meshes.

### Short navigation runs (pipeline check only, 30 iterations, 4,096 envs, seed 42)

Run through `Navigation/scripts/train.py --config <ID>.yaml --set agent.max_iterations=30 --run_tag pipeline_check`,
then moved to `tests/sim/results/m6_pipeline_check/` (git-ignored) so the experiment folders stay empty for the real
runs. Summaries are in `results/`.

| | Return it 0 → 29 | Goal rate it 29 | Throughput | `ref_chase_c10` eval of model_29 |
|---|---|---|---|---|
| NAV02 (OB1) | −29.6 → 73.2 | 72 % | ≈ 27k steps/s | 78/100 (22 timeouts) |
| NAV03 (OB1 + OB2) | −28.5 → 20.9 | 2 % | ≈ 25k steps/s | 94/100 (4 safety, 2 timeouts) |

Training, logging, resolved configs, evaluation (new protocol `ref_chase_c10`, per-episode obstacle settings in
`episodes.csv`) and videos with the moving obstacles all work. These are pipeline checks, not results.

## 3. Limitations

* **OB3** (`triggered_chase`) is not implemented. The reference ST04/ST05 experiments use OB1 and OB1 + OB2 only.
* **No radius randomization.** `obstacle_radius_dr` raises an error: the ray caster fixes mesh sizes at start-up. The
  radius is 0.17 m, and the reference configs have radius DR off.
* **Not bit-identical to the reference's random streams.** Obstacle draws use the scene's torch generator, so
  distributions and motion logic match the reference but individual episodes don't.
* **The reference's `lidar_hz_dr`** (native LiDAR timing) is not part of our sensor model; the LiDAR configuration was
  left unchanged, as requested.
* **Cylinder tessellation:** 32 facets, so near-tangent rays (< 1 mm from the edge) can miss. This affects about
  0.3 % of the rays that hit an obstacle, as for the M2 cylinders.
* **Obstacles are kinematic:** they push the robot on contact. With the default 0.05 m safety margin, episodes end
  before contact.

## 4. Follow-up: reproducible per-episode randomness, 70 s evaluation

* **Problem:** obstacle reversal draws came from one generator shared by all envs, so one env's reset shifted every
  other env's obstacle motion. LiDAR noise was unseeded. Evaluations were therefore not reproducible and different
  policies did not meet the same obstacles.
* **Fix (`st_nav/crng.py`, counter-based random numbers):**
  * episode seed = key(env seed, env index, episode number);
  * obstacle draws (presence, speed, path y, phase and the pre-generated table of reversal targets / pauses) use the
    episode's obstacle seed;
  * LiDAR noise of beam b acquired at physics tick t = uniform(episode noise seed, t, b), so the same measurement
    always gets the same noise, as adaptive sensing needs.
  * Distributions, motion model and dynamics are unchanged. No synchronization in the per-step update.
* **Checks:**
  * unit tests: same results as the reference loop; distributions unchanged; bit-identical env trajectories while 61
    resets happen in other envs; same seeds reproduce; keyed noise reproducible and batch-independent;
  * dynamic scene validation 20/20; static env validation 31/31;
  * the same checkpoint evaluated twice: obstacles, noise, outcomes and episode lengths identical in 100/100 episodes;
  * NAV03 `model_400` vs `model_500`: identical obstacle trajectories over the common duration and identical noise
    draws.
* **Throughput (4,096 envs):** dynamic 26.7k (OB1 + OB2) / 27.2k (OB1), static 20.7k env steps/s, within about 2 %
  of before.
* **Evaluation timeout:** 70 s (= training). NAV01 ×2, NAV02 and NAV03 were re-evaluated (98 %, 100 %, 100 %, 97 %).
