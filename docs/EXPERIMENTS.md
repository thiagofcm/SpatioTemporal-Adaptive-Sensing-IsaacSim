# Experiments: conventions and registry

This is the entry point for every experiment: what was run, why, and where its files are. The conventions follow the
reference simplified environment (`~/adaptive_sensor_policy`, branch `spatiotemporal`, `docs/EXPERIMENTS.md`). Paths
and IDs are defined in one place, [tools/experiments.py](../tools/experiments.py), which the training and evaluation
scripts import.

## Registry

| ID | Line | Status | Question | Config |
|---|---|---|---|---|
| [NAV01_static_initial](../Navigation/eval/NAV01_static_initial/README.md) | Navigation | done | Can PPO learn collision-free navigation in the randomized static corridor (2D LiDAR, start-pose DR) at 4,096 envs? | [config](../Navigation/configs/NAV01_static_initial.yaml) |
| [NAV02_chase-ob1_dr](../Navigation/eval/NAV02_chase-ob1_dr/README.md) | Navigation | evaluating | Can PPO navigate the dynamic corridor with one moving obstacle (OB1; speed / path-y DR)? | [config](../Navigation/configs/NAV02_chase-ob1_dr.yaml) |
| [NAV03_chase-ob1ob2_dr](../Navigation/eval/NAV03_chase-ob1ob2_dr/README.md) | Navigation | evaluating | Can PPO navigate the dynamic corridor with two moving obstacles (OB1 + OB2; speed / path-y DR)? | [config](../Navigation/configs/NAV03_chase-ob1ob2_dr.yaml) |
| [NAV04_static_lidar3d-r4_cnn](../Navigation/eval_3d/NAV04_static_lidar3d-r4_cnn/README.md) | Navigation (3D) | prepared | Can PPO with a CNN on the fixed 3D LiDAR range image (R4 mount, 4 Hz) learn the NAV01 task? | [config](../Navigation/configs_3d/NAV04_static_lidar3d-r4_cnn.yaml) |
| [NAV05_chase-ob1ob2_lidar3d-r4_cnn-from-nav04](../Navigation/eval_3d/NAV05_chase-ob1ob2_lidar3d-r4_cnn-from-nav04/README.md) | Navigation (3D) | prepared | Warm-started from NAV04 model_499, can the 3D-LiDAR CNN policy learn the NAV03 task (OB1 + OB2 moving)? | [config](../Navigation/configs_3d/NAV05_chase-ob1ob2_lidar3d-r4_cnn-from-nav04.yaml) |
| [TMP01_chase-ob1ob2_fixed-rate-nav05](../AdaptiveSensing/eval/temporal/TMP01_chase-ob1ob2_fixed-rate-nav05/README.md) | AdaptiveSensing (temporal) | done | How does the frozen NAV05 policy degrade when the 4 Hz LiDAR is gated to 4 / 2 / 1 / 0.5 / 0.2 Hz (real-robot scan gate)? | none (`evaluate_fixed_rate.py --fixed <hz>`) |
| [TMP02_chase-ob1ob2_adaptive-nav05](../AdaptiveSensing/eval/temporal/TMP02_chase-ob1ob2_adaptive-nav05/README.md) | AdaptiveSensing (temporal) | implemented, not trained | Can a recurrent sensing policy choose the LiDAR rate (0.2-4 Hz) of the frozen NAV05 policy to save frames while keeping navigation performance? | `AdaptiveSensing/configs/temporal/TMP02_chase-ob1ob2_adaptive-nav05.yaml` |
| [STS01_chase-ob1ob2_fixed-sections-nav05](../AdaptiveSensing/eval/spatiotemporal/STS01_chase-ob1ob2_fixed-sections-nav05/README.md) | AdaptiveSensing (spatiotemporal) | done | How does the frozen NAV05 policy behave when the left / center / right sections of the 3D range image are gated independently at constant rates (all 125 combinations of 0.2-4 Hz)? | none (`evaluate_fixed_sections.py --sections <fL> <fC> <fR>`) |

**3D-LiDAR experiments (M8)** use their own scripts and roots, separate from the 2D pipeline: `Navigation/scripts/train_3d.py`,
`evaluate_3d.py`, configs in `Navigation/configs_3d/`, runs in `Navigation/runs_3d/<ID>/`, evaluations in
`Navigation/eval_3d/<ID>/` (`tools/experiments.py: paths_3d`). Task `STNav-TB4-Corridor-Lidar3D-v0` (same env class,
`obs_layout: image`), agent `CorridorNavCNNPPORunnerCfg` (`st_nav/rl/lidar_cnn.py`).
`train_3d.py` can start from a checkpoint: config key `init_from: {checkpoint: <path>, mode: weights | resume}` or
`--init_from <checkpoint> [--resume]`. `weights` (warm start) loads the actor and critic (with normalizer statistics
and action std) and starts a new run at iteration 0 with a fresh optimizer; `resume` also restores the optimizer, the
iteration and the adaptive learning rate (`agent.max_iterations` = additional iterations). The network must match.

Training curves: [TENSORBOARD.md](TENSORBOARD.md) explains every TensorBoard scalar.

## Experiment ID

```
<PREFIX><NN>_<scene>_<short-desc>          e.g. NAV01_static_initial, NAV02_chase-ob1_dr, STS01_static_lidar-dr
```

| Prefix | Research line | Folder |
|---|---|---|
| `NAV` | navigation policies | `Navigation/` |
| `TMP` | temporal adaptive sensing | `AdaptiveSensing/{configs,runs,eval}/temporal/` |
| `SPA` | spatial adaptive sensing | `AdaptiveSensing/{configs,runs,eval}/spatial/` |
| `STS` | spatiotemporal adaptive sensing | `AdaptiveSensing/{configs,runs,eval}/spatiotemporal/` |

* `NN` is the next free number of the prefix. It is never reused; a retired experiment keeps its number and is
  marked `superseded by <ID>`.
* `<scene>` is `static` (corridor_static_random) or `chase-<obstacles>` (corridor_dynamic_chase_random, e.g.
  `chase-ob1`, `chase-ob1ob2`), as in the reference.
* `<short-desc>` says what makes the experiment different, in lowercase words joined by `-`.
* When an experiment corresponds to one in the reference repository (e.g. `ST04`), the card says so.
* The ID is the only name used for the experiment's config, runs folder, eval folder and registry row.

## Where everything lives (Navigation; AdaptiveSensing adds the `<mode>/` level)

```
Navigation/
├── configs/<ID>.yaml                     the experiment config (one file per experiment); _templates/ = starting points
├── runs/<ID>/<date>_<run_name>/          one folder per training run:
│     params/env.yaml, params/agent.yaml    fully resolved Isaac Lab configs
│     params/config.yaml                    the experiment YAML + --set overrides used
│     params/run_info.json                  command, git state of this repo, versions
│     model_*.pt, events.out.tfevents.*     checkpoints and TensorBoard (only listed checkpoints are in git)
└── eval/<ID>/README.md                   the experiment card (from docs/templates/experiment_card.md)
    eval/<ID>/<protocol>/summary.csv      one row per evaluated policy (same episodes: directly comparable)
    eval/<ID>/<protocol>/<run>_<ckpt>/    one folder per evaluated policy: run_config.json (resolved configuration),
                                          episodes.csv, summary.json, world_geometry.json, figures, videos and
      episode_NNNN/                       one folder per episode (format: tools/episode_data.py): metadata.json,
                                          world_geometry.json, trajectory.csv, steps.csv, lidar.npz, obstacles.npz
models/navigation/<ID>_<ckpt>.pt + .json  navigation policies promoted for reuse by AdaptiveSensing
```

* **`<protocol>`:** a named evaluation protocol defined in `Navigation/scripts/evaluate.py` (`PROTOCOLS`). It fixes
  episodes, seeds, goal and timeout, so all policies in one protocol folder are directly comparable.
  * `ref_static_c10`: the reference protocol. 100 episodes; obstacles from reference seeds 3000+i (held out from
    training); start poses from reference seeds 4000+i; goal (−2, −2); 70 s timeout (= training episode length;
    the reference evaluation used 80 s).
  * `ref_chase_c10`: dynamic scenes (reference ST04 / ST05 evaluation). Same start seeds, goal and timeout; the
    moving obstacles follow the run's scene settings.
  * **Reproducibility:** episode i has per-episode seeds derived from (`--seed`, i) (`st_nav/crng.py`, counter-based).
    They fix its whole obstacle realization and its LiDAR noise (per acquisition time and beam), independently of all
    other envs, so every policy evaluated with the same protocol and seed meets identical obstacle trajectories and
    identical noise for identical measurements. The seeds are recorded in each episode's `metadata.json`.
  * Overriding any protocol setting on the command line writes to `<protocol>_custom-<keys>`.
* **`<run_name>`:** `ppo_n<num_envs>_s<seed>` by default (or `run_name` from the config), plus `_<run_tag>` if one
  is given.

## Evaluation recordings and visualization

Every episode folder is self-contained: the visualization tools rebuild the scene from it alone (numpy + matplotlib;
no st_nav, Isaac Lab or PyTorch), as in the reference's AdaptiveFPS/eval/<...>/episode_NNNN/.

```bash
python tools/make_video.py <eval_dir>/episode_0025                 # one episode: map + timelines (MP4 in the folder)
python tools/make_video.py --eval <eval_dir> --episodes 0 1 25     # several episodes side by side
python tools/plot_episode.py <eval_dir>/episode_0025               # static figure of one episode
python tools/plot_eval.py <eval_dir> [--episodes 0 1 25]           # whole evaluation (or a selection)
```

3D LiDAR episodes (`lidar.npz` per-scan layout, schema 1.1) are drawn as the scan's 3D returns projected top-down
(coloured by height, floor returns grey); single-episode videos and figures add the range image.
`tools/make_pov_video.py` (3D only) renders the LiDAR's point of view: first-person views from the optical centre with
the returns coloured by ray class (mount-study colours: floor, wall, static obstacle, moving obstacle, escaping ray) and
by range, plus a top-down map, the range image the policy received and the ray-class image.

```bash
python tools/make_pov_video.py <eval_dir>/episode_0005                  # -> episode_0005/episode_0005_pov.mp4
python tools/make_pov_video.py --eval <eval_dir> --episodes 5 7 42      # one video per episode
```

`tools/make_cloud_video.py` (3D only) draws the point cloud actually captured (observed ranges rebuilt as
origin + range x direction), coloured by range, in the robot frame (camera behind the sensor, no scene geometry), with a
miniature map. `--depth` adds the depth image the policy received; `--scene` shows instead a world-frame 3D view with
the walls, robot and trail (`--camera follow|fixed`); `--accumulate N` also draws the N previous scans (faded).

```bash
python tools/make_cloud_video.py <eval_dir>/episode_0005                # -> episode_0005/episode_0005_cloud.mp4
python tools/make_cloud_video.py <eval_dir>/episode_0005 --depth --accumulate 8
python tools/make_cloud_video.py <eval_dir>/episode_0005 --scene        # -> episode_0005/episode_0005_cloud_scene.mp4
```

`Navigation/scripts/replay_episode_isaac.py` (needs Isaac Sim; tested on 3D episodes, 2D ones have the same pose files) renders a recorded episode with RTX
cameras (chase + overhead): the scene is rebuilt from the episode folder and the TurtleBot4 USD (physics removed) and the
moving obstacles are placed at their recorded poses; no policy, no physics. About 25 s for a 40 s episode.

```bash
bin/isaacrun -g <gpu> -- python Navigation/scripts/replay_episode_isaac.py --headless --enable_cameras <eval_dir>/episode_0000 --compare
# -> episode_0000_isaac.mp4; --compare adds episode_0000_recreation_2x.mp4 (make_video.py, same speed) and
#    episode_0000_isaac_vs_recreation.mp4 (side by side, frames in sync)
```

## Configs

Defaults live in the Isaac Lab configclasses registered with the config's Gym `task` (for navigation:
`CorridorNavEnvCfg`, `CorridorNavPPORunnerCfg`). The YAML only overrides them:

```yaml
exp_id: NAV01_static_initial
task: STNav-TB4-Corridor-Static-v0
run_name: null                 # default ppo_n<num_envs>_s<seed>
env:   {scene.num_envs: 4096, start_pose_dr: true, rewards.goal_reward: 30.0}    # dotted paths (nested maps also work)
agent: {seed: 42, max_iterations: 1500, algorithm.learning_rate: 3.0e-4}
sweep: {}                      # reserved: {agent.seed: [42, 43, 44]} -> one run per combination
```

* Sensor: `env.sensor: lidar2d` (default; 40-beam 2D LiDAR, `lidar_period_s`) or `lidar3d` (fixed 3D LiDAR, all its
  settings under `env.lidar3d.*`: FoV, rows / cols, placement, ray order or `ray_table`, mount position and
  roll/pitch/yaw, range, noise type / sigma, period, floor target). Templates with every field:
  `Navigation/configs/_templates/{static,chase}_lidar3d.yaml`. `env.lidar_noise: false` disables either sensor's
  noise. Object heights: `env.wall_height`, `env.baffle_height`, `env.obstacle_height`.
* Precedence: configclass defaults < YAML < `--set <env|agent>.<path>=<value>` on the command line.
* Unknown keys are errors.
* Each run records the resolved configs in `params/`.
* Future sweeps need no structural change: a sweep launcher calls `train.py --config <ID>.yaml --set ...
  --run_tag ...` once per combination.

## Workflow for a new experiment

1. **Create it:**
   * copy a template from `<line>/configs/_templates/` to `<line>/configs/<ID>.yaml` and set `exp_id` (next free
     number);
   * create `<line>/eval/<ID>/README.md` from [templates/experiment_card.md](templates/experiment_card.md);
   * add a row to the registry above.
2. **Fill in the card:** the question, hypothesis and what differs from the previous experiment, before training.
3. **Train:** `bin/isaacrun -g <gpu> -- python Navigation/scripts/train.py --headless --config Navigation/configs/<ID>.yaml`.
4. **Evaluate:** `bin/isaacrun -g <gpu> -- python Navigation/scripts/evaluate.py --headless --checkpoint Navigation/runs/<ID>/<run>/model_<k>.pt`.
5. **Write up:** fill in the card's run and evaluation tables, results and conclusion, and update the status here.
6. **Version the checkpoints:** list the evaluated checkpoints in `.gitignore` with a `!` line. Promote a policy for
   AdaptiveSensing by copying it to `models/navigation/<ID>_<ckpt>.pt` with a `.json` description.

**Rules:**
* **Never edit a config that has training runs:** create a new experiment, and say in the card what differs.
  Replicates (other seeds, a re-run) are new runs of the same experiment, distinguished by `--set` and `--run_tag`.
* **Re-evaluating** into an existing policy folder needs `--overwrite`. Say why in the card.
* **Ad-hoc checks** that are not part of an experiment use `--out_dir` outside `Navigation/eval/` (e.g.
  `tests/sim/results/`).
