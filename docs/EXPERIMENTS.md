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
