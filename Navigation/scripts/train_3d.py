"""Navigation with the 3D LiDAR: PPO training of the CNN policy (RSL-RL OnPolicyRunner) from a 3D experiment config.
The 2D pipeline (train.py, configs/, runs/) is not used or changed; this script differs from train.py only in the task
family (STNav-TB4-Corridor-Lidar3D-v0: range-image observations, CNN agent config) and the output root.

    isaacrun -g 5 -- python Navigation/scripts/train_3d.py --headless --config Navigation/configs_3d/NAV04_static_lidar3d-r4_cnn.yaml
        [--set agent.max_iterations=3 --set env.scene.num_envs=256 ...] [--run_tag <tag>]
        [--init_from <checkpoint> [--resume]]

Initialization from a checkpoint (config key `init_from: {checkpoint: <path>, mode: weights | resume}`, or the
command-line options, which override it):
    weights (warm start): actor and critic weights, including the observation-normalizer statistics and the action
        std; fresh optimizer, iteration counter from 0, learning rate from the config. The network must be identical
        (same ray grid, navigation features, CNN / MLP sizes): loading is strict.
    resume: also the Adam state, the iteration counter (logging continues at the checkpoint's iteration) and the
        adaptive learning rate reached (restored from the optimizer state; RSL-RL would otherwise restart at the
        config value). agent.max_iterations = number of ADDITIONAL iterations.
    Every loaded tensor is compared with the checkpoint before training; the source and mode are recorded in
    params/run_info.json (init_from) and params/config.yaml.

Output: Navigation/runs_3d/<ID>/<date>_<run_name>[_<tag>]/  TensorBoard events, model_*.pt,
    params/env.yaml, params/agent.yaml (resolved), params/config.yaml (YAML + --set), params/run_info.json,
    params/model.txt (actor / critic structure and parameter counts)
"""
import argparse
import json
import os
import socket
import sys
import time
from datetime import datetime

from isaaclab.app import AppLauncher

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tools import experiments as X  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--config", required=True, help="3D experiment YAML (Navigation/configs_3d/<ID>.yaml)")
p.add_argument("--set", action="append", default=[], metavar="SECTION.PATH=VALUE",
               help="override, e.g. --set agent.seed=43 --set env.scene.num_envs=1024 (applied after the YAML)")
p.add_argument("--run_tag", default=None, help="suffix of the run folder name (e.g. smoke)")
p.add_argument("--out_root", default=None, help="parent folder of the run folder (default Navigation/runs_3d/<ID>)")
p.add_argument("--init_from", default=None, metavar="CHECKPOINT",
               help="start from this checkpoint's weights (overrides the config's init_from)")
p.add_argument("--resume", action="store_true", help="with --init_from: also restore optimizer, iteration and learning rate")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
exp = X.load_config(args.config)
if exp["sweep"]:
    raise SystemExit("this config has a sweep: section; run one member per call with --set")
sets = X.parse_set(args.set)
init = dict(exp.get("init_from") or {})
if args.init_from:
    init = {"checkpoint": args.init_from, "mode": "resume" if args.resume else "weights"}
elif args.resume:
    raise SystemExit("--resume needs --init_from <checkpoint>")
if init:
    init.setdefault("mode", "weights")
    if init["mode"] not in ("weights", "resume") or not init.get("checkpoint"):
        raise SystemExit(f"init_from must be {{checkpoint: <path>, mode: weights | resume}}, got {init}")
    ck = init["checkpoint"]
    init["checkpoint_abs"] = os.path.abspath(ck if os.path.exists(ck) else os.path.join(X.REPO_ROOT, ck))  # cwd or repo
    if not os.path.isfile(init["checkpoint_abs"]):
        raise SystemExit(f"init_from checkpoint not found: {ck}")
app = AppLauncher(args).app

from importlib.metadata import version  # noqa: E402

import gymnasium as gym  # noqa: E402
import hashlib  # noqa: E402

import torch  # noqa: E402

import st_nav.tasks.nav2d  # noqa: E402,F401  (registers the navigation Gym tasks)
from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

exp_id, task = exp["exp_id"], exp["task"]
if "Lidar3D" not in task:
    raise SystemExit(f"train_3d.py is for the 3D-LiDAR tasks (STNav-TB4-Corridor-Lidar3D-v0), got {task}; use train.py")
env_cfg, agent = X.task_cfgs(task)
for section, cfg_obj in (("env", env_cfg), ("agent", agent)):
    X.apply_overrides(cfg_obj, exp[section], section)
    X.apply_overrides(cfg_obj, sets[section], section)
if env_cfg.sensor != "lidar3d" or env_cfg.obs_layout != "image":
    raise SystemExit("train_3d.py expects env.sensor = lidar3d and env.obs_layout = image")
agent.device = args.device
env_cfg.seed = agent.seed
env_cfg.sim.device = args.device
agent = handle_deprecated_rsl_rl_cfg(agent, version("rsl-rl-lib"))   # strips legacy model fields (rsl-rl 5)
run_name = exp.get("run_name") or f"ppo_cnn_n{env_cfg.scene.num_envs}_s{agent.seed}"
run_name += f"_{args.run_tag}" if args.run_tag else ""
root = args.out_root or str(X.paths_3d(exp_id)["runs"])
log_dir = os.path.join(root, f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{run_name}")
env_cfg.log_dir = log_dir
print(f"[TRAIN3D] {exp_id} task {task} -> {log_dir}", flush=True)
t0 = time.time()
env = gym.make(task, cfg=env_cfg)
env = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
runner = OnPolicyRunner(env, agent.to_dict(), log_dir=log_dir, device=agent.device)
if init:
    resume = init["mode"] == "resume"
    try:
        runner.load(init["checkpoint_abs"], load_cfg={"actor": True, "critic": True, "optimizer": resume,
                                                      "iteration": resume}, strict=True, map_location=agent.device)
    except RuntimeError as e:
        raise SystemExit(f"init_from: the checkpoint does not match this network (same ray grid, features and sizes "
                         f"needed):\n{e}")
    ck = torch.load(init["checkpoint_abs"], map_location=agent.device, weights_only=False)
    n_t, same = 0, True
    for key, m in (("actor_state_dict", runner.alg.actor), ("critic_state_dict", runner.alg.critic)):
        sd = m.state_dict()
        same &= sd.keys() == ck[key].keys()
        for k, v in ck[key].items():
            n_t += 1
            same &= bool(torch.equal(sd[k], v.to(sd[k].device)))
    if not same:
        raise SystemExit("init_from: loaded parameters differ from the checkpoint")
    if resume:
        runner.alg.learning_rate = float(runner.alg.optimizer.param_groups[0]["lr"])
    with open(init["checkpoint_abs"], "rb") as fh:
        init["sha256"] = hashlib.sha256(fh.read()).hexdigest()
    init.update(source_iteration=int(ck.get("iter", -1)), tensors_loaded=n_t, tensors_identical=True,
                start_iteration=int(runner.current_learning_iteration), learning_rate_start=float(runner.alg.learning_rate))
    print(f"[TRAIN3D] init_from ({init['mode']}): {init['checkpoint']} (iteration {init['source_iteration']}); "
          f"{n_t} tensors loaded, identical to the checkpoint; start iteration {init['start_iteration']}, "
          f"learning rate {init['learning_rate_start']:.3g}", flush=True)
dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent)
with open(os.path.join(log_dir, "params", "config.yaml"), "w") as f:
    X.yaml.safe_dump({"source": os.path.relpath(os.path.abspath(args.config), X.REPO_ROOT), **exp,
                      "init_from": {k: init[k] for k in ("checkpoint", "mode")} if init else None,
                      "set": {s: v for s, v in sets.items() if v}}, f, sort_keys=False)
with open(os.path.join(log_dir, "params", "model.txt"), "w") as f:
    for name, m in (("actor", runner.alg.actor), ("critic", runner.alg.critic)):
        f.write(f"{name}: {sum(p.numel() for p in m.parameters())} parameters\n{m}\n\n")
with open(os.path.join(log_dir, "params", "run_info.json"), "w") as f:
    json.dump({"exp_id": exp_id, "task": task, "run_name": run_name, "command": sys.argv, "host": socket.gethostname(),
               "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "repo": X.git_state(X.REPO_ROOT),
               "init_from": init or None,
               "versions": {k: version(k) for k in ("rsl-rl-lib", "torch", "isaaclab")}}, f, indent=1)
print(f"[TRAIN3D] setup {time.time() - t0:.0f} s; starting {agent.max_iterations} iterations", flush=True)
t1 = time.time()
runner.learn(num_learning_iterations=agent.max_iterations, init_at_random_ep_len=True)
print(f"[TRAIN3D] training time {time.time() - t1:.0f} s", flush=True)
os._exit(0)
