"""Navigation: PPO training (RSL-RL OnPolicyRunner) from an experiment config. Mirrors Isaac Lab's rsl_rl/train.py.

    isaacrun -g 5 -- python Navigation/scripts/train.py --headless --config Navigation/configs/NAV01_static_initial.yaml
        [--set agent.max_iterations=201 --set env.scene.num_envs=1024 ...] [--run_tag <tag>]

Defaults come from the configclasses registered with the config's Gym task; the YAML `env:` / `agent:` sections and
then `--set` override them (tools/experiments.py, docs/EXPERIMENTS.md). Output:
    Navigation/runs/<ID>/<date>_<run_name>[_<tag>]/  TensorBoard events, model_*.pt,
        params/env.yaml, params/agent.yaml   fully resolved configs (Isaac Lab dump)
        params/config.yaml                   the experiment YAML + the --set overrides applied
        params/run_info.json                 command, git state of this repo and of the reference repo, versions
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
p.add_argument("--config", required=True, help="experiment YAML (Navigation/configs/<ID>.yaml)")
p.add_argument("--set", action="append", default=[], metavar="SECTION.PATH=VALUE",
               help="override, e.g. --set agent.seed=43 --set env.scene.num_envs=1024 (applied after the YAML)")
p.add_argument("--run_tag", default=None, help="suffix of the run folder name (e.g. a sweep member or a check)")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
exp = X.load_config(args.config)
if exp["sweep"]:
    raise SystemExit("this config has a sweep: section; run one member per call with --set (sweep.py not implemented)")
sets = X.parse_set(args.set)
app = AppLauncher(args).app

from importlib.metadata import version  # noqa: E402

import gymnasium as gym  # noqa: E402

import st_nav.tasks.nav2d  # noqa: E402,F401  (registers the navigation Gym tasks)
from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

exp_id, task = exp["exp_id"], exp["task"]
env_cfg, agent = X.task_cfgs(task)
for section, cfg_obj in (("env", env_cfg), ("agent", agent)):
    X.apply_overrides(cfg_obj, exp[section], section)
    X.apply_overrides(cfg_obj, sets[section], section)
agent.device = args.device
env_cfg.seed = agent.seed                       # as Isaac Lab's train.py
env_cfg.sim.device = args.device
agent = handle_deprecated_rsl_rl_cfg(agent, version("rsl-rl-lib"))
run_name = exp.get("run_name") or f"ppo_n{env_cfg.scene.num_envs}_s{agent.seed}"
run_name += f"_{args.run_tag}" if args.run_tag else ""
log_dir = str(X.paths(exp_id)["runs"] / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{run_name}")
env_cfg.log_dir = log_dir
print(f"[TRAIN] {exp_id} task {task} -> {log_dir}", flush=True)
t0 = time.time()
env = gym.make(task, cfg=env_cfg)
env = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
runner = OnPolicyRunner(env, agent.to_dict(), log_dir=log_dir, device=agent.device)
dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent)
with open(os.path.join(log_dir, "params", "config.yaml"), "w") as f:
    X.yaml.safe_dump({"source": os.path.relpath(os.path.abspath(args.config), X.REPO_ROOT), **exp,
                      "set": {s: v for s, v in sets.items() if v}}, f, sort_keys=False)
with open(os.path.join(log_dir, "params", "run_info.json"), "w") as f:
    json.dump({"exp_id": exp_id, "task": task, "run_name": run_name, "command": sys.argv, "host": socket.gethostname(),
               "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "repo": X.git_state(X.REPO_ROOT),
               "reference_repo": X.git_state(X.REFERENCE_ROOT),        # obstacle-size bank is sampled from it
               "versions": {k: version(k) for k in ("rsl-rl-lib", "torch", "isaaclab")}}, f, indent=1)
print(f"[TRAIN] setup {time.time() - t0:.0f} s; starting {agent.max_iterations} iterations", flush=True)
t1 = time.time()
runner.learn(num_learning_iterations=agent.max_iterations, init_at_random_ep_len=True)
print(f"[TRAIN] training time {time.time() - t1:.0f} s", flush=True)
os._exit(0)
