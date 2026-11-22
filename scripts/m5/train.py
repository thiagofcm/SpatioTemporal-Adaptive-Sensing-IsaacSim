"""M5: PPO training (RSL-RL OnPolicyRunner) on STNav-TB4-Corridor-Static-v0. Mirrors Isaac Lab's rsl_rl/train.py.

    isaacrun -g 5 -- python scripts/m5/train.py --headless --num_envs 4096 --run_name m5_initial
Logs/checkpoints: results/m5/runs/<date>_<run_name>/ (tensorboard events, model_*.pt, params/env.yaml, agent.yaml)
"""
import argparse
import os
import time
from datetime import datetime

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--num_envs", type=int, default=4096)
p.add_argument("--max_iterations", type=int, default=None)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--run_name", default="m5")
p.add_argument("--log_root", default=os.path.join(os.environ.get("ST_ROOT", "."), "results", "m5", "runs"))
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app

from importlib.metadata import version  # noqa: E402

import gymnasium as gym  # noqa: E402

import st_nav.tasks.nav2d  # noqa: E402,F401
from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from st_nav.tasks.nav2d.agents.rsl_rl_ppo_cfg import CorridorNavPPORunnerCfg  # noqa: E402
from st_nav.tasks.nav2d.corridor_env_cfg import CorridorNavEnvCfg  # noqa: E402

env_cfg = CorridorNavEnvCfg()
agent = CorridorNavPPORunnerCfg()
if args.max_iterations is not None:
    agent.max_iterations = args.max_iterations
agent.seed = args.seed
agent.device = args.device
agent = handle_deprecated_rsl_rl_cfg(agent, version("rsl-rl-lib"))
env_cfg.scene.num_envs = args.num_envs
env_cfg.seed = agent.seed
env_cfg.sim.device = args.device
log_dir = os.path.abspath(os.path.join(args.log_root, datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{args.run_name}"))
env_cfg.log_dir = log_dir
print(f"[M5] log_dir {log_dir}", flush=True)
t0 = time.time()
env = gym.make("STNav-TB4-Corridor-Static-v0", cfg=env_cfg)
env = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
runner = OnPolicyRunner(env, agent.to_dict(), log_dir=log_dir, device=agent.device)
dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent)
print(f"[M5] setup {time.time() - t0:.0f} s; starting {agent.max_iterations} iterations", flush=True)
t1 = time.time()
runner.learn(num_learning_iterations=agent.max_iterations, init_at_random_ep_len=True)
print(f"[M5] training time {time.time() - t1:.0f} s", flush=True)
os._exit(0)
