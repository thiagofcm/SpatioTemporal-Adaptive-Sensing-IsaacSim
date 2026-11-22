"""M4 integration + throughput test of the complete RL environment through the RSL-RL wrapper.

A randomly initialized RSL-RL actor-critic (inference only: no learning, nothing saved) produces the actions, so the
loop matches a PPO rollout: policy forward -> env.step (20 physics steps, command model, LiDAR with noise, rewards,
terminations, partial resets, observations). Reports throughput, outcome counts, reward-term means and stability.

    isaacrun -g auto -- python tools/benchmark_env.py --headless --num_envs 16384
"""
import argparse
import json
import os
import subprocess
import sys
import time

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--num_envs", type=int, default=4096)
p.add_argument("--steps", type=int, default=200)
p.add_argument("--out", default=os.path.join(os.environ.get("ST_ROOT", "."), "tests", "sim", "results", "benchmark_env.jsonl"))
p.add_argument("--set", action="append", default=[], metavar="env.PATH=VALUE",
               help="env cfg override, e.g. --set env.scene_name=corridor_dynamic_chase_random --set env.active_obstacles=[1,2]")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tools import experiments as X  # noqa: E402

sets = X.parse_set(args.set)
t_launch = time.time()
app = AppLauncher(args).app

from importlib.metadata import version  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import st_nav.tasks.nav2d  # noqa: E402,F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from st_nav.tasks.nav2d.agents.rsl_rl_ppo_cfg import CorridorNavPPORunnerCfg  # noqa: E402
from st_nav.tasks.nav2d.corridor_env import OUTCOME_NAMES  # noqa: E402
from st_nav.tasks.nav2d.corridor_env_cfg import CorridorNavEnvCfg  # noqa: E402

cfg = CorridorNavEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.sim.device = args.device
cfg.seed = 11
X.apply_overrides(cfg, sets["env"], "env")
t0 = time.time()
genv = gym.make("STNav-TB4-Corridor-Static-v0", cfg=cfg)
agent = handle_deprecated_rsl_rl_cfg(CorridorNavPPORunnerCfg(), version("rsl-rl-lib"))
agent.device = args.device
env = RslRlVecEnvWrapper(genv, clip_actions=agent.clip_actions)
runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=args.device)
policy = runner.get_inference_policy(device=args.device)
t_setup = time.time() - t0
N = genv.unwrapped.num_envs
obs = env.get_observations()
outcomes = {name: 0 for name in OUTCOME_NAMES.values()}
term_sums = {}
bad = 0
for _ in range(10):
    with torch.inference_mode():
        obs, rew, dones, extras = env.step(policy(obs))
torch.cuda.synchronize()
t1 = time.time()
for k in range(args.steps):
    with torch.inference_mode():
        obs, rew, dones, extras = env.step(policy(obs))
    bad += int((~torch.isfinite(obs["policy"])).sum() + (~torch.isfinite(rew)).sum())
    d = dones.bool()
    if d.any():
        oc = extras["outcome"][d]
        for code, name in OUTCOME_NAMES.items():
            outcomes[name] += int((oc == code).sum())
    for kk, v in genv.unwrapped.last_reward_terms.items():
        term_sums[kk] = term_sums.get(kk, 0.0) + float(v.mean())
torch.cuda.synchronize()
el = time.time() - t1
mem = None
out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                     capture_output=True, text=True).stdout
for line in out.strip().splitlines():
    pid, m = (x.strip() for x in line.split(","))
    if int(pid) == os.getpid():
        mem = int(m)
res = {"num_envs": N, "steps": args.steps, "control_steps_per_s": args.steps * N / el, "wall_s": el,
       "setup_s": t_setup, "episodes_ended": sum(outcomes.values()), "outcomes": outcomes,
       "mean_reward_terms_per_step": {k: v / args.steps for k, v in term_sums.items()}, "non_finite": bad,
       "gpu_mem_process_mib": mem, "rsl_rl_version": version("rsl-rl-lib"), "lidar_noise": cfg.lidar_noise, "scene": cfg.scene_name, "set": args.set}
print("[BENCH]", json.dumps(res), flush=True)
os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, "a") as f:
    f.write(json.dumps(res) + "\n")
os._exit(0)
