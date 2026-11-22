"""Golden snapshot of configuration, scene generation, reset observations and policy outputs (refactoring check).

    isaacrun -g 5 -- python tests/golden/capture_golden.py --headless --checkpoint <model_200.pt> --tag pre
Writes tests/golden/<tag>.npz + <tag>_cfg.json. Compare two snapshots with compare_golden.py (no simulator needed).
Policy outputs are computed on a FIXED observation batch: --obs_from <snapshot.npz> reuses the batch of an earlier
snapshot, so the network check is bit-exact even though the LiDAR noise generator is unseeded.
"""
import argparse
import json
import math
import os

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--tag", required=True)
p.add_argument("--obs_from", default=None)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app

from importlib.metadata import version  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import st_nav.tasks.nav2d  # noqa: E402,F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from st_nav.scenes import corridor_static_random as S  # noqa: E402
from st_nav.tasks.nav2d.agents.rsl_rl_ppo_cfg import CorridorNavPPORunnerCfg  # noqa: E402
from st_nav.tasks.nav2d.corridor_env_cfg import CorridorNavEnvCfg  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
cfg_dump = {"env_default": CorridorNavEnvCfg().to_dict(), "agent_default": CorridorNavPPORunnerCfg().to_dict()}


def ref_start(seed):
    rng = np.random.default_rng(seed)
    i = int(rng.integers(len(S.START_POSE_LIST)))
    dyaw = float(rng.uniform(-S.START_YAW_NOISE, S.START_YAW_NOISE))
    return (*S.START_POSE_LIST[i], (S.START_YAW_CENTER + dyaw + math.pi) % (2 * math.pi) - math.pi)


n = 100
cfg = CorridorNavEnvCfg()
cfg.scene.num_envs, cfg.sim.device, cfg.seed, cfg.episode_length_s = n, args.device, 42, 80.0
cfg.bank_seed, cfg.bank_size, cfg.bank_assignment, cfg.offsets_from_bank = 3000, n, "sequential", True
cfg.start_poses = tuple(ref_start(4000 + i) for i in range(n))
cfg.goals = ((-2.0, -2.0),)
genv = gym.make("STNav-TB4-Corridor-Static-v0", cfg=cfg)
agent = handle_deprecated_rsl_rl_cfg(CorridorNavPPORunnerCfg(), version("rsl-rl-lib"))
agent.device = args.device
env = RslRlVecEnvWrapper(genv, clip_actions=agent.clip_actions)
runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=args.device)
runner.load(args.checkpoint)
policy = runner.get_inference_policy(device=args.device)
u = genv.unwrapped
corr = u.corr
with torch.inference_mode():
    obs = env.get_observations()["policy"].clone()
    g = torch.Generator(device=args.device).manual_seed(0)
    rand_obs = torch.randn(4096, 47, generator=g, device=args.device) * 0.5 + 0.5
    if args.obs_from:
        old = np.load(args.obs_from)
        batch = torch.tensor(np.concatenate([old["obs_reset"], old["rand_obs"]]), device=args.device)
    else:
        batch = torch.cat([obs, rand_obs])
    from tensordict import TensorDict
    act = policy(TensorDict({"policy": batch}, batch_size=[batch.shape[0]]))
out = dict(obs_reset=obs.cpu().numpy(), scan_clean=u.scan_clean.cpu().numpy(), rand_obs=rand_obs.cpu().numpy(),
           policy_batch=batch.cpu().numpy(), actions=act.cpu().numpy(), start=corr.start.cpu().numpy(),
           goal=corr.goal.cpu().numpy(), walls=corr.walls.cpu().numpy(), z_len=corr.z_len.cpu().numpy(),
           z_thick=corr.z_thick.cpu().numpy(), z_off=corr.z_off.cpu().numpy(),
           robot_pose=torch.cat([corr.robot_pose2d()[0], corr.robot_pose2d()[1][:, None]], 1).cpu().numpy())
np.savez(os.path.join(HERE, f"{args.tag}.npz"), **out)
json.dump(cfg_dump, open(os.path.join(HERE, f"{args.tag}_cfg.json"), "w"), indent=1, default=str)
print(f"[GOLDEN] wrote {args.tag}: obs {obs.shape}, actions {tuple(act.shape)}", flush=True)
os._exit(0)
