"""Mount study, input 1: base_link attitude (pitch, roll) and height while a trained policy drives (the LiDAR is rigidly
attached to base_link, so body pitch tilts the whole ray fan). Physics does not depend on the sensor (M7 L6), so the
2D policies are used.

    isaacrun -g <gpu> -- python infrastructure/M7_lidar3d/mount_study/scripts/measure_base_attitude.py --headless \
        --checkpoint Navigation/runs/<ID>/<run>/model_<N>.pt --tag nav03
Writes results/base_attitude_<tag>.npz (+ .json percentiles). Angles: pitch > 0 = nose DOWN (REP-103, as the mount).
"""
import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
from tools import experiments as X  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--tag", required=True)
p.add_argument("--num_envs", type=int, default=1024)
p.add_argument("--steps", type=int, default=350)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app

from importlib.metadata import version  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import st_nav.tasks.nav2d  # noqa: E402,F401
from isaaclab.utils.math import euler_xyz_from_quat  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from st_nav.tasks.nav2d.agents.rsl_rl_ppo_cfg import CorridorNavPPORunnerCfg  # noqa: E402
from st_nav.tasks.nav2d.corridor_env_cfg import CorridorNavEnvCfg  # noqa: E402

run_dir = os.path.dirname(os.path.abspath(args.checkpoint))
exp = X.yaml.safe_load(open(os.path.join(run_dir, "params", "config.yaml")))
cfg = CorridorNavEnvCfg()
X.apply_overrides(cfg, {k: v for k, v in X.flatten(exp.get("env") or {}).items() if k != "scene.num_envs"}, "env")
cfg.scene.num_envs, cfg.sim.device, cfg.seed = args.num_envs, args.device, 5
genv = gym.make("STNav-TB4-Corridor-v0", cfg=cfg)
agent = handle_deprecated_rsl_rl_cfg(CorridorNavPPORunnerCfg(), version("rsl-rl-lib"))
agent.device = args.device
env = RslRlVecEnvWrapper(genv, clip_actions=agent.clip_actions)
runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=args.device)
runner.load(args.checkpoint)
policy = runner.get_inference_policy(device=args.device)
u = genv.unwrapped
rd = u.corr.robot.data
pitch, roll, z, v = [], [], [], []
with torch.inference_mode():
    obs = env.get_observations()
    for k in range(args.steps):
        obs, _, _, _ = env.step(policy(obs))
        r, pt, _ = euler_xyz_from_quat(rd.root_link_quat_w)
        wrap = lambda a: torch.remainder(a + torch.pi, 2 * torch.pi) - torch.pi  # noqa: E731
        # isaaclab euler: rotation about +y; positive = nose DOWN for x forward / z up (right-handed), same as REP-103
        pitch.append(torch.rad2deg(wrap(pt)).cpu().numpy()); roll.append(torch.rad2deg(wrap(r)).cpu().numpy())
        z.append(rd.root_link_pos_w[:, 2].cpu().numpy()); v.append(rd.root_link_lin_vel_b[:, 0].cpu().numpy())
pitch, roll, z, v = (np.stack(a) for a in (pitch, roll, z, v))
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", f"base_attitude_{args.tag}")
np.savez_compressed(out + ".npz", pitch_deg=pitch, roll_deg=roll, base_z_m=z, v_mps=v)
q = [0.1, 1, 5, 50, 95, 99, 99.9]
stats = {"checkpoint": args.checkpoint, "samples": int(pitch.size),
         "pitch_deg_percentiles": dict(zip(map(str, q), np.percentile(pitch, q).round(4).tolist())),
         "roll_deg_percentiles": dict(zip(map(str, q), np.percentile(roll, q).round(4).tolist())),
         "base_z_m_percentiles": dict(zip(map(str, q), np.percentile(z, q).round(5).tolist())),
         "abs_pitch_max_deg": float(np.abs(pitch).max()), "abs_roll_max_deg": float(np.abs(roll).max())}
json.dump(stats, open(out + ".json", "w"), indent=1)
print(json.dumps(stats, indent=1), flush=True)
genv.close()
app.close()
