"""M5 pre-check: start-pose randomization (default ON) in the RL env. Verifies the 5 reference points, yaw in
pi +- 10 deg (uniform, wrapped), per-reset resampling, robot placed exactly at the sampled pose, safe initial clearance.

    isaacrun -g 5 -- python scripts/m5/m5_verify_start_dr.py --headless --num_envs 1024
"""
import argparse
import json
import math
import os

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--num_envs", type=int, default=1024)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import st_nav.tasks.nav2d  # noqa: E402,F401
from st_nav.scenes import corridor_static_random as S  # noqa: E402
from st_nav.tasks.nav2d.corridor_env_cfg import CorridorNavEnvCfg  # noqa: E402

cfg = CorridorNavEnvCfg()                 # defaults as used for training
cfg.scene.num_envs = args.num_envs
cfg.sim.device = args.device
cfg.seed = 3
genv = gym.make("STNav-TB4-Corridor-Static-v0", cfg=cfg)
_IM = torch.inference_mode()
_IM.__enter__()
u = genv.unwrapped
corr, N, dev = u.corr, u.num_envs, u.device
pts = torch.tensor(S.START_POSE_LIST, device=dev)
starts, place_err, clear0 = [], [], []
first_start = None
per_env_starts = [[] for _ in range(8)]


def collect(ids):
    st = corr.start[ids].clone()
    starts.append(st)
    pp, yaw = corr.robot_pose2d()
    place_err.append(torch.cat([(pp[ids] - st[:, :2]).norm(dim=1),
                                (torch.remainder(yaw[ids] - st[:, 2] + math.pi, 2 * math.pi) - math.pi).abs()]))
    clear0.append(corr.footprint_clearance()[ids])
    for e in range(8):
        if bool((ids == e).any()):
            per_env_starts[e].append(tuple(round(x, 3) for x in corr.start[e].tolist()))


genv.reset()
collect(torch.arange(N, device=dev))
# drive with random actions; collect the start pose of every env that was reset (state right after reset)
orig_reset = u._reset_idx


def hook(env_ids):
    orig_reset(env_ids)
    ids = env_ids if env_ids is not None else torch.arange(N, device=dev)
    collect(ids)


u._reset_idx = hook
for k in range(150):
    genv.step(torch.clamp(torch.randn(N, 2, device=dev) * 2, -1, 1))
ST = torch.cat(starts)
idx = torch.cdist(ST[:, :2], pts).argmin(1)
dpos = torch.cdist(ST[:, :2], pts).min(1).values
dyaw = torch.remainder(ST[:, 2] - S.START_YAW_CENTER + math.pi, 2 * math.pi) - math.pi
PE = torch.cat(place_err)
C0 = torch.cat(clear0)
res = {
    "cfg_start_pose_dr": cfg.start_pose_dr, "resets": int(ST.shape[0]),
    "position_counts": {str(S.START_POSE_LIST[i]): int((idx == i).sum()) for i in range(len(pts))},
    "max_dist_to_nearest_reference_point_m": float(dpos.max()),
    "yaw_minus_pi_deg_min_max_mean_std": [math.degrees(float(dyaw.min())), math.degrees(float(dyaw.max())),
                                          math.degrees(float(dyaw.mean())), math.degrees(float(dyaw.std()))],
    "uniform_std_expected_deg": 10 / math.sqrt(3),
    "max_placement_error_pos_m_yaw_rad": float(PE.max()),
    "initial_clearance_min_m": float(C0.min()), "safety_margin": cfg.safety_margin,
    "per_env_start_sequences_env0_3": per_env_starts[:4],
}
res["pass"] = bool(cfg.start_pose_dr and dpos.max() < 1e-5 and all(v > 0 for v in res["position_counts"].values())
                   and dyaw.abs().max() <= math.radians(10) + 1e-5 and PE.max() < 1e-3 and C0.min() > cfg.safety_margin
                   and any(len(set(s)) > 1 for s in per_env_starts))
os.makedirs("results/m5", exist_ok=True)
json.dump(res, open("results/m5/start_dr_check.json", "w"), indent=1)
print("[START-DR]", json.dumps(res), flush=True)
os._exit(0)
