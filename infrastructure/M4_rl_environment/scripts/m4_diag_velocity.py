"""M4 follow-up D3b: body rocking and reset transient behind the CoM-velocity excursions (per physics substep).

Records base_link roll/pitch, body-frame angular rates, link height, CoM vs link forward velocity, and the reset
spawn height vs the settled height. Fixed actions per group: (1,0) start-from-rest, (1,+-1) turn reversals, (0,0).

    isaacrun -g 5 -- python infrastructure/M4_rl_environment/scripts/m4_diag_velocity.py --headless --num_envs 256
"""
import argparse
import json
import os

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--num_envs", type=int, default=256)
p.add_argument("--out_dir", default=os.path.join(os.environ.get("ST_ROOT", "."), "infrastructure", "M4_rl_environment", "results", "diag"))
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import st_nav.tasks.nav2d  # noqa: E402,F401
from isaaclab.utils.math import euler_xyz_from_quat  # noqa: E402
from st_nav.tasks.nav2d.corridor_env_cfg import CorridorNavEnvCfg  # noqa: E402

cfg = CorridorNavEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.sim.device = args.device
cfg.seed = 11
genv = gym.make("STNav-TB4-Corridor-Static-v0", cfg=cfg)
_IM = torch.inference_mode()
_IM.__enter__()
u = genv.unwrapped
corr, rd, N, dev = u.corr, u.corr.robot.data, u.num_envs, u.device
G = N // 4
grp = torch.arange(N, device=dev) // G
names = ["v_com", "v_link", "roll", "pitch", "wx", "wy", "wz", "z", "since_reset", "cmd_v"]
rec = {n: [] for n in names}
_orig_apply = u._apply_action


def snap():
    r, pch, _ = euler_xyz_from_quat(rd.root_link_quat_w)
    wb = rd.root_link_ang_vel_b
    vals = [rd.root_com_lin_vel_b[:, 0], rd.root_link_lin_vel_b[:, 0], torch.atan2(torch.sin(r), torch.cos(r)),
            torch.atan2(torch.sin(pch), torch.cos(pch)), wb[:, 0], wb[:, 1], wb[:, 2],
            rd.root_link_pos_w[:, 2], u.episode_length_buf.float(), corr.cm.state[:, 0]]
    for n, v in zip(names, vals):
        rec[n].append(v.clone())


u._apply_action = lambda: (snap(), _orig_apply())
genv.reset()
z_spawn = float(rd.root_link_pos_w[:, 2].mean())
R = {"num_envs": N, "z_spawn_link_m": z_spawn, "default_root_z": float(rd.default_root_state[0, 2]),
     "com_offset_b": rd.body_com_pos_b[0, 0].tolist(), "groups": ["(0,0)", "(1,0)", "(1,+1)/(1,-1) alternating 1 s", "random"]}
for k in range(120):
    a = torch.zeros(N, 2, device=dev)
    a[grp == 1] = torch.tensor([1.0, 0.0], device=dev)
    a[grp == 2] = torch.tensor([1.0, 1.0 if (k // 5) % 2 == 0 else -1.0], device=dev)
    a[grp == 3] = torch.clamp(torch.randn(int((grp == 3).sum()), 2, device=dev) * 2, -1, 1)
    genv.step(a)
T = {n: torch.stack(v) for n, v in rec.items()}
diff = T["v_com"] - T["v_link"]
r_z = R["com_offset_b"][2]
pred = T["wy"] * r_z - T["wz"] * R["com_offset_b"][1]                    # (w x r)_x in the body frame
R["com_minus_link_vs_(w x r)_x_max_err"] = float((pred - diff).abs().max())
R["fraction_of_|com-link|_explained_by_pitch_rate"] = float(1 - ((T["wy"] * r_z - diff).abs().sum() / diff.abs().sum()))
first = T["since_reset"] == 0                                             # first control step of each episode
for g in range(4):
    m = grp == g
    ss = m[None] & (T["since_reset"] >= 3)
    R[f"group{g}"] = {
        "max_v_com": float(T["v_com"][:, m].max()), "max_v_link": float(T["v_link"][:, m].max()),
        "max_com_minus_link": float(diff[:, m].abs().max()),
        "max_abs_pitch_deg": float(T["pitch"][:, m].abs().max() * 57.2958), "max_abs_roll_deg": float(T["roll"][:, m].abs().max() * 57.2958),
        "max_abs_wy": float(T["wy"][:, m].abs().max()), "max_abs_wx": float(T["wx"][:, m].abs().max()),
        "steady(>=3 steps)_max_abs_pitch_deg": float((T["pitch"].abs() * ss).max() * 57.2958),
        "steady_max_com_minus_link": float((diff.abs() * ss).max()),
        "first_step_max_abs_v_link_(cmd=0)": float((T["v_link"].abs() * (first & m[None] & (T["cmd_v"].abs() < 1e-6))).max()),
        "first_step_max_abs_v_com_(cmd=0)": float((T["v_com"].abs() * (first & m[None] & (T["cmd_v"].abs() < 1e-6))).max()),
    }
zs = T["z"][first.any(1)] if first.any() else T["z"]
R["z_first_step_range"] = [float(T["z"][first].min()), float(T["z"][first].max())]
R["z_settled_mean"] = float(T["z"][T["since_reset"] >= 5].mean())
g1 = int((grp == 1).nonzero()[0])
g2 = int((grp == 2).nonzero()[0])
np.savez(os.path.join(args.out_dir, "rocking_traces.npz"), dt=corr.dt, r_z=r_z,
         **{f"g1_{n}": T[n][:, g1].cpu().numpy() for n in names}, **{f"g2_{n}": T[n][:, g2].cpu().numpy() for n in names})
with open(os.path.join(args.out_dir, "m4_diag_velocity.json"), "w") as f:
    json.dump(R, f, indent=1)
print("[DIAG-V]", json.dumps(R), flush=True)
os._exit(0)
