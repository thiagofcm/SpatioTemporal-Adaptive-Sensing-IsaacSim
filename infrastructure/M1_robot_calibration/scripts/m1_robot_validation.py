"""M1 validation of the articulated TurtleBot4 Standard (no command-model delay/lag: raw drive + contact behaviour).

T1 rest stability      : zero command for 60 s; drift, speed jitter, pitch/roll
T2 steady-state grid   : (v, w) grid incl. infeasible combinations -> measured vs commanded vs scale-saturation model
T3 intrinsic response  : in-motion steps of v and w (command applied directly) -> delay / rise of the drive+contacts
T4 frames              : rplidar_link pose in base_link from the USD; collision footprint radii

    isaacrun -g auto -- python infrastructure/M1_robot_calibration/scripts/m1_robot_validation.py --headless [--physics_hz 200]
"""
import argparse
import json
import math
import os
import time

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--physics_hz", type=float, default=200.0)
p.add_argument("--out", default=os.path.join(os.environ.get("ST_ROOT", "."), "infrastructure", "M1_robot_calibration", "results"))
p.add_argument("--pos_iters", type=int, default=16)
p.add_argument("--vel_iters", type=int, default=2)
p.add_argument("--tag", default="", help="suffix for output files (keeps the approved M1 results intact)")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402

from st_nav.assets.turtlebot4 import TB4_CFG  # noqa: E402
from st_nav.calibration.real_data import step_metrics  # noqa: E402
from st_nav.calibration.sim_harness import Driver, default_geometry, make_scene  # noqa: E402
from st_nav.control.diff_drive import CommandModel, scale_saturation  # noqa: E402

os.makedirs(args.out, exist_ok=True)
DT = 1.0 / args.physics_hz
geo = default_geometry()
V_GRID = [-0.22, -0.10, 0.0, 0.10, 0.22]
W_GRID = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
GRID = [(v, w) for v in V_GRID for w in W_GRID]
# T3 in-motion steps: (v0, w0) -> (v1, w1)
STEPS = [(0.10, 0.0, 0.15, 0.0), (0.15, 0.0, 0.20, 0.0), (0.20, 0.0, 0.10, 0.0),
         (0.0, 0.2, 0.0, 0.4), (0.0, 0.4, 0.0, 0.6), (0.0, 0.6, 0.0, 0.2), (0.0, 0.4, 0.0, -0.4)]
N = len(GRID) + len(STEPS) + 8          # + 8 envs that stay at rest for T1
dev = args.device
_rc = TB4_CFG.copy()
_rc.spawn.articulation_props = _rc.spawn.articulation_props.replace(solver_position_iteration_count=args.pos_iters,
                                                                    solver_velocity_iteration_count=args.vel_iters)
sim, scene, robot, wheel_ids = make_scene(N, DT, dev, robot_cfg=_rc)
cm = CommandModel(N, DT, geo, device=dev)            # raw: no delay, no lag, gains 1 (saturation on)
drv = Driver(sim, scene, robot, wheel_ids, cm)
res = {"physics_hz": args.physics_hz, "num_envs": N}

# ---------------------------------------------------------------- T4 frames (USD, before stepping) -----------------
stage = sim_utils_stage = __import__("isaaclab.sim", fromlist=["get_current_stage"]).get_current_stage()
base = stage.GetPrimAtPath("/World/envs/env_0/Robot/base_link")
lidar = None
for prim in Usd.PrimRange(base):
    if prim.GetName() == "rplidar_link":
        lidar = prim
        break
xc = UsdGeom.XformCache()
rel = xc.GetLocalToWorldTransform(lidar) * xc.GetLocalToWorldTransform(base).GetInverse()
t = rel.ExtractTranslation()
q = rel.ExtractRotationQuat()
yaw = 2 * math.atan2(q.GetImaginary()[2], q.GetReal())
res["T4_lidar_frame"] = {"prim": str(lidar.GetPath()), "xyz_in_base": [round(v, 5) for v in t], "yaw_rad": yaw,
                         "expected_xyz": [-0.04, 0.0, 0.192915], "expected_yaw": math.pi / 2}
print("[T4]", res["T4_lidar_frame"], flush=True)

# ---------------------------------------------------------------- settle -------------------------------------------
for _ in range(int(1.0 / DT)):
    drv.physics_step()
p0, r0, pch0, y0 = drv.pose2d()
p0 = p0.clone()
res["settled_base_z_m"] = float(p0[:, 2].mean())
res["settled_pitch_deg"] = float(torch.rad2deg(pch0).mean())

# ---------------------------------------------------------------- run T1/T2/T3 together ----------------------------
i_grid = torch.arange(len(GRID), device=dev)
i_step = torch.arange(len(GRID), len(GRID) + len(STEPS), device=dev)
i_rest = torch.arange(len(GRID) + len(STEPS), N, device=dev)
cmd_v = torch.zeros(N, device=dev)
cmd_w = torch.zeros(N, device=dev)
cmd_v[i_grid] = torch.tensor([g[0] for g in GRID], device=dev)
cmd_w[i_grid] = torch.tensor([g[1] for g in GRID], device=dev)
step_v0 = torch.tensor([s[0] for s in STEPS], device=dev)
step_w0 = torch.tensor([s[1] for s in STEPS], device=dev)
step_v1 = torch.tensor([s[2] for s in STEPS], device=dev)
step_w1 = torch.tensor([s[3] for s in STEPS], device=dev)
T_STEP = 3.0          # T3 step time (s after settle)
T_TOTAL = 60.0        # T1 duration
n_steps = int(T_TOTAL / DT)
log_t, log_v, log_w = [], [], []
rest_speed_max = torch.zeros(len(i_rest), device=dev)
wall0 = time.time()
for k in range(n_steps):
    tk = (k + 1) * DT
    step_on = tk >= T_STEP
    cmd_v[i_step] = torch.where(torch.tensor(step_on, device=dev), step_v1, step_v0)
    cmd_w[i_step] = torch.where(torch.tensor(step_on, device=dev), step_w1, step_w0)
    if tk > 8.0:                      # T2 grid runs 8 s, then stops (keeps the robots inside their cells)
        cmd_v[i_grid] = 0.0
        cmd_w[i_grid] = 0.0
    if tk > 6.0:
        cmd_v[i_step] = 0.0
        cmd_w[i_step] = 0.0
    cm.set_command(cmd_v, cmd_w)
    drv.physics_step()
    v, w = drv.twist()
    if tk <= 8.0:
        log_t.append(tk)
        log_v.append(v.clone())
        log_w.append(w.clone())
    lin = torch.linalg.norm(robot.data.root_lin_vel_w[i_rest, :2], dim=-1)
    if tk > 3.0:
        rest_speed_max = torch.maximum(rest_speed_max, lin)
wall = time.time() - wall0
T = np.array(log_t)
V = torch.stack(log_v).cpu().numpy()
W = torch.stack(log_w).cpu().numpy()

# T1
p1, r1, pch1, _ = drv.pose2d()
drift = torch.linalg.norm((p1 - p0)[i_rest, :2], dim=-1)
res["T1_rest"] = {"duration_s": T_TOTAL, "jitter_window": "t > 3 s (2 s after settling)", "max_drift_m": float(drift.max()), "max_planar_speed_mps": float(rest_speed_max.max()),
                  "max_abs_pitch_deg": float(torch.rad2deg(pch1[i_rest]).abs().max()),
                  "max_abs_roll_deg": float(torch.rad2deg(r1[i_rest]).abs().max())}
print("[T1]", res["T1_rest"], flush=True)

# T2: mean over 4..8 s
sel = (T >= 4.0) & (T <= 8.0)
gv = torch.tensor([g[0] for g in GRID], device=dev)
gw = torch.tensor([g[1] for g in GRID], device=dev)
sv, sw = scale_saturation(gv, gw, geo)
rows = []
for j, (v, w) in enumerate(GRID):
    mv, mw = float(V[sel, j].mean()), float(W[sel, j].mean())
    ev, ew = float(sv[j]), float(sw[j])
    rows.append(dict(v_cmd=v, w_cmd=w, v_expected=ev, w_expected=ew, v_meas=mv, w_meas=mw,
                     v_err_pct=100 * (mv - ev) / ev if abs(ev) > 1e-6 else None,
                     w_err_pct=100 * (mw - ew) / ew if abs(ew) > 1e-6 else None,
                     v_abs_err=mv - ev, w_abs_err=mw - ew, saturated=bool(abs(ev - v) > 1e-6 or abs(ew - w) > 1e-6)))
res["T2_steady_state"] = rows
errs_v = [abs(r["v_err_pct"]) for r in rows if r["v_err_pct"] is not None]
errs_w = [abs(r["w_err_pct"]) for r in rows if r["w_err_pct"] is not None]
res["T2_summary"] = {"max_abs_v_err_pct": max(errs_v), "max_abs_w_err_pct": max(errs_w),
                     "max_abs_v_err_unsat_pct": max(abs(r["v_err_pct"]) for r in rows if r["v_err_pct"] is not None and not r["saturated"]),
                     "max_abs_w_err_unsat_pct": max(abs(r["w_err_pct"]) for r in rows if r["w_err_pct"] is not None and not r["saturated"]),
                     "max_abs_v_err_zero_cmd": max(abs(r["v_abs_err"]) for r in rows if r["v_expected"] == 0),
                     "max_abs_w_err_zero_cmd": max(abs(r["w_abs_err"]) for r in rows if r["w_expected"] == 0)}
print("[T2]", res["T2_summary"], flush=True)

# T3
t3 = []
for j, (v0, w0, v1, w1) in enumerate(STEPS):
    col = len(GRID) + j
    if v0 != v1:
        d, r = step_metrics(T, V[:, col], T_STEP, v0, v1, T_STEP + 2.0)
        comp = "v"
    else:
        d, r = step_metrics(T, W[:, col], T_STEP, w0, w1, T_STEP + 2.0)
        comp = "w"
    t3.append(dict(step=f"{comp}: {v0 if comp == 'v' else w0} -> {v1 if comp == 'v' else w1}", delay_s=d, rise_s=r))
res["T3_intrinsic_steps"] = t3
print("[T3]", t3, flush=True)
res["wallclock_s"] = wall
res["physics_steps_per_s_total"] = n_steps * N / wall
np.savez(os.path.join(args.out, f"m1_validation_{int(args.physics_hz)}hz{args.tag}.npz"), t=T, v=V, w=W,
         grid=np.array(GRID), steps=np.array(STEPS), n_grid=len(GRID))
with open(os.path.join(args.out, f"m1_validation_{int(args.physics_hz)}hz{args.tag}.json"), "w") as f:
    json.dump(res, f, indent=1, default=float)
print("[DONE]", flush=True)
os._exit(0)   # app.close() can hang on this server (see README); results are already written
