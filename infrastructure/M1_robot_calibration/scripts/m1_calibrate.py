"""M1 actuator calibration: replay every real TurtleBot4 command sequence in simulation, with the physics in the loop,
for a grid of CommandModel parameters, and compare the complete simulated twist responses with the real odometry.

Envs = (real run) x (parameter combination). Because the fit is done with the PhysX drive + contacts in the loop, the
fitted delay/lag are only what the simulated robot lacks: actuator dynamics are not counted twice.

Scoring (per parameter combination, over all runs):
  RMS_v : linear-velocity error over linear + combined runs
  RMS_w : angular-velocity error over angular + combined runs
  samples within 2.5 s after a start-from-rest command (firmware start-up regime, 0.3-1.9 s real delays) are
  excluded from the fit and reported separately.

    isaacrun -g auto -- python infrastructure/M1_robot_calibration/scripts/m1_calibrate.py --headless
"""
import argparse
import itertools
import json
import os
import time

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--physics_hz", type=float, default=200.0)
p.add_argument("--out", default=os.path.join(os.environ.get("ST_ROOT", "."), "infrastructure", "M1_robot_calibration", "results"))
p.add_argument("--delays", default="0,0.02,0.04,0.06,0.08,0.10,0.12,0.14,0.16,0.18,0.20,0.22,0.24")
p.add_argument("--taus", default="0,0.02,0.04,0.06,0.08,0.10")
p.add_argument("--kws", default="0.85,0.90,0.95,1.00")
p.add_argument("--amax", default="0", help="grid: linear acceleration limits [m/s^2] (0 = none)")
p.add_argument("--w_fixed", default="", help="grid: fix the angular channel to delay_w,tau_w,k_w (only v is gridded)")
p.add_argument("--mode", choices=["grid", "final"], default="grid")
p.add_argument("--final", default="", help="final mode: delay_v,tau_v,delay_w,tau_w,k_w[,accel_v_max]")
p.add_argument("--pos_iters", type=int, default=16)
p.add_argument("--vel_iters", type=int, default=2)
p.add_argument("--tag", default="", help="suffix for output files (keeps the approved M1 results intact)")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

from st_nav.assets.turtlebot4 import TB4_CFG  # noqa: E402
from st_nav.calibration.real_data import load_all, startup_mask  # noqa: E402
from st_nav.calibration.sim_harness import Driver, default_geometry, make_scene  # noqa: E402
from st_nav.control.diff_drive import CommandModel  # noqa: E402

os.makedirs(args.out, exist_ok=True)
dev = args.device
DT = 1.0 / args.physics_hz
REC_EVERY = 2                                     # record twist at 100 Hz
runs = [r for r in load_all() if r.duration > 5.0]
delays = [float(x) for x in args.delays.split(",")]
taus = [float(x) for x in args.taus.split(",")]
kws = [float(x) for x in args.kws.split(",")]
amaxs = [float(x) for x in args.amax.split(",")]
if args.mode == "grid" and args.w_fixed:
    dw, tw, kw = (float(x) for x in args.w_fixed.split(","))
    params = list(itertools.product(delays, taus, amaxs))     # (delay_v, tau_v, accel_v_max)
    params5 = [(d_, t_, dw, tw, kw, a_) for d_, t_, a_ in params]
elif args.mode == "grid":
    params = list(itertools.product(delays, taus, kws))       # (delay, tau, k_w): same delay/tau on both channels
    params5 = [(d_, t_, d_, t_, k_, 0.0) for d_, t_, k_ in params]
else:
    fv = tuple(float(x) for x in args.final.split(","))
    fv = fv + (0.0,) * (6 - len(fv))
    params5 = [fv, (0.0, 0.0, 0.0, 0.0, 1.0, 0.0)]           # selected per-channel model, and no command model
    params = [(fv[0], fv[1], fv[4]), (0.0, 0.0, 1.0)]
R, P = len(runs), len(params)
N = R * P
T_END = max(r.duration for r in runs) + 0.5
n_steps = int(T_END / DT)
print(f"[CAL] runs={R} params={P} envs={N} sim_steps={n_steps}", flush=True)

# commands at every physics step (ZOH of the real 5 Hz command stream), per run
t_grid = (np.arange(n_steps) + 1) * DT
cmd = np.zeros((R, n_steps, 2), dtype=np.float32)
for i, r in enumerate(runs):
    cmd[i, :, 0], cmd[i, :, 1] = r.command_at(t_grid)
cmd_t = torch.tensor(cmd, device=dev)
run_of_env = torch.arange(N, device=dev) // P
par = torch.tensor(params5, device=dev, dtype=torch.float32)[torch.arange(N, device=dev) % P]

geo = default_geometry()
_rc = TB4_CFG.copy()
_rc.spawn.articulation_props = _rc.spawn.articulation_props.replace(solver_position_iteration_count=args.pos_iters,
                                                                    solver_velocity_iteration_count=args.vel_iters)
sim, scene, robot, wheel_ids = make_scene(N, DT, dev, env_spacing=4.0, robot_cfg=_rc)
cm = CommandModel(N, DT, geo, max_delay_s=float(par[:, [0, 2]].max()) + 0.05, device=dev)
cm.set_params(delay_s=par[:, 0], tau_v_s=par[:, 1], delay_w_s=par[:, 2], tau_w_s=par[:, 3], k_w=par[:, 4],
              accel_v_max=par[:, 5])
drv = Driver(sim, scene, robot, wheel_ids, cm)
zero = torch.zeros(N, device=dev)
for _ in range(int(0.5 / DT)):                     # settle at rest before t = 0 (real runs start with a pre-roll)
    cm.set_command(zero, zero)
    drv.physics_step()
rec_v = torch.zeros(N, n_steps // REC_EVERY + 1, device=dev)
rec_w = torch.zeros_like(rec_v)
wall0 = time.time()
for k in range(n_steps):
    c = cmd_t[run_of_env, k]
    cm.set_command(c[:, 0], c[:, 1])
    drv.physics_step()
    if k % REC_EVERY == 0:
        v, w = drv.twist()
        rec_v[:, k // REC_EVERY] = v
        rec_w[:, k // REC_EVERY] = w
wall = time.time() - wall0
print(f"[CAL] simulated {n_steps * DT:.1f} s x {N} envs in {wall:.1f} s ({n_steps * N / wall / 1e6:.2f} M physics env-steps/s)",
      flush=True)
t_rec = (np.arange(rec_v.shape[1]) * REC_EVERY + 1) * DT
SV = rec_v.cpu().numpy().reshape(R, P, -1)
SW = rec_w.cpu().numpy().reshape(R, P, -1)

# ---------------- scoring against the real odometry ----------------
sse = {k: np.zeros(P) for k in ("v", "w", "v_start", "w_start")}
cnt = {k: 0 for k in sse}
for i, r in enumerate(runs):
    ok = (r.odo_t > 0.2) & (r.odo_t < t_rec[-1])
    t = r.odo_t[ok]
    sv = np.stack([np.interp(t, t_rec, SV[i, j]) for j in range(P)])
    sw = np.stack([np.interp(t, t_rec, SW[i, j]) for j in range(P)])
    m_start = startup_mask(r, t)
    for comp, s, real, use in (("v", sv, r.odo_v[ok], r.kind in ("linear", "combined")),
                               ("w", sw, r.odo_w[ok], r.kind in ("angular", "combined"))):
        if not use:
            continue
        e2 = (s - real[None]) ** 2
        sse[comp] += e2[:, ~m_start].sum(1)
        cnt[comp] += int((~m_start).sum())
        sse[comp + "_start"] += e2[:, m_start].sum(1)
        cnt[comp + "_start"] += int(m_start.sum())
rms = {k: np.sqrt(sse[k] / max(cnt[k], 1)) for k in sse}
# joint objective: normalized by the typical command magnitude of each channel (0.15 m/s, 0.4 rad/s)
J = rms["v"] / 0.15 + rms["w"] / 0.4
best = int(np.argmin(J)) if args.mode == "grid" else 0
best_v = int(np.argmin(rms["v"])) if args.mode == "grid" else 0
best_w = int(np.argmin(rms["w"])) if args.mode == "grid" else 0
raw = params.index((0.0, 0.0, 1.0)) if (0.0, 0.0, 1.0) in params else best
res = {
    "runs": [r.name for r in runs], "grid": {"delays": delays, "taus": taus, "k_w": kws},
    "best_joint": dict(zip(("delay_s", "tau_s", "k_w"), params[best])),
    "best_for_v": dict(zip(("delay_s", "tau_s", "k_w"), params[best_v])),
    "best_for_w": dict(zip(("delay_s", "tau_s", "k_w"), params[best_w])),
    "rms": {name: {k: float(rms[k][j]) for k in rms} for name, j in
            (("best_joint", best), ("best_for_v", best_v), ("best_for_w", best_w), ("raw_no_model", raw))},
    "samples": cnt, "sim_wallclock_s": wall,
}
print("[CAL]", json.dumps(res["best_joint"]), json.dumps(res["rms"]), flush=True)
if args.mode == "final":
    res["final_model"] = dict(zip(("delay_v_s", "tau_v_s", "delay_w_s", "tau_w_s", "k_w", "accel_v_max"), params5[0]))
tag = ("grid_v" if args.w_fixed else "grid") if args.mode == "grid" else "final"
np.savez_compressed(os.path.join(args.out, f"m1_calibration_{tag}{args.tag}.npz"), t_rec=t_rec,
                    sim_v_best=SV[:, best], sim_w_best=SW[:, best], sim_v_raw=SV[:, raw], sim_w_raw=SW[:, raw],
                    params=np.array(params), rms_v=rms["v"], rms_w=rms["w"], rms_v_start=rms["v_start"],
                    rms_w_start=rms["w_start"], run_names=np.array([r.name for r in runs]), best=best, raw=raw)
with open(os.path.join(args.out, f"m1_calibration_{tag}{args.tag}.json"), "w") as f:
    json.dump(res, f, indent=1)
print("[DONE]", flush=True)
os._exit(0)
