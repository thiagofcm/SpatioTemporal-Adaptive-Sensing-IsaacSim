"""M6 validation of the dynamic scene corridor_dynamic_chase_random (moving OB1 / OB2) in the full RL env.

    isaacrun -g <gpu> -- python tests/sim/dynamic_scene_validation.py --headless --num_envs 1024

V1 structure: no baffles, 2 kinematic obstacle bodies, fixed walls only
V2 bodies follow the motion model every step (active: on the path; inactive: parked below the floor)
V3 trajectories (100 s, robots standing): on the episode segment, speed bound, reversal regions, turnaround pauses
V4 LiDAR vs the analytic oracle with the ACTIVE obstacles (inactive ones must be invisible)
V5 independent partial resets (other envs' obstacles untouched, reset envs redrawn)
V6 collisions with obstacles: safety violation (margin 0.05); physical contact (contact sensor) where the analytic clearance reaches 0
V7 presence randomization statistics
Writes tests/sim/results/dynamic_scene_validation.json (+ figure).
"""
import argparse
import json
import math
import os

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--num_envs", type=int, default=1024)
p.add_argument("--out", default=os.path.join(os.environ.get("ST_ROOT", "."), "tests", "sim", "results"))
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import st_nav.tasks.nav2d  # noqa: E402,F401
from isaaclab.utils.math import euler_xyz_from_quat  # noqa: E402
from st_nav.geometry.analytic2d import circle_clearance, raycast_2d, wall_clearance  # noqa: E402
from st_nav.scenes import corridor_dynamic_chase_random as D  # noqa: E402
from st_nav.scenes._obstacle_motion import OBSTACLE_RADIUS, OBSTACLE_TURN_PAUSE_RANGE_S  # noqa: E402
from st_nav.tasks.nav2d.corridor_env import PHYSICAL_COLLISION, SAFETY_VIOLATION  # noqa: E402
from st_nav.tasks.nav2d.corridor_env_cfg import CorridorNavEnvCfg  # noqa: E402

os.makedirs(args.out, exist_ok=True)
cfg = CorridorNavEnvCfg()
cfg.scene.num_envs, cfg.sim.device, cfg.seed = args.num_envs, args.device, 7
cfg.scene_name = "corridor_dynamic_chase_random"
cfg.active_obstacles = (1, 2)
cfg.obstacle_presence_dr = cfg.obstacle_speed_dr = cfg.obstacle_y_dr = True
cfg.goals = D.GOAL_LIST_ST04
cfg.episode_length_s = 400.0                       # long episodes: trajectories without timeouts
genv = gym.make("STNav-TB4-Corridor-v0", cfg=cfg)
_IM = torch.inference_mode()
_IM.__enter__()
u = genv.unwrapped
corr, N, dev = u.corr, u.num_envs, u.device
om = corr.obs_model
CTRL = u.step_dt
R, checks = {}, {}


def ok(name, cond, detail=None):
    checks[name] = {"pass": bool(cond), "detail": detail}
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail if detail is not None else ''}", flush=True)


def bodies():
    return torch.stack([ob.data.root_pos_w - corr.origins for ob in corr.obstacles], 1)   # (N, K, 3)


AZ = torch.deg2rad(torch.tensor(corr.lidar.cfg.azimuths_deg, device=dev))


def oracle(circles):
    org = corr.lidar.origin_w - corr.origins
    _, _, ly = euler_xyz_from_quat(corr.lidar.rc.data.quat_w)
    return raycast_2d(org[:, :2], ly[:, None] + AZ[None], corr.walls, circles, max_range=3.5)


zero = torch.zeros(N, 2, device=dev)
genv.reset()
# ---------------------------------------------------------------- V1 / V7
ok("V1_no_baffles_two_obstacle_bodies_fixed_walls", (not corr.has_baffles) and len(corr.obstacles) == 2
   and corr.walls.shape[1] == 10, {"walls": corr.walls.shape[1], "obstacles": len(corr.obstacles)})
combo = (om.active[:, 0].long() + 2 * om.active[:, 1].long())
frac = (torch.bincount(combo, minlength=4).float() / N).tolist()
ok("V7_presence_dr_one_third_each_never_empty", frac[0] == 0 and max(abs(f - 1 / 3) for f in frac[1:]) < 0.06,
   {"none/OB1/OB2/both": [round(f, 3) for f in frac]})

# ---------------------------------------------------------------- V2 / V3 / V4: 100 s, robots standing
T = int(100 / CTRL)
xy_model, xy_body, z_body, act_log, s_log = [], [], [], [], []
lidar_err_obs, lidar_err_wall, inactive_rays, inactive_err, obs_rays = [], [], 0, [], 0
dones_total = 0
for k in range(T):
    _, _, term, trunc, _ = genv.step(zero)
    dones_total += int((term | trunc).sum())
    pm, b = om.positions(), bodies()
    xy_model.append(pm[:64].cpu()); xy_body.append(b[:64, :, :2].cpu()); z_body.append(b[:, :, 2].cpu())
    act_log.append(om.active.cpu()); s_log.append(om.s[:64].cpu())
    err = torch.where(om.active, (b[..., :2] - pm).norm(dim=-1), torch.zeros_like(om.s))
    R.setdefault("body_model_err_max", 0.0)
    R["body_model_err_max"] = max(R["body_model_err_max"], float(err.max()))
    if k % 25 == 0:                                          # LiDAR vs oracle every 5 s
        clean = corr.lidar.read().ranges_clean
        ora = oracle(corr.obstacle_circles())
        circ_all = corr.obstacle_circles().clone(); circ_all[..., 2] = OBSTACLE_RADIUS
        ora_all = oracle(circ_all)
        ora_walls = oracle(None)
        on_obs = ora < ora_walls - 1e-4                      # nearest hit is an active obstacle
        lidar_err_obs.append((clean - ora)[on_obs].abs().cpu()); lidar_err_wall.append((clean - ora)[~on_obs].abs().cpu())
        # grazing test: the oracle's hit circle passes the ray at a perpendicular distance in the annulus between the
        # 32-facet cylinder mesh (inradius r cos(pi/32) = 0.1692 m) and the analytic circle (0.17 m)
        org = (corr.lidar.origin_w - corr.origins)[:, :2]
        _, _, ly = euler_xyz_from_quat(corr.lidar.rc.data.quat_w)
        ang = ly[:, None] + AZ[None]
        uvec = torch.stack([torch.cos(ang), torch.sin(ang)], -1)                     # (N, B, 2)
        cc = corr.obstacle_circles()                                                  # (N, K, 3)
        rel = cc[:, None, :, :2] - org[:, None, None, :]                              # (N, 1, K, 2)
        perp = (uvec[:, :, None, 0] * rel[..., 1] - uvec[:, :, None, 1] * rel[..., 0]).abs()
        along = uvec[:, :, None, 0] * rel[..., 0] + uvec[:, :, None, 1] * rel[..., 1]
        entry = along - torch.sqrt(torch.clamp(cc[:, None, :, 2] ** 2 - perp ** 2, min=0))
        is_hit = (cc[:, None, :, 2] > 0) & (along > 0) & (perp <= cc[:, None, :, 2]) & ((entry - ora[..., None]).abs() < 1e-3)
        depth = torch.where(is_hit, cc[:, None, :, 2] - perp, torch.full_like(perp, float("inf"))).min(-1).values
        bad = on_obs & ((clean - ora).abs() > 0.01)
        R.setdefault("bad_ray_depth_mm", []).extend((1e3 * depth[bad]).tolist())
        obs_rays += int(on_obs.sum())
        hidden = ora_all < ora - 0.05                        # an inactive obstacle would block this ray
        inactive_rays += int(hidden.sum())
        inactive_err.append((clean - ora)[hidden].abs().cpu())
ok("V2_bodies_follow_model", R["body_model_err_max"] < 1e-4, {"max_err_m": R["body_model_err_max"]})
Z, A = torch.stack(z_body), torch.stack(act_log)
ok("V2_inactive_parked_below_floor_active_on_floor", bool((Z[~A] < 0).all() and (Z[A] > 0).all()),
   {"inactive_z_max": float(Z[~A].max()) if (~A).any() else None, "active_z_min": float(Z[A].min())})
ok("V3_no_terminations_while_standing", dones_total == 0, dones_total)
XM, S_ = torch.stack(xy_model), torch.stack(s_log)                       # (T, 64, K, 2), (T, 64, K)
Y = om.path_y[:64].cpu()
on_seg = ((XM[..., 1] - Y).abs() < 1e-5) & (XM[..., 0] >= 1.4 - 1e-4) & (XM[..., 0] <= 2.4 + 1e-4)
act64 = om.active[:64].cpu()
ok("V3_on_episode_segment", bool(on_seg[:, act64].all()))
spd = om.speed[:64].cpu()
ds = (S_[1:] - S_[:-1]).abs()
ok("V3_speed_bound", bool((ds <= spd * CTRL + 1e-4).all()), {"max_ds/(v*dt)": float((ds / (spd * CTRL)).max())})
L = float(om.L[0])
revs, pauses = [], []
for e in range(64):
    for j in range(2):
        if not act64[e, j]:
            continue
        s = S_[:, e, j].numpy()
        d = np.sign(np.diff(s))
        mv = np.where(d != 0)[0]
        dd = d[mv]
        flips = mv[1:][dd[1:] != dd[:-1]]                                 # first moving step after a reversal
        for f in flips:
            prev = mv[mv < f][-1]
            revs.append((s[prev + 1] / L, dd[mv < f][-1]))
            pauses.append((f - prev - 1) * CTRL)
rv = np.array([r[0] for r in revs]); rd = np.array([r[1] for r in revs])
pz = np.array(pauses)
lo, hi = OBSTACLE_TURN_PAUSE_RANGE_S
ok("V3_reversal_regions", len(rv) > 50 and rv[rd > 0].min() >= 0.75 - 0.02 and rv[rd < 0].max() <= 0.25 + 0.02,
   {"n": len(rv), "fwd_min": float(rv[rd > 0].min()), "bwd_max": float(rv[rd < 0].max())})
ok("V3_turnaround_pause_4.6_5.0s", len(pz) > 50 and pz.min() >= lo - 2 * CTRL and pz.max() <= hi + CTRL,
   {"n": len(pz), "min": float(pz.min()), "max": float(pz.max()), "mean": float(pz.mean())})
eo, ew, ei = torch.cat(lidar_err_obs), torch.cat(lidar_err_wall), torch.cat(inactive_err) if inactive_err else torch.zeros(1)
bd = np.array(R["bad_ray_depth_mm"]) if R.get("bad_ray_depth_mm") else np.zeros(0)
ok("V4_lidar_matches_oracle_on_obstacles", obs_rays > 1000 and float(eo.quantile(0.99)) < 0.01
   and float((eo > 0.01).float().mean()) < 1e-2 and (len(bd) == 0 or bd.max() < 0.17e3 * (1 - math.cos(math.pi / 32)) + 1e3 * 0.08 * u.physics_dt),
   {"rays": obs_rays, "p50_mm": 1e3 * float(eo.median()), "p99_mm": 1e3 * float(eo.quantile(0.99)), "max_mm": 1e3 * float(eo.max()),
    "rays_err_gt_1cm": int((eo > 0.01).sum()), "frac_err_gt_1cm": float((eo > 0.01).float().mean()), "their_depth_inside_circle_mm_max": float(bd.max()) if len(bd) else None, "mesh_circle_gap_mm": 0.17e3 * (1 - math.cos(math.pi / 32)), "tolerance_mm": "gap + one physics step of obstacle motion (0.8 mm)",
    "note": "errors > 1 cm only on grazing rays (cylinder mesh = 32-facet polygon vs analytic circle)"})
ok("V4_lidar_matches_oracle_on_walls", float(ew.quantile(0.999)) < 0.005,
   {"p999_mm": 1e3 * float(ew.quantile(0.999)), "max_mm": 1e3 * float(ew.max())})
ok("V4_inactive_obstacles_invisible", inactive_rays > 100 and float(ei.quantile(0.99)) < 0.01,
   {"rays_where_inactive_would_block": inactive_rays, "p99_mm": 1e3 * float(ei.quantile(0.99))})

# ---------------------------------------------------------------- V5: independent partial resets
snap = {k: getattr(om, k).clone() for k in ("s", "dir", "target", "pause", "speed", "path_y", "active")}
b0 = bodies().clone()
ids = torch.arange(0, N, 2, device=dev)
u._reset_idx(ids)
keep = torch.ones(N, dtype=torch.bool, device=dev); keep[ids] = False
same = all(torch.equal(getattr(om, k)[keep], v[keep]) for k, v in snap.items())
ok("V5_other_envs_untouched", same and torch.equal(bodies()[keep], b0[keep]))
ok("V5_reset_envs_redrawn", not torch.equal(om.s[ids], snap["s"][ids]) and float((om.s[ids] - snap["s"][ids]).abs().mean()) > 0.05)
ok("V5_reset_bodies_at_new_state", float(torch.where(om.active[ids], (bodies()[ids, :, :2] - om.positions()[ids]).norm(dim=-1),
                                                     torch.zeros_like(om.s[ids])).max()) < 1e-4)

# ---------------------------------------------------------------- V6: collisions with OB1
info = {}
_orig = u._get_dones


def _hook():
    out = _orig()
    pr, _ = corr.robot_pose2d()
    info["c"] = u._clear.clone()
    info["c_obs"] = circle_clearance(pr, corr.obstacle_circles()) - corr.cfg.footprint_radius
    info["c_wall"] = wall_clearance(pr, corr.walls) - corr.cfg.footprint_radius
    info["f"] = corr.contact_force_xy().clone()
    return out


u._get_dones = _hook


def place_for_collision():
    """OB1 set to the left end of its path moving +x (pause 0, target in the right reversal region); robot standing on
    OB1's path at x = 1.9 (0.5 m ahead, clearance 0.153 m), facing +x. Envs with OB1 inactive: same robot spot."""
    genv.reset()
    om.s[:, 0], om.dir[:, 0], om.pause[:, 0], om.target[:, 0] = 0.0, 1.0, 0.0, 0.9 * om.L[0]
    corr._write_obstacle_poses()
    st = torch.stack([torch.full((N,), 1.9, device=dev), om.path_y[:, 0]], 1)
    q = torch.zeros(N, 4, device=dev); q[:, 0] = 1
    corr.robot.write_root_pose_to_sim(torch.cat([corr.origins[:, :2] + st, torch.zeros(N, 1, device=dev), q], 1))
    corr.robot.write_root_velocity_to_sim(torch.zeros(N, 6, device=dev))
    return om.active[:, 0].clone()                         # clone: presence is redrawn when an env resets


u.cfg.safety_margin = 0.05
m = place_for_collision()
first = torch.full((N,), -1, dtype=torch.long, device=dev)
oc = torch.zeros(N, dtype=torch.long, device=dev)
c_t, co_t, cw_t = (torch.zeros(N, device=dev) for _ in range(3))
for k in range(int(15 / CTRL)):
    _, _, term, trunc, ext = genv.step(zero)
    new = (term | trunc) & (first < 0)
    if new.any():
        first[new], oc[new] = k, ext["outcome"][new]
        c_t[new], co_t[new], cw_t[new] = info["c"][new], info["c_obs"][new], info["c_wall"][new]
hit = m & (first >= 0)
ok("V6_safety_violation_on_approaching_OB1", bool(hit.sum() == m.sum()) and bool((oc[hit] == SAFETY_VIOLATION).all()),
   {"envs_OB1_active": int(m.sum()), "terminated": int(hit.sum()), "outcome_safety": int((oc[hit] == SAFETY_VIOLATION).sum()),
    "time_to_violation_s_min_max": [float(first[hit].min() + 1) * CTRL, float(first[hit].max() + 1) * CTRL]})
step_move = float(om.speed[:, 0].max()) * CTRL
ok("V6_terminates_at_margin_against_obstacle", bool(((c_t[hit] < 0.05) & (c_t[hit] > 0.05 - step_move - 1e-3)).all()
                                                   and (co_t[hit] < cw_t[hit]).all() and (co_t[hit] - c_t[hit]).abs().max() < 1e-5),
   {"clearance_at_term_min_max": [float(c_t[hit].min()), float(c_t[hit].max())]})
ok("V6_inactive_OB1_no_collision", bool((first[~m] < 0).all()), {"envs_OB1_inactive": int((~m).sum()), "terminated": int((first[~m] >= 0).sum())})
# physical contact: no termination (margin -1), first contact-sensor force > 1 N vs the analytic clearance
u.cfg.safety_margin = -1.0
m = place_for_collision()
c_first = torch.full((N,), float("nan"), device=dev)
co_first = torch.full((N,), float("nan"), device=dev)
touched = torch.zeros(N, dtype=torch.bool, device=dev)
for k in range(int(15 / CTRL)):
    genv.step(zero)
    new = (info["f"] > 1.0) & ~touched
    c_first[new], co_first[new] = info["c"][new], info["c_obs"][new]
    touched |= new
cf = c_first[m & touched]
ok("V6_physical_contact_where_analytic_clearance_reaches_zero",
   float((m & touched).float().sum() / m.float().sum()) > 0.97 and float(cf.min()) > -0.03 and float(cf.max()) < 0.01
   and bool(((co_first - c_first).abs()[m & touched] < 1e-5).all()),
   {"envs_OB1_active": int(m.sum()), "contact": int((m & touched).sum()),
    "clearance_at_first_contact_min_median_max": [float(cf.min()), float(cf.median()), float(cf.max())]})
ok("V6_no_contact_with_inactive_OB1", bool((~touched[~m]).all()), {"envs": int((~m).sum()), "contact": int(touched[~m].sum())})
u._get_dones = _orig
u.cfg.safety_margin = 0.05

# ---------------------------------------------------------------- figure + report
fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
t = np.arange(T) * CTRL
for e in range(6):
    for j, c in ((0, "tab:blue"), (1, "tab:orange")):
        if act64[e, j]:
            ax[0].plot(t, XM[:, e, j, 0].numpy(), color=c, lw=0.9, alpha=0.8)
ax[0].set_xlabel("time [s]"); ax[0].set_ylabel("x [m]"); ax[0].set_title("OB1 (blue) / OB2 (orange) x(t), 6 envs: random phase, reversals, 4.6-5.0 s pauses", fontsize=9)
for e in range(64):
    for j, c in ((0, "tab:blue"), (1, "tab:orange")):
        if act64[e, j]:
            ax[1].plot(XM[:, e, j, 0].numpy(), XM[:, e, j, 1].numpy(), color=c, lw=0.6)
ax[1].set_xlim(1.0, 2.8); ax[1].set_ylim(0.4, 2.0); ax[1].set_aspect("equal"); ax[1].grid(lw=0.3)
ax[1].set_title("paths of 64 envs (path y randomized per episode)", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(args.out, "dynamic_scene_trajectories.png"), dpi=110)
R["checks"] = checks
R["all_pass"] = all(c["pass"] for c in checks.values())
json.dump(R, open(os.path.join(args.out, "dynamic_scene_validation.json"), "w"), indent=1, default=float)
print(f"[DONE] all_pass = {R['all_pass']}", flush=True)
os._exit(0)
