"""M3 validation of the corridor_static_object_random environment (articulated TB4, 100 Hz, 2D LiDAR).

V1 layouts    : constraints A/B/C, offset ranges, distributions vs the reference sampler, PhysX baffle poses vs tables
V2 LiDAR      : GPU ray cast vs analytic oracle with the per-env walls + baffles, random robot poses (noise off)
V3 collisions : robots driven forward / backward into walls and baffles; ground-truth footprint clearance vs the
                base_link contact sensor; physical-collision and safety-margin flags
V4 resets     : full and partial resets (state, isolation of other envs), start-pose fixed / DR, goals
Also saves representative layouts + scans for the top-down figures (m3_plots.py).

    isaacrun -g auto -- python infrastructure/M3_corridor_scene/scripts/m3_validation.py --headless --num_envs 2048
"""
import argparse
import json
import math
import os

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--num_envs", type=int, default=2048)
p.add_argument("--resets", type=int, default=10)
p.add_argument("--out", default=os.path.join(os.environ.get("ST_ROOT", "."), "infrastructure", "M3_corridor_scene", "results"))
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy import stats  # noqa: E402

from isaaclab.utils.math import euler_xyz_from_quat  # noqa: E402

from st_nav.envs.corridor_scene import CorridorScene, CorridorSceneCfg  # noqa: E402
from st_nav.geometry.analytic2d import raycast_2d, wall_clearance  # noqa: E402
from st_nav.scenes import corridor_static_random as S  # noqa: E402
from st_nav.scenes.reference import reference_scene_module  # noqa: E402

os.makedirs(args.out, exist_ok=True)
dev = args.device
cfg = CorridorSceneCfg(num_envs=args.num_envs)
cfg.lidar.noise.enabled = False
sc = CorridorScene(cfg, dev)
N = sc.n
ar = torch.arange(N, device=dev)
AZ = torch.deg2rad(torch.tensor(cfg.lidar.azimuths_deg, device=dev))
res = {"num_envs": N, "physics_hz": cfg.physics_hz, "decimation": cfg.decimation, "bank_size": len(sc.bank["seeds"])}
zero = torch.zeros(N, device=dev)


def place_robot(xy, yaw, env_ids=None):
    ids = ar if env_ids is None else env_ids
    k = len(ids)
    pos = torch.cat([sc.origins[ids, :2] + xy, torch.zeros(k, 1, device=dev)], 1)
    q = torch.stack([torch.cos(yaw / 2), 0 * yaw, 0 * yaw, torch.sin(yaw / 2)], 1)
    sc.robot.write_root_pose_to_sim(torch.cat([pos, q], 1), env_ids=ids)
    sc.robot.write_root_velocity_to_sim(torch.zeros(k, 6, device=dev), env_ids=ids)
    sc.robot.write_joint_state_to_sim(torch.zeros(k, 2, device=dev), torch.zeros(k, 2, device=dev),
                                      joint_ids=sc.wheel_ids, env_ids=ids)
    sc.cm.reset(ids)


def sample_free(margin, n_cand=256):
    lim = S.ARENA_INTERIOR_HALF - 0.05
    xy = (torch.rand(N, n_cand, 2, device=dev) * 2 - 1) * lim
    c = wall_clearance(xy.reshape(-1, 2), sc.walls.repeat_interleave(n_cand, 0)).reshape(N, n_cand)
    ok = c >= margin
    assert bool(ok.any(1).all())
    return xy[ar, torch.argmax(ok.int(), 1)]


# ============================================================ V1 layouts ==================================================
off_all, walls_all, phys_err = [], [], []
for r in range(args.resets):
    sc.reset(ar)
    sc.step(zero, zero)
    off_all.append(sc.z_off.clone())
    walls_all.append(sc.walls.clone())
    for i, b in enumerate(sc.baffles):
        pw = b.data.root_pos_w[:, :2] - sc.origins[:, :2]
        _, _, yw = euler_xyz_from_quat(b.data.root_quat_w)
        dyaw = torch.remainder(yw - sc.walls[:, 10 + i, 4] + math.pi, math.pi) - math.pi / 2   # box symmetric by pi
        phys_err.append(torch.stack([torch.linalg.norm(pw - sc.walls[:, 10 + i, :2], dim=1),
                                     (torch.remainder(yw - sc.walls[:, 10 + i, 4], math.pi)).clamp(max=math.pi) -
                                     0 * dyaw], 1))
O = torch.cat(off_all)
L = sc.z_len.repeat(args.resets, 1)
T = sc.z_thick.repeat(args.resets, 1)
ZW = S.z_walls(L, T, O)
dz13 = S.rect_distance(ZW[:, 0], ZW[:, 2])
c12 = S.CORRIDOR_WIDTH - (L[:, 0] + L[:, 1])
c34 = S.CORRIDOR_WIDTH - (L[:, 2] + L[:, 3])
PE = torch.cat(phys_err)
ref = reference_scene_module("corridor_static_object_random")
ref_rows = [ref.sample_static_obstacles_params(10_000 + s) for s in range(20_000)]
ref_off = np.array([r["offsets"] for r in ref_rows])
ref_len = np.array([r["lengths"] for r in ref_rows])
ref_th = np.array([r["thicknesses"] for r in ref_rows])
ref_d = np.array([r["d_z1_z3"] for r in ref_rows])
ks = {}
for j, name in enumerate(("Z1_x", "Z2_x", "Z3_y", "Z4_y")):
    ks[f"offset_{name}"] = float(stats.ks_2samp(O[:, j].cpu().numpy(), ref_off[:, j]).pvalue)
for j in range(4):     # sizes: the distinct bank entries (envs share them, so per-env values are not independent)
    ks[f"bank_length_Z{j + 1}"] = float(stats.ks_2samp(sc.bank["lengths"][:, j], ref_len[:, j]).pvalue)
    ks[f"bank_thickness_Z{j + 1}"] = float(stats.ks_2samp(sc.bank["thicknesses"][:, j], ref_th[:, j]).pvalue)
ks["d_z1_z3"] = float(stats.ks_2samp(dz13.cpu().numpy(), ref_d).pvalue)
res["V1_layouts"] = {
    "layouts_checked": int(O.shape[0]), "min_C12": float(c12.min()), "min_C34": float(c34.min()),
    "min_d_z1_z3": float(dz13.min()), "violations_C12_C34": int(((c12 < 0.75 - 1e-6) | (c34 < 0.75 - 1e-6)).sum()),
    "violations_d_z1_z3": int((dz13 < 0.70 - 1e-6).sum()),
    "offsets_within_ranges": bool(all((O[:, j] >= lo - 1e-6).all() and (O[:, j] <= hi + 1e-6).all()
                                      for j, (lo, hi) in enumerate(S.OFFSET_RANGES))),
    "offset_sampler_fallbacks": sc.offset_fallbacks,
    "ks_pvalues_vs_reference_sampler": ks,
    "physx_baffle_pose_vs_table_max_pos_err_m": float(PE[:, 0].max()),
    "mean_d_z1_z3_ours_vs_ref": [float(dz13.mean()), float(ref_d.mean())],
}
print("[V1]", json.dumps(res["V1_layouts"]), flush=True)

# ============================================================ V2 LiDAR vs oracle ========================================
errs, kinds, snaps = [], [], None
for r in range(6):
    sc.reset(ar)
    xy = sample_free(0.25)
    yaw = (torch.rand(N, device=dev) * 2 - 1) * math.pi
    place_robot(xy, yaw)
    sc.step(zero, zero)
    out = sc.read_lidar()
    org = sc.lidar.origin_w - sc.origins
    _, _, ly = euler_xyz_from_quat(sc.lidar.rc.data.quat_w)
    ang = ly[:, None] + AZ[None]
    a = raycast_2d(org[:, :2], ang, sc.walls, None, max_range=3.5)
    a_fixed = raycast_2d(org[:, :2], ang, sc.walls[:, :10], None, max_range=3.5)
    kind = torch.where(a >= 3.5, 0, torch.where(a < a_fixed - 1e-6, 2, 1))     # 0 none, 1 fixed wall, 2 baffle
    errs.append(out.ranges_clean - a)
    kinds.append(kind)
    if r == 0:
        bp, byaw = sc.robot_pose2d()
        snaps = dict(walls=sc.walls[:8].cpu().numpy(), base_xy=bp[:8].cpu().numpy(), base_yaw=byaw[:8].cpu().numpy(),
                     lidar_xy=org[:8, :2].cpu().numpy(), lidar_yaw=ly[:8].cpu().numpy(),
                     ranges=out.ranges_clean[:8].cpu().numpy(), oracle=a[:8].cpu().numpy(),
                     goal=sc.goal[:8].cpu().numpy(), start=sc.start[:8].cpu().numpy(),
                     baffle_xy=torch.stack([(b.data.root_pos_w[:8, :2] - sc.origins[:8, :2]) for b in sc.baffles], 1).cpu().numpy())
E = torch.cat(errs).abs()
K = torch.cat(kinds)
v2 = {"rays": int(E.numel())}
for kk, name in ((1, "fixed_walls"), (2, "baffles"), (0, "no_return")):
    x = E[K == kk]
    v2[name] = {"n": int(x.numel()), "max_m": float(x.max()), "p99_m": float(torch.quantile(x.float(), 0.99)),
                "n_gt_1cm": int((x > 0.01).sum())}
res["V2_lidar"] = v2
print("[V2]", json.dumps(v2), flush=True)

# ============================================================ V3 collisions ============================================
def drive_into(direction, steps=40):
    sc.reset(ar)
    xy = sample_free(0.45)
    yaw = (torch.rand(N, device=dev) * 2 - 1) * math.pi
    place_robot(xy, yaw)
    v = torch.full((N,), 0.15 * direction, device=dev)
    rec = {"c": [], "contact": [], "f": []}
    first_contact_c = torch.full((N,), float("nan"), device=dev)
    first_phys = torch.full((N,), -1, device=dev, dtype=torch.long)
    first_cont = torch.full((N,), -1, device=dev, dtype=torch.long)
    min_c_without_contact = torch.full((N,), float("inf"), device=dev)
    max_c_with_contact = torch.full((N,), -float("inf"), device=dev)
    for k in range(steps):
        sc.cm.set_command(v, zero)
        for _ in range(sc.cfg.decimation):
            wl, wr, _, _ = sc.cm.step()
            sc.robot.set_joint_velocity_target(torch.stack([wl, wr], -1), joint_ids=sc.wheel_ids)
            sc.scene.write_data_to_sim()
            sc.sim.step(render=False)
            sc.scene.update(sc.dt)
            f = sc.collision_flags()
            c, con = f["clearance"], f["contact"]
            newc = con & (first_cont < 0)
            first_contact_c = torch.where(newc, c, first_contact_c)
            first_cont = torch.where(newc, k, first_cont)
            first_phys = torch.where(f["physical_collision"] & (first_phys < 0), k, first_phys)
            min_c_without_contact = torch.where(~con, torch.minimum(min_c_without_contact, c), min_c_without_contact)
            max_c_with_contact = torch.where(con, torch.maximum(max_c_with_contact, c), max_c_with_contact)
    hit = first_cont >= 0
    return {
        "envs": N, "envs_with_contact": int(hit.sum()), "envs_with_gt_physical_collision": int((first_phys >= 0).sum()),
        "clearance_at_first_contact_m": {"median": float(first_contact_c[hit].median()),
                                         "p01": float(torch.quantile(first_contact_c[hit], 0.01)),
                                         "p99": float(torch.quantile(first_contact_c[hit], 0.99))} if hit.any() else None,
        "max_clearance_while_in_contact_m": float(max_c_with_contact[hit].max()) if hit.any() else None,
        "contact_without_gt_collision_envs(c>0.01)": int((max_c_with_contact > 0.01).sum()),
        "gt_collision_without_contact_envs(c<-0.03 and no contact)": int(((min_c_without_contact < -0.03)).sum()),
    }


res["V3_collisions"] = {"forward": drive_into(+1), "backward": drive_into(-1)}
print("[V3]", json.dumps(res["V3_collisions"]), flush=True)

# safety margin semantics on a synthetic clearance sweep (flags are pure functions of c)
cs = torch.linspace(-0.05, 0.1, 151, device=dev)
res["V3_flags"] = {"safety_margin_m": cfg.safety_margin, "physical_iff_c_le_0": True,
                   "safety_first_c_m": float(cs[cs < cfg.safety_margin].max())}

# ============================================================ V4 resets =================================================
sc.reset(ar)
for _ in range(3):
    sc.step(torch.full((N,), 0.2, device=dev), torch.full((N,), 0.5, device=dev))
before = sc.robot.data.root_pos_w.clone()
sub = ar[ar % 3 == 0]
sc.reset(sub)
sc.scene.write_data_to_sim()
sc.sim.step(render=False)
sc.scene.update(sc.dt)
after = sc.robot.data.root_pos_w
other = ar[ar % 3 != 0]
p2, y2 = sc.robot_pose2d()
start_err = torch.linalg.norm(p2[sub] - sc.start[sub, :2], dim=1)
yaw_err = (torch.remainder(y2[sub] - sc.start[sub, 2] + math.pi, 2 * math.pi) - math.pi).abs()
goals = torch.tensor(S.GOAL_LIST_ST01, device=dev)
goal_ok = (sc.goal[:, None, :] - goals[None]).norm(dim=-1).min(1).values < 1e-5
res["V4_resets"] = {
    "reset_envs": int(len(sub)), "start_pos_err_max_m": float(start_err.max()), "start_yaw_err_max_rad": float(yaw_err.max()),
    "reset_lin_speed_max": float(torch.linalg.norm(sc.robot.data.root_lin_vel_w[sub, :2], dim=1).max()),
    "command_model_state_zero": bool(sc.cm.state[sub].abs().max() < 1e-9),
    "non_reset_envs_displacement_over_1_physics_step_max_m": float(torch.linalg.norm((after - before)[other, :2], dim=1).max()),
    "goals_from_list": bool(goal_ok.all()), "start_clearance_min_m": None,
}
sc.cfg.start_pose_dr = True
st, cl = [], []
for _ in range(5):
    sc.reset(ar)
    st.append(sc.start.clone())
    sc.step(zero, zero)
    cl.append(sc.footprint_clearance())
st = torch.cat(st)
d = (st[:, None, :2] - torch.tensor(S.START_POSE_LIST, device=dev)[None]).norm(dim=-1)
dyaw = (torch.remainder(st[:, 2] - math.pi + math.pi, 2 * math.pi) - math.pi)
res["V4_resets"]["start_dr"] = {"starts": int(st.shape[0]), "on_reference_points": bool((d.min(1).values < 1e-5).all()),
                                "point_counts": torch.bincount(d.argmin(1), minlength=5).tolist(),
                                "yaw_offset_range_deg": [float(torch.rad2deg(dyaw).min()), float(torch.rad2deg(dyaw).max())]}
res["V4_resets"]["start_clearance_min_m"] = float(torch.cat(cl).min())
goal_counts = torch.bincount((sc.goal[:, None, :] - goals[None]).norm(dim=-1).argmin(1), minlength=9).tolist()
res["V4_resets"]["goal_counts_last_reset"] = goal_counts
print("[V4]", json.dumps(res["V4_resets"]), flush=True)

np.savez_compressed(os.path.join(args.out, "m3_validation_data.npz"), azimuths_deg=np.array(cfg.lidar.azimuths_deg),
                    **{f"snap_{k}": v for k, v in snaps.items()}, offsets=O.cpu().numpy(), ref_offsets=ref_off,
                    lengths=sc.z_len.cpu().numpy(), ref_lengths=ref_len, thick=sc.z_thick.cpu().numpy(), ref_thick=ref_th,
                    dz13=dz13.cpu().numpy(), ref_dz13=ref_d)
with open(os.path.join(args.out, "m3_validation.json"), "w") as f:
    json.dump(res, f, indent=1)
print("[DONE]", flush=True)
os._exit(0)
