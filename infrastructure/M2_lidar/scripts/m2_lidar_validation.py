"""M2 LiDAR validation: GPU mesh ray casting vs the analytic 2D oracle (port of the reference ray cast).

V1  random robot poses (position, yaw) and random mover positions, static scene      -> per-beam error (noise off)
V2  beam ordering: mover placed on the bearing of beam k = env % 40                    -> argmin beam == k
V3  moving robot (calibrated command model, random commands) + moving obstacle        -> per-beam error over time
V4  range limits (no-return rays = range_max, flags) and mounting (sensor pose vs base pose + URDF offset)
V5  noise model statistics (reference datasheet-uniform model)                         -> U(-1, 1) of e / (a(d) d)

    isaacrun -g auto -- python infrastructure/M2_lidar/scripts/m2_lidar_validation.py --headless
"""
import argparse
import json
import math
import os

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--num_envs", type=int, default=1024)
p.add_argument("--physics_hz", type=float, default=200.0)
p.add_argument("--scene", default="corridor_static_object")
p.add_argument("--debug", action="store_true")
p.add_argument("--out", default=os.path.join(os.environ.get("ST_ROOT", "."), "infrastructure", "M2_lidar", "results"))
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy import stats  # noqa: E402

from isaaclab.utils.math import euler_xyz_from_quat, quat_apply  # noqa: E402

from st_nav.assets.turtlebot4 import TB4_COMMAND_MODEL, TB4_GEOMETRY  # noqa: E402
from st_nav.control.diff_drive import CommandModel  # noqa: E402
from st_nav.geometry.analytic2d import circle_clearance, raycast_2d, wall_clearance, walls_from_specs  # noqa: E402
from st_nav.scenes.builder import SceneSpec  # noqa: E402
from st_nav.scenes.harness import make_nav_scene  # noqa: E402
from st_nav.scenes.reference import reference_scene_module  # noqa: E402
from st_nav.sensors.lidar2d import TB4_LIDAR_MOUNT, Lidar2DCfg  # noqa: E402
from st_nav.sensors.noise import DatasheetUniformRangeNoise, DatasheetUniformRangeNoiseCfg  # noqa: E402

os.makedirs(args.out, exist_ok=True)
dev = args.device
N = args.num_envs
DT = 1.0 / args.physics_hz
DEC = int(round(0.2 / DT))                     # physics steps per 5 Hz control step
STATIC = [(1.9, 1.6, 0.17), (-0.6, -0.9, 0.10), (2.2, -1.2, 0.05)]
MOVER_R = 0.17
ref_walls = list(reference_scene_module(args.scene).WALLS)
spec = SceneSpec(walls=ref_walls, static_cylinders=STATIC, moving_cylinders=[MOVER_R])
lcfg = Lidar2DCfg()
lcfg.noise.enabled = False
sim, scene, robot, wid, lidar, (mover,) = make_nav_scene(N, DT, dev, spec, lcfg)
origins = scene.env_origins
walls = walls_from_specs(ref_walls, device=dev)
static_c = torch.tensor(STATIC, device=dev)
AZ = torch.deg2rad(torch.tensor(lcfg.azimuths_deg, device=dev))
R = len(lcfg.azimuths_deg)
cm = CommandModel(N, DT, TB4_GEOMETRY, TB4_COMMAND_MODEL, device=dev)
res = {"num_envs": N, "physics_hz": args.physics_hz, "scene": args.scene, "beams": R}
ar = torch.arange(N, device=dev)


def circles_now():
    m = mover.data.root_pos_w[:, :2] - origins[:, :2]
    mc = torch.cat([m, torch.full((N, 1), MOVER_R, device=dev)], 1)
    return torch.cat([static_c[None].expand(N, -1, -1), mc[:, None]], 1)


def sample_free(n_cand=128, margin=0.25, extra_circles=None, wall_margin=None):
    lim = 2.835 - 0.05
    xy = (torch.rand(N, n_cand, 2, device=dev) * 2 - 1) * lim
    pts = xy.reshape(-1, 2)
    wc = wall_clearance(pts, walls).reshape(N, n_cand)
    cc = circle_clearance(pts, static_c[None].expand(pts.shape[0], -1, -1)).reshape(N, n_cand)
    ok = (wc >= (wall_margin or margin)) & (cc >= margin)
    if extra_circles is not None:
        ec = torch.linalg.norm(xy - extra_circles[:, None, :2], dim=-1) - extra_circles[:, None, 2]
        ok &= ec >= margin
    first = torch.argmax(ok.int(), dim=1)
    assert bool(ok.any(1).all()), "free-space sampling failed"
    return xy[ar, first]


def place_robot(xy, yaw):
    pos = torch.cat([origins[:, :2] + xy, torch.zeros(N, 1, device=dev)], 1)
    q = torch.stack([torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2)], 1)
    robot.write_root_pose_to_sim(torch.cat([pos, q], 1))
    robot.write_root_velocity_to_sim(torch.zeros(N, 6, device=dev))
    robot.write_joint_state_to_sim(torch.zeros(N, 2, device=dev), torch.zeros(N, 2, device=dev), joint_ids=wid)
    cm.reset()


def place_mover(xy):
    pos = torch.cat([origins[:, :2] + xy, torch.full((N, 1), spec.obstacle_height / 2 + 0.001, device=dev)], 1)
    q = torch.tensor([[1.0, 0, 0, 0]], device=dev).expand(N, -1)
    mover.write_root_pose_to_sim(torch.cat([pos, q], 1))
    mover.write_root_velocity_to_sim(torch.zeros(N, 6, device=dev))


def physics(steps, v=None, w=None):
    for _ in range(steps):
        cm.set_command(v if v is not None else torch.zeros(N, device=dev), w if w is not None else torch.zeros(N, device=dev))
        wl, wr, _, _ = cm.step()
        robot.set_joint_velocity_target(torch.stack([wl, wr], -1), joint_ids=wid)
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(DT)


def compare():
    """Ray caster (clean) vs oracle at the sensor pose reported by the ray caster (from PhysX)."""
    out = lidar.read()
    sp = lidar.origin_w - origins
    _, _, yaw = euler_xyz_from_quat(lidar.rc.data.quat_w)
    circ = circles_now()
    ang = yaw[:, None] + AZ[None]
    a_all = raycast_2d(sp[:, :2], ang, walls, circ, max_range=lidar.range_max)
    a_wall = raycast_2d(sp[:, :2], ang, walls, None, max_range=lidar.range_max)
    a_stat = raycast_2d(sp[:, :2], ang, None, circ[:, :-1], max_range=lidar.range_max)
    a_mov = raycast_2d(sp[:, :2], ang, None, circ[:, -1:], max_range=lidar.range_max)
    kind = torch.where(a_all >= lidar.range_max, 0,
                       torch.where((a_wall <= a_stat) & (a_wall <= a_mov), 1, torch.where(a_stat <= a_mov, 2, 3)))
    # sensor origin inside / touching an obstacle (possible while driving: the test mover has no collision)
    inside = (wall_clearance(sp[:, :2], walls) <= 1e-3) | (circle_clearance(sp[:, :2], circ) <= 1e-3)
    return out, a_all, kind, sp, yaw, inside


def err_stats(e, kind, hit, a):
    d = {}
    e = e.abs()
    for name, sel in (("all", torch.ones_like(kind, dtype=torch.bool)), ("wall", kind == 1),
                      ("static_cylinder", kind == 2), ("moving_cylinder", kind == 3), ("no_return", kind == 0)):
        x = e[sel]
        d[name] = {"n": int(x.numel()), "max_abs_m": float(x.max()) if x.numel() else None,
                   "p99_abs_m": float(torch.quantile(x[:min(x.numel(), 10_000_000)].float(), 0.99)) if x.numel() else None,
                   "mean_abs_m": float(x.mean()) if x.numel() else None}
    d["hit_flag_mismatch"] = int(((a < lidar.range_max) != hit).sum())
    return d


# ---------------------------------------------------------------- settle ------------------------------------------------
place_robot(sample_free(), torch.zeros(N, device=dev))
place_mover(torch.full((N, 2), 10.0, device=dev))
physics(int(0.5 / DT))

# ---------------------------------------------------------------- V1 ----------------------------------------------------
v1_err, v1_kind, v1_hit, v1_a, snaps = [], [], [], [], []
for rnd in range(8):
    mxy = sample_free(margin=0.30)
    mc = torch.cat([mxy, torch.full((N, 1), MOVER_R, device=dev)], 1)
    rxy = sample_free(margin=0.25, extra_circles=mc)
    yaw = (torch.rand(N, device=dev) * 2 - 1) * math.pi
    place_mover(mxy)
    place_robot(rxy, yaw)
    physics(int(0.1 / DT))
    out, a, kind, sp, syaw, inside = compare()
    v1_err.append(out.ranges_clean - a)
    v1_kind.append(kind)
    v1_hit.append(out.hit)
    v1_a.append(a)
    if rnd == 0 and args.debug:
        torch.set_printoptions(precision=3, linewidth=200)
        print("[DBG] sensor xy/yaw", sp[0, :2].tolist(), float(syaw[0]), "base", (robot.data.root_pos_w[0] - origins[0]).tolist(),
              "mover", (mover.data.root_pos_w[0] - origins[0]).tolist(), flush=True)
        rc = lidar.rc
        print("[DBG] rc offset", [t[0].tolist() for t in rc._offset], flush=True)
        print("[DBG] rc pos_w", rc.data.pos_w[0].tolist(), "quat_w", rc.data.quat_w[0].tolist(), flush=True)
        print("[DBG] base pos_w", robot.data.root_pos_w[0].tolist(), "quat", robot.data.root_quat_w[0].tolist(), flush=True)
        print("[DBG] ray_starts_w[0,:3]", rc._ray_starts_w[0, :3].tolist(), flush=True)
        print("[DBG] view type", type(rc._view).__name__, flush=True)
        print("[DBG] lidar  ", out.ranges_clean[0], flush=True)
        print("[DBG] oracle ", a[0], flush=True)
        print("[DBG] kind   ", kind[0], flush=True)
        print("[DBG] per-env mean abs err (first 16)", (out.ranges_clean - a).abs().mean(1)[:16], flush=True)
        os._exit(0)
    if rnd == 0:
        snaps = dict(sensor_xy=sp[:8, :2].cpu().numpy(), sensor_yaw=syaw[:8].cpu().numpy(),
                     base_xy=(robot.data.root_pos_w[:8, :2] - origins[:8, :2]).cpu().numpy(),
                     base_yaw=euler_xyz_from_quat(robot.data.root_quat_w[:8])[2].cpu().numpy(),
                     mover_xy=(mover.data.root_pos_w[:8, :2] - origins[:8, :2]).cpu().numpy(),
                     ranges=out.ranges_clean[:8].cpu().numpy(), analytic=a[:8].cpu().numpy())
E1, K1, H1, A1 = (torch.cat(x) for x in (v1_err, v1_kind, v1_hit, v1_a))
res["V1_random_poses"] = {"rounds": 8, "rays": int(E1.numel()), **err_stats(E1, K1, H1, A1)}
print("[V1]", json.dumps(res["V1_random_poses"]), flush=True)

# ---------------------------------------------------------------- V2 beam ordering --------------------------------------
k_exp = ar % R
yaw = (torch.rand(N, device=dev) * 2 - 1) * math.pi
rxy = sample_free(margin=0.25, wall_margin=0.6)
bear = yaw + AZ[k_exp]
lid_xy = rxy + TB4_LIDAR_MOUNT[0] * torch.stack([torch.cos(yaw), torch.sin(yaw)], 1)
mxy = lid_xy + 0.5 * torch.stack([torch.cos(bear), torch.sin(bear)], 1)
place_mover(mxy)
place_robot(rxy, yaw)
physics(int(0.1 / DT))
out, a, kind, _, _, _ = compare()
arg = torch.argmin(out.ranges_clean, dim=1)
arg_a = torch.argmin(a, dim=1)
ok_mover = kind[ar, k_exp] == 3                      # the expected beam actually sees the mover (not occluded)
res["V2_beam_ordering"] = {"envs": N, "mover_visible_on_expected_beam": int(ok_mover.sum()),
                           "argmin_equals_expected_beam": int(((arg == k_exp) & ok_mover).sum()),
                           "argmin_equals_oracle_argmin": int((arg == arg_a).sum()),
                           "max_abs_err_expected_beam_m": float((out.ranges_clean[ar, k_exp] - a[ar, k_exp]).abs()[ok_mover].max()),
                           "expected_range_m": "0.5 - 0.17 = 0.33 (mover centre on the beam)",
                           "mean_range_expected_beam_m": float(out.ranges_clean[ar, k_exp][ok_mover].mean())}
print("[V2]", json.dumps(res["V2_beam_ordering"]), flush=True)
v2 = dict(k_exp=k_exp[:40].cpu().numpy(), ranges=out.ranges_clean[:40].cpu().numpy())

# ---------------------------------------------------------------- V3 moving robot + moving obstacle ---------------------
rxy = sample_free(margin=0.35)
place_robot(rxy, (torch.rand(N, device=dev) * 2 - 1) * math.pi)
place_mover(torch.full((N, 2), 10.0, device=dev))
physics(int(0.2 / DT))
c0 = sample_free(margin=0.6)
phase = torch.rand(N, device=dev) * 2 * math.pi
v3_e, v3_k, v3_h, v3_a, v3_bad = [], [], [], [], []
vcmd = (torch.rand(N, device=dev) * 2 - 1) * 0.22
wcmd = (torch.rand(N, device=dev) * 2 - 1) * 1.0
v3_trace = []
for k in range(50):                                   # 10 s of control steps
    if k % 10 == 0:
        vcmd = (torch.rand(N, device=dev) * 2 - 1) * 0.22
        wcmd = (torch.rand(N, device=dev) * 2 - 1) * 1.0
    th = phase + 0.12 * k * 0.2 / 0.4                  # mover on a 0.4 m circle at 0.12 m/s
    place_mover(c0 + 0.4 * torch.stack([torch.cos(th), torch.sin(th)], 1))
    physics(DEC, vcmd, wcmd)
    out, a, kind, sp, syaw, inside = compare()
    r_, p_, _ = euler_xyz_from_quat(robot.data.root_quat_w)
    tilt = torch.rad2deg(torch.maximum(torch.sin(r_).abs().asin(), torch.sin(p_).abs().asin()))
    v3_bad.append(inside | (tilt > 2.0))
    v3_e.append(out.ranges_clean - a)
    v3_k.append(kind)
    v3_h.append(out.hit)
    v3_a.append(a)
    v3_trace.append(dict(sensor_xy=sp[0, :2].cpu().numpy(), yaw=float(syaw[0]),
                         mover_xy=(mover.data.root_pos_w[0, :2] - origins[0, :2]).cpu().numpy(),
                         ranges=out.ranges_clean[0].cpu().numpy(), analytic=a[0].cpu().numpy()))
E3, K3, H3, A3 = (torch.cat(x) for x in (v3_e, v3_k, v3_h, v3_a))
moved = torch.linalg.norm(robot.data.root_pos_w[:, :2] - origins[:, :2] - rxy, dim=1)
B3 = torch.cat(v3_bad)
res["V3_moving"] = {"control_steps": 50, "rays": int(E3.numel()), "robot_displacement_mean_m": float(moved.mean()),
                    "scans_with_origin_inside_obstacle_or_tilt_gt_2deg": int(B3.sum()), "scans": int(B3.numel()),
                    "valid_scans": err_stats(E3[~B3], K3[~B3], H3[~B3], A3[~B3]),
                    "excluded_scans": err_stats(E3[B3], K3[B3], H3[B3], A3[B3]) if bool(B3.any()) else None}
print("[V3]", json.dumps(res["V3_moving"]), flush=True)

# ---------------------------------------------------------------- V4 range limits and mounting --------------------------
noret = ~H1                                   # rays the GPU ray caster reports without a return
G1 = E1 + A1                                   # GPU clean ranges
res["V4_range_and_mount"] = {
    "gpu_no_return_rays": int(noret.sum()),
    "gpu_no_return_reported_as_range_max": bool(torch.all((lidar.range_max - G1[noret]).abs() < 1e-6)),
    "oracle_no_return_rays": int((K1 == 0).sum()),
    "no_return_disagreements_gpu_vs_oracle": int(((K1 == 0) & H1).sum() + ((K1 != 0) & ~H1).sum()),
    "max_reported_range_m": float((E1 + A1).max()), "min_reported_range_m": float((E1 + A1).min())}
off = torch.tensor(TB4_LIDAR_MOUNT, device=dev).expand(N, -1)
expect = robot.data.root_pos_w + quat_apply(robot.data.root_quat_w, off)
res["V4_range_and_mount"]["origin_vs_base_plus_urdf_offset_max_m"] = float(
    torch.linalg.norm(lidar.origin_w - expect, dim=1).max())
res["V4_range_and_mount"]["origin_vs_raycaster_ray_starts_max_m"] = float(
    torch.linalg.norm(lidar.rc._ray_starts_w - lidar.origin_w[:, None], dim=-1).max())
res["V4_range_and_mount"]["origin_height_above_ground_mean_m"] = float(lidar.origin_w[:, 2].mean())
res["V4_range_and_mount"]["base_link_height_mean_m"] = float(robot.data.root_pos_w[:, 2].mean())
print("[V4]", json.dumps(res["V4_range_and_mount"]), flush=True)

# ---------------------------------------------------------------- V5 noise statistics -----------------------------------
noise = DatasheetUniformRangeNoise(DatasheetUniformRangeNoiseCfg(seed=1234), dev)
out = lidar.read()
d, hit = out.ranges_clean, out.hit
M = 200
draws = torch.stack([noise(d, hit, lidar.range_max) for _ in range(M)])          # (M, N, R)
e = draws - d[None]
a_rel = noise.relative_accuracy(d)
unclipped = hit & (d * (1 + a_rel) <= lidar.range_max)
u = (e / (a_rel * d).clamp(min=1e-9))[:, unclipped]                                  # should be U(-1, 1)
us = u.flatten()
sub = us[torch.randperm(us.numel(), device=dev)[:200_000]].cpu().numpy()
ks = stats.kstest(sub, "uniform", args=(-1, 2))
perm = torch.randperm(us.numel(), device=dev)
ks_reps = [float(stats.kstest(us[perm[i * 200_000:(i + 1) * 200_000]].double().cpu().numpy(), "uniform", args=(-1, 2)).pvalue)
           for i in range(10)]
bands = {}
for lo, hi in ((0.0, 3.0), (3.0, 3.5)):
    sel = unclipped & (d > lo) & (d <= hi)
    eb = e[:, sel]
    db = d[sel][None].expand_as(eb)
    if eb.numel():
        bands[f"({lo},{hi}] m"] = {"n": int(eb.numel()), "expected_rel_halfwidth": float(noise.relative_accuracy(db[0:1, :1]).item()),
                                   "max_abs_rel": float((eb.abs() / db).max()), "std_rel": float((eb / db).std()),
                                   "expected_std_rel": float(noise.relative_accuracy(db[0:1, :1]).item() / math.sqrt(3))}
adj = float(torch.corrcoef(torch.stack([u[:, :-1].flatten()[:500000], u[:, 1:].flatten()[:500000]]))[0, 1]) \
    if u.shape[1] > 1 else float("nan")
seq = float(torch.corrcoef(torch.stack([u[:-1].flatten()[:500000], u[1:].flatten()[:500000]]))[0, 1])
clipped = hit & ~unclipped
res["V5_noise"] = {
    "model": "e ~ U(-a(d) d, +a(d) d), a = 1 % (d <= 3 m), 2 % (3 < d <= 5 m); per beam, per scan; clip [0, 3.5]",
    "draws": M, "samples_unclipped": int(us.numel()), "u_mean": float(us.mean()), "u_var": float(us.var()),
    "u_var_expected": 1 / 3, "u_min": float(us.min()), "u_max": float(us.max()),
    "ks_statistic_vs_U(-1,1)": float(ks.statistic), "ks_pvalue": float(ks.pvalue),
    "ks_pvalues_10_disjoint_subsamples_200k": ks_reps, "bands": bands,
    "corr_adjacent_beams": adj, "corr_successive_scans": seq,
    "no_return_rays_unchanged": bool(torch.all(e[:, ~hit] == 0)),
    "rays_near_cap_subject_to_clipping": int(clipped.sum()),
    "max_noisy_range_m": float(draws.max()), "min_noisy_range_m": float(draws.min())}
print("[V5]", json.dumps(res["V5_noise"]), flush=True)

np.savez_compressed(os.path.join(args.out, "m2_validation_data.npz"),
                    e1=E1[:, :].cpu().numpy().astype(np.float32), k1=K1.cpu().numpy().astype(np.int8),
                    a1=A1.cpu().numpy().astype(np.float32), e3=E3.cpu().numpy().astype(np.float32),
                    u_sample=sub.astype(np.float32), noisy_example=draws[:5, :4].cpu().numpy(), clean_example=d[:4].cpu().numpy(),
                    v2_k=v2["k_exp"], v2_ranges=v2["ranges"], azimuths_deg=np.array(lcfg.azimuths_deg),
                    walls=np.array([list(w) + ([0.15] if len(w) == 4 else []) for w in ref_walls]),
                    static=np.array(STATIC), mover_r=MOVER_R, mount=np.array(TB4_LIDAR_MOUNT),
                    **{f"snap_{k}": v for k, v in snaps.items()},
                    v3_sensor_xy=np.array([t["sensor_xy"] for t in v3_trace]), v3_yaw=np.array([t["yaw"] for t in v3_trace]),
                    v3_mover_xy=np.array([t["mover_xy"] for t in v3_trace]),
                    v3_ranges=np.array([t["ranges"] for t in v3_trace]), v3_analytic=np.array([t["analytic"] for t in v3_trace]))
with open(os.path.join(args.out, f"m2_validation_{int(args.physics_hz)}hz.json"), "w") as f:
    json.dump(res, f, indent=1)
print("[DONE]", flush=True)
os._exit(0)
