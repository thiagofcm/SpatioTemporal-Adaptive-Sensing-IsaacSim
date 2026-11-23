"""M7 validation of the fixed 3D LiDAR (Mirrorcle raster approximation) in the full navigation env.

    isaacrun -g <gpu> -- python tests/sim/lidar3d_validation.py --headless --scene static  [--num_envs 64]
    isaacrun -g <gpu> -- python tests/sim/lidar3d_validation.py --headless --scene dynamic
    # physics / rewards / terminations independent of the sensor: same actions, 2D vs 3D (run both, then compare)
    isaacrun -g <gpu> -- python tests/sim/lidar3d_validation.py --headless --scene static --sensor lidar2d --traj-only

L1 structure: 4726 rays, observation 4726 + 7, floor is a ray-cast target, ray caster offset = configured mount
L2 mount and orientation: ray caster directions = sensor-frame grid rotated by the mount (pitch +5 deg = DOWN), origin
   above the top platform, floor returns at z = 0, boresight floor distance
L3 geometric accuracy (noise off, robots driving with random actions): ray caster vs the analytic 3D oracle (boxes with
   height, finite cylinders, floor plane), per hit class
L4 acquisition schedule: 4 Hz due times, scans at control instants (existing scheduler), observation held between scans
L5 noise on: observed - clean = sigma * z with z the keyed Box-Muller draw (recomputed independently), statistics,
   no-return rays untouched
L6 the same action sequence gives identical robot poses / rewards / terminations as the 2D sensor (needs the
   --sensor lidar2d --traj-only dump of the same scene and num_envs)
Writes tests/sim/results/lidar3d_validation_<scene>.json (+ figure).
"""
import argparse
import json
import math
import os

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--num_envs", type=int, default=64)
p.add_argument("--scene", choices=("static", "dynamic"), default="static")
p.add_argument("--sensor", choices=("lidar2d", "lidar3d"), default="lidar3d")
p.add_argument("--traj-only", action="store_true", help="only dump the L6 trajectory (for the 2D comparison)")
p.add_argument("--steps", type=int, default=60, help="L3 control steps with random actions")
p.add_argument("--out", default=os.path.join(os.environ.get("ST_ROOT", "."), "tests", "sim", "results"))
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import st_nav.tasks.nav2d  # noqa: E402,F401
from isaaclab.utils.math import quat_apply  # noqa: E402
from st_nav import crng  # noqa: E402
from st_nav.geometry.analytic3d import raycast_3d  # noqa: E402
from st_nav.scenes import corridor_dynamic_chase_random as D  # noqa: E402
from st_nav.sensors.lidar import FLOOR_PRIM  # noqa: E402
from st_nav.sensors.patterns import grid_pattern, rpy_to_quat_wxyz  # noqa: E402
from st_nav.tasks.nav2d.corridor_env_cfg import CorridorNavEnvCfg  # noqa: E402

os.makedirs(args.out, exist_ok=True)
cfg = CorridorNavEnvCfg()
cfg.scene.num_envs, cfg.sim.device, cfg.seed = args.num_envs, args.device, 11
cfg.sensor = args.sensor
cfg.lidar_noise = False                              # L1-L3 noise-free; L5 switches the noise model on at runtime
if args.scene == "dynamic":
    cfg.scene_name = "corridor_dynamic_chase_random"
    cfg.active_obstacles = (1, 2)
    cfg.obstacle_speed_dr = cfg.obstacle_y_dr = True
    cfg.goals = D.GOAL_LIST_ST04
genv = gym.make("STNav-TB4-Corridor-v0", cfg=cfg)
_IM = torch.inference_mode()
_IM.__enter__()
u = genv.unwrapped
corr, N, dev = u.corr, u.num_envs, u.device
lid = corr.lidar
checks, res = {}, {"scene": corr.cfg.scene_name, "num_envs": N, "sensor": args.sensor}


def ok(name, cond, detail=None):
    checks[name] = {"pass": bool(cond), "detail": detail}
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail if detail is not None else ''}", flush=True)


def actions(k):
    g = torch.Generator().manual_seed(1000 + k)
    a = torch.rand(N, 2, generator=g) * 2 - 1
    a[:, 0] = a[:, 0] * 0.5 + 0.5                    # mostly forward: the robots explore the corridor
    return a.to(dev)


# ---------------------------------------------------------------- L6 dump (first, identical in 2D and 3D runs)
obs, _ = genv.reset()
traj = {"pos": [], "quat": [], "rew": [], "term": [], "trunc": [], "clear": []}
for k in range(50):
    _, rew, term, trunc, _ = genv.step(actions(k))
    rd = corr.robot.data
    traj["pos"].append((rd.root_link_pos_w - corr.origins).cpu().numpy()); traj["quat"].append(rd.root_link_quat_w.cpu().numpy())
    traj["rew"].append(rew.cpu().numpy()); traj["term"].append(term.cpu().numpy()); traj["trunc"].append(trunc.cpu().numpy())
    traj["clear"].append(corr.footprint_clearance().cpu().numpy())
traj = {k: np.stack(v) for k, v in traj.items()}
dump = os.path.join(args.out, f"lidar3d_traj_{args.sensor}_{args.scene}_{N}.npz")
np.savez(dump, **traj)
print("trajectory dump:", dump, flush=True)
if args.traj_only:
    genv.close(); app.close(); raise SystemExit(0)
other = os.path.join(args.out, f"lidar3d_traj_lidar2d_{args.scene}_{N}.npz")
if os.path.exists(other):
    t2 = dict(np.load(other))
    diff = {k: float(np.abs(traj[k].astype(np.float64) - t2[k].astype(np.float64)).max()) for k in traj}
    ok("L6_same_actions_same_physics_rewards_terminations_as_2D", all(v == 0.0 for v in diff.values()),
       {"max_abs_diff": diff, "terminations": int(traj["term"].sum()), "truncations": int(traj["trunc"].sum())})
else:
    print(f"[SKIP] L6: run the --sensor lidar2d --traj-only dump first ({other})", flush=True)

# ---------------------------------------------------------------- L1 structure
lc, rc = lid.cfg, lid.rc
obs = u._get_observations()
ok("L1_rays_obs_dims", lid.num_rays == 4726 and u.cfg.observation_space == 4733 and obs["policy"].shape == (N, 4733)
   and tuple(lc.grid_shape) == (34, 139), {"num_rays": lid.num_rays, "obs": list(obs["policy"].shape)})
tgts = [getattr(t, "prim_expr", t) for t in rc.cfg.mesh_prim_paths]
ok("L1_floor_is_target", FLOOR_PRIM in tgts, tgts)
q_exp = rpy_to_quat_wxyz((0.0, 5.0, 0.0))
ok("L1_raycaster_offset_is_mount", np.allclose(rc.cfg.offset.pos, (0, 0, 0.3998)) and np.allclose(rc.cfg.offset.rot, q_exp)
   and lc.range_max == 12.0 and rc.cfg.max_distance == 12.0, {"pos": list(rc.cfg.offset.pos), "rot": list(rc.cfg.offset.rot)})

# ---------------------------------------------------------------- L2 mount / orientation
az, el, row, col = grid_pattern()
a_, e_ = torch.deg2rad(torch.tensor(az, device=dev)), torch.deg2rad(torch.tensor(el, device=dev))
dirs_s = torch.stack([torch.cos(e_) * torch.cos(a_), torch.cos(e_) * torch.sin(a_), torch.sin(e_)], -1)
rc_dirs = rc.ray_directions[0]                                   # what the GPU ray caster casts (base frame)
d_err = float((rc_dirs - lid.dirs_b).abs().max())
bore = quat_apply(torch.tensor(q_exp, device=dev)[None], torch.tensor([[1.0, 0, 0]], device=dev))[0]
ok("L2_raycaster_dirs_equal_rotated_grid", d_err < 1e-6 and float((lid.dirs_s - dirs_s).abs().max()) < 1e-6,
   {"max_abs": d_err})
ok("L2_pitch_plus5_points_down", abs(float(bore[2]) + math.sin(math.radians(5))) < 1e-6 and
   float(lid.dirs_b[:, 2].mean()) < -0.08, {"boresight_b": [round(float(v), 5) for v in bore],
                                            "mean_dir_z": round(float(lid.dirs_b[:, 2].mean()), 4)})
genv.reset()                                                     # robots at rest on their start poses
for _ in range(3):
    genv.step(torch.zeros(N, 2, device=dev))
data = rc.data
org = lid.origin_w
base = corr.robot.data.root_link_pos_w
org_rel_z = float((org[:, 2] - base[:, 2]).mean())
hits = data.ray_hits_w
H = float(org[:, 2].mean())
bore_w = quat_apply(data.quat_w, bore[None].expand(N, -1))
t_bore = (org[:, 2] / -bore_w[:, 2])
res["mount"] = {"origin_height_above_floor_m": H, "origin_minus_base_z_m": org_rel_z, "base_link_z_m": float(base[:, 2].mean()),
                "boresight_floor_distance_m": float(t_bore.mean()), "boresight_floor_distance_flat_m": H / math.tan(math.radians(5)),
                "lowest_ray_floor_distance_m": H / math.tan(math.radians(5 + 7 - 7 / 34))}
ok("L2_origin_above_top_platform", H > 0.3498 + 0.04 and abs(org_rel_z - 0.3998) < 0.01, res["mount"])

# ---------------------------------------------------------------- L3 accuracy vs the analytic oracle
W = corr.walls.shape[1]
hgt = torch.tensor([corr.cfg.wall_height] * 10 + [corr.cfg.baffle_height] * (W - 10), device=dev)
KIND = {1: "floor", 2: "box", 3: "cylinder"}
errs = {k: [] for k in ("floor", "wall", "baffle", "cylinder", "none")}
n_rays_total, kind_count = 0, {k: 0 for k in errs}
big, floor_zmax, n_mis, n_unexpl = [], 0.0, 0, 0
TOL_BOX, TOL_CYL = 1e-3, 2e-3      # edge tolerance: boxes exact; cylinders 32 facets (0.82 mm) + < 1 physics step of motion


def grown(boxes, cyl, eps_b, eps_c):
    b = boxes.clone(); b[..., 2:4] += eps_b; b[..., 5] += eps_b
    c = None if cyl is None else cyl.clone()
    if c is not None:
        c[..., 2] = torch.where(c[..., 2] > 0, c[..., 2] + eps_c, c[..., 2]); c[..., 3] += eps_c
    return b, c
for k in range(args.steps):
    genv.step(actions(100 + k))
    data = rc.data
    org = lid.origin_w
    d = torch.linalg.norm(data.ray_hits_w - org[:, None], dim=-1)
    d = torch.where(torch.isfinite(d) & (d <= 12.0), d, torch.full_like(d, 12.0))
    dirs_w = quat_apply(data.quat_w[:, None].expand(-1, lid.num_rays, -1).reshape(-1, 4), lid.dirs_b.repeat(N, 1)).view(N, -1, 3)
    boxes = torch.cat([corr.walls, hgt[None, :, None].expand(N, -1, -1)], -1)
    circ = corr.obstacle_circles()
    cyl = None if circ is None else torch.cat([circ, torch.full_like(circ[..., :1], corr.cfg.obstacle_height)], -1)
    o_rel = org - corr.origins
    t_or, kind = raycast_3d(o_rel, dirs_w, boxes, cyl, floor=True, max_range=12.0, return_kind=True)
    # which box: wall or baffle (recompute with walls only)
    t_w = raycast_3d(o_rel, dirs_w, boxes[:, :10], None, floor=False, max_range=12.0)
    e = (d - t_or)
    cls = torch.where(kind == 1, 0, torch.where(kind == 2, torch.where((t_w - t_or).abs() < 1e-6, 1, 2),
                                                torch.where(kind == 3, 3, 4)))
    for ci, name in enumerate(("floor", "wall", "baffle", "cylinder", "none")):
        m = cls == ci
        errs[name].append(e[m].abs().cpu())
        kind_count[name] += int(m.sum())
    fl = (kind == 1) & (e.abs() < 0.01)
    if fl.any():
        floor_zmax = max(floor_zmax, float(data.ray_hits_w[..., 2][fl].abs().max()))
    bad = e.abs() > 0.01
    if bad.any():                    # edge-grazing check: within the range of the primitives grown / shrunk by the tolerance
        bg, cg = grown(boxes, cyl, TOL_BOX, TOL_CYL)
        bs, cs = grown(boxes, cyl, -TOL_BOX, -TOL_CYL)
        t_g = raycast_3d(o_rel, dirs_w, bg, cg, floor=True, max_range=12.0)
        t_s = raycast_3d(o_rel, dirs_w, bs, cs, floor=True, max_range=12.0)
        lo, hi = torch.minimum(t_g, t_s) - 1e-3, torch.maximum(t_g, t_s) + 1e-3
        expl = (d >= lo) & (d <= hi)
        n_mis += int(bad.sum()); n_unexpl += int((bad & ~expl).sum())
    if bad.any() and len(big) < 2000:
        ii = torch.nonzero(bad)[:200]
        for i, r in ii.cpu().tolist():
            big.append({"env": i, "ray": r, "row": r // 139, "col": r % 139, "raycaster": float(d[i, r]), "oracle": float(t_or[i, r]),
                        "class": ("floor", "wall", "baffle", "cylinder", "none")[int(cls[i, r])]})
    n_rays_total += d.numel()
acc = {}
for name, v in errs.items():
    v = torch.cat(v).double() if v else torch.zeros(0)
    if v.numel():
        acc[name] = {"n": int(v.numel()), "p50_mm": 1e3 * float(v.quantile(0.5)) if v.numel() < 16_000_000 else None,
                     "p99_mm": 1e3 * float(torch.quantile(v[:16_000_000], 0.99)), "p999_mm": 1e3 * float(torch.quantile(v[:16_000_000], 0.999)),
                     "max_mm": 1e3 * float(v.max()), "frac_gt_1cm": float((v > 0.01).double().mean())}
res["accuracy"] = acc
res["mismatches_gt_1cm"] = {"n": n_mis, "unexplained_by_edge_tolerance": n_unexpl, "tol_box_m": TOL_BOX, "tol_cyl_m": TOL_CYL}
ok("L2_floor_returns_at_z0", floor_zmax < 1e-4 and kind_count["floor"] > 0, {"max_abs_z_m": floor_zmax, "n": kind_count["floor"]})
ok("L3_all_mismatches_are_edge_grazing_rays", n_unexpl == 0, res["mismatches_gt_1cm"])
res["hit_class_fraction"] = {k: v / n_rays_total for k, v in kind_count.items()}
res["mismatch_examples"] = big[:50]
print(json.dumps(acc, indent=1), flush=True)
print("hit classes:", res["hit_class_fraction"], flush=True)
for name in ("floor", "wall", "baffle"):
    if name in acc:
        ok(f"L3_{name}_matches_oracle", acc[name]["p999_mm"] < 1.0 and acc[name]["frac_gt_1cm"] < 1e-4, acc[name])
if "cylinder" in acc:
    ok("L3_cylinder_matches_oracle", acc["cylinder"]["p99_mm"] < 5.0 and acc["cylinder"]["frac_gt_1cm"] < 0.01, acc["cylinder"])
if "none" in acc:
    ok("L3_no_return_agrees", acc["none"]["frac_gt_1cm"] < 1e-4, acc["none"])

# ---------------------------------------------------------------- L4 schedule (+ L5 noise on)
lid.noise.cfg.enabled = True
genv.reset()
zero = torch.zeros(N, 2, device=dev)
t_acq, held_ok, changed_ok = [], True, True
noise_err, z_all, noret_ok = 0.0, [], True
prev = u.scan.clone()
for k in range(50):                                              # 10 s, robots standing
    o, _, term, trunc, _ = genv.step(zero)
    fresh = (u.scan_t - u.ep_time).abs() < 1e-6
    alive = ~(term | trunc)
    if bool(fresh[0]) and bool(alive[0]):
        t_acq.append(round(float(u.ep_time[0]), 4))
    blk = o["policy"][:, :lid.num_rays]
    held_ok &= bool(torch.equal(blk[alive & ~fresh], prev[alive & ~fresh]))
    if (alive & fresh).any():
        changed_ok &= bool((blk[alive & fresh] != prev[alive & fresh]).any(1).all())   # new noise draw on a new scan
        m = alive & fresh
        clean = u.scan_clean[m] * 12.0
        obs_r = u.scan[m] * 12.0
        tick = torch.round(u.ep_time[m] / u.physics_dt).to(torch.int64)
        seed = corr.lidar_seed[m]
        ray = torch.arange(lid.num_rays, device=dev)[None]
        u1 = crng.uniform(seed[:, None], tick[:, None], ray, 0); u2 = crng.uniform(seed[:, None], tick[:, None], ray, 1)
        z = torch.sqrt(-2.0 * torch.log1p(-u1)) * torch.cos(2.0 * torch.pi * u2)
        hit = clean < 12.0 - 1e-4
        exp = torch.where(hit, torch.clamp(clean + 0.01 * z, 0.0, 12.0), clean)
        noise_err = max(noise_err, float((obs_r - exp).abs().max()))
        noret_ok &= bool(torch.equal(obs_r[~hit], clean[~hit]))
        z_all.append(((obs_r - clean)[hit] / 0.01).cpu())
    prev = blk.clone()
dt_acq = np.diff(t_acq)
due = np.arange(0, 10.0001, 0.25)
expected = sorted({round(math.ceil(round(t / 0.2, 6)) * 0.2, 4) for t in due if math.ceil(round(t / 0.2, 6)) * 0.2 <= 10.0 + 1e-9})
res["schedule"] = {"acquisitions_env0": t_acq, "intervals": sorted(set(np.round(dt_acq, 4).tolist())), "count": len(t_acq)}
ok("L4_4Hz_control_quantized", t_acq == [t for t in expected if t > 0] and set(np.round(dt_acq, 4)) <= {0.2, 0.4}
   and abs(len(t_acq) / 10.0 - 4.0) <= 0.1, {"count_10s": len(t_acq), "intervals": res["schedule"]["intervals"],
                                            "first": t_acq[:8]})
ok("L4_observation_held_between_scans_and_new_at_scans", held_ok and changed_ok)
z = torch.cat(z_all).double()
res["noise"] = {"n": int(z.numel()), "mean_sigma_units": float(z.mean()), "std_sigma_units": float(z.std()),
                "max_abs_vs_independent_recomputation_m": noise_err}
ok("L5_noise_equals_keyed_gaussian_draws", noise_err < 5e-6 and noret_ok, {"max_abs_m": noise_err})
ok("L5_noise_statistics", abs(float(z.mean())) < 0.01 and abs(float(z.std()) - 1.0) < 0.01, res["noise"])

# ---------------------------------------------------------------- figure: one range image + top-down returns
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    e0 = 0
    rng_img = np.full((34, 139), np.nan)
    rng_img[np.array(row), np.array(col)] = (u.scan_clean[e0] * 12.0).cpu().numpy()
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5), gridspec_kw={"width_ratios": [2, 1]})
    im = ax[0].imshow(rng_img, cmap="viridis_r", vmin=0, vmax=12, aspect="auto", extent=(max(az), min(az), min(el), max(el)))
    ax[0].set_xlabel("azimuth [deg]"); ax[0].set_ylabel("elevation [deg]"); fig.colorbar(im, ax=ax[0]).set_label("range [m]")
    ax[0].set_title(f"{corr.cfg.scene_name}, env {e0}: noise-free range image")
    data = rc.data
    pts = (data.ray_hits_w[e0] - corr.origins[e0]).cpu().numpy()
    f = np.isfinite(pts).all(1)
    sc = ax[1].scatter(pts[f, 0], pts[f, 1], c=pts[f, 2], s=1, cmap="plasma", vmin=0, vmax=0.6)
    ax[1].set_aspect("equal"); ax[1].set_xlim(-3, 3); ax[1].set_ylim(-3, 3); ax[1].set_title("returns, top-down (colour = z)")
    fig.colorbar(sc, ax=ax[1]).set_label("z [m]")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, f"lidar3d_validation_{args.scene}.png"), dpi=110)
except Exception as ex:  # noqa: BLE001
    print("figure failed:", ex)

n_pass = sum(c["pass"] for c in checks.values())
res["checks"] = checks
print(f"{n_pass}/{len(checks)} checks passed", flush=True)
with open(os.path.join(args.out, f"lidar3d_validation_{args.scene}.json"), "w") as fh:
    json.dump(res, fh, indent=1, default=float)
genv.close()
app.close()
