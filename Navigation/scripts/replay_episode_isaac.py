"""Isaac Sim (RTX) replay of a recorded evaluation episode: the scene is rebuilt from the episode folder only
(world_geometry.json walls / baffles, metadata.json heights and goal) and the TurtleBot4 (official USD, physics removed:
a visual model posed directly) and the moving obstacles are placed at their RECORDED poses at every control step
(trajectory.csv, obstacles.npz). No policy, no physics: the video shows exactly the evaluated episode.

    isaacrun -g <gpu> -- python Navigation/scripts/replay_episode_isaac.py --headless --enable_cameras \
        <eval_dir>/episode_0000 [--speed 2] [--compare]

Output (in the episode folder): <episode>_isaac.mp4: chase camera (behind the robot, smoothed heading) | overhead
camera, with time / speed / clearance / outcome overlay. One frame per control step (5 Hz) + 1.5 s hold at the end, the
same frame timing as tools/make_video.py, so the two videos run in sync.
--compare: also <episode>_recreation_<speed>x.mp4 (tools/make_video.py at the same speed, so frames match) and
<episode>_isaac_vs_recreation.mp4 = the two side by side.
Not shown: wheel rotation and body pitch / roll (not recorded); the base is placed on the floor (z = 0).
--lidar_points (3D LiDAR, per-scan recordings): the scan in the policy's observation at each step is drawn into the
camera images at the world points where it was measured (held between acquisitions, so a stale scan stays where the
scene was), coloured by range; projected with each camera's intrinsics / pose and depth-tested against the rendered
depth (hidden points are not drawn). Overlay: scan acquired at this step / held, and its age. Output <episode>_isaac_lidar.mp4.
"""
import argparse
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time

from isaaclab.app import AppLauncher

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "tools"))

p = argparse.ArgumentParser()
p.add_argument("episode_dir")
p.add_argument("--speed", type=float, default=2.0, help="playback speed vs simulated time (default 2, as make_video)")
p.add_argument("--chase_res", type=int, nargs=2, default=(1280, 720))
p.add_argument("--top_res", type=int, default=720, help="overhead camera (square) resolution")
p.add_argument("--renders_per_frame", type=int, default=3, help="RTX updates per video frame (temporal convergence)")
p.add_argument("--compare", action="store_true", help="also write the side-by-side with tools/make_video.py's video")
p.add_argument("--out", default=None)
p.add_argument("--lidar_points", action="store_true", help="draw the observed LiDAR scan into the camera images")
p.add_argument("--lidar_views", default="both", choices=("chase", "both"), help="--lidar_points: chase camera or both")
p.add_argument("--point_px", type=int, default=2, help="--lidar_points: point radius (pixels) in the chase view")
p.add_argument("--max_frames", type=int, default=None, help="render only the first N frames (tests)")
p.add_argument("--depth_max", type=float, default=8.0, help="--lidar_points: range of the colour scale (m)")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from pxr import Gf, UsdGeom, UsdPhysics  # noqa: E402

import episode_data as ED  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors.camera import Camera, CameraCfg  # noqa: E402
from isaaclab.sim.utils import get_current_stage  # noqa: E402
from make_video import ffmpeg_path  # noqa: E402
from st_nav.assets.turtlebot4 import TB4_USD_PATH  # noqa: E402

ep_dir = os.path.abspath(args.episode_dir)
ep = ED.load_episode(ep_dir)
meta, geo, T, O = ep["meta"], ep["geometry"], ep["traj"], ep["obstacles"]
H = meta.get("geometry_heights_m", {})
dt = float(meta["control"]["control_dt_s"])
n = int(meta["n_steps"])
frames = n + 1 + int(1.5 / dt)
if args.max_frames:
    frames = min(frames, args.max_frames)
name = os.path.basename(ep_dir.rstrip("/"))
out = args.out or os.path.join(ep_dir, f"{name}_isaac{'_lidar' if args.lidar_points else ''}.mp4")
if args.lidar_points and not ED.is_per_scan(ep):
    raise SystemExit("--lidar_points needs a per-scan (3D LiDAR) recording")
dev = args.device

# ------------------------------------------------------------------------------------------------ scene
sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device=dev, use_fabric=False,
                                                          render=sim_utils.RenderCfg(rendering_mode="quality")))
stage = get_current_stage()
sim_utils.GroundPlaneCfg(color=(0.32, 0.33, 0.35)).func("/World/ground", sim_utils.GroundPlaneCfg(color=(0.32, 0.33, 0.35)))
dome = sim_utils.DomeLightCfg(intensity=900.0, color=(0.95, 0.97, 1.0))
dome.func("/World/dome", dome)
sun = sim_utils.DistantLightCfg(intensity=2200.0, angle=1.0)
sun.func("/World/sun", sun, orientation=(0.88, 0.25, 0.25, 0.30))


def yaw_quat(yaw):
    return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))


for i, w in enumerate(geo["walls"]):
    baffle = w.get("name", "").startswith("Z")
    h = float(H.get("baffle" if baffle else "wall", 0.6))
    c = sim_utils.CuboidCfg(size=(w["length"], w["thickness"], h), visual_material=sim_utils.PreviewSurfaceCfg(
        diffuse_color=(0.92, 0.55, 0.18) if baffle else (0.78, 0.78, 0.80), roughness=0.7))
    c.func(f"/World/walls/w{i:02d}", c, translation=(w["cx"], w["cy"], h / 2), orientation=yaw_quat(w["yaw"]))
g = meta["goal"]
gc = sim_utils.CylinderCfg(radius=float(meta["terminations"]["goal_tolerance_m"]), height=0.004, axis="Z",
                           visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.75, 0.25), opacity=0.8))
gc.func("/World/goal", gc, translation=(g["x"], g["y"], 0.002))
obst_paths = []
h_ob = float(H.get("moving_obstacle", 0.5))
for j in range(len(O["ids"])):
    if not bool(O["active"][j]):
        continue
    oc = sim_utils.CylinderCfg(radius=float(O["radius"][j]), height=h_ob, axis="Z",
                               visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.30, 0.75), roughness=0.5))
    path = f"/World/obstacles/{O['ids'][j]}"
    oc.func(path, oc, translation=(float(O["x"][0, j]), float(O["y"][0, j]), h_ob / 2))
    obst_paths.append((j, path))

# robot: official USD as a visual model (physics schemas removed, so it is posed directly)
rob = sim_utils.UsdFileCfg(usd_path=TB4_USD_PATH)
rob.func("/World/Robot", rob, translation=(float(T["x"][0]), float(T["y"][0]), 0.0), orientation=yaw_quat(float(T["yaw"][0])))
for prim in stage.Traverse():
    if not str(prim.GetPath()).startswith("/World/Robot"):
        continue
    for api in (UsdPhysics.ArticulationRootAPI, UsdPhysics.RigidBodyAPI, UsdPhysics.MassAPI, UsdPhysics.CollisionAPI):
        if prim.HasAPI(api):
            prim.RemoveAPI(api)
    if prim.IsA(UsdPhysics.Joint):
        UsdPhysics.Joint(prim).CreateJointEnabledAttr(False)
    for a in prim.GetAppliedSchemas():
        if a.startswith("Physx"):
            prim.RemoveAppliedSchema(a)


def set_pose(path, x, y, z, yaw=None):
    prim = stage.GetPrimAtPath(path)
    prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(x, y, z))
    if yaw is not None:
        q = yaw_quat(yaw)
        prim.GetAttribute("xformOp:orient").Set(Gf.Quatd(q[0], Gf.Vec3d(*q[1:])))


cw, ch = args.chase_res
DT_ = ["rgb", "distance_to_image_plane"] if args.lidar_points else ["rgb"]
cam_c = Camera(CameraCfg(prim_path="/World/cam_chase", height=ch, width=cw, data_types=DT_,
                         update_latest_camera_pose=args.lidar_points,
                         spawn=sim_utils.PinholeCameraCfg(focal_length=16.0, clipping_range=(0.05, 50.0))))
cam_t = Camera(CameraCfg(prim_path="/World/cam_top", height=args.top_res, width=args.top_res, data_types=DT_,
                         update_latest_camera_pose=args.lidar_points,
                         spawn=sim_utils.PinholeCameraCfg(focal_length=15.0, clipping_range=(0.1, 50.0))))
t0 = time.time()
sim.reset()


def heading_smoothed(tau=1.2):
    yaw = np.unwrap(np.asarray(T["yaw"], float))
    out_ = yaw.copy()
    a = 1 - math.exp(-dt / tau)
    for i in range(1, len(yaw)):
        out_[i] = out_[i - 1] + a * (yaw[i] - out_[i - 1])
    return out_


HEAD = heading_smoothed()
tt = lambda v: torch.tensor([v], dtype=torch.float32, device=dev)  # noqa: E731
half = float(geo.get("arena_interior_half_m", 2.835))
cam_t.set_world_poses_from_view(tt([0.0, -0.9, 7.4]), tt([0.0, 0.0, 0.0]))


def place(k):
    r = min(k, n)
    x, y, yaw = float(T["x"][r]), float(T["y"][r]), float(T["yaw"][r])
    set_pose("/World/Robot", x, y, 0.0, yaw)
    for j, path in obst_paths:
        rr = min(r, O["x"].shape[0] - 1)
        while rr > 0 and not np.isfinite(O["x"][rr, j]):
            rr -= 1
        set_pose(path, float(O["x"][rr, j]), float(O["y"][rr, j]), h_ob / 2)
    hd = float(HEAD[r])
    eye = [x - 1.1 * math.cos(hd), y - 1.1 * math.sin(hd), 1.9]       # high enough to see over the 0.6 m walls
    tgt = [x + 0.9 * math.cos(hd), y + 0.9 * math.sin(hd), 0.0]
    cam_c.set_world_poses_from_view(tt(eye), tt(tgt))
    return r


def rgb(cam):
    return cam.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)


CMAP = None
if args.lidar_points:
    import matplotlib
    CMAP = matplotlib.colormaps["turbo"]


def quat_to_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def draw_points(cam, img, P, rng, px, depth_test=True):
    """Project world points P (N, 3) into the camera image (ROS optical frame: x right, y down, z forward), keep those
    in front, inside the image and (depth_test) not behind the rendered surface, draw far to near, coloured by range.
    The overhead view draws without the depth test: the scan hits vertical faces, which a top-down camera cannot see."""
    K = cam.data.intrinsic_matrices[0].cpu().numpy()
    pos = cam.data.pos_w[0].cpu().numpy()
    Rm = quat_to_mat(cam.data.quat_w_ros[0].cpu().numpy())
    pc = (P - pos) @ Rm                                               # world -> camera (R^T (p - pos))
    z = pc[:, 2]
    ok = z > 0.05
    u = np.full(len(P), -1.0)
    v = np.full(len(P), -1.0)
    u[ok] = K[0, 0] * pc[ok, 0] / z[ok] + K[0, 2]
    v[ok] = K[1, 1] * pc[ok, 1] / z[ok] + K[1, 2]
    h, w = img.shape[:2]
    ui, vi = np.round(u).astype(int), np.round(v).astype(int)
    ok &= (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    D = cam.data.output["distance_to_image_plane"][0, ..., 0].cpu().numpy()
    d = np.full(len(P), np.inf)
    d[ok] = D[vi[ok], ui[ok]]
    if depth_test:                                                    # points lie on surfaces: small tolerance
        ok &= z <= np.where(np.isfinite(d), d, np.inf) + 0.03 + 0.02 * z
    idx = np.nonzero(ok)[0]
    idx = idx[np.argsort(-z[idx])]
    col = (CMAP(np.clip(rng[idx] / args.depth_max, 0, 1))[:, :3] * 255).astype(int)
    for i, c in zip(idx, col):
        cv2.circle(img, (int(ui[i]), int(vi[i])), px, (int(c[0]), int(c[1]), int(c[2])), -1, cv2.LINE_AA)
    return len(idx)


def lidar_state(r):
    """Scan in the policy's observation at row r: (points, ranges, acquired at this step, age s) or None."""
    s = ED.scan_index(ep, r)
    if s < 0:
        return None
    sc = ED.lidar_scan(ep, s)
    hit = sc["hit"]
    fresh = bool(ep["lidar"]["fresh"][r])
    return sc["points"][hit], sc["ranges"][hit], fresh, float(T["t_sim"][r]) - sc["t_sim"]


def overlay(img, lines, scale=0.8):
    y = 32
    for s in lines:
        cv2.putText(img, s, (14, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, s, (14, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
        y += int(34 * scale / 0.8)
    return img


place(0)
for _ in range(40):                                                  # warm-up: shaders, RTX accumulation
    sim.step(render=True)
cam_c.update(0.0); cam_t.update(0.0)
print(f"[REPLAY] setup {time.time() - t0:.0f} s; rendering {frames} frames", flush=True)
tmp = tempfile.mkdtemp(prefix="isaac_replay_")
title = f"{meta.get('exp_id', '')}  {meta.get('protocol', '')}  {ED.episode_dir_name(meta['episode_index'])}"
t1 = time.time()
try:
    for k in range(frames):
        r = place(k)
        for _ in range(args.renders_per_frame):
            sim.step(render=True)
        cam_c.update(dt); cam_t.update(dt)
        a, b = rgb(cam_c), rgb(cam_t)
        lid_line = []
        if args.lidar_points:
            st = lidar_state(r)
            if st is not None:
                P, rr, fresh, age = st
                draw_points(cam_c, a, P, rr, args.point_px)
                if args.lidar_views == "both":
                    draw_points(cam_t, b, P, rr, max(args.point_px - 1, 1), depth_test=False)
                lid_line = ["LiDAR scan ACQUIRED" if fresh else f"LiDAR scan held, age {age:3.1f} s"]
        if b.shape[0] != a.shape[0]:
            b = cv2.resize(b, (int(b.shape[1] * a.shape[0] / b.shape[0]), a.shape[0]))
        done = k >= n
        overlay(a, [title, f"t {float(T['t_sim'][r]):5.1f} s   v {float(T['v_mps'][r]):4.2f} m/s   "
                           f"clearance {float(T['clearance_m'][r]):5.3f} m"
                    + (f"   {meta['outcome'].upper().replace('_', ' ')}" if done else ""),
                    f"Isaac Sim RTX replay of the recorded poses ({args.speed:g}x)"] + lid_line)
        overlay(b, ["overhead"], 0.7)
        cv2.imwrite(os.path.join(tmp, f"f{k:05d}.png"), cv2.cvtColor(np.concatenate([a, b], 1), cv2.COLOR_RGB2BGR))
        if k % 25 == 0:
            print(f"[REPLAY] frame {k}/{frames}  {time.time() - t1:.0f} s", flush=True)
    subprocess.run([ffmpeg_path(), "-y", "-loglevel", "error", "-framerate", f"{args.speed / dt:g}", "-i",
                    os.path.join(tmp, "f%05d.png"), "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p",
                    "-c:v", "libx264", "-crf", "18", out], check=True)
finally:
    shutil.rmtree(tmp, ignore_errors=True)
print(f"[REPLAY] {out} ({frames} frames, {time.time() - t1:.0f} s)", flush=True)

if args.compare:
    rec = os.path.join(ep_dir, f"{name}_recreation_{args.speed:g}x.mp4")      # same speed = same frame timing
    subprocess.run([sys.executable, os.path.join(ROOT, "tools", "make_video.py"), ep_dir, "--speed", f"{args.speed:g}",
                    "--out", rec], check=True)
    cmp_out = os.path.join(ep_dir, f"{name}_isaac_vs_recreation.mp4")
    subprocess.run([ffmpeg_path(), "-y", "-loglevel", "error", "-i", out, "-i", rec, "-filter_complex",
                    "[0:v]scale=-2:720[a];[1:v]scale=-2:720[b];[a][b]hstack=inputs=2,pad=ceil(iw/2)*2:ceil(ih/2)*2",
                    "-pix_fmt", "yuv420p", "-c:v", "libx264", "-crf", "20", cmp_out], check=True)
    print(f"[REPLAY] {cmp_out}", flush=True)
os._exit(0)
