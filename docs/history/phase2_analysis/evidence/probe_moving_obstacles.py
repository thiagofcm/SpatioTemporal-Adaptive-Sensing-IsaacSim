"""Feasibility probe (not migration code): MultiMeshRayCaster vs. per-env moving kinematic obstacles.

Per env: static wall at x=1.5 (thickness 0.1), kinematic cube (0.2 m) at x=1.0 sweeping y=A*sin(w t + phase).
2D LiDAR (1 channel, 360 rays, 1 deg) at env origin, z=0.1. Forward ray (0 deg) expected range:
  0.90 m if |y_cube| < 0.1 (cube blocks), else 1.45 m (wall). Edge band |y|~0.1 is excluded.
"""
import argparse, math, time

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--num_envs", type=int, default=1024)
p.add_argument("--steps", type=int, default=300)
p.add_argument("--no_lidar", action="store_true")
p.add_argument("--no_track", action="store_true")
p.add_argument("--no_move", action="store_true")
p.add_argument("--dynamic", action="store_true")
p.add_argument("--vel", action="store_true")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

KIN = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)
OBST_RB = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False, disable_gravity=True) if args.dynamic else KIN


@configclass
class SceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane")
    base = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Base",
        spawn=sim_utils.CuboidCfg(size=(0.05, 0.05, 0.05), rigid_props=KIN,
                                  collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.1)),
    )
    wall = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Wall",
        spawn=sim_utils.CuboidCfg(size=(0.1, 3.0, 0.5), rigid_props=KIN,
                                  collision_props=sim_utils.CollisionPropertiesCfg()),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.5, 0.0, 0.25)),
    )
    obstacle = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle",
        spawn=sim_utils.CuboidCfg(size=(0.2, 0.2, 0.2), rigid_props=OBST_RB, mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
                                  collision_props=sim_utils.CollisionPropertiesCfg()),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.0, 0.0, 0.1)),
    )
    lidar = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Base",
        mesh_prim_paths=[
            "/World/ground",
            MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/Wall", track_mesh_transforms=False),
            MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/Obstacle", track_mesh_transforms=not args.no_track),
        ],
        ray_alignment="yaw",
        pattern_cfg=patterns.LidarPatternCfg(channels=1, vertical_fov_range=(0.0, 0.0),
                                             horizontal_fov_range=(-180.0, 180.0), horizontal_res=1.0),
        max_distance=3.0,
    )


dt = 1.0 / 120.0
sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=dt, device=args.device))
scfg = SceneCfg(num_envs=args.num_envs, env_spacing=4.0)
if args.no_lidar:
    scfg.lidar = None
scene = InteractiveScene(scfg)
sim.reset()
obst = scene["obstacle"]
lidar = None if args.no_lidar else scene["lidar"]
N = args.num_envs
print(f"[PROBE] dynamic={args.dynamic} vel={args.vel} envs={N} lidar={not args.no_lidar} track={not args.no_track} move={not args.no_move}", flush=True)

# forward ray index (angle closest to 0 deg)
fwd = 0 if lidar is None else int(torch.argmax(lidar.ray_directions[0][:, 0]))
phase = torch.linspace(0, 2 * math.pi, N, device=sim.device)
A, w = 0.6, 2 * math.pi * 0.5  # 0.6 m amplitude, 0.5 Hz -> up to ~1.9 m/s
origins = scene.env_origins
quat = torch.tensor([1.0, 0, 0, 0], device=sim.device).repeat(N, 1)

checked = wrong = 0
t_sim = 0.0
torch.cuda.synchronize(); t0 = time.time()
for k in range(args.steps):
    y = A * torch.sin(w * t_sim + phase)
    pose = torch.cat([origins + torch.stack([torch.full_like(y, 1.0), y, torch.full_like(y, 0.1)], -1), quat], -1)
    if not args.no_move:
        if args.vel:
            vy = A * w * torch.cos(w * t_sim + phase)
            v = torch.zeros(N, 6, device=sim.device); v[:, 1] = vy
            obst.write_root_velocity_to_sim(v)
        else:
            obst.write_root_pose_to_sim(pose)
    scene.write_data_to_sim()
    sim.step(render=False)
    scene.update(dt)
    t_sim += dt
    if lidar is None:
        continue
    d = torch.linalg.norm(lidar.data.ray_hits_w[:, fwd] - lidar.data.pos_w, dim=-1)
    yc = obst.data.root_pos_w[:, 1] - origins[:, 1]
    valid = (yc.abs() < 0.09) | (yc.abs() > 0.11)
    expect = torch.where(yc.abs() < 0.1, 0.90, 1.45)
    err = (d - expect).abs()
    checked += int(valid.sum()); wrong += int(((err > 0.02) & valid).sum())
    if k in (0, 50, 150):
        i = int(torch.argmin(yc.abs()))
        print(f"[PROBE] step {k}: env {i} y_cube={yc[i]:+.3f} fwd_range={d[i]:.3f} expect={expect[i]:.2f}", flush=True)
torch.cuda.synchronize(); el = time.time() - t0
print(f"[PROBE] checks={checked} mismatches={wrong} ({100*wrong/max(checked,1):.3f}%)", flush=True)
print(f"[PROBE] {args.steps} steps in {el:.2f}s -> {args.steps/el:.1f} steps/s, {args.steps*N/el/1e3:.1f}k env-steps/s, "
      f"{(args.steps*N*lidar.num_rays/el/1e6 if lidar else 0):.0f} Mrays/s", flush=True)
app.close()
