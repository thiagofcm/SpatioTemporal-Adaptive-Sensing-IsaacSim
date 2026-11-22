"""Headless GPU ray-caster (LiDAR pattern) benchmark with no rendering and no network assets."""
import argparse, time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--steps", type=int, default=200)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors.ray_caster import RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass


@configclass
class SceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane")
    body = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Body",
        spawn=sim_utils.CuboidCfg(
            size=(0.3, 0.3, 0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5)),
    )
    lidar = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Body",
        mesh_prim_paths=["/World/ground"],
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.2)),
        ray_alignment="base",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=16, vertical_fov_range=(-15.0, 15.0), horizontal_fov_range=(-180.0, 180.0), horizontal_res=1.0
        ),
        max_distance=30.0,
        debug_vis=False,
    )


sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device=args.device))
scene = InteractiveScene(SceneCfg(num_envs=args.num_envs, env_spacing=2.0))
sim.reset()
lidar = scene["lidar"]
print(f"[DIAG] envs={args.num_envs} rays/env={lidar.num_rays} total_rays={args.num_envs*lidar.num_rays}", flush=True)
for _ in range(10):
    sim.step(); scene.update(0.01)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(args.steps):
    sim.step()
    scene.update(0.01)
    hits = lidar.data.ray_hits_w
torch.cuda.synchronize()
dt = time.time() - t0
d = torch.linalg.norm(hits - lidar.data.pos_w.unsqueeze(1), dim=-1)
valid = torch.isfinite(d)
print(f"[DIAG] device={hits.device} shape={tuple(hits.shape)} valid_frac={valid.float().mean():.3f} "
      f"range[min/mean]={d[valid].min():.3f}/{d[valid].mean():.3f}", flush=True)
print(f"[DIAG] {args.steps} sim+lidar steps in {dt:.2f}s -> {args.steps/dt:.1f} steps/s, "
      f"{args.steps*args.num_envs*lidar.num_rays/dt/1e6:.1f} Mrays/s", flush=True)
app.close()
