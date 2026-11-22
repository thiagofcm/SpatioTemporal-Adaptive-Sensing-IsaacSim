"""M3 smoke test: build, reset, step, reset again (kinematic baffle pose change mid-simulation), step."""
import argparse, os, time
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); p.add_argument("--num_envs", type=int, default=64)
AppLauncher.add_app_launcher_args(p); a = p.parse_args(); app = AppLauncher(a).app
import torch
from st_nav.envs.corridor_scene import CorridorScene, CorridorSceneCfg
t0 = time.time()
sc = CorridorScene(CorridorSceneCfg(num_envs=a.num_envs), a.device)
print("[SMOKE] built in", round(time.time() - t0, 1), "s", flush=True)
z = torch.zeros(sc.n, device=a.device)
for k in range(5):
    sc.step(z, z)
b0 = sc.baffles[0].data.root_pos_w[:, :2] - sc.origins[:, :2]
print("[SMOKE] Z1 pos (env 0..2)", b0[:3].tolist(), "expected", sc.walls[:3, 10, :2].tolist(), flush=True)
sc.reset(torch.arange(sc.n, device=a.device))
for k in range(5):
    sc.step(z + 0.1, z)
b1 = sc.baffles[0].data.root_pos_w[:, :2] - sc.origins[:, :2]
print("[SMOKE] after 2nd reset Z1 pos", b1[:3].tolist(), "expected", sc.walls[:3, 10, :2].tolist(), flush=True)
out = sc.read_lidar()
f = sc.collision_flags()
print("[SMOKE] lidar", out.ranges.shape, "min/max", float(out.ranges.min()), float(out.ranges.max()),
      "clearance min", float(f["clearance"].min()), "fallbacks", sc.offset_fallbacks, flush=True)
print("[SMOKE] OK", flush=True)
os._exit(0)
