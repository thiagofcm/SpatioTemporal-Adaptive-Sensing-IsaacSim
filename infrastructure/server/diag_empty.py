"""Diagnostic copy of create_empty.py: prints timing and dumps Python stacks if stuck."""
import argparse, faulthandler, sys, time

faulthandler.dump_traceback_later(75, repeat=True, file=sys.stderr)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=200)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
t0 = time.time()
app = AppLauncher(args).app
print(f"[DIAG] app launched in {time.time()-t0:.1f}s", flush=True)

import torch
from isaaclab.sim import SimulationCfg, SimulationContext

cfg = SimulationCfg(dt=0.01, device=args.device)
t0 = time.time()
sim = SimulationContext(cfg)
print(f"[DIAG] SimulationContext created in {time.time()-t0:.1f}s, device={sim.device}", flush=True)

import carb
s = carb.settings.get_settings()
for k in ["/physics/suppressReadback", "/physics/cudaDevice", "/renderer/multiGpu/enabled", "/app/window/enabled"]:
    print(f"[DIAG] {k} = {s.get(k)}", flush=True)

t0 = time.time()
sim.reset()
print(f"[DIAG] sim.reset() in {time.time()-t0:.1f}s", flush=True)

from omni.physx import get_physx_interface  # noqa
try:
    from pxr import PhysxSchema
    scene = sim.stage.GetPrimAtPath(sim.cfg.physics_prim_path)
    api = PhysxSchema.PhysxSceneAPI(scene)
    print(f"[DIAG] enableGPUDynamics={api.GetEnableGPUDynamicsAttr().Get()} broadphase={api.GetBroadphaseTypeAttr().Get()}", flush=True)
except Exception as e:  # noqa
    print("[DIAG] scene attr read failed", e)

t0 = time.time()
for i in range(args.steps):
    sim.step()
    if i in (0, 9, 99):
        print(f"[DIAG] step {i+1} done at {time.time()-t0:.2f}s", flush=True)
print(f"[DIAG] {args.steps} steps in {time.time()-t0:.2f}s", flush=True)
faulthandler.cancel_dump_traceback_later()
app.close()
