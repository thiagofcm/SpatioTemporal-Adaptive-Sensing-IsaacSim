"""Read-only inspection of the converted TB4 USD: prims (incl. instance proxies), colliders with world bounds,
frames (rplidar, base_footprint), joint drive and limit attributes."""
import argparse, os
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p); a = p.parse_args(); app = AppLauncher(a).app
from pxr import Usd, UsdGeom, UsdPhysics
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "usd", "turtlebot4_standard.usd")
st = Usd.Stage.Open(path)
cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.guide])
for prim in Usd.PrimRange(st.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
    pth = str(prim.GetPath()); d = pth.count("/")
    if prim.HasAPI(UsdPhysics.CollisionAPI):
        r = cache.ComputeWorldBound(prim).ComputeAlignedRange(); mn, mx = r.GetMin(), r.GetMax()
        ap = prim.GetAttribute("physics:approximation").Get() if prim.HasAPI(UsdPhysics.MeshCollisionAPI) else "-"
        print(f"[COLL] {pth} [{prim.GetTypeName()}] approx={ap} x[{mn[0]:+.3f},{mx[0]:+.3f}] y[{mn[1]:+.3f},{mx[1]:+.3f}] z[{mn[2]:+.3f},{mx[2]:+.3f}]", flush=True)
    elif d <= 3 and prim.IsA(UsdGeom.Xformable) and not prim.IsA(UsdGeom.Gprim):
        t = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        q = t.ExtractRotationQuat()
        print(f"[XFRM] {pth} pos={tuple(round(v,4) for v in t.ExtractTranslation())} quat_wxyz=({q.GetReal():.3f},{q.GetImaginary()[0]:.3f},{q.GetImaginary()[1]:.3f},{q.GetImaginary()[2]:.3f})", flush=True)
    if prim.IsA(UsdPhysics.RevoluteJoint):
        print(f"[JNT ] {pth} " + " ".join(f"{at.GetName()}={at.Get()}" for at in prim.GetAttributes() if any(k in at.GetName() for k in ("drive", "limit", "axis", "maxJointVelocity", "localPos"))), flush=True)
app.close()
