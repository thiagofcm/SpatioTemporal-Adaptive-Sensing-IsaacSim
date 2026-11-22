"""Read-only USD inspection: list prims, types, transforms, sizes, physics APIs, references."""
import argparse, sys
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("files", nargs="+")
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
app = AppLauncher(a).app

from pxr import Usd, UsdGeom, UsdPhysics

for f in a.files:
    print(f"\n######## {f}", flush=True)
    st = Usd.Stage.Open(f)
    if st is None:
        print("  could not open"); continue
    print("  defaultPrim:", st.GetDefaultPrim().GetPath() if st.GetDefaultPrim() else None,
          "upAxis:", UsdGeom.GetStageUpAxis(st), "metersPerUnit:", UsdGeom.GetStageMetersPerUnit(st))
    n = 0
    for prim in st.Traverse():
        n += 1
        if n > 120:
            print("  ... (truncated)"); break
        info = [prim.GetTypeName() or "-"]
        if prim.IsA(UsdGeom.Xformable):
            m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            t = m.ExtractTranslation()
            info.append(f"pos=({t[0]:.2f},{t[1]:.2f},{t[2]:.2f})")
            ops = {op.GetOpName(): op.Get() for op in UsdGeom.Xformable(prim).GetOrderedXformOps()}
            if "xformOp:scale" in ops:
                s = ops["xformOp:scale"]; info.append(f"scale=({s[0]:.2f},{s[1]:.2f},{s[2]:.2f})")
        if prim.IsA(UsdGeom.Cube):
            info.append(f"size={UsdGeom.Cube(prim).GetSizeAttr().Get()}")
        apis = [s for s in prim.GetAppliedSchemas()]
        if apis: info.append("apis=" + ",".join(apis))
        if prim.HasAuthoredReferences():
            refs = prim.GetMetadata("references")
            info.append(f"refs={refs}")
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            kin = UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Get()
            info.append(f"kinematic={kin}")
        if prim.GetTypeName() in ("RevoluteJoint",):
            info.append(f"body0={UsdPhysics.Joint(prim).GetBody0Rel().GetTargets()} body1={UsdPhysics.Joint(prim).GetBody1Rel().GetTargets()}")
        for attr in ("radius", "height", "horizontalFov", "maxRange", "minRange", "horizontalResolution", "highLod", "verticalFov",
                     "drive:angular:physics:damping", "drive:angular:physics:stiffness", "drive:angular:physics:maxForce"):
            at = prim.GetAttribute(attr)
            if at and at.HasAuthoredValue():
                info.append(f"{attr}={at.Get()}")
        print(f"  {prim.GetPath()}  " + "  ".join(info), flush=True)
app.close()
