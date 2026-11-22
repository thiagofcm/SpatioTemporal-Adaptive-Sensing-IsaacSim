"""Read-only: print every xformOp and the world-axis-aligned bounds of each Gprim in the given USD files."""
import argparse
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("files", nargs="+")
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
app = AppLauncher(a).app

from pxr import Usd, UsdGeom

for f in a.files:
    st = Usd.Stage.Open(f)
    print(f"\n######## {f}", flush=True)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    for prim in st.Traverse():
        if not prim.IsA(UsdGeom.Gprim):
            continue
        ops = [(op.GetOpName(), op.Get()) for op in UsdGeom.Xformable(prim).GetOrderedXformOps()]
        r = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        mn, mx = r.GetMin(), r.GetMax()
        print(f"  {prim.GetPath()} [{prim.GetTypeName()}]  x[{mn[0]:+.3f},{mx[0]:+.3f}] y[{mn[1]:+.3f},{mx[1]:+.3f}] "
              f"z[{mn[2]:+.3f},{mx[2]:+.3f}]", flush=True)
        for n, v in ops:
            print(f"      {n} = {v}", flush=True)
app.close()
