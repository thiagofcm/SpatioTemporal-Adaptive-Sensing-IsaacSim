"""Export the converted TurtleBot4 USD geometry (visual meshes, collision shapes as PhysX sees them, key frames) into
per-body triangle meshes, for offline (GPU-free) visualization. Read-only on the USD.

    isaacrun -g auto -- python infrastructure/robot_viz/scripts/export_robot_geometry.py --headless

Output: results/viz/tb4_geometry.npz (+ .glb files for interactive viewing in any glTF viewer)
"""
import argparse
import os

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--out", default=os.path.join(os.environ.get("ST_ROOT", "."), "infrastructure", "robot_viz", "results"))
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import trimesh  # noqa: E402
from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: E402

from st_nav.assets.turtlebot4 import TB4_USD_PATH  # noqa: E402

os.makedirs(args.out, exist_ok=True)
stage = Usd.Stage.Open(TB4_USD_PATH)
root = stage.GetDefaultPrim()
xc = UsdGeom.XformCache()
BODIES = ["base_link", "left_wheel", "right_wheel"]
body_prims = {b: stage.GetPrimAtPath(f"{root.GetPath()}/{b}") for b in BODIES}
body_inv = {b: np.array(xc.GetLocalToWorldTransform(p).GetInverse()).T for b, p in body_prims.items()}


def to_np(m):  # Gf.Matrix4d (row-vector convention) -> 4x4 column-vector matrix
    return np.array(m).T


def prim_mesh(prim):
    """Triangle mesh of a gprim in its own local frame (None if unsupported)."""
    if prim.IsA(UsdGeom.Mesh):
        m = UsdGeom.Mesh(prim)
        pts = np.array(m.GetPointsAttr().Get(), dtype=np.float64)
        counts = np.array(m.GetFaceVertexCountsAttr().Get())
        idx = np.array(m.GetFaceVertexIndicesAttr().Get())
        faces, k = [], 0
        for c in counts:
            f = idx[k:k + c]
            faces += [[f[0], f[i], f[i + 1]] for i in range(1, c - 1)]
            k += c
        return trimesh.Trimesh(pts, np.array(faces), process=False)
    if prim.IsA(UsdGeom.Cube):
        s = UsdGeom.Cube(prim).GetSizeAttr().Get()
        return trimesh.creation.box(extents=(s, s, s))
    if prim.IsA(UsdGeom.Sphere):
        return trimesh.creation.icosphere(subdivisions=2, radius=UsdGeom.Sphere(prim).GetRadiusAttr().Get())
    if prim.IsA(UsdGeom.Cylinder):
        c = UsdGeom.Cylinder(prim)
        m = trimesh.creation.cylinder(radius=c.GetRadiusAttr().Get(), height=c.GetHeightAttr().Get(), sections=32)
        ax = c.GetAxisAttr().Get()
        if ax == "X":
            m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
        elif ax == "Y":
            m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
        return m
    return None


def owner_body(path):
    for b in BODIES:
        if f"/{b}/" in str(path) + "/":
            return b
    return None


out = {}
summary = []
for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
    if not prim.IsA(UsdGeom.Gprim):
        continue
    path = str(prim.GetPath())
    kind = "collision" if "/collisions/" in path else "visual" if "/visuals/" in path else None
    body = owner_body(path)
    if kind is None or body is None:
        continue
    mesh = prim_mesh(prim)
    if mesh is None or len(mesh.faces) == 0:
        continue
    T = body_inv[body] @ to_np(xc.GetLocalToWorldTransform(prim))
    mesh.apply_transform(T)
    approx = "-"
    if kind == "collision" and prim.IsA(UsdGeom.Mesh):
        approx = prim.GetAttribute("physics:approximation").Get() if prim.HasAPI(UsdPhysics.MeshCollisionAPI) else "none"
        if approx in ("convexHull", None, "none"):
            # triangle meshes are not allowed on dynamic bodies: PhysX collides with their convex hull
            mesh = mesh.convex_hull
    key = f"{body}|{kind}"
    out.setdefault(key, []).append(mesh)
    summary.append((body, kind, path.split(str(root.GetPath()))[1], prim.GetTypeName(), approx, len(mesh.faces)))

# frames in base_link coordinates
frames = {}
for name in ("base_link", "rplidar_link", "shell_link", "front_caster_link", "wheel_drop_left", "wheel_drop_right",
             "base_footprint"):
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if prim.GetName() == name and prim.IsA(UsdGeom.Xformable):
            frames[name] = body_inv["base_link"] @ to_np(xc.GetLocalToWorldTransform(prim))
            break

save = {}
for key, meshes in out.items():
    m = trimesh.util.concatenate(meshes)
    save[f"{key}|v"] = np.asarray(m.vertices, dtype=np.float32)
    save[f"{key}|f"] = np.asarray(m.faces, dtype=np.int32)
    m.export(os.path.join(args.out, f"tb4_{key.replace('|', '_')}.glb"))
for k, T in frames.items():
    save[f"frame|{k}"] = T
for b in BODIES[1:]:   # wheel body pose in base_link at rest (for the static figure)
    save[f"rest|{b}"] = body_inv["base_link"] @ np.linalg.inv(body_inv[b])
np.savez_compressed(os.path.join(args.out, "tb4_geometry.npz"), **save)
# full robot at rest, visual + collision, for glTF viewers
scene = trimesh.Scene()
for key, meshes in out.items():
    body, kind = key.split("|")
    T = np.eye(4) if body == "base_link" else save[f"rest|{body}"]
    m = trimesh.util.concatenate(meshes).copy()
    m.apply_transform(T)
    if kind == "collision":
        m.visual.face_colors = [255, 80, 80, 110]
    scene.add_geometry(m, node_name=key.replace("|", "_"))
scene.export(os.path.join(args.out, "tb4_full_at_rest.glb"))
for s in summary:
    print("[GEOM]", *s, flush=True)
for k, T in frames.items():
    print("[FRAME]", k, np.round(T[:3, 3], 4).tolist(), flush=True)
print("[DONE]", args.out, flush=True)
os._exit(0)
