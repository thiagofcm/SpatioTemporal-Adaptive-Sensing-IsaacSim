"""M3 figures (CPU): randomized environments with robot, start, goal and LiDAR; randomization distributions vs the
reference sampler; collision-detection summary.

    python infrastructure/M3_corridor_scene/scripts/m3_plots.py
"""
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon

ROOT = os.environ.get("ST_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
RES = os.path.join(ROOT, "infrastructure", "M3_corridor_scene", "results")
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)
D = dict(np.load(os.path.join(RES, "m3_validation_data.npz")))
V = json.load(open(os.path.join(RES, "m3_validation.json")))
AZ = np.deg2rad(D["azimuths_deg"])
GOALS = [(-1.80, -2.50), (-1.50, -2.50), (-2.10, -2.50), (-1.80, -2.20), (-1.60, -2.30), (-2.00, -2.30),
         (-1.50, -2.20), (-2.10, -2.20), (-1.80, -2.00)]


def box(ax, w, **kw):
    cx, cy, hx, hy, yaw = w
    c, s = math.cos(yaw), math.sin(yaw)
    pts = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
    ax.add_patch(Polygon([(cx + c * x - s * y, cy + s * x + c * y) for x, y in pts], **kw))


def robot(ax, xy, yaw, col, alpha=0.35, r=0.177):
    ax.add_patch(Circle(xy, r, fc=col, alpha=alpha, ec=col, lw=0.8))
    ax.arrow(xy[0], xy[1], 0.22 * math.cos(yaw), 0.22 * math.sin(yaw), width=0.012, color=col, length_includes_head=True)


fig, axes = plt.subplots(2, 4, figsize=(20, 10.6))
for i, ax in enumerate(axes.ravel()):
    W = D["snap_walls"][i]
    for j, w in enumerate(W):
        box(ax, w, fc="0.45" if j < 10 else "tab:orange", ec="k", lw=0.4, alpha=0.95)
        if j >= 10:
            ax.text(w[0], w[1], f"Z{j - 9}", fontsize=7, ha="center", va="center", color="k")
    for g in GOALS:
        ax.plot(*g, ".", color="0.6", ms=3)
    ax.add_patch(Circle(D["snap_goal"][i], 0.20, fc="tab:green", alpha=0.35, ec="tab:green"))
    ax.plot(*D["snap_goal"][i], "*", color="darkgreen", ms=10)
    st = D["snap_start"][i]
    robot(ax, st[:2], st[2], "tab:purple", alpha=0.15)
    robot(ax, D["snap_base_xy"][i], D["snap_base_yaw"][i], "tab:blue")
    o, y = D["snap_lidar_xy"][i], D["snap_lidar_yaw"][i]
    r, a = D["snap_ranges"][i], np.minimum(D["snap_oracle"][i], 3.5)
    for k in range(len(AZ)):
        ang = y + AZ[k]
        e = o + r[k] * np.array([math.cos(ang), math.sin(ang)])
        ax.plot([o[0], e[0]], [o[1], e[1]], color="tab:red" if r[k] < 3.5 - 1e-6 else "0.8", lw=0.5)
        ax.plot(*(o + a[k] * np.array([math.cos(ang), math.sin(ang)])), "k+", ms=3.5)
    err = np.abs(r - a).max()
    ax.set_title(f"env {i}: max |GPU - oracle| {err * 1e3:.2f} mm", fontsize=9)
    ax.set_xlim(-3.05, 3.05)
    ax.set_ylim(-3.05, 3.05)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)
fig.suptitle("M3: corridor_static_object_random in Isaac Lab (articulated TB4, 100 Hz). Grey walls 0.6 m, orange Z1-Z4 "
             "(per-env sizes, per-reset offsets).\nBlue: robot (footprint r = 0.177 m) at a random validation pose; purple: "
             "reference start pose; green: goal (r = 0.20 m), grey dots: 9-goal list; red: GPU LiDAR rays; black +: oracle",
             fontsize=10.5)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "m3_randomized_envs_topview.png"), dpi=110)
plt.close(fig)

# ---------------- distributions ----------------
fig, axes = plt.subplots(2, 5, figsize=(22, 8))
names = ("Z1 x offset", "Z2 x offset", "Z3 y offset", "Z4 y offset")
for j in range(4):
    ax = axes[0, j]
    ax.hist(D["ref_offsets"][:, j], bins=40, density=True, histtype="step", lw=1.5, color="k", label="reference sampler")
    ax.hist(D["offsets"][:, j], bins=40, density=True, histtype="step", lw=1.5, color="tab:orange", label="Isaac Lab resets")
    ax.set_title(f"{names[j]} (KS p = {V['V1_layouts']['ks_pvalues_vs_reference_sampler'][f'offset_{names[j][:2]}_{names[j][3]}']:.2f})",
                 fontsize=9)
    ax.legend(fontsize=7)
ax = axes[0, 4]
ax.hist(D["ref_dz13"], bins=40, density=True, histtype="step", lw=1.5, color="k", label="reference")
ax.hist(D["dz13"], bins=40, density=True, histtype="step", lw=1.5, color="tab:orange", label="Isaac Lab")
ax.axvline(0.70, color="r", ls="--", lw=1)
ax.set_title(f"d(Z1, Z3) [m] (constraint >= 0.70; min {V['V1_layouts']['min_d_z1_z3']:.3f})", fontsize=9)
ax.legend(fontsize=7)
for j in range(4):
    ax = axes[1, j]
    ax.hist(D["ref_lengths"][:, j], bins=30, density=True, histtype="step", lw=1.5, color="k", label="ref length")
    ax.hist(D["lengths"][:, j], bins=30, density=True, histtype="step", lw=1.5, color="tab:blue", label="per-env length")
    ax.hist(D["ref_thick"][:, j], bins=30, density=True, histtype="step", lw=1.5, color="0.5", ls="--", label="ref thickness")
    ax.hist(D["thick"][:, j], bins=30, density=True, histtype="step", lw=1.5, color="tab:red", ls="--", label="per-env thickness")
    ax.set_title(f"Z{j + 1} length / thickness [m]", fontsize=9)
    ax.legend(fontsize=6)
ax = axes[1, 4]
ax.axis("off")
v3 = V["V3_collisions"]
lines = ["Collision detection (V3):",
         f"  forward : contact in {v3['forward']['envs_with_contact']} envs; footprint clearance",
         f"            at first contact: median {v3['forward']['clearance_at_first_contact_m']['median'] * 1e3:+.2f} mm",
         f"            (p01 {v3['forward']['clearance_at_first_contact_m']['p01'] * 1e3:+.2f}, p99 "
         f"{v3['forward']['clearance_at_first_contact_m']['p99'] * 1e3:+.2f} mm)",
         f"  backward: median {v3['backward']['clearance_at_first_contact_m']['median'] * 1e3:+.2f} mm",
         "            (rear base r = 0.164 < 0.177: GT is 13 mm conservative)",
         f"  contact while GT clearance > 1 cm: {v3['forward']['contact_without_gt_collision_envs(c>0.01)'] + v3['backward']['contact_without_gt_collision_envs(c>0.01)']} envs",
         f"  GT clearance < -3 cm without contact: {v3['forward']['gt_collision_without_contact_envs(c<-0.03 and no contact)'] + v3['backward']['gt_collision_without_contact_envs(c<-0.03 and no contact)']} envs",
         "", "Layouts (V1):",
         f"  {V['V1_layouts']['layouts_checked']} randomized layouts, 0 constraint violations",
         f"  min C12 {V['V1_layouts']['min_C12']:.3f}, min C34 {V['V1_layouts']['min_C34']:.3f} m",
         f"  PhysX baffle pose vs table: {V['V1_layouts']['physx_baffle_pose_vs_table_max_pos_err_m'] * 1e6:.0f} um"]
ax.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=8.5)
fig.suptitle("M3: randomization vs the reference sampler (offsets per reset; sizes per env from a bank of reference "
             "samples)", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "m3_randomization_distributions.png"), dpi=110)
plt.close(fig)
print("figures written to", FIG)
