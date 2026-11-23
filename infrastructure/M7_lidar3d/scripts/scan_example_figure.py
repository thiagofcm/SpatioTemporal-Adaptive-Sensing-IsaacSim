"""M7 report figure: one recorded 3D LiDAR scan (the scan with the most floor returns in an evaluation folder) as a
range image, a top-down view and a side view, reconstructed from the episode files only (tools/episode_data.py).

    python infrastructure/M7_lidar3d/scripts/scan_example_figure.py <eval_dir> <out.png>
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "tools"))
import episode_data as ED  # noqa: E402
import viz_common as V  # noqa: E402

ev, out = sys.argv[1], sys.argv[2]
best = (-1, None, None)
for d in ED.episode_dirs(ev):
    ep = ED.load_episode(d)
    for i in range(ep["lidar"]["ranges"].shape[0]):
        s = ED.lidar_scan(ep, i)
        n = int((np.abs(s["points"][s["hit"], 2]) < 0.03).sum())
        if n > best[0]:
            best = (n, ep, i)
n, ep, i = best
s = ED.lidar_scan(ep, i)
p = s["points"][s["hit"]]
fig = plt.figure(figsize=(15, 8))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.4])
ax = fig.add_subplot(gs[0, :])
im = V.range_image_axes(ax, ep, f"{ep['dir'].name}, scan {i} (t = {s['t_sim']:.2f} s): observed range image "
                                f"(34 x 139, row 0 = top, col 0 = left)")
im.set_data(ED.range_image(ep, i))
fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01).set_label("range [m]")
a2 = fig.add_subplot(gs[1, 0])
V.draw_walls(a2, ep["geometry"], alpha=0.5)
row = int(ep["lidar"]["scan_step"][i])
V.set_obstacles(V.obstacle_artists(a2, ep), ep, row)
floor = p[:, 2] < V.FLOOR_Z_TOL
a2.scatter(p[:, 0], p[:, 1], c=V.scan3d_colors(p[:, 2], floor, 0.6), s=1.5, linewidths=0)
a2.plot(*s["origin"][:2], "k^", ms=6)
V.setup_map(a2, ep["geometry"], "top-down: returns coloured by height, floor grey")
a3 = fig.add_subplot(gs[1, 1])
q = ep["lidar"]["base_quat_wxyz"][i]
yaw = np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
fwd = (p[:, 0] - s["origin"][0]) * np.cos(yaw) + (p[:, 1] - s["origin"][1]) * np.sin(yaw)
a3.scatter(fwd, p[:, 2], s=1, c=V.scan3d_colors(p[:, 2], floor, 0.6), linewidths=0)
a3.plot([0], [s["origin"][2]], "k^", ms=6)
a3.axhline(0, color="0.3", lw=0.8); a3.axhline(0.6, color="0.6", lw=0.6, ls="--")
a3.set_xlabel("distance along robot heading [m]"); a3.set_ylabel("z [m]"); a3.set_title("side view (wall height 0.6 m dashed)")
a3.grid(lw=0.3)
fig.tight_layout()
fig.savefig(out, dpi=110)
print(out, "episode", ep["dir"].name, "scan", i, "floor returns", n)
