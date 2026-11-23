"""Mount study, summary map over the (mount height, pitch) grid (results/sweep_grid_*.csv): floor fraction, escaping
fraction (nominal body attitude), the zero-escape frontiers (nominal / worst observed body pitch), the analytic
zero-escape bound for any pose (arena diagonal), self-occlusion, and the near-field obstacle blind zone.

    python infrastructure/M7_lidar3d/mount_study/scripts/mount_map.py
"""
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RES = Path(__file__).resolve().parents[1] / "results"
rows = [r for f in sorted(RES.glob("sweep_grid_*.csv")) for r in csv.DictReader(open(f))]
F = lambda r, k: float(r[k])  # noqa: E731
zs = sorted({F(r, "z_mount_m") for r in rows}); ps = sorted({F(r, "pitch_deg") for r in rows})
grid = {k: np.full((len(ps), len(zs)), np.nan) for k in ("floor", "esc", "escw", "occ", "blind", "ob")}
for r in rows:
    i, j = ps.index(F(r, "pitch_deg")), zs.index(F(r, "z_mount_m"))
    grid["floor"][i, j] = F(r, "both_floor_pct"); grid["esc"][i, j] = F(r, "both_escape_pct")
    grid["escw"][i, j] = 0.5 * (F(r, "static_escape_pct_worst_body_pitch") + F(r, "dynamic_escape_pct_worst_body_pitch"))
    grid["occ"][i, j] = F(r, "self_occluded_rays"); grid["blind"][i, j] = F(r, "obstacle_blind_below_m")
Z, P = np.array(zs), np.array(ps)
front_n = [P[np.argmax((grid["esc"][:, j] == 0))] if (grid["esc"][:, j] == 0).any() else np.nan for j in range(len(zs))]
front_w = [P[np.argmax((grid["escw"][:, j] == 0) & (grid["esc"][:, j] == 0))] for j in range(len(zs))]
EL = 6.794118                                                     # top / bottom row elevation (cell centres)
zo = Z - 0.00445
bound_n = EL + 0.252 - np.degrees(np.arctan((0.6 - zo) / 8.02))   # analytic: top row below 0.6 m at the arena diagonal
bound_w = EL + 1.31 - np.degrees(np.arctan((0.6 - zo) / 8.02))
fig, axs = plt.subplots(1, 2, figsize=(15, 6.2), sharey=True)
ext = (Z[0] - 0.01, Z[-1] + 0.01, P[0] - 0.125, P[-1] + 0.125)
im0 = axs[0].imshow(grid["floor"], origin="lower", aspect="auto", extent=ext, cmap="Greys", vmin=0, vmax=35)
fig.colorbar(im0, ax=axs[0]).set_label("rays hitting the floor [%] (static + dynamic poses)")
esc = np.where(grid["esc"] > 0, grid["esc"], np.nan)
im1 = axs[1].imshow(esc, origin="lower", aspect="auto", extent=ext, cmap="Reds", vmin=0, vmax=1.0)
fig.colorbar(im1, ax=axs[1]).set_label("rays escaping above walls [%] (nominal attitude; blank = 0)")
for a in axs:
    a.plot(Z, front_n, "o-", color="tab:blue", ms=3, label="zero escape, nominal body pitch (-0.25 deg)")
    a.plot(Z, front_w, "s-", color="tab:green", ms=3, label="zero escape, also at worst body pitch (-1.31 deg)")
    a.plot(Z, bound_n, "--", color="tab:blue", lw=0.8, label="analytic bound, any pose (8.02 m diagonal), nominal")
    a.plot(Z, bound_w, "--", color="tab:green", lw=0.8, label="analytic bound, worst body pitch")
    occ = np.ma.masked_where(grid["occ"] == 0, np.ones_like(grid["occ"]))
    a.contourf(Z, P, occ, levels=[0.5, 1.5], hatches=["xxx"], colors="none")
    a.contour(Z, P, (grid["occ"] > 0).astype(float), levels=[0.5], colors="k", linewidths=0.8)
    a.axvspan(0.5045, Z[-1] + 0.01, color="tab:purple", alpha=0.08)
    a.plot([0.3998], [5.0], "*", color="gold", mec="k", ms=14, label="current mount (0.40 m, 5 deg)")
    a.set_xlabel("mount height z in base_link [m] (top platform at 0.350 m)")
    a.text(0.552, 0.4, "sensor above the 0.5 m\nobstacles: near-field\nblind zone", fontsize=7, color="tab:purple")
    a.text(0.362, 15.0, "self-\nocclusion", fontsize=7)
axs[0].set_ylabel("mount pitch [deg] (down = +)")
axs[0].set_title("floor returns")
axs[1].set_title("escaping rays")
axs[0].legend(fontsize=7, loc="upper right")
fig.suptitle("3D LiDAR mount study: x = y = 0, roll = yaw = 0; full FoV 28 x 14 deg, 4,726 rays; NAV01 / NAV03 poses + random poses")
fig.tight_layout()
fig.savefig(RES / "mount_map.png", dpi=110)
print(RES / "mount_map.png")
