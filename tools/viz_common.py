"""Drawing helpers shared by make_video.py, plot_episode.py and plot_eval.py (numpy + matplotlib only; the scene is
reconstructed from the recorded episode files, see episode_data.py)."""
from __future__ import annotations

import math

import numpy as np
from matplotlib.patches import Circle, Polygon

import episode_data as ED

OUTCOME_COLORS = {"goal": "tab:green", "safety_violation": "tab:red", "physical_collision": "darkred", "timeout": "tab:orange"}


def draw_walls(ax, geometry: dict, alpha: float = 1.0, baffle_color="tab:orange"):
    for w in geometry["walls"]:
        z = w.get("name", "").startswith("Z")
        ax.add_patch(Polygon(ED.wall_polygon(w), closed=True, fc=baffle_color if z else "0.45", ec="k", lw=0.3,
                             alpha=alpha * (0.85 if z else 1.0)))


def draw_obstacle_paths(ax, geometry: dict, alpha=0.25):
    for o in geometry.get("moving_obstacles", []):
        if o["active"]:
            (x0, y0), (x1, y1) = o["path"]
            ax.plot([x0, x1], [y0, y1], color="tab:purple", lw=6, alpha=alpha, solid_capstyle="round")


def setup_map(ax, geometry: dict, title=None, fontsize=9):
    h = geometry.get("arena_interior_half_m", 2.835) + 0.17
    ax.set_xlim(-h, h); ax.set_ylim(-h, h); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=fontsize)


def draw_goal(ax, meta: dict):
    g, tol = meta["goal"], meta["terminations"]["goal_tolerance_m"]
    ax.add_patch(Circle((g["x"], g["y"]), tol, fc="tab:green", alpha=0.2, ec="darkgreen"))
    ax.plot(g["x"], g["y"], "*", color="darkgreen", ms=10)


def episode_title(meta: dict) -> str:
    return (f"{ED.episode_dir_name(meta['episode_index'])}: {meta['outcome']}, {meta['duration_s']:.1f} s, "
            f"min clr {meta['min_clearance_m']:.3f} m")


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def robot_artists(ax, meta: dict, color="tab:blue"):
    body = Circle((0, 0), meta["robot"]["footprint_radius_m"], fc=color, alpha=0.35, ec="navy")
    ax.add_patch(body)
    head, = ax.plot([], [], color="navy", lw=2)
    return body, head


def set_robot(body, head, x, y, yaw, r=0.25):
    body.center = (x, y)
    head.set_data([x, x + r * math.cos(yaw)], [y, y + r * math.sin(yaw)])


def obstacle_artists(ax, ep: dict):
    O = ep["obstacles"]
    return [ax.add_patch(Circle((0, 0), float(O["radius"][j]), fc="tab:purple", alpha=0.6, ec="k", visible=False))
            for j in range(len(O["ids"]))]


def set_obstacles(arts, ep: dict, row: int):
    O = ep["obstacles"]
    for j, c in enumerate(arts):
        r = row
        while r > 0 and not np.isfinite(O["x"][r, j]):        # unknown row (converted legacy terminal row): last known
            r -= 1
        on = bool(O["active"][j]) and np.isfinite(O["x"][r, j])
        c.set_visible(on)
        if on:
            c.center = (float(O["x"][r, j]), float(O["y"][r, j]))


def last_obstacle_xy(ep: dict, j: int):
    """Last recorded position of obstacle j (converted legacy episodes have no terminal-row obstacle positions)."""
    O = ep["obstacles"]
    ok = np.isfinite(O["x"][:, j])
    i = int(np.where(ok)[0][-1]) if ok.any() else 0
    return float(O["x"][i, j]), float(O["y"][i, j])


# --------------------------------------------------------------------------------------------- 3D LiDAR
FLOOR_Z_TOL = 0.02          # m: 3D end points below this height are drawn as floor returns


def scan3d_topdown(ep: dict, row: int, clean: bool = False):
    """Per-scan (3D) episodes: top-down view of the scan in the observation at control step `row`.
    Returns (origin (2,), xy (M, 2), z (M,), floor (M,) bool) of the rays with a return."""
    sc = ED.lidar_scan(ep, max(ED.scan_index(ep, row), 0), clean)
    p = sc["points"][sc["hit"]]
    return sc["origin"][:2], p[:, :2], p[:, 2], p[:, 2] < FLOOR_Z_TOL


def scan3d_colors(z, floor, zmax=0.6):
    """RGBA per point: floor returns light grey, others coloured by height (0..zmax, plasma)."""
    import matplotlib.cm as cm
    c = cm.plasma(np.clip(z / zmax, 0, 1) * 0.9)
    c[floor] = (0.6, 0.6, 0.6, 0.35)
    return c


def range_image_axes(ax, ep: dict, title="range image (row 0 = top, col 0 = left)"):
    """Image artist for ED.range_image (call .set_data each frame)."""
    L = ep["lidar"]
    h, w = (int(v) for v in L["grid_shape"])
    az, el = L["azimuth_deg"], L["elevation_deg"]
    im = ax.imshow(np.full((h, w), np.nan), cmap="viridis_r", vmin=0, vmax=float(L["range_max"]), aspect="auto",
                   interpolation="nearest", extent=(float(az.max()), float(az.min()), float(el.min()), float(el.max())))
    ax.set_xlabel("azimuth [deg] (sensor frame, left = +)", fontsize=7); ax.set_ylabel("elev. [deg]", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_title(title, fontsize=8)
    return im
