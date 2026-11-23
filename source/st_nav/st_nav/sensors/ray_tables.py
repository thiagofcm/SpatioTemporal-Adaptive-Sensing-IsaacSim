"""Ray tables of 3D LiDAR sensors and the mount rotation convention: pure Python (no Isaac Lab / Kit), so they can be
used and tested without the simulator. Re-exported by st_nav.sensors.patterns."""
from __future__ import annotations

import math


RAY_ORDERS = ("row_major_top_left", "row_major_bottom_left", "serpentine_top_left", "column_major_top_left")


def grid_pattern(h_fov_deg=(-14.0, 14.0), v_fov_deg=(-7.0, 7.0), cols: int = 139, rows: int = 34,
                 placement: str = "cell_center", order: str = "row_major_top_left"):
    """Uniform rectangular (azimuth x elevation) grid in the SENSOR frame (x forward, y left, z up; azimuth CCW from +x,
    elevation up from the x-y plane).

    placement: "cell_center" (FoV split into rows x cols equal cells, one ray per cell centre) or "endpoints"
    (linspace including both FoV edges).
    order (ray index = position in the returned lists; fixes observation / recording / noise-key indices):
      row_major_top_left     row 0 = highest elevation, column 0 = leftmost azimuth (largest), index = row*cols + col
      row_major_bottom_left  row 0 = lowest elevation, column 0 = leftmost azimuth
      serpentine_top_left    as row_major_top_left, odd rows run right -> left
      column_major_top_left  column 0 = leftmost, each column top -> bottom
    Returns (azimuths_deg, elevations_deg, grid_row, grid_col): lists of length rows*cols; grid_row / grid_col give each
    ray's cell (row 0 = top, column 0 = left), independent of the order, to rebuild the range image.
    """
    if order not in RAY_ORDERS:
        raise ValueError(f"unknown ray order {order!r}; valid: {RAY_ORDERS}")

    def axis(lo, hi, n):
        if placement == "cell_center":
            step = (hi - lo) / n
            return [lo + (i + 0.5) * step for i in range(n)]
        if placement == "endpoints":
            return [lo + i * (hi - lo) / (n - 1) for i in range(n)] if n > 1 else [(lo + hi) / 2]
        raise ValueError(f"unknown placement {placement!r} (cell_center | endpoints)")

    az_left_to_right = axis(h_fov_deg[0], h_fov_deg[1], cols)[::-1]      # left = positive azimuth
    el_top_to_bottom = axis(v_fov_deg[0], v_fov_deg[1], rows)[::-1]      # top = positive elevation
    cells = []
    if order == "column_major_top_left":
        cells = [(r, c) for c in range(cols) for r in range(rows)]
    else:
        row_seq = range(rows) if order != "row_major_bottom_left" else range(rows - 1, -1, -1)
        for r in row_seq:
            cs = range(cols)
            if order == "serpentine_top_left" and r % 2 == 1:
                cs = range(cols - 1, -1, -1)
            cells += [(r, c) for c in cs]
    return ([az_left_to_right[c] for _, c in cells], [el_top_to_bottom[r] for r, _ in cells],
            [r for r, _ in cells], [c for _, c in cells])


def rpy_to_quat_wxyz(rpy_deg) -> tuple:
    """(roll, pitch, yaw) in degrees -> quaternion (w, x, y, z) of R = Rz(yaw) Ry(pitch) Rx(roll) (fixed axes x, y, z;
    REP-103). Right-handed: a POSITIVE pitch rotates the sensor's +x (forward) axis toward -z, i.e. tilts it DOWN."""
    r, p, y = (math.radians(a) / 2.0 for a in rpy_deg)
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return (cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy)
