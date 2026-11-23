"""Mount study part 2: markdown tables from results/sweep_xzfov_dense*.csv.
    python infrastructure/M7_lidar3d/mount_study/scripts/mount_tables_xzfov.py > .../results/tables_xzfov.md"""
import csv
from pathlib import Path

RES = Path(__file__).resolve().parents[1] / "results"
rows = list(csv.DictReader(open(RES / "sweep_xzfov_dense.csv"))) + list(csv.DictReader(open(RES / "sweep_xzfov_dense_wide.csv")))
F = lambda r, k: float(r[k]) if r[k] not in ("", "inf") else float("inf")  # noqa: E731
NAMES = {(0.086, 0.298, 3.25, 8.0): "**B8**", (0.086, 0.298, 4.0, 10.0): "**B10**"}
print("| | x [m] | z [m] | pitch | v. FoV | floor st / dyn | escape nominal / worst tilt | moving obst. | rays on nearest obstacle 0.5-1 / 1-2 / 2-3 / >3 m | "
      "wall height covered at 1 / 2 / 4 m | obstacle height covered at 0.5 / 1 / 2 m | floor from | row spacing | full robot width covered from (ahead of the front) | self-occl. |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
def key(r):
    return (r["label"] == "", F(r, "both_floor_pct"))
for r in sorted(rows, key=key):
    k = (F(r, "x_m"), F(r, "z_m"), F(r, "pitch_deg"), F(r, "vfov_deg"))
    lab = r["label"].split(" (")[0] if r["label"] else NAMES.get(k, "")
    print(f"| {lab} | {k[0]:.3f} | {k[1]:.3f} | {k[2]:g} | {k[3]:g} | {F(r,'static_floor_pct'):.1f} / {F(r,'dynamic_floor_pct'):.1f} % | "
          f"{F(r,'both_escape_pct'):.3f} / {F(r,'both_escape_pct_worst'):.3f} % | {F(r,'dynamic_moving_obstacle_pct'):.2f} % | "
          + " / ".join(f"{F(r, 'dynamic_obst_rays_' + b):.0f}" for b in ("0.5-1m", "1-2m", "2-3m", "3-12m")) + " | "
          + " / ".join(f"{F(r, f'wall_cover_{d}m_pct'):.0f}" for d in ("1", "2", "4")) + " % | "
          + " / ".join(f"{F(r, f'obst_cover_{d}m_pct'):.0f}" for d in ("0.5", "1", "2")) + " % | "
          f"{F(r,'floor_from_m'):.2f} m | {F(r,'row_spacing_deg'):.3f} deg | {F(r,'full_width_ahead_of_front_m'):.2f} m | {int(F(r,'self_occluded_rays'))} |")
