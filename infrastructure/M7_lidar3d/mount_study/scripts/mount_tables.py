"""Mount study, report tables (markdown) from results/sweep_shortlist_dense.csv and results/sweep_grid_*.csv.
    python infrastructure/M7_lidar3d/mount_study/scripts/mount_tables.py > infrastructure/M7_lidar3d/mount_study/results/tables.md"""
import csv
from pathlib import Path

RES = Path(__file__).resolve().parents[1] / "results"
F = lambda r, k: float(r[k]) if r[k] not in ("", "inf") else float("inf")  # noqa: E731
LAB = {(0.3998, 5.0): "current", (0.4, 5.75): "N4", (0.4, 6.75): "R4", (0.46, 6.25): "N3", (0.46, 7.25): "R3",
       (0.5, 6.5): "N1", (0.5, 7.5): "**R1**", (0.5, 8.0): "R1+", (0.56, 8.0): "R2"}
R = list(csv.DictReader(open(RES / "sweep_shortlist_dense.csv")))
for sc in ("static", "dynamic"):
    print(f"\n**{sc.capitalize()} scene** ({'NAV01 ref_static_c10' if sc == 'static' else 'NAV03 ref_chase_c10'} poses)\n")
    print("| | z [m] | pitch [deg] | floor | wall | static obst. | moving obst. | escape | escape at worst tilt | no return | top-face | poses with escape | self-occl. |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in R:
        k = (round(F(r, "z_mount_m"), 4), F(r, "pitch_deg"))
        v = [F(r, f"{sc}_{x}") for x in ("floor_pct", "wall_pct", "static_obstacle_pct", "moving_obstacle_pct", "escape_pct",
                                          "escape_pct_worst_body_pitch", "no_return_pct", "top_face_pct", "poses_with_escape_pct")]
        print(f"| {LAB.get(k, '')} | {k[0]:.3f} | {k[1]:g} | " + " | ".join(f"{x:.2f} %" if i < 4 else f"{x:.3f} %" for i, x in enumerate(v))
              + f" | {int(F(r, 'self_occluded_rays'))} |")
print("\n**Coverage and fan geometry** (dynamic scene for the obstacle columns; flat floor, nominal attitude for the distances)\n")
print("| | z [m] | pitch [deg] | top / bottom row [deg] | floor from [m] | top row above 0.6 m from [m] | obstacle blind zone [m] | structure returns 0-1 / 1-2 / 2-4 / 4-12 m (static) | rays on the nearest moving obstacle at 0.5-1 / 1-2 / 2-3 m |")
print("|---|---|---|---|---|---|---|---|---|")
for r in R:
    k = (round(F(r, "z_mount_m"), 4), F(r, "pitch_deg"))
    tw = F(r, "top_above_wall_from_m")
    print(f"| {LAB.get(k, '')} | {k[0]:.3f} | {k[1]:g} | {F(r,'top_row_elev_deg'):+.2f} / {F(r,'bottom_row_elev_deg'):+.2f} | "
          f"{F(r,'floor_from_m'):.2f} | {'never' if tw == float('inf') else f'{tw:.1f}'} | {F(r,'obstacle_blind_below_m'):.2f} | "
          + " / ".join(f"{F(r, f'static_struct_{b}m_pct'):.1f}" for b in ("0-1", "1-2", "2-4", "4-12"))
          + " % | " + " / ".join(f"{F(r, f'dynamic_obst_rays_{b}m'):.0f}" for b in ("0.5-1", "1-2", "2-3")) + " |")
rows = [r for f in sorted(RES.glob("sweep_grid_*.csv")) for r in csv.DictReader(open(f))]
print("\n**Zero-escape frontier per mount height** (coarse grid, 0.25 deg pitch steps; floor = mean of both scenes)\n")
print("| z [m] | min. pitch, nominal attitude | floor | top-face | min. pitch, also worst tilt | floor | top-face | self-occlusion from pitch |")
print("|---|---|---|---|---|---|---|---|")
for z in sorted({F(r, "z_mount_m") for r in rows}):
    S = sorted([r for r in rows if F(r, "z_mount_m") == z], key=lambda r: F(r, "pitch_deg"))
    n = next(r for r in S if F(r, "static_escape_pct") == 0 and F(r, "dynamic_escape_pct") == 0)
    w = next(r for r in S if all(F(r, k) == 0 for k in ("static_escape_pct", "dynamic_escape_pct",
                                                         "static_escape_pct_worst_body_pitch", "dynamic_escape_pct_worst_body_pitch")))
    occ = [F(r, "pitch_deg") for r in S if F(r, "self_occluded_rays") > 0]
    print(f"| {z:.2f} | {F(n,'pitch_deg'):g} | {F(n,'both_floor_pct'):.1f} % | {F(n,'both_top_face_pct'):.2f} % | {F(w,'pitch_deg'):g} | "
          f"{F(w,'both_floor_pct'):.1f} % | {F(w,'both_top_face_pct'):.2f} % | {min(occ) if occ else 'none'} |")
