"""Horizontal sections of the 3D-LiDAR range image (spatiotemporal sensing, STS).

The range image (1, rows, cols) is split into contiguous, non-overlapping column ranges, left to right (column 0 =
leftmost = largest azimuth, st_nav.sensors.ray_tables.grid_pattern). The layout is derived from the env's ACTUAL ray
table (`grid_col` of every ray), so ray counts stay correct for a measured Mirrorcle table as well as for the uniform
grid. Default: 46 / 47 / 46 columns of the 139-column grid = 1,564 / 1,598 / 1,564 rays (34 per column), symmetric about
the forward axis (docs/spatiotemporal_sensing_plan.md, section 2.1).

Consumption is counted in rays: full-scan equivalents FSE = (rays acquired, reset scan included) / total rays, i.e.
sum_s w_s n_s with w_s = rays_s / total. Integer ray counts keep it exact: when every section acquires together, FSE
equals the number of acquisitions exactly (no float rounding of the weights). Torch / Python only (no Isaac Sim).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

DEFAULT_SECTION_COLUMNS = (46, 47, 46)
DEFAULT_SECTION_NAMES = ("left", "center", "right")


@dataclass(frozen=True)
class SectionLayout:
    names: tuple            # section names, left to right
    col_bounds: tuple       # ((c0, c1), ...) half-open column ranges, left to right
    n_cols: int             # image width
    rays: tuple             # rays per section (from the ray table)

    @property
    def n_sections(self) -> int:
        return len(self.names)

    @property
    def total_rays(self) -> int:
        return int(sum(self.rays))

    @property
    def weights(self) -> tuple:
        """w_s = rays_s / total (float64; for reporting -- consumption itself is counted in integer rays)."""
        return tuple(r / self.total_rays for r in self.rays)

    def column_section(self, device="cpu") -> torch.Tensor:
        """(n_cols,) section index of every image column."""
        out = torch.empty(self.n_cols, dtype=torch.long, device=device)
        for s, (c0, c1) in enumerate(self.col_bounds):
            out[c0:c1] = s
        return out

    def rays_tensor(self, device="cpu") -> torch.Tensor:
        return torch.tensor(self.rays, dtype=torch.int64, device=device)

    def to_dict(self) -> dict:
        return {"names": list(self.names), "columns": [[c0, c1] for c0, c1 in self.col_bounds],
                "n_columns": [c1 - c0 for c0, c1 in self.col_bounds], "image_columns": self.n_cols,
                "rays": list(self.rays), "total_rays": self.total_rays, "weights": list(self.weights),
                "column_order": "column 0 = leftmost (largest azimuth); sections left to right",
                "fse": "full-scan equivalents = rays acquired (reset scan included) / total_rays = sum_s w_s n_s"}

    @classmethod
    def from_grid(cls, grid_col, n_cols: int, columns=DEFAULT_SECTION_COLUMNS, names=None) -> "SectionLayout":
        """Layout of `columns` (column counts, left to right) on an image of `n_cols` columns whose rays sit in the
        columns `grid_col` (one entry per ray)."""
        columns = tuple(int(c) for c in columns)
        if any(c < 1 for c in columns) or sum(columns) != n_cols:
            raise ValueError(f"section column counts {columns} must be >= 1 and sum to the image width {n_cols}")
        if names is None:
            names = DEFAULT_SECTION_NAMES if len(columns) == 3 else tuple(f"s{i}" for i in range(len(columns)))
        if len(names) != len(columns):
            raise ValueError(f"{len(names)} section names for {len(columns)} sections")
        gc = [int(c) for c in grid_col]
        if not gc or min(gc) < 0 or max(gc) >= n_cols:
            raise ValueError(f"ray table columns must lie in [0, {n_cols})")
        bounds, c0 = [], 0
        for c in columns:
            bounds.append((c0, c0 + c))
            c0 += c
        rays = tuple(sum(b0 <= c < b1 for c in gc) for b0, b1 in bounds)
        if any(r == 0 for r in rays):
            raise ValueError(f"a section has no ray: rays per section {rays}")
        return cls(tuple(names), tuple(bounds), int(n_cols), rays)
