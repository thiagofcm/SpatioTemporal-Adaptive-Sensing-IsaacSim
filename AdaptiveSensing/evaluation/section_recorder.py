"""Episode recording of section-gated 3D-LiDAR evaluations (STS): Navigation/evaluation/recorder.EpisodeRecorder with
section-aware LiDAR data. Same files and format as the temporal fixed-rate evaluation (tools/episode_data.py, per-scan
layout); differences:

lidar.npz
    ranges, ranges_clean, origin_xyz, base_quat_wxyz, scan_step, scan_t_sim   (S, ...) every NATIVE scan accepted by at
                                       least one section, with the sensor pose of THAT acquisition (as the base layout)
    scan_sections (S, n_sections) bool which sections accepted the scan (row 0 = reset scan: all)
    row_section_scan_index (rows, n_sections)   per control-step row k and section s: the scan whose section-s rays
                                       are in the observation after step k
    ray_section (R,)                   section of each ray; section_names, section_columns (n_sections, 2), section_rays
    row_scan_index                     (base layout) latest scan accepted by ANY section: for a composite it does not
                                       describe the whole observation; use row_section_scan_index
    composite observation after step k (ray r):  ranges[row_section_scan_index[k, ray_section[r]], r]
steps.csv
    scan_fresh = at least one section accepted the scan of this step; beams_acquired = rays acquired at this step
    (sum of the accepting sections' rays); min_lidar_m = minimum of the composite observation (terminal row: last held)
All other columns, trajectory.csv and obstacles.npz are written by the base recorder unchanged; with all sections
synchronized every file equals the temporal fixed-rate recording (tests/sim/sts01_validation.py).
"""
from __future__ import annotations

import csv
import os

import numpy as np

from Navigation.evaluation.recorder import EpisodeRecorder
from tools import episode_data as ED

LIDAR_SECTIONS_NOTE = ("section-gated per-scan layout: ranges / scan_step / origin_xyz / ... = native scans accepted by "
                       ">= 1 section (pose of that acquisition); scan_sections (S, n_sections) = sections that accepted "
                       "each scan; row_section_scan_index[k, s] = scan whose section-s rays are in the observation after "
                       "step k; composite ray r after step k = ranges[row_section_scan_index[k, ray_section[r]], r]; "
                       "row_scan_index = latest scan accepted by any section (not the whole composite)")


class SectionEpisodeRecorder(EpisodeRecorder):
    def __init__(self, env_unwrapped, layout, capture_lidar: bool = True):
        """capture_lidar=False: no LiDAR scans are kept (no export; the in-memory trajectory / step records and the
        terminal-state hook are kept, e.g. for episodes.csv)."""
        super().__init__(env_unwrapped)
        self.capture_lidar = bool(capture_lidar)
        if not self.per_scan:
            raise ValueError("SectionEpisodeRecorder needs the per-scan (3D) LiDAR layout")
        lc = self.corr.lidar.cfg
        self.layout = layout
        cs = layout.column_section().numpy()
        self.ray_section = cs[np.asarray(lc.grid_col, dtype=np.int64)]              # (R,)
        self.section_rays = np.asarray(layout.rays, dtype=np.int64)
        if np.bincount(self.ray_section, minlength=layout.n_sections).tolist() != list(layout.rays):
            raise ValueError("section layout does not match the env's ray table")
        self.comp_min, self.row_beams = {}, {}

    def _capture(self, row, mask, obs, st):
        if self.capture_lidar:
            super()._capture(row, mask, obs, st)

    def _tag_new_scans(self, before, sec):
        if not self.capture_lidar:
            return
        for e, nb in enumerate(before):
            if len(self.scans[e]) > nb:
                self.scans[e][-1]["sections"] = sec[e].copy()

    def record_reset(self, obs):
        before = [len(s) for s in self.scans]
        super().record_reset(obs)
        self._tag_new_scans(before, np.ones((len(before), self.layout.n_sections), dtype=bool))

    def record_step_sections(self, actions, rew, dones, ext, native_obs, accepted, t_scan=None):
        """native_obs: the env observation of this step (its native scan); accepted (N, S): sections that accepted it."""
        before = [len(s) for s in self.scans]
        fresh = accepted.any(dim=1)
        super().record_step(actions, rew, dones, ext, native_obs, fresh=fresh, t_scan=t_scan)
        if self.capture_lidar:
            self._tag_new_scans(before, accepted.cpu().numpy())

    def _scan_arrays(self, e: int, n: int, dt: float) -> dict:
        base = super()._scan_arrays(e, n, dt)
        sc = [s for s in self.scans[e] if s["row"] <= n]
        sec = np.stack([s["sections"] for s in sc]).astype(bool)                   # (S_scans, n_sections)
        rows = base["scan_step"]
        idx = np.arange(len(rows))
        rssi = np.zeros((n + 1, self.layout.n_sections), dtype=np.int64)
        for s in range(self.layout.n_sections):
            rs, ix = rows[sec[:, s]], idx[sec[:, s]]
            rssi[:, s] = ix[np.searchsorted(rs, np.arange(n + 1), side="right") - 1]
        comp = base["ranges"][rssi[:, self.ray_section], np.arange(self.NR)]       # (rows, R) composite observation
        self.comp_min[e] = comp.min(1)[1:]
        beams = np.zeros(n + 1, dtype=np.int64)
        beams[rows] = (sec * self.section_rays).sum(1)
        self.row_beams[e] = beams[1:]
        L = self.layout
        return {**base, "scan_sections": sec, "row_section_scan_index": rssi, "ray_section": self.ray_section,
                "section_names": np.array(L.names), "section_columns": np.array(L.col_bounds, dtype=np.int64),
                "section_rays": self.section_rays, "lidar_sections_note": np.array(LIDAR_SECTIONS_NOTE)}

    def export(self, out_dir, n_steps, ep_rows: list, static_meta: dict, episode_meta_fn):
        if not self.capture_lidar:
            raise RuntimeError("export needs capture_lidar=True")
        super().export(out_dir, n_steps, ep_rows, static_meta, episode_meta_fn)
        for e, n in enumerate(n_steps):                 # steps.csv: rays acquired and composite minimum, same formatting
            path = os.path.join(out_dir, ED.episode_dir_name(e), "steps.csv")
            with open(path, newline="") as f:
                rows = list(csv.reader(f))
            hdr = rows[0]
            ib, im = hdr.index("beams_acquired"), hdr.index("min_lidar_m")
            for k, r in enumerate(rows[1:]):
                r[ib] = str(int(self.row_beams[e][k]))
                v = float(self.comp_min[e][k])
                r[im] = repr(round(v, 6)) if np.isfinite(v) else ""
            with open(path, "w", newline="") as f:
                csv.writer(f).writerows(rows)


def composite_from_recording(lidar: dict, row: int) -> np.ndarray:
    """(R,) composite observation after step `row` of a section-gated lidar.npz (ray order)."""
    rssi, rs = lidar["row_section_scan_index"], lidar["ray_section"]
    return lidar["ranges"][rssi[row, rs], np.arange(len(rs))]

