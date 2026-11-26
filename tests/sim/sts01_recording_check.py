"""STS01 validation of a section-gated recording (evaluate_fixed_sections.py --record), offline (NumPy / csv only):

    python tests/sim/sts01_recording_check.py <sts01 run dir>

Per episode, from lidar.npz / steps.csv / episodes.csv:
  * acquisitions per section (scan_sections, reset scan included) = episodes.csv acq_<s>
  * section acquisitions follow the section's constant rate on the lattice t0 + m / f (deadline advanced by whole
    periods, so a late native scan does not shift it): every acquisition is at or after the next deadline after the
    previous acquisition, and within one native interval (<= 0.4 s) of it
  * rays: total rays + sum of steps.csv beams_acquired = FSE x total rays (episodes.csv episode_frame_count)
  * accepted_scans = number of stored scans; beams_acquired > 0 exactly at the stored scans' rows
  * composite (ranges[row_section_scan_index[k, ray_section]]) minimum = steps.csv min_lidar_m, and each section's
    index points to the latest scan that section accepted at or before row k
Prints a JSON report; exit code 0 = no violation.
"""
import csv
import json
import os
import sys

import numpy as np


def main():
    d = os.path.abspath(sys.argv[1])
    cfg = json.load(open(os.path.join(d, "run_config.json")))
    hz = cfg["settings"]["sections"]
    rows = list(csv.DictReader(open(os.path.join(d, "episodes.csv"))))
    names = cfg["sensing"]["sections"]["names"]
    total = cfg["sensing"]["sections"]["total_rays"]
    bad = {k: [] for k in ("acq", "period", "rays", "scans", "beams_rows", "min_lidar", "index")}
    for i, r in enumerate(rows):
        ep = os.path.join(d, f"episode_{i:04d}")
        L = np.load(os.path.join(ep, "lidar.npz"))
        st = list(csv.DictReader(open(os.path.join(ep, "steps.csv"))))
        sec, t = L["scan_sections"], L["scan_t_sim"]
        if [int(sec[:, s].sum()) for s in range(len(names))] != [int(r[f"acq_{nm}"]) for nm in names]:
            bad["acq"].append(i)
        for s, h in enumerate(hz):                     # lattice deadlines t0 + m / h (real-robot gate, no drift)
            ts = np.round(t[sec[:, s]] / 0.01).astype(np.int64)      # integer physics ticks, as the gate
            per = int(round(1.0 / (h * 0.01)))
            for a, b in zip(ts[:-1], ts[1:]):
                nxt = ts[0] + ((a - ts[0]) // per + 1) * per
                if not (nxt <= b < nxt + 40):              # first native scan at / after the deadline (gap <= 0.4 s)
                    bad["period"].append((i, s))
                    break
        beams = np.array([int(x["beams_acquired"]) for x in st])
        fse = float(r["episode_frame_count"])
        if abs((total + beams.sum()) / total - fse) > 1e-6:
            bad["rays"].append(i)
        if len(t) != int(r["accepted_scans"]):
            bad["scans"].append(i)
        if sorted(np.nonzero(beams)[0] + 1) != sorted(L["scan_step"][1:].tolist()):
            bad["beams_rows"].append(i)
        rs, rssi = L["ray_section"], L["row_section_scan_index"]
        comp = L["ranges"][rssi[:, rs], np.arange(len(rs))]
        mins = np.array([float(x["min_lidar_m"]) for x in st])
        if np.abs(comp.min(1)[1:] - mins).max() > 1e-5:
            bad["min_lidar"].append(i)
        rows_k = np.arange(rssi.shape[0])
        for s in range(len(names)):
            rs_s = L["scan_step"][sec[:, s]]
            idx_s = np.nonzero(sec[:, s])[0]
            exp = idx_s[np.searchsorted(rs_s, rows_k, side="right") - 1]
            if not np.array_equal(exp, rssi[:, s]):
                bad["index"].append((i, s))
    rep = {"run": d, "section_hz": hz, "episodes": len(rows), "violations": {k: v[:10] for k, v in bad.items()},
           "pass": not any(bad.values())}
    print(json.dumps(rep, indent=1))
    sys.exit(0 if rep["pass"] else 1)


if __name__ == "__main__":
    main()
