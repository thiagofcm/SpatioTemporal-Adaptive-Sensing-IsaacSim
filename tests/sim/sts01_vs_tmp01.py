"""STS01 validation: a synchronized fixed-section evaluation (every section at the same rate f) must reproduce the
TMP01 fixed-rate recording at f episode by episode (no simulator needed; NumPy / csv only):

    python tests/sim/sts01_vs_tmp01.py <sts01 run dir> [<tmp01 run dir>]     (default: the stored TMP01 f Hz run)

Compared:
  episodes.csv   every reference-schema column (policy_type excepted): seed, outcome, collision_type, n_steps,
                 adaptive / nav return, frame count (= FSE), distances, final pose, start, goal
  sensing        accepted / native scans, mean / max observation age per episode (TMP01 metadata.json scan_gate)
  recordings     (when the STS01 run was made with --record) trajectory.csv, steps.csv and obstacles.npz identical;
                 lidar.npz: every TMP01 array identical, and the composite rebuilt from the section arrays
                 (row_section_scan_index) equals the TMP01 observation at every row
Prints a JSON report; exit code 0 = identical.
"""
import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TMP01 = os.path.join(ROOT, "AdaptiveSensing/eval/temporal/TMP01_chase-ob1ob2_fixed-rate-nav05/ref_chase_c10")
SKIP = {"policy_type"}


def main():
    a = os.path.abspath(sys.argv[1])
    cfg = json.load(open(os.path.join(a, "run_config.json")))
    hz = cfg["settings"]["sections"]
    if len(set(hz)) != 1:
        raise SystemExit(f"{a}: sections are not synchronized ({hz})")
    f = hz[0]
    b = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else next(
        os.path.join(TMP01, x) for x in sorted(os.listdir(TMP01)) if x.endswith(f"_fixed{f:g}Hz"))
    ra, rb = list(csv.DictReader(open(os.path.join(a, "episodes.csv")))), list(csv.DictReader(open(os.path.join(b, "episodes.csv"))))
    rep = {"sts01": os.path.relpath(a, ROOT), "tmp01": os.path.relpath(b, ROOT), "hz": f, "episodes": len(ra),
           "episodes_csv_mismatch": {}, "sensing_mismatch": {}, "recordings": None}
    assert len(ra) == len(rb), "different episode counts"
    for col in rb[0]:
        if col in SKIP:
            continue
        bad = [i for i, (x, y) in enumerate(zip(ra, rb)) if x[col] != y[col]]
        if bad:
            rep["episodes_csv_mismatch"][col] = bad[:20]
    sens = {"accepted_scans": "accepted_scans", "native_scans": "native_scans", "mean_obs_age_s": "mean_obs_age_s",
            "max_obs_age_s": "max_obs_age_s"}
    for i, r in enumerate(ra):
        m = json.load(open(os.path.join(b, f"episode_{i:04d}", "metadata.json")))["scan_gate"]
        for ka, kb in sens.items():
            if abs(float(r[ka]) - float(m[kb])) > 1e-9:
                rep["sensing_mismatch"].setdefault(ka, []).append(i)
    if os.path.isdir(os.path.join(a, "episode_0000")):
        bad_files, comp_bad, arr_bad = [], [], []
        for i in range(len(ra)):
            da, db = os.path.join(a, f"episode_{i:04d}"), os.path.join(b, f"episode_{i:04d}")
            for fn in ("trajectory.csv", "steps.csv"):
                if open(os.path.join(da, fn)).read() != open(os.path.join(db, fn)).read():
                    bad_files.append(f"{i}/{fn}")
            oa, ob = np.load(os.path.join(da, "obstacles.npz")), np.load(os.path.join(db, "obstacles.npz"))
            if set(oa.files) != set(ob.files) or any(not np.array_equal(oa[k], ob[k], equal_nan=oa[k].dtype.kind == "f")
                                                     for k in ob.files):
                bad_files.append(f"{i}/obstacles.npz")
            la, lb = np.load(os.path.join(da, "lidar.npz")), np.load(os.path.join(db, "lidar.npz"))
            for k in lb.files:
                x, y = la[k], lb[k]
                if x.shape != y.shape or not np.array_equal(x, y, equal_nan=x.dtype.kind == "f"):
                    arr_bad.append(f"{i}/{k}")
            rs, rssi = la["ray_section"], la["row_section_scan_index"]
            comp = la["ranges"][rssi[:, rs], np.arange(len(rs))]
            ref = lb["ranges"][lb["row_scan_index"]]
            if not np.array_equal(comp, ref):
                comp_bad.append(i)
        rep["recordings"] = {"files_differ": bad_files, "lidar_arrays_differ": arr_bad,
                             "composite_ne_tmp01_observation": comp_bad}
    ok = not rep["episodes_csv_mismatch"] and not rep["sensing_mismatch"] and (
        rep["recordings"] is None or not any(rep["recordings"].values()))
    rep["identical"] = ok
    print(json.dumps(rep, indent=1))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
