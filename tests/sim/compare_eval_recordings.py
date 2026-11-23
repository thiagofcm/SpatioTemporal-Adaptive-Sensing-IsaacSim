"""Compare two recorded evaluations file by file (regression check after code changes):
    python tests/sim/compare_eval_recordings.py <eval_dir_a> <eval_dir_b>
episodes.csv, every episode's trajectory.csv, steps.csv, lidar.npz and obstacles.npz must be identical (NaN == NaN);
metadata.json differences are listed (expected: schema version / documentation fields only)."""
import csv
import json
import sys
from pathlib import Path

import numpy as np

A, B = Path(sys.argv[1]), Path(sys.argv[2])


def same_arr(x, y):
    if x.shape != y.shape or x.dtype.kind != y.dtype.kind:
        return False
    if x.dtype.kind in "fc":
        return bool(np.array_equal(x, y, equal_nan=True))
    return bool(np.array_equal(x, y))


def diff_json(x, y, path=""):
    if isinstance(x, dict) and isinstance(y, dict):
        return [d for k in sorted(set(x) | set(y)) for d in diff_json(x.get(k, "<missing>"), y.get(k, "<missing>"), f"{path}.{k}")]
    return [] if x == y else [path]


ra, rb = list(csv.DictReader(open(A / "episodes.csv"))), list(csv.DictReader(open(B / "episodes.csv")))
skip = {"wall_time_s", "eval_time_s"}
ep_diff = [i for i, (x, y) in enumerate(zip(ra, rb)) if {k: v for k, v in x.items() if k not in skip}
           != {k: v for k, v in y.items() if k not in skip}]
bad, meta_paths = [], set()
for i in range(len(ra)):
    da, db = A / f"episode_{i:04d}", B / f"episode_{i:04d}"
    for f in ("trajectory.csv", "steps.csv"):
        if (da / f).read_text() != (db / f).read_text():
            bad.append(f"{da.name}/{f}")
    for f in ("lidar.npz", "obstacles.npz"):
        x, y = np.load(da / f), np.load(db / f)
        for k in sorted(set(x.files) | set(y.files)):
            if k not in x.files or k not in y.files or not same_arr(x[k], y[k]):
                bad.append(f"{da.name}/{f}:{k}")
    meta_paths |= set(diff_json(json.load(open(da / "metadata.json")), json.load(open(db / "metadata.json"))))
print(f"episodes.csv rows differing (except timing): {len(ep_diff)} of {len(ra)}")
print(f"data files differing: {len(bad)}" + (f" e.g. {bad[:5]}" if bad else ""))
print("metadata.json fields differing:", sorted(meta_paths))
sys.exit(0 if not ep_diff and not bad else 1)
