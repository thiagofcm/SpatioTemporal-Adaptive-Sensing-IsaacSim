"""Compare two golden snapshots (capture_golden.py). python tests/golden/compare_golden.py pre post"""
import json, os, sys
import numpy as np
H = os.path.dirname(os.path.abspath(__file__)); a, b = sys.argv[1], sys.argv[2]
A, B = np.load(f"{H}/{a}.npz"), np.load(f"{H}/{b}.npz")
ca, cb = json.load(open(f"{H}/{a}_cfg.json")), json.load(open(f"{H}/{b}_cfg.json"))
ok = True
def chk(name, cond, detail=""):
    global ok; ok &= bool(cond); print(f"{'PASS' if cond else 'FAIL'}  {name} {detail}")
def diff(x, y, path=""):
    if isinstance(x, dict) and isinstance(y, dict):
        return [d for k in sorted(set(x) | set(y)) for d in diff(x.get(k, "<missing>"), y.get(k, "<missing>"), f"{path}.{k}")]
    return [] if x == y else [f"{path}: {x!r} -> {y!r}"]
d = [x for x in diff(ca, cb) if "log_dir" not in x]
chk("resolved default env/agent configs identical", not d, "\n   " + "\n   ".join(d[:20]) if d else "")
for k in ("start", "goal", "walls", "z_len", "z_thick", "z_off"):
    chk(f"scene generation: {k} identical", np.array_equal(A[k], B[k]))
chk("robot pose at reset", np.abs(A["robot_pose"] - B["robot_pose"]).max() <= 1e-6, f"max {np.abs(A['robot_pose'] - B['robot_pose']).max():.2e}")
chk("noise-free LiDAR scan at reset", np.abs(A["scan_clean"] - B["scan_clean"]).max() <= 1e-6, f"max {np.abs(A['scan_clean'] - B['scan_clean']).max():.2e}")
chk("non-LiDAR observations [40:47] at reset", np.abs(A["obs_reset"][:, 40:] - B["obs_reset"][:, 40:]).max() <= 1e-6,
    f"max {np.abs(A['obs_reset'][:, 40:] - B['obs_reset'][:, 40:]).max():.2e}")
nd = np.abs(B["obs_reset"][:, :40] - B["scan_clean"]).max()
chk("noisy LiDAR within noise model of the clean scan (post)", nd <= 0.05, f"max |noisy - clean| {nd:.4f} (normalized)")
chk("policy input batch identical", np.array_equal(A["policy_batch"], B["policy_batch"]))
chk("policy actions bit-identical", np.array_equal(A["actions"], B["actions"]), f"max diff {np.abs(A['actions'] - B['actions']).max():.2e}")
print("ALL PASS" if ok else "SOME CHECKS FAILED"); sys.exit(0 if ok else 1)
