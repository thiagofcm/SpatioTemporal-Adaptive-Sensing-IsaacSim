"""One-time: Z1-Z4 configurations from the REFERENCE sampler (read-only import of ~/adaptive_sensor_policy), stored as
the golden values that st_nav's own sampler must reproduce bit-for-bit (tests/unit/test_static_obstacle_sampler.py).
    python -B tests/golden/make_reference_bank.py"""
import json, os, subprocess, sys
import numpy as np
REF = os.path.expanduser("~/adaptive_sensor_policy")
sys.dont_write_bytecode = True
sys.path.insert(0, REF)
from envs.scenes import get_scene  # noqa: E402
ref = get_scene("corridor_static_object_random")
seeds = np.concatenate([np.arange(0, 4096), np.arange(3000, 3100)])
rows = [ref.sample_static_obstacles_params(int(s)) for s in seeds]
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference_bank.npz")
np.savez_compressed(out, seeds=seeds, lengths=np.array([r["lengths"] for r in rows]),
                    thicknesses=np.array([r["thicknesses"] for r in rows]), offsets=np.array([r["offsets"] for r in rows]),
                    hard=np.array([[m == "hard" for m in r["pair_modes"]] for r in rows]),
                    c12=np.array([r["c12"] for r in rows]), c34=np.array([r["c34"] for r in rows]),
                    d_z1_z3=np.array([r["d_z1_z3"] for r in rows]),
                    source=json.dumps({"repo": REF, "commit": subprocess.run(["git", "-C", REF, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
                                       "file": "envs/scenes/corridor_static_object_random.py:sample_static_obstacles_params"}))
print(out, len(seeds), "configurations")
