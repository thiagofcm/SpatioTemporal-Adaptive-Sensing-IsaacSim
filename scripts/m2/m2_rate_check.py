"""Per-transition timing (delay / rise) of a calibrated-replay npz vs the real recordings (same metric as M1)."""
import json, os, sys
import numpy as np
from st_nav.calibration.real_data import load_all, step_metrics, transitions
RES = os.path.join(os.environ.get("ST_ROOT", "."), "results", "m1")
runs = {r.name: r for r in load_all() if r.duration > 5.0}
out = {}
for tag in sys.argv[1:]:
    d = np.load(os.path.join(RES, f"m1_calibration_final{tag}.npz"))
    t = d["t_rec"]; names = [str(n) for n in d["run_names"]]
    rows = []
    for i, n in enumerate(names):
        r = runs[n]; trs = transitions(r)
        for j, tr in enumerate(trs):
            t_next = trs[j + 1]["t"] if j + 1 < len(trs) else r.duration
            for comp, y0, y1 in (("v", tr["v0"], tr["v1"]), ("w", tr["w0"], tr["w1"])):
                if abs(y1 - y0) < 1e-6 or tr["from_rest"]:
                    continue
                real = r.odo_v if comp == "v" else r.odo_w
                rd, rr = step_metrics(r.odo_t, real, tr["t"], y0, y1, t_next)
                sd, sr = step_metrics(t, d[f"sim_{comp}_best"][i], tr["t"], y0, y1, t_next)
                rows.append((comp, rd, rr, sd, sr))
    res = {}
    for comp in ("v", "w"):
        a = np.array([x[1:] for x in rows if x[0] == comp], dtype=float)
        m = np.nanmean(a, 0)
        res[comp] = {"real_delay": m[0], "sim_delay": m[2], "delay_diff": m[2] - m[0], "real_rise": m[1], "sim_rise": m[3],
                     "rise_diff": m[3] - m[1]}
    out[tag or "200hz"] = res
    print(tag or "200hz", json.dumps({c: {k: round(v, 3) for k, v in x.items()} for c, x in res.items()}))
json.dump(out, open(os.path.join(os.environ.get("ST_ROOT", "."), "results", "m2", "rate_check_timing.json"), "w"), indent=1)
