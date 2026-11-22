"""Compare two training runs (TensorBoard events): learning curves, convergence, final metrics, throughput.

    python tests/golden/compare_training.py <run_a> <run_b> --labels original reorganized --out tests/golden/training_comparison
"""
import argparse
import glob
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: E402

TAGS = {"return": "Train/mean_reward", "length": "Train/mean_episode_length", "goal": "Episode_Termination/goal",
        "safety": "Episode_Termination/safety_violation", "timeout": "Episode_Termination/timeout",
        "min_clearance": "Episode/min_clearance_m", "std": "Policy/mean_std", "value_loss": "Loss/value",
        "lr": "Loss/learning_rate", "fps": "Perf/total_fps", "t_collect": "Perf/collection_time", "t_learn": "Perf/learning_time"}
p = argparse.ArgumentParser()
p.add_argument("runs", nargs=2)
p.add_argument("--labels", nargs=2, default=["a", "b"])
p.add_argument("--max_it", type=int, default=200)
p.add_argument("--out", default="training_comparison")
a = p.parse_args()


def load(run):
    ev = EventAccumulator(glob.glob(f"{run}/events.out.tfevents.*")[0], size_guidance={"scalars": 0})
    ev.Reload()
    out = {}
    for k, t in TAGS.items():
        d = {e.step: e.value for e in ev.Scalars(t) if e.step <= a.max_it}
        out[k] = np.array([d.get(i, np.nan) for i in range(a.max_it + 1)])
    return out


D = [load(r) for r in a.runs]
res = {"runs": dict(zip(a.labels, a.runs)), "iterations_compared": a.max_it}
for lab, d in zip(a.labels, D):
    def first(x, thr):
        idx = np.where(x >= thr)[0]
        return int(idx[0]) if len(idx) else None
    w = slice(150, a.max_it + 1)                                  # converged window
    res[lab] = {
        "first_iter_goal_ge_0.9": first(d["goal"], 0.9), "first_iter_goal_ge_0.98": first(d["goal"], 0.98),
        "first_iter_return_ge_80": first(d["return"], 80.0),
        "iter150_200_mean": {k: float(np.nanmean(d[k][w])) for k in ("return", "length", "goal", "safety", "timeout", "min_clearance", "std")},
        "final_iter": {k: float(d[k][a.max_it]) for k in ("return", "length", "goal", "safety", "std")},
        "fps_median": float(np.nanmedian(d["fps"][1:])), "fps_p10_p90": [float(np.nanpercentile(d["fps"][1:], q)) for q in (10, 90)],
        "s_per_iter_median": float(np.nanmedian((d["t_collect"] + d["t_learn"])[1:])),
        "collect_s_median": float(np.nanmedian(d["t_collect"][1:])), "learn_s_median": float(np.nanmedian(d["t_learn"][1:])),
        "total_train_time_s": float(np.nansum(d["t_collect"] + d["t_learn"])),
    }
fa, fb = res[a.labels[0]]["fps_median"], res[a.labels[1]]["fps_median"]
res["fps_change_pct"] = 100 * (fb - fa) / fa
res["curve_abs_diff_mean_iter_0_200"] = {k: float(np.nanmean(np.abs(D[0][k] - D[1][k]))) for k in ("return", "goal", "length")}
json.dump(res, open(a.out + ".json", "w"), indent=1)
fig, ax = plt.subplots(2, 3, figsize=(16, 8))
for (k, ttl), axx in zip([("return", "mean episode return"), ("goal", "goal rate"), ("length", "episode length [steps]"),
                          ("safety", "safety-violation rate"), ("std", "policy std"), ("fps", "throughput [env steps/s]")], ax.ravel()):
    for lab, d, c in zip(a.labels, D, ("tab:blue", "tab:orange")):
        axx.plot(d[k], label=lab, color=c, lw=1.2)
    axx.set_title(ttl, fontsize=10); axx.set_xlabel("iteration"); axx.grid(lw=0.3)
ax[0, 0].legend()
fig.suptitle(f"Training comparison: {a.labels[0]} vs {a.labels[1]} (same config, seed 42, 4,096 envs)")
fig.tight_layout(); fig.savefig(a.out + ".png", dpi=110)
print(json.dumps(res, indent=1))
