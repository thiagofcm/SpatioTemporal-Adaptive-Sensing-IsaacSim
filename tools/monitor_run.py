"""Summarize an M5 run from its tensorboard events (no Isaac Sim needed).
    python tools/monitor_run.py Navigation/runs/<ID>/<run> [--every 10]"""
import argparse, glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
p = argparse.ArgumentParser(); p.add_argument("run"); p.add_argument("--every", type=int, default=10); a = p.parse_args()
ev = EventAccumulator(glob.glob(f"{a.run}/events*")[0], size_guidance={"scalars": 0}); ev.Reload()
cols = [("Train/mean_reward", "reward"), ("Train/mean_episode_length", "len"), ("Episode_Termination/goal", "goal"),
        ("Episode_Termination/safety_violation", "safety"), ("Episode_Termination/physical_collision", "phys"),
        ("Episode_Termination/timeout", "tout"), ("Episode_Reward/progress", "r_prog"), ("Episode_Reward/proximity", "r_prox"),
        ("Episode/min_clearance_m", "minclr"), ("Policy/mean_std", "std"), ("Loss/value", "vloss"), ("Loss/learning_rate", "lr"),
        ("Perf/total_fps", "fps"), ("Perf/collection_time", "t_col"), ("Perf/learning_time", "t_lrn")]
tags = ev.Tags()["scalars"]
data = {t: {e.step: e.value for e in ev.Scalars(t)} for t, _ in cols if t in tags}
steps = sorted(data.get("Perf/total_fps", {}))
print("iter " + " ".join(f"{n:>8}" for _, n in cols))
for s in steps:
    if s % a.every == 0 or s == steps[-1]:
        print(f"{s:4d} " + " ".join((f"{data[t].get(s, float('nan')):8.1e}" if t == "Loss/learning_rate" else f"{data[t].get(s, float('nan')):8.3f}") if t in data else f"{'-':>8}" for t, _ in cols))
