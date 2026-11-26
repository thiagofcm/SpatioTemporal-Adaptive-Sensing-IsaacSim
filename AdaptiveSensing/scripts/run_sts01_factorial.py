"""STS01 factorial: every constant per-section rate configuration (default 5^3 = 125 over 0.2, 0.5, 1, 2, 4 Hz) through
AdaptiveSensing/scripts/evaluate_fixed_sections.py, one fresh process per configuration (identical conditions to the
TMP01 per-rate runs), distributed over the given GPUs (one worker per GPU, shared queue).

    python AdaptiveSensing/scripts/run_sts01_factorial.py --gpus 6 7 [--rates 0.2 0.5 1 2 4] [--dry_run]

Resumable: a configuration whose run folder already has episodes.csv is skipped. A configuration fails when the
process exits non-zero or its summary.json checks do not pass; failures are listed at the end (exit code 1) and can be
re-run with the same command after deleting / fixing them. Console logs: <eval>/<protocol>/console/<config>.log;
progress: <eval>/<protocol>/factorial_progress.jsonl. Standard library only (the evaluations run under bin/isaacrun).
"""
import argparse
import itertools
import json
import os
import queue
import subprocess
import sys
import threading
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXP = "STS01_chase-ob1ob2_fixed-sections-nav05"
RUN = "2026-11-23_20-21-14_ppo_cnn_n4096_s42_model_1499"      # NAV05 run + checkpoint (evaluate_fixed_sections.py default)

p = argparse.ArgumentParser()
p.add_argument("--gpus", type=int, nargs="+", required=True)
p.add_argument("--rates", type=float, nargs="+", default=[0.2, 0.5, 1.0, 2.0, 4.0])
p.add_argument("--protocol", default="ref_chase_c10")
p.add_argument("--timeout_s", type=int, default=1200, help="per configuration")
p.add_argument("--dry_run", action="store_true")
a = p.parse_args()

proto_dir = os.path.join(ROOT, "AdaptiveSensing", "eval", "spatiotemporal", EXP, a.protocol)
cons = os.path.join(proto_dir, "console")
os.makedirs(cons, exist_ok=True)
configs = list(itertools.product(a.rates, repeat=3))
name = lambda c: "-".join(f"{h:g}" for h in c)  # noqa: E731
run_dir = lambda c: os.path.join(proto_dir, f"{RUN}_sections{name(c)}Hz")  # noqa: E731
todo = [c for c in configs if not os.path.exists(os.path.join(run_dir(c), "episodes.csv"))]
print(f"[STS01] {len(configs)} configurations, {len(configs) - len(todo)} done, {len(todo)} to run on GPUs {a.gpus}",
      flush=True)
if a.dry_run:
    for c in todo:
        print("  ", name(c))
    sys.exit(0)
q = queue.Queue()
for c in todo:
    q.put(c)
lock, failures = threading.Lock(), []
t0 = time.time()
done_count = [0]


def worker(gpu):
    while True:
        try:
            c = q.get_nowait()
        except queue.Empty:
            return
        log = os.path.join(cons, f"sections{name(c)}Hz.log")
        cmd = [os.path.join(ROOT, "bin", "isaacrun"), "-g", str(gpu), "-t", str(a.timeout_s), "-l", log, "--",
               "python", "AdaptiveSensing/scripts/evaluate_fixed_sections.py", "--headless", "--protocol", a.protocol,
               "--sections", *[f"{h:g}" for h in c]]
        ts = time.time()
        rc = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
        ok, info = rc == 0, {}
        sj = os.path.join(run_dir(c), "summary.json")
        if ok and os.path.exists(sj):
            s = json.load(open(sj))
            ok = bool(s["checks"]["pass"])
            info = {"success_rate": s["success_rate"], "fse_mean": s["fse_mean_std"][0], "timing_s": s["timing_s"]}
        else:
            ok = False
        with lock:
            done_count[0] += 1
            rec = {"config": name(c), "gpu": gpu, "exit": rc, "ok": ok, "wall_s": round(time.time() - ts, 1), **info}
            if not ok:
                failures.append(name(c))
            with open(os.path.join(proto_dir, "factorial_progress.jsonl"), "a") as f:
                f.write(json.dumps(rec) + "\n")
            el = time.time() - t0
            print(f"[STS01] {done_count[0]}/{len(todo)} {name(c)} gpu{gpu} ok={ok} {rec['wall_s']} s "
                  f"(elapsed {el / 60:.1f} min, ~{el / done_count[0] * (len(todo) - done_count[0]) / 60:.0f} min left)",
                  flush=True)


threads = [threading.Thread(target=worker, args=(g,)) for g in a.gpus]
for t in threads:
    t.start()
for t in threads:
    t.join()
print(f"[STS01] finished in {(time.time() - t0) / 60:.1f} min; failures: {failures or 'none'}", flush=True)
sys.exit(1 if failures else 0)
