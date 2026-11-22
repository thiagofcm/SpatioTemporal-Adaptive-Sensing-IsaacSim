"""Coherence and repeatability checks of the clean M2 benchmark (CPU only).

Inputs : results/m2/benchmark_clean.jsonl (main sweep), benchmark_clean_repeats.jsonl (repeats), benchmark_clean_runlog.txt
Output : results/m2/benchmark_check.json and a markdown table on stdout.

Checks
  isolation   : every run started with no other process on its GPU (run log)
  stability   : 0 unstable envs, no NaN
  phases      : A >= B >= C  (adding work must not speed things up; 3 % tolerance for timing noise)
  scaling     : control steps/s non-decreasing with num_envs (per rate / solver setting)
  rate        : 100 Hz >= 200 Hz at equal num_envs and solver setting
  repeat      : |main - repeat| / mean <= 5 % per phase
"""
import json
import os
import re

RES = os.path.join(os.environ.get("ST_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))),
                   "results", "m2")
main = [json.loads(l) for l in open(os.path.join(RES, "benchmark_clean.jsonl")) if l.strip()]
rep_f = os.path.join(RES, "benchmark_clean_repeats.jsonl")
reps = [json.loads(l) for l in open(rep_f) if l.strip()] if os.path.exists(rep_f) else []
log = open(os.path.join(RES, "benchmark_clean_runlog.txt")).read()
key = lambda r: (r["num_envs"], int(r["physics_hz"]), tuple(r["solver_iters"]))
cs = lambda r, ph: r[ph]["control_steps_per_s"]
issues = []

# isolation
others = [int(x) for x in re.findall(r"other procs on GPU5: (\d+)", log)]
if any(o != 0 for o in others):
    issues.append(f"isolation: {sum(o != 0 for o in others)} runs started with another process on the GPU")
# duplicates in the main sweep
keys = [key(r) for r in main]
if len(keys) != len(set(keys)):
    issues.append("main sweep contains duplicated configurations")
for r in main + reps:
    if r["unstable_envs"] or r["nan_seen"]:
        issues.append(f"stability: {key(r)} unstable={r['unstable_envs']} nan={r['nan_seen']}")
    if not (cs(r, "A") * 1.03 >= cs(r, "B") and cs(r, "B") * 1.03 >= cs(r, "C")):
        issues.append(f"phases: {key(r)} A={cs(r, 'A'):.0f} B={cs(r, 'B'):.0f} C={cs(r, 'C'):.0f}")
groups = {}
for r in main:
    groups.setdefault((int(r["physics_hz"]), tuple(r["solver_iters"])), []).append(r)
for g, rr in groups.items():
    rr.sort(key=lambda r: r["num_envs"])
    for a, b in zip(rr, rr[1:]):
        for ph in "ABC":
            if cs(b, ph) < 0.97 * cs(a, ph):
                issues.append(f"scaling: {g} {ph} {a['num_envs']}->{b['num_envs']} drops {cs(a, ph):.0f}->{cs(b, ph):.0f}")
idx = {key(r): r for r in main}
for (n, hz, it), r in idx.items():
    if hz == 100 and (n, 200, it) in idx:
        for ph in "ABC":
            if cs(r, ph) < cs(idx[(n, 200, it)], ph):
                issues.append(f"rate: {n} envs {it} {ph}: 100 Hz slower than 200 Hz")
rep_dev = []
for r in reps:
    m = idx.get(key(r))
    if m is None:
        continue
    d = {ph: abs(cs(r, ph) - cs(m, ph)) / (0.5 * (cs(r, ph) + cs(m, ph))) for ph in "ABC"}
    rep_dev.append({"config": list(key(r)), **{f"dev_{ph}": d[ph] for ph in "ABC"},
                    **{f"main_{ph}": cs(m, ph) for ph in "ABC"}, **{f"repeat_{ph}": cs(r, ph) for ph in "ABC"}})
    for ph in "ABC":
        if d[ph] > 0.05:
            issues.append(f"repeat: {key(r)} {ph} differs {100 * d[ph]:.1f} % (main {cs(m, ph):.0f}, repeat {cs(r, ph):.0f})")
out = {"runs_main": len(main), "runs_repeat": len(reps), "issues": issues, "repeatability": rep_dev,
       "max_repeat_dev": max((max(x["dev_A"], x["dev_B"], x["dev_C"]) for x in rep_dev), default=None)}
json.dump(out, open(os.path.join(RES, "benchmark_check.json"), "w"), indent=1)

print("| Envs | Physics | Solver | A physics only | B + LiDAR | C full loop | C physics env-steps/s | LiDAR cost | loop cost | GPU mem | setup |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for r in sorted(main, key=lambda r: (-r["physics_hz"], r["solver_iters"], r["num_envs"])):
    print(f"| {r['num_envs']:,} | {int(r['physics_hz'])} Hz | {r['solver_iters'][0]}/{r['solver_iters'][1]} | "
          f"{cs(r, 'A') / 1e3:.1f}k | {cs(r, 'B') / 1e3:.1f}k | {cs(r, 'C') / 1e3:.1f}k | "
          f"{r['C']['physics_env_steps_per_s'] / 1e6:.2f} M | {100 * (1 - cs(r, 'B') / cs(r, 'A')):.0f} % | "
          f"{100 * (1 - cs(r, 'C') / cs(r, 'A')):.0f} % | {r['C']['gpu_mem_process_mib'] / 1024:.1f} GiB | {r['setup_s']:.0f} s |")
print()
for x in rep_dev:
    print("repeat", x["config"], " ".join(f"{ph}: {x[f'main_{ph}'] / 1e3:.1f}k vs {x[f'repeat_{ph}'] / 1e3:.1f}k "
                                           f"({100 * x[f'dev_{ph}']:.1f} %)" for ph in "ABC"))
print("ISSUES:", issues if issues else "none")
