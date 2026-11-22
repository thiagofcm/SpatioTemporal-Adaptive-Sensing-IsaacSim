"""M1 calibration plots and acceptance metrics (CPU only; reads results/m1/*.npz and the real recordings).

    python scripts/m1/m1_plots.py            (conda-env python with PYTHONPATH from env.sh)
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from st_nav.calibration.real_data import load_all, startup_mask, step_metrics, transitions

ROOT = os.environ.get("ST_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
RES = os.path.join(ROOT, "results", "m1")
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)
d = np.load(os.path.join(RES, "m1_calibration_final.npz"))          # selected model + no-model, all runs
g = np.load(os.path.join(RES, "m1_calibration_grid.npz"))           # error surfaces
cal = json.load(open(os.path.join(RES, "m1_calibration_final.json")))
runs = {r.name: r for r in load_all() if r.duration > 5.0}
names = [str(n) for n in d["run_names"]]
t = d["t_rec"]
params = g["params"]
fm = cal["final_model"]
lab_best = (f"sim calibrated (v: delay {fm['delay_v_s']:.2f} s, tau {fm['tau_v_s']:.2f} s, a_max {fm.get('accel_v_max', 0):.1f} m/s2 | "
            f"w: delay {fm['delay_w_s']:.2f} s, tau {fm['tau_w_s']:.2f} s, k_w {fm['k_w']:.2f})")
bj = {"delay_s": fm["delay_v_s"], "tau_s": fm["tau_v_s"], "k_w": fm["k_w"]}


def shade_startup(ax, r):
    for tr in transitions(r):
        if tr["from_rest"]:
            ax.axvspan(tr["t"], tr["t"] + 2.5, color="0.9", zorder=0)


# ---------------- Fig A: complete step responses, all runs ----------------
fig, axes = plt.subplots(len(names), 1, figsize=(13, 2.1 * len(names)), sharex=False)
for ax, (i, n) in zip(axes, enumerate(names)):
    r = runs[n]
    comp = "w" if r.kind == "angular" else "v"
    real = r.odo_w if comp == "w" else r.odo_v
    c = r.command_at(t)[1 if comp == "w" else 0]
    shade_startup(ax, r)
    ax.step(t, c, where="post", color="k", lw=0.8, ls="--", label="command")
    ax.plot(r.odo_t, real, ".", ms=2.5, color="tab:blue", label="real odometry")
    ax.plot(t, d[f"sim_{comp}_raw"][i], color="tab:gray", lw=0.8, label="sim, no command model")
    ax.plot(t, d[f"sim_{comp}_best"][i], color="tab:red", lw=1.1, label=lab_best)
    if r.kind == "combined":
        ax2 = ax.twinx()
        ax2.plot(r.odo_t, r.odo_w, ".", ms=2, color="tab:cyan")
        ax2.plot(t, d["sim_w_best"][i], color="tab:orange", lw=0.9)
        ax2.set_ylabel("w [rad/s]", fontsize=7, color="tab:orange")
        ax2.tick_params(labelsize=6)
    ax.set_xlim(0, r.duration)
    ax.set_ylabel(f"{n}\n{'w [rad/s]' if comp == 'w' else 'v [m/s]'}", fontsize=7)
    ax.tick_params(labelsize=6)
axes[0].legend(fontsize=7, ncol=4, loc="upper right")
axes[-1].set_xlabel("time [s]  (grey bands: 2.5 s after a start from rest, excluded from the fit)")
fig.suptitle("M1: complete step responses, real TurtleBot4 vs articulated simulation (all 16 recorded runs)")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "m1_step_responses_all_runs.png"), dpi=110)
plt.close(fig)

# ---------------- Fig B: three representative runs, zoomed ----------------
reps = [n for n in ("run_1", "angular_run_1", "combined_run_1") if n in names]
fig, axes = plt.subplots(len(reps), 2, figsize=(15, 3.2 * len(reps)))
for row, n in zip(axes, reps):
    i = names.index(n)
    r = runs[n]
    for ax, comp in zip(row, ("v", "w")):
        real = r.odo_v if comp == "v" else r.odo_w
        c = r.command_at(t)[0 if comp == "v" else 1]
        shade_startup(ax, r)
        ax.step(t, c, where="post", color="k", lw=0.8, ls="--", label="command")
        ax.plot(r.odo_t, real, ".-", ms=3, lw=0.4, color="tab:blue", label="real odometry (20 Hz)")
        ax.plot(t, d[f"sim_{comp}_raw"][i], color="tab:gray", lw=0.9, label="sim, no command model")
        ax.plot(t, d[f"sim_{comp}_best"][i], color="tab:red", lw=1.2, label="sim calibrated")
        ax.set_xlim(0, r.duration)
        ax.set_title(f"{n}: {'linear v [m/s]' if comp == 'v' else 'angular w [rad/s]'}", fontsize=9)
        ax.tick_params(labelsize=7)
axes[0, 0].legend(fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "m1_step_responses_representative.png"), dpi=120)
plt.close(fig)

# ---------------- Fig C: error surfaces ----------------
delays = sorted(set(params[:, 0]))
taus = sorted(set(params[:, 1]))
kw_best = bj["k_w"]
fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
for ax, key, ttl in ((axes[0], "rms_v", "RMS v [m/s] (k_w irrelevant)"), (axes[1], "rms_w", f"RMS w [rad/s] at k_w={kw_best}"),
                     (axes[2], "rms_w_start", f"RMS w in start-up windows, k_w={kw_best}")):
    sel = params[:, 2] == kw_best
    Z = np.full((len(taus), len(delays)), np.nan)
    for (dl, ta, _), z in zip(params[sel], g[key][sel]):
        Z[taus.index(ta), delays.index(dl)] = z
    hd, ht = (delays[1] - delays[0]) / 2, (taus[1] - taus[0]) / 2
    im = ax.imshow(Z, origin="lower", aspect="auto", cmap="viridis",
                   extent=(delays[0] - hd, delays[-1] + hd, taus[0] - ht, taus[-1] + ht))
    if key == "rms_v":
        ax.plot(fm["delay_v_s"], fm["tau_v_s"], "r*", ms=12)
    else:
        ax.plot(fm["delay_w_s"], fm["tau_w_s"], "r*", ms=12)
    ax.set_xlabel("pure delay [s]")
    ax.set_ylabel("residual lag tau [s]")
    ax.set_title(ttl, fontsize=9)
    fig.colorbar(im, ax=ax)
fig.suptitle("M1: fit error over the command-model grid (physics in the loop, 702 combinations x 16 runs; red star = selected)")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "m1_calibration_error_surfaces.png"), dpi=120)
plt.close(fig)

# ---------------- per-transition delay / rise: real vs calibrated sim ----------------
rows = []
for i, n in enumerate(names):
    r = runs[n]
    trs = transitions(r)
    for j, tr in enumerate(trs):
        t_next = trs[j + 1]["t"] if j + 1 < len(trs) else r.duration
        for comp, y0, y1 in (("v", tr["v0"], tr["v1"]), ("w", tr["w0"], tr["w1"])):
            if abs(y1 - y0) < 1e-6:
                continue
            real = r.odo_v if comp == "v" else r.odo_w
            dr, rr = step_metrics(r.odo_t, real, tr["t"], y0, y1, t_next)
            dsim, rsim = step_metrics(t, d[f"sim_{comp}_best"][i], tr["t"], y0, y1, t_next)
            draw, rraw = step_metrics(t, d[f"sim_{comp}_raw"][i], tr["t"], y0, y1, t_next)
            rows.append(dict(run=n, t=tr["t"], comp=comp, y0=y0, y1=y1, from_rest=tr["from_rest"], real_delay=dr,
                             real_rise=rr, sim_delay=dsim, sim_rise=rsim, raw_delay=draw, raw_rise=rraw))


def agg(sel, key):
    v = np.array([r[key] for r in sel], dtype=float)
    v = v[np.isfinite(v)]
    return (float(np.mean(v)), float(np.std(v)), int(v.size)) if v.size else (np.nan, np.nan, 0)


summary = {}
for regime, sel in (("in_motion", [r for r in rows if not r["from_rest"]]), ("from_rest", [r for r in rows if r["from_rest"]])):
    for comp in ("v", "w"):
        s = [r for r in sel if r["comp"] == comp]
        e = {k: agg(s, k) for k in ("real_delay", "sim_delay", "raw_delay", "real_rise", "sim_rise", "raw_rise")}
        e["delay_diff_s"] = e["sim_delay"][0] - e["real_delay"][0]
        e["rise_diff_s"] = e["sim_rise"][0] - e["real_rise"][0]
        summary[f"{regime}_{comp}"] = e
with open(os.path.join(RES, "m1_transition_metrics.json"), "w") as f:
    json.dump({"summary": summary, "transitions": rows}, f, indent=1, default=float)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, key, ttl in ((axes[0], "delay", "response delay (time to 10 %)"), (axes[1], "rise", "rise time (10 -> 90 %)")):
    for regime, mk in ((False, "o"), (True, "x")):
        for comp, col in (("v", "tab:blue"), ("w", "tab:red")):
            s = [r for r in rows if r["from_rest"] == regime and r["comp"] == comp]
            ax.plot([r[f"real_{key}"] for r in s], [r[f"sim_{key}"] for r in s], mk, color=col, ms=4, alpha=0.7,
                    label=f"{comp}, {'from rest' if regime else 'in motion'}")
    lim = 2.0 if key == "delay" else 1.0
    xx = np.linspace(0, lim, 2)
    ax.plot(xx, xx, "k-", lw=0.8)
    ax.fill_between(xx, xx - 0.03, xx + 0.03, color="0.85", label="±0.03 s")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel(f"real {key} [s]")
    ax.set_ylabel(f"calibrated sim {key} [s]")
    ax.set_title(ttl, fontsize=10)
axes[0].legend(fontsize=7)
fig.suptitle("M1: per-transition timing, real vs calibrated simulation")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "m1_transition_timing.png"), dpi=120)
plt.close(fig)

# ---------------- validation figure (T2 steady-state grid) ----------------
val = json.load(open(os.path.join(RES, "m1_validation_200hz.json")))
g = val["T2_steady_state"]
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, comp in zip(axes, ("v", "w")):
    ax.plot([x[f"{comp}_expected"] for x in g], [x[f"{comp}_meas"] for x in g], "o", ms=5)
    for x in g:
        if x["saturated"]:
            ax.plot(x[f"{comp}_cmd"], x[f"{comp}_meas"], "x", color="tab:red", ms=6)
    lim = 0.25 if comp == "v" else 2.1
    ax.plot([-lim, lim], [-lim, lim], "k-", lw=0.7)
    ax.set_xlabel(f"expected {comp} (command after scale saturation)")
    ax.set_ylabel(f"measured steady-state {comp}")
    ax.set_title(f"steady-state {comp}: dots = vs saturation model; red x = vs raw command (saturated cases)", fontsize=9)
fig.suptitle("M1 T2: steady-state tracking of the articulated TB4 (35 commands, no command-model lag)")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "m1_steady_state_tracking.png"), dpi=120)
plt.close(fig)
print(json.dumps(summary, indent=1, default=float))
