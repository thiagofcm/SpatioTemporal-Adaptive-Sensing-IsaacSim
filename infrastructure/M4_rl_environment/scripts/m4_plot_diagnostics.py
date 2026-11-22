"""Figures for the M4 follow-up diagnostics (reads infrastructure/M4_rl_environment/results/diag; no Isaac Sim needed)."""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
D = os.path.join(os.environ.get("ST_ROOT", "."), "infrastructure", "M4_rl_environment", "results", "diag")
fig, ax = plt.subplots(1, 3, figsize=(19, 5.2))
# (a) trajectories, deterministic vs stochastic policy (first 64 envs, first episode portion)
for tag, col in (("stochastic", "tab:blue"), ("deterministic", "tab:red")):
    tr = np.load(os.path.join(D, f"traj_{tag}.npy"))[:45]                      # (steps, 64, 2)
    for e in range(tr.shape[1]):
        ax[0].plot(tr[:, e, 0], tr[:, e, 1], color=col, lw=2.5 if tag == "deterministic" else 0.6, alpha=1 if tag == "deterministic" else 0.5, label=tag if e == 0 else None)
ax[0].axhline(2.835, color="k", lw=1); ax[0].axvline(2.835, color="k", lw=1)
ax[0].plot(2.5, 2.5, "ko"); ax[0].set_aspect("equal"); ax[0].set_xlim(1.6, 2.9); ax[0].set_ylim(1.6, 2.9)
ax[0].set_title("D2: first 45 steps, 64 envs (black lines: wall faces)\ndeterministic inference policy = one path for all envs", fontsize=9)
ax[0].legend(fontsize=8)
# (b) CoM vs link velocity at the largest excursion (random actions)
v = np.load(os.path.join(D, "velocity_traces.npz"))
i = int(np.nanargmax(v["v_com_b"])); s = slice(i - 60, i + 25); t = (np.arange(len(v["v_com_b"]))[s] - i) * float(v["dt"])
for n, lab, st in (("v_cmd_eff", "command model output", "k--"), ("v_com_b", "CoM velocity (old obs[45])", "tab:red"),
                   ("v_link_b", "base_link velocity (new obs[45])", "tab:blue"), ("v_wheel", "wheel odometry", "tab:green"),
                   ("v_fd", "finite difference of base position", "tab:orange")):
    ax[1].plot(t, v[n][s], st if n == "v_cmd_eff" else "-", color=None if n == "v_cmd_eff" else st, label=lab, lw=1.3)
ax[1].axhline(0.22, color="0.5", lw=0.8, ls=":"); ax[1].set_xlabel("time relative to peak [s]"); ax[1].set_ylabel("forward velocity [m/s]")
ax[1].set_title("D3: largest excursion (random actions): only the CoM velocity exceeds v_max", fontsize=9); ax[1].legend(fontsize=7.5)
# (c) reset drop + rocking (straight drive group, from rest)
r = np.load(os.path.join(D, "rocking_traces.npz")); n = 60; tt = np.arange(n) * float(r["dt"])
a2 = ax[2].twinx()
ax[2].plot(tt, r["g1_v_link"][:n], color="tab:blue", label="base_link v"); ax[2].plot(tt, r["g1_v_com"][:n], color="tab:red", label="CoM v")
ax[2].plot(tt, r["g1_cmd_v"][:n], "k--", label="command")
a2.plot(tt, np.degrees(r["g1_pitch"][:n]), color="tab:purple", lw=1, label="pitch [deg]"); a2.plot(tt, r["g1_z"][:n] * 1000, color="tab:brown", lw=1, label="base z [mm]")
ax[2].set_xlabel("time since reset [s]"); ax[2].set_ylabel("m/s"); a2.set_ylabel("deg / mm")
ax[2].set_title("D3b: reset spawns at z=0, rests at z=-4.45 mm -> drop + pitch transient\n(first 0.1 s, command still 0 due to the 0.14 s delay)", fontsize=9)
h1, l1 = ax[2].get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels(); ax[2].legend(h1 + h2, l1 + l2, fontsize=7.5, loc="center right")
fig.tight_layout(); out = os.path.join(D, "m4_followup_diagnostics.png"); fig.savefig(out, dpi=110); print(out)
