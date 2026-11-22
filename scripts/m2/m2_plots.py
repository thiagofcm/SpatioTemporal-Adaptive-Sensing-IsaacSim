"""M2 figures: LiDAR scans, validation errors, noise statistics, benchmark. CPU only.

    python scripts/m2/m2_plots.py
"""
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon

ROOT = os.environ.get("ST_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
RES = os.path.join(ROOT, "results", "m2")
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)
D = dict(np.load(os.path.join(RES, "m2_validation_data.npz")))
V = json.load(open(os.path.join(RES, "m2_validation_200hz.json")))
AZ = np.deg2rad(D["azimuths_deg"])
MOUNT = D["mount"]


def draw_walls(ax, walls):
    for cx, cy, L, yaw, th in walls:
        c, s = math.cos(yaw), math.sin(yaw)
        pts = [(-L / 2, -th / 2), (L / 2, -th / 2), (L / 2, th / 2), (-L / 2, th / 2)]
        ax.add_patch(Polygon([(cx + c * x - s * y, cy + s * x + c * y) for x, y in pts], fc="0.45", ec="k", lw=0.4))


def draw_robot(ax, bxy, byaw, color="tab:blue"):
    th = np.linspace(-np.pi / 2, np.pi / 2, 40)
    front = np.stack([0.177 * np.cos(th), 0.177 * np.sin(th)], 1)
    rear = np.stack([0.164 * np.cos(th + np.pi), 0.164 * np.sin(th + np.pi)], 1)
    poly = np.concatenate([front, rear])
    R = np.array([[math.cos(byaw), -math.sin(byaw)], [math.sin(byaw), math.cos(byaw)]])
    ax.add_patch(Polygon(poly @ R.T + bxy, fc=color, alpha=0.3, ec=color, lw=0.8))
    ax.arrow(bxy[0], bxy[1], 0.25 * math.cos(byaw), 0.25 * math.sin(byaw), width=0.01, color=color)


# ---------------- Fig 1: representative scans ----------------
fig, axes = plt.subplots(2, 4, figsize=(20, 10.5))
for i, ax in enumerate(axes.ravel()):
    draw_walls(ax, D["walls"])
    for cx, cy, r in D["static"]:
        ax.add_patch(Circle((cx, cy), r, fc="tab:orange", alpha=0.6, ec="k", lw=0.4))
    ax.add_patch(Circle(D["snap_mover_xy"][i], float(D["mover_r"]), fc="tab:red", alpha=0.5, ec="k", lw=0.4))
    draw_robot(ax, D["snap_base_xy"][i], D["snap_base_yaw"][i])
    o, yaw = D["snap_sensor_xy"][i], D["snap_sensor_yaw"][i]
    r, a = D["snap_ranges"][i], D["snap_analytic"][i]
    for k in range(len(AZ)):
        ang = yaw + AZ[k]
        e = o + r[k] * np.array([math.cos(ang), math.sin(ang)])
        ax.plot([o[0], e[0]], [o[1], e[1]], color="tab:green" if r[k] < 3.5 - 1e-6 else "0.75", lw=0.6)
        ea = o + min(a[k], 3.5) * np.array([math.cos(ang), math.sin(ang)])
        ax.plot(*ea, "k+", ms=4)
    e0 = o + r[0] * np.array([math.cos(yaw), math.sin(yaw)])
    ax.plot([o[0], e0[0]], [o[1], e0[1]], color="magenta", lw=1.6)
    err = np.abs(r - np.minimum(a, 3.5)).max()
    ax.set_title(f"env {i}: max |GPU - analytic| = {err * 1e3:.2f} mm", fontsize=9)
    ax.set_xlim(-3.1, 3.1)
    ax.set_ylim(-3.1, 3.1)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)
fig.suptitle("M2: 40-beam GPU LiDAR (Isaac Lab MultiMeshRayCaster) on the articulated TB4, random poses, noise off.\n"
             "green = GPU ray to hit (grey = no return, drawn to 3.5 m), black + = analytic oracle end point, "
             "magenta = beam 0 (robot forward); walls 0.6 m, cylinders 0.5 m", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "m2_lidar_scans_topview.png"), dpi=110)
plt.close(fig)

# ---------------- Fig 2: range vs beam index + V2 ordering ----------------
fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
for i, ax in zip((0, 1), axes[:2]):
    ax.plot(np.arange(40), D["snap_ranges"][i], "o-", ms=4, label="GPU ray cast")
    ax.plot(np.arange(40), np.minimum(D["snap_analytic"][i], 3.5), "k+", ms=8, label="analytic oracle")
    ax.set_xlabel("beam index (beam i at i x 9 deg CCW from forward)")
    ax.set_ylabel("range [m]")
    ax.set_title(f"env {i}: range per beam", fontsize=10)
    ax.grid(lw=0.3)
    ax.legend(fontsize=8)
ax = axes[2]
k = D["v2_k"]
am = D["v2_ranges"].argmin(1)
ax.plot(k, am, "o", ms=5)
ax.plot([0, 39], [0, 39], "k-", lw=0.6)
ax.set_xlabel("beam index the mover was placed on")
ax.set_ylabel("beam index of minimum GPU range")
v2 = V["V2_beam_ordering"]
ax.set_title(f"V2 beam ordering: {v2['argmin_equals_oracle_argmin']}/{v2['envs']} match oracle argmin\n"
             f"(mover visible on expected beam in {v2['mover_visible_on_expected_beam']}; first 40 envs shown)", fontsize=10)
ax.grid(lw=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "m2_lidar_ranges_and_ordering.png"), dpi=110)
plt.close(fig)

# ---------------- Fig 3: error distributions + moving trace ----------------
e1, k1 = D["e1"], D["k1"]
fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
ax = axes[0]
for kind, lab, col in ((1, "walls (boxes)", "tab:blue"), (2, "static cylinders", "tab:orange"), (3, "moving cylinder", "tab:red")):
    x = np.abs(e1[k1 == kind]) * 1e3
    x = x[x > 0] if (x > 0).any() else x
    ax.hist(np.clip(x, 1e-4, 100), bins=np.logspace(-4, 2, 80), histtype="step", lw=1.4, color=col,
            label=f"{lab}: median {np.median(np.abs(e1[k1 == kind])) * 1e3:.3f} mm, p99 {np.quantile(np.abs(e1[k1 == kind]), 0.99) * 1e3:.2f} mm")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("|GPU - analytic| [mm] (clipped to [1e-4, 100])")
ax.set_ylabel("rays")
ax.legend(fontsize=7)
ax.set_title("V1: 327,680 rays, 8 x 1024 random poses (noise off)", fontsize=10)
ax = axes[1]
th = np.linspace(-0.17, 0.17, 400)
ax.text(0.02, 0.95, "Cylinders: Isaac Lab tessellates USD cylinders into 32 facets.\n"
        "Expected range excess for r = 0.17 m (simulated): mean 0.85 mm,\np99 4.4 mm, rare grazing misses.\n"
        f"Measured (V1): p99 {V['V1_random_poses']['static_cylinder']['p99_abs_m'] * 1e3:.1f} mm (static), "
        f"{V['V1_random_poses']['moving_cylinder']['p99_abs_m'] * 1e3:.1f} mm (moving).\n"
        f"Walls: max {V['V1_random_poses']['wall']['max_abs_m'] * 1e3:.2f} mm (float32).\n"
        f"No-return rays: exactly 3.5 m, hit flag False ({V['V1_random_poses']['no_return']['n']} rays).\n"
        f"Hit/miss flag mismatches: {V['V1_random_poses']['hit_flag_mismatch']} of 327,680 (grazing rays on facets).\n\n"
        "V4 mounting: origin = base_link + R*(-0.040, 0, 0.1929) within\n"
        f"{V['V4_range_and_mount']['origin_vs_base_plus_urdf_offset_max_m'] * 1e6:.1f} um; equal to the ray caster's ray starts;\n"
        f"scan plane {V['V4_range_and_mount']['origin_height_above_ground_mean_m']:.4f} m above ground.",
        va="top", fontsize=9, family="monospace", transform=ax.transAxes)
ax.axis("off")
ax = axes[2]
r3, a3 = D["v3_ranges"], np.minimum(D["v3_analytic"], 3.5)
t3 = np.arange(len(r3)) * 0.2
for b, col in ((0, "tab:blue"), (10, "tab:green"), (20, "tab:orange"), (30, "tab:purple")):
    ax.plot(t3, r3[:, b], "-", color=col, lw=1.2, label=f"beam {b} ({b * 9} deg) GPU")
    ax.plot(t3, a3[:, b], "k+", ms=4)
ax.set_xlabel("time [s] (robot driving, mover circling)")
ax.set_ylabel("range [m]")
v3 = V["V3_moving"]["valid_scans"]
ax.set_title(f"V3 moving (env 0): GPU (lines) vs oracle (+)\nall envs: walls p99 {v3['wall']['p99_abs_m'] * 1e3:.3f} mm, "
             f"moving cyl. p99 {v3['moving_cylinder']['p99_abs_m'] * 1e3:.1f} mm", fontsize=10)
ax.legend(fontsize=7)
ax.grid(lw=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "m2_lidar_validation_errors.png"), dpi=110)
plt.close(fig)

# ---------------- Fig 4: noise ----------------
n5 = V["V5_noise"]
fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
ax = axes[0]
ax.hist(D["u_sample"], bins=100, density=True, histtype="stepfilled", alpha=0.6)
ax.plot([-1, -1, 1, 1], [0, 0.5, 0.5, 0], "k-", lw=1)
ax.set_xlabel("u = (noisy - clean) / (a(d) d)")
ax.set_title(f"normalized error: U(-1,1) expected; mean {n5['u_mean']:+.1e}, var {n5['u_var']:.4f} (1/3)\n"
             f"KS p-values over 10 subsamples: {min(n5['ks_pvalues_10_disjoint_subsamples_200k']):.2f}-"
             f"{max(n5['ks_pvalues_10_disjoint_subsamples_200k']):.2f}", fontsize=9)
ax = axes[1]
bands = n5["bands"]
names = list(bands)
x = np.arange(len(names))
ax.bar(x - 0.2, [bands[b]["std_rel"] * 100 for b in names], 0.4, label="measured std / d")
ax.bar(x + 0.2, [bands[b]["expected_std_rel"] * 100 for b in names], 0.4, label="expected a/sqrt(3)")
ax.set_xticks(x, names)
ax.set_ylabel("relative std [%]")
ax.legend(fontsize=8)
ax.set_title(f"per datasheet band; corr. adjacent beams {n5['corr_adjacent_beams']:+.4f},\n"
             f"successive scans {n5['corr_successive_scans']:+.4f}", fontsize=9)
ax = axes[2]
for j in range(min(5, len(D["noisy_example"]))):
    ax.plot(np.arange(40), D["noisy_example"][j, 0], ".", ms=4, alpha=0.7)
ax.plot(np.arange(40), D["clean_example"][0], "k-", lw=1, label="clean")
ax.set_xlabel("beam index")
ax.set_ylabel("range [m]")
ax.set_title("5 noisy scans of one pose (dots) vs clean (line)", fontsize=9)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "m2_lidar_noise.png"), dpi=110)
plt.close(fig)

# ---------------- Fig 5: benchmark ----------------
bf = os.path.join(RES, "benchmark_clean.jsonl")   # clean sweep: one process at a time on an otherwise idle GPU 5
if os.path.exists(bf):
    rows = [json.loads(l) for l in open(bf) if l.strip()]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for hz, ls in ((200.0, "-"), (100.0, "--")):
        rr = sorted([r for r in rows if r["physics_hz"] == hz and r["solver_iters"] == [16, 2]], key=lambda r: r["num_envs"])
        if not rr:
            continue
        n = [r["num_envs"] for r in rr]
        for ph, col in (("A", "tab:gray"), ("B", "tab:blue"), ("C", "tab:red")):
            axes[0].plot(n, [r[ph]["control_steps_per_s"] / 1e3 for r in rr], ls, marker="o", color=col,
                         label=f"{int(hz)} Hz, {dict(A='physics only', B='+ LiDAR', C='full control loop')[ph]}")
        axes[1].plot(n, [r["C"]["gpu_mem_process_mib"] / 1024 for r in rr], ls, marker="o", color="k", label=f"{int(hz)} Hz")
        axes[2].plot(n, [100 * (1 - r["B"]["control_steps_per_s"] / r["A"]["control_steps_per_s"]) for r in rr], ls, marker="o",
                     color="tab:blue", label=f"{int(hz)} Hz: LiDAR overhead")
        axes[2].plot(n, [100 * (1 - r["C"]["control_steps_per_s"] / r["A"]["control_steps_per_s"]) for r in rr], ls, marker="o",
                     color="tab:red", label=f"{int(hz)} Hz: full loop overhead")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.grid(lw=0.3)
        ax.set_xlabel("parallel environments")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("control steps (5 Hz) per second [k]")
    axes[0].set_title("throughput (1 control step = 0.2 s simulated)", fontsize=10)
    axes[1].set_ylabel("GPU memory of the process [GiB]")
    axes[1].set_title("GPU memory (nvidia-smi, per process)", fontsize=10)
    axes[2].set_ylabel("throughput reduction vs physics only [%]")
    axes[2].set_title("cost of LiDAR and of the full control loop", fontsize=10)
    fig.suptitle("M2 benchmark: articulated TB4 + 14-wall arena + 4 cylinders + 40-beam LiDAR, one RTX A6000", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "m2_benchmark.png"), dpi=110)
    plt.close(fig)
print("figures written to", FIG)
