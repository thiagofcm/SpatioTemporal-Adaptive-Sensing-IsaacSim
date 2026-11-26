"""Profile the TMP02 training loop (rollout and PPO update) without changing it: where does the time go?

Same setup and rollout code path as train_temporal_ppo.py (env from the TMP02 config, frozen NAV05, TemporalSensing,
SensingAgent, GPU buffers), at a chosen number of envs. Measurements:

  A. plain rollout (exactly the trainer's loop, no instrumentation): wall time per tick, transitions / s, process CPU
     time / wall time (CPU-bound indicator);
  B. instrumented rollout: CUDA-synchronized timers around every component (physics substeps, ray casting, env
     bookkeeping, NAV05, sensing agent and its CNN, gate, buffer writes); inclusive times per tick. The extra
     synchronizations make this slower than A; it is for attribution;
  C. host synchronizations per tick by source file (torch.cuda.set_sync_debug_mode);
  D. CNN reuse on unchanged held scans: share of (tick, env) whose held image is unchanged, encoder time on all images
     vs only the changed ones, and whether the cached features are bitwise equal to recomputed ones (and if not, the
     effect on logits / log-probs / sampled actions);
  E. PPO update: time of the full update (epochs x minibatches) and of one minibatch, on the rollout of A.

    bin/isaacrun -g 0 -- python AdaptiveSensing/scripts/profile_temporal_rollout.py --headless --num_envs 512
Output: AdaptiveSensing/eval/temporal/TMP02_chase-ob1ob2_adaptive-nav05/profiling/profile_n<N>[_<tag>].json
"""
import argparse
import json
import os
import sys
import time
import warnings
from collections import defaultdict

from isaaclab.app import AppLauncher

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tools import experiments as X  # noqa: E402

CONFIG = "AdaptiveSensing/configs/temporal/TMP02_chase-ob1ob2_adaptive-nav05.yaml"
p = argparse.ArgumentParser()
p.add_argument("--num_envs", type=int, required=True)
p.add_argument("--ticks", type=int, default=256, help="ticks per measured rollout (= num_steps)")
p.add_argument("--warmup", type=int, default=32)
p.add_argument("--sync_ticks", type=int, default=8, help="ticks with host-synchronization counting (C)")
p.add_argument("--skip_update", action="store_true")
p.add_argument("--detail", action="store_true", help="env internals: finer timers, host syncs by source line, torch.profiler kernel / launch counts (skips D and E)")
p.add_argument("--kernel_ticks", type=int, default=8)
p.add_argument("--tag", default=None)
p.add_argument("--fast_path", default="none", help="env fast path: none | all | comma-separated items")
p.add_argument("--out_dir", default="AdaptiveSensing/eval/temporal/TMP02_chase-ob1ob2_adaptive-nav05/profiling")
AppLauncher.add_app_launcher_args(p)
cli = p.parse_args()
app = AppLauncher(cli).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import torch.optim as optim  # noqa: E402

import st_nav.tasks.nav2d  # noqa: E402,F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from AdaptiveSensing.training.recurrent_ppo import compute_gae, minibatch_update  # noqa: E402
from AdaptiveSensing.training.sensing_agent import SensingAgent  # noqa: E402
from AdaptiveSensing.training.temporal_sensing import (TemporalSensing, load_nav_policy,  # noqa: E402
                                                       nav_contract_mismatches, nav_run_cfgs)

exp = X.load_config(os.path.join(X.REPO_ROOT, CONFIG))
env_cfg, _ = X.task_cfgs(exp["task"])
X.apply_overrides(env_cfg, exp["env"], "env")
from types import SimpleNamespace  # noqa: E402
agent_cfg = dict(exp["agent"])                                            # the TMP02 YAML lists every agent key
for k, v in exp["sweep"].items():                                         # first sweep member
    agent_cfg[k.partition(".")[2]] = v[0]
args = SimpleNamespace(**agent_cfg)
args.fps_choices = [float(x) for x in args.fps_choices]
N, T = cli.num_envs, cli.ticks
env_cfg.scene.num_envs = N
env_cfg.fast_path = () if cli.fast_path == "none" else tuple(x.strip() for x in cli.fast_path.split(","))
env_cfg.seed = args.seed
env_cfg.sim.device = cli.device
device = torch.device(cli.device)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic
nav_ck = os.path.join(X.REPO_ROOT, args.nav_checkpoint)
nav_agent_cfg, _, nav_env_cfg = nav_run_cfgs(nav_ck, device)
assert not nav_contract_mismatches(env_cfg, nav_env_cfg)
t0 = time.time()
genv = gym.make(exp["task"], cfg=env_cfg)
env = RslRlVecEnvWrapper(genv, clip_actions=nav_agent_cfg.clip_actions)
u = genv.unwrapped
nav = load_nav_policy(env, nav_agent_cfg, nav_ck, device)
sense = TemporalSensing(env, nav, args.fps_choices, args.frame_cost, args.budget, args.budget_penalty)
img_shape = tuple(env.get_observations()["lidar"].shape[1:])
agent = SensingAgent(img_shape, 3, len(args.fps_choices), args.lstm_hidden_size, tuple(args.cnn_channels),
                     args.cnn_kernel, args.cnn_stride, args.cnn_projection_dim).to(device)
optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
t_setup = time.time() - t0

f32 = dict(dtype=torch.float32, device=device)
buf = {"img": torch.zeros((T, N) + img_shape, **f32), "aux": torch.zeros((T, N, 3), **f32),
       "actions": torch.zeros((T, N), **f32), "logprobs": torch.zeros((T, N), **f32),
       "rewards": torch.zeros((T, N), **f32), "dones": torch.zeros((T, N), **f32),
       "values": torch.zeros((T, N), **f32), "masks": torch.zeros((T, N), **f32)}
EPK = ("return", "nav_return", "frame_cost_sum", "budget_penalty_sum", "length", "frames", "causal", "fps_inforce_sum",
       "fps_selected_sum", "fps_applied_sum", "age_ticks_sum", "budget_exceeded", "outcome")
ep_buf = {k: torch.zeros((T, N), **f32) for k in EPK}
ep_done = torch.zeros((T, N), dtype=torch.bool, device=device)


# ---------------------------------------------------------------------------------------------- timers
class Timers:
    def __init__(self):
        self.t, self.n, self.on = defaultdict(float), defaultdict(int), False

    def wrap(self, obj, attr, label):
        f = getattr(obj, attr)

        def w(*a, **k):
            if not self.on:
                return f(*a, **k)
            torch.cuda.synchronize()
            s = time.perf_counter()
            r = f(*a, **k)
            torch.cuda.synchronize()
            self.t[label] += time.perf_counter() - s
            self.n[label] += 1
            return r
        setattr(obj, attr, w)

    def block(self, label):
        tm = self

        class _B:
            def __enter__(self):
                if tm.on:
                    torch.cuda.synchronize()
                    self.s = time.perf_counter()

            def __exit__(self, *e):
                if tm.on:
                    torch.cuda.synchronize()
                    tm.t[label] += time.perf_counter() - self.s
                    tm.n[label] += 1
        return _B()


TM = Timers()
rc = u.corr.lidar.rc
for obj, attr, label in [
        (env, "step", "env.step (total)"),
        (u, "_pre_physics_step", "env: pre_physics (action -> diff-drive command)"),
        (u, "_apply_action", "env: apply_action x20 (obstacles + diff-drive + joint targets)"),
        (u.corr, "advance_obstacles", "env:   advance_obstacles x20"),
        (u.scene, "write_data_to_sim", "env: scene.write_data_to_sim x20"),
        (u.sim, "step", "env: sim.step x20 (PhysX)"),
        (u.scene, "update", "env: scene.update x20 (articulation / sensor buffers)"),
        (u, "_get_dones", "env: get_dones (clearance, goal)"),
        (u, "_get_rewards", "env: get_rewards"),
        (u, "_reset_idx", "env: reset_idx (episode resets)"),
        (u, "_get_observations", "env: get_observations (LiDAR read + features)"),
        (rc, "_update_buffers_impl", "env:   ray casting (MultiMeshRayCaster)"),
        (sense, "step", "sense.step (total)"),
        (sense.gate, "step", "gate: step"),
        (sense.gate, "set_rate", "gate: set_rate"),
        (sense.gate, "reset", "gate: reset"),
        (sense, "nav", "NAV05 inference"),
        (sense, "sensing_obs", "sensing obs build"),
        (agent, "get_action_and_value", "sensing agent (total)"),
        (agent.encoder, "forward", "sensing agent: CNN encoder")]:
    TM.wrap(obj, attr, label)
if cli.detail:                                                            # env internals (inclusive, nested in the above)
    import st_nav.crng as crng_mod
    c = u.corr
    det = [(c.obs_model, "advance", "obstacles: model advance x20"),
           (c, "_write_obstacle_poses", "obstacles: pose writes (both bodies) x20 (+resets)"),
           (c.cm, "step", "drive: command model step x20"),
           (c.robot, "set_joint_velocity_target", "drive: set_joint_velocity_target x20 (+resets)"),
           (c.robot, "write_data_to_sim", "write_data: robot articulation x20"),
           (c.robot, "update", "update: robot articulation x20"),
           (c.contact, "update", "update: contact sensor x20"),
           (c.contact, "_update_buffers_impl", "update:   contact sensor buffer fetch"),
           (rc, "update", "update: LiDAR ray caster x20 (lazy)"),
           (c, "reset", "reset: corr.reset"),
           (c.obs_model, "reset", "reset:   obstacle model reset"),
           (c.robot, "write_root_pose_to_sim", "reset:   robot root pose write"),
           (c.robot, "write_root_velocity_to_sim", "reset:   robot root velocity write"),
           (c.robot, "write_joint_state_to_sim", "reset:   robot joint state write"),
           (u.scene, "reset", "reset:   scene.reset"),
           (c.lidar, "read", "obs: lidar.read (ray cast + noise)"),
           (c.lidar, "noise", "obs:   range noise (crng Box-Muller)"),
           (c, "goal_distance", "dones: goal_distance"),
           (c, "footprint_clearance", "dones: footprint_clearance"),
           (crng_mod, "key", "crng.key (all callers)")]
    for j, ob in enumerate(c.obstacles):
        det += [(ob, "write_root_pose_to_sim", f"obstacles:   OB{j + 1} write_root_pose_to_sim x20 (+resets)"),
                (ob, "write_data_to_sim", f"write_data: OB{j + 1} rigid object x20"),
                (ob, "update", f"update: OB{j + 1} rigid object x20")]
    for obj, attr, label in det:
        TM.wrap(obj, attr, label)

state = {}


def reset_all():
    img, aux = sense.reset()
    state.update(img=img.clone(), aux=aux.clone(), done=torch.zeros(N, **f32), lstm=agent.initial_state(N, device))


def rollout(n_ticks, store=True):
    """The trainer's rollout loop (train_temporal_ppo.py), one tick per iteration."""
    for step in range(n_ticks):
        s = step % T
        with TM.block("buffer writes"):
            if store:
                buf["img"][s] = state["img"]
                buf["aux"][s] = state["aux"]
                buf["dones"][s] = state["done"]
        with torch.no_grad():
            action, logprob, _, value, state["lstm"] = agent.get_action_and_value(
                state["img"], state["aux"], state["lstm"], state["done"])
        with TM.block("buffer writes"):
            if store:
                buf["values"][s] = value.flatten()
                buf["actions"][s] = action.float()
                buf["logprobs"][s] = logprob
        (img, aux), reward, d, info = sense.step(action)
        state["img"], state["aux"] = img, aux
        with TM.block("buffer writes"):
            if store:
                buf["rewards"][s] = reward
                buf["masks"][s] = info["mask"]
                state["done"] = d.float()
                ep_done[s] = d
                for k in ep_buf:
                    ep_buf[k][s] = info["episode"][k]
            else:
                state["done"] = d.float()


out = {"num_envs": N, "fast_path": list(env_cfg.fast_path), "ticks": T, "device": cli.device, "gpu": torch.cuda.get_device_name(device),
       "cpu_count": os.cpu_count(), "setup_s": round(t_setup, 1)}
reset_all()
rollout(cli.warmup, store=False)                                          # warm-up (cuDNN autotune, caches)
# ---------------------------------------------------------------------------------------------- A. plain rollout
torch.cuda.synchronize()
w0, c0 = time.perf_counter(), time.process_time()
rollout(T)
torch.cuda.synchronize()
wall, cpu = time.perf_counter() - w0, time.process_time() - c0
out["A_plain"] = {"rollout_s": round(wall, 3), "ms_per_tick": round(1e3 * wall / T, 2),
                  "transitions_per_s": round(N * T / wall, 1), "process_cpu_over_wall": round(cpu / wall, 3),
                  "accepted_tick_fraction": round(float(buf["masks"].mean()), 4), "episodes_ended": int(ep_done.sum())}
print("[PROFILE] A", out["A_plain"], flush=True)
saved = {k: v.clone() for k, v in buf.items()}
# ---------------------------------------------------------------------------------------------- B. instrumented
TM.on = True
w0 = time.perf_counter()
rollout(T, store=True)
torch.cuda.synchronize()
wall_b = time.perf_counter() - w0
TM.on = False
out["B_instrumented"] = {"rollout_s": round(wall_b, 3), "ms_per_tick": round(1e3 * wall_b / T, 2),
                         "components_ms_per_tick (inclusive)": {k: round(1e3 * v / T, 3) for k, v in
                                                                 sorted(TM.t.items(), key=lambda kv: -kv[1])},
                         "calls_per_tick": {k: round(v / T, 2) for k, v in TM.n.items()}}
print("[PROFILE] B", json.dumps(out["B_instrumented"], indent=1), flush=True)
# ---------------------------------------------------------------------------------------------- C. host syncs
cnt = defaultdict(int)
lines = defaultdict(int)
torch.cuda.set_sync_debug_mode("warn")
with warnings.catch_warnings(record=True) as wl:
    warnings.simplefilter("always")
    rollout(cli.sync_ticks, store=False)
torch.cuda.set_sync_debug_mode(0)
for w in wl:
    if "synchroniz" in str(w.message):
        import linecache
        lines[f"{w.filename.split('site-packages/')[-1].split('IsaacLab_Fork/source/')[-1].split('spatiotemporal_sensing/')[-1]}:"
              f"{w.lineno}  {linecache.getline(w.filename, w.lineno).strip()[:90]}"] += 1
        cnt[os.path.relpath(w.filename, X.REPO_ROOT) if w.filename.startswith(str(X.REPO_ROOT)) else
            w.filename.split("site-packages/")[-1].split("source/")[-1]] += 1
out["C_host_syncs_per_tick"] = {k: round(v / cli.sync_ticks, 2) for k, v in sorted(cnt.items(), key=lambda kv: -kv[1])}
print("[PROFILE] C", out["C_host_syncs_per_tick"], flush=True)
if cli.detail:
    out["C_host_syncs_per_tick_by_line"] = {k: round(v / cli.sync_ticks, 2) for k, v in
                                            sorted(lines.items(), key=lambda kv: -kv[1])}
    print("[PROFILE] C-lines", json.dumps(out["C_host_syncs_per_tick_by_line"], indent=1), flush=True)
    # kernel-level view: GPU busy time vs wall, launches and copies per tick
    from torch.profiler import ProfilerActivity, profile
    torch.cuda.synchronize()
    w0 = time.perf_counter()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        rollout(cli.kernel_ticks, store=False)
        torch.cuda.synchronize()
    wk = time.perf_counter() - w0
    ev = [e for e in prof.events() if e.device_type.name == "CUDA"]
    kern = [e for e in ev if "memcpy" not in e.name.lower() and "memset" not in e.name.lower()]
    cp = [e for e in ev if "memcpy" in e.name.lower()]
    busy = sum(e.device_time for e in kern) / 1e3
    by = defaultdict(lambda: [0, 0.0])
    for e in kern:
        by[e.name[:80]][0] += 1
        by[e.name[:80]][1] += e.device_time / 1e3
    rt = defaultdict(int)
    for e in prof.events():
        if e.device_type.name == "CPU" and e.name in ("cudaLaunchKernel", "cuLaunchKernel", "cudaMemcpyAsync",
                                                     "cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaMemcpy"):
            rt[e.name] += 1
    out["F_kernels_per_tick"] = {
        "wall_ms_per_tick (profiler on)": round(1e3 * wk / cli.kernel_ticks, 2),
        "gpu_kernel_busy_ms_per_tick": round(busy / cli.kernel_ticks, 2),
        "kernels_per_tick": round(len(kern) / cli.kernel_ticks, 1),
        "memcpy_per_tick": round(len(cp) / cli.kernel_ticks, 1),
        "runtime_calls_per_tick": {k: round(v / cli.kernel_ticks, 1) for k, v in rt.items()},
        "top_kernels_by_time_ms_per_tick": {k: [round(v[0] / cli.kernel_ticks, 1), round(v[1] / cli.kernel_ticks, 3)]
                                            for k, v in sorted(by.items(), key=lambda kv: -kv[1][1])[:15]},
        "top_kernels_by_count_per_tick": {k: round(v[0] / cli.kernel_ticks, 1)
                                          for k, v in sorted(by.items(), key=lambda kv: -kv[1][0])[:15]}}
    print("[PROFILE] F", json.dumps(out["F_kernels_per_tick"], indent=1), flush=True)
# ---------------------------------------------------------------------------------------------- D. CNN reuse
if cli.detail:
    cli.skip_update = True
img, dn = saved["img"], saved["dones"]
with torch.no_grad():
    changed = torch.ones(T, N, dtype=torch.bool, device=device)
    changed[1:] = (img[1:] != img[:-1]).flatten(2).any(-1) | dn[1:].bool()
    frac = float(changed[1:].float().mean())
    enc = agent.encoder
    feats_full = torch.stack([enc(img[t]) for t in range(T)])
    feats_cache = torch.empty_like(feats_full)
    feats_cache[0] = enc(img[0])
    for t in range(1, T):
        f = feats_cache[t - 1].clone()
        m = changed[t]
        if bool(m.any()):
            f[m] = enc(img[t][m])
        feats_cache[t] = f
    bitwise = bool(torch.equal(feats_full, feats_cache))
    max_diff = float((feats_full - feats_cache).abs().max())

    def policy_from_feats(feats):
        lstm = agent.initial_state(N, device)
        logits_all = []
        for t in range(T):
            h = agent.network(torch.cat([feats[t], saved["aux"][t]], -1))
            d = dn[t]
            h, lstm = agent.lstm(h.unsqueeze(0), ((1 - d).view(1, -1, 1) * lstm[0], (1 - d).view(1, -1, 1) * lstm[1]))
            logits_all.append(agent.actor(h[0]))
        return torch.stack(logits_all)
    lf, lc = policy_from_feats(feats_full), policy_from_feats(feats_cache)
    lpf, lpc = torch.log_softmax(lf, -1), torch.log_softmax(lc, -1)

    def timed(fn, reps=3):
        torch.cuda.synchronize()
        s = time.perf_counter()
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - s) / reps

    t_full = timed(lambda: [enc(img[t]) for t in range(T)])

    def cached_pass():
        f = enc(img[0])
        for t in range(1, T):
            m = changed[t]
            idx = m.nonzero().squeeze(1)
            if idx.numel():
                f = f.index_copy(0, idx, enc(img[t].index_select(0, idx)))
    t_cache = timed(cached_pass)
out["D_cnn_reuse"] = {
    "changed_fraction (held image changed or reset)": round(frac, 4),
    "encoder_all_images_ms_per_tick": round(1e3 * t_full / T, 3),
    "encoder_changed_only_ms_per_tick (incl. index ops)": round(1e3 * t_cache / T, 3),
    "features_bitwise_equal": bitwise, "features_max_abs_diff": max_diff,
    "logits_max_abs_diff": float((lf - lc).abs().max()), "logprob_max_abs_diff": float((lpf - lpc).abs().max()),
    "argmax_action_mismatches": int((lf.argmax(-1) != lc.argmax(-1)).sum()),
    "note": "rollout policy only (no_grad); the update recomputes features from the stored raw images either way"}
print("[PROFILE] D", out["D_cnn_reuse"], flush=True)
# ---------------------------------------------------------------------------------------------- E. update
if not cli.skip_update:
    buf.update(saved)
    with torch.no_grad():
        nv = agent.get_value(state["img"], state["aux"], state["lstm"], state["done"]).reshape(1, -1)
        buf["advantages"], buf["returns"] = compute_gae(buf["rewards"], buf["values"], buf["dones"], nv,
                                                        state["done"], args.gamma, args.gae_lambda)
    epm = N // args.num_minibatches
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    s = time.perf_counter()
    mb_times = []
    for epoch in range(args.update_epochs):
        perm = torch.randperm(N, device=device)
        for s0 in range(0, N, epm):
            torch.cuda.synchronize()
            m0 = time.perf_counter()
            minibatch_update(agent, optimizer, buf, perm[s0:s0 + epm], (state["lstm"][0] * 0, state["lstm"][1] * 0),
                             args, chunk_envs=args.update_chunk_envs)
            torch.cuda.synchronize()
            mb_times.append(time.perf_counter() - m0)
    t_up = time.perf_counter() - s
    out["E_update"] = {"update_s": round(t_up, 3), "minibatch_s_mean": round(sum(mb_times) / len(mb_times), 3),
                       "minibatches": len(mb_times), "envs_per_minibatch": epm,
                       "chunk_envs": args.update_chunk_envs,
                       "peak_mem_gb": round(torch.cuda.max_memory_allocated(device) / 1e9, 2)}
    print("[PROFILE] E", out["E_update"], flush=True)
odir = cli.out_dir if os.path.isabs(cli.out_dir) else os.path.join(X.REPO_ROOT, cli.out_dir)
os.makedirs(odir, exist_ok=True)
out["command"] = sys.argv
out["repo"] = X.git_state(X.REPO_ROOT)
name = f"profile_n{N}" + (f"_{cli.tag}" if cli.tag else "") + ".json"
json.dump(out, open(os.path.join(odir, name), "w"), indent=1)
print(f"[PROFILE] written {os.path.join(odir, name)}", flush=True)
os._exit(0)
