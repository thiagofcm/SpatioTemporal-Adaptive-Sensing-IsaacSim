"""TMP02 simulator validation of the temporal sensing wrapper (AdaptiveSensing/training/temporal_sensing.py) in the real
Isaac Lab env with the frozen NAV05 policy, on the TMP01 evaluation protocol (ref_chase_c10: 100 envs, same start
poses, goal, seeds, 70 s).

  --fixed <hz>   constant action = <hz>, gate reset rate = <hz>: every first episode must equal the TMP01 recording
                 (outcome, steps, return, accepted / native scans, observation age) -> same navigation, same gate
  --adaptive     uniformly random sensing actions over several episodes per env: every tick checked against a
                 literal transcription of the reference / real-robot rule (accept, re-anchor, discard), masks, frame
                 accounting, reward decomposition, terminal ticks and reset attribution

    bin/isaacrun -g 0 -- python tests/sim/tmp02_validation.py --headless --fixed 2
    bin/isaacrun -g 0 -- python tests/sim/tmp02_validation.py --headless --adaptive
Results: AdaptiveSensing/eval/temporal/TMP02_chase-ob1ob2_adaptive-nav05/validation/tmp02_validation_<mode>[_<tag>].json
(--out_dir to write elsewhere; --tag to keep repeated runs apart, e.g. run2_gpu7)
"""
import argparse
import csv
import json
import math
import os
import sys

from isaaclab.app import AppLauncher

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from tools import experiments as X  # noqa: E402

NAV05 = ("Navigation/runs_3d/NAV05_chase-ob1ob2_lidar3d-r4_cnn-from-nav04/2026-11-23_20-21-14_ppo_cnn_n4096_s42/"
         "model_1499.pt")
TMP01 = "AdaptiveSensing/eval/temporal/TMP01_chase-ob1ob2_fixed-rate-nav05/ref_chase_c10"
FPS = [0.2, 0.5, 1.0, 2.0, 4.0]
p = argparse.ArgumentParser()
g = p.add_mutually_exclusive_group(required=True)
g.add_argument("--fixed", type=float)
g.add_argument("--adaptive", action="store_true")
p.add_argument("--ticks", type=int, default=1200, help="--adaptive: control ticks (several episodes per env)")
p.add_argument("--frame_cost", type=float, default=0.03)
p.add_argument("--fast_path", default="none", help="env performance fast path: none | all | comma-separated items (st_nav.envs.corridor_scene.FAST_PATH_ITEMS); same results, faster")
p.add_argument("--out_dir", default="AdaptiveSensing/eval/temporal/TMP02_chase-ob1ob2_adaptive-nav05/validation")
p.add_argument("--tag", default=None, help="suffix of the result file name")
p.add_argument("--compare_dir", default=None, help="--fixed: evaluate_fixed_rate.py output to compare with (default: the stored TMP01 recording)")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import st_nav.tasks.nav2d  # noqa: E402,F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from st_nav.scenes import corridor_static_random as S  # noqa: E402
from st_nav.tasks.nav2d.corridor_env import OUTCOME_NAMES  # noqa: E402
from AdaptiveSensing.training.temporal_sensing import TemporalSensing, load_nav_policy, nav_run_cfgs  # noqa: E402

dev = args.device
ck = os.path.join(ROOT, NAV05)
nav_agent_cfg, run_exp, nav_env_cfg = nav_run_cfgs(ck, dev)


def reference_start_pose(seed):                    # = evaluate_3d.py / evaluate_fixed_rate.py
    rng = np.random.default_rng(seed)
    i = int(rng.integers(len(S.START_POSE_LIST)))
    dyaw = float(rng.uniform(-S.START_YAW_NOISE, S.START_YAW_NOISE))
    return i, (*S.START_POSE_LIST[i], (S.START_YAW_CENTER + dyaw + math.pi) % (2 * math.pi) - math.pi)


n = 100                                           # ref_chase_c10
cfg = nav_env_cfg
cfg.scene.num_envs, cfg.sim.device, cfg.seed, cfg.episode_length_s = n, dev, 42, 70.0
cfg.start_poses = tuple(reference_start_pose(4000 + i)[1] for i in range(n))
cfg.goals = ((-2.0, -2.0),)
cfg.fast_path = () if args.fast_path == "none" else tuple(x.strip() for x in args.fast_path.split(","))
genv = gym.make("STNav-TB4-Corridor-Lidar3D-v0", cfg=cfg)
env = RslRlVecEnvWrapper(genv, clip_actions=nav_agent_cfg.clip_actions)
nav = load_nav_policy(env, nav_agent_cfg, ck, dev)
u = genv.unwrapped
nav_before = {k: v.clone() for k, v in nav.state_dict().items()}
out = {"mode": "fixed" if args.fixed else "adaptive", "checks": {}}
C = out["checks"]

if args.fixed:
    hz = args.fixed
    ts = TemporalSensing(env, nav, FPS, 0.0, 500, 15.0, reset_hz=hz)
    a = torch.full((n,), FPS.index(hz), device=dev)
    ts.reset()
    first = {k: None for k in ("outcome", "length", "nav_return", "frames", "native_scans", "age_ticks_sum",
                               "age_ticks_max", "causal")}
    res = {k: torch.zeros(n, device=dev) for k in first}
    seen = torch.zeros(n, dtype=torch.bool, device=dev)
    for k in range(400):
        _, r, d, info = ts.step(a)
        new_end = d & ~seen
        for key in res:
            res[key] = torch.where(new_end, info["episode"][key].float(), res[key])
        seen |= d
        if bool(seen.all()):
            break
    R = {k: v.cpu().numpy() for k, v in res.items()}
    tdir = [args.compare_dir] if args.compare_dir else [
        os.path.join(ROOT, TMP01, x) for x in os.listdir(os.path.join(ROOT, TMP01)) if x.endswith(f"_fixed{hz:g}Hz")]
    rows = list(csv.DictReader(open(os.path.join(tdir[0], "episodes.csv"))))
    mism = {"outcome": 0, "n_steps": 0, "episode_return": 0, "accepted_scans": 0, "native_scans": 0,
            "mean_obs_age_s": 0, "max_obs_age_s": 0}
    for i, row in enumerate(rows):
        L = R["length"][i]
        mism["outcome"] += OUTCOME_NAMES.get(int(R["outcome"][i]), "running") != row["outcome"]
        mism["n_steps"] += int(L) != int(row["n_steps"])
        mism["episode_return"] += abs(round(float(R["nav_return"][i]), 3) - float(row["episode_return"])) > 2e-3
        mism["accepted_scans"] += int(R["frames"][i]) != int(row["accepted_scans"])
        mism["native_scans"] += int(R["native_scans"][i]) != int(row["native_scans"])
        mism["mean_obs_age_s"] += abs(round(R["age_ticks_sum"][i] * u.physics_dt / L, 4) - float(row["mean_obs_age_s"])) > 1e-4
        mism["max_obs_age_s"] += abs(round(R["age_ticks_max"][i] * u.physics_dt, 4) - float(row["max_obs_age_s"])) > 1e-4
    succ = float(np.mean([OUTCOME_NAMES.get(int(o), "") == "goal" for o in R["outcome"]]))
    tmp01_succ = float(np.mean([r_["success"] == "True" for r_ in rows]))
    C.update(hz=hz, tmp01_dir=os.path.relpath(tdir[0], ROOT), episodes=n, mismatches_vs_tmp01=mism,
             success=succ, success_tmp01=tmp01_succ, accepted_total=int(R["frames"].sum()),
             causal_total=int(R["causal"].sum()))
    ok = all(v == 0 for v in mism.values())
else:
    gen = torch.Generator(device=dev).manual_seed(0)
    fc, P = args.frame_cost, [500, 200, 100, 50, 25]
    ts = TemporalSensing(env, nav, FPS, fc, 500, 15.0)
    img, aux = ts.reset()
    log = {k: [] for k in ("now", "native", "a", "mask", "period", "r", "r_env", "d", "held", "aux", "nav_held",
                           "ep_frames", "ep_causal", "ep_len", "ep_return")}
    for k in range(args.ticks):
        now = ts.gate.ticks(u.ep_time).clone()
        native = ts.gate.ticks(u.scan_t).clone()
        held_t = ts.gate.last_forward.clone()
        a = torch.randint(0, 5, (n,), device=dev, generator=gen)
        log["aux"].append(aux.clone())
        (img, aux), r, d, info = ts.step(a)
        for key, v in (("now", now), ("native", native), ("a", a), ("mask", info["mask"]), ("r", r),
                       ("r_env", info["r_env"]), ("d", d), ("held", held_t), ("nav_held", ts.gate.last_forward),
                       ("ep_frames", info["episode"]["frames"]), ("ep_causal", info["episode"]["causal"]),
                       ("ep_len", info["episode"]["length"]), ("ep_return", info["episode"]["return"])):
            log[key].append(v.clone())
        log["period"].append(torch.where(d, torch.full_like(ts.gate.periods, -1), ts.gate.periods))
    L = {k: torch.stack(v).cpu().numpy() for k, v in log.items()}
    bad = {"rule": 0, "native_interval": 0, "reward": 0, "frames": 0, "causal": 0, "return": 0, "reset_obs": 0,
           "age_ratio": 0, "frame_ratio": 0, "fps_ratio": 0, "mask_without_native": 0}
    n_ep = n_term_fwd = n_ticks = n_mask = 0
    intervals = set()
    for i in range(n):
        start = 0
        for k in range(args.ticks):
            if not (L["d"][k, i] or k == args.ticks - 1):
                continue
            seg = range(start, k + 1)
            period, nxt, last_fwd, prev_native, since, frames = 25, int(L["now"][start, i]) + 25, int(L["now"][start, i]), int(L["native"][start, i]), 0, 1
            ret = 0.0
            for j in seg:
                nat, now_, a_ = int(L["native"][j, i]), int(L["now"][j, i]), int(L["a"][j, i])
                fps_r, age_r, fr_r = L["aux"][j][i]
                bad["fps_ratio"] += abs(fps_r - (1.0 / (period * 0.01)) / 4.0) > 1e-5
                bad["age_ratio"] += abs(age_r - min(since / 25.0, 1.0)) > 1e-6
                bad["frame_ratio"] += abs(fr_r - min(frames / 500.0, 1.0)) > 1e-6
                new = nat != prev_native
                if new and j > start:
                    intervals.add(nat - prev_native)
                    bad["native_interval"] += (nat - prev_native) not in (20, 40)
                prev_native = nat if new else prev_native
                since += 1
                fwd = new and nat >= nxt
                bad["mask_without_native"] += bool(L["mask"][j, i]) and nat != now_
                if fwd:
                    last_fwd = nat
                    while nxt <= nat:
                        nxt += period
                    if P[a_] != period:
                        period, nxt = P[a_], last_fwd + P[a_]
                    since, frames = 0, frames + 1
                bad["rule"] += bool(L["mask"][j, i]) != fwd
                if j < k or not L["d"][k, i]:
                    bad["rule"] += int(L["period"][j, i]) != period
                bad["reward"] += abs(L["r"][j, i] - (L["r_env"][j, i] - fc * fwd)) > 1e-5
                ret += float(L["r"][j, i])
                n_ticks += 1
                n_mask += fwd
            if L["d"][k, i]:
                n_ep += 1
                n_term_fwd += bool(L["mask"][k, i])
                bad["frames"] += int(L["ep_frames"][k, i]) != frames
                bad["causal"] += int(L["ep_causal"][k, i]) != frames - 1
                bad["return"] += abs(L["ep_return"][k, i] - ret) > 1e-2
                if k + 1 < args.ticks:                    # next tick: reset scan of the new episode, 4 Hz, frame 1
                    fps_r, age_r, fr_r = L["aux"][k + 1][i]
                    bad["reset_obs"] += not (int(L["now"][k + 1, i]) == 0 and int(L["held"][k + 1, i]) == 0
                                             and abs(fps_r - 1.0) < 1e-6 and age_r == 0.0
                                             and abs(fr_r - 1 / 500) < 1e-6 and not L["mask"][k + 1, i])
            start = k + 1
    C.update(ticks=n_ticks, episodes_completed=n_ep, accepted=int(n_mask), causal_fraction=n_mask / n_ticks,
             terminal_ticks_with_accepted_scan=n_term_fwd, native_intervals_ticks=sorted(intervals), violations=bad,
             applied_action_hist=np.bincount(L["a"][L["mask"].astype(bool)].astype(int), minlength=5).tolist())
    ok = all(v == 0 for v in bad.values()) and n_ep > 0
C["nav05_state_unchanged"] = all(torch.equal(v, nav_before[k]) for k, v in nav.state_dict().items())
out["pass"] = bool(ok and C["nav05_state_unchanged"])
odir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(ROOT, args.out_dir)
os.makedirs(odir, exist_ok=True)
tag = (f"fixed{args.fixed:g}Hz" + ("_vs_" + os.path.basename(os.path.normpath(args.compare_dir)) if args.compare_dir else "")
       if args.fixed else "adaptive")
J = lambda o: o.item() if hasattr(o, "item") else str(o)  # noqa: E731  (numpy / torch scalars)
tag += f"_{args.tag}" if args.tag else ""
out["command"] = sys.argv
out["repo"] = X.git_state(X.REPO_ROOT)
json.dump(out, open(os.path.join(odir, f"tmp02_validation_{tag}.json"), "w"), indent=1, default=J)
print("[TMP02-VALIDATION]", json.dumps(out, default=J), flush=True)
os._exit(0)
