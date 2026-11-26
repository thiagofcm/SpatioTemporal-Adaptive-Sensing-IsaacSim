"""Env fast path: bitwise equivalence of whole-environment traces (st_nav.envs.corridor_scene.FAST_PATH_ITEMS).

Record: the TMP02 training env (scene / DR from the TMP02 config, N envs, auto-resets) driven by the frozen NAV05
policy through the TMP02 sensing wrapper with seeded random sensing actions, for K control ticks. Saved per tick:
robot root pose / velocity and wheel joint velocities, obstacle bodies as Isaac Lab holds them and the obstacle model
positions, an exact digest of every LiDAR image (int32 view summed in int64, per env), navigation features, rewards,
dones, outcomes, sensing masks, timings.

    bin/isaacrun -g 0 -- python tests/sim/fast_path_equivalence.py --headless --fast_path none --out <a.pt>
    bin/isaacrun -g 0 -- python tests/sim/fast_path_equivalence.py --headless --fast_path all  --out <b.pt>
    python tests/sim/fast_path_equivalence.py --compare <a.pt> <b.pt>        (no simulator; exit 1 if different)
"""
import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
CONFIG = "AdaptiveSensing/configs/temporal/TMP02_chase-ob1ob2_adaptive-nav05.yaml"
FIELDS = ("robot_pose", "robot_vel", "wheel_vel", "ob_body_pose", "ob_model_pos", "lidar_digest", "nav_features",
          "reward", "done", "outcome", "mask", "scan_t")


def compare(a_path, b_path):
    import torch
    a, b = torch.load(a_path), torch.load(b_path)
    print(f"A: {a['meta']}\nB: {b['meta']}")
    ok = True
    for f in FIELDS:
        x, y = a[f], b[f]
        if x.shape != y.shape:
            print(f"  {f}: shape {tuple(x.shape)} vs {tuple(y.shape)}")
            ok = False
            continue
        same = (x == y) | (torch.isnan(x) & torch.isnan(y)) if x.is_floating_point() else (x == y)
        bad = ~same.reshape(x.shape[0], -1).all(1)
        if bool(bad.any()):
            t = int(bad.nonzero()[0])
            print(f"  {f}: DIFFERENT from tick {t} ({int(bad.sum())} ticks), max |diff| "
                  f"{float((x.double() - y.double()).abs().nan_to_num().max()):.3e}")
            ok = False
        else:
            print(f"  {f}: identical ({tuple(x.shape)})")
    print("EQUIVALENT (bitwise)" if ok else "NOT EQUIVALENT")
    return ok


if "--compare" in sys.argv:
    i = sys.argv.index("--compare")
    sys.exit(0 if compare(sys.argv[i + 1], sys.argv[i + 2]) else 1)

from isaaclab.app import AppLauncher  # noqa: E402

from tools import experiments as X  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--fast_path", default="none", help="none | all | comma-separated FAST_PATH_ITEMS")
p.add_argument("--num_envs", type=int, default=512)
p.add_argument("--ticks", type=int, default=800)
p.add_argument("--seed", type=int, default=1)
p.add_argument("--out", required=True)
AppLauncher.add_app_launcher_args(p)
cli = p.parse_args()
app = AppLauncher(cli).app

import time  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import st_nav.tasks.nav2d  # noqa: E402,F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from AdaptiveSensing.training.temporal_sensing import TemporalSensing, load_nav_policy, nav_run_cfgs  # noqa: E402

fp = () if cli.fast_path == "none" else tuple(x.strip() for x in cli.fast_path.split(","))
exp = X.load_config(os.path.join(ROOT, CONFIG))
cfg, _ = X.task_cfgs(exp["task"])
X.apply_overrides(cfg, exp["env"], "env")
cfg.scene.num_envs, cfg.seed, cfg.sim.device, cfg.fast_path = cli.num_envs, cli.seed, cli.device, fp
dev = torch.device(cli.device)
nav_ck = os.path.join(ROOT, exp["agent"]["nav_checkpoint"])
nav_cfg, _, _ = nav_run_cfgs(nav_ck, dev)
genv = gym.make(exp["task"], cfg=cfg)
env = RslRlVecEnvWrapper(genv, clip_actions=nav_cfg.clip_actions)
u = genv.unwrapped
nav = load_nav_policy(env, nav_cfg, nav_ck, dev)
sense = TemporalSensing(env, nav, [float(f) for f in exp["agent"]["fps_choices"]], 0.03, 500, 15.0)
corr = u.corr
gen = torch.Generator(device=dev).manual_seed(1234)
rec = {f: [] for f in FIELDS}
sense.reset()
torch.cuda.synchronize()
t0 = time.perf_counter()
for k in range(cli.ticks):
    a = torch.randint(0, 5, (cli.num_envs,), device=dev, generator=gen)
    _, r, d, info = sense.step(a)
    rd = corr.robot.data
    rec["robot_pose"].append(torch.cat([rd.root_pos_w, rd.root_quat_w], 1).cpu())
    rec["robot_vel"].append(torch.cat([rd.root_lin_vel_w, rd.root_ang_vel_w], 1).cpu())
    rec["wheel_vel"].append(rd.joint_vel[:, corr.wheel_ids].cpu())
    rec["ob_body_pose"].append(torch.stack([ob.data.root_pos_w for ob in corr.obstacles], 1).cpu())
    rec["ob_model_pos"].append(corr.obs_model.positions().cpu())
    lid = sense.obs["lidar"].contiguous()
    rec["lidar_digest"].append(lid.view(torch.int32).reshape(lid.shape[0], -1).long().sum(1).cpu())
    rec["nav_features"].append(sense.obs["policy"].cpu())
    rec["reward"].append(r.cpu())
    rec["done"].append(d.cpu())
    rec["outcome"].append(info["episode"]["outcome"].cpu())
    rec["mask"].append(info["mask"].cpu())
    rec["scan_t"].append(u.scan_t.cpu())
torch.cuda.synchronize()
wall = time.perf_counter() - t0
out = {f: torch.stack(v) for f, v in rec.items()}
out["meta"] = {"fast_path": list(fp), "num_envs": cli.num_envs, "ticks": cli.ticks, "seed": cli.seed,
               "loop_s": round(wall, 2), "ms_per_tick_incl_recording": round(1e3 * wall / cli.ticks, 2),
               "episodes_ended": int(out["done"].sum()), "gpu": torch.cuda.get_device_name(dev)}
os.makedirs(os.path.dirname(os.path.abspath(cli.out)), exist_ok=True)
torch.save(out, cli.out)
print("[FASTPATH-TRACE]", out["meta"], flush=True)
os._exit(0)
