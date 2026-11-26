"""Temporal adaptive sensing: recurrent PPO (LSTM, causal-tick masking) of the sensing-rate policy over the frozen
NAV05 3D-LiDAR navigation policy in Isaac Lab (TMP02).

Port of the reference temporal trainer (adaptive_sensor_policy@dc5b53f AdaptiveFPS/scripts/train_adaptive_fps_ppo.py:
rollout 570-631, GAE 647, recurrent masked update 677-748, logging 632 / 777, checkpoints 751) onto GPU tensors with
the Isaac Lab vectorized env. Only the sensing policy is trained; NAV05 is frozen inside the sensing wrapper
(AdaptiveSensing/training/temporal_sensing.py: timing, gate, reward). Differences from the reference: CNN encoder in
front of the reference network (sensing_agent.py), gradient accumulation over env chunks (recurrent_ppo.py), Isaac
Lab env and its NAV05 reward, rates {0.2, 0.5, 1, 2, 4} Hz with native 4 Hz LiDAR.

    bin/isaacrun -g <gpu> -- python AdaptiveSensing/scripts/train_temporal_ppo.py --headless \
        --config AdaptiveSensing/configs/temporal/TMP02_chase-ob1ob2_adaptive-nav05.yaml --sweep_index 0 \
        [--set agent.num_iterations=3 --set env.scene.num_envs=64 ...] [--run_tag smoke]

A config with a sweep runs one member per process (--sweep_index i: i-th combination of the sweep lists, in YAML
order, last key fastest). Output: AdaptiveSensing/runs/temporal/<ID>/<date>_<run_name>[_<tag>]/ (TensorBoard events,
checkpoints/, params/).
"""
import argparse
import itertools
import json
import os
import socket
import sys
import time
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime

from isaaclab.app import AppLauncher

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tools import experiments as X  # noqa: E402


@dataclass
class Args:
    """Sensing-policy and recurrent-PPO settings (`agent:` section; names as the reference trainer's Args)."""
    nav_checkpoint: str = ("Navigation/runs_3d/NAV05_chase-ob1ob2_lidar3d-r4_cnn-from-nav04/2026-11-23_20-21-14_ppo_cnn_"
                           "n4096_s42/model_1499.pt")
    fps_choices: list = field(default_factory=lambda: [0.2, 0.5, 1.0, 2.0, 4.0])
    frame_cost: float = 0.0
    budget: int = 500
    budget_penalty: float = 15.0
    seed: int = 1
    torch_deterministic: bool = True
    total_timesteps: int = 100_000_000
    num_iterations: int = 0                 # 0: total_timesteps // batch_size (reference); > 0 overrides (smoke tests)
    num_steps: int = 256
    learning_rate: float = 3.0e-4
    anneal_lr: bool = False
    gamma: float = 0.99
    gae_lambda: float = 0.98
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.05
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = None
    lstm_hidden_size: int = 64
    cnn_channels: list = field(default_factory=lambda: [16, 32, 64])
    cnn_kernel: int = 3
    cnn_stride: int = 2
    cnn_projection_dim: int = 128
    update_chunk_envs: int = 64             # envs per gradient-accumulation chunk (memory only; same update)
    checkpoint_interval: int = 25           # iterations


def make_args(d: dict) -> Args:
    known = {f.name for f in fields(Args)}
    bad = sorted(set(d) - known)
    if bad:
        raise SystemExit(f"agent: unknown keys {bad} (known: {sorted(known)})")
    a = Args(**d)
    a.fps_choices = [float(x) for x in a.fps_choices]
    return a


p = argparse.ArgumentParser()
p.add_argument("--config", required=True)
p.add_argument("--sweep_index", type=int, default=None)
p.add_argument("--set", action="append", default=[], metavar="SECTION.PATH=VALUE")
p.add_argument("--run_tag", default=None)
p.add_argument("--out_root", default=None)
AppLauncher.add_app_launcher_args(p)
cli = p.parse_args()
exp = X.load_config(cli.config)
sweep_keys = list(exp["sweep"])
members = list(itertools.product(*(exp["sweep"][k] for k in sweep_keys))) if sweep_keys else [()]
if sweep_keys and cli.sweep_index is None:
    raise SystemExit("this config has a sweep; choose one member with --sweep_index:\n" + "\n".join(
        f"  {i}: {dict(zip(sweep_keys, m))}" for i, m in enumerate(members)))
if cli.sweep_index is not None and not 0 <= cli.sweep_index < len(members):
    raise SystemExit(f"--sweep_index must be in [0, {len(members) - 1}]")
member = dict(zip(sweep_keys, members[cli.sweep_index or 0])) if sweep_keys else {}
sets = X.parse_set(cli.set)
for k, v in member.items():                          # sweep value, then --set (highest precedence)
    sec, _, sub = k.partition(".")
    if sec not in X.SECTIONS:
        raise SystemExit(f"sweep key {k}: expected <env|agent>.<path>")
    sets[sec] = {sub: v, **sets[sec]}
app = AppLauncher(cli).app

from importlib.metadata import version  # noqa: E402

import gymnasium as gym  # noqa: E402
import hashlib  # noqa: E402
import numpy as np  # noqa: E402
import random  # noqa: E402
import torch  # noqa: E402
import torch.optim as optim  # noqa: E402
from torch.utils.tensorboard import SummaryWriter  # noqa: E402

import st_nav.tasks.nav2d  # noqa: E402,F401
from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from st_nav.tasks.nav2d.corridor_env import OUTCOME_NAMES  # noqa: E402
from AdaptiveSensing.training.recurrent_ppo import compute_gae, minibatch_update  # noqa: E402
from AdaptiveSensing.training.sensing_agent import SensingAgent  # noqa: E402
from AdaptiveSensing.training.temporal_sensing import (TemporalSensing, load_nav_policy,  # noqa: E402
                                                       nav_contract_mismatches, nav_run_cfgs)

exp_id, task = exp["exp_id"], exp["task"]
if not exp_id.startswith("TMP"):
    raise SystemExit("train_temporal_ppo.py trains temporal sensing experiments (TMP<NN>_...)")
env_cfg, _ = X.task_cfgs(task)
X.apply_overrides(env_cfg, exp["env"], "env")
X.apply_overrides(env_cfg, sets["env"], "env")
args = make_args({**exp["agent"], **sets["agent"]})
if env_cfg.sensor != "lidar3d" or env_cfg.obs_layout != "image":
    raise SystemExit("expects env.sensor = lidar3d and env.obs_layout = image")
if abs(env_cfg.lidar3d.period_s - 0.25) > 1e-9:
    raise SystemExit(f"native LiDAR period is {env_cfg.lidar3d.period_s} s; TMP02 keeps it at 0.25 s (4 Hz)")
env_cfg.seed = args.seed
env_cfg.sim.device = cli.device
N, T = env_cfg.scene.num_envs, args.num_steps
batch_size = N * T
num_iterations = args.num_iterations or args.total_timesteps // batch_size
if N % args.num_minibatches:
    raise SystemExit("num_envs must be divisible by num_minibatches")
nav_ck = args.nav_checkpoint if os.path.isabs(args.nav_checkpoint) else os.path.join(X.REPO_ROOT, args.nav_checkpoint)
run_name = exp.get("run_name") or f"ppo_lstm_n{N}_s{args.seed}_fc{args.frame_cost:g}"
run_name += f"_{cli.run_tag}" if cli.run_tag else ""
root = cli.out_root or str(X.paths(exp_id)["runs"])
log_dir = os.path.join(root, f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{run_name}")
print(f"[TMP-TRAIN] {exp_id} {member or ''} -> {log_dir}", flush=True)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic
device = torch.device(cli.device)
t0 = time.time()
nav_agent_cfg, nav_run_exp, nav_env_cfg = nav_run_cfgs(nav_ck, device)
bad = nav_contract_mismatches(env_cfg, nav_env_cfg)
if bad:
    raise SystemExit(f"env settings that define NAV05's input differ from its training run: {bad}")
genv = gym.make(task, cfg=env_cfg)
env = RslRlVecEnvWrapper(genv, clip_actions=nav_agent_cfg.clip_actions)
u = genv.unwrapped
if abs(u.step_dt - 0.2) > 1e-9 or abs(u.scan_period - 0.25) > 1e-9:
    raise SystemExit(f"unexpected timing: control {u.step_dt} s, native LiDAR {u.scan_period} s")
nav = load_nav_policy(env, nav_agent_cfg, nav_ck, device)
nav_params_before = {k: v.detach().clone() for k, v in nav.state_dict().items()}
sense = TemporalSensing(env, nav, args.fps_choices, args.frame_cost, args.budget, args.budget_penalty)
img_shape = tuple(env.get_observations()["lidar"].shape[1:])
agent = SensingAgent(img_shape, 3, len(args.fps_choices), args.lstm_hidden_size, tuple(args.cnn_channels),
                     args.cnn_kernel, args.cnn_stride, args.cnn_projection_dim).to(device)
optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
nav_ids = {id(q) for q in nav.parameters()}
assert not any(id(q) in nav_ids for g in optimizer.param_groups for q in g["params"]), "NAV05 in the optimizer"

# ---- run folder
os.makedirs(os.path.join(log_dir, "checkpoints"), exist_ok=True)
dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
with open(nav_ck, "rb") as fh:
    nav_sha = hashlib.sha256(fh.read()).hexdigest()
with open(os.path.join(log_dir, "params", "config.yaml"), "w") as f:
    X.yaml.safe_dump({"source": os.path.relpath(os.path.abspath(cli.config), X.REPO_ROOT), **exp,
                      "sweep_member": {"index": cli.sweep_index, "values": member} if member else None,
                      "set": {s: v for s, v in X.parse_set(cli.set).items() if v},
                      "agent_resolved": asdict(args)}, f, sort_keys=False)
with open(os.path.join(log_dir, "params", "model.txt"), "w") as f:
    f.write(f"sensing agent: {sum(q.numel() for q in agent.parameters())} trainable parameters\n{agent}\n")
with open(os.path.join(log_dir, "params", "run_info.json"), "w") as f:
    json.dump({"exp_id": exp_id, "task": task, "run_name": run_name, "command": sys.argv, "host": socket.gethostname(),
               "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "repo": X.git_state(X.REPO_ROOT),
               "nav_checkpoint": {"path": args.nav_checkpoint, "sha256": nav_sha},
               "num_envs": N, "num_steps": T, "batch_size": batch_size, "num_iterations": num_iterations,
               "versions": {k: version(k) for k in ("rsl-rl-lib", "torch", "isaaclab")}}, f, indent=1)
writer = SummaryWriter(log_dir)
writer.add_text("hyperparameters", "|param|value|\n|-|-|\n" + "\n".join(f"|{k}|{v}|" for k, v in asdict(args).items()))

# ---- storage (GPU): images of the held scan are stored raw; the CNN is re-run with gradients in the update
f32 = dict(dtype=torch.float32, device=device)
buf = {"img": torch.zeros((T, N) + img_shape, **f32), "aux": torch.zeros((T, N, 3), **f32),
       "actions": torch.zeros((T, N), **f32), "logprobs": torch.zeros((T, N), **f32),
       "rewards": torch.zeros((T, N), **f32), "dones": torch.zeros((T, N), **f32),
       "values": torch.zeros((T, N), **f32), "masks": torch.zeros((T, N), **f32)}
ep_buf = {k: torch.zeros((T, N), **f32) for k in ("return", "nav_return", "frame_cost_sum", "budget_penalty_sum",
                                                  "length", "frames", "causal", "fps_inforce_sum", "fps_selected_sum",
                                                  "fps_applied_sum", "age_ticks_sum", "budget_exceeded", "outcome")}
ep_done = torch.zeros((T, N), dtype=torch.bool, device=device)
print(f"[TMP-TRAIN] setup {time.time() - t0:.0f} s; {N} envs x {T} steps = {batch_size} per iteration, "
      f"{num_iterations} iterations; image buffer {buf['img'].numel() * 4 / 1e9:.2f} GB; sensing agent "
      f"{sum(q.numel() for q in agent.parameters())} parameters", flush=True)

img, aux = sense.reset()
next_img, next_aux = img.clone(), aux.clone()
next_done = torch.zeros(N, **f32)
next_lstm_state = agent.initial_state(N, device)
global_step, start_time = 0, time.time()
dt_phys = u.physics_dt


def save_checkpoint(iteration):
    path = os.path.join(log_dir, "checkpoints", f"model_{iteration:05d}.pt")
    torch.save({"model_state_dict": agent.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "args": asdict(args), "global_step": global_step, "iteration": iteration,
                "next_lstm_state_h": next_lstm_state[0].cpu(), "next_lstm_state_c": next_lstm_state[1].cpu(),
                "fps_choices": args.fps_choices, "img_shape": list(img_shape),
                "obs_layout": ["image (held scan)", "fps_ratio", "obs_age_ratio", "frame_ratio"],
                "max_obs_interval_steps": sense.max_obs_interval, "budget": args.budget, "frame_cost": args.frame_cost,
                "nav_checkpoint": {"path": args.nav_checkpoint, "sha256": nav_sha}, "exp_id": exp_id}, path)
    return path


for iteration in range(1, num_iterations + 1):
    torch.cuda.reset_peak_memory_stats(device)
    t_it = time.time()
    initial_lstm_state = (next_lstm_state[0].clone(), next_lstm_state[1].clone())
    if args.anneal_lr:
        optimizer.param_groups[0]["lr"] = (1.0 - (iteration - 1.0) / num_iterations) * args.learning_rate
    # ------------------------------------------------------------------ rollout
    for step in range(T):
        global_step += N
        buf["img"][step] = next_img
        buf["aux"][step] = next_aux
        buf["dones"][step] = next_done
        with torch.no_grad():
            action, logprob, _, value, next_lstm_state = agent.get_action_and_value(
                next_img, next_aux, next_lstm_state, next_done)
        buf["values"][step] = value.flatten()
        buf["actions"][step] = action.float()
        buf["logprobs"][step] = logprob
        (next_img, next_aux), reward, d, info = sense.step(action)
        buf["rewards"][step] = reward
        buf["masks"][step] = info["mask"]
        next_done = d.float()
        ep_done[step] = d
        for k in ep_buf:
            ep_buf[k][step] = info["episode"][k]
    t_roll = time.time() - t_it
    mem_roll = torch.cuda.max_memory_allocated(device) / 1e9
    # ------------------------------------------------------------------ GAE
    with torch.no_grad():
        next_value = agent.get_value(next_img, next_aux, next_lstm_state, next_done).reshape(1, -1)
        adv, ret = compute_gae(buf["rewards"], buf["values"], buf["dones"], next_value, next_done,
                               args.gamma, args.gae_lambda)
    buf["advantages"], buf["returns"] = adv, ret
    # ------------------------------------------------------------------ update
    t_up = time.time()
    epm = N // args.num_minibatches
    clipfracs, stats = [], None
    for epoch in range(args.update_epochs):
        perm = torch.randperm(N, device=device)
        for s0 in range(0, N, epm):
            stats = minibatch_update(agent, optimizer, buf, perm[s0:s0 + epm], initial_lstm_state, args,
                                     chunk_envs=args.update_chunk_envs)
            clipfracs.append(stats["clipfrac"])
        if args.target_kl is not None and float(stats["approx_kl"]) > args.target_kl:
            break
    t_upd = time.time() - t_up
    mem_peak = torch.cuda.max_memory_allocated(device) / 1e9
    # ------------------------------------------------------------------ logging (one host transfer per iteration)
    y_pred, y_true = buf["values"].reshape(-1), buf["returns"].reshape(-1)
    var_y = torch.var(y_true, unbiased=False)
    explained_var = float("nan") if float(var_y) == 0 else float(1 - torch.var(y_true - y_pred, unbiased=False) / var_y)
    W = lambda tag, v: writer.add_scalar(tag, float(v), global_step)  # noqa: E731
    W("charts/learning_rate", optimizer.param_groups[0]["lr"])
    for k in ("value_loss", "policy_loss", "entropy", "old_approx_kl", "approx_kl"):
        W(f"losses/{k}", stats[k])
    W("losses/clipfrac", torch.stack(clipfracs).mean())
    W("losses/explained_variance", explained_var)
    W("losses/grad_norm_last", stats["grad_norm"])
    sps = int(global_step / (time.time() - start_time))
    W("charts/SPS", sps)
    m = buf["masks"]
    W("rollout/causal_tick_fraction", m.mean())
    applied = buf["actions"][m.bool()]
    for i, f in enumerate(args.fps_choices):
        W(f"rollout/applied_action_frac_{f:g}Hz", (applied == i).float().mean() if applied.numel() else 0.0)
        W(f"rollout/selected_action_frac_{f:g}Hz", (buf["actions"] == i).float().mean())
    n_ep = int(ep_done.sum())
    if n_ep:
        E = {k: v[ep_done] for k, v in ep_buf.items()}
        L = E["length"]
        W("charts/episodes", n_ep)
        W("charts/episodic_return", E["return"].mean())
        W("charts/episodic_length", L.mean())
        W("charts/mean_chosen_fps", (E["fps_inforce_sum"] / L).mean())        # reference: rate in force per tick
        W("charts/mean_selected_fps", (E["fps_selected_sum"] / L).mean())
        W("charts/mean_applied_fps", (E["fps_applied_sum"][E["causal"] > 0] / E["causal"][E["causal"] > 0]).mean()
          if bool((E["causal"] > 0).any()) else 0.0)
        W("charts/mean_nav_reward", (E["nav_return"] / L).mean())
        W("charts/episodic_nav_reward", E["nav_return"].mean())
        W("charts/episodic_frame_penalty", -E["frame_cost_sum"].mean())
        W("charts/episodic_budget_penalty", -E["budget_penalty_sum"].mean())
        W("charts/episode_frame_count", E["frames"].mean())
        W("charts/budget_exceeded", E["budget_exceeded"].mean())
        W("charts/measured_hz", ((E["frames"] - 1) / (L * u.step_dt)).mean())
        W("charts/mean_obs_age_s", (E["age_ticks_sum"] / L).mean() * dt_phys)
        W("charts/causal_decisions_per_episode", E["causal"].mean())
        for code, name in OUTCOME_NAMES.items():
            W(f"outcomes/{name}", (E["outcome"] == code).float().mean())
    W("perf/rollout_s", t_roll)
    W("perf/update_s", t_upd)
    W("perf/gpu_mem_peak_rollout_gb", mem_roll)
    W("perf/gpu_mem_peak_gb", mem_peak)
    print(f"iteration={iteration}/{num_iterations} global_step={global_step} SPS={sps} rollout {t_roll:.1f} s "
          f"update {t_upd:.1f} s mem {mem_roll:.1f}/{mem_peak:.1f} GB episodes {n_ep}"
          + (f" return {float(E['return'].mean()):.2f} success {float((E['outcome'] == 1).float().mean()):.2f}"
             f" frames {float(E['frames'].mean()):.1f}" if n_ep else ""), flush=True)
    if iteration % args.checkpoint_interval == 0:
        print(f"  checkpoint -> {save_checkpoint(iteration)}", flush=True)

path = save_checkpoint(num_iterations)
nav_same = all(torch.equal(v, nav_params_before[k]) for k, v in nav.state_dict().items())
with open(os.path.join(log_dir, "params", "final_checks.json"), "w") as f:
    json.dump({"nav05_state_unchanged": nav_same, "final_checkpoint": path, "global_step": global_step}, f, indent=1)
print(f"[TMP-TRAIN] done; NAV05 unchanged: {nav_same}; final checkpoint {path}", flush=True)
writer.close()
os._exit(0)
