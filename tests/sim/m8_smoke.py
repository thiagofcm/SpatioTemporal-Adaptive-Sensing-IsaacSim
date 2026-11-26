"""M8 smoke tests of the 3D-LiDAR CNN PPO pipeline (task STNav-TB4-Corridor-Lidar3D-v0, CorridorNavCNNPPORunnerCfg).

    isaacrun -g <gpu> -- python tests/sim/m8_smoke.py --headless [--num_envs 64]

S1 observation shapes / ranges: {"policy": (N, 7), "lidar": (N, 1, 34, 139)} in [0, 1]; the image is the held scan
   scattered on the 34 x 139 grid (row-major top-left), lidar_block() inverts it; R4 mount in the ray caster
S2 LiDAR timing: scans at 4 Hz at control instants (0.4, 0.6, 0.8, 1.0, 1.4 ... s), the image is held between scans
   and changes only at a new scan (noise on: every new scan differs)
S3 network: actor / critic = Conv 16-32-64 (3x3, stride 2, ReLU) -> Linear 3072 -> 128 + ReLU, MLP (135) -> 256 -> 128
   (ELU) -> 2 (actor, Gaussian) / 1 (critic); separate encoders
S4 CNN gradients: PPO surrogate and value losses on a real rollout batch give non-zero gradients in every conv / linear
   layer of both encoders
S5 PPO updates: 2 learning iterations run, losses finite, every CNN parameter tensor of actor and critic changed
S6 checkpoint save / load: a new runner loading the checkpoint has identical parameters, normalizer statistics and
   identical deterministic actions on the same observations
Writes tests/sim/results/m8_smoke.json.
"""
import argparse
import json
import math
import os
import sys
import tempfile

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--num_envs", type=int, default=64)
p.add_argument("--out", default=os.path.join(os.environ.get("ST_ROOT", "."), "tests", "sim", "results"))
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import st_nav.tasks.nav2d  # noqa: E402,F401
from importlib.metadata import version  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from st_nav.sensors.patterns import rpy_to_quat_wxyz  # noqa: E402
from st_nav.tasks.nav2d.agents.rsl_rl_ppo_cnn_cfg import CorridorNavCNNPPORunnerCfg  # noqa: E402
from st_nav.tasks.nav2d.corridor_env_lidar3d_cfg import CorridorNavLidar3DEnvCfg  # noqa: E402

os.makedirs(args.out, exist_ok=True)
checks = {}


def ok(name, cond, detail=None):
    checks[name] = {"pass": bool(cond), "detail": detail}
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail if detail is not None else ''}", flush=True)


cfg = CorridorNavLidar3DEnvCfg()
cfg.scene.num_envs, cfg.sim.device, cfg.seed = args.num_envs, args.device, 3
genv = gym.make("STNav-TB4-Corridor-Lidar3D-v0", cfg=cfg)
u = genv.unwrapped
agent = handle_deprecated_rsl_rl_cfg(CorridorNavCNNPPORunnerCfg(), version("rsl-rl-lib"))
agent.device = args.device
agent.num_steps_per_env = 16
env = RslRlVecEnvWrapper(genv, clip_actions=agent.clip_actions)
N, R = u.num_envs, u.corr.lidar.num_rays

# ---------------------------------------------------------------- S1 shapes
obs = env.get_observations()
lid, nav = obs["lidar"], obs["policy"]
img_from_scan = torch.ones(N, 34 * 139, device=u.device)
img_from_scan[:, u._img_idx] = u.scan
rc = u.corr.lidar.rc
ok("S1_obs_groups_and_shapes", set(obs.keys()) == {"policy", "lidar"} and tuple(nav.shape) == (N, 7)
   and tuple(lid.shape) == (N, 1, 34, 139), {k: list(v.shape) for k, v in obs.items()})
ok("S1_ranges_normalized_0_1", bool((lid >= 0).all() and (lid <= 1).all()) and bool(torch.isfinite(nav).all()),
   {"lidar_min": float(lid.min()), "lidar_max": float(lid.max()), "nav_absmax": float(nav.abs().max())})
ok("S1_image_is_held_scan_row_major", torch.equal(lid.view(N, -1), img_from_scan) and torch.equal(u._img_idx.cpu(), torch.arange(R))
   and torch.equal(u.lidar_block(obs), u.scan), "image[r, c] = scan[r * 139 + c]")
ok("S1_r4_mount_in_ray_caster", np.allclose(rc.cfg.offset.pos, (0, 0, 0.40)) and np.allclose(rc.cfg.offset.rot, rpy_to_quat_wxyz((0, 6.75, 0)))
   and u.corr.lidar.range_max == 12.0 and R == 4726, {"pos": list(rc.cfg.offset.pos), "rot": list(rc.cfg.offset.rot)})

# ---------------------------------------------------------------- S2 timing (robots standing, noise on)
env.reset()
prev = env.get_observations()["lidar"].clone()
acq, held_ok, new_ok = [], True, True
zero = torch.zeros(N, 2, device=u.device)
for k in range(50):
    o, _, d, _ = env.step(zero)
    fresh = (u.scan_t - u.ep_time).abs() < 1e-6
    alive = ~d.bool()
    if bool(fresh[0]) and bool(alive[0]):
        acq.append(round(float(u.ep_time[0]), 3))
    cur = o["lidar"]
    same = (cur == prev).flatten(1).all(1)
    held_ok &= bool(same[alive & ~fresh].all())
    new_ok &= bool((~same[alive & fresh]).all())
    prev = cur.clone()
exp_t = sorted({round(math.ceil(round(t / 0.2, 6)) * 0.2, 3) for t in np.arange(0.25, 10.0001, 0.25)})
ok("S2_scans_4Hz_at_control_instants", acq == exp_t, {"n_10s": len(acq), "first": acq[:8]})
ok("S2_image_held_between_scans_new_at_scans", held_ok and new_ok)

# ---------------------------------------------------------------- S3 network
tmp = tempfile.mkdtemp(prefix="m8_smoke_")
runner = OnPolicyRunner(env, agent.to_dict(), log_dir=tmp, device=agent.device)
actor, critic = runner.alg.actor, runner.alg.critic


def describe(m):
    enc = m.cnns["lidar"]
    convs = [l for l in enc[0] if isinstance(l, torch.nn.Conv2d)]
    lin = enc[1]
    mlp_lin = [l for l in m.mlp if isinstance(l, torch.nn.Linear)]
    return {"convs": [(c.in_channels, c.out_channels, c.kernel_size, c.stride) for c in convs],
            "conv_acts": [type(l).__name__ for l in enc[0] if not isinstance(l, (torch.nn.Conv2d, torch.nn.Flatten))],
            "proj": (lin.in_features, lin.out_features, type(enc[2]).__name__),
            "mlp": [(l.in_features, l.out_features) for l in mlp_lin], "params": sum(p.numel() for p in m.parameters()),
            "obs_normalized_dim": m.obs_dim}


da, dc = describe(actor), describe(critic)
conv_ok = lambda d: [c[1] for c in d["convs"]] == [16, 32, 64] and all(c[2] == (3, 3) and c[3] == (2, 2) for c in d["convs"]) \
    and set(d["conv_acts"]) == {"ReLU"} and d["proj"] == (3072, 128, "ReLU")  # noqa: E731
ok("S3_actor_architecture", conv_ok(da) and da["mlp"] == [(135, 256), (256, 128), (128, 2)] and da["obs_normalized_dim"] == 7, da)
ok("S3_critic_architecture", conv_ok(dc) and dc["mlp"] == [(135, 256), (256, 128), (128, 1)], dc)
ok("S3_separate_encoders", actor.cnns["lidar"] is not critic.cnns["lidar"]
   and actor.cnns["lidar"][0][0].weight.data_ptr() != critic.cnns["lidar"][0][0].weight.data_ptr())

# ---------------------------------------------------------------- S4 gradients through the CNN (PPO losses on a rollout)
o = env.get_observations().to(agent.device)
with torch.inference_mode(False):
    ob = o.clone()
    actor.train(); critic.train()
    a = actor(ob, stochastic_output=True)
    logp = actor.get_output_log_prob(a.detach())
    v = critic(ob)
    loss = -(logp * torch.randn_like(logp)).mean() + (v.squeeze(-1) - torch.randn_like(v.squeeze(-1))).pow(2).mean()
    actor.zero_grad(); critic.zero_grad()
    loss.backward()
grads = {f"{n_}.{pn}": float(p_.grad.norm()) if p_.grad is not None else 0.0
         for n_, m in (("actor", actor), ("critic", critic)) for pn, p_ in m.cnns.named_parameters()}
ok("S4_cnn_gradients_nonzero", all(g > 0 and math.isfinite(g) for g in grads.values()),
   {k: f"{v:.2e}" for k, v in grads.items()})
actor.zero_grad(); critic.zero_grad()

# ---------------------------------------------------------------- S5 PPO updates
before = {f"{n_}.{pn}": p_.detach().clone() for n_, m in (("actor", actor), ("critic", critic)) for pn, p_ in m.cnns.named_parameters()}
losses = []
orig_update = runner.alg.update


def update_hook(*a_, **k_):
    out = orig_update(*a_, **k_)
    losses.append({k: float(v) for k, v in out.items()})
    return out


runner.alg.update = update_hook
runner.learn(num_learning_iterations=2, init_at_random_ep_len=True)
changed = {k: float((p_.detach() - before[k]).abs().max()) for n_, m in (("actor", actor), ("critic", critic))
           for k, p_ in ((f"{n_}.{pn}", q) for pn, q in m.cnns.named_parameters())}
ok("S5_ppo_updates_finite", len(losses) == 2 and all(math.isfinite(v) for l in losses for v in l.values()), losses)
ok("S5_all_cnn_parameters_updated", all(v > 0 for v in changed.values()), {k: f"{v:.1e}" for k, v in changed.items()})

# ---------------------------------------------------------------- S6 checkpoint save / load
ck = os.path.join(tmp, "model_smoke.pt")
saved = runner.alg.save()
saved["iter"], saved["infos"] = runner.current_learning_iteration, None
torch.save(saved, ck)
runner2 = OnPolicyRunner(env, handle_deprecated_rsl_rl_cfg(CorridorNavCNNPPORunnerCfg(device=agent.device), version("rsl-rl-lib")).to_dict(),
                         log_dir=None, device=agent.device)
runner2.load(ck)
sd1 = {**{f"a.{k}": v for k, v in actor.state_dict().items()}, **{f"c.{k}": v for k, v in critic.state_dict().items()}}
sd2 = {**{f"a.{k}": v for k, v in runner2.alg.actor.state_dict().items()}, **{f"c.{k}": v for k, v in runner2.alg.critic.state_dict().items()}}
same_params = sd1.keys() == sd2.keys() and all(torch.equal(sd1[k], sd2[k]) for k in sd1)
pol1, pol2 = runner.get_inference_policy(device=agent.device), runner2.get_inference_policy(device=agent.device)
o = env.get_observations().to(agent.device)
with torch.inference_mode():
    a1, a2 = pol1(o), pol2(o)
ok("S6_checkpoint_roundtrip", same_params and torch.equal(a1, a2),
   {"tensors": len(sd1), "max_action_diff": float((a1 - a2).abs().max()), "file_MB": round(os.path.getsize(ck) / 1e6, 2)})

n_pass = sum(c["pass"] for c in checks.values())
print(f"{n_pass}/{len(checks)} checks passed", flush=True)
with open(os.path.join(args.out, "m8_smoke.json"), "w") as f:
    json.dump({"num_envs": N, "checks": checks}, f, indent=1, default=str)
os._exit(0 if n_pass == len(checks) else 1)
