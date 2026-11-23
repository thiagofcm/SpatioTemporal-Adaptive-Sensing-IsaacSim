"""Untrained (randomly initialized) policy checkpoint for an experiment config, for pipeline checks of evaluation,
recording and visualization with a new observation layout (e.g. the 3D LiDAR, 4733 inputs). No training step.

    isaacrun -g <gpu> -- python tests/sim/make_untrained_checkpoint.py --headless \
        --config Navigation/configs/_templates/static_lidar3d.yaml --out tests/sim/results/m7/runs/static_lidar3d

Writes <out>/model_0.pt and <out>/params/config.yaml (read by Navigation/scripts/evaluate.py; evaluate with --out_dir).
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tools import experiments as X  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--config", required=True)
p.add_argument("--out", required=True)
p.add_argument("--num_envs", type=int, default=8)
p.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app

from importlib.metadata import version  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import st_nav.tasks.nav2d  # noqa: E402,F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from st_nav.tasks.nav2d.agents.rsl_rl_ppo_cfg import CorridorNavPPORunnerCfg  # noqa: E402
from st_nav.tasks.nav2d.corridor_env_cfg import CorridorNavEnvCfg  # noqa: E402

exp = X.yaml.safe_load(open(args.config))
env_over = {k: v for k, v in X.flatten(exp.get("env") or {}).items() if k != "scene.num_envs"}
cfg = CorridorNavEnvCfg()
X.apply_overrides(cfg, env_over, "env")
cfg.scene.num_envs, cfg.sim.device, cfg.seed = args.num_envs, args.device, args.seed
genv = gym.make(exp.get("task", "STNav-TB4-Corridor-v0"), cfg=cfg)
agent = handle_deprecated_rsl_rl_cfg(CorridorNavPPORunnerCfg(), version("rsl-rl-lib"))
agent.device, agent.seed = args.device, args.seed
torch.manual_seed(args.seed)
env = RslRlVecEnvWrapper(genv, clip_actions=agent.clip_actions)
os.makedirs(os.path.join(args.out, "params"), exist_ok=True)
runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=args.device)
saved = runner.alg.save()                          # = OnPolicyRunner.save without the logger (no log_dir here)
saved["iter"], saved["infos"] = 0, None
torch.save(saved, os.path.join(args.out, "model_0.pt"))
with open(os.path.join(args.out, "params", "config.yaml"), "w") as f:
    X.yaml.safe_dump({"exp_id": exp.get("exp_id"), "task": exp.get("task"), "env": env_over,
                      "note": "UNTRAINED checkpoint (random initialization) for pipeline checks"}, f, sort_keys=False)
print("observation dim", genv.unwrapped.cfg.observation_space, "->", os.path.join(args.out, "model_0.pt"), flush=True)
genv.close()
app.close()
