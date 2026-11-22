"""2D-LiDAR navigation tasks (DirectRLEnv). Importing this package registers the gym tasks."""
import gymnasium as gym

from . import agents

gym.register(
    id="STNav-TB4-Corridor-Static-v0",
    entry_point=f"{__name__}.corridor_env:CorridorNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.corridor_env_cfg:CorridorNavEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CorridorNavPPORunnerCfg",
    },
)
