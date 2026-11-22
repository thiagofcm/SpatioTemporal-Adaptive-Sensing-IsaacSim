"""2D-LiDAR navigation tasks (DirectRLEnv). Importing this package registers the gym tasks."""
import gymnasium as gym

from . import agents

# Same env and cfg classes; the scene is chosen by the cfg's scene_name. "Static" is kept for NAV01 (static default).
for _task_id in ("STNav-TB4-Corridor-Static-v0", "STNav-TB4-Corridor-v0"):
    gym.register(
        id=_task_id,
        entry_point=f"{__name__}.corridor_env:CorridorNavEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.corridor_env_cfg:CorridorNavEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CorridorNavPPORunnerCfg",
        },
    )
