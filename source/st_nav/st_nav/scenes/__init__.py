"""Scene registry. Every scene is one file in this folder with the same interface (as in the reference
envs/scenes/__init__.py):

    NAME              scene name (the env cfg's `scene_name`)
    WALLS             static walls (cx, cy, length, yaw), thickness WALL_THICKNESS
    HAS_BAFFLES       True: the Z1-Z4 kinematic baffles with the bank / offset randomization of corridor_static_random
    MOVING_OBSTACLES  moving-obstacle specs (motion: _obstacle_motion.py); empty for static scenes

Shared pieces: _common.py (arena, corridor walls, start poses, goals) and _obstacle_motion.py (obstacle motion).
To add a scene: create <name>.py with these fields and add the module to SCENE_MODULES below.
"""
from . import corridor_dynamic_chase_random, corridor_static_random

SCENE_MODULES = {m.NAME: m for m in (
    corridor_static_random,            # corridor + Z1-Z4 baffles (sizes from the bank, offsets per reset)   [NAV01]
    corridor_dynamic_chase_random,     # corridor + moving OB1 / OB2 (random_ping_pong)                        [NAV02, NAV03]
)}

for _m in SCENE_MODULES.values():
    for _field in ("NAME", "WALLS", "HAS_BAFFLES", "MOVING_OBSTACLES"):
        if not hasattr(_m, _field):
            raise AttributeError(f"scene file {_m.__name__} has no {_field}")


def get_scene(name: str):
    if name not in SCENE_MODULES:
        raise ValueError(f"unknown scene {name!r}; valid scenes: {sorted(SCENE_MODULES)}")
    return SCENE_MODULES[name]
