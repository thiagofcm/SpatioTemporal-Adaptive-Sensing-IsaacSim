"""Read-only access to the reference simplified environment's scene tables (adaptive_sensor_policy/envs/scenes).

Imported without writing bytecode into the reference repository. Only plain data (wall tuples, obstacle specs, DR
ranges) is used; the reference code is not executed beyond module import.
"""
import os
import sys

REFERENCE_ROOT = os.environ.get("ST_REFERENCE_ROOT", os.path.expanduser("~/adaptive_sensor_policy"))


def reference_scene_module(name: str):
    prev = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        if REFERENCE_ROOT not in sys.path:
            sys.path.insert(0, REFERENCE_ROOT)
        from envs.scenes import get_scene  # noqa: E402
        return get_scene(name)
    finally:
        sys.dont_write_bytecode = prev
