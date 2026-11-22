"""Run a script with ANY access to the reference repository (~/adaptive_sensor_policy) forbidden.

    isaacrun -g <gpu> -- python tests/sim/no_reference_guard.py <script.py> [script args]

An audit hook raises on imports, file opens and directory listings whose path contains the reference root, so the
wrapped run fails if any code path still depends on the reference. Prints [NO-REF] OK at exit if nothing was touched.
"""
import atexit
import os
import runpy
import sys

REF = os.path.realpath(os.path.expanduser("~/adaptive_sensor_policy"))
hits = []


def _block(event, what):
    hits.append((event, what))
    raise PermissionError(f"[NO-REF] reference access blocked: {event} {what}")


def hook(event, args):
    if event in ("open", "os.listdir", "os.scandir") and args and isinstance(args[0], (str, bytes, os.PathLike)):
        path = os.fsdecode(args[0])
        if "adaptive_sensor_policy" in path:
            _block(event, path)
    elif event == "import" and args:            # (module, filename, sys.path, ...)
        mod, fname = args[0], args[1] if len(args) > 1 else None
        if mod == "envs" or str(mod).startswith("envs.") or (isinstance(fname, str) and "adaptive_sensor_policy" in fname):
            _block(event, f"{mod} {fname}")


sys.addaudithook(hook)


@atexit.register
def _report():
    leaked = [p for p in sys.path if "adaptive_sensor_policy" in p] + [m for m in sys.modules if m.startswith("envs")]
    print(f"[NO-REF] {'OK: no reference access' if not hits and not leaked else f'FAIL {hits[:3]} {leaked[:3]}'}", flush=True)


_os_exit = os._exit


def _exit(code):                               # the project scripts end with os._exit (skips atexit): report first
    _report()
    _os_exit(code)


os._exit = _exit
script = sys.argv[1]
sys.argv = sys.argv[1:]
sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
runpy.run_path(script, run_name="__main__")
