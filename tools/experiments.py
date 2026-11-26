#!/usr/bin/env python3
"""Single source of truth for experiment IDs, paths and YAML configs (conventions: docs/EXPERIMENTS.md).

    experiment ID   <PREFIX><NN>_<scene>_<short-desc>     NAV01_static_initial, STS01_static_lidar-dr
                    NAV -> Navigation/   TMP / SPA / STS -> AdaptiveSensing/ {temporal, spatial, spatiotemporal}
    config          <line>/configs[/<mode>]/<ID>.yaml
    training runs   <line>/runs[/<mode>]/<ID>/<date>_<run_name>/
    evaluations     <line>/eval[/<mode>]/<ID>/<protocol>/<policy>/   (experiment card: <line>/eval[/<mode>]/<ID>/README.md)

Experiment YAML (overrides on top of the Isaac Lab configclass defaults registered with the Gym task):

    exp_id: NAV01_static_initial
    task: STNav-TB4-Corridor-Static-v0
    run_name: ppo                  # optional; default ppo_n<num_envs>_s<seed>
    env:   {start_pose_dr: true, scene.num_envs: 4096}      # dotted paths into the env cfg (nested maps also accepted)
    agent: {max_iterations: 1500, seed: 42, algorithm.learning_rate: 3.0e-4}
    sweep: {}                      # reserved: {"agent.seed": [42, 43]} -> one run per combination (future sweep.py,
                                   # which will call train.py with one --set per swept key)

Precedence: configclass defaults < YAML < --set section.path=value. Unknown keys are errors.

CLI (from the repo root):  python tools/experiments.py list   |   python tools/experiments.py paths NAV01_static_initial
"""
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
LINES = {"NAV": ("Navigation", None), "TMP": ("AdaptiveSensing", "temporal"), "SPA": ("AdaptiveSensing", "spatial"),
         "STS": ("AdaptiveSensing", "spatiotemporal")}
EXP_ID_RE = re.compile(r"^(NAV|TMP|SPA|STS)\d{2}_[a-z0-9][a-z0-9.-]*(_[a-z0-9][a-z0-9.-]*)*$")
SECTIONS = ("env", "agent")


# ------------------------------------------------------------------------------------------------ IDs and paths
def check_exp_id(exp_id: str) -> str:
    if not exp_id or not EXP_ID_RE.match(exp_id):
        raise SystemExit(f"invalid experiment ID {exp_id!r}: expected <NAV|TMP|SPA|STS><NN>_<scene>_<desc> "
                         "(lowercase, words joined by '-'), see docs/EXPERIMENTS.md")
    return exp_id


def paths(exp_id: str) -> dict:
    line, mode = LINES[check_exp_id(exp_id)[:3]]
    sub = (mode,) if mode else ()
    root = REPO_ROOT / line
    ev = root.joinpath("eval", *sub, exp_id)
    return {"config": root.joinpath("configs", *sub, f"{exp_id}.yaml"), "runs": root.joinpath("runs", *sub, exp_id),
            "eval": ev, "card": ev / "README.md"}


def exp_id_from_run_path(path) -> str | None:
    """Experiment ID of a checkpoint / run folder inside <line>/runs[/<mode>]/<ID>/<run>/."""
    parts = Path(path).resolve().parts
    for i, p in enumerate(parts[:-1]):
        if p == "runs":
            for cand in parts[i + 1:i + 3]:
                if EXP_ID_RE.match(cand):
                    return cand
    return None


def paths_3d(exp_id: str) -> dict:
    """3D-LiDAR (CNN) navigation experiments (M8): separate roots so they never mix with the 2D ones:
    <line>/configs_3d/<ID>.yaml, <line>/runs_3d/<ID>/, <line>/eval_3d/<ID>/."""
    line, mode = LINES[check_exp_id(exp_id)[:3]]
    if mode:
        raise SystemExit(f"{exp_id}: 3D paths are defined for the Navigation line only")
    root = REPO_ROOT / line
    ev = root / "eval_3d" / exp_id
    return {"config": root / "configs_3d" / f"{exp_id}.yaml", "runs": root / "runs_3d" / exp_id, "eval": ev,
            "card": ev / "README.md"}


def exp_id_from_run_path_3d(path) -> str | None:
    """Experiment ID of a checkpoint / run folder inside Navigation/runs_3d/<ID>/<run>/."""
    parts = Path(path).resolve().parts
    for i, p in enumerate(parts[:-1]):
        if p == "runs_3d" and i + 1 < len(parts) and EXP_ID_RE.match(parts[i + 1]):
            return parts[i + 1]
    return None


# ------------------------------------------------------------------------------------------------ YAML configs
def load_config(path) -> dict:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text()) or {}
    unknown = set(cfg) - {"exp_id", "task", "run_name", "sweep", "description", "init_from", *SECTIONS}  # init_from: train_3d.py only
    if unknown:
        raise SystemExit(f"{path}: unknown top-level keys {sorted(unknown)}")
    check_exp_id(cfg.get("exp_id"))
    if path.stem != cfg["exp_id"] and not path.parent.name.startswith("_"):
        raise SystemExit(f"{path}: file name must equal exp_id ({cfg['exp_id']})")
    if "task" not in cfg:
        raise SystemExit(f"{path}: missing 'task' (Gym task ID)")
    for s in SECTIONS:
        cfg[s] = flatten(cfg.get(s) or {})
    cfg["sweep"] = cfg.get("sweep") or {}
    return cfg


def flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def parse_set(items) -> dict:
    """['env.scene.num_envs=1024', 'agent.max_iterations=201'] -> {'env': {...}, 'agent': {...}} (YAML-typed values)."""
    out = {s: {} for s in SECTIONS}
    for it in items or []:
        key, _, val = it.partition("=")
        section, _, sub = key.partition(".")
        if section not in SECTIONS or not sub or not _:
            raise SystemExit(f"--set {it!r}: expected <env|agent>.<path>=<value>")
        out[section][sub] = yaml.safe_load(val)
    return out


def _coerce(old, new):
    if isinstance(old, tuple) and isinstance(new, list):
        return tuple(_coerce(old[0] if old else None, x) if isinstance(x, list) else x for x in new)
    if isinstance(old, float) and isinstance(new, int) and not isinstance(new, bool):
        return float(new)
    return new


def apply_overrides(obj, overrides: dict, label: str):
    """Set dotted attribute paths on a configclass instance; every path must already exist."""
    for path, value in overrides.items():
        *parents, leaf = path.split(".")
        tgt = obj
        for p in parents:
            if not hasattr(tgt, p):
                raise SystemExit(f"{label}.{path}: unknown key '{p}' in {type(tgt).__name__}")
            tgt = getattr(tgt, p)
        if not hasattr(tgt, leaf):
            raise SystemExit(f"{label}.{path}: unknown key '{leaf}' in {type(tgt).__name__}")
        setattr(tgt, leaf, _coerce(getattr(tgt, leaf), value))


def task_cfgs(task: str):
    """Default env and RSL-RL runner configclass instances registered with a Gym task (Isaac Lab entry points)."""
    import importlib

    import gymnasium as gym

    kw = gym.spec(task).kwargs

    def make(entry):
        mod, _, cls = entry.partition(":")
        return getattr(importlib.import_module(mod), cls)()

    return make(kw["env_cfg_entry_point"]), make(kw["rsl_rl_cfg_entry_point"])


def git_state(path) -> dict:
    def run(*a):
        r = subprocess.run(["git", "-C", str(path), *a], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None
    commit = run("rev-parse", "HEAD")
    return {"path": str(path), "commit": commit, "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(run("status", "--porcelain")) if commit else None}


# ------------------------------------------------------------------------------------------------ CLI
def _all_ids():
    ids = []
    for line, mode in sorted(set(LINES.values()), key=str):
        d = REPO_ROOT.joinpath(line, "configs", *((mode,) if mode else ()))
        ids += [p.stem for p in sorted(d.glob("*.yaml"))] if d.is_dir() else []
    return ids


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = p.add_subparsers(dest="cmd", required=True)
    sp.add_parser("list")
    pp = sp.add_parser("paths")
    pp.add_argument("exp_id")
    a = p.parse_args()
    if a.cmd == "list":
        for i in _all_ids():
            pth = paths(i)
            n_runs = len([d for d in pth["runs"].iterdir() if d.is_dir()]) if pth["runs"].is_dir() else 0
            print(f"{i:40s} runs: {n_runs:2d}   card: {'yes' if pth['card'].exists() else 'no'}")
    else:
        for k, v in paths(a.exp_id).items():
            print(f"{k:7s} {v.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
