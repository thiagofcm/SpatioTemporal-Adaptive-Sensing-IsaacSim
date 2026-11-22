# Spatiotemporal Adaptive Sensing: Development Workspace

Workspace for the spatiotemporal adaptive sensing project: RL mobile-robot navigation with a programmable Mirrorcle SyMPL 3D LiDAR, built on Isaac Lab 2.3.2 / Isaac Sim 5.1.

This folder currently holds the **headless launch tooling** for the shared GPU server. Navigation code (phase 2, migration from OmniIsaacGymEnvs) and the Mirrorcle sensor (phase 3) will be added here later.

> **Status (2026-11-21):** headless GPU physics, parallel RL training and Warp ray casting work using the CUDA workaround below.
> Rendering (cameras, RTX LiDAR, `--video`, `--livestream`) is **unavailable** until the server administrator fixes the driver and render-device permissions.
> See `../phase1_diagnostics/PHASE1_INSTALLATION_REPORT.md`.

---

## Folder layout

```
spatiotemporal_sensing/
├── README.md              this file
├── env.sh                 session-only environment: python env + CUDA workaround + GPU selection
├── bin/
│   ├── isaacrun           launcher: GPU restriction, logging, timeout, clean shutdown
│   └── check_physx_gpu    checks the latest Isaac Sim log to confirm PhysX ran on the GPU
├── validation/
│   ├── diag_empty.py      empty scene, GPU physics stepping test
│   ├── diag_raycaster.py  headless GPU LiDAR (Warp ray caster) benchmark
│   └── vkprobe.py         Vulkan device check (rendering readiness)
├── logs/launcher/         one log per isaacrun call (git-ignored)
└── .shim/                 created automatically: libcuda.so symlink (git-ignored)
```

None of these tools modifies the system, the `env_isaaclab` conda environment, or `IsaacLab_Fork`. Everything they create stays inside this folder.

---

## 1. The CUDA library workaround

### Why it's needed
The October driver upgrade (550 → 580) left the unversioned driver library `/usr/lib/x86_64-linux-gnu/libcuda.so` missing. PyTorch and Warp load `libcuda.so.1`, which is present, so they work. PhysX looks for `libcuda.so`, can't find it, prints

```
PhysX warning: GPU solver pipeline failed, switching to software
```

and the simulation hangs on its first step.

### What the workaround does
`env.sh` creates a private symlink

```
spatiotemporal_sensing/.shim/libcuda.so  ->  /usr/lib/x86_64-linux-gnu/libcuda.so.1
```

and puts `.shim/` first on `LD_LIBRARY_PATH` **for the current shell or process only**. PhysX then finds the same driver library everything else uses.

* Nothing is installed or changed system-wide. Close the shell and the workaround is gone.
* **It turns itself off** once the administrator restores `/usr/lib/x86_64-linux-gnu/libcuda.so`. `env.sh` then reports `CUDA shim : not needed`.
* To disable it by hand: `export ST_CUDA_SHIM=0` before sourcing `env.sh`.

### How to activate it
Use one of two methods; both apply the workaround automatically.

| Method | When to use | How |
|---|---|---|
| **A. `isaacrun` launcher (recommended)** | any simulation or training run | `~/isaac_lab/spatiotemporal_sensing/bin/isaacrun -g <GPU> <script.py> [args]` |
| **B. Source `env.sh`** | interactive work: Python shells, quick scripts, `isaaclab.sh` | `source ~/isaac_lab/spatiotemporal_sensing/env.sh <GPU>` |

`env.sh` also activates the Python environment, so `conda activate env_isaaclab` is **optional** with either method. Running it anyway is harmless.

### How to confirm it worked
After any run:
```bash
~/isaac_lab/spatiotemporal_sensing/bin/check_physx_gpu
# RESULT: PASS (PhysX on GPU)    <- expected
```
FAIL means PhysX fell back to the CPU, usually because the run was started without `env.sh`/`isaacrun`.

---

## 2. Reusable launch procedure

### One-time, per login session (optional convenience)
```bash
alias isaacrun=~/isaac_lab/spatiotemporal_sensing/bin/isaacrun
```
This lasts only for the current session. If you want it permanently, you can add the line to your own `~/.bashrc`. This workspace doesn't do that for you.

### Launching anything
```bash
cd ~/isaac_lab/IsaacLab_Fork            # or your project folder; outputs go to ./logs
isaacrun -g <GPU|auto> [-t SECONDS] [--] <command or script.py> [args...]
```

| Option | Meaning |
|---|---|
| `-g N` / `--gpu N` | use physical GPU N (0–7, numbered as in `nvidia-smi`) |
| `-g N,M` | several GPUs (distributed training) |
| `-g auto` | pick the GPU with the least memory in use right now |
| `-t SEC` / `--timeout SEC` | stop the whole run after SEC seconds (exit code 124) |
| `-l FILE` / `--log FILE` | custom launcher log path |
| `--no-log` | don't write a launcher log |
| `ST_GPU=N` (env var) | default GPU when `-g` is omitted |

If `-g` is missing and no GPU is otherwise selected, `isaacrun` **refuses to start** and shows current GPU usage. This prevents accidentally using all 8 shared GPUs.

What `isaacrun` does on every run:
1. Sources `env.sh` (Python env, CUDA workaround, GPU restriction).
2. Runs `*.py` scripts with the env's `python`. Any other command (e.g. `./isaaclab.sh -p ...`) runs as given.
3. Writes a header (date, host, GPU, command, Isaac Lab commit) and all output to `logs/launcher/<time>_gpu<N>_<name>.log`.
4. Starts the run in its own process group, so **Ctrl-C, `--timeout`, or killing `isaacrun` stops every process of the run**.
   * The first Ctrl-C asks the run to stop.
   * If it's still running after 20 s it is force-killed.
   * A second Ctrl-C force-kills immediately.

   This avoids a problem seen in phase 1: wrapping `./isaaclab.sh` in `timeout` left an orphaned Python process spinning on the GPU.

### Examples
```bash
cd ~/isaac_lab/IsaacLab_Fork

# Validation: empty scene with GPU physics (should finish 1000 steps in ~1 s)
isaacrun -g 3 -t 180 ~/isaac_lab/spatiotemporal_sensing/validation/diag_empty.py --headless --steps 1000
~/isaac_lab/spatiotemporal_sensing/bin/check_physx_gpu

# Official tutorial
isaacrun -g 3 -t 120 -- ./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --headless

# RL training (RSL-RL), headless
isaacrun -g 3 scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Cartpole-Direct-v0 --headless --num_envs 4096 --max_iterations 300

# Policy evaluation without rendering (small env count)
isaacrun -g 3 scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Cartpole-Direct-v0 --headless --num_envs 16

# Headless GPU LiDAR benchmark
isaacrun -g 3 -t 240 ~/isaac_lab/spatiotemporal_sensing/validation/diag_raycaster.py --headless --num_envs 1024
```

### Interactive use (method B)
```bash
source ~/isaac_lab/spatiotemporal_sensing/env.sh 3
python -c "import torch; print(torch.cuda.device_count())"     # -> 1
cd ~/isaac_lab/IsaacLab_Fork && ./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --headless
```
Without `isaacrun`, aborted Isaac runs can leave orphaned processes. Check with
`ps -u $USER -o pid,etime,pcpu,cmd | grep python` and `kill <pid>` if needed.

### Long runs over SSH
A training run dies if its SSH session closes. Run long jobs inside `tmux` (installed):
```bash
tmux new -s train_gpu3                # start a named session
isaacrun -g 3 scripts/reinforcement_learning/rsl_rl/train.py --task ... --headless
# detach: Ctrl-b then d       | reattach later: tmux attach -t train_gpu3
```
Or fire and forget: `nohup isaacrun -g 3 <...> > /dev/null 2>&1 &`. Output still goes to `logs/launcher/`.

### Monitoring
```bash
tail -f ~/isaac_lab/spatiotemporal_sensing/logs/launcher/<latest>.log
tensorboard --logdir logs --port 6006 --bind_all
#   on your laptop: ssh -L 6006:localhost:6006 <user>@<server>   then open http://localhost:6006
```

---

## 3. Restricting a run to one GPU

### Use `CUDA_VISIBLE_DEVICES`, not just `--device`
Measured on this server (2026-11-21):

| How the GPU was selected | GPUs touched by the process |
|---|---|
| `--device cuda:5` only | **all 8 GPUs**: ~2.6 GB on GPU 5 plus ~260 MB of CUDA context on each of the other 7 (Kit enables peer access to every GPU) |
| `CUDA_VISIBLE_DEVICES=5` (`isaacrun -g 5`) | **only GPU 5** (bus `A1:00.0`) |

`isaacrun -g N` and `source env.sh N` set:
```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID   # CUDA numbering == nvidia-smi numbering
export CUDA_VISIBLE_DEVICES=N         # the process can only see GPU N
```

### Rules of thumb
* **Inside the process the selected GPU is always `cuda:0`.** Omit `--device` (Isaac Lab defaults to `cuda:0`) or pass `--device cuda:0`. Passing `--device cuda:3` with `-g 3` fails because only one GPU is visible. `isaacrun` warns about this.
* Check free GPUs before starting:
  ```bash
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
  nvidia-smi --query-compute-apps=pid,gpu_bus_id,used_memory --format=csv   # who is using what
  ```
  GPU index ↔ PCI bus on this server: 0=`01` 1=`23` 2=`41` 3=`61` 4=`81` 5=`A1` 6=`C1` 7=`E1`.
* **Several experiments in parallel:** one GPU each, e.g. two tmux sessions with `isaacrun -g 2 ...` and `isaacrun -g 3 ...`.
* **One experiment on several GPUs** (Isaac Lab distributed training): pass a list. The GPUs appear as `cuda:0`, `cuda:1`, … and each rank picks its own:
  ```bash
  isaacrun -g 2,3 -- python -m torch.distributed.run --nproc_per_node=2 \
      scripts/reinforcement_learning/rsl_rl/train.py --task ... --headless --distributed
  ```
  This isn't tested yet. Validate it before relying on it.
* **When rendering is available again:** Kit's renderer selects GPUs through Vulkan, which doesn't fully honor `CUDA_VISIBLE_DEVICES`, and by default it may claim all GPUs (`/renderer/multiGpu/enabled=True`). Re-test GPU isolation with `--enable_cameras` / `--video` after the admin fix before using rendering on the shared server.

---

## 4. What works headless right now

| Capability | Status | Notes |
|---|---|---|
| GPU PhysX simulation | ✅ (with workaround) | 1000 empty-scene steps ≈ 0.5–0.9 s |
| GPU-parallel RL training | ✅ | Cartpole, 4096 envs: ~225–240k steps/s |
| Warp ray caster / LiDAR pattern | ✅ | 1024 envs × 5,760 rays ≈ 365 Mrays/s, no rendering needed |
| Cameras, RTX LiDAR, `--video`, `--livestream` | ❌ | waiting for the admin fix (driver cleanup + `render` group) |
| Offline inspection (save tensors → plot locally) | ✅ | the recommended way to inspect LiDAR data until rendering returns |

### After the administrator's fix
1. `python validation/vkprobe.py` (after `source env.sh`) should list 8× NVIDIA RTX A6000.
2. `env.sh` should print `CUDA shim : not needed`.
3. Re-run the validation commands above, then test `--enable_cameras` / `--video` / `--livestream` (see the phase-1 report).

---

## 5. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `GPU solver pipeline failed, switching to software`, run hangs | started without the workaround. Use `isaacrun` or `source env.sh` |
| `isaacrun: choose a GPU with --gpu N\|auto` | intentional. Pick a GPU |
| `invalid device` / CUDA error with `--device cuda:N` | inside a restricted run only `cuda:0` exists |
| Hangs at `sim.reset()` with `--enable_cameras` / `--video` | rendering unavailable (Vulkan). Expected until the admin fix |
| Run finished its work but doesn't exit | known Kit shutdown hang. Use `-t` or Ctrl-C; `isaacrun` cleans up |
| Exit code 124 / 130 | timeout / interrupted by Ctrl-C |
| Disk space | `/md127` is ~95 % full. Prune old `logs/`, checkpoints and videos regularly |
