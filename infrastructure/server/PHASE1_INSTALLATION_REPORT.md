# Phase 1 – Isaac Sim / Isaac Lab Installation Inspection

**Date:** 2026-11-21 · **Host:** remote headless Ubuntu 24.04.5 server, kernel 7.0.0-38, 8× RTX A6000 (48 GB), 256 CPU threads, 944 GB RAM
**Scope:** inspection only. No source code, packages, drivers or system configuration were changed. Diagnostic scripts live in this folder.

---

## 1. Summary

| Area | Status |
|---|---|
| Python stack (Isaac Sim 5.1.0, Isaac Lab 2.3.2, torch 2.7.0+cu128, RSL-RL 5.0.1, SKRL 2.1.0, Warp 1.14) | ✅ Consistent with the fork's pins |
| CUDA from Python (torch, Warp, raw `cuInit`) | ✅ Works on all 8 GPUs, peer access OK |
| **PhysX GPU simulation (as installed)** | ❌ **Broken.** Falls back to CPU, then hangs on the first step |
| PhysX GPU simulation with a per-process workaround | ✅ Works (GPU dynamics + GPU broadphase) |
| GPU-parallel RL training with the workaround | ✅ Cartpole, 4096 envs: ~225k steps/s, converges |
| Warp ray caster (LiDAR pattern), headless, no rendering | ✅ 1024 envs × 5,760 rays: ~365 Mrays/s |
| **RTX rendering (cameras, RTX LiDAR, video recording, livestream)** | ❌ **Broken.** Vulkan can't see the NVIDIA GPUs |
| Isaac Lab fork customizations | ✅ No functional changes |
| NVIDIA cloud asset server | ✅ Reachable (HTTP 200) |

**Root cause:** on **2026-10-08** the system NVIDIA driver was upgraded from **550.144 to 580.178.04** (with kernel 6.11 to 7.0). The upgrade left both the 550 and 580 user-space packages installed, which caused two separate faults. Everything worked under driver 550 in June. The Kit logs show 28 successful June runs, including camera RL, MP4 recording and WebRTC livestreaming.

Neither fault is caused by Isaac Lab, by the conda environment or by your code.

---

## 2. Problems found

### P1: Damaged driver install means PhysX can't load CUDA (causes the reported warnings)

* `dpkg -V libnvidia-compute-580` shows that files the 580 package should install are **missing**:
  `/usr/lib/x86_64-linux-gnu/libcuda.so`, `libnvidia-ml.so`, `libnvidia-nvvm.so`, `libnvidia-ptxjitcompiler.so`, `/etc/OpenCL/vendors/nvidia.icd`. Other 580 packages are also missing files, including `libnvcuvid.so` and `/usr/share/glvnd/egl_vendor.d/10_nvidia.json`.
  This happens when two driver branches (550 and 580) are installed side by side and the package manager deletes files they share.
* PyTorch and Warp load `libcuda.so.1`, which is present, so they work. The `omni.physx` plugin looks up the unversioned `libcuda.so`. The Kit log shows:
  `omni.physx handle on CUDA lib is (nil)` → `PhysX warning: GPU solver pipeline failed, switching to software` → `GPU Bp pipeline failed`.
* Isaac Lab still configures the scene for the GPU pipeline (`device=cuda:0`, `suppressReadback=True`, GPU broadphase). With PhysX now on the CPU fallback, the **first `fetch_results()` call never returns**. A Python stack dump shows the hang at `isaacsim.core.simulation_manager … simulation_manager.py:237` (`initialize_physics`). This is the "simulation did not progress" symptom.
* **Proof:** I ran the same script with `LD_LIBRARY_PATH` pointing to a folder containing only `libcuda.so → /usr/lib/x86_64-linux-gnu/libcuda.so.1`. The CUDA handle was valid, there were no fallback warnings, and 1000 steps ran in 0.54 s.

### P2: NVIDIA Vulkan driver unavailable to this user, so all RTX rendering fails

* Kit: `[gpu.foundation.plugin] No device could be created`. The probe in this folder (`vkprobe.py`) confirms that the Vulkan loader rejects the NVIDIA driver (`Could not get 'vkCreateInstance' … for ICD libGLX_nvidia.so.0`) and only sees `llvmpipe`, the CPU software renderer.
* `strace` shows the 580 Vulkan driver opening `/dev/dri/renderD128–135` and getting **`EACCES`**. Those nodes are `root:render 0660`. Your account is not in `render` or `video`. The `+` ACL is the `uaccess` rule, which only covers users logged in at the physical console, not SSH sessions. `nvidia_drm modeset=1` is set.
* Effect: `--enable_cameras`, `--video`, `--livestream`, TiledCamera/Camera sensors and RTX LiDAR all hang or fail. A launch with `--enable_cameras` hangs at `sim.reset()` even with the P1 workaround.
* Confidence: high but not fully confirmed. Confirming it needs root access (see C2). The packaging damage from P1 (for example the missing EGL vendor file) is a secondary suspect, and fix C1 covers it.

### Minor observations (non-blocking)

* **`isaaclab.sh` + `timeout`:** killing `isaaclab.sh` leaves the child `python` process running at 100% CPU. Today's hung run had to be killed by hand. Check with `ps -u $USER | grep python` after any aborted run.
* My standalone diagnostic scripts sometimes hang in `app.close()` after finishing, while `train.py` exits cleanly. This is probably related to P2. Re-check after the fix.
* Benign log noise: `ICpuInfo circular dependency`, `GLFW initialization failed`, `failed to open the default display`.
* **Disk:** `/md127` is **95% full (395 GB free)**. Checkpoints, videos and Kit logs (`…/site-packages/isaacsim/kit/logs`) will add up.
* Kernel 7.0 is newer than what Isaac Sim 5.1 was validated on. The 580 driver branch is the branch Isaac Sim 5.1 recommends, and the modules load correctly, so I see no evidence the kernel is a problem.

---

## 3. Fork and tutorial project review

**IsaacLab_Fork** (`origin: thiagofcm/IsaacLab_Fork`, branch `main` at upstream `c2277524`, 2026-06-02, v2.3.2 / `isaaclab` 0.54.4):
* The only tracked change is an **accidental edit in the copyright comment** on line 1 of `scripts/reinforcement_learning/rsl_rl/train.py` (`CONTRIBUTORSagent_cfg.md`). It has no functional effect. Revert it with `git checkout -- scripts/reinforcement_learning/rsl_rl/train.py`.
* Untracked `${data}/` folder (~180 MB of `NvStreamer-*.etli` traces) from the June livestream sessions. The literal `${data}` token was never expanded. Safe to delete or add to `.gitignore`. There's also an untracked `camera_frames/` folder.
* `apps/*.kit` files are unmodified and already point the asset root to the 5.1 S3 bucket.

**isaac_lab_tutorial** (external project template):
* Jetbot direct-RL task (modified from the template), plus an untracked `jetbot_camera` task with SKRL camera train/play scripts, a custom `robots/jetbot.py`, and logs, checkpoints and MP4s from 4–8 June. These confirm the stack worked end to end under driver 550.
* `possibe_solution_for_streaming.txt` describes copying Isaac Sim's streaming `.kit` to set the asset root. That's only needed when launching raw Isaac Sim kits. Isaac Lab's own kits (used by `--livestream`) already set the asset root.
* A tracked `__pycache__/*.pyc` file is modified. Add `__pycache__/` to `.gitignore`.
* `note_for_raw_isaacsim_launching` uses `omni.services.livestream.nvcf`, which is NVIDIA's cloud-function streaming. It isn't needed for local WebRTC streaming.

---

## 4. Corrective actions (proposed, not applied)

**Needs a sysadmin (root):**

* **C1. Clean up the driver (fixes P1, likely part of P2).** Remove the leftover 550 user-space packages and reinstall 580 so the missing files come back:
  ```bash
  sudo apt purge 'nvidia-driver-550' 'libnvidia-*-550' 'nvidia-*-550' 'xserver-xorg-video-nvidia-550' 'linux-modules-nvidia-550-*'
  sudo apt install --reinstall nvidia-driver-580 libnvidia-compute-580 libnvidia-gl-580 libnvidia-decode-580 libnvidia-extra-580 nvidia-kernel-common-580
  dpkg -V $(dpkg -l | awk '/^ii/ && /nvidia/ {print $2}')   # should print nothing
  ```
  The admin should review the purge list against what else the server uses. A reboot is advisable.
* **C2. Give SSH users render-node access (fixes P2).** Either `sudo usermod -aG render,video <user>` (takes effect at next login) or a udev rule making `/dev/dri/renderD*` mode `0666`, which is common on shared GPU compute servers. Then confirm with `python vkprobe.py`, which should list 8 × "NVIDIA RTX A6000".

**User-level, no root (temporary):**

* **W1. CUDA shim for P1.** Restores GPU physics and headless training right away:
  ```bash
  mkdir -p ~/isaac_lab/phase1_diagnostics/libshim
  ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 ~/isaac_lab/phase1_diagnostics/libshim/libcuda.so
  export LD_LIBRARY_PATH=~/isaac_lab/phase1_diagnostics/libshim${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
  ```
  Run this per shell, or (if you agree) add it to `env_isaaclab`'s `etc/conda/activate.d/`. Remove it once C1 is done.
* There is **no user-level workaround for P2**. Rendering stays unavailable until C2 is done.

---

## 5. Headless development: how it works and what it means for this project

**What runs where.** `--headless` loads `apps/isaaclab.python.headless.kit`: no window and no RTX renderer. PhysX runs on the GPU and exchanges state with PyTorch through GPU tensors (`suppressReadback`), so nothing is copied back to the CPU. Isaac Lab's `RayCaster` / `MultiMeshRayCaster` are **Warp/CUDA kernels that ray-cast against meshes** and need no rendering at all. Rendering is only loaded when you ask for it with `--enable_cameras` (→ `headless.rendering.kit`), `--video`, or `--livestream`.

**A GUI or stream is not needed for training.** Physics, rewards, observations, ray-cast sensors and PPO updates all run without a display. This is the normal mode for large-scale RL.

**GPU ray casting / LiDAR without rendering works.** Verified here: 1024 envs × 16-channel 360° LiDAR, 5.9 M rays per step, ~62 sim+sensor steps/s (~365 Mrays/s), all on `cuda:0`.

**Limitations to plan for (Mirrorcle SyMPL work):**
* **Warp ray caster (recommended for training):** geometric hits and distances only. No intensity or reflectivity, materials, beam divergence or multi-return. Add noise and dropout models yourself. `RayCaster` handles a single static mesh. Use **`MultiMeshRayCaster`** for multiple or moving obstacles. Patterns are just ray-direction tensors, so a **programmable, time-varying scan pattern** (spatiotemporal adaptive sensing) maps onto it naturally. Phase 3 will need a custom pattern or sensor class.
* **RTX LiDAR** (Isaac Sim `RtxLidar`): more physically realistic, but **requires rendering** (blocked until C2) and costs far more per env. Use it only for validation and sim-to-real fidelity checks, not for massively parallel training.
* RGB/depth cameras always require rendering.
* `debug_vis` markers only appear in rendered output (viewport, livestream, video).

**Visual inspection options (all require C2 except option 4):**
1. **Video recording:** `train.py`/`play.py --video --video_length N --video_interval M` writes MP4s to `logs/<lib>/<exp>/<run>/videos/`. Best fit for a headless server. Worked in June.
2. **WebRTC livestream:** `--livestream 2`, then connect with the *Isaac Sim WebRTC Streaming Client* on your laptop. It needs the server's signaling (TCP 49100) and media (UDP 47998) ports reachable. An SSH `-L` tunnel only carries TCP, so use the campus VPN or direct access. Use for interactive debugging with few envs. Worked in June (the NvStreamer traces).
3. **Remote desktop (VNC/NoMachine) + Isaac Sim GUI:** possible, but heavy. X11 forwarding won't work for Vulkan.
4. **Offline visualization (works now):** save poses, LiDAR hit points and actions to `.npz`/HDF5 during `play.py`, then plot locally (matplotlib, Open3D, rerun, Foxglove). Training curves: TensorBoard over `ssh -L 6006:localhost:6006` (or W&B, already installed).

**Performance cost of visualization:**
* `--enable_cameras` loads the RTX renderer for the **whole run**: extra GPU memory, longer startup, and per-step render cost, even if no camera is used.
* `--video` renders only during recording windows, but still needs the rendering kit. Cost grows with resolution and the number of visible envs.
* Livestream renders every frame, which is a large slowdown. Use 1–16 envs.
* Kit sets `/renderer/multiGpu/enabled=True` and claims **all 8 GPUs** when rendering. On a shared server, pin a run with `CUDA_VISIBLE_DEVICES=N` (or `--device cuda:N`). `--distributed` with `torch.distributed.run` is available for multi-GPU training.

**Assessment:** training fully headless and using visualization only for debugging, environment validation and policy evaluation is the **right workflow**, and it's how Isaac Lab is meant to be used at scale. Recommended loop:
1. Train headless, physics + Warp LiDAR only, large `num_envs`, one GPU per experiment, TensorBoard for monitoring.
2. Evaluate with `play.py --num_envs 4–16 --video` (MP4s), plus logged LiDAR/pose data for offline plots.
3. Livestream only for interactive scene building or debugging.

---

## 6. Validation commands

Setup for every shell (`W1` is only needed until C1 is done):
```bash
conda activate env_isaaclab
export LD_LIBRARY_PATH=~/isaac_lab/phase1_diagnostics/libshim${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}   # W1
cd ~/isaac_lab/IsaacLab_Fork
LOG() { ls -t $CONDA_PREFIX/lib/python3.11/site-packages/isaacsim/kit/logs/Kit/Isaac-Sim/5.1/*.log | head -1; }
```

| Check | Command | Pass criterion |
|---|---|---|
| CUDA / torch / Warp | `python -c "import torch,warp as wp; print(torch.cuda.device_count()); wp.init()"` | 8 devices |
| Driver package integrity (after C1) | `dpkg -V $(dpkg -l \| awk '/^ii/ && /nvidia/ {print $2}')` | no output |
| Vulkan (after C2) | `python ~/isaac_lab/phase1_diagnostics/vkprobe.py` | 8× NVIDIA RTX A6000 |
| Empty sim, GPU physics | `timeout -k 10 180 python ~/isaac_lab/phase1_diagnostics/diag_empty.py --headless --steps 1000` | `enableGPUDynamics=True`, 1000 steps < 1 s |
| PhysX really on GPU | `grep -E 'handle on CUDA\|GPU solver' "$(LOG)"` | non-nil handle, **no** "GPU solver pipeline failed" |
| Official tutorial | `./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --headless` | runs and exits (Ctrl-C) |
| GPU-parallel RL | `python scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Cartpole-Direct-v0 --headless --num_envs 4096 --max_iterations 30` | ~2×10⁵ steps/s, reward rises |
| Headless GPU LiDAR | `timeout -k 10 240 python ~/isaac_lab/phase1_diagnostics/diag_raycaster.py --headless --num_envs 1024` | `device=cuda:0`, hundreds of Mrays/s |
| Cameras (after C2) | `python scripts/tutorials/04_sensors/run_usd_camera.py --headless --enable_cameras` | renders without hang |
| Video (after C2) | `python scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Cartpole-Direct-v0 --headless --num_envs 16 --video --video_length 300` | MP4 under `logs/rsl_rl/.../videos/play/` |
| Livestream (after C2) | `python scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Cartpole-Direct-v0 --num_envs 4 --livestream 2` | WebRTC client connects |
| Training curves | `tensorboard --logdir logs --port 6006`, with `ssh -L 6006:localhost:6006 <server>` on your laptop | curves visible |

Run training from a directory where you want `logs/` to be created. After any aborted run, check for orphaned processes: `ps -u $USER -o pid,etime,pcpu,cmd | grep python`.
