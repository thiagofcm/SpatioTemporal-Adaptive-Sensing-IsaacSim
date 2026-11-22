# shellcheck shell=bash
# -----------------------------------------------------------------------------
# env.sh: session-only environment for Isaac Lab on this server.
#
#   source env.sh            # env + CUDA workaround, all GPUs visible (warns)
#   source env.sh 3          # ... and restrict this shell to physical GPU 3
#   source env.sh 2,3        # ... several GPUs (distributed training)
#   source env.sh auto       # ... and pick the GPU with the least memory in use
#
# Only affects the current shell. Nothing is written outside this folder, and
# neither the conda env nor Isaac Lab is modified. Open a new shell to undo.
# -----------------------------------------------------------------------------

ST_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ST_ROOT
export ISAACLAB_PATH="${ISAACLAB_PATH:-$HOME/isaac_lab/IsaacLab_Fork}"
export ISAAC_CONDA_ENV="${ISAAC_CONDA_ENV:-$HOME/.conda/envs/env_isaaclab}"

# --- 1. Python environment (same effect as `conda activate env_isaaclab`) ----
if [ ! -x "$ISAAC_CONDA_ENV/bin/python" ]; then
    echo "[env.sh] ERROR: conda env not found at $ISAAC_CONDA_ENV" >&2
    return 1 2>/dev/null || exit 1
fi
if [ "$(realpath -q "${CONDA_PREFIX:-/nonexistent}")" != "$(realpath "$ISAAC_CONDA_ENV")" ]; then
    export CONDA_PREFIX="$ISAAC_CONDA_ENV"
    export CONDA_DEFAULT_ENV="env_isaaclab"
    export PATH="$ISAAC_CONDA_ENV/bin:$PATH"
fi

# Project code (source/st_nav) is put on PYTHONPATH instead of being pip-installed into the shared conda env.
case ":${PYTHONPATH:-}:" in
    *":$ST_ROOT/source/st_nav:"*) ;;
    *) export PYTHONPATH="$ST_ROOT/source/st_nav${PYTHONPATH:+:$PYTHONPATH}" ;;
esac

# --- 2. CUDA library workaround -----------------------------------------------
# The driver package is missing the unversioned /usr/lib/x86_64-linux-gnu/libcuda.so,
# which PhysX needs to use the GPU. We put a symlink to the real driver
# library (libcuda.so.1) in a private folder and put that folder first on the
# library search path. It turns itself off once the admin restores the file.
# Set ST_CUDA_SHIM=0 to disable it by hand.
_st_sys_libcuda="/usr/lib/x86_64-linux-gnu/libcuda.so"
_st_shim_dir="$ST_ROOT/.shim"
if [ "${ST_CUDA_SHIM:-1}" = "0" ]; then
    _st_shim_state="disabled (ST_CUDA_SHIM=0)"
elif [ -e "$_st_sys_libcuda" ]; then
    _st_shim_state="not needed (system libcuda.so present)"
else
    mkdir -p "$_st_shim_dir"
    ln -sfn /usr/lib/x86_64-linux-gnu/libcuda.so.1 "$_st_shim_dir/libcuda.so"
    case ":${LD_LIBRARY_PATH:-}:" in
        *":$_st_shim_dir:"*) ;;
        *) export LD_LIBRARY_PATH="$_st_shim_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
    esac
    _st_shim_state="ACTIVE ($_st_shim_dir/libcuda.so -> libcuda.so.1)"
fi

# --- 3. GPU selection ----------------------------------------------------------
# PCI_BUS_ID ordering makes CUDA's numbering match nvidia-smi's.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
_st_gpu_req="${1:-}"
if [ "$_st_gpu_req" = "auto" ]; then
    _st_gpu_req="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
                   | sort -t, -k2 -n | head -1 | cut -d, -f1 | tr -d ' ')"
fi
if [ -n "$_st_gpu_req" ]; then
    _st_ngpu="$(nvidia-smi -L | wc -l)"
    _st_bad=0
    if ! [[ "$_st_gpu_req" =~ ^[0-9]+(,[0-9]+)*$ ]]; then _st_bad=1; else
        for _st_g in ${_st_gpu_req//,/ }; do [ "$_st_g" -ge "$_st_ngpu" ] && _st_bad=1; done
    fi
    if [ "$_st_bad" = 1 ]; then
        echo "[env.sh] ERROR: invalid GPU '$_st_gpu_req' (valid: 0..$((_st_ngpu-1)), a list like 2,3, or auto)" >&2
        unset _st_bad _st_g
        return 1 2>/dev/null || exit 1
    fi
    unset _st_bad _st_g
    export CUDA_VISIBLE_DEVICES="$_st_gpu_req"
fi

if [ "${ST_QUIET:-0}" != "1" ]; then
    echo "[env.sh] python      : $(command -v python)"
    echo "[env.sh] Isaac Lab   : $ISAACLAB_PATH"
    echo "[env.sh] CUDA shim   : $_st_shim_state"
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        echo "[env.sh] GPU         : physical GPU(s) $CUDA_VISIBLE_DEVICES (renumbered from cuda:0 inside the process)"
    else
        echo "[env.sh] GPU         : WARNING all GPUs visible. Use 'source env.sh <N|auto>' on this shared server." >&2
    fi
fi
unset _st_sys_libcuda _st_shim_dir _st_shim_state _st_gpu_req _st_ngpu
