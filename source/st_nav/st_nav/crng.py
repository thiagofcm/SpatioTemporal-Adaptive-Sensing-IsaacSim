"""Counter-based random numbers (batched torch, GPU or CPU, bit-identical on both).

A random number is a pure function of integer keys, u = uniform(key_0, key_1, ...): no generator state, so
  * every environment / episode / stream gets its own reproducible sequence (no sharing between parallel envs: a reset
    or termination elsewhere cannot shift it),
  * a value can be addressed directly by what it is for (e.g. obstacle j, reversal k; or beam b acquired at physics tick
    t), so the same measurement always gets the same noise, whatever else was sampled.

Hash: SplitMix64 finalizer (Steele et al. 2014) on int64 tensors (multiplication wraps modulo 2^64 in torch); keys are
combined by iterated hashing. uniform() uses the top 24 bits -> exact float32 in [0, 1).
"""
from __future__ import annotations

import torch

_M64 = (1 << 64) - 1


def _s(c: int) -> int:                       # unsigned 64-bit constant -> signed int64 value
    c &= _M64
    return c - (1 << 64) if c >= (1 << 63) else c


_GOLDEN, _C1, _C2 = _s(0x9E3779B97F4A7C15), _s(0xBF58476D1CE4E5B9), _s(0x94D049BB133111EB)


def _shr(z: torch.Tensor, n: int) -> torch.Tensor:
    """Logical right shift of int64 (torch's >> is arithmetic)."""
    return (z >> n) & ((1 << (64 - n)) - 1)


def mix64(x: torch.Tensor) -> torch.Tensor:
    z = x + _GOLDEN
    z = (z ^ _shr(z, 30)) * _C1
    z = (z ^ _shr(z, 27)) * _C2
    return z ^ _shr(z, 31)


def key(*parts) -> torch.Tensor:
    """Hash of integer keys (tensors broadcast; python ints allowed) -> int64 tensor."""
    dev = next((p.device for p in parts if isinstance(p, torch.Tensor)), torch.device("cpu"))
    h = None
    for p in parts:
        if not isinstance(p, torch.Tensor):
            p = torch.tensor(_s(int(p)), dtype=torch.int64, device=dev)
        p = p.to(device=dev, dtype=torch.int64)
        h = mix64(p) if h is None else mix64(h ^ mix64(p))
    return h


def uniform(*parts) -> torch.Tensor:
    """U[0, 1) float32, a pure function of the integer keys (broadcast shape of the parts)."""
    return _shr(key(*parts), 40).to(torch.float32) * (1.0 / (1 << 24))


# stream identifiers (second key after the episode seed)
OBSTACLES, LIDAR_NOISE = 1, 2
