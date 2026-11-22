"""The ported Z1-Z4 sampler reproduces the reference sampler bit-for-bit (golden output, no reference access needed).

tests/golden/reference_bank.npz: reference `sample_static_obstacles_params` (adaptive_sensor_policy, commit d58be3fadb)
for seeds 0..4095 (training banks up to 4,096) and 3000..3099 (evaluation protocol ref_static_c10).
"""
import os

import numpy as np

from st_nav.scenes import corridor_static_random as S

GOLD = np.load(os.path.join(os.path.dirname(__file__), "..", "golden", "reference_bank.npz"))


def test_sampler_bit_identical_to_reference():
    rows = [S.sample_static_obstacles_params(int(s)) for s in GOLD["seeds"]]
    assert np.array_equal(np.array([r["lengths"] for r in rows]), GOLD["lengths"])
    assert np.array_equal(np.array([r["thicknesses"] for r in rows]), GOLD["thicknesses"])
    assert np.array_equal(np.array([r["offsets"] for r in rows]), GOLD["offsets"])
    assert np.array_equal(np.array([[m == "hard" for m in r["pair_modes"]] for r in rows]), GOLD["hard"])
    for k in ("c12", "c34", "d_z1_z3"):
        assert np.array_equal(np.array([r[k] for r in rows]), GOLD[k]), k


def test_bank_matches_golden_and_constraints():
    for seed, k in ((0, 1024), (3000, 100)):
        bank = S.sample_bank(k, seed)
        idx = np.searchsorted(GOLD["seeds"][:4096], np.arange(seed, seed + k)) if seed == 0 else 4096 + np.arange(k)
        assert np.array_equal(bank["lengths"], GOLD["lengths"][idx])
        assert np.array_equal(bank["offsets"], GOLD["offsets"][idx])
        L = bank["lengths"]
        assert (S.CORRIDOR_WIDTH - (L[:, 0] + L[:, 1]) >= S.MIN_NOMINAL_CLEARANCE - 1e-9).all()      # constraint A
        assert (S.CORRIDOR_WIDTH - (L[:, 2] + L[:, 3]) >= S.MIN_NOMINAL_CLEARANCE - 1e-9).all()      # constraint B


def test_no_reference_import():
    """Building a bank in a fresh interpreter touches neither the reference path nor its modules."""
    import subprocess
    import sys
    code = ("import sys; from st_nav.scenes import corridor_static_random as S; S.sample_bank(64, 0); "
            "assert not any('adaptive_sensor_policy' in p for p in sys.path); "
            "assert not any(m == 'envs' or m.startswith('envs.') for m in sys.modules); print('clean')")
    r = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True)
    assert r.returncode == 0 and "clean" in r.stdout, r.stderr
