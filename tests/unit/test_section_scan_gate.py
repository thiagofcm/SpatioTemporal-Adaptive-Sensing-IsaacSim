"""STS01: section layout (AdaptiveSensing/evaluation/sections.py) and per-section scan gate
(AdaptiveSensing/evaluation/section_scan_gate.py).

  * layout from the real ray table: 46 / 47 / 46 columns -> 1,564 / 1,598 / 1,564 rays, independent of the ray order
  * every section = the literal real-robot gate (real_eval_fixed.py) at its own rate, for all 125 rate combinations
  * composite image: every column holds its section's latest accepted scan; other sections' columns never change
  * synchronized sections = ScanGate (TMP01 / TMP02 gate) bitwise, fixed and with rate changes
  * per-section set_target_fps = literal real_eval_adaptive.py, independent random actions per section
  * counters: acquisitions per section, scans accepted by >= 1 section, rays, full-scan equivalents (exact)
"""
import itertools
import os
import sys

import pytest
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))
from AdaptiveSensing.evaluation.scan_gate import ScanGate  # noqa: E402
from AdaptiveSensing.evaluation.section_scan_gate import SectionScanGate  # noqa: E402
from AdaptiveSensing.evaluation.sections import SectionLayout  # noqa: E402
from st_nav.sensors.ray_tables import RAY_ORDERS, grid_pattern  # noqa: E402
from test_scan_gate import DT, native_scan_ticks, real_adaptive_gate, real_gate  # noqa: E402

FPS = (0.2, 0.5, 1.0, 2.0, 4.0)
PERIODS = [500, 200, 100, 50, 25]
H, W = 34, 139


def layout(columns=(46, 47, 46), order="row_major_top_left"):
    _, _, _, col = grid_pattern((-14.0, 14.0), (-7.0, 7.0), W, H, "cell_center", order)
    return SectionLayout.from_grid(col, W, columns)


def st(t, n=1):
    return torch.full((n,), t * DT, dtype=torch.float32)       # float32 seconds, as env.scan_t


def img(t, n=1):
    """A native 'scan' whose every cell carries its own acquisition tick."""
    return torch.full((n, 1, H, W), float(t))


# ------------------------------------------------------------------------------------------------ layout
def test_default_layout_from_ray_table():
    L = layout()
    assert L.col_bounds == ((0, 46), (46, 93), (93, 139)) and L.names == ("left", "center", "right")
    assert L.rays == (1564, 1598, 1564) and L.total_rays == 4726
    assert abs(sum(L.weights) - 1.0) < 1e-15
    cs = L.column_section()
    assert cs.tolist() == [0] * 46 + [1] * 47 + [2] * 46


@pytest.mark.parametrize("order", RAY_ORDERS)
def test_layout_independent_of_ray_order(order):
    assert layout(order=order).rays == (1564, 1598, 1564)


def test_layout_rejects_bad_partitions():
    for cols in ((46, 47, 47), (0, 93, 46), (139,) + (0,)):
        with pytest.raises(ValueError):
            layout(cols)
    assert layout((139,)).rays == (4726,)                       # one section = the whole image


# ------------------------------------------------------------------------------------------------ fixed rates
def run_sections(hz3, ticks):
    """One env; returns the gate and per tick (accepted (S,), composite image)."""
    g = SectionScanGate(1, layout(), hz3, DT)
    g.reset(torch.tensor([0]), img(ticks[0]), st(ticks[0]))
    hist = [(torch.ones(3, dtype=torch.bool), g.last_obs.clone())]
    for t in ticks[1:]:
        before = g.last_obs.clone()
        _, fwd = g.step(img(t), st(t))
        cm = g.column_mask(fwd, 4)[0, 0, 0]                       # (W,)
        assert torch.equal(g.last_obs[..., ~cm], before[..., ~cm])  # non-accepting sections untouched
        hist.append((fwd[0].clone(), g.last_obs.clone()))
    return g, hist


def test_every_combination_matches_literal_gate_per_section():
    ticks = native_scan_ticks(350)
    L = layout()
    for hz3 in itertools.product(FPS, repeat=3):
        g, hist = run_sections(hz3, ticks)
        exp = [real_gate(ticks, round(1 / (h * DT))) for h in hz3]
        for s in range(3):
            got = [ticks[k] for k, (f, _) in enumerate(hist) if bool(f[s])]
            assert got == exp[s], (hz3, s)
            assert int(g.n_forward[0, s]) == len(exp[s])
            # composite image: section s's columns = its latest accepted scan, at every tick
            c0, c1 = L.col_bounds[s]
            for k, (_, im) in enumerate(hist):
                held = max(f for f in exp[s] if f <= ticks[k])
                assert bool((im[..., c0:c1] == held).all()), (hz3, s, k)
        union = sorted(set().union(*exp))
        assert int(g.n_any[0]) == len(union) and int(g.n_native[0]) == 281
        assert int(g.rays[0]) == sum(r * len(e) for r, e in zip(L.rays, exp))
        assert float(g.fse()[0]) == int(g.rays[0]) / 4726


def test_synchronized_sections_equal_scan_gate_bitwise():
    ticks = native_scan_ticks(350)
    for hz in FPS:
        g3 = SectionScanGate(1, layout(), hz, DT)
        g1 = ScanGate(1, hz, DT)
        for g in (g3, g1):
            g.reset(torch.tensor([0]), img(0), st(0))
        for t in ticks[1:]:
            new3, f3 = g3.step(img(t), st(t))
            new1, f1 = g1.step(img(t), st(t))
            assert bool(new3) == bool(new1) and f3[0].tolist() == [bool(f1)] * 3
            assert torch.equal(g3.last_obs, g1.last_obs)
            assert g3.next_sample[0].tolist() == [int(g1.next_sample)] * 3
            assert g3.last_forward[0].tolist() == [int(g1.last_forward)] * 3
        assert g3.n_forward[0].tolist() == [int(g1.n_forward)] * 3 and int(g3.n_any) == int(g1.n_forward)
        assert int(g3.n_native) == int(g1.n_native)
        assert float(g3.fse()) == float(int(g1.n_forward))      # FSE = frames exactly when synchronized
        assert g3.int_min[0].tolist() == [int(g1.int_min)] * 3 and g3.int_max[0].tolist() == [int(g1.int_max)] * 3


# ------------------------------------------------------------------------------------------------ rate changes
def test_independent_set_rate_matches_literal_adaptive_gate():
    ticks = native_scan_ticks(350)
    rng = torch.Generator().manual_seed(0)
    for trial in range(10):
        acts = torch.randint(0, 5, (len(ticks), 3), generator=rng)
        exp = [real_adaptive_gate(ticks, acts[:, s].tolist(), PERIODS, 25) for s in range(3)]
        g = SectionScanGate(1, layout(), 4.0, DT)
        g.reset(torch.tensor([0]), img(0), st(0))
        got = [[(False, 25)] for _ in range(3)]
        for k in range(1, len(ticks)):
            _, fwd = g.step(img(ticks[k]), st(ticks[k]))
            g.set_rate(fwd, torch.tensor([PERIODS[a] for a in acts[k].tolist()])[None])
            for s in range(3):
                got[s].append((bool(fwd[0, s]), int(g.periods[0, s])))
        assert got == exp, trial


def test_uniform_actions_equal_scan_gate_with_rate_changes():
    ticks = native_scan_ticks(350)
    rng = torch.Generator().manual_seed(1)
    acts = torch.randint(0, 5, (len(ticks),), generator=rng).tolist()
    g3, g1 = SectionScanGate(1, layout(), 4.0, DT), ScanGate(1, 4.0, DT)
    for g in (g3, g1):
        g.reset(torch.tensor([0]), img(0), st(0))
    for k in range(1, len(ticks)):
        _, f3 = g3.step(img(ticks[k]), st(ticks[k]))
        _, f1 = g1.step(img(ticks[k]), st(ticks[k]))
        c3 = g3.set_rate(f3, torch.full((1, 3), PERIODS[acts[k]]))
        c1 = g1.set_rate(f1, torch.tensor([PERIODS[acts[k]]]))
        assert f3[0].tolist() == [bool(f1)] * 3 and c3[0].tolist() == [bool(c1)] * 3
        assert torch.equal(g3.last_obs, g1.last_obs) and g3.next_sample[0].tolist() == [int(g1.next_sample)] * 3


# ------------------------------------------------------------------------------------------------ vectorization
def test_envs_independent_and_masked_reset():
    ticks = native_scan_ticks(60)
    g = SectionScanGate(2, layout(), (4.0, 1.0, 0.2), DT)
    g.reset(torch.tensor([0, 1]), img(0, 2), st(0, 2))
    for k, t in enumerate(ticks[1:], 1):
        if k == 30:                                             # env 1 restarts its episode at step 30 (6 s)
            g.reset(torch.tensor([False, True]), img(t, 2), st(t, 2))
            continue
        g.step(img(t, 2), st(t, 2))
    exp0 = [len(real_gate(ticks, p)) for p in (25, 100, 500)]
    assert g.n_forward[0].tolist() == exp0
    after = ticks[30:]                                          # env 1's new episode starts with the scan of 6 s
    fw1 = [real_gate(after, p) for p in (25, 100, 500)]
    assert g.n_forward[1].tolist() == [len(f) for f in fw1]
    assert g.last_forward[1].tolist() == [f[-1] for f in fw1]
    assert g.next_sample[1].tolist()[2] == fw1[2][-1] + 500     # 0.2 Hz section: lattice anchored at the reset scan
    assert int(g.rays[1]) == sum(r * len(f) for r, f in zip((1564, 1598, 1564), fw1))
