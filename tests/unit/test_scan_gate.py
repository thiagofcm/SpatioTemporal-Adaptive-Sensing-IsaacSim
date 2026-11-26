"""FixedRateScanGate (AdaptiveSensing/evaluation/scan_gate.py) against a literal transcription of the real-robot gate
(real_eval_fixed.py `_on_raw_scan`) on the Isaac Lab native timing: 4 Hz LiDAR held between 5 Hz control steps."""
import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from AdaptiveSensing.evaluation.scan_gate import FixedRateScanGate  # noqa: E402

DT = 0.01                  # physics step
CTRL = 20                  # physics ticks per control step (5 Hz)
NATIVE = 25                # native LiDAR period in ticks (4 Hz)


def native_scan_ticks(n_steps, native=NATIVE):
    """scan_t (ticks) held after each control step k = 0..n_steps, with corridor_env's scheduler: a scan is taken at the
    first control instant on or after each lattice point; between scans the previous scan is returned again."""
    nxt, held, out = 0, None, []
    for k in range(n_steps + 1):
        t = k * CTRL
        if t >= nxt:
            held = t
            while nxt <= t:
                nxt += native
        out.append(held)
    return out


def real_gate(scan_ticks, period_ticks):
    """Literal real_eval_fixed.py gate, fed only on genuinely new scans. Returns the forwarded scan times."""
    forwarded, nst, force, last = [], None, True, None
    for now in scan_ticks:
        if now == last:
            continue                                   # no new physical scan: the callback does not run
        last = now
        if force:
            forwarded.append(now)
            force, nst = False, now + period_ticks
            continue
        if now < nst:
            continue
        forwarded.append(now)
        while nst <= now:
            nst += period_ticks
    return forwarded


def run_gate(hz, scan_ticks):
    g = FixedRateScanGate(1, hz, DT)
    img = lambda t: torch.full((1, 1, 2, 3), float(t))  # noqa: E731  (the "scan" carries its own time)
    st = lambda t: torch.tensor([t * DT], dtype=torch.float32)  # noqa: E731  (float32, as env.scan_t)
    g.reset(torch.tensor([0]), img(scan_ticks[0]), st(scan_ticks[0]))
    forwarded, held = [scan_ticks[0]], [scan_ticks[0]]
    for t in scan_ticks[1:]:
        new, fwd = g.step(img(t), st(t))
        if bool(fwd):
            forwarded.append(t)
        held.append(int(g.last_obs[0, 0, 0, 0]))
    return g, forwarded, held


def test_native_timing():
    ticks = native_scan_ticks(14)                              # steps 0..14 = 0..2.8 s
    assert sorted(set(ticks)) == [0, 40, 60, 80, 100, 140, 160, 180, 200, 240, 260, 280]


def test_matches_real_gate_all_rates():
    ticks = native_scan_ticks(350)
    for hz in (4, 2, 1, 0.5, 0.2):
        g, fwd, _ = run_gate(hz, ticks)
        assert fwd == real_gate(ticks, round(1 / (hz * DT))), hz


def test_4hz_forwards_every_native_scan():
    ticks = native_scan_ticks(350)
    g, fwd, _ = run_gate(4, ticks)
    assert fwd == sorted(set(ticks))
    assert int(g.n_native) == int(g.n_forward) == 281          # reset scan + 280


def test_expected_schedules():
    ticks = native_scan_ticks(350)
    exp = {2: ([0, 60, 100, 160, 200, 260], 141), 1: ([0, 100, 200, 300], 71), 0.5: ([0, 200, 400], 36),
           0.2: ([0, 500, 1000], 15)}
    for hz, (first, n) in exp.items():
        g, fwd, _ = run_gate(hz, ticks)
        assert fwd[:len(first)] == first and len(fwd) == n and int(g.n_forward) == n, hz
        assert int(g.n_native) == 281                          # repeats never counted


def test_hold_between_forwards():
    ticks = native_scan_ticks(100)
    _, fwd, held = run_gate(1, ticks)
    for k, h in enumerate(held):                               # held scan = latest forwarded scan at or before step k
        assert h == max(f for f in fwd if f <= ticks[k])


def test_intervals_and_reset():
    ticks = native_scan_ticks(350)
    g, _, _ = run_gate(2, ticks)
    assert (int(g.int_min), int(g.int_max)) == (40, 60)
    g.reset(torch.tensor([0]), torch.zeros(1, 1, 2, 3), torch.zeros(1))
    assert int(g.n_forward) == 1 and int(g.n_native) == 1 and int(g.next_sample) == 50


def test_vectorized_envs_independent():
    ticks = native_scan_ticks(60)
    g = FixedRateScanGate(2, 1, DT)
    g.reset(torch.tensor([0, 1]), torch.zeros(2, 1), torch.zeros(2))
    for k, t in enumerate(ticks[1:], 1):
        st = torch.tensor([t * DT, (t if k < 30 else ticks[k] - ticks[30]) * DT])
        if k == 30:                                            # env 1 restarts its episode at step 30
            g.reset(torch.tensor([1]), torch.zeros(2, 1), torch.zeros(2))
            continue
        g.step(torch.zeros(2, 1), st)
    assert int(g.n_forward[0]) == 13 and int(g.n_native[0]) == 49   # 0..12 s: reset + 12; 49 native scans
    assert int(g.n_forward[1]) == 7 and int(g.n_native[1]) == 25    # new episode at 6 s: reset + 1..6 s; 6 s of native scans


# ------------------------------------------------------------------------------------------------ adaptive rate (TMP02)
def real_adaptive_gate(scan_ticks_per_tick, actions, periods, reset_period):
    """Literal real_eval_adaptive.py: `_on_raw_scan` on genuinely new scans + `set_target_fps(requested)` at the first
    forwarded scan of the decision's window. Here the window of decision k holds the scan of instant k (if new).
    Returns per tick (forwarded, period in force after the tick)."""
    period, nst, last_fwd, last = reset_period, None, None, None
    out = []
    for k, (now, a) in enumerate(zip(scan_ticks_per_tick, actions)):
        fwd = False
        if k == 0:                                     # reset scan: force_next_fresh
            last, last_fwd, nst = now, now, now + period
            out.append((False, period))
            continue
        if now != last:                                # the callback runs on new physical scans only
            last = now
            if now >= nst:
                last_fwd, fwd = now, True
                while nst <= now:
                    nst += period
        if fwd:                                        # set_target_fps(requested_fps)
            req = periods[a]
            if req != period:
                period = req
                nst = last_fwd + period
        out.append((fwd, period))
    return out


def test_set_rate_matches_real_adaptive_gate():
    from AdaptiveSensing.evaluation.scan_gate import ScanGate
    periods = [500, 200, 100, 50, 25]                  # 0.2, 0.5, 1, 2, 4 Hz
    ticks = native_scan_ticks(350)
    g_rng = torch.Generator().manual_seed(0)
    for trial in range(20):
        acts = torch.randint(0, 5, (len(ticks),), generator=g_rng).tolist()
        exp = real_adaptive_gate(ticks, acts, periods, 25)
        g = ScanGate(1, 4, DT)
        st = lambda t: torch.tensor([t * DT], dtype=torch.float32)  # noqa: E731
        g.reset(torch.tensor([0]), torch.zeros(1, 1), st(ticks[0]))
        got = [(False, 25)]
        for k in range(1, len(ticks)):
            _, fwd = g.step(torch.zeros(1, 1), st(ticks[k]))
            g.set_rate(fwd, torch.tensor([periods[acts[k]]]))
            got.append((bool(fwd), int(g.periods[0])))
        assert got == exp, trial


def test_constant_rate_equals_fixed_gate():
    """set_rate with the rate in force never changes the schedule (the TMP01 fixed-rate gate)."""
    from AdaptiveSensing.evaluation.scan_gate import ScanGate
    ticks = native_scan_ticks(350)
    for hz in (4, 2, 1, 0.5, 0.2):
        g = ScanGate(1, hz, DT)
        st = lambda t: torch.tensor([t * DT], dtype=torch.float32)  # noqa: E731
        g.reset(torch.tensor([0]), torch.zeros(1, 1), st(0))
        fwd_all = [0]
        for t in ticks[1:]:
            _, fwd = g.step(torch.zeros(1, 1), st(t))
            assert not bool(g.set_rate(fwd, torch.tensor([g.period])))
            if bool(fwd):
                fwd_all.append(t)
        assert fwd_all == real_gate(ticks, round(1 / (hz * DT))), hz


def test_rate_change_reanchors_and_reset_restores():
    from AdaptiveSensing.evaluation.scan_gate import ScanGate
    g = ScanGate(2, 4, DT)
    g.reset(torch.tensor([0, 1]), torch.zeros(2, 1), torch.zeros(2))
    _, fwd = g.step(torch.zeros(2, 1), torch.tensor([0.4, 0.4]))           # both forward at tick 40
    assert fwd.tolist() == [True, True]
    ch = g.set_rate(torch.tensor([True, False]), torch.tensor([100, 100]))  # env 1: action not effective
    assert ch.tolist() == [True, False] and g.periods.tolist() == [100, 25]
    assert g.next_sample.tolist() == [140, 50]                             # re-anchored at the forwarded scan
    g.reset(torch.tensor([True, False]), torch.zeros(2, 1), torch.zeros(2))
    assert g.periods.tolist() == [25, 25] and g.next_sample.tolist() == [25, 50]
