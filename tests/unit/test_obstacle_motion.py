"""Moving obstacles (random_ping_pong, corridor_dynamic_chase_random): vectorized motion vs a scalar transcription of the
reference _advance_random_ping_pong loop, reversal regions, pauses, speed, presence / speed / path-y randomization and
independent per-env resets. CPU only, no simulator."""
import math

import numpy as np
import torch

from st_nav import crng
from st_nav.scenes import corridor_dynamic_chase_random as D
from st_nav.scenes._obstacle_motion import OBSTACLE_TURN_PAUSE_RANGE_S, RandomPingPongObstacles

DT = 0.01


def seeds(ids, base=0, episode=1):
    return crng.key(base, ids.to(torch.int64), episode)


def reset(ob, ids, base=0, episode=1):
    ob.reset(ids, seeds(ids, base, episode))


def make(n=8, seed=0, **kw):
    kw.setdefault("speed_dr", True)
    kw.setdefault("speed_ranges", D.OBSTACLE_SPEED_RANGES)
    kw.setdefault("y_dr", True)
    kw.setdefault("y_sampler", D.sample_obstacle_path_y)
    ob = RandomPingPongObstacles(D.MOVING_OBSTACLES, n, "cpu", **kw)
    ob._base = seed
    return ob


def test_matches_scalar_reference_loop():
    """Scalar transcription of the reference loop (float64), fed the targets / pauses drawn by the vectorized model."""
    ob = make()
    reset(ob, torch.arange(ob.n), ob._base)
    S = ob.s.double().numpy().copy(); Dir = ob.dir.double().numpy().copy(); T = ob.target.double().numpy().copy()
    P = np.zeros_like(S); V = ob.speed.double().numpy()
    max_err = 0.0
    for _ in range(int(120 / DT)):
        ob.advance(DT)
        for e in range(ob.n):
            for i in range(ob.k):
                time_left = DT
                while time_left > 0.0:                               # reference _advance_random_ping_pong
                    if P[e, i] > 0.0:
                        used = min(P[e, i], time_left); P[e, i] -= used; time_left -= used
                        continue
                    gap = abs(T[e, i] - S[e, i])
                    if V[e, i] * time_left < gap:
                        S[e, i] += Dir[e, i] * V[e, i] * time_left
                        break
                    time_left -= gap / V[e, i]
                    S[e, i] = T[e, i]; Dir[e, i] = -Dir[e, i]
                    T[e, i] = float(ob.target[e, i])                 # the vectorized model's draw
                    P[e, i] = float(ob.pause[e, i]) + time_left      # its pause draw U (pause left = U - leftover)
        max_err = max(max_err, float(np.abs(S - ob.s.double().numpy()).max()))
        S = ob.s.double().numpy().copy()                             # resync float32 rounding, keep events
        assert np.array_equal(Dir, ob.dir.double().numpy())
    assert max_err < 1e-5, max_err
    assert int(ob.reversals.sum()) > 2 * ob.n * ob.k                 # many reversals exercised


def test_reversals_pauses_and_speed():
    ob = make(n=256, seed=1)
    reset(ob, torch.arange(ob.n), ob._base)
    L = ob.L
    prev_s, prev_dir = ob.s.clone(), ob.dir.clone()
    pause_len = torch.zeros_like(ob.s)
    pauses, rev_fwd, rev_bwd, step_speed = [], [], [], []
    for _ in range(int(150 / DT)):
        ob.advance(DT)
        flipped = ob.dir != prev_dir
        if flipped.any():
            s = ob.s[flipped] / L.expand_as(ob.s)[flipped]
            rev_fwd.extend(s[prev_dir[flipped] > 0].tolist())
            rev_bwd.extend(s[prev_dir[flipped] < 0].tolist())
        moving_whole_step = (~flipped) & (ob.pause == 0) & (pause_len == 0)
        step_speed.append(((ob.s - prev_s).abs() / DT)[moving_whole_step] / ob.speed[moving_whole_step])
        still = (ob.s == prev_s) & ~flipped
        pause_len = torch.where(still, pause_len + DT, pause_len)
        ended = (~still) & (pause_len > 0)
        pauses.extend(pause_len[ended].tolist())
        pause_len = torch.where(ended, torch.zeros_like(pause_len), pause_len)
        assert ((ob.s >= -1e-5) & (ob.s <= L + 1e-5)).all()
        prev_s, prev_dir = ob.s.clone(), ob.dir.clone()
    rev_fwd, rev_bwd = np.array(rev_fwd), np.array(rev_bwd)
    assert rev_fwd.min() >= 0.75 - 1e-5 and rev_fwd.max() <= 1 + 1e-5          # reversal_end region
    assert rev_bwd.min() >= -1e-5 and rev_bwd.max() <= 0.25 + 1e-5               # reversal_start region
    p = np.array(pauses)                                                          # standing-still spans (step-quantized)
    lo, hi = OBSTACLE_TURN_PAUSE_RANGE_S
    assert p.min() >= lo - 2 * DT and p.max() <= hi + 2 * DT and len(p) > 500, (p.min(), p.max(), len(p))
    sp = torch.cat(step_speed)
    assert (sp - 1).abs().max() < 1e-3                                           # constant episode speed while moving


def test_randomization_and_independent_resets():
    ob = make(n=4096, seed=2)
    reset(ob, torch.arange(ob.n), ob._base)
    sp = ob.speed
    assert sp.min() >= 0.04 and sp.max() <= 0.08 and sp.std() > 0.01
    y = ob.path_y
    gap = y[:, 0] - y[:, 1]
    assert (y[:, 0] >= 1.2).all() and (y[:, 0] <= 1.7).all() and (y[:, 1] >= 0.6).all() and (y[:, 1] <= 1.1).all()
    assert (gap >= 0.55 - 1e-6).all() and (gap <= 0.65 + 1e-6).all()
    s0 = ob.s / ob.L
    assert s0.min() < 0.01 and s0.max() > 0.99 and abs(float((ob.dir > 0).float().mean()) - 0.5) < 0.05
    pos = ob.positions()                                                          # horizontal segments at path y
    assert torch.allclose(pos[..., 1], y) and (pos[..., 0] >= 1.4 - 1e-5).all() and (pos[..., 0] <= 2.4 + 1e-5).all()
    for _ in range(50):
        ob.advance(DT)
    before = {k: getattr(ob, k).clone() for k in ("s", "dir", "target", "pause", "speed", "path_y")}
    ids = torch.arange(0, ob.n, 2)
    reset(ob, ids, ob._base, episode=2)
    keep = torch.ones(ob.n, dtype=torch.bool); keep[ids] = False
    for k, v in before.items():                                                   # untouched envs unchanged
        assert torch.equal(getattr(ob, k)[keep], v[keep]), k
    assert not torch.equal(ob.s[ids], before["s"][ids])                           # reset envs redrawn


def test_presence_flags():
    for active, expect in (((1,), [True, False]), ((2,), [False, True]), ((1, 2), [True, True])):
        ob = make(n=64, active=active)
        reset(ob, torch.arange(64), ob._base)
        assert (ob.active == torch.tensor(expect)).all()
        r = ob.circles()[..., 2]
        assert ((r > 0) == torch.tensor(expect)).all()                           # inactive: r = 0 (no collision / ray)
        s_inactive = ob.s[:, ~torch.tensor(expect)].clone()
        for _ in range(100):
            ob.advance(DT)
        assert torch.equal(ob.s[:, ~torch.tensor(expect)], s_inactive)            # inactive obstacles do not move
    ob = make(n=20000, presence_dr=True, presence_prob=D.OBSTACLE_PRESENCE_PROB)
    reset(ob, torch.arange(ob.n), ob._base)
    combos = ob.active[:, 0].long() + 2 * ob.active[:, 1].long()
    assert (combos > 0).all()                                                     # never obstacle-free
    frac = torch.bincount(combos, minlength=4)[1:].float() / ob.n                 # {OB1}, {OB2}, {OB1, OB2}: 1/3 each
    assert (frac - 1 / 3).abs().max() < 0.02, frac


def run(ob, steps, resets=None):
    """x(t) of every env; resets: {step: env_ids} forced resets (new episode numbers) during the run."""
    out = []
    for t in range(steps):
        if resets and t in resets:
            ids = resets[t]
            reset(ob, ids, ob._base, episode=2 + t)
        ob.advance(DT)
        out.append(ob.positions()[..., 0].clone())
    return torch.stack(out)


def test_resets_elsewhere_do_not_affect_an_env():
    """Envs 0..63: identical trajectories whether or not the other envs are reset / terminated during the run."""
    a, b = make(n=128, seed=5), make(n=128, seed=5)
    reset(a, torch.arange(128), 5); reset(b, torch.arange(128), 5)
    others = torch.arange(64, 128)
    xa = run(a, 6000)
    xb = run(b, 6000, {t: others[torch.randperm(64, generator=torch.Generator().manual_seed(t))[:16]] for t in range(50, 6000, 97)})
    assert torch.equal(xa[:, :64], xb[:, :64])                     # bit-identical (60 s, many reversals)
    assert int(a.reversals[:64].sum()) > 200
    assert not torch.equal(xa[:, 64:], xb[:, 64:])                 # the reset envs did change


def test_same_seeds_reproduce_and_different_seeds_differ():
    a, b, c = make(n=64, seed=9), make(n=64, seed=9), make(n=64, seed=10)
    for o in (a, b, c):
        reset(o, torch.arange(64), o._base)
    xa, xb, xc = run(a, 3000), run(b, 3000), run(c, 3000)
    assert torch.equal(xa, xb) and not torch.equal(xa, xc)


def test_keyed_lidar_noise():
    from st_nav.sensors.noise import DatasheetUniformRangeNoise, DatasheetUniformRangeNoiseCfg
    nm = DatasheetUniformRangeNoise(DatasheetUniformRangeNoiseCfg(), "cpu")
    d = torch.full((256, 40), 2.0)
    hit = torch.ones_like(d, dtype=torch.bool)
    seed = crng.key(3, torch.arange(256), 1)
    tick = torch.full((256,), 40, dtype=torch.int64)
    n1 = nm(d, hit, 3.5, keys=(seed, tick))
    n2 = nm(d, hit, 3.5, keys=(seed, tick))
    assert torch.equal(n1, n2)                                         # reproducible
    sub = nm(d[10:20], hit[10:20], 3.5, keys=(seed[10:20], tick[10:20]))
    assert torch.equal(sub, n1[10:20])                                 # independent of the other envs in the batch
    n3 = nm(d, hit, 3.5, keys=(seed, tick + 20))
    assert not torch.equal(n1, n3)                                     # a new acquisition time -> new noise
    e = (n1 - d) / (0.01 * 2.0)                                        # a(2 m) = 1 %: e ~ U(-1, 1)
    assert e.abs().max() <= 1.0 + 1e-5 and abs(float(e.mean())) < 0.02 and abs(float(e.std()) - 1 / 3 ** 0.5) < 0.02
