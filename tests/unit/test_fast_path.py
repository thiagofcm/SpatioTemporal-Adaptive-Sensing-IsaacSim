"""Env fast path (st_nav.envs.corridor_scene.FAST_PATH_ITEMS), simulator-free parts: every cached / precomputed value
must equal the original computation bit for bit. Simulator parts (lazy contact sensor, joint indices, obstacle pose
buffers, whole-env traces): tests/sim/fast_path_equivalence.py."""
import torch

from st_nav import crng
from st_nav.control.diff_drive import CommandModel, CommandModelParams, DiffDriveGeometry
from st_nav.scenes import _common as SC
from st_nav.scenes import corridor_dynamic_chase_random as DYN

GEO = DiffDriveGeometry(wheel_radius=0.03575, track_width=0.233, wheel_speed_max=0.305)


def test_rng_key_cache_bitwise():
    seeds = torch.randint(-2 ** 62, 2 ** 62, (257,), dtype=torch.int64)
    ray = torch.arange(4726).view(1, -1)
    cases = [lambda: crng.key(7, seeds, 3), lambda: crng.uniform(seeds.view(-1, 1), 12, ray, 0),
             lambda: crng.uniform(seeds.view(-1, 1), 12, ray, 1), lambda: crng.key(seeds, crng.OBSTACLES),
             lambda: crng.key(2 ** 63 + 5, seeds), lambda: crng.key(0, 1, 2)]
    crng.set_key_cache(False)
    ref = [f() for f in cases]
    crng.set_key_cache(True)
    try:
        for _ in range(2):                                   # first call fills the cache, second uses it
            for f, r in zip(cases, ref):
                assert torch.equal(f(), r)
    finally:
        crng.set_key_cache(False)


def _cm_pair(n, params, **set_params):
    a = CommandModel(n, 0.01, GEO, params, device="cpu")
    b = CommandModel(n, 0.01, GEO, params, device="cpu")
    b.enable_constant_cache()
    if set_params:
        a.set_params(**set_params)
        b.set_params(**set_params)
    return a, b


def test_cm_constants_bitwise():
    g = torch.Generator().manual_seed(0)
    for params, sp in ((CommandModelParams(), {}),
                       (CommandModelParams(delay_s=0.05, tau_v_s=0.08, tau_w_s=0.03, accel_v_max=1.0, accel_w_max=0.0),
                        {}),
                       (CommandModelParams(), dict(env_ids=torch.tensor([1, 3]), tau_v_s=0.0, accel_v_max=2.5,
                                                   tau_w_s=torch.tensor([0.07, 0.02]), k_v=1.01))):
        a, b = _cm_pair(5, params, **sp)
        for k in range(400):
            if k % 20 == 0:
                v, w = torch.rand(5, generator=g) * 0.5 - 0.1, torch.rand(5, generator=g) * 4 - 2
                a.set_command(v, w)
                b.set_command(v, w)
            if k == 200:                                     # parameters changed mid-run: the cache follows
                a.set_params(tau_w_s=0.11, accel_w_max=3.0)
                b.set_params(tau_w_s=0.11, accel_w_max=3.0)
            for x, y in zip(a.step(), b.step()):
                assert torch.equal(x, y)


def test_reset_constants_bitwise():
    for dr in (False, True):
        ga, gb = torch.Generator().manual_seed(5), torch.Generator().manual_seed(5)
        cache = {}
        for _ in range(3):
            assert torch.equal(SC.sample_start(17, dr, "cpu", ga), SC.sample_start(17, dr, "cpu", gb, cache=cache))
    goals = ((-1.8, -1.7), (-2.0, -1.8), (-2.0, -2.0))
    ga, gb, cache = torch.Generator().manual_seed(9), torch.Generator().manual_seed(9), {}
    for _ in range(3):
        assert torch.equal(SC.sample_goal(11, goals, "cpu", ga), SC.sample_goal(11, goals, "cpu", gb, cache=cache))
    u = torch.rand(40, 64, 2, generator=torch.Generator().manual_seed(1))
    u[0] = 0.999                                             # no feasible candidate: fallback row
    cache = {}
    for _ in range(2):
        assert torch.equal(DYN.sample_obstacle_path_y(u), DYN.sample_obstacle_path_y(u, cache=cache))
