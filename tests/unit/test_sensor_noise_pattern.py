"""CPU tests of the LiDAR noise model and beam pattern (no simulator)."""
import math

import pytest
import torch

from st_nav.sensors.noise import DatasheetUniformRangeNoise, DatasheetUniformRangeNoiseCfg


def test_bands_match_reference_selection():
    n = DatasheetUniformRangeNoise(DatasheetUniformRangeNoiseCfg(seed=0), "cpu")
    d = torch.tensor([0.5, 3.0, 3.0001, 3.5, 5.0, 6.0])
    assert torch.allclose(n.relative_accuracy(d), torch.tensor([0.01, 0.01, 0.02, 0.02, 0.02, 0.025]))


def test_noise_bounds_no_return_and_clip():
    n = DatasheetUniformRangeNoise(DatasheetUniformRangeNoiseCfg(seed=1), "cpu")
    d = torch.tensor([[1.0, 2.0, 3.49, 3.5]]).repeat(20000, 1)
    hit = torch.tensor([[True, True, True, False]]).repeat(20000, 1)
    y = n(d, hit, 3.5)
    assert (y[:, 0] - 1.0).abs().max() <= 0.01 + 1e-6
    assert (y[:, 1] - 2.0).abs().max() <= 0.02 + 1e-6
    assert y[:, 2].max() <= 3.5                     # clipped at the cap
    assert torch.all(y[:, 3] == 3.5)                # no return: unchanged
    u = (y[:, 1] - 2.0) / 0.02
    assert abs(u.mean().item()) < 0.02 and abs(u.var().item() - 1 / 3) < 0.01


def test_seed_reproducible_and_disabled():
    a = DatasheetUniformRangeNoise(DatasheetUniformRangeNoiseCfg(seed=7), "cpu")
    b = DatasheetUniformRangeNoise(DatasheetUniformRangeNoiseCfg(seed=7), "cpu")
    d = torch.full((4, 40), 1.5)
    h = torch.ones_like(d, dtype=torch.bool)
    assert torch.equal(a(d, h, 3.5), b(d, h, 3.5))
    off = DatasheetUniformRangeNoise(DatasheetUniformRangeNoiseCfg(enabled=False), "cpu")
    assert torch.equal(off(d, h, 3.5), d)


def test_beam_pattern_order():
    pat = pytest.importorskip("st_nav.sensors.patterns", reason="needs the Isaac Sim (Kit) runtime")
    BeamListPatternCfg, beam_list_pattern, uniform_planar_azimuths = (pat.BeamListPatternCfg, pat.beam_list_pattern,
                                                                     pat.uniform_planar_azimuths)
    az = uniform_planar_azimuths(40)
    assert az[0] == 0.0 and az[10] == 90.0 and az[39] == 351.0
    starts, dirs = beam_list_pattern(BeamListPatternCfg(azimuths_deg=az), "cpu")
    assert torch.allclose(dirs[0], torch.tensor([1.0, 0, 0]), atol=1e-6)        # beam 0 forward
    assert torch.allclose(dirs[10], torch.tensor([0.0, 1.0, 0]), atol=1e-6)     # beam 10 left (CCW)
    assert torch.all(starts == 0)
