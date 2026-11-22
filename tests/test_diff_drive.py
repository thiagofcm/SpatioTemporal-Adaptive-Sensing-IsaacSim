"""Unit tests for the TurtleBot4 command path (CPU, no simulator)."""
import math

import torch

from st_nav.control.diff_drive import (CommandModel, CommandModelParams, DiffDriveGeometry, scale_saturation,
                                       unicycle_to_wheels)

GEO = DiffDriveGeometry(wheel_radius=0.03575, track_width=0.233, wheel_speed_max=0.305)
DT = 0.005


def run(params, v, w, n):
    cm = CommandModel(1, DT, GEO, params, device="cpu")
    cm.set_command(torch.tensor([v]), torch.tensor([w]))
    out = [tuple(x.clone() for x in cm.step()) for _ in range(n)]
    return [o[2].item() for o in out], [o[3].item() for o in out], out


def test_wheel_kinematics():
    vl, vr = unicycle_to_wheels(torch.tensor([0.2]), torch.tensor([1.0]), GEO)
    assert math.isclose(vl.item(), 0.2 - 0.1165, abs_tol=1e-6) and math.isclose(vr.item(), 0.2 + 0.1165, abs_tol=1e-6)


def test_scale_saturation_preserves_ratio_and_limit():
    v, w = scale_saturation(torch.tensor([0.22]), torch.tensor([2.0]), GEO)
    vl, vr = unicycle_to_wheels(v, w, GEO)
    assert math.isclose(max(abs(vl.item()), abs(vr.item())), 0.305, rel_tol=1e-5)
    assert math.isclose((w / v).item(), 2.0 / 0.22, rel_tol=1e-5)
    v2, w2 = scale_saturation(torch.tensor([0.1]), torch.tensor([0.5]), GEO)       # feasible: unchanged
    assert math.isclose(v2.item(), 0.1, rel_tol=1e-6) and math.isclose(w2.item(), 0.5, rel_tol=1e-6)


def test_per_channel_delay():
    v, w, _ = run(CommandModelParams(delay_s=0.15, delay_w_s=0.10), 0.1, 0.5, 40)
    assert next(i for i, x in enumerate(v) if x > 0) == 30          # 0.15 s / 5 ms
    assert next(i for i, x in enumerate(w) if x > 0) == 20          # 0.10 s / 5 ms


def test_first_order_lag():
    v, _, _ = run(CommandModelParams(tau_v_s=0.04), 0.1, 0.0, 400)
    i63 = next(i for i, x in enumerate(v) if x >= 0.1 * (1 - math.exp(-1)))
    assert abs((i63 + 1) * DT - 0.04) <= DT
    assert abs(v[-1] - 0.1) < 1e-6


def test_acceleration_limit():
    v, _, _ = run(CommandModelParams(accel_v_max=1.0), 0.15, 0.0, 60)
    dv = [b - a for a, b in zip([0.0] + v[:-1], v)]
    assert max(dv) <= 1.0 * DT * (1 + 1e-5)
    assert abs(v[29] - 0.15) < 1e-6                               # 0.15 m/s reached after 0.15 s


def test_wheel_targets_rad_per_s():
    _, _, out = run(CommandModelParams(), 0.2, 0.0, 1)
    wl, wr = out[0][0].item(), out[0][1].item()
    assert math.isclose(wl, 0.2 / 0.03575, rel_tol=1e-5) and math.isclose(wr, wl)
