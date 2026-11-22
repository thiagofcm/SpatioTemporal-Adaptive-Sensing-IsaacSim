"""Differential-drive command path for the TurtleBot4 (batched, torch).

    policy / script (v_cmd, w_cmd) at the control rate
      -> CommandModel (runs every physics step):
           1. pure delay        theta_v, theta_w (FIFO of physics steps, per channel)
           2. gains             k_v, k_w
           3. wheel saturation  "scale": if max(|v_l|, |v_r|) > v_wheel_max both wheels are scaled by the same factor
                                (best fit to the real robot, saturation_models.csv)
           4. first-order lag   tau_v, tau_w (exact discrete form, per physics step)
           5. acceleration limit accel_v_max, accel_w_max (rate limit on the lagged velocities; firmware-like)
      -> wheel angular-velocity targets [rad/s] for the PhysX velocity drives

The command model only adds what the PhysX drive + contact dynamics do not already produce: its parameters are
fitted with the physics in the loop (infrastructure/M1_robot_calibration/scripts/m1_calibrate.py), so delay/lag are never counted twice.
Every parameter is a per-env tensor so it can be randomized per episode later.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class DiffDriveGeometry:
    wheel_radius: float = 0.03575      # m, irobot_create_description wheel.urdf.xacro
    track_width: float = 0.233         # m, wheel joints at y = +-0.1165 (create3 URDF)
    wheel_speed_max: float = 0.305     # m/s, real-robot saturation fit ("scale" model, best score)


@dataclass
class CommandModelParams:
    delay_s: float = 0.0               # pure delay of the linear-velocity channel
    delay_w_s: float | None = None     # pure delay of the angular channel (None = same as delay_s)
    tau_v_s: float = 0.0               # residual first-order lag of the linear velocity
    tau_w_s: float = 0.0               # residual first-order lag of the angular velocity
    k_v: float = 1.0
    k_w: float = 1.0
    accel_v_max: float = 0.0           # m/s^2, rate limit on the lagged linear velocity (0 = none)
    accel_w_max: float = 0.0           # rad/s^2, rate limit on the lagged angular velocity (0 = none)


def unicycle_to_wheels(v: torch.Tensor, w: torch.Tensor, geo: DiffDriveGeometry) -> tuple[torch.Tensor, torch.Tensor]:
    """(v, w) -> left/right wheel rim speeds [m/s]."""
    half = 0.5 * geo.track_width
    return v - w * half, v + w * half


def scale_saturation(v: torch.Tensor, w: torch.Tensor, geo: DiffDriveGeometry) -> tuple[torch.Tensor, torch.Tensor]:
    """Scale (v, w) jointly so that no wheel exceeds wheel_speed_max (direction of motion preserved)."""
    vl, vr = unicycle_to_wheels(v, w, geo)
    s = torch.clamp(torch.maximum(vl.abs(), vr.abs()) / geo.wheel_speed_max, min=1.0)
    return v / s, w / s


class CommandModel:
    """Batched delay -> gain -> saturation -> lag model, stepped at the physics rate."""

    def __init__(self, num_envs: int, physics_dt: float, geo: DiffDriveGeometry, params: CommandModelParams | None = None,
                 max_delay_s: float = 0.5, device: str | torch.device = "cuda:0"):
        self.n, self.dt, self.geo, self.device = num_envs, physics_dt, geo, device
        self.max_delay_steps = int(round(max_delay_s / physics_dt))
        p = params or CommandModelParams()
        self.delay_steps = torch.full((num_envs,), int(round(p.delay_s / physics_dt)), dtype=torch.long, device=device)
        dw = p.delay_s if p.delay_w_s is None else p.delay_w_s
        self.delay_steps_w = torch.full((num_envs,), int(round(dw / physics_dt)), dtype=torch.long, device=device)
        self.tau_v = torch.full((num_envs,), p.tau_v_s, device=device)
        self.tau_w = torch.full((num_envs,), p.tau_w_s, device=device)
        self.k_v = torch.full((num_envs,), p.k_v, device=device)
        self.k_w = torch.full((num_envs,), p.k_w, device=device)
        self.accel_v_max = torch.full((num_envs,), p.accel_v_max, device=device)
        self.accel_w_max = torch.full((num_envs,), p.accel_w_max, device=device)
        # FIFO of commanded (v, w); index 0 = newest
        self._fifo = torch.zeros(num_envs, self.max_delay_steps + 1, 2, device=device)
        self.state = torch.zeros(num_envs, 2, device=device)       # lagged (v, w) actually sent to the wheels
        self.cmd = torch.zeros(num_envs, 2, device=device)         # latest command (held between control ticks)
        self._ar = torch.arange(num_envs, device=device)

    def set_params(self, env_ids=None, *, delay_s=None, delay_w_s=None, tau_v_s=None, tau_w_s=None, k_v=None, k_w=None,
                   accel_v_max=None, accel_w_max=None):
        """delay_s sets both channels unless delay_w_s is also given."""
        ids = slice(None) if env_ids is None else env_ids
        for attr, val in (("delay_steps", delay_s), ("delay_steps_w", delay_w_s if delay_w_s is not None else delay_s)):
            if val is None:
                continue
            steps = torch.round(torch.as_tensor(val, device=self.device) / self.dt).long()
            if int(steps.max()) > self.max_delay_steps:
                raise ValueError(f"delay exceeds max_delay_s ({self.max_delay_steps * self.dt:.3f} s)")
            getattr(self, attr)[ids] = steps
        for name, val in (("tau_v", tau_v_s), ("tau_w", tau_w_s), ("k_v", k_v), ("k_w", k_w), ("accel_v_max", accel_v_max),
                          ("accel_w_max", accel_w_max)):
            if val is not None:
                getattr(self, name)[ids] = torch.as_tensor(val, device=self.device, dtype=torch.float32)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self._fifo[ids] = 0.0
        self.state[ids] = 0.0
        self.cmd[ids] = 0.0

    def set_command(self, v: torch.Tensor, w: torch.Tensor):
        """Latest (v, w) command; held until the next call (zero-order hold at the control rate)."""
        self.cmd[:, 0] = v
        self.cmd[:, 1] = w

    def step(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Advance one physics step. Returns (wheel_left, wheel_right) angular-velocity targets [rad/s] and the
        effective (v, w) sent to the drives."""
        self._fifo = torch.roll(self._fifo, 1, dims=1)
        self._fifo[:, 0] = self.cmd
        v_d = self._fifo[self._ar, self.delay_steps, 0]
        w_d = self._fifo[self._ar, self.delay_steps_w, 1]
        v, w = scale_saturation(self.k_v * v_d, self.k_w * w_d, self.geo)
        a_v = torch.where(self.tau_v > 0, torch.exp(-self.dt / self.tau_v.clamp(min=1e-6)), torch.zeros_like(self.tau_v))
        a_w = torch.where(self.tau_w > 0, torch.exp(-self.dt / self.tau_w.clamp(min=1e-6)), torch.zeros_like(self.tau_w))
        nv = a_v * self.state[:, 0] + (1.0 - a_v) * v
        nw = a_w * self.state[:, 1] + (1.0 - a_w) * w
        lim_v = torch.where(self.accel_v_max > 0, self.accel_v_max * self.dt, torch.full_like(nv, float("inf")))
        lim_w = torch.where(self.accel_w_max > 0, self.accel_w_max * self.dt, torch.full_like(nw, float("inf")))
        self.state[:, 0] = self.state[:, 0] + torch.clamp(nv - self.state[:, 0], -lim_v, lim_v)
        self.state[:, 1] = self.state[:, 1] + torch.clamp(nw - self.state[:, 1], -lim_w, lim_w)
        vl, vr = unicycle_to_wheels(self.state[:, 0], self.state[:, 1], self.geo)
        return vl / self.geo.wheel_radius, vr / self.geo.wheel_radius, self.state[:, 0], self.state[:, 1]
