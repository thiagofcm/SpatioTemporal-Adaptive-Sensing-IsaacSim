"""Moving-obstacle motion shared by the dynamic scenes: GPU-vectorized port of the reference `random_ping_pong` motion
(adaptive_sensor_policy envs/scenes/_obstacle_motion.py, commit d58be3fadb: _reset_random_ping_pong,
_advance_random_ping_pong, _rpp_sample_target, _rpp_geometry, presence / speed / path-y randomization).

Spec (scene file, MOVING_OBSTACLES):
    {type="random_ping_pong", start=(x0, y0), end=(x1, y1), speed, reversal_start=(0, f1), reversal_end=(f2, 1)}

Per environment and obstacle (independent state and draws for every parallel env):
  * reset: path coordinate s0 ~ U(0, L), direction ~ {-1, +1}, first reversal target drawn from the region ahead,
    no pause; presence, speed and path y of the episode (when their randomization is on);
  * advance(dt): spend exactly dt: first any turnaround pause left (standing still), then motion at the episode speed;
    on reaching the target: stop there, flip direction, draw a new target from the opposite reversal region and a
    pause ~ U(OBSTACLE_TURN_PAUSE_RANGE_S); the leftover time of the step goes into that pause.
Only `random_ping_pong` is implemented (the reference's keyframe / ping_pong / triggered types are not used by OB1/OB2).

Randomness (same distributions as the reference): every draw is a counter-based random number (st_nav.crng) addressed by
the env's per-episode obstacle seed, a stream id, the obstacle index and a counter (e.g. the reversal number). An
episode's whole obstacle realization is therefore fixed at its reset by its seed, and is unaffected by the resets,
terminations or behaviour of any other env. Nothing about future motion is exposed: the env only sees positions.
"""
from __future__ import annotations

import math

import torch

from st_nav import crng

OBSTACLE_RADIUS = 0.17                     # m (reference OBSTACLE_RADIUS)
OBSTACLE_TURN_PAUSE_RANGE_S = (4.6, 5.0)   # s, turnaround pause at every reversal (reference)
# streams of the per-episode obstacle seed
_PRESENCE, _SPEED, _S0, _DIR, _TARGET, _PAUSE, _PATH_Y = range(7)
PRESENCE_ATTEMPTS = 64                     # rejection attempts for presence DR (all-absent draws are redrawn)
PATH_Y_CANDIDATES = 64                     # candidates of the path-y rejection sampler


class RandomPingPongObstacles:
    """State and motion of K random_ping_pong obstacles in N environments (env-local frame, metres, seconds)."""

    def __init__(self, specs, n: int, device, *, active=None, presence_dr=False,
                 presence_prob=None, speed_dr=False, speed_ranges=None, y_dr=False, y_sampler=None, max_episode_s=None):
        for o in specs:
            (f0, f1), (f2, f3) = o["reversal_start"], o["reversal_end"]
            if o.get("type") != "random_ping_pong" or not (o["speed"] > 0 and f0 == 0.0 and f0 <= f1 < f2 <= f3 and f3 == 1.0):
                raise ValueError(f"unsupported moving-obstacle spec {o}")
            if y_dr and o["start"][1] != o["end"][1]:
                raise ValueError("path-y randomization needs horizontal segments")
        k = len(specs)
        self.n, self.k, self.device = n, k, device
        f = lambda v: torch.tensor(v, dtype=torch.float32, device=device)  # noqa: E731
        self.a0 = f([o["start"] for o in specs])                       # (K, 2)
        self.b0 = f([o["end"] for o in specs])
        self.L = torch.linalg.norm(self.b0 - self.a0, dim=1)           # (K,), unchanged by the path-y shift
        self.rev_start = f([o["reversal_start"] for o in specs])       # (K, 2) fractions of L
        self.rev_end = f([o["reversal_end"] for o in specs])
        self.speed_cfg = f([o["speed"] for o in specs])
        active = tuple(range(1, k + 1)) if active is None else tuple(active)
        if not active or not set(active) <= set(range(1, k + 1)):
            raise ValueError(f"active obstacles must be a non-empty subset of 1..{k}, got {active}")
        if presence_dr and len(active) != k:
            raise ValueError("obstacle presence randomization replaces the fixed active set: keep all obstacles listed")
        self.fixed_active = torch.tensor([i + 1 in active for i in range(k)], device=device)
        self.presence_dr, self.speed_dr, self.y_dr = presence_dr, speed_dr, y_dr
        self.presence_prob = f(presence_prob[:k]) if presence_dr else None
        self.speed_ranges = f(speed_ranges[:k]) if speed_dr else None   # (K, 2)
        self.y_sampler = y_sampler
        z = lambda: torch.zeros(n, k, device=device)  # noqa: E731
        self.s, self.dir, self.target, self.pause = z(), torch.ones(n, k, device=device), z(), z()
        self.speed = self.speed_cfg.expand(n, k).clone()
        self.path_y = self.a0[:, 1].expand(n, k).clone()
        self.active = self.fixed_active.expand(n, k).clone()
        self.reversals = torch.zeros(n, k, dtype=torch.long, device=device)   # reversal counter (random-number index)
        self.seed = torch.zeros(n, dtype=torch.int64, device=device)          # per-episode obstacle seed
        self._j = torch.arange(k, device=device)
        # Pre-generated reversal draws: enough for any episode of max_episode_s. Consecutive reversals are at least
        # (reversal_end[0] - reversal_start[1]) * L / v_max + min pause apart, so an episode has at most 1 + T / cycle.
        v_max = float((self.speed_ranges[:, 1] if speed_dr else self.speed_cfg).max())
        cycle = float(((self.rev_end[:, 0] - self.rev_start[:, 1]) * self.L).min()) / v_max + OBSTACLE_TURN_PAUSE_RANGE_S[0]
        self.table_size = int(math.ceil(1 + (max_episode_s if max_episode_s else 600.0) / cycle)) + 1
        self._kk = torch.arange(self.table_size, device=device)
        self.u_target_tab = torch.zeros(n, k, self.table_size, device=device)   # target draw of reversal i
        self.u_pause_tab = torch.zeros(n, k, self.table_size, device=device)    # pause draw of reversal i

    # ------------------------------------------------------------------ helpers
    def _u(self, seed, stream, *counters):
        """(m, K) uniforms for the seeds (m,) and every obstacle; counters: ints or tensors broadcastable to (m, K)."""
        return crng.uniform(seed.view(-1, 1), stream, self._j.view(1, -1), *counters)

    def _sample_target(self, s, d, u):
        """Target in the part of the reversal region (toward direction d) that lies ahead of s (u ~ U[0, 1))."""
        L = self.L.expand_as(s)
        fwd = d > 0
        lo = torch.where(fwd, torch.maximum(self.rev_end[:, 0] * L, s), self.rev_start[:, 0] * L)
        hi = torch.where(fwd, self.rev_end[:, 1] * L, torch.minimum(self.rev_start[:, 1] * L, s))
        return lo + u * (hi - lo)

    # ------------------------------------------------------------------ reset / advance
    def reset(self, env_ids: torch.Tensor, seeds: torch.Tensor):
        """New episode for env_ids; seeds (len(env_ids),) int64 per-episode obstacle seeds fix the whole realization."""
        m = len(env_ids)
        if m == 0:
            return
        seeds = seeds.to(torch.int64)
        self.seed[env_ids] = seeds
        if self.y_dr:
            cand = crng.uniform(seeds.view(-1, 1, 1), _PATH_Y, torch.arange(PATH_Y_CANDIDATES, device=self.device).view(1, -1, 1),
                                torch.arange(2, device=self.device).view(1, 1, -1))
            self.path_y[env_ids, :2] = self.y_sampler(cand)
        else:
            self.path_y[env_ids] = self.a0[:, 1]
        s0 = self._u(seeds, _S0) * self.L
        d0 = torch.where(self._u(seeds, _DIR) < 0.5, 1.0, -1.0)
        self.s[env_ids], self.dir[env_ids] = s0, d0
        self.target[env_ids] = self._sample_target(s0, d0, self._u(seeds, _TARGET, 0))
        self.pause[env_ids] = 0.0                                  # every episode starts with the obstacles moving
        jj, kk = self._j.view(1, -1, 1), self._kk.view(1, 1, -1)    # pre-generate this episode's reversal draws
        self.u_target_tab[env_ids] = crng.uniform(seeds.view(-1, 1, 1), _TARGET, jj, kk + 1)
        self.u_pause_tab[env_ids] = crng.uniform(seeds.view(-1, 1, 1), _PAUSE, jj, kk)
        self.reversals[env_ids] = 0
        if self.presence_dr:                                       # >= 1 obstacle: first attempt with any obstacle present
            att = torch.arange(PRESENCE_ATTEMPTS, device=self.device)
            act = crng.uniform(seeds.view(-1, 1, 1), _PRESENCE, att.view(1, -1, 1), self._j.view(1, 1, -1)) < self.presence_prob
            first = torch.argmax(act.any(-1).int(), 1)
            self.active[env_ids] = act[torch.arange(m, device=self.device), first]
        else:
            self.active[env_ids] = self.fixed_active
        if self.speed_dr:
            lo, hi = self.speed_ranges[:, 0], self.speed_ranges[:, 1]
            self.speed[env_ids] = lo + self._u(seeds, _SPEED) * (hi - lo)
        else:
            self.speed[env_ids] = self.speed_cfg

    def advance(self, dt: float):
        """Advance every obstacle of every env by dt (dt << turnaround pause, so at most one reversal per call)."""
        t = torch.full_like(self.s, dt)
        used = torch.minimum(self.pause, t)                        # turnaround pause: no motion
        self.pause -= used
        t -= used
        moving = (t > 0) & self.active
        gap = (self.target - self.s).abs()
        reach = moving & (self.speed * t >= gap)
        go = moving & ~reach
        self.s = torch.where(go, self.s + self.dir * self.speed * t, self.s)
        # reversal (masked, no host synchronization): stop at the target, flip, next target and pause from the episode's
        # pre-generated draws of reversal k, leftover step time spent in the pause
        t_left = t - gap / self.speed
        k = self.reversals.clamp(max=self.table_size - 1).unsqueeze(-1)
        u_t = self.u_target_tab.gather(2, k).squeeze(-1)
        u_p = self.u_pause_tab.gather(2, k).squeeze(-1)
        self.s = torch.where(reach, self.target, self.s)
        self.dir = torch.where(reach, -self.dir, self.dir)
        self.target = torch.where(reach, self._sample_target(self.s, self.dir, u_t), self.target)
        lo, hi = OBSTACLE_TURN_PAUSE_RANGE_S
        self.pause = torch.where(reach, lo + u_p * (hi - lo) - t_left, self.pause)
        self.reversals += reach.long()

    # ------------------------------------------------------------------ geometry
    def positions(self) -> torch.Tensor:
        """(N, K, 2) obstacle centres (env-local); inactive obstacles keep their (unused) path position."""
        a = torch.stack([self.a0[:, 0].expand_as(self.s), self.path_y], -1)
        b = torch.stack([self.b0[:, 0].expand_as(self.s), self.path_y + (self.b0[:, 1] - self.a0[:, 1])], -1)
        return a + (b - a) * (self.s / self.L)[..., None]

    def circles(self, radius: float = OBSTACLE_RADIUS) -> torch.Tensor:
        """(N, K, 3) [cx, cy, r] for the analytic geometry; inactive obstacles have r = 0 (ignored)."""
        r = torch.where(self.active, torch.full_like(self.s, radius), torch.zeros_like(self.s))
        return torch.cat([self.positions(), r[..., None]], -1)
