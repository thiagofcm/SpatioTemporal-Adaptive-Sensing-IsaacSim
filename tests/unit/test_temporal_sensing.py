"""TemporalSensing (AdaptiveSensing/training/temporal_sensing.py) on a torch-only stand-in for the Isaac Lab env:
the corridor_env native-scan scheduler (4 Hz, period 0.25 s, scans at the 5 Hz control instants), DirectRLEnv
step / auto-reset order (rewards and outcome before the reset, returned observation = next episode's reset
observation), RslRlVecEnvWrapper outputs. Each image carries (scan tick, episode id) so attribution is checked.

Checks: decision timing (the action selected at an instant gates that instant's scan; observation = held scan before
gating, reference age counter), real-robot rate re-anchoring, discarded actions and masks, frame accounting, reward
decomposition, budget penalty, terminal tick, reset attribution, NAV05 input = most recent accepted scan."""
import os
import sys

import torch
from tensordict import TensorDict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from AdaptiveSensing.training.temporal_sensing import TemporalSensing  # noqa: E402

FPS = [0.2, 0.5, 1.0, 2.0, 4.0]
PERIODS = [500, 200, 100, 50, 25]


class FakeEnv:
    """Native LiDAR scheduler of corridor_env._get_observations; episodes end after `ep_len[i]` control steps."""

    def __init__(self, n, ep_lens, r_env=1.0):
        self.num_envs, self.device = n, "cpu"
        self.unwrapped = self
        self.step_dt, self.physics_dt, self.scan_period = 0.2, 0.01, 0.25
        self.ep_lens = [list(x) for x in ep_lens]                # episode lengths per env, consumed in order
        self.ep_time = torch.zeros(n)
        self.next_scan_t = torch.zeros(n)
        self.scan_t = torch.zeros(n)
        self.ep_len = torch.zeros(n, dtype=torch.long)
        self.ep_id = torch.zeros(n)
        self.r_env = r_env
        self.lidar = torch.zeros(n, 1, 2, 3)
        self.nav_inputs = []
        self._obs()

    def _scan(self):
        tol = 0.5 * self.physics_dt
        due = self.ep_time + tol >= self.next_scan_t
        self.scan_t = torch.where(due, self.ep_time, self.scan_t)
        self.next_scan_t = torch.where(due, self.next_scan_t + self.scan_period * torch.ceil(
            (self.ep_time + tol - self.next_scan_t) / self.scan_period).clamp(min=1), self.next_scan_t)
        tick = torch.round(self.scan_t / self.physics_dt)
        new = self.lidar.clone()
        new[:, 0, 0, 0], new[:, 0, 0, 1] = tick, self.ep_id
        self.lidar = torch.where(due.view(-1, 1, 1, 1), new, self.lidar)

    def _obs(self):
        self._scan()
        self.obs = TensorDict({"policy": torch.stack([self.ep_time, self.ep_id], 1), "lidar": self.lidar.clone()},
                              batch_size=[self.num_envs])
        return self.obs

    def get_observations(self):
        return self.obs

    def step(self, act):
        self.ep_len += 1
        self.ep_time += self.step_dt
        lim = torch.tensor([x[0] if x else 10 ** 9 for x in self.ep_lens])
        done = self.ep_len >= lim
        rew = torch.full((self.num_envs,), self.r_env) + 0.01 * self.ep_len.float()
        outcome = torch.where(done, torch.ones_like(self.ep_len), torch.zeros_like(self.ep_len))
        for i in torch.nonzero(done).flatten().tolist():           # auto-reset (DirectRLEnv: after rewards)
            self.ep_lens[i].pop(0)
            self.ep_time[i], self.next_scan_t[i], self.ep_len[i] = 0.0, 0.0, 0
            self.ep_id[i] += 1
        obs = self._obs()
        return obs, rew, done.long(), {"outcome": outcome, "time_outs": torch.zeros_like(done)}


def make(n, ep_lens, fc=0.1, budget=500, bp=15.0, reset_hz=None):
    env = FakeEnv(n, ep_lens)

    def nav(o):
        env.nav_inputs.append(o["lidar"][:, 0, 0, :2].clone())
        return torch.zeros(n, 2)

    return env, TemporalSensing(env, nav, FPS, fc, budget, bp, reset_hz=reset_hz)


def run(env, ts, actions):
    """actions: (K, N) indices. Returns per-tick records."""
    img, aux = ts.reset()
    recs = []
    for k in range(actions.shape[0]):
        rec = {"held_scan": img[:, 0, 0, 0].clone(), "held_ep": img[:, 0, 0, 1].clone(), "aux": aux.clone(),
               "now": torch.round(env.ep_time / env.physics_dt), "ep": env.ep_id.clone(),
               "native_tick": torch.round(env.scan_t / env.physics_dt), "a": actions[k].clone()}
        (img, aux), r, d, info = ts.step(actions[k])
        rec.update(r=r, d=d, mask=info["mask"], period=ts.gate.periods.clone(), nav=env.nav_inputs[-1],
                   episode=info["episode"], r_env=info["r_env"])
        recs.append(rec)
    return recs


def test_user_example_timing_1hz():
    """Last acquisition at 4.0 s with 1 Hz: the next accepted scan is at 5.0 s and the action that sets the interval
    after it is the one selected AT 5.0 s, from the scan held since 4.0 s with age counter 4 (reference: 4)."""
    K = 40
    env, ts = make(1, [[10 ** 6]], fc=0.0)
    acts = torch.full((K, 1), 4)                                   # 4 Hz until 4.0 s
    acts[20, 0] = 2                                                # selected at 4.0 s (4.0 s scan accepted): 1 Hz
    acts[21:25, 0] = 0                                             # 4.2-4.8 s: 0.2 Hz, must be discarded
    acts[25, 0] = 3                                                # selected at 5.0 s: 2 Hz
    recs = run(env, ts, acts)
    assert int(recs[20]["now"]) == 400 and bool(recs[20]["mask"]) and int(recs[20]["period"]) == 100
    for k in range(21, 25):                                        # 4.4 / 4.6 / 4.8 s native scans rejected
        assert not bool(recs[k]["mask"]) and int(recs[k]["period"]) == 100
    r5 = recs[25]
    assert int(r5["now"]) == 500 and bool(r5["mask"])              # accepted at the deadline instant
    assert int(r5["held_scan"]) == 400                             # decision sees the held 4.0 s scan
    assert abs(float(r5["aux"][0, 1]) - 4 / 25) < 1e-6             # obs_age_ratio counter 4 (reference: 4)
    assert abs(float(r5["aux"][0, 0]) - 1.0 / 4) < 1e-6            # fps_ratio of the 1 Hz in force
    assert int(r5["period"]) == 50                                 # the 5.0 s decision set 2 Hz ...
    nxt = [int(r["now"]) for r in recs[26:] if bool(r["mask"])][0]
    assert nxt == 560                                              # ... re-anchored at 5.0: first native >= 5.5 s
    assert int(r5["nav"][0, 0]) == 500                             # NAV05 uses the 5.0 s scan at 5.0 s
    assert int(recs[24]["nav"][0, 0]) == 400                       # and the held 4.0 s scan at 4.8 s


def literal_reference(native_now, new_flags, actions, reset_period=25):
    """Reference / real-robot rule, per env, per tick (window of tick k = the scan of instant k)."""
    period, nxt, last_fwd = reset_period, reset_period, 0
    out = []
    for now, new, a in zip(native_now, new_flags, actions):
        fwd = bool(new) and now >= nxt
        if fwd:
            last_fwd = now
            while nxt <= now:
                nxt += period
            req = PERIODS[a]
            if req != period:
                period, nxt = req, last_fwd + req
        out.append((fwd, period))
    return out


def test_random_actions_against_literal_rule_and_accounting():
    n, K = 6, 900
    lens = [[37, 350, 121], [350, 350, 350], [5, 7, 300], [100, 2, 1, 400], [349, 351], [60] * 20]
    env, ts = make(n, lens, fc=0.25)
    acts = torch.randint(0, 5, (K, n), generator=torch.Generator().manual_seed(3))
    recs = run(env, ts, acts)
    for i in range(n):
        # split ticks into episodes, check each against the literal rule from its reset
        start = 0
        for k in range(K):
            if bool(recs[k]["d"][i]) or k == K - 1:
                seg = recs[start:k + 1]
                nat = [int(r["native_tick"][i]) for r in seg]
                prev = [0] + nat[:-1]
                new = [nat[j] != prev[j] for j in range(len(seg))]
                exp = literal_reference(nat, new, [int(r["a"][i]) for r in seg])
                got = [(bool(r["mask"][i]), int(r["period"][i])) for r in seg]
                if bool(recs[k]["d"][i]):                          # period after the terminal tick = reset rate
                    got[-1] = (got[-1][0], exp[-1][1])
                assert got == exp, (i, start)
                ep = recs[k]["episode"]
                if bool(recs[k]["d"][i]):
                    nmask = sum(float(r["mask"][i]) for r in seg)
                    assert int(ep["frames"][i]) == 1 + nmask       # reset scan + accepted scans incl. the last tick
                    assert int(ep["length"][i]) == len(seg)
                    assert int(ep["causal"][i]) == nmask
                    exp_ret = sum(float(r["r_env"][i]) - 0.25 * float(r["mask"][i]) for r in seg)
                    assert abs(float(ep["return"][i]) - exp_ret) < 1e-3
                    # next tick: reset observation of the NEW episode, frame 1, age 0, 4 Hz, not charged
                    if k + 1 < K:
                        nx = recs[k + 1]
                        assert int(nx["held_ep"][i]) == int(nx["ep"][i]) == int(recs[k]["ep"][i]) + 1
                        assert int(nx["held_scan"][i]) == 0
                        assert abs(float(nx["aux"][i, 0]) - 1.0) < 1e-6 and float(nx["aux"][i, 1]) == 0.0
                        assert abs(float(nx["aux"][i, 2]) - 1 / 500) < 1e-6
                        assert not bool(nx["mask"][i]) or int(nx["now"][i]) > 0
                start = k + 1
    for k, r in enumerate(recs):
        # rewards: env reward - frame cost on accepted scans only; mask only where a new native scan was there
        assert torch.allclose(r["r"], r["r_env"] - 0.25 * r["mask"])
        assert bool(((r["mask"] > 0) <= (r["native_tick"] == r["now"])).all())
        # NAV05 input: held scan after gating = accepted scan at or before now, same episode
        if k + 1 < K:
            nxt = recs[k + 1]
            same = ~r["d"]
            assert bool((nxt["held_scan"][same] == r["nav"][same, 0]).all())
        held = r["nav"][:, 0]
        assert bool((held <= r["now"]).all())
        assert bool((r["nav"][:, 1] == r["ep"]).all())             # never a scan of another episode


def test_terminal_tick_follows_normal_mechanism():
    """Episode of 5 ticks (decisions at 0..0.8 s), 4 Hz: scans at 0.4, 0.6, 0.8 s accepted, the 0.8 s one on the
    terminal tick (counted, charged, mask 1). The 1.0 s instant is the episode end: never gated."""
    env, ts = make(1, [[5, 10 ** 6]], fc=0.5)
    recs = run(env, ts, torch.full((8, 1), 4))
    assert [bool(r["mask"]) for r in recs[:5]] == [False, False, True, True, True]
    last = recs[4]
    assert bool(last["d"]) and abs(float(last["r"]) - (float(last["r_env"]) - 0.5)) < 1e-6
    assert int(last["episode"]["frames"]) == 4                     # reset + 0.4 + 0.6 + 0.8
    assert not bool(recs[5]["mask"])                               # new episode tick 0: reset scan is not new
    assert int(recs[5]["held_ep"]) == 1 and abs(float(recs[5]["aux"][0, 2]) - 1 / 500) < 1e-6


def test_budget_penalty_once():
    env, ts = make(1, [[200, 10 ** 6]], fc=0.0, budget=10, bp=15.0)
    recs = run(env, ts, torch.full((250, 1), 4))
    pen = [k for k, r in enumerate(recs[:200]) if abs(float(r["r"] - r["r_env"]) + 15.0) < 1e-6]
    assert len(pen) == 1
    k = pen[0]
    assert bool(recs[k]["mask"]) and abs(float(recs[k]["aux"][0, 2]) - 1.0) < 1e-6   # 10 frames held -> 11th
    assert float(recs[199]["episode"]["budget_exceeded"]) == 1.0
    pen2 = [k for k, r in enumerate(recs[200:]) if abs(float(r["r"] - r["r_env"]) + 15.0) < 1e-6]
    assert pen2 == [k]                                             # flag reset with the episode: once per episode


def test_fixed_rate_reset_hz():
    """Fixed-rate validation: reset rate = the fixed rate and constant actions -> the TMP01 schedule."""
    for hz, idx, first in ((1.0, 2, [100, 200, 300]), (0.2, 0, [500, 1000])):
        env, ts = make(1, [[10 ** 6]], reset_hz=hz)
        recs = run(env, ts, torch.full((60, 1), idx))
        got = [int(r["now"]) for r in recs if bool(r["mask"])]
        assert got[:len(first)] == first, hz
        assert abs(float(recs[0]["aux"][0, 0]) - hz / 4) < 1e-6
