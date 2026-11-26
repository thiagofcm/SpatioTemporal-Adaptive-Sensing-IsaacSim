"""Sensing agent (AdaptiveSensing/training/sensing_agent.py) and recurrent PPO update (recurrent_ppo.py) against the
reference temporal trainer (adaptive_sensor_policy@dc5b53f AdaptiveFPS/scripts/train_adaptive_fps_ppo.py).

Checks: CNN architecture = NAV05 encoder (parameter shapes, layer order; weights NOT taken from NAV05), LSTM reset by
done affects only those envs, rollout log-probs reproduced by the minibatch replay (ratio 1), the chunked update
equals the literal reference minibatch update (loss, diagnostics and gradients), gradients reach the CNN, GAE equals
the reference loop."""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from AdaptiveSensing.training.recurrent_ppo import compute_gae, minibatch_update  # noqa: E402
from AdaptiveSensing.training.sensing_agent import SensingAgent  # noqa: E402

NAV05 = os.path.join(ROOT, "Navigation/runs_3d/NAV05_chase-ob1ob2_lidar3d-r4_cnn-from-nav04/2026-11-23_20-21-14_ppo_cnn_"
                     "n4096_s42/model_1499.pt")
ARGS = SimpleNamespace(clip_coef=0.2, norm_adv=True, clip_vloss=True, ent_coef=0.05, vf_coef=0.5, max_grad_norm=0.5)
IMG = (1, 34, 139)


def small_agent(seed=0, img=(1, 12, 20)):
    torch.manual_seed(seed)
    return SensingAgent(img, 3, 5, 16, (4, 8), 3, 2, 16)


@pytest.mark.skipif(not os.path.exists(NAV05), reason="NAV05 checkpoint not available")
def test_encoder_architecture_equals_nav05():
    agent = SensingAgent(IMG)
    ck = torch.load(NAV05, map_location="cpu", weights_only=False)["actor_state_dict"]
    nav = [tuple(ck[f"cnns.lidar.0.{i}.weight"].shape) for i in (0, 2, 4)] + [tuple(ck["cnns.lidar.1.weight"].shape)]
    convs = [m for m in agent.encoder if isinstance(m, (nn.Conv2d, nn.Linear))]
    assert [tuple(m.weight.shape) for m in convs] == nav
    assert [type(m).__name__ for m in agent.encoder] == ["Conv2d", "ReLU", "Conv2d", "ReLU", "Conv2d", "ReLU",
                                                         "Flatten", "Linear", "ReLU"]
    assert all(m.padding == (0, 0) and m.stride == (2, 2) for m in convs[:3])
    assert agent.cnn_flat_dim == 3072
    assert not torch.equal(convs[0].weight, ck["cnns.lidar.0.0.weight"])          # random init, not NAV05's
    out = agent.get_action_and_value(torch.rand(4, *IMG), torch.rand(4, 3), agent.initial_state(4, "cpu"),
                                     torch.zeros(4))
    assert out[0].shape == (4,) and out[3].shape == (4, 1)


def test_done_resets_only_those_envs():
    agent = small_agent()
    n = 4
    img, aux = torch.rand(n, 1, 12, 20), torch.rand(n, 3)
    state = (torch.randn(1, n, 16), torch.randn(1, n, 16))
    done = torch.tensor([0.0, 1.0, 0.0, 1.0])
    h, _ = agent.get_states(img, aux, state, done)
    h0, _ = agent.get_states(img, aux, (torch.zeros(1, n, 16), torch.zeros(1, n, 16)), torch.zeros(n))
    hs, _ = agent.get_states(img, aux, state, torch.zeros(n))
    assert torch.allclose(h[[1, 3]], h0[[1, 3]]) and torch.allclose(h[[0, 2]], hs[[0, 2]])
    assert not torch.allclose(h[[0, 2]], h0[[0, 2]])


def rollout(agent, T=12, N=8, seed=1):
    g = torch.Generator().manual_seed(seed)
    buf = {"img": torch.rand(T, N, 1, 12, 20, generator=g), "aux": torch.rand(T, N, 3, generator=g),
           "dones": (torch.rand(T, N, generator=g) < 0.1).float(), "masks": (torch.rand(T, N, generator=g) < 0.4).float(),
           "rewards": torch.randn(T, N, generator=g)}
    state0 = (torch.randn(1, N, 16, generator=g) * 0.1, torch.randn(1, N, 16, generator=g) * 0.1)
    st = state0
    acts, lps, vals = [], [], []
    with torch.no_grad():
        for t in range(T):
            a, lp, _, v, st = agent.get_action_and_value(buf["img"][t], buf["aux"][t], st, buf["dones"][t])
            acts.append(a.float()), lps.append(lp), vals.append(v.flatten())
    buf.update(actions=torch.stack(acts), logprobs=torch.stack(lps), values=torch.stack(vals))
    nv = agent.get_value(buf["img"][-1], buf["aux"][-1], st, buf["dones"][-1]).reshape(1, -1).detach()
    buf["advantages"], buf["returns"] = compute_gae(buf["rewards"], buf["values"], buf["dones"], nv,
                                                    torch.zeros(N), 0.99, 0.98)
    return buf, state0


def reference_minibatch_loss(agent, buf, mb_envs, state0, args):
    """Literal reference update body (train_adaptive_fps_ppo.py 686-748) for one minibatch, flat time-major data."""
    T, N = buf["masks"].shape
    flat = lambda k: buf[k].reshape(T * N, *buf[k].shape[2:])  # noqa: E731
    grid = np.arange(T * N).reshape(T, N)
    idx = torch.as_tensor(grid[:, mb_envs.numpy()].ravel())
    mb_dones = buf["dones"][:, mb_envs].reshape(-1)
    _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
        flat("img")[idx], flat("aux")[idx], (state0[0][:, mb_envs], state0[1][:, mb_envs]), mb_dones,
        flat("actions").long()[idx])
    logratio = newlogprob - flat("logprobs")[idx]
    ratio = logratio.exp()
    mb_mask = flat("masks")[idx]
    mb_mask_sum = mb_mask.sum().clamp(min=1.0)
    approx_kl = (mb_mask * ((ratio - 1) - logratio)).sum() / mb_mask_sum
    mb_adv = flat("advantages")[idx]
    valid = mb_adv[mb_mask.bool()]
    if valid.numel() > 0:
        mb_adv = (mb_adv - valid.mean()) / (valid.std(unbiased=False) + 1e-8)
    pg_loss = (mb_mask * torch.max(-mb_adv * ratio, -mb_adv * torch.clamp(ratio, 1 - args.clip_coef,
                                                                          1 + args.clip_coef))).sum() / mb_mask_sum
    newvalue = newvalue.view(-1)
    ret, val = flat("returns")[idx], flat("values")[idx]
    v_clipped = val + torch.clamp(newvalue - val, -args.clip_coef, args.clip_coef)
    v_loss = 0.5 * torch.max((newvalue - ret) ** 2, (v_clipped - ret) ** 2).mean()
    entropy_loss = (mb_mask * entropy).sum() / mb_mask_sum
    return pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef, pg_loss, v_loss, entropy_loss, approx_kl


def grads(agent):
    return [p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in agent.parameters()]


def test_replay_reproduces_rollout_logprobs():
    agent = small_agent()
    buf, state0 = rollout(agent)
    out = minibatch_update(agent, torch.optim.SGD(agent.parameters(), lr=0.0), buf, torch.arange(8), state0, ARGS,
                           step=False)
    assert abs(float(out["approx_kl"])) < 1e-6 and abs(float(out["old_approx_kl"])) < 1e-6
    assert float(out["clipfrac"]) == 0.0


@pytest.mark.parametrize("chunk", [None, 1, 3, 4])
def test_chunked_update_equals_reference(chunk):
    agent = small_agent()
    buf, state0 = rollout(agent)
    with torch.no_grad():                         # move the policy so ratios differ from 1
        for p in agent.parameters():
            p.add_(0.05 * torch.randn_like(p))
    mb = torch.tensor([5, 1, 6, 2])
    agent.zero_grad()
    loss, pg, v, ent, kl = reference_minibatch_loss(agent, buf, mb, state0, ARGS)
    loss.backward()
    g_ref = grads(agent)
    opt = torch.optim.SGD(agent.parameters(), lr=0.0)
    out = minibatch_update(agent, opt, buf, mb, state0, ARGS, chunk_envs=chunk, step=False)
    g = grads(agent)
    for name, a, b in (("pg", out["policy_loss"], pg), ("v", out["value_loss"], v), ("ent", out["entropy"], ent),
                       ("kl", out["approx_kl"], kl)):
        assert torch.allclose(a, b.detach(), rtol=1e-5, atol=1e-6), name
    for (name, _), a, b in zip(agent.named_parameters(), g, g_ref):
        assert torch.allclose(a, b, rtol=1e-4, atol=1e-7), name
    enc = [a for (name, _), a in zip(agent.named_parameters(), g) if name.startswith("encoder.")]
    assert all(float(x.abs().sum()) > 0 for x in enc)                 # gradients reach every CNN layer


def test_gae_equals_reference_loop():
    T, N = 9, 3
    g = torch.Generator().manual_seed(7)
    r, v, d = torch.randn(T, N, generator=g), torch.randn(T, N, generator=g), (torch.rand(T, N, generator=g) < .2).float()
    nv, nd = torch.randn(1, N, generator=g), torch.tensor([0.0, 1.0, 0.0])
    adv, ret = compute_gae(r, v, d, nv, nd, 0.99, 0.98)
    for i in range(N):                                                  # per-env scalar loop
        last = 0.0
        for t in reversed(range(T)):
            nnt = 1.0 - (float(nd[i]) if t == T - 1 else float(d[t + 1, i]))
            nval = float(nv[0, i]) if t == T - 1 else float(v[t + 1, i])
            delta = float(r[t, i]) + 0.99 * nval * nnt - float(v[t, i])
            last = delta + 0.99 * 0.98 * nnt * last
            assert abs(float(adv[t, i]) - last) < 1e-5
    assert torch.allclose(ret, adv + v)
