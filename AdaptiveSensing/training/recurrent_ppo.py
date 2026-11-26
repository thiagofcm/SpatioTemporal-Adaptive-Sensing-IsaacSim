"""Recurrent PPO with causal-tick masking (TMP02): GAE and the minibatch update of the reference temporal trainer,
on GPU tensors, with the minibatch split into env chunks whose gradients are accumulated.

Reference: adaptive_sensor_policy@dc5b53f AdaptiveFPS/scripts/train_adaptive_fps_ppo.py
    GAE (line 647): gamma, gae_lambda; done = terminated | truncated (no bootstrap through a time-out)
    update (677-748): minibatches of whole envs replayed over the full rollout from the rollout-start LSTM state;
        mask = 1 on ticks whose sensing action was applied (frame_consumed). Policy loss, entropy, approx KL,
        clipfrac and advantage normalization use the masked ticks only; the value loss uses every tick;
        loss = pg_loss - ent_coef * entropy + vf_coef * v_loss; clip_grad_norm_(max_grad_norm); one Adam step.

Chunking: the CNN must be re-run with gradients on every image of a minibatch (256 envs x 256 steps = 65,536 range
images), too many activations for one pass. The minibatch is split into chunks of whole envs; every per-tick term is
summed per chunk and divided by the MINIBATCH totals (masked-tick count, tick count), the advantage statistics are
taken over the whole minibatch first, and the gradients of all chunks are accumulated before the single
clip_grad_norm_ / optimizer step. Envs are independent in the LSTM, so the result is the reference minibatch update
(up to floating-point summation order). Torch only (no Isaac Sim).
"""
from __future__ import annotations

import torch
import torch.nn as nn


@torch.no_grad()
def compute_gae(rewards, values, dones, next_value, next_done, gamma, gae_lambda):
    """(T, N) tensors; dones[t] = done flag stored WITH obs[t] (reference convention). Returns advantages, returns."""
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    lastgaelam = torch.zeros_like(next_value)
    for t in reversed(range(T)):
        if t == T - 1:
            nextnonterminal, nextvalues = 1.0 - next_done, next_value
        else:
            nextnonterminal, nextvalues = 1.0 - dones[t + 1], values[t + 1]
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
    return advantages, advantages + values


def minibatch_update(agent, optimizer, buf, mb_envs, initial_lstm_state, args, chunk_envs=None, step=True):
    """One reference minibatch (the envs `mb_envs`, all T steps) with gradient accumulation over env chunks.

    buf: dict of (T, N, ...) tensors: img, aux, actions, logprobs, advantages, returns, values, dones, masks.
    args: clip_coef, norm_adv, clip_vloss, ent_coef, vf_coef, max_grad_norm.
    step=False leaves the accumulated gradients in place (tests). Returns the reference diagnostics (floats are left
    as 0-dim tensors, no host sync)."""
    T, E = buf["masks"].shape[0], len(mb_envs)
    chunk_envs = E if not chunk_envs else min(int(chunk_envs), E)
    mask = buf["masks"][:, mb_envs]
    mask_sum = mask.sum().clamp(min=1.0)
    n_ticks = float(T * E)
    adv_all = buf["advantages"][:, mb_envs]
    if args.norm_adv:                                    # causal-tick statistics of the whole minibatch
        mb = mask.bool()
        cnt = mb.sum()
        mean = (adv_all * mask).sum() / cnt.clamp(min=1)
        std = (((adv_all - mean) ** 2) * mask).sum().div(cnt.clamp(min=1)).sqrt()
        norm = cnt > 0
        adv_all = torch.where(norm, (adv_all - mean) / (std + 1e-8), adv_all)
    optimizer.zero_grad(set_to_none=True)
    acc = {k: torch.zeros((), device=mask.device) for k in
           ("pg", "v", "ent", "old_kl", "kl", "clip")}
    for c0 in range(0, E, chunk_envs):
        ce = mb_envs[c0:c0 + chunk_envs]
        C = len(ce)
        sel = lambda k: buf[k][:, ce].reshape(T * C, *buf[k].shape[2:])  # noqa: E731  (time-major)
        m = sel("masks")
        _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
            sel("img"), sel("aux"), (initial_lstm_state[0][:, ce], initial_lstm_state[1][:, ce]), sel("dones"),
            sel("actions").long())
        logratio = newlogprob - sel("logprobs")
        ratio = logratio.exp()
        with torch.no_grad():
            acc["old_kl"] += (m * (-logratio)).sum()
            acc["kl"] += (m * ((ratio - 1) - logratio)).sum()
            acc["clip"] += (m * ((ratio - 1.0).abs() > args.clip_coef).float()).sum()
        mb_adv = adv_all[:, c0:c0 + C].reshape(-1)
        pg_loss1 = -mb_adv * ratio
        pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
        pg_sum = (m * torch.max(pg_loss1, pg_loss2)).sum()
        newvalue = newvalue.view(-1)
        ret, val = sel("returns"), sel("values")
        if args.clip_vloss:
            v_unclipped = (newvalue - ret) ** 2
            v_clipped = val + torch.clamp(newvalue - val, -args.clip_coef, args.clip_coef)
            v_sum = 0.5 * torch.max(v_unclipped, (v_clipped - ret) ** 2).sum()
        else:
            v_sum = 0.5 * ((newvalue - ret) ** 2).sum()
        ent_sum = (m * entropy).sum()
        loss = (pg_sum - args.ent_coef * ent_sum) / mask_sum + args.vf_coef * v_sum / n_ticks
        loss.backward()
        acc["pg"] += pg_sum.detach()
        acc["v"] += v_sum.detach()
        acc["ent"] += ent_sum.detach()
    grad_norm = None
    if step:
        grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
        optimizer.step()
    return {"policy_loss": acc["pg"] / mask_sum, "value_loss": acc["v"] / n_ticks, "entropy": acc["ent"] / mask_sum,
            "old_approx_kl": acc["old_kl"] / mask_sum, "approx_kl": acc["kl"] / mask_sum,
            "clipfrac": acc["clip"] / mask_sum, "grad_norm": grad_norm, "causal_ticks": mask.sum()}
