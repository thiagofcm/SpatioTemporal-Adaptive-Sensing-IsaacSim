"""Recurrent sensing policy (TMP02): the reference temporal Agent with a CNN encoder in front.

Reference: adaptive_sensor_policy@dc5b53f AdaptiveFPS/scripts/train_adaptive_fps_ppo.py `Agent` (line 256),
`get_states` (289), `get_action_and_value` (322); cleanRL layer_init (orthogonal, std sqrt(2), bias 0).

    held range image (1, 34, 139)
      -> Conv 3x3/2 (16) ReLU -> Conv 3x3/2 (32) ReLU -> Conv 3x3/2 (64) ReLU -> flatten (64 x 3 x 16 = 3072)
      -> Linear (128) ReLU                                   NAV05 encoder ARCHITECTURE (no padding / norm / pooling);
                                                             randomly initialized, trained with the rest, NAV05's
                                                             weights are never read
      ++ [fps_ratio, obs_age_ratio, frame_ratio]  (131)
      -> Linear (64) Tanh -> LSTM (64)                       reference Agent from here on (shared by actor and critic)
      -> actor  Linear 64 Tanh -> Linear 64 Tanh -> Linear n_actions (std 0.01)
      -> critic Linear 64 Tanh -> Linear 64 Tanh -> Linear 1 (std 1.0)

Inputs are (B, 1, H, W) images and (B, 3) ratios with B = T * n_envs in time-major order (as the reference
`b_obs[timestep_env_grid[:, envs].ravel()]`); the LSTM state is reset where done = 1 before the step (reference
`get_states`). Torch only (no Isaac Sim).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def conv_out(hw, k, s):
    return tuple((x - k) // s + 1 for x in hw)


class SensingAgent(nn.Module):
    def __init__(self, img_shape=(1, 34, 139), n_aux=3, n_actions=5, lstm_hidden_size=64,
                 cnn_channels=(16, 32, 64), cnn_kernel=3, cnn_stride=2, cnn_projection_dim=128):
        super().__init__()
        c, hw, layers = img_shape[0], tuple(img_shape[1:]), []
        for ch in cnn_channels:
            layers += [layer_init(nn.Conv2d(c, ch, cnn_kernel, cnn_stride)), nn.ReLU()]
            c, hw = ch, conv_out(hw, cnn_kernel, cnn_stride)
        self.cnn_flat_dim = c * hw[0] * hw[1]
        self.encoder = nn.Sequential(*layers, nn.Flatten(1), layer_init(nn.Linear(self.cnn_flat_dim, cnn_projection_dim)),
                                     nn.ReLU())
        self.network = nn.Sequential(layer_init(nn.Linear(cnn_projection_dim + n_aux, 64)), nn.Tanh())
        self.lstm = nn.LSTM(64, lstm_hidden_size)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(lstm_hidden_size, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0))
        self.actor = nn.Sequential(
            layer_init(nn.Linear(lstm_hidden_size, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, n_actions), std=0.01))

    def initial_state(self, n_envs, device):
        z = torch.zeros(self.lstm.num_layers, n_envs, self.lstm.hidden_size, device=device)
        return z, z.clone()

    def get_states(self, img, aux, lstm_state, done):
        hidden = self.network(torch.cat([self.encoder(img), aux], dim=-1))
        batch_size = lstm_state[0].shape[1]
        hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size))
        new_hidden = []
        for h, d in zip(hidden, done):
            h, lstm_state = self.lstm(
                h.unsqueeze(0),
                ((1.0 - d).view(1, -1, 1) * lstm_state[0], (1.0 - d).view(1, -1, 1) * lstm_state[1]))
            new_hidden.append(h)
        return torch.flatten(torch.cat(new_hidden), 0, 1), lstm_state

    def get_value(self, img, aux, lstm_state, done):
        hidden, _ = self.get_states(img, aux, lstm_state, done)
        return self.critic(hidden)

    def get_action_and_value(self, img, aux, lstm_state, done, action=None, deterministic=False):
        hidden, lstm_state = self.get_states(img, aux, lstm_state, done)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = logits.argmax(dim=-1) if deterministic else probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden), lstm_state
