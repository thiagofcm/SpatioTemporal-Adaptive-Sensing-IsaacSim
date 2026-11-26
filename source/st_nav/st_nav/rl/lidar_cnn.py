"""CNN policy / value model for the 3D LiDAR range image (RSL-RL 5.x).

RSL-RL's CNNModel encodes every 2D observation group with a CNN, flattens it and concatenates it with the (optionally
normalized) 1D groups before the MLP head. LidarCNNModel adds a linear projection after each flattened CNN:

    range image (1, 34, 139) -> Conv 3x3/2 (16) -> ReLU -> Conv 3x3/2 (32) -> ReLU -> Conv 3x3/2 (64) -> ReLU
        -> flatten (64 x 3 x 16 = 3072) -> Linear (128) -> ReLU
    [128 image features | 7 navigation features (normalized)] -> MLP (256, 128, ELU) -> output

Used by name from the agent config: class_name = "st_nav.rl.lidar_cnn:LidarCNNModel". Actor and critic each build
their own encoder (PPO share_cnn_encoders = False).
"""
from __future__ import annotations

import torch.nn as nn
from rsl_rl.models import CNNModel
from rsl_rl.utils import resolve_nn_activation
from tensordict import TensorDict


class ProjectedCNN(nn.Sequential):
    """CNN (flattened output) followed by Linear(projection_dim) and an activation. Exposes the attributes CNNModel
    reads from an encoder (output_channels = None: flattened; output_dim)."""

    def __init__(self, cnn: nn.Module, projection_dim: int, activation: str):
        super().__init__(cnn, nn.Linear(int(cnn.output_dim), projection_dim), resolve_nn_activation(activation))
        self.output_channels = None
        self.output_dim = projection_dim


class LidarCNNModel(CNNModel):
    def __init__(self, obs: TensorDict, obs_groups: dict, obs_set: str, output_dim: int,
                 hidden_dims=(256, 128), activation: str = "elu", obs_normalization: bool = False,
                 distribution_cfg: dict | None = None, cnn_cfg: dict | None = None, cnns=None,
                 projection_dim: int = 128, projection_activation: str = "relu") -> None:
        self.projection_dim = int(projection_dim)
        self.projection_activation = projection_activation
        super().__init__(obs, obs_groups, obs_set, output_dim, hidden_dims, activation, obs_normalization,
                         distribution_cfg, cnn_cfg=cnn_cfg, cnns=cnns)
        if cnns is None:                                   # own encoders: append the projection to each CNN
            self.cnns = nn.ModuleDict({g: ProjectedCNN(self.cnns[g], self.projection_dim, projection_activation)
                                       for g in self.obs_groups_2d})

    def _get_latent_dim(self) -> int:
        """MLP input = 1D observations + one projected feature vector per 2D group (called inside MLPModel.__init__)."""
        return self.obs_dim + self.projection_dim * len(self.obs_groups_2d)
