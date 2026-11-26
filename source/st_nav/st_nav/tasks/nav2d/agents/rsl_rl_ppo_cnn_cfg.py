"""RSL-RL PPO configuration with the CNN LiDAR encoder (3D LiDAR range image, M8).

Actor and critic (separate encoders): range image (1, 34, 139) -> Conv 3x3 stride 2 (16, 32, 64, ReLU) -> flatten ->
Linear 128 -> ReLU, concatenated with the 7 navigation features (empirically normalized), -> MLP 256, 128 (ELU).
Gaussian policy over (v, w) in [-1, 1]^2 (clipped; scaled by v_max / w_max in the env). PPO hyper-parameters as the 2D
policies (CorridorNavPPORunnerCfg), so only the observation encoder differs.
"""
from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlCNNModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class LidarCNNModelCfg(RslRlCNNModelCfg):
    class_name: str = "st_nav.rl.lidar_cnn:LidarCNNModel"
    projection_dim: int = 128
    projection_activation: str = "relu"


CNN = RslRlCNNModelCfg.CNNCfg(output_channels=[16, 32, 64], kernel_size=3, stride=2, padding="none",
                              activation="relu", flatten=True)


@configclass
class CorridorNavCNNPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 1500
    save_interval = 100
    experiment_name = "stnav_tb4_corridor_lidar3d"
    obs_groups = {"actor": ["lidar", "policy"], "critic": ["lidar", "policy"]}
    clip_actions = 1.0
    actor = LidarCNNModelCfg(hidden_dims=[256, 128], activation="elu", obs_normalization=True, cnn_cfg=CNN,
                             distribution_cfg=RslRlCNNModelCfg.GaussianDistributionCfg(init_std=1.0))
    critic = LidarCNNModelCfg(hidden_dims=[256, 128], activation="elu", obs_normalization=True, cnn_cfg=CNN)
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        share_cnn_encoders=False,
    )
