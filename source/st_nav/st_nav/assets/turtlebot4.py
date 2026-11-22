"""TurtleBot4 Standard articulation (converted from the official URDF, see assets/turtlebot4_standard/README.md)."""
import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

_ST_ROOT = os.environ.get("ST_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))
TB4_USD_PATH = os.path.join(_ST_ROOT, "assets", "turtlebot4_standard", "usd", "turtlebot4_standard.usd")

# Geometry from the URDF / converted USD (verified in M1)
TB4_WHEEL_RADIUS = 0.03575            # m
TB4_TRACK_WIDTH = 0.233               # m
TB4_LIDAR_OFFSET = (-0.04, 0.0, 0.192915)   # m, rplidar_link in base_link (shell 0.0942 + 0.098715), yaw +pi/2
TB4_LIDAR_YAW = 1.5707963267948966
TB4_FOOTPRINT_RADIUS_FRONT = 0.177    # m, bumper convex hull (x >= 0 half)
TB4_FOOTPRINT_RADIUS_BASE = 0.164     # m, create3 base cylinder
TB4_CASTER_COLLIDER = "base_link/collisions/mesh_2"   # front caster sphere (r = 0.01 m); bind a frictionless material
TB4_PHYSICS_DT = 1.0 / 100.0          # s, adopted after M2 (M1 criteria re-validated at 100 Hz, 16/2 iterations)
TB4_WHEEL_SPEED_LIMIT_RAD = 0.305 / TB4_WHEEL_RADIUS  # rad/s (real saturation 0.305 m/s)

TB4_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=TB4_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False, max_depenetration_velocity=1.0, enable_gyroscopic_forces=True),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=16, solver_velocity_iteration_count=2,
            sleep_threshold=0.0, stabilization_threshold=0.001),
    ),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.005), joint_pos={".*": 0.0}, joint_vel={".*": 0.0}),
    actuators={
        # Velocity-controlled wheels: torque = damping * (w_target - w), clipped at effort_limit.
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=["left_wheel_joint", "right_wheel_joint"],
            stiffness=0.0,
            damping=5.0,
            effort_limit_sim=1.0,
            velocity_limit_sim=1.2 * TB4_WHEEL_SPEED_LIMIT_RAD,
        ),
    },
)
"""Solver: 16 position / 2 velocity iterations (M1: at 4/1 the contact + drive solve does not converge and in-place
turning reaches only ~80 % of the command; 16/2 gives <= 3 %, 32/4 adds < 1 %).
Gains: damping 5 N.m.s/rad gives a drive time constant of ~1 ms on the wheel (I_eff ~ 0.0035 kg m^2 per wheel);
the effort limit (1 N.m per wheel, ~ 7 m/s^2 for 5.4 kg) is not binding at the commanded accelerations, so the
drive is near-ideal and the robot's lag is set by the calibrated CommandModel (st_nav.control.diff_drive)."""

# ---------------------------------------------------------------------------------------------------------------------
# Calibrated command model (M1, infrastructure/M1_robot_calibration/scripts/m1_calibrate.py): fitted with the PhysX drive + contacts in the loop against
# 16 real TurtleBot4 step-response recordings (motion_dynamic_charac/runs), so the simulated drive dynamics (rise
# ~17-38 ms, no delay) are not counted twice. In-motion transitions: delay/rise within +-0.02 s of the real robot.
# Not modelled: the firmware start-up delay after >= 0.8 s at rest (real 0.53 +- 0.59 s, highly variable).
from st_nav.control.diff_drive import CommandModelParams, DiffDriveGeometry  # noqa: E402

TB4_GEOMETRY = DiffDriveGeometry(wheel_radius=TB4_WHEEL_RADIUS, track_width=TB4_TRACK_WIDTH, wheel_speed_max=0.305)
TB4_COMMAND_MODEL = CommandModelParams(
    delay_s=0.14,          # linear channel pure delay
    delay_w_s=0.12,        # angular channel pure delay
    tau_v_s=0.01,
    tau_w_s=0.05,
    k_v=1.0,
    k_w=0.90,              # RMS-optimal global angular gain (real gain is command-dependent, 0.8-1.3)
    accel_v_max=1.0,       # m/s^2 (removes the start-from-rest overshoot; real from-rest rise matched within 0.003 s)
    accel_w_max=0.0,
)
