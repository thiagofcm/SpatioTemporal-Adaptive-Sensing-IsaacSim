# Revised Project Direction — Native Isaac Lab Navigation and Spatiotemporal Adaptive Sensing Framework

After reviewing the Phase 2 and Phase 2B investigations, I have decided to change the direction of the project.

We will no longer pursue a migration or reproduction of the original RL-Navigation repository. Although the investigation identified useful methodological aspects, particularly regarding observation formulation, reward design, PPO configuration and network architecture, the missing assets, implementation inconsistencies and legacy dependencies considerably reduce the benefits of migrating that repository.

Instead, we will develop our own native Isaac Lab framework, leveraging our previous research and the technical findings from your Isaac Lab feasibility investigation.

## 1. Overall research objective

Our ultimate objective is to develop and evaluate spatiotemporal adaptive sensing policies for mobile robot navigation using the Mirrorcle programmable 3D LiDAR.

The motivation for adopting Isaac Lab is primarily its ability to perform massively parallel, GPU-accelerated reinforcement learning training while maintaining physically realistic robot and obstacle dynamics.

The implementation must therefore take full advantage of Isaac Lab's native GPU architecture rather than reproduce a conventional CPU-based or ROS-dependent simulation pipeline.

The project should eventually support:

- Conventional 2D LiDAR navigation.
- Fixed-pattern Mirrorcle 3D LiDAR navigation.
- Temporal adaptive sensing.
- Spatial adaptive sensing.
- Joint spatiotemporal adaptive sensing.
- Eventual deployment on a physical TurtleBot4 equipped with the Mirrorcle LiDAR.

However, the immediate objective is to establish a functional and computationally efficient PPO navigation baseline in Isaac Lab using conventional 2D LiDAR.

## 2. Reuse our existing experimental environment

Instead of reconstructing the environment from the original paper, I want to reuse the geometry of the experimental environment from my previous temporal adaptive sensing research, which has already been implemented in Gazebo and in a simplified simulation environment.

I will provide access to the corresponding repositories.

Before implementing the Isaac Lab environment, investigate the existing implementations and identify:

- Exact wall and obstacle geometry, dimensions and positions.
- Robot initial poses and goal configurations.
- Static and moving-obstacle scenarios.
- OB1, OB2 and OB3 configurations.
- Robot dimensions, velocity constraints and sensor mounting.
- Existing domain randomization settings.
- Existing navigation reward and observation formulations.
- Existing Markovian moving-obstacle implementations.
- Existing evaluation methodology and logging formats.

Compare the Gazebo and simplified environments and identify any discrepancies between their geometry, timing, dynamics or assumptions.

Do not assume that the two implementations are identical without verifying them.

We should preserve the existing geometry as closely as possible, while keeping obstacle configurations, start/goal sampling and motion models configurable.

There is no longer any requirement to reconstruct or reproduce the original paper's experimental arena.

## 3. Native Isaac Lab implementation

Use the existing Isaac Lab installation and the launcher infrastructure established during Phase 1.

The implementation should be based on Native DirectRLEnv and adopt the following principles:

- GPU-accelerated physics.
- Fully vectorized PyTorch operations.
- GPU-resident robot and obstacle states.
- GPU-based raycasting using the previously validated MultiMeshRayCaster approach.
- Independent, batched environment resets.
- No per-environment Python loops in the training hot path.
- No unnecessary CPU-GPU transfers.
- No ROS communication during training.
- Headless execution without rendering.
- Configurable number of parallel environments.
- Configurable physics and navigation control frequencies.

Reuse the validated mechanisms from the previous feasibility investigations, particularly those related to moving-obstacle raycasting and GPU physics.

Keep the new project independent of modifications to the shared Isaac Lab installation whenever possible.

The intended physical platform is the TurtleBot4 Standard. Investigate whether an appropriate robot asset is available or can be generated from its existing URDF. Prefer the actual target robot over an unrelated robot approximation when feasible.

Report any limitations associated with actuator modeling, contacts, wheel geometry or articulation performance before making significant compromises.

## 4. Initial navigation problem

The initial implementation should use a conventional 360-degree 2D LiDAR with 40 beams, corresponding as closely as possible to the sensing configuration used in my previous navigation research.

Initially preserve the existing beam angles, sensor mounting, range limits and preprocessing conventions. Verify these from the existing source code.

The navigation control frequency should initially be 5 Hz, but remain configurable independently of the physics integration frequency.

The navigation action space should consist of continuous linear and angular velocity commands.

The robot should be able to stop and rotate in place. Initially, reverse motion can remain disabled unless the investigation identifies a compelling reason to enable it.

This capability will be particularly important when transitioning to a limited-field-of-view sensor.

## 5. Navigation observation formulation

For the new PPO navigation policy, I propose the following initial observation:

o_t = [LiDAR ranges, goal distance, sin(relative goal angle), cos(relative goal angle), current linear velocity, current angular velocity, previous navigation action]

With 40 LiDAR beams, this corresponds to 47 observation dimensions.

Use appropriate normalization and consistent coordinate and action conventions.

This formulation is a starting proposal. Compare it with the observations used in my previous TD3 navigation implementation and the original paper, and report any important implications before finalizing it.

Keep the observation construction modular and independent of the specific LiDAR implementation.

The environment should also support constructing the original TD3 observation separately if we decide to perform zero-shot evaluation of the existing trained navigation actor.

## 6. Navigation reward formulation

I want to use a simple, sensor-independent navigation reward composed of four terms:

r_nav = r_progress + r_proximity + r_time + r_terminal

### Goal progress

r_progress = k_p * (d_previous - d_current)

where d represents the Euclidean distance to the goal.

### Obstacle proximity

r_proximity = -k_o * max(0, 1 - c / c_safe)^2

where c represents the minimum ground-truth clearance between the robot footprint and the surrounding obstacles, including moving obstacles and walls.

The proximity penalty should be zero outside the desired safety clearance and increase smoothly as the robot approaches an obstacle.

### Time penalty

r_time = -k_t * control_dt

This should discourage unnecessary waiting without preventing the robot from stopping when waiting is appropriate.

### Terminal rewards

- Goal reached: +R_goal.
- Collision: -R_collision.
- Otherwise: 0.

Initially treat the maximum episode duration as a training truncation rather than an additional terminal penalty, provided the task does not impose an actual mission deadline.

Proposed initial coefficients:

- k_p = 10.0
- k_o = 1.0
- c_safe = 0.25 m
- k_t = 0.1
- R_goal = 30.0
- R_collision = 30.0
- control_dt = 0.2 s

These values are initial experimental settings, not final optimized parameters. All coefficients must be configurable.

Ensure that the previous goal distance is correctly initialized during each environment reset.

Collision detection should account for physical contacts and should not rely exclusively on LiDAR measurements.

Ground-truth clearance should account for the robot footprint and obstacle geometry rather than simply measuring center-to-center distances.

The most important requirement is that the navigation reward and termination functions remain independent of the sensing configuration.

When we eventually introduce adaptive sensing, the sensing policy must not be able to avoid penalties by choosing not to observe obstacles.

Initially avoid introducing additional rewards for heading alignment, angular velocity, action smoothness or exploration unless a concrete training problem motivates them.

## 7. Existing TD3 navigation policy

My previous navigation policy was trained using TD3 in Gazebo with conventional 2D LiDAR observations.

I do not intend to migrate the TD3 training algorithm into Isaac Lab.

However, it may be useful to evaluate the existing trained TD3 actor directly in the new Isaac Lab environment as an engineering validation experiment.

Investigate the feasibility and expected implementation effort of loading the trained actor and constructing its original observations, action scaling and timing conventions.

This would allow us to identify discrepancies between Gazebo and Isaac Lab without introducing the additional uncertainty of a newly trained policy.

Treat this as an optional validation milestone. It should not become a major dependency or delay the development of the PPO baseline.

I currently prefer training the new PPO navigation policy from scratch instead of transferring TD3 weights or performing behavior cloning.

## 8. PPO training

Initially use RSL-RL for PPO training, assuming it remains the most appropriate framework for the native Isaac Lab implementation.

Start with a relatively simple MLP navigation policy and a reasonable PPO configuration informed by the original paper and Isaac Lab's existing implementations.

The architecture should support recurrent policies, particularly LSTM, in the future.

Do not invest excessive effort in optimizing the conventional 2D navigation policy. Its primary purpose is to establish that the following components function correctly:

- Robot dynamics and control.
- Environment geometry.
- Dynamic obstacles.
- LiDAR observations.
- Reward and termination functions.
- Parallel environment execution.
- PPO training and evaluation.

Once reliable navigation is demonstrated, we will transition to the Mirrorcle sensor.

Keep the future adaptive sensing policy independent of the navigation policy. The navigation policy will eventually be frozen while a separate policy learns spatial and temporal sensing decisions.

Consequently, do not assume that we require a joint hybrid continuous-discrete navigation and sensing action space.

## 9. Computational performance

One of the main motivations for this transition is achieving substantially faster training.

Benchmark actual environment throughput and end-to-end PPO training throughput at different parallel environment counts, initially considering:

- 64 environments.
- 256 environments.
- 1024 environments.
- 2048 environments.
- 4096 environments, if supported by the available GPU resources.

For each configuration, report:

- Environment transitions per second.
- Physics and raycasting performance.
- PPO rollout collection time.
- PPO optimization time.
- End-to-end training throughput.
- GPU utilization and memory consumption where practical.
- Any computational bottlenecks.

Distinguish clearly between physics-only throughput, raycasting benchmarks, environment stepping throughput and actual end-to-end RL training throughput.

Do not extrapolate the previous isolated raycasting benchmarks directly into claims about complete navigation training performance.

Determine the practical number of parallel environments and identify the best configuration for our hardware.

## 10. Future Mirrorcle integration

The subsequent development phase will replace the conventional 2D LiDAR with the Mirrorcle programmable 3D LiDAR.

Its nominal field of view is approximately:

- Horizontal azimuth: -15 to +15 degrees.
- Vertical elevation: -8 to +8 degrees.

The future sensor implementation should support configurable ray directions in both azimuth and elevation.

We will characterize the actual hardware acquisition patterns, timing, resolution, range measurements and invalid-return behavior separately.

The sensor abstraction should eventually accommodate:

- Fixed and programmable ray patterns.
- Range measurements and corresponding ray directions.
- Measurement validity masks.
- Acquisition timestamps.
- Partial observations.
- Spatially varying sampling density.
- Temporally varying acquisition frequency.

For fixed-pattern 3D navigation, we may represent the measurements as an angular range image or another compact representation rather than directly processing raw point clouds.

This transition will likely require adapting the observation encoder and retraining the navigation policy.

The main research challenge is the reduced observability caused by the Mirrorcle's narrow field of view. The navigation policy may need to learn active rotation, stopping, memory-based navigation and conservative behavior under uncertainty.

Do not implement or train the Mirrorcle navigation policy yet. However, avoid architectural decisions during the initial implementation that would unnecessarily complicate this future transition.

## 11. Evaluation

Develop a reproducible evaluation pipeline with independent seeds and configurable static and dynamic scenarios.

Initially consider the following metrics:

- Navigation success rate.
- Collision rate.
- Timeout rate.
- Mean episode duration.
- Distance traveled.
- Minimum and mean obstacle clearance.
- Navigation return.
- Training throughput.

Preserve episode-level results and relevant trajectory, observation and action data in formats suitable for offline analysis.

Since rendering remains a potential limitation on the server, provide offline trajectory and scene visualization tools that do not depend on Isaac Sim's graphical rendering capabilities.

## 12. Immediate next steps and requested deliverables

Before proceeding with the full implementation:

1. Review the previous Phase 1, Phase 2 and Phase 2B investigations and identify which findings remain applicable.
2. Inspect my existing Gazebo and simplified environment repositories once I provide their locations.
3. Identify the components that can be reused directly, those that require adaptations and those that must be implemented from scratch.
4. Propose a revised project architecture and directory structure.
5. Establish a concrete implementation plan with incremental validation milestones.
6. Identify unresolved technical decisions, dependencies and potential blockers.
7. Provide updated estimates of implementation effort and expected computational resources.
8. Explain which aspects of the implementation can be completed autonomously and which require my input.

The plan should prioritize an efficient path toward a working 2D PPO navigation baseline rather than an exhaustive reproduction of another research project.

Please distinguish verified findings from assumptions and proposed design decisions.

Do not modify the original reference repositories or shared Isaac Lab installation without explicit approval.

For now, focus on the investigation of our existing environment and producing the revised implementation plan. Do not begin the complete implementation until we have reviewed and agreed on the remaining decisions.



M1 — Build and calibrate the robot

Import the physical TurtleBot4 model and implement differential-drive control. Test straight motion, rotation, curved trajectories, saturation and actuator dynamics.
Deliverable: functional TurtleBot4 USD asset and calibration report.Estimated implementation: 6–10 hours


2


M2 — Benchmark computational performance

Run hundreds or thousands of articulated robots simultaneously, with basic walls and LiDAR. Measure simulation speed, stability and GPU memory usage.
Deliverable: measured throughput and selected number of parallel environments.Estimated implementation: 2–3 hours


3


M3 — Reconstruct your navigation environment

Port the S-shaped corridor, randomized static obstacles, starting positions, goals and ground-truth clearance computation.
Deliverable: physically simulated version of your existing static scenario.Estimated implementation: 5–7 hours


4


M4 — Complete the reinforcement learning environment

Implement the 40-beam LiDAR, 47-dimensional observation, navigation reward, termination conditions and resets.
Deliverable: functional Isaac Lab DirectRLEnv ready for PPO training.Estimated implementation: 4–6 hours


5


M5 — Train the PPO navigation policy

Train directly on the randomized static-obstacle scenario. Evaluate success, collisions, path length, navigation time and clearance.
Deliverable: trained navigation checkpoints, learning curves and evaluation results.Estimated implementation: 3–5 hours, plus training


6

M6 — Introduce moving obstacles

Port the stochastic obstacle motion models, train on dynamic scenarios and compare against our previous navigation results.
Deliverable: navigation policy capable of operating in dynamic environments.Estimated implementation: 5–7 hours, plus training