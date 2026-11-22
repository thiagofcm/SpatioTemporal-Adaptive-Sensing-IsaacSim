# TurtleBot4 Standard asset (M1)

Articulated TurtleBot4 Standard for Isaac Lab 2.3.2 / Isaac Sim 5.1, built from the **official** robot descriptions.

## Provenance

| Source | Repository | Branch | Commit | Licence |
|---|---|---|---|---|
| TurtleBot4 Standard | `github.com/turtlebot/turtlebot4` → `turtlebot4_description` | `jazzy` | `7fd29fb420e906f3aca4a904adb54b69b11c7c00` (2025-09-09) | Apache-2.0 |
| Create3 base | `github.com/iRobotEducation/create3_sim` → `irobot_create_description` | `jazzy` | `6c305843afcfe094bf8eb2a0c6997f531e526b9d` (2026-09-24) | BSD-3-Clause |

The clones are in `third_party/` (not modified).

## Build pipeline

1. **xacro → URDF** in an isolated venv (`third_party/.venv_xacro`, `pip install xacro`; the conda env is untouched).
   * `$(find pkg)` is resolved by a 20-line stand-in for `ament_index_python` (`third_party/xacro_shim/`) that maps package names to the clones. There is no ROS on this machine.
   ```bash
   cd third_party
   PYTHONPATH=$PWD/xacro_shim .venv_xacro/bin/xacro turtlebot4/turtlebot4_description/urdf/standard/turtlebot4.urdf.xacro \
       gazebo:=ignition -o ../assets/turtlebot4_standard/turtlebot4_standard_raw.urdf
   ```
2. **Clean:** `python3 clean_urdf.py turtlebot4_standard_raw.urdf turtlebot4_standard.urdf`.
3. **Convert:**
   ```bash
   isaacrun -g auto -- python assets/turtlebot4_standard/convert_urdf.py --headless
   ```
   This uses Isaac Lab's `UrdfConverter`: floating base, fixed joints merged, convex-hull colliders, velocity drives on the wheels. Output: `usd/turtlebot4_standard.usd`, with three rigid bodies: `base_link` (4.99 kg, all fixed links merged), `left_wheel` and `right_wheel` (0.2 kg each).
4. **Runtime config:** `source/st_nav/st_nav/assets/turtlebot4.py` (`TB4_CFG`).

## Deviations from the upstream description (all deliberate, all in our copy only)

| # | Change | Reason |
|---|---|---|
| 1 | `<gazebo>` / `<ros2_control>` elements removed | Simulator plugins (no geometry or physics) |
| 2 | `gazebo:=ignition` parameter set used (not `classic`) | Both are upstream. They differ only in the Create3 body CoM x offset (classic 0 cm, ignition +2.28 cm) and caster height (−5.2 / −5.03 cm). With `classic` the CoM sits over the wheel axle and the simulated robot tips backwards onto its base (5.2° pitch). With `ignition` it rests on wheels + caster (0.25° pitch; caster 5.6 N, wheels 23.6 N each). **Masses are identical in both** |
| 3 | Wheel-drop prismatic joints fixed at 0 | Upstream relies on a Gazebo plugin spring; without it the joints collapse |
| 4 | Wheel colliders: cylinder → sphere of the same radius (0.03575 m) | 15 mm-wide cylinder line contacts scrub when turning (in-place turns reached only ~86 % of the command even with converged solver). The sphere gives one rolling contact point; it lies inside the footprint (bumper r = 0.177 m) |
| 5 | Caster: frictionless physics material (bound at spawn, collider group de-instanced) | The real caster is a free-rolling ball. Without it the caster drags: reverse at 80 %, in-place turns at 20 % of the command |
| 6 | Solver iterations 16 position / 2 velocity, physics 200 Hz | At 4/1 the contact + drive solve does not converge (in-place turns at ~80 %) |

## Known limitations

* **Mass (kept unchanged on request):** the URDF totals **5.390 kg**, vs the manufacturer's ≈ 3.945 kg.
  * The tower standoffs declare 0.260 kg each (1.04 kg in total). For a Ø12 × 205 mm rod that implies 11,200 kg/m³; solid aluminium would be 0.063 kg each.
  * 22 placeholder sensor frames carry 0.01 kg each (0.22 kg).
  * The excess sits high on the tower (simulated CoM z = 0.137 m above `base_link`).
  * To revisit when measured masses are available.
* **Footprint** is not a circle: front bumper r = 0.177 m, base cylinder r = 0.164 m, shell r = 0.137 m (convex hulls of the URDF collision meshes).
* **LiDAR frame:** `base_link/shell_link/rplidar_link` at (−0.040, 0, 0.1929) m with yaw +π/2 (verified in the simulated stage).
* Two small URDF collision boxes (RPLIDAR housing, OAK-D) report odd bounding boxes after conversion. They do not touch the ground and are irrelevant for navigation, but will be checked again when the robot's collision footprint is used against walls (M3).
