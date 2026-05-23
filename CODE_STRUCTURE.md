# Code Structure

## Overall Pipeline

```text
BVH files in intput/
  -> general_motion_retargeting/utils/lafan1.py
  -> general_motion_retargeting/motion_retarget.py
  -> general_motion_retargeting/ik_configs/bvh_lafan1_to_r2v2.json
  -> R2V2 MuJoCo qpos
  -> general_motion_retargeting/utils/motion_export.py
  -> npz files in output/output_npz/
```

The conversion first parses each LAFAN1-style BVH frame into global human bone
poses, then solves task-space inverse kinematics on the R2V2 MuJoCo model. The
export step expands the resulting MuJoCo `qpos` into root pose, DoF positions,
finite-difference velocities, FK-derived body poses, and name metadata.

## Repository Layout

- `intput/`: source BVH motions used for the current R2V2 retargeting checks.
- `output/output_npz/`: generated R2V2 `.npz` motions.
- `scripts/`: CLI entry points for BVH conversion, dataset conversion, replay,
  visualization, and format conversion.
- `general_motion_retargeting/`: core retargeting package.
- `general_motion_retargeting/ik_configs/`: human-to-robot task maps and task
  weights.
- `assets/r2_v2_with_shell_no_hand/`: R2V2 MuJoCo/URDF model and meshes. The
  MuJoCo joint ranges here are part of the `.npz` data contract.
- `third_party/`: bundled helpers such as poselib.

## Modified File: `general_motion_retargeting/ik_configs/bvh_lafan1_to_r2v2.json`

This JSON config maps LAFAN1 BVH bones to R2V2 MuJoCo bodies. Each task entry is:

```text
"robot_body_name": [
  "human_body_name",
  position_weight,
  orientation_weight,
  position_offset_xyz,
  rotation_offset_quat_wxyz
]
```

### Top-Level Fields

- `_comment_r2v2_arm_waist_tuning`: Records why this config was tuned. The
  evidence came from local diagnostics comparing `dof_pos` against MuJoCo joint
  ranges and from `mink.ConfigurationLimit` behavior in the IK loop.
- `robot_root_name`: R2V2 root body used by IK.
- `human_root_name`: BVH root bone used for scaling and relative positions.
- `ground_height`: Ground reference offset.
- `human_height_assumption`: Height used to scale human targets.
- `use_ik_match_table1`: Enables the first IK task set.
- `use_ik_match_table2`: Enables the second IK task set.
- `posture_task`: Optional low-priority `mink.PostureTask` regularizer. It
  biases selected arm joints toward a MuJoCo qpos-space reference when
  task-space targets conflict. Waist posture cost is kept at `0` so the torso
  can follow BVH body orientation rather than being pulled to neutral.
- `human_scale_table`: Per-bone scale factors before setting IK targets.
- `ik_match_table1`: First-pass tasks, mainly orientation alignment and coarse
  body placement.
- `ik_match_table2`: Second-pass tasks, mainly stronger positional refinement.

### R2V2 Arm And Waist Tuning

- `waist_pitch_link <- Spine2`: orientation cost is kept at `10` in both IK
  passes because the body angle must follow the BVH torso. Hand tasks are
  instead weakened/disabled so they cannot pull the torso away from this target.
- `left_arm_pitch_link <- LeftForeArm` and
  `right_arm_pitch_link <- RightForeArm`: orientation cost was reduced to `0`
  and position cost to `0.5`. R2V2 no-hand kinematics cannot exactly reproduce
  LAFAN1 forearm orientation, and the previous task pushed
  `right_arm_pitch_joint` into its lower limit.
- `left_hand_roll_link <- LeftHand` and
  `right_hand_roll_link <- RightHand`: orientation cost was reduced to `0` and
  position cost to `0`. These are fixed downstream bodies in the no-hand model,
  so hand end-body tracking would over-constrain the shoulder/elbow chain and
  borrow torso motion.
- `posture_task.costs`: waist joints use cost `0`, while shoulder and arm joints
  use cost `0.05`. This keeps the arm solution near neutral only when the motion
  targets leave redundant freedom; it is intentionally weaker than the main
  `FrameTask` tracking costs.
- `posture_task.target_offsets`: adds a small `+0.2` rad qpos-space target bias
  to both R2V2 `arm_pitch_joint`s. Local diagnostics showed this lowers the walk
  hands/forearms by about 1.4-1.5 cm while leaving the body-angle task
  effectively unchanged, because waist posture cost remains `0`.
- `direction_tasks`: second-pass elbow-to-hand direction constraints. These
  align each robot forearm direction with the matching BVH forearm-to-hand
  direction without enforcing full wrist orientation, which would over-constrain
  the R2V2 no-hand model.

## Relevant Runtime Files

### `general_motion_retargeting/utils/lafan1.py`

- `load_bvh_file(bvh_file, format="lafan1")`: Reads a BVH file, runs FK on the
  human skeleton, converts coordinates from BVH centimeters to MuJoCo-style
  meters, adds `LeftFootMod` and `RightFootMod`, and returns frame dictionaries
  plus the human-height value used for scaling.

### `general_motion_retargeting/motion_retarget.py`

- `BodyDirectionTask`: Custom `mink.Task` that minimizes the difference between
  a robot body-to-body unit vector and a BVH-derived target unit vector. Its
  Jacobian uses MuJoCo translational body Jacobians and the normalized-vector
  derivative `(I - uu^T) / ||v||`, so it controls forearm direction rather than
  absolute hand pose.
- `GeneralMotionRetargeting.__init__`: Loads the robot XML, IK config, scale
  table, MuJoCo joint limits, and task lists.
- `setup_retarget_configuration`: Converts each IK table entry into a
  `mink.FrameTask` with position and orientation costs, then appends optional
  second-pass `BodyDirectionTask`s from `direction_tasks`.
- `create_posture_task`: Builds the optional `mink.PostureTask` from the IK
  config, maps joint names to MuJoCo DoF indices, targets `model.qpos0` plus
  optional `target_offsets`, and raises an error if a configured joint name does
  not exist.
- `update_targets`: Scales human data, applies position/rotation offsets, and
  writes target SE(3) poses into the active IK tasks.
- `retarget`: Solves the two-stage IK problem with `mink.solve_ik`, integrates
  the result into MuJoCo `qpos`, and returns a copy of that `qpos`.
- `scale_human_data`: Scales each tracked human bone relative to the human root.
- `offset_human_data`: Applies configured local position and rotation offsets
  before target assignment. Zero-cost IK entries are allowed to remain in the
  config for documentation/tuning context; they do not create `FrameTask`
  offsets, so this function preserves their scaled pose without applying a
  missing offset.
- `offset_human_data_to_ground`: Optionally shifts the motion so the lowest foot
  target sits above the ground.
- `set_ground_offset`: Sets a global z-offset used during target updates.
- `apply_ground_offset`: Applies the configured z-offset to all targets.

### `general_motion_retargeting/utils/motion_export.py`

- `finite_difference`: Computes endpoint one-sided and interior centered finite
  differences for velocities.
- `quaternion_angular_velocity`: Converts frame-to-frame quaternion changes into
  angular velocity while avoiding quaternion sign flips.
- `extract_dof_names`: Reads MuJoCo joint names in `qpos` order, skipping the
  floating base.
- `build_npz_motion_data`: Splits `qpos` into root pose and DoF arrays, derives
  velocities, runs FK for body pose fields, and returns the `.npz` dictionary.

### `scripts/bvh_to_robot.py`

- CLI entry point for single-file BVH conversion.
- Loads BVH frames with `load_bvh_file`.
- Creates `GeneralMotionRetargeting`.
- Steps the optional viewer.
- Saves either legacy `.pkl` fields or the expanded `.npz` fields from
  `build_npz_motion_data`.

### `scripts/npz_to_robot_video_offscreen.py`

- `_configure_gl_backend_from_argv`: Reads `--gl_backend` before importing
  MuJoCo so the requested OpenGL backend actually takes effect. On Windows it
  defaults to `glfw`, because MuJoCo 3.x rejects `egl` there and Windows
  Graphics Preferences can route `python.exe` to the NVIDIA high-performance
  adapter.
- `load_npz_motion`: Loads root pose, root rotation, DoF positions, and FPS from
  an `.npz` file.
- Main CLI: checks DoF count against the current MuJoCo model and renders a
  headless MP4 by writing `root_pos`, `root_rot_wxyz`, and `dof_pos` back into
  `data.qpos`.
