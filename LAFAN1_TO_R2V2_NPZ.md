# LAFAN1 到 R2V2 npz 转化流程

> 代码里使用的格式名是 `lafan1`。如果口头写成 `lanfan1`，本文统一按 `LAFAN1/LaFAN1` 理解。

## 一句话流程

LAFAN1 的 `.bvh` 会先被解析成人体各骨骼的全局位姿字典，然后通过 GMR 的两阶段 IK 映射到 R2V2 的 MuJoCo `qpos`，最后把 `qpos` 拆成根节点、关节、刚体位姿和速度等字段保存为 `.npz`。

```text
LAFAN1 .bvh
  -> load_bvh_file()
  -> 每帧人体骨骼 {bone: [position, quaternion_wxyz]}
  -> GeneralMotionRetargeting(src_human="bvh_lafan1", tgt_robot="r2v2")
  -> 每帧 R2V2 qpos = [root_pos(3), root_quat_wxyz(4), dof_pos(D)]
  -> build_npz_motion_data()
  -> R2V2 motion .npz
```

## 入口命令

单个 BVH 转 R2V2 npz：

```powershell
python scripts/bvh_to_robot.py `
  --bvh_file dataset/jumps1_subject2.bvh `
  --format lafan1 `
  --robot r2v2 `
  --motion_fps 30 `
  --save_path output/r2v2_jumps1_subject2.npz
```

如果 `--save_path` 后缀是 `.npz`，`scripts/bvh_to_robot.py` 会自动选择 `npz`；也可以显式传：

```powershell
--save_format npz
```

生成后可以用最新新增的离屏脚本检查 npz 是否能被 R2V2 MuJoCo 模型播放：

```powershell
python scripts/npz_to_robot_video_offscreen.py `
  --npz_file output/r2v2_jumps1_subject2.npz `
  --robot r2v2 `
  --video_path output/r2v2_jumps1_subject2_offscreen.mp4
```

## 1. LAFAN1 BVH 解析成人体帧

代码位置：`general_motion_retargeting/utils/lafan1.py`

`load_bvh_file(bvh_file, format="lafan1")` 做了四件事：

1. 用 `read_bvh()` 读取 BVH，得到局部关节位移、旋转和骨架父子关系。
2. 用 `utils.quat_fk()` 做人体骨架正运动学，得到每一帧每个 bone 的全局位置和全局四元数。
3. 做坐标系转换和单位转换：
   - 旋转矩阵为 `[[1,0,0], [0,0,-1], [0,1,0]]`
   - 位置从厘米转成米：`position / 100`
   - 四元数使用 `wxyz`，也就是 scalar first。
4. 对 LAFAN1 补出两个脚部目标：
   - `LeftFootMod = [LeftFoot 的位置, LeftToe 的旋转]`
   - `RightFootMod = [RightFoot 的位置, RightToe 的旋转]`

输出是：

```python
frames, actual_human_height = load_bvh_file(...)
```

其中 `frames` 是一个列表，每个元素是一帧人体数据：

```python
{
    "Hips": [position_xyz_m, quaternion_wxyz],
    "Spine2": [position_xyz_m, quaternion_wxyz],
    "LeftFootMod": [position_xyz_m, quaternion_wxyz],
    ...
}
```

当前代码里 `actual_human_height` 固定为 `1.75`，后续 GMR 会用它和 IK 配置里的身高假设做比例缩放。

## 2. R2V2 在 GMR 参数里的接入

代码位置：`general_motion_retargeting/params.py`

R2V2 是通过 `r2v2` 这个 robot key 接进 GMR 的：

```python
ROBOT_XML_DICT["r2v2"] = assets/r2_v2_with_shell_no_hand/r2v2_with_shell.xml
IK_CONFIG_DICT["bvh_lafan1"]["r2v2"] = general_motion_retargeting/ik_configs/bvh_lafan1_to_r2v2.json
ROBOT_BASE_DICT["r2v2"] = "base_link"
```

也就是说，转换时 `--robot r2v2` 会选中：

- R2V2 MuJoCo 模型：`assets/r2_v2_with_shell_no_hand/r2v2_with_shell.xml`
- LAFAN1 到 R2V2 的 IK 映射：`general_motion_retargeting/ik_configs/bvh_lafan1_to_r2v2.json`
- 播放和跟随视角的基座 body：`base_link`

## 3. LAFAN1 到 R2V2 的 IK 映射

代码位置：`general_motion_retargeting/ik_configs/bvh_lafan1_to_r2v2.json`

这个配置描述了人体 bone 和 R2V2 body 的对应关系。核心字段是：

- `human_root_name`: `Hips`
- `robot_root_name`: `base_link`
- `human_height_assumption`: `1.8`
- `human_scale_table`: 每个参与 IK 的人体部位缩放系数，当前主体为 `0.85`
- `ik_match_table1` 和 `ik_match_table2`: 两阶段 IK 任务表

主要映射关系如下：

| R2V2 body | LAFAN1 bone |
| --- | --- |
| `base_link` | `Hips` |
| `left_hip_yaw_link` | `LeftUpLeg` |
| `left_knee_link` | `LeftLeg` |
| `left_ankle_roll_link` | `LeftFootMod` |
| `right_hip_yaw_link` | `RightUpLeg` |
| `right_knee_link` | `RightLeg` |
| `right_ankle_roll_link` | `RightFootMod` |
| `waist_pitch_link` | `Spine2` |
| `left_shoulder_yaw_link` | `LeftArm` |
| `left_arm_pitch_link` | `LeftForeArm` |
| `left_hand_roll_link` | `LeftHand` |
| `right_shoulder_yaw_link` | `RightArm` |
| `right_arm_pitch_link` | `RightForeArm` |
| `right_hand_roll_link` | `RightHand` |

每一项的结构是：

```json
"robot_body_name": [
  "human_body_name",
  position_weight,
  rotation_weight,
  position_offset_xyz,
  rotation_offset_quat_wxyz
]
```

两阶段的区别：

- `ik_match_table1` 更偏向先把姿态方向对齐，脚的位置也有一定权重。
- `ik_match_table2` 提高了根节点、脚、手臂等位置权重，用于把整体位置和末端约束进一步拉准。

## 4. GMR 每帧求 R2V2 qpos

代码位置：`general_motion_retargeting/motion_retarget.py`

`scripts/bvh_to_robot.py` 初始化：

```python
retargeter = GMR(
    src_human="bvh_lafan1",
    tgt_robot="r2v2",
    actual_human_height=actual_human_height,
)
```

GMR 初始化时会：

1. 根据 `ROBOT_XML_DICT["r2v2"]` 加载 R2V2 MuJoCo 模型。
2. 根据 `IK_CONFIG_DICT["bvh_lafan1"]["r2v2"]` 加载 IK 配置。
3. 用 `actual_human_height / human_height_assumption` 修正人体比例。
4. 根据 `ik_match_table1/2` 创建 `mink.FrameTask`。

每一帧转换时调用：

```python
qpos = retargeter.retarget(lafan1_data_frames[i])
```

内部流程是：

1. `update_targets()` 把人体数据转 numpy、按人体根节点局部坐标缩放、套用位置/旋转 offset，并写入 IK task target。
2. `mink.solve_ik()` 求解第一阶段任务，并把速度积分到当前 `configuration`。
3. 如果启用第二阶段，再对 `ik_match_table2` 重复 IK 求解。
4. 返回 `self.configuration.data.qpos.copy()`。

R2V2 的 `qpos` 约定来自 MuJoCo：

```text
qpos[:, 0:3]  = floating base position xyz
qpos[:, 3:7]  = floating base quaternion wxyz
qpos[:, 7:]   = R2V2 joints, order follows MuJoCo XML qpos order
```

关节顺序不要手写假设，应该以导出的 `dof_names` 为准；它由 `general_motion_retargeting/utils/motion_export.py` 从 MuJoCo model 里按 `qpos_adr` 排序提取。

这里的 DoF 不是从 `.pkl` 或 `.npz` 里“反推”出来的，而是在 IK 求解时直接产生的。流程是：

1. `update_targets()` 根据 LAFAN1 人体帧设置一组 R2V2 body 的目标位姿，例如脚、髋、膝、腰、手臂等。
2. `mink.solve_ik()` 在 R2V2 MuJoCo 模型上求解一组关节速度，使这些 body 尽量贴近目标。
3. `self.configuration.integrate_inplace(vel, dt)` 把求出来的关节速度积分进当前 MuJoCo configuration。
4. 积分后的完整机器人状态存在 `self.configuration.data.qpos`。
5. `retarget()` 返回这份 `qpos.copy()`。

也就是说，每一帧的 `dof_pos` 来源是：

```python
qpos = retargeter.retarget(lafan1_data_frames[i])
dof_pos_frame = qpos[7:]
```

这里 `qpos[:7]` 是 floating base，剩下的 `qpos[7:]` 才是 R2V2 的关节 DoF。`.pkl` 里保存的 `dof_pos` 和 `.npz` 里保存的 `dof_pos` 本质上是同一个东西：都是所有帧的 `qpos[:, 7:]`。

旧 `.pkl` 只保存这些基础字段：

```python
{
    "fps": motion_fps,
    "root_pos": qpos_array[:, :3],
    "root_rot": qpos_array[:, 3:7][:, [1, 2, 3, 0]],
    "dof_pos": qpos_array[:, 7:],
    "local_body_pos": local_body_pos,
    "link_body_list": body_names,
}
```

所以 `.pkl` 里没有 `dof_vel` 是正常的；`dof_vel` 是 `.npz` 导出阶段基于 `dof_pos` 和 `fps` 做差分补出来的派生量。

### DoF 是怎么对上的

这里的“对上”分两层：

第一层是人体到机器人的运动目标对齐。LAFAN1 的 bone 并不会直接变成 R2V2 的某个关节角，也就是说没有 `LeftLeg angle -> left_knee_joint` 这种一对一公式。GMR 做的是 body-level IK：`bvh_lafan1_to_r2v2.json` 只指定“哪个 R2V2 body 要去跟哪个 LAFAN1 bone 的位置/姿态”，例如：

```text
R2V2 left_ankle_roll_link  <-  LAFAN1 LeftFootMod
R2V2 right_ankle_roll_link <-  LAFAN1 RightFootMod
R2V2 waist_pitch_link      <-  LAFAN1 Spine2
R2V2 left_hand_roll_link   <-  LAFAN1 LeftHand
```

然后 `mink.solve_ik()` 在 R2V2 的所有可动关节上一起优化，求出一组 R2V2 自己的关节角，使这些 body 尽量贴近目标。所以 DoF 的来源是机器人 IK 解，不是人体关节角拷贝。

第二层是 R2V2 关节向量内部的顺序对齐。这个顺序完全由 R2V2 的 MuJoCo XML 决定：

```python
self.model = mj.MjModel.from_xml_path(self.xml_file)
...
qpos = self.configuration.data.qpos.copy()
```

MuJoCo 会根据 XML 里的 joint 定义建立 `qpos` 布局。R2V2 是 floating-base 机器人，所以：

```text
qpos[:3]  -> base 位置
qpos[3:7] -> base 四元数
qpos[7:]  -> XML 中各个可动关节的 qpos，按 MuJoCo 模型顺序排列
```

导出 `.npz` 时没有重新排序 DoF，只是直接切片：

```python
dof_pos = qpos[:, 7:]
```

为了知道 `dof_pos` 每一列对应哪个 R2V2 关节，`motion_export.py` 会从同一个 MuJoCo model 里提取 `dof_names`。提取时按 `model.jnt_qposadr` 排序，并跳过 floating base 的 free joint：

```python
for joint_id in range(model.njnt):
    joint_type = int(model.jnt_type[joint_id])
    joint_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, joint_id)
    qpos_adr = int(model.jnt_qposadr[joint_id])

    if joint_type == mj.mjtJoint.mjJNT_FREE:
        continue
    if joint_type in (mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE):
        dof_entries.append((qpos_adr, joint_name))

dof_entries.sort(key=lambda x: x[0])
dof_names = [name for _, name in dof_entries]
```

因此对齐关系是：

```text
dof_pos[t, 0]  <-> dof_names[0]
dof_pos[t, 1]  <-> dof_names[1]
...
dof_pos[t, k]  <-> dof_names[k]
```

回放时也不需要按名字再匹配一次，只要使用同一个 R2V2 XML，把这一整条向量按原顺序塞回 MuJoCo 即可：

```python
data.qpos[:3] = root_pos[i]
data.qpos[3:7] = root_rot_wxyz[i]
data.qpos[7:] = dof_pos[i]
```

所以 DoF 能对上的关键条件是：生成 `.npz` 和回放 `.npz` 使用的是同一个 R2V2 XML，或者至少 joint 数量和 `qpos` 顺序完全一致。如果改了 `r2v2_with_shell.xml` 的关节顺序、删了关节、加了关节，旧 `.npz` 的 `dof_pos` 就不能直接保证语义一致，需要重新导出；`scripts/npz_to_robot_video_offscreen.py` 也会检查 `dof_pos.shape[1]` 是否等于当前模型的 `model.nq - 7`。

## 5. qpos 打包成 npz

代码位置：`general_motion_retargeting/utils/motion_export.py`

`scripts/bvh_to_robot.py` 会收集所有帧：

```python
qpos_list.append(qpos.copy())
qpos_array = np.asarray(qpos_list, dtype=np.float32)
```

当保存格式是 `npz` 时：

```python
npz_data = build_npz_motion_data(
    retargeter.xml_file,
    qpos_array,
    motion_fps,
    fk_device="cpu",
)
np.savez(args.save_path, **npz_data)
```

`build_npz_motion_data()` 的输入不是原始 BVH，而是 GMR 已经求好的 R2V2 `qpos`。所以 `.npz` 转化可以理解为“把 MuJoCo 运动状态展开成下游更容易读的 numpy 字段”。

输入参数含义：

| 参数 | 说明 |
| --- | --- |
| `xml_file` | R2V2 的 MuJoCo XML，用来读取关节顺序，并对 `qpos` 做正运动学 |
| `qpos_list` | shape 为 `(T, 7 + D)` 的整段运动，`T` 是帧数，`D` 是 R2V2 的关节自由度 |
| `fps` | 输出运动帧率，用来计算 `dt` 和速度 |
| `fk_device` | 正运动学使用的设备，单条转换默认可以用 `cpu` |

第一步是把 MuJoCo `qpos` 拆开。MuJoCo 的 floating base 机器人把根节点和关节放在同一个向量里，R2V2 也是这个结构：

```python
qpos = np.asarray(qpos_list, dtype=np.float32)
dt = 1.0 / float(fps)

root_pos = qpos[:, :3]
root_rot_wxyz = _normalize_quat_wxyz(qpos[:, 3:7]).astype(np.float32)
root_rot_xyzw = _wxyz_to_xyzw(root_rot_wxyz).astype(np.float32)
dof_pos = qpos[:, 7:].astype(np.float32)
```

这里有一个容易混淆的点：MuJoCo `qpos` 里的根节点四元数是 `wxyz`，但有些训练或数据处理代码习惯用 `xyzw`。所以导出时同时保存两份：

| 字段 | 四元数顺序 | 用途 |
| --- | --- | --- |
| `root_rot_wxyz` | `w, x, y, z` | 和 MuJoCo `data.qpos[3:7]` 一致，回放时优先用它 |
| `root_rot` | `x, y, z, w` | 给习惯 `xyzw` 的下游代码使用 |

第二步是从位置差分得到关节速度。`dof_pos` 是每帧关节角，`dof_vel` 不是 IK 直接输出的，而是根据相邻帧做有限差分：

```python
dof_vel = finite_difference(dof_pos, dt)
```

`finite_difference()` 的策略是：

| 帧位置 | 速度计算方式 |
| --- | --- |
| 第一帧 | 用第 1 帧到第 2 帧的前向差分 |
| 最后一帧 | 用倒数第 2 帧到最后一帧的后向差分 |
| 中间帧 | 用前后两帧的中心差分 |

因此 `fps` 很重要：同一段 `dof_pos`，`fps` 越高，`dt` 越小，计算出的速度数值越大。

第三步是用 R2V2 模型做正运动学。单纯的 `qpos` 只包含根节点和关节角，不直接包含每个 body 的全局位置和全局姿态。为了让下游不必重新加载 MuJoCo，也能拿到每个连杆的位置和姿态，导出器会调用：

```python
kinematics_model = KinematicsModel(xml_file, device=fk_device)
body_pos_t, body_rot_xyzw_t = kinematics_model.forward_kinematics(
    torch.from_numpy(root_pos).to(device=fk_device, dtype=torch.float32),
    torch.from_numpy(root_rot_xyzw).to(device=fk_device, dtype=torch.float32),
    torch.from_numpy(dof_pos).to(device=fk_device, dtype=torch.float32),
)
```

这里传给 `KinematicsModel` 的根节点旋转是 `xyzw`，因为这个 kinematics 工具内部使用的是 `xyzw`。得到的 `body_rot_xyzw_t` 会再转回 `wxyz` 存入 `.npz`：

```python
body_pos = body_pos_t.detach().cpu().numpy().astype(np.float32)
body_rot_xyzw = body_rot_xyzw_t.detach().cpu().numpy().astype(np.float32)
body_rot_wxyz = _normalize_quat_wxyz(_xyzw_to_wxyz(body_rot_xyzw)).astype(np.float32)
```

第四步是补 body 速度。body 线速度直接对 `body_positions` 做有限差分；body 角速度会先计算相邻帧四元数的相对旋转，再转成旋转向量除以 `dt`：

```python
body_lin_vel = finite_difference(body_pos, dt)
body_ang_vel = quaternion_angular_velocity(body_rot_wxyz, dt)
```

`quaternion_angular_velocity()` 里会先检查相邻四元数的点积。如果点积小于 0，会把后一帧四元数取反，保证走较短旋转弧，减少四元数符号翻转造成的角速度尖峰。

第五步是写入名字表。`dof_names` 从 MuJoCo model 里按 `qpos_adr` 排序提取，跳过 floating base 的 free joint，只保留 hinge/slide 关节：

```python
model = mj.MjModel.from_xml_path(xml_file)
dof_names = np.array(extract_dof_names(model), dtype=object)
body_names = np.array(kinematics_model.body_names, dtype=object)
```

`dof_names[i]` 对应 `dof_pos[:, i]` 和 `dof_vel[:, i]`；`body_names[j]` 对应 `body_positions[:, j]`、`body_rotations[:, j]` 等 body 字段。这样下游读取 `.npz` 时不用猜 XML 里的关节顺序。

这里也可以换成按帧来理解：`.npz` 中每一个 frame 都有一整条 R2V2 的 DoF 信息。第 `t` 帧的关节角是 `dof_pos[t, :]`，它是一个长度为 `D` 的向量；第 `k` 个值对应的关节名是 `dof_names[k]`。如果要取某一帧某个关节的角度，可以按下面的方式理解：

```python
joint_name = dof_names[k]
joint_angle = dof_pos[t, k]
joint_velocity = dof_vel[t, k]
```

所以 `dof_pos` 不是只存一份全局关节列表，而是按时间展开后的二维数组：

```text
dof_pos =
[
  frame_0: [dof_0, dof_1, ..., dof_D-1],
  frame_1: [dof_0, dof_1, ..., dof_D-1],
  ...
  frame_T-1: [dof_0, dof_1, ..., dof_D-1],
]
```

最后，所有字段通过 `np.savez()` 写成一个 `.npz` 文件：

```python
return {
    "fps": np.array([fps], dtype=np.float32),
    "dt": np.array([dt], dtype=np.float32),
    "root_pos": root_pos.astype(np.float32),
    "root_rot": root_rot_xyzw.astype(np.float32),
    "root_rot_wxyz": root_rot_wxyz.astype(np.float32),
    "dof_pos": dof_pos.astype(np.float32),
    "dof_vel": dof_vel.astype(np.float32),
    "dof_positions": dof_pos.astype(np.float32),
    "dof_velocities": dof_vel.astype(np.float32),
    "body_positions": body_pos,
    "body_rotations": body_rot_wxyz,
    "body_linear_velocities": body_lin_vel,
    "body_angular_velocities": body_ang_vel,
    "dof_names": dof_names,
    "body_names": body_names,
}
```

其中 `dof_positions/dof_velocities` 是 `dof_pos/dof_vel` 的别名，主要是为了兼容不同下游命名习惯。

最终 `.npz` 字段：

| key | shape | 说明 |
| --- | --- | --- |
| `fps` | `(1,)` | 帧率 |
| `dt` | `(1,)` | `1 / fps` |
| `root_pos` | `(T, 3)` | R2V2 floating base 位置 |
| `root_rot` | `(T, 4)` | 根节点四元数，`xyzw` |
| `root_rot_wxyz` | `(T, 4)` | 根节点四元数，MuJoCo `qpos` 使用的 `wxyz` |
| `dof_pos` | `(T, D)` | R2V2 关节位置 |
| `dof_vel` | `(T, D)` | 关节速度 |
| `dof_positions` | `(T, D)` | `dof_pos` 别名 |
| `dof_velocities` | `(T, D)` | `dof_vel` 别名 |
| `body_positions` | `(T, B, 3)` | R2V2 各 body 的全局位置 |
| `body_rotations` | `(T, B, 4)` | R2V2 各 body 的全局四元数，`wxyz` |
| `body_linear_velocities` | `(T, B, 3)` | body 线速度 |
| `body_angular_velocities` | `(T, B, 3)` | body 角速度 |
| `dof_names` | `(D,)` | 关节名，顺序对应 `dof_pos` |
| `body_names` | `(B,)` | body 名，顺序对应 body 相关字段 |

读取时如果要直接塞回 MuJoCo `data.qpos`，只需要使用 `.npz` 里的三组核心字段：`root_pos`、`root_rot_wxyz`、`dof_pos`。最新的 `scripts/npz_to_robot_video_offscreen.py` 就是这么做的：

```python
data.qpos[:3] = root_pos[i]
data.qpos[3:7] = root_rot_wxyz[i]
data.qpos[7:] = dof_pos[i]
```

也就是说，`.npz` 里虽然保存了很多派生字段，但真正还原 MuJoCo 动作的最小字段是：

| 字段 | 还原到 MuJoCo 的位置 |
| --- | --- |
| `root_pos[i]` | `data.qpos[:3]` |
| `root_rot_wxyz[i]` | `data.qpos[3:7]` |
| `dof_pos[i]` | `data.qpos[7:]` |

`body_positions`、`body_rotations` 和速度字段主要服务于训练、分析、调试和不用 MuJoCo 的下游消费场景。

## 和原 GMR 流程的差异

原 GMR 的核心没有变：仍然是“人体 motion -> IK target -> MuJoCo qpos”。这次为了支持 LAFAN1 转 R2V2 npz，主要增加和改造了这些点：

1. 增加 R2V2 模型资产  
   `assets/r2_v2_with_shell_no_hand/r2v2_with_shell.xml`

2. 在 GMR 参数里注册 R2V2  
   `general_motion_retargeting/params.py` 增加 `r2v2` 的 XML、IK config、base body 和 viewer 距离。

3. 增加 LAFAN1 到 R2V2 的 IK 配置  
   `general_motion_retargeting/ik_configs/bvh_lafan1_to_r2v2.json`

4. 扩展 BVH 转换脚本支持 R2V2 和 npz  
   `scripts/bvh_to_robot.py` 增加 `--robot r2v2`、`--save_format`，并在 `.npz` 输出时调用 `build_npz_motion_data()`。

5. 新增统一 npz 导出器  
   `general_motion_retargeting/utils/motion_export.py` 负责从 qpos 生成下游更容易使用的 npz 字段，不再只保存旧的 `.pkl`。

6. 新增 npz 离屏预览脚本  
   `scripts/npz_to_robot_video_offscreen.py` 可验证 npz 的 `root_pos/root_rot_wxyz/dof_pos` 能否正确驱动 R2V2 模型。

## 常见检查

查看 npz 里有哪些字段：

```powershell
python -c "import numpy as np; d=np.load('output/r2v2_jumps1_subject2.npz', allow_pickle=True); print(d.files)"
```

查看 R2V2 关节顺序：

```powershell
python -c "import numpy as np; d=np.load('output/r2v2_jumps1_subject2.npz', allow_pickle=True); print(list(d['dof_names']))"
```

检查关节维度是否和模型匹配：

```powershell
python scripts/npz_to_robot_video_offscreen.py `
  --npz_file output/r2v2_jumps1_subject2.npz `
  --robot r2v2 `
  --video_path output/check.mp4 `
  --max_frames 120
```

如果维度不匹配，离屏脚本会报出：

```text
DoF mismatch for robot=r2v2: npz dof=..., model dof=...
```
