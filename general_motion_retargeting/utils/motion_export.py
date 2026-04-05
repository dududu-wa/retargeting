import numpy as np
import torch
import mujoco as mj

from general_motion_retargeting.kinematics_model import KinematicsModel


def _normalize_quat_wxyz(quat_wxyz):
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64)
    norm = np.linalg.norm(quat_wxyz, axis=-1, keepdims=True)
    norm = np.clip(norm, 1e-8, None)
    return quat_wxyz / norm


def _xyzw_to_wxyz(quat_xyzw):
    quat_xyzw = np.asarray(quat_xyzw)
    return quat_xyzw[..., [3, 0, 1, 2]]


def _wxyz_to_xyzw(quat_wxyz):
    quat_wxyz = np.asarray(quat_wxyz)
    return quat_wxyz[..., [1, 2, 3, 0]]


def _quat_conjugate_wxyz(quat_wxyz):
    out = np.array(quat_wxyz, copy=True)
    out[..., 1:] *= -1.0
    return out


def _quat_mul_wxyz(q1, q2):
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def _quat_to_rotvec_wxyz(quat_wxyz):
    quat_wxyz = _normalize_quat_wxyz(quat_wxyz)
    w = np.clip(quat_wxyz[..., 0], -1.0, 1.0)
    xyz = quat_wxyz[..., 1:]
    sin_half = np.linalg.norm(xyz, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(sin_half[..., 0], w)

    small = sin_half[..., 0] < 1e-8
    axis = np.zeros_like(xyz)
    axis[~small] = xyz[~small] / sin_half[~small]
    axis[small] = np.array([1.0, 0.0, 0.0], dtype=axis.dtype)

    rotvec = axis * angle[..., None]
    rotvec[small] = 2.0 * xyz[small]
    return rotvec


def finite_difference(values, dt):
    values = np.asarray(values, dtype=np.float64)
    vel = np.zeros_like(values)
    if values.shape[0] < 2:
        return vel.astype(np.float32)

    vel[0] = (values[1] - values[0]) / dt
    vel[-1] = (values[-1] - values[-2]) / dt
    if values.shape[0] > 2:
        vel[1:-1] = (values[2:] - values[:-2]) / (2.0 * dt)
    return vel.astype(np.float32)


def quaternion_angular_velocity(body_rot_wxyz, dt):
    body_rot_wxyz = _normalize_quat_wxyz(body_rot_wxyz)
    num_frames = body_rot_wxyz.shape[0]
    ang_vel = np.zeros(body_rot_wxyz.shape[:-1] + (3,), dtype=np.float64)
    if num_frames < 2:
        return ang_vel.astype(np.float32)

    q_curr = body_rot_wxyz[:-1]
    q_next = body_rot_wxyz[1:]

    # Keep interpolation on the shorter arc to reduce artificial spikes.
    flip_mask = np.sum(q_curr * q_next, axis=-1, keepdims=True) < 0.0
    q_next = np.where(flip_mask, -q_next, q_next)

    q_rel = _quat_mul_wxyz(q_next, _quat_conjugate_wxyz(q_curr))
    interval_omega = _quat_to_rotvec_wxyz(q_rel) / dt

    ang_vel[0] = interval_omega[0]
    ang_vel[-1] = interval_omega[-1]
    if num_frames > 2:
        ang_vel[1:-1] = 0.5 * (interval_omega[:-1] + interval_omega[1:])

    return ang_vel.astype(np.float32)


def extract_dof_names(model):
    dof_entries = []
    for joint_id in range(model.njnt):
        joint_type = int(model.jnt_type[joint_id])
        joint_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, joint_id)
        qpos_adr = int(model.jnt_qposadr[joint_id])

        if joint_type == mj.mjtJoint.mjJNT_FREE:
            continue
        if joint_type in (mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE):
            dof_entries.append((qpos_adr, joint_name))

    dof_entries.sort(key=lambda x: x[0])
    return [name for _, name in dof_entries]


def build_npz_motion_data(xml_file, qpos_list, fps, fk_device="cpu"):
    qpos = np.asarray(qpos_list, dtype=np.float32)
    if qpos.ndim != 2 or qpos.shape[1] < 8:
        raise ValueError("qpos_list must be shaped [num_frames, 7 + dof]")

    dt = 1.0 / float(fps)

    root_pos = qpos[:, :3]
    root_rot_wxyz = _normalize_quat_wxyz(qpos[:, 3:7]).astype(np.float32)
    root_rot_xyzw = _wxyz_to_xyzw(root_rot_wxyz).astype(np.float32)
    dof_pos = qpos[:, 7:].astype(np.float32)
    dof_vel = finite_difference(dof_pos, dt)

    kinematics_model = KinematicsModel(xml_file, device=fk_device)
    with torch.no_grad():
        body_pos_t, body_rot_xyzw_t = kinematics_model.forward_kinematics(
            torch.from_numpy(root_pos).to(device=fk_device, dtype=torch.float32),
            torch.from_numpy(root_rot_xyzw).to(device=fk_device, dtype=torch.float32),
            torch.from_numpy(dof_pos).to(device=fk_device, dtype=torch.float32),
        )

    body_pos = body_pos_t.detach().cpu().numpy().astype(np.float32)
    body_rot_xyzw = body_rot_xyzw_t.detach().cpu().numpy().astype(np.float32)
    body_rot_wxyz = _normalize_quat_wxyz(_xyzw_to_wxyz(body_rot_xyzw)).astype(np.float32)

    body_lin_vel = finite_difference(body_pos, dt)
    body_ang_vel = quaternion_angular_velocity(body_rot_wxyz, dt)

    model = mj.MjModel.from_xml_path(xml_file)
    dof_names = np.array(extract_dof_names(model), dtype=object)
    if dof_names.shape[0] != dof_pos.shape[1]:
        dof_names = np.array([f"dof_{i}" for i in range(dof_pos.shape[1])], dtype=object)

    body_names = np.array(kinematics_model.body_names, dtype=object)

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
