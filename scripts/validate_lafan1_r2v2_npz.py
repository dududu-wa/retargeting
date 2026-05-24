import argparse
import csv
import sys
from pathlib import Path

import mujoco as mj
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from general_motion_retargeting.params import ROBOT_XML_DICT
from general_motion_retargeting.utils.lafan1 import load_bvh_file


def unit(v):
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(norm, 1e-8)


def angle_deg(a, b):
    cos = np.sum(unit(a) * unit(b), axis=-1)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def joint_angle_deg(a, b, c):
    return angle_deg(a - b, c - b)


def bvh_positions(frames, names):
    out = {name: [] for name in names}
    for frame in frames:
        for name in names:
            out[name].append(np.asarray(frame[name][0], dtype=np.float64))
    return {name: np.asarray(values) for name, values in out.items()}


def robot_positions(model, data, root_pos, root_rot, dof_pos, names):
    out = {name: np.empty((len(root_pos), 3), dtype=np.float64) for name in names}
    ids = {name: model.body(name).id for name in names}
    for i in range(len(root_pos)):
        data.qpos[:3] = root_pos[i]
        data.qpos[3:7] = root_rot[i]
        data.qpos[7:] = dof_pos[i]
        mj.mj_forward(model, data)
        for name, body_id in ids.items():
            out[name][i] = data.xpos[body_id]
    return out


def percentile(x, q):
    if len(x) == 0:
        return float("nan")
    return float(np.percentile(x, q))


def summarize_file(bvh_path, npz_path, model, data):
    frames, _ = load_bvh_file(str(bvh_path), format="lafan1")
    npz = np.load(npz_path, allow_pickle=True)
    root_pos = np.asarray(npz["root_pos"], dtype=np.float64)
    root_rot = np.asarray(npz["root_rot_wxyz"], dtype=np.float64)
    dof_pos = np.asarray(npz["dof_pos"], dtype=np.float64)
    n = min(len(frames), len(root_pos))

    frames = frames[:n]
    root_pos = root_pos[:n]
    root_rot = root_rot[:n]
    dof_pos = dof_pos[:n]

    human_names = [
        "Hips", "Spine2", "Head",
        "LeftArm", "LeftForeArm", "LeftHand",
        "RightArm", "RightForeArm", "RightHand",
        "LeftLeg", "RightLeg", "LeftFoot", "RightFoot",
    ]
    robot_names = [
        "base_link", "waist_pitch_link", "head_pitch_link",
        "left_shoulder_yaw_link", "left_arm_pitch_link", "left_hand_roll_link",
        "right_shoulder_yaw_link", "right_arm_pitch_link", "right_hand_roll_link",
        "left_knee_link", "right_knee_link", "left_ankle_roll_link", "right_ankle_roll_link",
    ]
    hp = bvh_positions(frames, human_names)
    rp = robot_positions(model, data, root_pos, root_rot, dof_pos, robot_names)

    torso_err = angle_deg(hp["Spine2"] - hp["Hips"], rp["waist_pitch_link"] - rp["base_link"])
    head_dir_err = angle_deg(hp["Head"] - hp["Hips"], rp["head_pitch_link"] - rp["base_link"])

    left_elbow_err = np.abs(
        joint_angle_deg(hp["LeftArm"], hp["LeftForeArm"], hp["LeftHand"])
        - joint_angle_deg(rp["left_shoulder_yaw_link"], rp["left_arm_pitch_link"], rp["left_hand_roll_link"])
    )
    right_elbow_err = np.abs(
        joint_angle_deg(hp["RightArm"], hp["RightForeArm"], hp["RightHand"])
        - joint_angle_deg(rp["right_shoulder_yaw_link"], rp["right_arm_pitch_link"], rp["right_hand_roll_link"])
    )
    left_forearm_err = angle_deg(
        hp["LeftHand"] - hp["LeftForeArm"],
        rp["left_hand_roll_link"] - rp["left_arm_pitch_link"],
    )
    right_forearm_err = angle_deg(
        hp["RightHand"] - hp["RightForeArm"],
        rp["right_hand_roll_link"] - rp["right_arm_pitch_link"],
    )

    bvh_foot_z = np.minimum(hp["LeftFoot"][:, 2], hp["RightFoot"][:, 2])
    bvh_knee_rel = np.minimum(hp["LeftLeg"][:, 2], hp["RightLeg"][:, 2]) - bvh_foot_z
    bvh_hip_rel = hp["Hips"][:, 2] - bvh_foot_z
    bvh_head_rel = hp["Head"][:, 2] - bvh_foot_z

    robot_foot_z = np.minimum(rp["left_ankle_roll_link"][:, 2], rp["right_ankle_roll_link"][:, 2])
    robot_knee_rel = np.minimum(rp["left_knee_link"][:, 2], rp["right_knee_link"][:, 2]) - robot_foot_z
    robot_base_rel = rp["base_link"][:, 2] - robot_foot_z
    robot_head_rel = rp["head_pitch_link"][:, 2] - robot_foot_z

    # Proxy checks compare robot low poses against BVH low poses so real source
    # crouch or landing frames are not counted as retargeting failures.
    bvh_low = (bvh_head_rel < 0.70) | (bvh_hip_rel < 0.25) | (bvh_knee_rel < 0.15)
    robot_fall = (robot_head_rel < 0.70) | ((head_dir_err > 65.0) & (robot_head_rel < 1.0))
    robot_kneel = (robot_knee_rel < 0.10) & (robot_base_rel < 0.55)
    bad_vs_bvh = (robot_fall | robot_kneel) & ~bvh_low

    qpos = np.concatenate([root_pos, root_rot, dof_pos], axis=1)
    limit_margin = []
    near_limit = []
    for joint_id in range(1, model.njnt):
        qadr = int(model.jnt_qposadr[joint_id])
        lo, hi = model.jnt_range[joint_id]
        values = qpos[:, qadr]
        span = max(float(hi - lo), 1e-8)
        limit_margin.append(float(np.min(np.minimum(values - lo, hi - values))))
        near_limit.append(float(np.mean((values - lo < 0.05 * span) | (hi - values < 0.05 * span))))

    return {
        "file": bvh_path.stem,
        "bvh_frames": len(frames),
        "npz_frames": len(root_pos),
        "dof": dof_pos.shape[1],
        "finite": bool(np.isfinite(qpos).all()),
        "min_limit_margin_rad": min(limit_margin),
        "max_near_limit_ratio": max(near_limit),
        "torso_mae_deg": float(np.mean(torso_err)),
        "torso_p95_deg": percentile(torso_err, 95),
        "head_dir_mae_deg": float(np.mean(head_dir_err)),
        "head_dir_p95_deg": percentile(head_dir_err, 95),
        "left_elbow_mae_deg": float(np.mean(left_elbow_err)),
        "left_elbow_p95_deg": percentile(left_elbow_err, 95),
        "right_elbow_mae_deg": float(np.mean(right_elbow_err)),
        "right_elbow_p95_deg": percentile(right_elbow_err, 95),
        "left_forearm_mae_deg": float(np.mean(left_forearm_err)),
        "left_forearm_p95_deg": percentile(left_forearm_err, 95),
        "right_forearm_mae_deg": float(np.mean(right_forearm_err)),
        "right_forearm_p95_deg": percentile(right_forearm_err, 95),
        "robot_base_rel_min_m": float(np.min(robot_base_rel)),
        "robot_base_rel_p1_m": percentile(robot_base_rel, 1),
        "robot_knee_rel_min_m": float(np.min(robot_knee_rel)),
        "robot_knee_rel_p1_m": percentile(robot_knee_rel, 1),
        "bvh_low_count": int(np.sum(bvh_low)),
        "bad_vs_bvh_count": int(np.sum(bad_vs_bvh)),
        "robot_fall_count": int(np.sum(robot_fall)),
        "robot_kneel_count": int(np.sum(robot_kneel)),
        "worst_bad_frame": int(np.argmax(bad_vs_bvh)) if np.any(bad_vs_bvh) else -1,
    }


def main():
    parser = argparse.ArgumentParser(description="Validate LAFAN1 BVH to R2V2 npz retargeting outputs.")
    parser.add_argument("--bvh_dir", default="intput/lafan1")
    parser.add_argument("--npz_dir", default="output/output_npz/lafan1_r2v2")
    parser.add_argument("--csv_path", default=None)
    args = parser.parse_args()

    bvh_dir = Path(args.bvh_dir)
    npz_dir = Path(args.npz_dir)
    bvh_files = sorted(
        p for p in bvh_dir.glob("*.bvh")
        if p.name.startswith(("walk", "run", "jumps"))
    )
    model = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT["r2v2"]))
    data = mj.MjData(model)

    rows = []
    for bvh_path in bvh_files:
        npz_path = npz_dir / f"{bvh_path.stem}.npz"
        if not npz_path.exists():
            rows.append({"file": bvh_path.stem, "missing_npz": True})
            print(f"{bvh_path.stem}: missing npz")
            continue
        row = summarize_file(bvh_path, npz_path, model, data)
        rows.append(row)
        print(
            f"{row['file']}: bad={row['bad_vs_bvh_count']} fall={row['robot_fall_count']} "
            f"kneel={row['robot_kneel_count']} torso={row['torso_mae_deg']:.2f}/{row['torso_p95_deg']:.2f} "
            f"elbowL={row['left_elbow_mae_deg']:.2f}/{row['left_elbow_p95_deg']:.2f} "
            f"elbowR={row['right_elbow_mae_deg']:.2f}/{row['right_elbow_p95_deg']:.2f}"
        )

    if args.csv_path:
        csv_path = Path(args.csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted({key for row in rows for key in row.keys()})
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        print(f"saved {csv_path}")

    bad_total = sum(row.get("bad_vs_bvh_count", 0) for row in rows)
    missing = sum(1 for row in rows if row.get("missing_npz"))
    print(f"summary: files={len(rows)} missing={missing} bad_vs_bvh_total={bad_total}")
    if missing or bad_total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
