#!/usr/bin/env python3
"""Create a quality-check CSV template from generated robot motion files."""

from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


ISSUE_TYPES = [
    "OK",
    "FOOT_SLIDE",
    "GROUND_PENETRATION",
    "ARM_EXPLOSION",
    "TORSO_TILT_TOO_LARGE",
    "ROOT_DRIFT",
    "JUMP_LANDING_BAD",
    "UNUSABLE",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate quality-check template CSV.")
    parser.add_argument("--motion_dir", type=str, required=True, help="Folder containing generated .pkl files")
    parser.add_argument("--manifest", type=str, default="", help="Optional manifest CSV for category/source mapping")
    parser.add_argument("--output_csv", type=str, required=True, help="Output quality-check CSV")
    parser.add_argument("--pattern", type=str, default="*.pkl", help="Glob pattern for motion files")
    parser.add_argument("--no_recursive", action="store_true", help="Disable recursive scan")
    parser.add_argument("--min_frames", type=int, default=30, help="Minimum valid frame count")
    parser.add_argument(
        "--teleport_threshold",
        type=float,
        default=0.35,
        help="Max frame-to-frame XY translation before auto-flagging ROOT_DRIFT",
    )
    parser.add_argument(
        "--ground_threshold",
        type=float,
        default=-0.03,
        help="Root Z lower than this value will be auto-flagged as GROUND_PENETRATION",
    )
    parser.add_argument(
        "--dof_limit",
        type=float,
        default=4.5,
        help="Absolute DOF value larger than this will be auto-flagged as ARM_EXPLOSION",
    )
    return parser.parse_args()


def load_manifest_mapping(path: Path) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Dict[str, str]]]:
    mapping_by_motion_id: Dict[str, Dict[str, str]] = {}
    mapping_by_source_stem: Dict[str, Dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as infile:
        for row in csv.DictReader(infile):
            motion_id = (row.get("motion_id") or "").strip()
            source_path = (row.get("source_path") or "").strip()
            if motion_id:
                mapping_by_motion_id[motion_id] = row
            if source_path:
                mapping_by_source_stem[Path(source_path).stem] = row
    return mapping_by_motion_id, mapping_by_source_stem


def infer_category_from_path(path: Path) -> str:
    parent_name = path.parent.name.lower()
    if parent_name in {"walk", "jump", "other"}:
        return parent_name
    return "unknown"


def compute_metrics(path: Path) -> Dict[str, float | int | str]:
    with path.open("rb") as infile:
        motion_data = pickle.load(infile)

    root_pos = np.asarray(motion_data.get("root_pos", []), dtype=float)
    dof_pos = np.asarray(motion_data.get("dof_pos", []), dtype=float)
    fps_raw = motion_data.get("fps", 0.0)
    try:
        fps = float(fps_raw)
    except (TypeError, ValueError):
        fps = 0.0

    num_frames = int(root_pos.shape[0]) if root_pos.ndim >= 2 else 0
    duration_sec = float(num_frames / fps) if fps > 0 and num_frames > 0 else 0.0

    if num_frames >= 2:
        delta = np.diff(root_pos[:, :3], axis=0)
        xy_step = np.linalg.norm(delta[:, :2], axis=1)
        max_xy_step = float(np.max(xy_step))
        mean_xy_speed = float(np.mean(xy_step) * fps) if fps > 0 else 0.0
        max_z_step = float(np.max(np.abs(delta[:, 2])))
    else:
        max_xy_step = 0.0
        mean_xy_speed = 0.0
        max_z_step = 0.0

    if num_frames > 0:
        root_xy_disp = float(np.linalg.norm(root_pos[-1, :2] - root_pos[0, :2]))
        root_z_min = float(np.min(root_pos[:, 2]))
        root_z_max = float(np.max(root_pos[:, 2]))
    else:
        root_xy_disp = 0.0
        root_z_min = 0.0
        root_z_max = 0.0

    dof_abs_max = float(np.max(np.abs(dof_pos))) if dof_pos.size > 0 else 0.0
    has_nan = bool(
        (root_pos.size > 0 and np.isnan(root_pos).any()) or (dof_pos.size > 0 and np.isnan(dof_pos).any())
    )

    return {
        "num_frames": num_frames,
        "fps": fps,
        "duration_sec": duration_sec,
        "root_xy_displacement": root_xy_disp,
        "root_z_min": root_z_min,
        "root_z_max": root_z_max,
        "max_xy_step": max_xy_step,
        "mean_xy_speed": mean_xy_speed,
        "max_z_step": max_z_step,
        "dof_abs_max": dof_abs_max,
        "has_nan": int(has_nan),
    }


def auto_flags(metrics: Dict[str, float | int | str], args: argparse.Namespace) -> List[str]:
    flags: List[str] = []
    if int(metrics["has_nan"]) == 1:
        flags.append("UNUSABLE")
    if int(metrics["num_frames"]) < args.min_frames:
        flags.append("UNUSABLE")
    if float(metrics["max_xy_step"]) > args.teleport_threshold:
        flags.append("ROOT_DRIFT")
    if float(metrics["root_z_min"]) < args.ground_threshold:
        flags.append("GROUND_PENETRATION")
    if float(metrics["dof_abs_max"]) > args.dof_limit:
        flags.append("ARM_EXPLOSION")
    return sorted(set(flags))


def main() -> int:
    args = parse_args()
    motion_dir = Path(args.motion_dir).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()
    recursive = not args.no_recursive

    if not motion_dir.exists():
        raise FileNotFoundError(f"motion_dir does not exist: {motion_dir}")

    manifest_by_id: Dict[str, Dict[str, str]] = {}
    manifest_by_source_stem: Dict[str, Dict[str, str]] = {}
    if args.manifest:
        manifest_path = Path(args.manifest).expanduser().resolve()
        if not manifest_path.exists():
            raise FileNotFoundError(f"manifest does not exist: {manifest_path}")
        manifest_by_id, manifest_by_source_stem = load_manifest_mapping(manifest_path)

    if recursive:
        motion_files = sorted(motion_dir.rglob(args.pattern))
    else:
        motion_files = sorted(motion_dir.glob(args.pattern))

    motion_files = [path for path in motion_files if path.is_file()]
    if not motion_files:
        print(f"[quality] no files matched under {motion_dir} pattern={args.pattern}")
        return 0

    rows: List[Dict[str, str]] = []
    auto_flag_counter: Dict[str, int] = {issue: 0 for issue in ISSUE_TYPES}

    for motion_path in motion_files:
        motion_name = motion_path.stem
        manifest_row = manifest_by_id.get(motion_name) or manifest_by_source_stem.get(motion_name) or {}
        category = (manifest_row.get("category") or "").strip().lower() or infer_category_from_path(motion_path)
        source_path = (manifest_row.get("source_path") or "").strip()

        try:
            metrics = compute_metrics(motion_path)
            flags = auto_flags(metrics, args=args)
            issue_type = "|".join(flags) if flags else "OK"
            for flag in (flags or ["OK"]):
                if flag in auto_flag_counter:
                    auto_flag_counter[flag] += 1
            comments = "" if not flags else "AUTO_FLAGGED"
        except Exception as exc:  # pylint: disable=broad-except
            metrics = {
                "num_frames": 0,
                "fps": 0.0,
                "duration_sec": 0.0,
                "root_xy_displacement": 0.0,
                "root_z_min": 0.0,
                "root_z_max": 0.0,
                "max_xy_step": 0.0,
                "mean_xy_speed": 0.0,
                "max_z_step": 0.0,
                "dof_abs_max": 0.0,
                "has_nan": 0,
            }
            issue_type = "UNUSABLE"
            comments = f"LOAD_ERROR: {type(exc).__name__}: {exc}"
            auto_flag_counter["UNUSABLE"] += 1

        row = {
            "file_name": motion_path.name,
            "category": category,
            "result": "PENDING",
            "issue_type": issue_type,
            "comments": comments,
            "motion_name": motion_name,
            "file_path": str(motion_path),
            "source_path": source_path,
            "num_frames": str(int(metrics["num_frames"])),
            "fps": f"{float(metrics['fps']):.6f}",
            "duration_sec": f"{float(metrics['duration_sec']):.6f}",
            "root_xy_displacement": f"{float(metrics['root_xy_displacement']):.6f}",
            "root_z_min": f"{float(metrics['root_z_min']):.6f}",
            "root_z_max": f"{float(metrics['root_z_max']):.6f}",
            "max_xy_step": f"{float(metrics['max_xy_step']):.6f}",
            "mean_xy_speed": f"{float(metrics['mean_xy_speed']):.6f}",
            "max_z_step": f"{float(metrics['max_z_step']):.6f}",
            "dof_abs_max": f"{float(metrics['dof_abs_max']):.6f}",
            "has_nan": str(int(metrics["has_nan"])),
        }
        rows.append(row)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "file_name",
        "category",
        "result",
        "issue_type",
        "comments",
        "motion_name",
        "file_path",
        "source_path",
        "num_frames",
        "fps",
        "duration_sec",
        "root_xy_displacement",
        "root_z_min",
        "root_z_max",
        "max_xy_step",
        "mean_xy_speed",
        "max_z_step",
        "dof_abs_max",
        "has_nan",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"[quality] scanned={len(motion_files)} output={output_csv}")
    print(
        "[quality] auto_flags="
        + ", ".join(f"{issue}:{count}" for issue, count in auto_flag_counter.items() if count > 0)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
