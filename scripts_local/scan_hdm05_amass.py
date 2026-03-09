#!/usr/bin/env python3
"""Scan AMASS MPI-HDM05 motions and export a metadata CSV."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


DEFAULT_SCAN_EXTENSIONS = (".npz",)
REQUIRED_KEYS = ("pose_body", "root_orient", "trans", "betas")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan AMASS MPI-HDM05 files into CSV.")
    parser.add_argument("--input_dir", type=str, required=True, help="Root folder of MPI-HDM05 files.")
    parser.add_argument("--output_csv", type=str, required=True, help="Output metadata CSV path.")
    parser.add_argument(
        "--output_errors",
        type=str,
        default="",
        help="Optional CSV for unreadable files and parse errors.",
    )
    parser.add_argument(
        "--extensions",
        type=str,
        nargs="+",
        default=list(DEFAULT_SCAN_EXTENSIONS),
        help="File extensions to scan. Default: .npz",
    )
    parser.add_argument("--id_prefix", type=str, default="MPI_HDM05", help="Prefix for generated motion_id.")
    parser.add_argument(
        "--skip_stagei",
        action="store_true",
        help="Skip files with `_stagei` in filename.",
    )
    return parser.parse_args()


def as_scalar(value):
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        if value.size == 1:
            return value.reshape(-1)[0].item()
        return value
    try:
        return value.item()
    except AttributeError:
        return value


def as_float(value) -> float | None:
    if value is None:
        return None
    scalar = as_scalar(value)
    try:
        return float(scalar)
    except (TypeError, ValueError):
        return None


def as_text(value) -> str:
    if value is None:
        return ""
    scalar = as_scalar(value)
    if isinstance(scalar, bytes):
        return scalar.decode("utf-8", errors="ignore")
    return str(scalar)


def guess_category(path_text: str) -> Tuple[str, str, str]:
    text = path_text.lower()
    walk_keywords = ("walk", "locomotion", "step", "turn", "sidestep")
    jump_keywords = ("jump", "hop", "jacks", "leap")
    exclude_keywords = ("crawl", "roll", "flip", "cartwheel", "handstand", "_lie", "lie_")

    if any(keyword in text for keyword in exclude_keywords):
        return "exclude", "low", "contains high-risk motion keywords"
    if any(keyword in text for keyword in jump_keywords):
        return "jump", "high", ""
    if any(keyword in text for keyword in walk_keywords):
        return "walk", "high", ""
    return "other", "medium", ""


def find_first_frame_count(data: np.lib.npyio.NpzFile) -> int | None:
    frame_keys = ("pose_body", "root_orient", "trans", "poses")
    for key in frame_keys:
        if key not in data.files:
            continue
        value = data[key]
        if isinstance(value, np.ndarray) and value.ndim > 0:
            return int(value.shape[0])
    return None


def find_fps(data: np.lib.npyio.NpzFile) -> float | None:
    for key in ("mocap_frame_rate", "mocap_framerate", "fps", "frame_rate"):
        if key in data.files:
            return as_float(data[key])
    return None


def sorted_files(root: Path, extensions: Iterable[str]) -> List[Path]:
    ext_set = {ext.lower() for ext in extensions}
    files = [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in ext_set]
    return sorted(files)


def scan_file(path: Path, input_dir: Path, index: int, id_prefix: str) -> Dict[str, str]:
    npz = np.load(path, allow_pickle=True)
    try:
        keys = sorted(npz.files)
        num_frames = find_first_frame_count(npz)
        fps = find_fps(npz)
        duration_sec = ""
        if fps and fps > 0 and num_frames:
            duration_sec = f"{num_frames / fps:.4f}"
        gender = as_text(npz["gender"]) if "gender" in npz.files else ""
        relative_path = path.relative_to(input_dir).as_posix()
        path_text = f"{relative_path} {path.stem}"
        category_guess, priority_guess, rule_note = guess_category(path_text)
        is_stagei = "_stagei" in path.stem.lower()
        missing_required = [key for key in REQUIRED_KEYS if key not in npz.files]

        notes = []
        if rule_note:
            notes.append(rule_note)
        if is_stagei:
            notes.append("contains _stagei")
        if missing_required:
            notes.append(f"missing required keys: {','.join(missing_required)}")

        return {
            "motion_id": f"{id_prefix}_{index:06d}",
            "source_path": str(path.resolve()),
            "relative_path": relative_path,
            "file_name": path.name,
            "file_stem": path.stem,
            "category_guess": category_guess,
            "priority_guess": priority_guess,
            "num_frames": "" if num_frames is None else str(num_frames),
            "fps": "" if fps is None else f"{fps:.6f}",
            "duration_sec": duration_sec,
            "gender": gender,
            "is_stagei": "1" if is_stagei else "0",
            "has_pose_body": "1" if "pose_body" in npz.files else "0",
            "has_root_orient": "1" if "root_orient" in npz.files else "0",
            "has_trans": "1" if "trans" in npz.files else "0",
            "has_betas": "1" if "betas" in npz.files else "0",
            "key_count": str(len(keys)),
            "keys": "|".join(keys),
            "status": "ok",
            "notes": "; ".join(notes),
        }
    finally:
        npz.close()


def write_csv(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()
    output_errors = Path(args.output_errors).expanduser().resolve() if args.output_errors else None

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    files = sorted_files(input_dir, args.extensions)
    print(f"[scan] input_dir={input_dir}")
    print(f"[scan] found_files={len(files)} extensions={args.extensions}")

    rows: List[Dict[str, str]] = []
    errors: List[Dict[str, str]] = []
    category_counter: Counter[str] = Counter()

    for index, path in enumerate(files, start=1):
        if args.skip_stagei and "_stagei" in path.stem.lower():
            continue
        try:
            row = scan_file(path=path, input_dir=input_dir, index=index, id_prefix=args.id_prefix)
            rows.append(row)
            category_counter[row["category_guess"]] += 1
        except Exception as exc:  # pylint: disable=broad-except
            errors.append(
                {
                    "source_path": str(path.resolve()),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )

    metadata_fields = [
        "motion_id",
        "source_path",
        "relative_path",
        "file_name",
        "file_stem",
        "category_guess",
        "priority_guess",
        "num_frames",
        "fps",
        "duration_sec",
        "gender",
        "is_stagei",
        "has_pose_body",
        "has_root_orient",
        "has_trans",
        "has_betas",
        "key_count",
        "keys",
        "status",
        "notes",
    ]
    write_csv(output_csv, rows, metadata_fields)

    if output_errors:
        error_fields = ["source_path", "error_type", "error_message"]
        write_csv(output_errors, errors, error_fields)

    print(f"[scan] wrote metadata: {output_csv} ({len(rows)} rows)")
    if output_errors:
        print(f"[scan] wrote errors:   {output_errors} ({len(errors)} rows)")
    if rows:
        print(
            "[scan] category_counts="
            + ", ".join(f"{key}:{category_counter[key]}" for key in sorted(category_counter.keys()))
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
