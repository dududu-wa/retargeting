#!/usr/bin/env python3
"""Batch-convert manifest-listed SMPL-X motions to Unitree G1 robot motions."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List


@dataclass
class ConversionItem:
    index: int
    motion_id: str
    source_path: Path
    category: str
    priority: str
    notes: str
    target_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch convert motions from manifest CSV.")
    parser.add_argument("--manifest_csv", type=str, required=True, help="Manifest CSV path")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for .pkl files")
    parser.add_argument(
        "--gmr_root",
        type=str,
        default="",
        help="GMR root directory. Default: inferred from this script location.",
    )
    parser.add_argument(
        "--robot",
        type=str,
        default="unitree_g1_29dof_lock_waist_fixed_hand_feet_inertia",
        help="Target robot name",
    )
    parser.add_argument("--python_bin", type=str, default=sys.executable, help="Python executable used for conversion")
    parser.add_argument("--limit", type=int, default=0, help="Process only first N rows (0 means all)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing target .pkl files")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without executing")
    parser.add_argument(
        "--stop_on_error",
        action="store_true",
        help="Stop immediately on first conversion error. Default behavior is continue.",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="",
        help="Log directory. Default: <gmr_root>/logs/batch_convert_<timestamp>",
    )
    return parser.parse_args()


def sanitize_motion_id(raw_motion_id: str, source_path: Path, index: int) -> str:
    motion_id = raw_motion_id.strip() if raw_motion_id else source_path.stem
    if not motion_id:
        motion_id = f"MOTION_{index:06d}"
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", motion_id)


def normalize_category(raw_category: str) -> str:
    category = (raw_category or "").strip().lower()
    if not category:
        return "uncategorized"
    return re.sub(r"[^a-z0-9_.-]+", "_", category)


def build_items(manifest_rows: Iterable[Dict[str, str]], output_dir: Path) -> List[ConversionItem]:
    items: List[ConversionItem] = []
    for index, row in enumerate(manifest_rows, start=1):
        source_raw = (row.get("source_path") or "").strip()
        if not source_raw:
            continue
        source_path = Path(source_raw).expanduser().resolve()
        motion_id = sanitize_motion_id(row.get("motion_id", ""), source_path=source_path, index=index)
        category = normalize_category(row.get("category", ""))
        target_path = output_dir / category / f"{motion_id}.pkl"
        items.append(
            ConversionItem(
                index=index,
                motion_id=motion_id,
                source_path=source_path,
                category=category,
                priority=(row.get("priority") or "").strip(),
                notes=(row.get("notes") or "").strip(),
                target_path=target_path,
            )
        )
    return items


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> int:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    gmr_root = Path(args.gmr_root).expanduser().resolve() if args.gmr_root else script_dir.parent
    manifest_csv = Path(args.manifest_csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_csv.exists():
        raise FileNotFoundError(f"Manifest CSV not found: {manifest_csv}")

    convert_script = gmr_root / "scripts" / "smplx_to_robot.py"
    if not convert_script.exists():
        raise FileNotFoundError(f"Conversion script not found: {convert_script}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.log_dir).expanduser().resolve() if args.log_dir else (gmr_root / "logs" / f"batch_convert_{timestamp}")
    stdout_dir = log_dir / "stdout"
    stderr_dir = log_dir / "stderr"
    stdout_dir.mkdir(parents=True, exist_ok=True)
    stderr_dir.mkdir(parents=True, exist_ok=True)

    with manifest_csv.open("r", encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    items = build_items(rows, output_dir=output_dir)
    if args.limit > 0:
        items = items[: args.limit]

    if not items:
        print("[convert] no valid rows in manifest, nothing to do")
        return 0

    summary_rows: List[Dict[str, str]] = []
    success_count = 0
    fail_count = 0
    skipped_count = 0

    print(f"[convert] manifest={manifest_csv}")
    print(f"[convert] output_dir={output_dir}")
    print(f"[convert] robot={args.robot} total_items={len(items)}")
    print(f"[convert] log_dir={log_dir}")

    for position, item in enumerate(items, start=1):
        started = time.time()
        stdout_log = stdout_dir / f"{position:05d}_{item.motion_id}.log"
        stderr_log = stderr_dir / f"{position:05d}_{item.motion_id}.log"
        status = "success"
        error_message = ""
        return_code = 0

        if not item.source_path.exists():
            status = "failed_missing_source"
            error_message = f"source path does not exist: {item.source_path}"
            fail_count += 1
            write_text(stderr_log, error_message + "\n")
            print(f"[{position}/{len(items)}] {status} motion_id={item.motion_id}")
        elif item.target_path.exists() and not args.overwrite:
            status = "skipped_existing"
            skipped_count += 1
            print(f"[{position}/{len(items)}] {status} motion_id={item.motion_id}")
        else:
            item.target_path.parent.mkdir(parents=True, exist_ok=True)
            command = [
                args.python_bin,
                str(convert_script),
                "--smplx_file",
                str(item.source_path),
                "--robot",
                args.robot,
                "--save_path",
                str(item.target_path),
                "--headless",
            ]
            print(f"[{position}/{len(items)}] running motion_id={item.motion_id}")
            print("  " + " ".join(command))
            if args.dry_run:
                status = "dry_run"
                skipped_count += 1
            else:
                proc = subprocess.run(command, capture_output=True, text=True, check=False)
                return_code = proc.returncode
                write_text(stdout_log, proc.stdout)
                write_text(stderr_log, proc.stderr)
                if proc.returncode == 0 and item.target_path.exists():
                    success_count += 1
                    status = "success"
                else:
                    fail_count += 1
                    status = "failed_runtime"
                    error_message = f"return_code={proc.returncode}"
                    if not item.target_path.exists():
                        error_message += " target_not_generated"
                    print(f"  [error] {error_message}")
                    if args.stop_on_error:
                        elapsed = time.time() - started
                        summary_rows.append(
                            {
                                "index": str(position),
                                "motion_id": item.motion_id,
                                "category": item.category,
                                "source_path": str(item.source_path),
                                "target_path": str(item.target_path),
                                "status": status,
                                "return_code": str(return_code),
                                "error_message": error_message,
                                "duration_sec": f"{elapsed:.3f}",
                                "stdout_log": str(stdout_log),
                                "stderr_log": str(stderr_log),
                            }
                        )
                        break

        elapsed = time.time() - started
        summary_rows.append(
            {
                "index": str(position),
                "motion_id": item.motion_id,
                "category": item.category,
                "source_path": str(item.source_path),
                "target_path": str(item.target_path),
                "status": status,
                "return_code": str(return_code),
                "error_message": error_message,
                "duration_sec": f"{elapsed:.3f}",
                "stdout_log": str(stdout_log),
                "stderr_log": str(stderr_log),
            }
        )

    summary_path = log_dir / "summary.csv"
    fieldnames = [
        "index",
        "motion_id",
        "category",
        "source_path",
        "target_path",
        "status",
        "return_code",
        "error_message",
        "duration_sec",
        "stdout_log",
        "stderr_log",
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as out_csv:
        writer = csv.DictWriter(out_csv, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    print(
        f"[convert] done success={success_count} failed={fail_count} skipped={skipped_count} summary={summary_path}"
    )
    return 1 if fail_count > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
