#!/usr/bin/env python3
"""Build a first-batch manifest from AMASS scan metadata."""

from __future__ import annotations

import argparse
import csv
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List


PRIORITY_RANK = {"high": 0, "medium": 1, "low": 2}
DEFAULT_EXCLUDE_KEYWORDS = (
    "crawl",
    "_lie",
    "lie_",
    "upstairs",
    "downstairs",
    "flip",
    "cartwheel",
    "handstand",
    "roll",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate first-batch manifest CSV from scan CSV.")
    parser.add_argument("--scan_csv", type=str, required=True, help="Input CSV from scan_hdm05_amass.py")
    parser.add_argument("--output_csv", type=str, required=True, help="Output manifest CSV")
    parser.add_argument("--walk_count", type=int, default=30, help="Target number of walk clips")
    parser.add_argument("--jump_count", type=int, default=15, help="Target number of jump clips")
    parser.add_argument("--other_count", type=int, default=15, help="Target number of other clips")
    parser.add_argument("--min_frames", type=int, default=30, help="Minimum frames for a valid clip")
    parser.add_argument("--max_frames", type=int, default=0, help="Max frames filter. 0 means no max.")
    parser.add_argument("--min_fps", type=float, default=0.0, help="Minimum FPS filter. 0 means no min.")
    parser.add_argument(
        "--exclude_keywords",
        type=str,
        default=",".join(DEFAULT_EXCLUDE_KEYWORDS),
        help="Comma-separated keywords excluded from source path and file stem.",
    )
    parser.add_argument(
        "--allow_stagei",
        action="store_true",
        help="Include files that contain `_stagei` in filename.",
    )
    parser.add_argument(
        "--require_exists",
        action="store_true",
        help="Keep only rows where source_path exists on disk.",
    )
    parser.add_argument("--shuffle", action="store_true", help="Shuffle each category before selection.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used when --shuffle is set.")
    parser.add_argument("--output_walk_txt", type=str, default="", help="Optional plain list for walk clips.")
    parser.add_argument("--output_jump_txt", type=str, default="", help="Optional plain list for jump clips.")
    parser.add_argument("--output_mixed_txt", type=str, default="", help="Optional plain list for all clips.")
    return parser.parse_args()


def parse_int(text: str, default: int = 0) -> int:
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return default


def parse_float(text: str, default: float = 0.0) -> float:
    try:
        return float(text)
    except (TypeError, ValueError):
        return default


def normalize_motion_id(raw_motion_id: str, source_path: str, fallback_index: int) -> str:
    motion_id = (raw_motion_id or "").strip()
    if not motion_id:
        motion_id = Path(source_path).stem or f"MOTION_{fallback_index:06d}"
    motion_id = re.sub(r"[^a-zA-Z0-9_.-]+", "_", motion_id)
    return motion_id


def guess_category(path_text: str) -> str:
    text = path_text.lower()
    if any(keyword in text for keyword in ("jump", "hop", "jacks", "leap")):
        return "jump"
    if any(keyword in text for keyword in ("walk", "locomotion", "step", "turn", "sidestep")):
        return "walk"
    if any(keyword in text for keyword in ("crawl", "roll", "flip", "cartwheel", "handstand", "_lie", "lie_")):
        return "exclude"
    return "other"


def normalize_category(row: Dict[str, str]) -> str:
    for key in ("category", "category_guess", "manual_category"):
        value = row.get(key, "").strip().lower()
        if value in {"walk", "jump", "other", "exclude"}:
            return value
    path_text = f"{row.get('source_path', '')} {row.get('file_stem', '')}"
    return guess_category(path_text)


def normalize_priority(row: Dict[str, str]) -> str:
    for key in ("priority", "priority_guess"):
        value = row.get(key, "").strip().lower()
        if value in PRIORITY_RANK:
            return value
    return "medium"


def should_exclude_by_keyword(text: str, keywords: Iterable[str]) -> bool:
    lower_text = text.lower()
    return any(keyword and keyword in lower_text for keyword in keywords)


def write_plain_list(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(f"{row['motion_id']},{row['source_path']}\n")


def main() -> int:
    args = parse_args()
    scan_csv = Path(args.scan_csv).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()
    exclude_keywords = [keyword.strip().lower() for keyword in args.exclude_keywords.split(",") if keyword.strip()]

    if not scan_csv.exists():
        raise FileNotFoundError(f"Scan CSV not found: {scan_csv}")

    with scan_csv.open("r", encoding="utf-8", newline="") as infile:
        input_rows = list(csv.DictReader(infile))

    selected_by_category: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    seen_source_paths = set()
    filtered_out = 0

    for index, row in enumerate(input_rows, start=1):
        source_path = row.get("source_path", "").strip()
        if not source_path:
            filtered_out += 1
            continue

        if source_path in seen_source_paths:
            filtered_out += 1
            continue

        status = row.get("status", "ok").strip().lower()
        if status and status != "ok":
            filtered_out += 1
            continue

        is_stagei = row.get("is_stagei", "0").strip() in {"1", "true", "True"}
        if is_stagei and not args.allow_stagei:
            filtered_out += 1
            continue

        if args.require_exists and not Path(source_path).exists():
            filtered_out += 1
            continue

        full_text = f"{source_path} {row.get('file_stem', '')}"
        if should_exclude_by_keyword(full_text, exclude_keywords):
            filtered_out += 1
            continue

        category = normalize_category(row)
        if category == "exclude":
            filtered_out += 1
            continue

        num_frames = parse_int(row.get("num_frames", ""), default=0)
        if num_frames < args.min_frames:
            filtered_out += 1
            continue
        if args.max_frames > 0 and num_frames > args.max_frames:
            filtered_out += 1
            continue

        fps = parse_float(row.get("fps", ""), default=0.0)
        if args.min_fps > 0 and fps < args.min_fps:
            filtered_out += 1
            continue

        motion_id = normalize_motion_id(row.get("motion_id", ""), source_path, index)
        priority = normalize_priority(row)
        notes = row.get("notes", "").strip()

        normalized = {
            "motion_id": motion_id,
            "source_path": source_path,
            "category": category,
            "priority": priority,
            "notes": notes,
            "num_frames": str(num_frames),
            "fps": f"{fps:.6f}" if fps > 0 else "",
            "duration_sec": row.get("duration_sec", "").strip(),
            "source_dataset": "AMASS_MPI_HDM05",
        }
        selected_by_category[category].append(normalized)
        seen_source_paths.add(source_path)

    # Deterministic ordering unless --shuffle is requested.
    rng = random.Random(args.seed)
    for category, rows in selected_by_category.items():
        if args.shuffle:
            rng.shuffle(rows)
        else:
            rows.sort(
                key=lambda row: (
                    PRIORITY_RANK.get(row["priority"], 99),
                    -parse_int(row.get("num_frames", ""), default=0),
                    row["source_path"],
                )
            )
        selected_by_category[category] = rows

    targets = {"walk": args.walk_count, "jump": args.jump_count, "other": args.other_count}
    selected_rows: List[Dict[str, str]] = []
    for category in ("walk", "jump", "other"):
        selected_rows.extend(selected_by_category.get(category, [])[: targets[category]])

    # Keep a stable grouped order: walk -> jump -> other.
    selected_rows.sort(key=lambda row: ("walk", "jump", "other").index(row["category"]))

    # Ensure unique motion IDs in output.
    used_motion_ids = set()
    for row in selected_rows:
        base_id = row["motion_id"]
        motion_id = base_id
        suffix = 1
        while motion_id in used_motion_ids:
            suffix += 1
            motion_id = f"{base_id}_{suffix}"
        row["motion_id"] = motion_id
        used_motion_ids.add(motion_id)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "motion_id",
        "source_path",
        "category",
        "priority",
        "notes",
        "num_frames",
        "fps",
        "duration_sec",
        "source_dataset",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in selected_rows:
            writer.writerow(row)

    print(f"[manifest] scan_rows={len(input_rows)} filtered_out={filtered_out} selected={len(selected_rows)}")
    print(
        "[manifest] selected_counts="
        + ", ".join(
            f"{category}:{sum(1 for row in selected_rows if row['category'] == category)}"
            for category in ("walk", "jump", "other")
        )
    )
    print(f"[manifest] output={output_csv}")

    if args.output_walk_txt:
        walk_rows = [row for row in selected_rows if row["category"] == "walk"]
        write_plain_list(Path(args.output_walk_txt).expanduser().resolve(), walk_rows)
    if args.output_jump_txt:
        jump_rows = [row for row in selected_rows if row["category"] == "jump"]
        write_plain_list(Path(args.output_jump_txt).expanduser().resolve(), jump_rows)
    if args.output_mixed_txt:
        write_plain_list(Path(args.output_mixed_txt).expanduser().resolve(), selected_rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
