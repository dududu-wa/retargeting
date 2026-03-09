#!/usr/bin/env python3
"""Challenge-agent checks for manifest quality and risky assumptions."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


RISK_KEYWORDS = ("crawl", "roll", "flip", "cartwheel", "handstand", "_lie", "lie_", "acrobatic")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run challenge checks on a manifest CSV.")
    parser.add_argument("--manifest_csv", type=str, required=True, help="Manifest CSV to challenge")
    parser.add_argument("--output_csv", type=str, required=True, help="Output challenge report CSV")
    parser.add_argument("--require_exists", action="store_true", help="Treat missing source files as blockers")
    parser.add_argument("--max_duration", type=float, default=20.0, help="Flag clips longer than this duration")
    parser.add_argument("--min_fps", type=float, default=20.0, help="Flag clips with fps lower than this")
    return parser.parse_args()


def parse_float(text: str, default: float = 0.0) -> float:
    try:
        return float(text)
    except (TypeError, ValueError):
        return default


def issue(row: Dict[str, str], level: str, reason: str, action: str) -> Dict[str, str]:
    return {
        "motion_id": (row.get("motion_id") or "").strip(),
        "source_path": (row.get("source_path") or "").strip(),
        "category": (row.get("category") or "").strip(),
        "challenge_level": level,
        "challenge_reason": reason,
        "recommended_action": action,
    }


def main() -> int:
    args = parse_args()
    manifest_csv = Path(args.manifest_csv).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()

    if not manifest_csv.exists():
        raise FileNotFoundError(f"Manifest CSV does not exist: {manifest_csv}")

    with manifest_csv.open("r", encoding="utf-8", newline="") as infile:
        rows = list(csv.DictReader(infile))

    challenges: List[Dict[str, str]] = []
    for row in rows:
        motion_id = (row.get("motion_id") or "").strip()
        source_path = (row.get("source_path") or "").strip()
        category = (row.get("category") or "").strip().lower()
        path_text = f"{motion_id} {source_path}".lower()

        if args.require_exists and source_path and not Path(source_path).exists():
            challenges.append(
                issue(
                    row,
                    "BLOCKER",
                    "source_path does not exist on disk",
                    "Fix path or remove this row from manifest before conversion.",
                )
            )

        if any(keyword in path_text for keyword in RISK_KEYWORDS):
            challenges.append(
                issue(
                    row,
                    "HIGH",
                    "motion name/path contains high-risk keyword",
                    "Move this clip to a later batch or mark as exclude.",
                )
            )

        guessed_jump = any(keyword in path_text for keyword in ("jump", "hop", "jacks", "leap"))
        guessed_walk = any(keyword in path_text for keyword in ("walk", "locomotion", "step", "turn"))
        if category == "walk" and guessed_jump:
            challenges.append(
                issue(
                    row,
                    "MEDIUM",
                    "category walk conflicts with jump-like filename",
                    "Review category assignment manually.",
                )
            )
        if category == "jump" and guessed_walk:
            challenges.append(
                issue(
                    row,
                    "MEDIUM",
                    "category jump conflicts with walk-like filename",
                    "Review category assignment manually.",
                )
            )

        duration_sec = parse_float(row.get("duration_sec", ""), default=0.0)
        if duration_sec > args.max_duration > 0:
            challenges.append(
                issue(
                    row,
                    "MEDIUM",
                    f"clip duration too long ({duration_sec:.2f}s)",
                    "Trim the clip before conversion.",
                )
            )

        fps = parse_float(row.get("fps", ""), default=0.0)
        if 0 < fps < args.min_fps:
            challenges.append(
                issue(
                    row,
                    "LOW",
                    f"low fps ({fps:.2f}) may hurt tracking quality",
                    "Keep but prioritize manual quality checks.",
                )
            )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "motion_id",
        "source_path",
        "category",
        "challenge_level",
        "challenge_reason",
        "recommended_action",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fields)
        writer.writeheader()
        for row in challenges:
            writer.writerow(row)

    print(f"[challenge] rows_in_manifest={len(rows)}")
    print(f"[challenge] challenges_found={len(challenges)}")
    print(f"[challenge] output={output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
