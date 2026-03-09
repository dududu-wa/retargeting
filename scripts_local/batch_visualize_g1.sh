#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <motion_dir> <video_dir> <gmr_root> [robot=unitree_g1_29dof_lock_waist_fixed_hand_feet_inertia] [python_bin=python] [max_files=0]"
  echo "Example:"
  echo "  $0 data/gmr_output_g1/batch videos /path/to/GMR unitree_g1_29dof_lock_waist_fixed_hand_feet_inertia python 20"
  exit 1
fi

MOTION_DIR="$1"
VIDEO_DIR="$2"
GMR_ROOT="$3"
ROBOT="${4:-unitree_g1_29dof_lock_waist_fixed_hand_feet_inertia}"
PYTHON_BIN="${5:-python}"
MAX_FILES="${6:-0}"

if [[ ! -d "$MOTION_DIR" ]]; then
  echo "Motion directory not found: $MOTION_DIR"
  exit 1
fi

mkdir -p "$VIDEO_DIR"

mapfile -t MOTION_FILES < <(find "$MOTION_DIR" -type f -name '*.pkl' | sort)
if [[ ${#MOTION_FILES[@]} -eq 0 ]]; then
  echo "No .pkl files found under $MOTION_DIR"
  exit 1
fi

echo "[viz] total_found=${#MOTION_FILES[@]} robot=$ROBOT"

DONE=0
FAILED=0
TOTAL=${#MOTION_FILES[@]}

for MOTION_FILE in "${MOTION_FILES[@]}"; do
  if [[ "$MAX_FILES" -gt 0 && "$DONE" -ge "$MAX_FILES" ]]; then
    break
  fi

  REL_PATH="${MOTION_FILE#"$MOTION_DIR"/}"
  VIDEO_NAME="${REL_PATH//\//__}"
  VIDEO_NAME="${VIDEO_NAME//\\/__}"
  VIDEO_NAME="${VIDEO_NAME%.pkl}.mp4"
  VIDEO_PATH="$VIDEO_DIR/$VIDEO_NAME"

  CUR_INDEX=$((DONE + FAILED + 1))
  echo "[viz] [$CUR_INDEX/$TOTAL] $MOTION_FILE -> $VIDEO_PATH"

  if "$PYTHON_BIN" "$GMR_ROOT/scripts/vis_robot_motion.py" \
      --robot "$ROBOT" \
      --robot_motion_path "$MOTION_FILE" \
      --record_video \
      --video_path "$VIDEO_PATH" \
      --num_loops 1; then
    DONE=$((DONE + 1))
  else
    FAILED=$((FAILED + 1))
  fi
done

echo "[viz] finished done=$DONE failed=$FAILED video_dir=$VIDEO_DIR"
