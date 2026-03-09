#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <manifest_csv> <output_dir> <gmr_root> [robot=unitree_g1_29dof_lock_waist_fixed_hand_feet_inertia] [python_bin=python] [extra args...]"
  echo "Example:"
  echo "  $0 data/selected_hdm05_motion_list/manifest_first_batch.csv data/gmr_output_g1/batch /path/to/GMR unitree_g1_29dof_lock_waist_fixed_hand_feet_inertia python --overwrite"
  exit 1
fi

MANIFEST_CSV="$1"
OUTPUT_DIR="$2"
GMR_ROOT="$3"
ROBOT="${4:-unitree_g1_29dof_lock_waist_fixed_hand_feet_inertia}"
PYTHON_BIN="${5:-python}"
EXTRA_ARGS=("${@:6}")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$PYTHON_BIN" "$SCRIPT_DIR/batch_convert_hdm05_to_g1.py" \
  --manifest_csv "$MANIFEST_CSV" \
  --output_dir "$OUTPUT_DIR" \
  --gmr_root "$GMR_ROOT" \
  --robot "$ROBOT" \
  --python_bin "$PYTHON_BIN" \
  "${EXTRA_ARGS[@]}"
