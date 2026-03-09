# Implementation Agent

## Goal

Build and maintain the local scripts required by `GMR_HDM05_to_G1_plan.md`.

## Inputs

- plan document
- existing GMR scripts
- AMASS folder layout

## Outputs

- `scripts_local/scan_hdm05_amass.py`
- `scripts_local/make_motion_manifest.py`
- `scripts_local/batch_convert_hdm05_to_g1.sh`
- `scripts_local/batch_visualize_g1.sh`
- `scripts_local/quality_check_report.py`
- changelog in README

## Quality Bar

- Scripts provide clear CLI help.
- Failures are logged with path and reason.
- CSV output columns are stable and machine-readable.
