# README: Convert MPI-HDM05 (AMASS) Motions to Unitree G1 with GMR

## 1. Overview

This workflow converts locomotion clips from **MPI-HDM05 (inside AMASS)** into **Unitree G1** robot motion references (`.pkl`) with GMR.

```text
MPI-HDM05 (AMASS)
  -> SMPL-X motion
  -> GMR retargeting
  -> Unitree G1 motion (.pkl)
```

Use this route because:

- Original HDM05 is distributed as legacy mocap formats (`C3D`, `ASF/AMC`).
- AMASS provides a modern and unified SMPL-X representation.
- GMR already supports SMPL-X to Unitree G1 retargeting.

---

## 2. Mandatory Robot Model

For this project, the G1 model must be:

- `g1_description/g1_29dof_lock_waist_with_fixed_hand_rev_1_0_feet_inertia.urdf`

This repository is configured so `--robot unitree_g1` uses that URDF.

Implementation location:

- `general_motion_retargeting/params.py`
- `ROBOT_XML_DICT["unitree_g1"]`

Current mapping:

```python
"unitree_g1": G1_DESCRIPTION_ROOT / "g1_29dof_lock_waist_with_fixed_hand_rev_1_0_feet_inertia.urdf"
```

---

## 3. Goal

Build a repeatable pipeline for:

- selecting walk/run/jump clips from MPI-HDM05 (AMASS),
- converting them into G1-compatible robot references,
- validating quality,
- preparing data for downstream WBC, imitation learning, and RL training.

---

## 4. Dataset Links

AMASS:

- https://amass.is.tue.mpg.de/

Original HDM05 reference:

- https://resources.mpi-inf.mpg.de/HDM05/
- https://resources.mpi-inf.mpg.de/HDM05/07_MuRoClEbKrWe_HDM05.pdf

Use AMASS for conversion. Use original HDM05 docs mainly for motion category lookup.

---

## 5. Recommended Layout

```text
project_root/
|-- README.md
|-- data/
|   |-- amass/
|   |   `-- MPI_HDM05/
|   |-- selected_sequences/
|   `-- g1_output/
|-- scripts/
|   |-- scan_hdm05_amass.py
|   |-- select_motion_subset.py
|   |-- convert_single_motion.sh
|   |-- batch_convert_hdm05_to_g1.sh
|   `-- validate_g1_motion.py
`-- external/
    `-- GMR/
```

---

## 6. Environment Setup

### 6.1 Clone GMR

```bash
git clone https://github.com/YanjieZe/GMR.git
cd GMR
```

### 6.2 Install dependencies

Follow the official repository environment instructions.

At minimum verify:

- Python and package versions match the repo requirement.
- `scripts/smplx_to_robot.py` can run.
- `--robot unitree_g1` resolves to the required G1 URDF above.

---

## 7. Data Preparation

### 7.1 Download AMASS + MPI-HDM05 subset

Place data under:

```text
data/amass/MPI_HDM05/
```

### 7.2 Select pilot clips first

Start with only:

- 1 walking clip
- 1 running clip
- 1 jumping clip

Do not start with full-batch conversion.

---

## 8. Conversion Commands

### 8.1 Single sequence (recommended first)

```bash
python scripts/smplx_to_robot.py \
  --smplx_file <path_to_smplx_data> \
  --robot unitree_g1 \
  --save_path <path_to_save_robot_data.pkl> \
  --rate_limit
```

### 8.2 Batch conversion

```bash
python scripts/smplx_to_robot_dataset.py \
  --src_folder <path_to_dir_of_smplx_data> \
  --tgt_folder <path_to_dir_to_save_robot_data> \
  --robot unitree_g1
```

### 8.3 Optional wrappers

```bash
bash scripts/convert_single_motion.sh
```

```bash
bash scripts/batch_convert_hdm05_to_g1.sh
```

---

## 9. Practical Workflow

1. Prepare environment and verify model mapping (`unitree_g1` -> required URDF).
2. Prepare AMASS MPI-HDM05 data.
3. Select walk/run/jump pilot clips.
4. Convert one sequence first.
5. Validate quality.
6. Expand to selected batch.

---

## 10. Validation Checklist

### 10.1 Semantics

- walking still looks like walking
- running still looks like running
- jumping still has clear takeoff/flight/landing phases

### 10.2 Foot behavior

- limited stance-foot sliding
- stable landing

### 10.3 Joint behavior

- no exploding joints
- no impossible folding
- smooth frame-to-frame continuity

### 10.4 Usability

- output is usable as G1 reference motion
- output quality is sufficient for tracking/imitation tasks

---

## 11. Common Pitfalls

1. Starting from original HDM05 instead of AMASS.
2. Running full-batch conversion before pilot validation.
3. Assuming all locomotion clips are equally retargetable.
4. Treating generated `.pkl` as automatically valid without visual checks.

---

## 12. First Milestone

Convert and validate these 3 motions:

- 1 walk
- 1 run
- 1 jump

Only then start larger batch jobs.

---

## 13. Expected Output

```text
data/g1_output/
|-- walk_001.pkl
|-- run_001.pkl
`-- jump_001.pkl
```

Each output should contain:

- `root_pos`
- `root_rot`
- `dof_pos`
- `local_body_pos`
- `link_body_list`
- `fps`
