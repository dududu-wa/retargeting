# README: Convert MPI-HDM05 (AMASS) Motions to Unitree G1 with GMR

## 1. Overview

This project uses the following pipeline to convert human motion data into robot motion references for **Unitree G1**:

```text
MPI-HDM05 (inside AMASS)
    -> SMPL-X motion data
    -> GMR retargeting
    -> Unitree G1 motion (.pkl)
```

This route is chosen because:

- **Original HDM05** is distributed in formats such as **C3D** and **ASF/AMC**, which are not the most convenient direct inputs for GMR.
- **AMASS** provides a unified motion representation and explicitly includes **MPI-HDM05**.
- **GMR** supports **SMPL-X to robot** conversion and supports **`unitree_g1`** as a target robot.

So instead of wrestling with legacy mocap formats, the practical route is:

> use **MPI-HDM05 from AMASS**, then convert it with **GMR** to **Unitree G1**.

---

## 2. Goal

The goal is to build a clean and repeatable workflow for:

- selecting **walk / run / jump** motions from **MPI-HDM05 (AMASS version)**,
- converting them into **G1-compatible robot motion references**,
- validating the output,
- and preparing the converted motions for downstream use in:
  - whole-body control (WBC),
  - imitation learning,
  - RL policy training,
  - motion prior experiments.

---

## 3. Why use MPI-HDM05 (AMASS version)

### 3.1 Why not use original HDM05 directly

Original HDM05 is excellent as a motion dataset, but its native formats are older mocap formats, mainly:

- `C3D`
- `ASF/AMC`

These are inconvenient if the target pipeline is already built around **SMPL-X**.

### 3.2 Why AMASS is better here

AMASS unifies many mocap datasets into a common parametric body representation. It includes **MPI-HDM05**, which makes it much easier to integrate with modern humanoid motion pipelines.

### 3.3 Why this is useful for humanoid locomotion

HDM05 contains locomotion-relevant motion categories such as:

- walking
- running
- jumping
- hopping / jumping-related motions

That makes it suitable for G1 locomotion retargeting experiments.

---

## 4. Recommended codebase

### GMR repository

Use the official GMR repository:

- GitHub: https://github.com/YanjieZe/GMR

GMR provides direct scripts for converting **SMPL-X motion** to **robot motion**, including **Unitree G1**.

---

## 5. Dataset links

### AMASS

- Website: https://amass.is.tue.mpg.de/

Use the **MPI-HDM05 subset inside AMASS**.

### Original HDM05 reference

- Website: https://resources.mpi-inf.mpg.de/HDM05/
- Motion documentation PDF: https://resources.mpi-inf.mpg.de/HDM05/07_MuRoClEbKrWe_HDM05.pdf

The original HDM05 site is mainly useful for:

- understanding motion categories,
- looking up sequence names,
- checking whether a sequence is likely to contain walk / run / jump motions.

---

## 6. Project structure

A suggested project layout:

```text
project_root/
├── README.md
├── data/
│   ├── amass/
│   │   └── MPI_HDM05/
│   ├── selected_sequences/
│   └── g1_output/
├── scripts/
│   ├── scan_hdm05_amass.py
│   ├── select_motion_subset.py
│   ├── convert_single_motion.sh
│   ├── batch_convert_hdm05_to_g1.sh
│   └── validate_g1_motion.py
└── external/
    └── GMR/
```

---

## 7. Environment setup

### 7.1 Clone GMR

```bash
git clone https://github.com/YanjieZe/GMR.git
cd GMR
```

### 7.2 Create environment

Follow the environment instructions in the GMR repository.

At minimum, verify that:

- Python version matches the repo requirements
- required dependencies are installed
- the `unitree_g1` robot target is available in the repo configuration

---

## 8. Data preparation

### 8.1 Download AMASS

Download AMASS and obtain the **MPI-HDM05** subset.

Place it under something like:

```text
data/amass/MPI_HDM05/
```

### 8.2 Inspect available sequences

Before conversion, scan the files and identify motion sequences related to:

- walk
- run
- jump

You may use:

- sequence names,
- metadata,
- motion documentation,
- quick visualization if needed.

### 8.3 Build a small pilot subset first

Do **not** start with full-batch conversion.

Start with only:

- 1 walking sequence
- 1 running sequence
- 1 jumping sequence

This saves time and prevents debugging inside a pile of broken outputs.

---

## 9. Conversion pipeline

### 9.1 Single motion conversion

GMR provides a script for single-file SMPL-X to robot conversion:

```bash
python scripts/smplx_to_robot.py \
  --smplx_file <path_to_smplx_data> \
  --robot unitree_g1 \
  --save_path <path_to_save_robot_data.pkl> \
  --rate_limit
```

### 9.2 Batch conversion

For batch conversion:

```bash
python scripts/smplx_to_robot_dataset.py \
  --src_folder <path_to_dir_of_smplx_data> \
  --tgt_folder <path_to_dir_to_save_robot_data> \
  --robot unitree_g1
```

### 9.3 Suggested wrapper script

Create a wrapper script such as:

```bash
bash scripts/convert_single_motion.sh
```

or

```bash
bash scripts/batch_convert_hdm05_to_g1.sh
```

This is useful for keeping your actual experiment paths separate from the raw GMR commands.

---

## 10. Practical workflow

### Step 1 — Prepare GMR

- clone the repository
- install dependencies
- verify `unitree_g1` works in the GMR setup

### Step 2 — Prepare MPI-HDM05 from AMASS

- download AMASS
- locate the MPI-HDM05 subset
- organize the files under a stable directory structure

### Step 3 — Select motion samples

Choose a few pilot motions:

- walk
- run
- jump

### Step 4 — Convert a single motion first

Run `smplx_to_robot.py` on one sequence and save the G1 result.

### Step 5 — Validate the result

Check whether the retargeted motion:

- looks physically reasonable,
- preserves the motion type,
- does not show severe jitter,
- does not show obvious foot sliding,
- does not collapse into an impossible G1 posture.

### Step 6 — Expand to batch conversion

Only after single-sequence validation passes, run the batch conversion on the selected subset.

---

## 11. Validation checklist

After conversion, validate each output on at least the following points.

### 11.1 Motion semantics

- walking still looks like walking
- running still looks like running
- jumping still has visible takeoff / flight / landing structure

### 11.2 Foot behavior

- stance feet do not slide too much
- landing does not look unstable or broken

### 11.3 Joint behavior

- no obvious exploding joints
- no severe self-intersection or impossible limb folding
- motion remains continuous frame to frame

### 11.4 Robot suitability

- output can be used as a reference motion for G1
- motion is acceptable for later tracking by WBC or RL-based imitation

---

## 12. Suggested helper scripts

### `scan_hdm05_amass.py`

Purpose:

- recursively scan the MPI-HDM05 directory,
- list all available motion files,
- optionally extract names and metadata.

### `select_motion_subset.py`

Purpose:

- filter candidate motions for walk / run / jump,
- write a selected file list for later conversion.

### `convert_single_motion.sh`

Purpose:

- wrap one GMR conversion command,
- keep paths and experiment settings reproducible.

### `batch_convert_hdm05_to_g1.sh`

Purpose:

- convert all selected files to G1 `.pkl` outputs.

### `validate_g1_motion.py`

Purpose:

- load converted motions,
- summarize basic statistics,
- optionally run visualization or structural checks.

---

## 13. Common pitfalls

### Pitfall 1 — Starting from original HDM05

This usually adds unnecessary format conversion pain.

### Pitfall 2 — Running full batch first

Do not do this unless you enjoy debugging a warehouse full of broken `.pkl` files.

### Pitfall 3 — Assuming all locomotion clips are equally usable

Some clips may be harder to retarget cleanly because of:

- unusual style,
- strong upper-body motion,
- rapid transitions,
- contact ambiguity.

### Pitfall 4 — Skipping output validation

A file being generated successfully does **not** mean the motion is good.

---

## 14. Recommended first milestone

A realistic first milestone is:

- convert **1 walk** motion from MPI-HDM05 (AMASS) to G1,
- convert **1 run** motion,
- convert **1 jump** motion,
- verify all three visually,
- save outputs in a clean experiment folder.

Only after this should you move on to larger batches.

---

## 15. Expected output

Expected output format:

- robot motion data saved as `.pkl`
- retargeted for **Unitree G1**
- ready for downstream visualization, reference tracking, or policy training

Example output folder:

```text
data/g1_output/
├── walk_001.pkl
├── run_001.pkl
└── jump_001.pkl
```

---

## 16. Summary

This project uses the following practical route:

```text
MPI-HDM05 (AMASS version)
    -> SMPL-X motion
    -> GMR retargeting
    -> Unitree G1 reference motion
```

This is the preferred path because it avoids legacy HDM05 format friction and plugs directly into a modern humanoid retargeting pipeline.

The key advice is simple:

- use **AMASS MPI-HDM05**, not raw HDM05,
- use **GMR** as the conversion engine,
- start with **small walk/run/jump samples**,
- validate carefully before batch conversion.

---

## 17. Next steps

Recommended next additions to this repository:

1. a sequence scanner for MPI-HDM05 (AMASS)
2. a motion selector for walk/run/jump samples
3. a batch conversion shell script
4. an output validator / visualizer
5. optional notes for downstream WBC or RL use

