# HDM05 -> Unitree G1 Retargeting Workspace

本仓库按 `GMR_HDM05_to_G1_plan.md` 落地，目标是基于 AMASS(MPI-HDM05) 通过 GMR 生成 Unitree G1 可用的 `robot motion .pkl`。

## 核心流程

1. 扫描 AMASS 数据：`scripts_local/scan_hdm05_amass.py`
2. 生成可筛选 manifest：`scripts_local/make_motion_manifest.py`
3. 批量转换到 G1：`scripts_local/batch_convert_hdm05_to_g1.sh`
4. 批量可视化：`scripts_local/batch_visualize_g1.sh`
5. 产出质检模板：`scripts_local/quality_check_report.py`
6. 质疑代理审查：`scripts_local/challenge_manifest.py`

## 目录

- `data/amass_raw/MPI_HDM05/`: AMASS 中 MPI-HDM05 源数据
- `data/selected_hdm05_motion_list/`: 扫描结果与 manifest
- `data/gmr_output_g1/single/`: 单条输出
- `data/gmr_output_g1/batch/`: 批量输出
- `logs/`: 批处理日志
- `videos/`: 可视化导出
- `teams/`: 执行分工（含质疑团队）

## 快速开始

### 1) 扫描数据并输出索引

```bash
python scripts_local/scan_hdm05_amass.py \
  --input_dir data/amass_raw/MPI_HDM05 \
  --output_csv data/selected_hdm05_motion_list/amass_scan.csv \
  --output_errors data/selected_hdm05_motion_list/amass_scan_errors.csv
```

### 2) 自动生成 first batch manifest

```bash
python scripts_local/make_motion_manifest.py \
  --scan_csv data/selected_hdm05_motion_list/amass_scan.csv \
  --output_csv data/selected_hdm05_motion_list/manifest_first_batch.csv \
  --walk_count 30 --jump_count 15 --other_count 15
```

### 3) 批量转换（需要 GMR）

```bash
bash scripts_local/batch_convert_hdm05_to_g1.sh \
  data/selected_hdm05_motion_list/manifest_first_batch.csv \
  data/gmr_output_g1/batch \
  /path/to/GMR
```

PowerShell 等价命令：

```powershell
.\scripts_local\batch_convert_hdm05_to_g1.ps1 `
  -ManifestCsv data/selected_hdm05_motion_list/manifest_first_batch.csv `
  -OutputDir data/gmr_output_g1/batch `
  -GmrRoot .
```

### 4) 生成质检模板

```bash
python scripts_local/quality_check_report.py \
  --motion_dir data/gmr_output_g1/batch \
  --manifest data/selected_hdm05_motion_list/manifest_first_batch.csv \
  --output_csv data/selected_hdm05_motion_list/quality_check_template.csv
```

### 5) 运行质疑代理（高风险输入拦截）

```bash
python scripts_local/challenge_manifest.py \
  --manifest_csv data/selected_hdm05_motion_list/manifest_first_batch.csv \
  --output_csv data/selected_hdm05_motion_list/manifest_challenge_report.csv \
  --require_exists
```

### 6) 批量渲染可视化视频（可选）

```bash
bash scripts_local/batch_visualize_g1.sh \
  data/gmr_output_g1/batch \
  videos \
  /path/to/GMR \
  unitree_g1 \
  python \
  20
```

PowerShell 等价命令：

```powershell
.\scripts_local\batch_visualize_g1.ps1 `
  -MotionDir data/gmr_output_g1/batch `
  -VideoDir videos `
  -GmrRoot . `
  -MaxFiles 20
```

## 执行分工

团队分工见 `teams/README.md`，其中 `Challenge Team` 负责质疑假设、识别高风险输入并触发停线复盘。
