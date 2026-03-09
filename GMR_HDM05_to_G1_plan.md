# GMR 转换 HDM05（到 Unitree G1）的实施计划

## 1. 目标

基于 **GMR (General Motion Retargeting)**，将 **HDM05** 中适合 humanoid locomotion / whole-body motion 的动作样本转换为 **Unitree G1** 可用的机器人参考运动（robot motion `.pkl`），并完成可视化、质量筛选、批处理与后续训练接口准备。

本计划优先采用：

> **AMASS 中的 MPI-HDM05 子集 → GMR 的 SMPL-X to Robot pipeline → Unitree G1**

而不是直接从 HDM05 原始 `ASF/AMC` 或 `C3D` 走自定义转换。

---

## 2. 为什么选这条路线

### 2.1 推荐主路线：AMASS(MPI-HDM05) → GMR
原因：
- GMR 原生支持 **SMPL-X → robot** 的转换脚本，命令路径清晰。
- GMR 已明确支持 **Unitree G1 (`unitree_g1`)**。
- **AMASS** 已将多个传统 mocap 数据集统一到 SMPL 表达中，其中包含 **MPI-HDM05**。
- 相比直接解析 HDM05 的 `ASF/AMC` 或 `C3D`，这条路线实现风险更低、工程工作量更小、结果更容易对齐到 GMR 现有输入格式。

### 2.2 不推荐第一版直接做：HDM05 原始格式 → 自己写转换器
原因：
- HDM05 官方主要提供 `C3D` 与 `ASF/AMC`。
- GMR 当前公开 README 中并未把 HDM05 的 `ASF/AMC` / `C3D` 列为直接支持输入。
- 若走原始格式，通常还需要额外做：
  - skeleton 解析
  - joint naming 对齐
  - 坐标系对齐
  - root 轨迹重建
  - 可能的 BVH/SMPL 中间转换
- 第一版容易把时间花在“格式考古”而不是 retargeting 本身。

---

## 3. 总体技术路线

```text
HDM05 / AMASS(MPI-HDM05)
        ↓
筛选 locomotion / jump / useful whole-body motions
        ↓
整理为 GMR 可读的 SMPL-X 输入
        ↓
调用 GMR: scripts/smplx_to_robot.py
        ↓
输出 Unitree G1 robot motion (.pkl)
        ↓
调用 GMR: scripts/vis_robot_motion.py 可视化检查
        ↓
批量处理 scripts/smplx_to_robot_dataset.py
        ↓
构建可用于 tracking / imitation / policy 的 motion dataset
```

---

## 4. 项目目录规划

建议建立如下目录：

```text
project_gmr_hdm05/
├── README.md
├── env/
│   └── environment_notes.md
├── data/
│   ├── amass_raw/
│   │   └── MPI_HDM05/
│   ├── smplx_body_models/
│   ├── selected_hdm05_motion_list/
│   │   ├── walk.txt
│   │   ├── jump.txt
│   │   └── mixed_useful.txt
│   └── gmr_output_g1/
│       ├── single/
│       └── batch/
├── scripts_local/
│   ├── scan_hdm05_amass.py
│   ├── make_motion_manifest.py
│   ├── batch_convert_hdm05_to_g1.sh
│   ├── batch_visualize_g1.sh
│   └── quality_check_report.py
├── logs/
├── videos/
└── notes/
    ├── motion_selection_rules.md
    ├── failure_cases.md
    └── tuning_notes.md
```

---

## 5. Phase 0：环境准备

### 5.1 克隆 GMR

```bash
git clone https://github.com/YanjieZe/GMR.git
cd GMR
```

### 5.2 创建环境

```bash
conda create -n gmr python=3.10 -y
conda activate gmr
pip install -e .
conda install -c conda-forge libstdcxx-ng -y
```

### 5.3 下载 SMPL-X body model
按 GMR 要求放到：

```text
assets/body_models/smplx/
  ├── SMPLX_NEUTRAL.pkl
  ├── SMPLX_FEMALE.pkl
  └── SMPLX_MALE.pkl
```

### 5.4 注意事项
如果使用 SMPL-X pkl 文件，按 GMR README 说明，可能需要调整 `smplx/body_models.py` 里的 `ext` 设置。

---

## 6. Phase 1：数据来源策略

### 主方案：使用 AMASS 中的 MPI-HDM05

#### 目标
不直接处理 HDM05 原始 `ASF/AMC` 或 `C3D`，而是优先使用 **AMASS 已统一后的 MPI-HDM05 数据**。

#### 原因
- GMR 的 `smplx_to_robot.py` 是现成主路。
- 避免额外写原始 mocap 格式解析器。
- 对后续批量转换、统一质量控制、训练数据整理更友好。

#### 需要确认的事项
- 你是否有 **AMASS 访问权限**。
- 本地下载的 MPI-HDM05 子集路径与文件组织形式。
- 单个 motion 文件的命名规则、帧率、是否有 gender/shape 参数。

### 备选方案：原始 HDM05 → 自定义中间表示 → GMR

仅当以下条件成立时再走：
- 拿不到 AMASS 版本；或
- 你必须保留 HDM05 原始骨架/标注。

备选中间层可考虑：
1. `ASF/AMC -> BVH -> GMR BVH pipeline`
2. `ASF/AMC or C3D -> SMPL/SMPL-X -> GMR SMPL-X pipeline`

这两条都比 AMASS 方案更费劲。

---

## 7. Phase 2：动作筛选计划

### 7.1 筛选目标
先不要全量上 2000+ clips。第一版只挑：
- **walk**
- **jump / jumping-jacks / dynamic jumps**
- **转身+行走**
- **少量 whole-body upper-lower coordinated motions**

### 7.2 第一批建议规模
- walk: 20–40 条
- jump: 10–20 条
- 其他过渡动作: 10–20 条

总计：
> **40–80 条** 作为 first batch

这很重要。别一上来全量批跑，把自己变成日志管理员。

### 7.3 筛选标准
保留：
- 动作语义清晰
- root trajectory 合理
- 没有长时间躺地/翻滚/极端手部接触
- 对 G1 的腿部与躯干自由度相对友好

暂时排除：
- 大幅地面翻滚
- 四肢强接触地面动作
- 柔术/匍匐/极端柔韧动作
- 大量手指依赖动作
- 人能做但 G1 大概率做不出的夸张高抬腿/扭腰动作

### 7.4 输出清单文件
生成文本或 CSV：

```text
motion_id, source_path, category, priority, notes
```

例如：

```text
MPI_HDM05_001, /data/amass_raw/MPI_HDM05/xxx.npz, walk, high, clean forward walk
MPI_HDM05_002, /data/amass_raw/MPI_HDM05/yyy.npz, jump, high, landing needs check
```

---

## 8. Phase 3：单条样本打通

### 8.1 先做单条样本
目标：先用 1 条 walk 和 1 条 jump 打通全流程。

### 8.2 单条转换命令模板

```bash
python scripts/smplx_to_robot.py \
  --smplx_file <path_to_one_amass_mpi_hdm05_motion> \
  --robot unitree_g1 \
  --save_path <output_path.pkl> \
  --rate_limit
```

### 8.3 单条可视化

```bash
python scripts/vis_robot_motion.py \
  --robot unitree_g1 \
  --robot_motion_path <output_path.pkl>
```

### 8.4 单条样本验收标准
对于 walk：
- base/root 朝向正常
- 双脚没有明显穿地
- 没有持续性 foot skating
- 手臂摆动不至于爆炸
- 姿态连续，没有帧间明显抖动

对于 jump：
- 下蹲、起跳、腾空、落地阶段可辨认
- 落地后没有立刻姿态发散
- 躯干没有过度后仰/前折

---

## 9. Phase 4：批处理脚本

### 9.1 批处理命令

```bash
python scripts/smplx_to_robot_dataset.py \
  --src_folder <path_to_mpi_hdm05_folder> \
  --tgt_folder <path_to_output_folder> \
  --robot unitree_g1
```

### 9.2 不建议直接全量盲跑
建议分三批：

#### Batch A（最干净）
- 20 条 walk / jump
- 用于验证 pipeline 稳定性

#### Batch B（中等复杂度）
- 30–50 条 locomotion + transition
- 用于观察失败模式

#### Batch C（扩展）
- 其余可用动作
- 只在 A/B 质量足够后再做

### 9.3 建议写本地包装脚本
文件：`scripts_local/batch_convert_hdm05_to_g1.sh`

功能：
- 逐条遍历 motion list
- 失败时记录日志
- 自动保存 stdout/stderr
- 统计成功数/失败数

---

## 10. Phase 5：质量检查与失败模式记录

### 10.1 人工检查维度
对每条输出至少检查：
- root translation 是否离谱
- 脚是否打滑
- 双足是否穿地
- 膝盖方向是否异常
- 手臂是否自碰撞严重
- jump 的落地是否有大抖动

### 10.2 失败类型标签
建议建立：

```text
OK
FOOT_SLIDE
GROUND_PENETRATION
ARM_EXPLOSION
TORSO_TILT_TOO_LARGE
ROOT_DRIFT
JUMP_LANDING_BAD
UNUSABLE
```

### 10.3 质量报告文件
输出：

```text
file_name, category, result, issue_type, comments
```

这一步别偷懒。否则后面训练时会把脏 reference 当圣经，结果策略学成抽风宗师。

---

## 11. Phase 6：调参计划

### 11.1 优先调的不是“更好看”，而是“更稳”
GMR 已有较成熟默认配置，但对 HDM05 某些动作仍可能需要额外调参。

优先检查：
- robot IK config
- joint limit behavior
- root/base scaling
- velocity limit
- lower-body vs upper-body 权重

### 11.2 调参顺序建议
1. 先让 walk 稳定
2. 再处理 jump
3. 最后再看复杂 upper-body 动作

### 11.3 每次只改一个变量
建议日志格式：

```text
Date:
Motion:
Robot:
Changed parameter:
Old value:
New value:
Observation:
Result:
```

不然过几天你会忘记自己到底动了哪颗螺丝，系统就会变成哲学问题。

---

## 12. Phase 7：输出格式与下游接口

### 12.1 预期输出
GMR 的 robot motion 输出可理解为：
- robot base translation
- robot base rotation
- robot joint positions

保存为 `.pkl`。

### 12.2 下游用途
这些输出可用于：
- MuJoCo / visualization
- RL tracking reference
- imitation learning dataset
- motion prior preprocessing
- whole-body controller 参考轨迹

### 12.3 建议额外导出 manifest
建议为每个 `.pkl` 配一行元数据：

```text
motion_name, class_label, source_dataset, source_path, robot, num_frames, fps, quality_flag
```

---

## 13. 关键风险与对策

### 风险 1：AMASS 中的 MPI-HDM05 文件组织与预期不一致
**对策：**
先写 `scan_hdm05_amass.py`，打印字段、shape、帧数、fps、gender、可用键名。

### 风险 2：某些 HDM05 动作对 G1 不友好
**对策：**
先专注 locomotion / jump / transition，避免极端 acrobatic clips。

### 风险 3：jump 落地后不稳
**对策：**
- 先保留但打低质量标签
- 后续用于 reference 清洗
- 训练时可只取起跳前后稳定段或做 clip trimming

### 风险 4：批处理成功率低
**对策：**
- 分批跑
- 逐步扩大动作范围
- 先定位失败模式再扩充数据

### 风险 5：想直接从原始 HDM05 转，结果工作量爆炸
**对策：**
坚持主路线：
> **AMASS(MPI-HDM05) → GMR SMPL-X → G1**

---

## 14. 第一周最小可交付成果（MVP）

### Day 1–2
- 安装 GMR
- 配好 SMPL-X body model
- 拿到 AMASS 中 MPI-HDM05 数据
- 写数据扫描脚本

### Day 3
- 选 1 条 walk + 1 条 jump
- 单条转换到 G1
- 成功可视化

### Day 4
- 选 10 条 walk / jump
- 小批量转换
- 建立质量检查表

### Day 5–7
- 统计成功率
- 记录失败模式
- 输出 first batch 可用 `.pkl` 数据集
- 写简短实验笔记

---

## 15. 本地需要补的辅助脚本

### 15.1 `scan_hdm05_amass.py`
用途：
- 扫描 MPI-HDM05 子集目录
- 列出所有 motion 文件
- 提取帧数、时长、fps、可用字段

### 15.2 `make_motion_manifest.py`
用途：
- 从扫描结果里生成可人工筛选的 manifest CSV

### 15.3 `batch_convert_hdm05_to_g1.sh`
用途：
- 从 manifest 中逐条调用 GMR 转换

### 15.4 `quality_check_report.py`
用途：
- 汇总输出目录
- 生成需要人工复查的视频/文件列表

---

## 16. 推荐执行顺序（极简版）

```text
1. 搭环境
2. 拿到 AMASS 的 MPI-HDM05
3. 扫描文件并筛选 walk/jump
4. 单条打通到 G1
5. 10 条小批量
6. 质量检查
7. 40–80 条 first batch
8. 准备下游 tracking / imitation
```

---

## 17. 最终建议

对当前任务，最合理的工程决策是：

> **不要先做“原始 HDM05 格式解析工程”；先用 AMASS 中的 MPI-HDM05，走 GMR 已支持的 SMPL-X 管道，快速打通 HDM05 → G1。**

这条路线最符合“成熟、可用、能尽快产出 reference motion”的目标。

等你把第一批数据跑通、质量看顺眼了，再决定要不要回头攻原始 `ASF/AMC` / `C3D`。那时候你是在做增强，不是在和文件格式摔跤。

