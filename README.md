# HDC VR Pilot

## 项目概述

本项目面向 VR 飞行任务中的多模态生理与行为数据，构建四分类工作负荷代理识别流程。当前研究目标是基于 PhysioNet VR Piloting 数据，将任务难度等级 1、2、3、4 映射为四个分类标签，并为后续传统机器学习基线与 Hyperdimensional Computing, HDC, 分类模型准备可复现实验数据。

当前仓库已完成数据盘点、原始模态审计、多模态特征提取，以及四分类建模数据集构建。模型训练、HDC 分类器筛选、多模态融合和在线模拟仍属于后续阶段。

## 研究任务

- 任务类型：四分类 workload proxy classification
- 标签来源：VR 飞行任务难度等级，而不是问卷或直接心理压力标注
- 标签映射：`difficulty_level` 1, 2, 3, 4 分别映射为 `target` 0, 1, 2, 3
- 主要方法方向：Hyperdimensional Computing, HDC
- 对比方法：传统机器学习基线、单模态贡献分析、多模态融合分析

## 数据说明

原始数据来自 PhysioNet VR Piloting 数据集。本项目遵守数据安全规则：

- `vrdataset` 作为只读数据源使用
- 所有生成的实验脚本、结果、图表、日志和报告均保存在 `experiments`
- GitHub 仓库中未上传完整原始数据目录 `vrdataset/dataPackage/`
- 超大文件也未纳入 Git 跟踪，例如 `DataQualityReport.pdf` 和 `feature_extraction_long_table.csv`

如需完整复现实验，请先按照数据集许可要求自行获取原始数据，并将数据放置到本地 `vrdataset/dataPackage/`。

## 项目结构

```text
.
├── EXPERIMENT_STATUS.md
├── requirements.txt
├── experiments/
│   ├── phase_00_project_setup/
│   ├── phase_01_raw_data_modality_audit/
│   ├── phase_02_full_multimodal_feature_extraction/
│   ├── phase_03_multimodal_dataset_labeling/
│   ├── phase_04_ml_baseline_four_class/
│   ├── phase_05_basic_hdc_four_class/
│   ├── phase_06_hdc_classifier_screening/
│   ├── phase_07_single_modality_contribution/
│   ├── phase_08_multimodal_fusion/
│   ├── phase_09_robustness_missing_modality/
│   ├── phase_10_onlinehd_lsl_simulation/
│   └── shared/
└── vrdataset/
    ├── referenceDocuments/
    └── starterCode/
```

## 实验阶段

| 阶段 | 目录 | 状态 | 内容 |
|---:|---|---|---|
| 00 | `phase_00_project_setup` | 已完成 | 项目结构、文件盘点、实验计划 |
| 01 | `phase_01_raw_data_modality_audit` | 已完成 | 原始数据扫描、模态覆盖审计、运行级清单 |
| 02 | `phase_02_full_multimodal_feature_extraction` | 已完成 | 多模态运行级聚合特征提取 |
| 03 | `phase_03_multimodal_dataset_labeling` | 已完成 | 四分类标签构建、泄漏风险分离、建模数据集输出 |
| 04 | `phase_04_ml_baseline_four_class` | 未开始 | 传统机器学习四分类基线 |
| 05 | `phase_05_basic_hdc_four_class` | 未开始 | 基础 HDC 四分类模型 |
| 06 | `phase_06_hdc_classifier_screening` | 未开始 | HDC 分类器筛选 |
| 07 | `phase_07_single_modality_contribution` | 未开始 | 单模态贡献分析 |
| 08 | `phase_08_multimodal_fusion` | 未开始 | 多模态融合策略比较 |
| 09 | `phase_09_robustness_missing_modality` | 未开始 | 缺失模态鲁棒性分析 |
| 10 | `phase_10_onlinehd_lsl_simulation` | 未开始 | OnlineHD 与 LSL 模拟 |

## 当前数据集状态

Phase 03 已生成两个四分类数据集版本：

- 主数据集：`experiments/phase_03_multimodal_dataset_labeling/results/cleaned_multimodal_four_class_without_performance.csv`
- 辅助数据集：`experiments/phase_03_multimodal_dataset_labeling/results/cleaned_multimodal_four_class_with_performance.csv`

主数据集用于后续 Phase 04 和 Phase 05，已排除 performance metrics，以降低模型学习任务表现捷径的风险。辅助数据集保留 performance metrics，可用于上界对照和泄漏风险分析。

当前 Phase 03 摘要：

- 样本数：419
- 类别数：4
- 主数据集特征数：1,176
- 含 performance metrics 的辅助数据集特征数：1,235
- 被试数：35
- 建议验证方式：subject-wise GroupKFold，建议 5 折
- 类别分布：104、106、104、105，整体较均衡

## 环境安装

建议使用 Python 3.10 或更新版本。

```bash
pip install -r requirements.txt
```

主要依赖包括：

- pandas
- numpy
- scipy
- scikit-learn
- matplotlib
- pyyaml
- jupyter
- nbformat
- nbclient
- ipykernel
- pyxdf

## 使用方式

1. 准备数据

   将完整原始数据放入本地：

   ```text
   vrdataset/dataPackage/
   ```

2. 查看项目状态

   ```text
   EXPERIMENT_STATUS.md
   ```

3. 按阶段查看和运行实验

   每个阶段目录均包含对应的 README、notebook、脚本、结果、表格、图和日志。

4. 后续建模建议

   从 Phase 04 开始，应使用主数据集：

   ```text
   experiments/phase_03_multimodal_dataset_labeling/results/cleaned_multimodal_four_class_without_performance.csv
   ```

   所有缺失值填补、标准化、特征选择和模型训练都应放在交叉验证折内完成，避免数据泄漏。

## 关键设计原则

- 原始数据只读，不在 `vrdataset` 中写入实验产物
- 建模主线不使用 performance metrics，避免捷径学习
- 缺失值保留到建模阶段处理
- 使用 subject-wise 划分评估泛化能力
- 阶段产物保存在对应 phase 目录中，便于审计和复现

## 后续工作

- Phase 04：训练传统机器学习四分类基线
- Phase 05：实现基础 HDC 四分类模型
- Phase 06：筛选不同 HDC 分类器与编码策略
- Phase 07：分析单模态贡献
- Phase 08：比较多模态融合策略
- Phase 09：测试缺失模态鲁棒性
- Phase 10：探索 OnlineHD 与 LSL 场景模拟

## 许可与数据限制

代码和实验产物可用于研究复现与方法开发。原始数据受 PhysioNet Restricted Health Data License 约束，使用者应自行确认数据访问权限、许可要求和共享限制。
