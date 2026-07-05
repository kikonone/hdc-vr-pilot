# HDC VR Pilot

## Project Overview
This project focuses on multimodal physiological and behavioral data from VR flight missions, constructing a four-class workload proxy identification process. The current research objective is to map task difficulty levels 1, 2, 3, and 4 to four classification labels based on PhysioNet VR Piloting data, and to prepare reproducible experimental data for subsequent traditional machine learning baselines and Hyperdimensional Computing (HDC) classification models.
The current repository has completed data inventory, raw modality auditing, multimodal feature extraction, and the construction of the four-class modeling dataset. Model training, HDC classifier selection, multimodal fusion, and online simulation are still in the later stages.
## Research Task
- Task Type: Four-class workload proxy classification
- Tag Source: VR flight mission difficulty level, not questionnaires or direct psychological stress labeling
- Tag Mapping: `difficulty_level` 1, 2, 3, 4 are mapped to `target` 0, 1, 2, 3 respectively
- Main Methodology: Hyperdimensional Computing, HDC
- Comparison Methods: Traditional machine learning baseline, single-modal contribution analysis, multimodal fusion analysis
## Data Description
The raw data comes from the PhysioNet VR Piloting dataset. This project adheres to the following data security rules:
- `vrdataset` is used as a read-only data source.
- All generated experimental scripts, results, graphs, logs, and reports are stored in `experiments`.
- The complete raw data directory `vrdataset/dataPackage/` is not uploaded to the GitHub repository.
- Extremely large files, such as `DataQualityReport.pdf` and `feature_extraction_long_table.csv`, are not included in Git tracking.
To fully reproduce the experiments, please obtain the raw data yourself according to the dataset license requirements and place the data locally in `vrdataset/dataPackage/`.

## Project Structure
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

## Experimental stage

| Phase | Table of Contents | Status | Content |

|---:|---|---|---|
| 00 | `phase_00_project_setup` | Completed | Project structure, file inventory, experimental plan |
| 01 | `phase_01_raw_data_modality_audit` | Completed | Raw data scanning, modality coverage audit, run-level inventory |
| 02 | `phase_02_full_multimodal_feature_extraction` | Completed | Multimodal run-level aggregated feature extraction |
| 03 | `phase_03_multimodal_dataset_labeling` | Completed | Four-class label construction, leakage risk separation, modeling dataset output |
| 04 | `phase_04_ml_baseline_four_class` | Not started | Traditional machine learning four-class baseline |
| 05 | `phase_05_basic_hdc_four_class` | Not started | Basic HDC Four-Classification Model |
| 06 | `phase_06_hdc_classifier_screening` | Not Started | HDC Classifier Screening |
| 07 | `phase_07_single_modality_contribution` | Not Started | Single-Modality Contribution Analysis |
| 08 | `phase_08_multimodal_fusion` | Not Started | Comparison of Multimodal Fusion Strategies |
| 09 | `phase_09_robustness_missing_modality` | Not Started | Robustness Analysis of Missing Modalities |
| 10 | `phase_10_onlinehd_lsl_simulation` | Not Started | OnlineHD and LSL Simulation |

## Current dataset status
Phase 03 has generated two versions of the four-class classification dataset:
- Main dataset: `experiments/phase_03_multimodal_dataset_labeling/results/cleaned_multimodal_four_class_without_performance.csv`
- Auxiliary dataset: `experiments/phase_03_multimodal_dataset_labeling/results/cleaned_multimodal_four_class_with_performance.csv`
The main dataset is used for subsequent Phases 04 and 05 and has excluded performance metrics to reduce the risk of performance shortcuts in the model learning task. The auxiliary dataset retains performance metrics and can be used for upper bound comparison and leakage risk analysis.
Current Phase 03 Summary:
- Number of samples: 419
- Number of classes: 4
- Number of features in the main dataset: 1,176
- Number of features in the auxiliary dataset including performance metrics: 1,235
- Number of participants: 35
- Recommended validation method: subject-wise GroupKFold, 50% recommended
- Class distribution: 104, 106, 104, 105, generally balanced

## 环境安装

It is recommended to use Python 3.10 or later.
``bash
pip install -r requirements.txt
```
Main dependencies include:
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

## Key Design Principles

- The raw data is read-only; experimental outputs are not written into `vrdataset`
- Avoid shortcut learning by excluding performance metrics from the main modeling pipeline
- Retain missing values to be handled during the modeling stage
- Adopt subject-wise splitting to evaluate generalization capability
- Store stage outputs in their corresponding phase directories for easy auditing and reproducibility

## Subsequent Work
- Train a traditional machine learning four-classification baseline
- Implement a basic HDC four-classification model
- Screen different HDC classifiers and encoding strategies
- Analyze the contribution of single modalities
- Compare multi-modal fusion strategies
- Test the robustness against missing modalities
- Explore scenario simulation of OnlineHD and LSL

## Permissions and Data Restrictions
Codes and experimental outputs may be used for research reproduction and method development. Raw data is subject to the PhysioNet Restricted Health Data License, and users shall independently verify data access rights, licensing requirements and sharing restrictions.
