# Experiment Status

## Project
- Task: Four-class workload proxy classification for PhysioNet VR Piloting data
- Target labels: difficulty levels 1, 2, 3, and 4
- Final method family: Hyperdimensional Computing (HDC)
- Current completed scope: Phase 03 multimodal dataset construction and four-class labeling
- Data rule: `vrdataset` is read-only source data; generated artifacts belong under `experiments`

## Source Data Positioning
- `vrdataset/dataPackage`: main source for the full multimodal experiment
- `vrdataset/starterCode`: eye-tracking and oculomotor pilot reference only, not the final full multimodal dataset
- `vrdataset/referenceDocuments`: documentation and data dictionary reference

## Revised Phase Status
| Phase | Folder | Status | Notes |
|---:|---|---|---|
| 00 | `phase_00_project_setup` | Complete | Revised folder structure, full inventory, phase structure CSV, setup report, README, and executed notebook created. |
| 01 | `phase_01_raw_data_modality_audit` | Complete | Raw dataPackage scan, file-level inventory, run-level modality availability, summaries, figures, JSON report, README, and executed notebook created. |
| 02 | `phase_02_full_multimodal_feature_extraction` | Complete | Full multimodal run-level aggregate feature extraction completed from `vrdataset/dataPackage`; missing modalities retained as NaN. |
| 03 | `phase_03_multimodal_dataset_labeling` | Complete | Four-class workload proxy datasets created with and without performance metrics; no model training performed. |
| 04 | `phase_04_ml_baseline_four_class` | Not started | No ML training in Phase 00. |
| 05 | `phase_05_basic_hdc_four_class` | Not started | No HDC training in Phase 00. |
| 06 | `phase_06_hdc_classifier_screening` | Not started | No classifier screening in Phase 00. |
| 07 | `phase_07_single_modality_contribution` | Not started | No modality contribution analysis in Phase 00. |
| 08 | `phase_08_multimodal_fusion` | Not started | No fusion comparison in Phase 00. |
| 09 | `phase_09_robustness_missing_modality` | Not started | No robustness experiments in Phase 00. |
| 10 | `phase_10_onlinehd_lsl_simulation` | Not started | No OnlineHD or LSL simulation in Phase 00. |

## Phase 00 Outputs
- Executed notebook: `experiments/phase_00_project_setup/notebooks/00_revised_project_setup_and_inventory.ipynb`
- Full data inventory: `experiments/phase_00_project_setup/results/data_inventory.csv`
- Revised phase structure: `experiments/phase_00_project_setup/results/revised_phase_structure.csv`
- Setup report: `experiments/phase_00_project_setup/results/project_setup_report.json`
- File type table: `experiments/phase_00_project_setup/tables/file_type_summary.csv`
- Folder summary table: `experiments/phase_00_project_setup/tables/folder_summary.csv`
- Phase log: `experiments/phase_00_project_setup/logs/phase_00_log.txt`
- Revised plan: `experiments/shared/reports/REVISED_EXPERIMENT_PLAN.md`

## Inventory Summary
- Total files under `vrdataset`: 9,022
- Files under `vrdataset/dataPackage`: 9,003
- Files under `vrdataset/starterCode`: 12
- Files under `vrdataset/starterCode/data_feats`: 1
- Files under `vrdataset/referenceDocuments`: 5
- Requested extension counts: csv 9,004; mat 0; xdf 0; txt 2; npy 0; npz 0; pkl 0; parquet 0; py 4; m 1; ipynb 6


## Phase 01 Outputs
- Executed notebook: `experiments/phase_01_raw_data_modality_audit/notebooks/01_raw_data_modality_audit.ipynb`
- File inventory: `experiments/phase_01_raw_data_modality_audit/results/raw_file_inventory.csv`
- Run modality availability: `experiments/phase_01_raw_data_modality_audit/results/run_modality_availability.csv`
- Audit report: `experiments/phase_01_raw_data_modality_audit/results/raw_data_audit_report.json`
- Unknown review list: `experiments/phase_01_raw_data_modality_audit/results/unknown_files_for_review.csv`
- Unreadable file list: `experiments/phase_01_raw_data_modality_audit/results/unreadable_files.csv`
- Figures: `experiments/phase_01_raw_data_modality_audit/figures/modality_availability_bar.png`, `experiments/phase_01_raw_data_modality_audit/figures/difficulty_distribution.png`
- README: `experiments/phase_01_raw_data_modality_audit/README.md`

## Phase 01 Summary
- Files audited: 9003
- Readable files: 9003
- Subjects: 35
- Sessions: 35
- Runs: 487
- Unknown files requiring manual review: 908
- Unreadable files: 0
- Strict complete multimodal runs: 0
- Core complete multimodal runs without control input: 408


## Phase 02 Outputs
- Executed notebook: `experiments/phase_02_full_multimodal_feature_extraction/notebooks/02_full_multimodal_feature_extraction.ipynb`
- Full run-level feature table: `experiments/phase_02_full_multimodal_feature_extraction/results/full_multimodal_run_level_features.csv`
- Long feature table: `experiments/phase_02_full_multimodal_feature_extraction/results/feature_extraction_long_table.csv`
- Feature groups: `experiments/phase_02_full_multimodal_feature_extraction/results/feature_groups.json`
- Failed file log: `experiments/phase_02_full_multimodal_feature_extraction/results/failed_files.csv`
- Extraction report: `experiments/phase_02_full_multimodal_feature_extraction/results/extraction_report.json`
- Summary tables: `features_per_modality.csv`, `runs_with_extracted_modalities.csv`, `missing_modalities_after_extraction.csv`
- Figures: `extracted_features_per_modality.png`, `runs_per_modality_after_extraction.png`
- Phase log: `experiments/phase_02_full_multimodal_feature_extraction/logs/phase_02_log.txt`
- README: `experiments/phase_02_full_multimodal_feature_extraction/README.md`

## Phase 02 Summary
- Runs in output: 487
- Extracted feature columns excluding identifiers: 1,247
- Long-table feature rows: 599,569
- Processed feature files: 4,945
- Skipped header/non-feature files: 4,057
- Failed files: 0
- Extracted modalities: eye_tracking, ecg, eda, emg, respiration, head_movement, xplane, performance, unknown
- Explicit control input streams: unavailable after extraction
- Performance features retained but grouped separately for leakage-sensitive Phase 03 dataset versions
- No final imputation, scaling, feature selection, four-class labeling, ML training, or HDC training was performed


## Phase 03 Outputs
- Executed notebook: `experiments/phase_03_multimodal_dataset_labeling/notebooks/03_multimodal_dataset_labeling_four_class.ipynb`
- Primary dataset: `experiments/phase_03_multimodal_dataset_labeling/results/cleaned_multimodal_four_class_without_performance.csv`
- Auxiliary dataset: `experiments/phase_03_multimodal_dataset_labeling/results/cleaned_multimodal_four_class_with_performance.csv`
- Feature columns and groups saved for both dataset versions.
- Target mapping: `experiments/phase_03_multimodal_dataset_labeling/results/target_mapping.json`
- Leakage report: `experiments/phase_03_multimodal_dataset_labeling/results/leakage_check_report.json`
- Construction report: `experiments/phase_03_multimodal_dataset_labeling/results/dataset_construction_report.json`
- Summary tables, figures, README, and phase log were created under Phase 03.

## Phase 03 Summary
- Completion status: Complete; notebook executed and saved with visible outputs.
- Samples in `cleaned_multimodal_four_class_without_performance.csv`: 419
- Samples in `cleaned_multimodal_four_class_with_performance.csv`: 419
- Features without performance metrics: 1,176
- Features with performance metrics: 1,235
- Number of classes: 4
- Class distribution: difficulty 1 / target 0 = 104; difficulty 2 / target 1 = 106; difficulty 3 / target 2 = 104; difficulty 4 / target 3 = 105.
- Subjects available: 35
- Subject-wise GroupKFold feasible: yes; recommended splits: 5.
- Target encoding verified: difficulty_level 1, 2, 3, and 4 map to target 0, 1, 2, and 3.
- Identifier and label columns are excluded from feature column files.
- Performance features in without-performance feature list: 0.
- Performance features in with-performance feature list and group: 59.
- Recommended main dataset for Phase 04 and Phase 05: `cleaned_multimodal_four_class_without_performance.csv`
- Auxiliary upper-bound dataset: `cleaned_multimodal_four_class_with_performance.csv`
- No global imputation, global scaling, ML training, or HDC training was performed.
- Warnings: incidental `_write_test.tmp` and a Python cache temp file remain under the Phase 03 folder from Windows filesystem troubleshooting; they are not requested result files and do not affect the datasets. No empty Phase 03 result, table, figure, or log files were found.
- Recommended next step: Phase 04 traditional ML four-class baseline using the without-performance dataset.

## Phase 00-02 Verification Audit
- Audit report: `experiments/shared/reports/PHASE_00_02_AUDIT_REPORT.json`
- Audit notes: `experiments/shared/reports/PHASE_00_02_AUDIT_NOTES.md`
- Audit date: 2026-07-02

### Completed Phases
- Phase 00 `phase_00_project_setup`: Complete; required folders exist, README exists, and notebook is executed with visible outputs. The `figures` folder exists but contains 0 files.
- Phase 01 `phase_01_raw_data_modality_audit`: Complete; required folders exist, README exists, notebook is executed with visible outputs, and required audit outputs are present.
- Phase 02 `phase_02_full_multimodal_feature_extraction`: Complete; required folders exist, README exists, notebook is executed with visible outputs, and required feature extraction outputs are present.

### Failed Or Incomplete Phases
- No failed Phase 00-02 phase was found by this audit.
- Phases 03-10 remain not started.

### Folder And Notebook Check
- Phase 00: README=yes; notebooks executed with visible outputs=yes; notebooks=yes (1 files); scripts=yes (2 files); results=yes (3 files); figures=yes (0 files); tables=yes (2 files); logs=yes (3 files)
- Phase 01: README=yes; notebooks executed with visible outputs=yes; notebooks=yes (1 files); scripts=yes (3 files); results=yes (6 files); figures=yes (2 files); tables=yes (3 files); logs=yes (1 files)
- Phase 02: README=yes; notebooks executed with visible outputs=yes; notebooks=yes (1 files); scripts=yes (2 files); results=yes (7 files); figures=yes (2 files); tables=yes (3 files); logs=yes (5 files)

### Missing Required Files
None among the requested Phase 01 and Phase 02 required outputs.

### Required Phase 01 Outputs
- `raw_file_inventory.csv` present (6022253 bytes), `run_modality_availability.csv` present (97324 bytes), `modality_summary.csv` present (156 bytes), `unknown_files_for_review.csv` present (593789 bytes)

### Required Phase 02 Outputs
- `full_multimodal_run_level_features.csv` present (8848511 bytes), `feature_groups.json` present (63934 bytes), `failed_files.csv` present (96 bytes), `extraction_report.json` present (5073 bytes)

### Identifier Column Check
- Phase 01 run availability identifiers present: {'subject_id': True, 'session_id': True, 'run_id': True, 'difficulty_level': True, 'run_key': True}; rows: 487; unique runs: 487.
- Phase 02 feature table identifiers present: {'subject_id': True, 'session_id': True, 'run_id': True, 'difficulty_level': True, 'run_key': True}; rows: 487; total columns: 1252; feature columns excluding identifiers: 1247; duplicate columns: 0.

### Extracted Modalities And Feature Counts
- eye_tracking: 426 features; 487 runs with features
- ecg: 57 features; 487 runs with features
- eda: 44 features; 475 runs with features
- emg: 84 features; 487 runs with features
- respiration: 48 features; 487 runs with features
- head_movement: 159 features; 487 runs with features
- xplane: 328 features; 487 runs with features
- control_input: 0 features; 0 runs with features
- performance: 59 features; 419 runs with features
- unknown: 42 features; 454 runs with features

### Feature Group Separation
- Feature group counts: {'identifier_columns': 5, 'physiological_features': 233, 'eye_tracking_features': 426, 'head_movement_features': 159, 'flight_parameter_features': 328, 'control_input_features': 0, 'performance_features': 59, 'unknown_features': 42}
- Performance features are separated in `feature_groups.json`: 59 features under `performance_features`.
- Unknown files were not forced into known modalities: 42 torso accelerometer-derived features are under `unknown_features`.
- The extracted feature dataset is multimodal: True; extracted modalities with data are eye_tracking, ecg, eda, emg, respiration, head_movement, xplane, performance, unknown.

### Empty Files
- Phase 02: `experiments\phase_02_full_multimodal_feature_extraction\logs\notebook_execute_stdout.txt`
- Phase 02: `experiments\phase_02_full_multimodal_feature_extraction\logs\notebook_execute_stdout_rerun.txt`

### vrdataset Read-Only Evidence
- Files found under `vrdataset`: 9022.
- Latest source-data file timestamp found by audit: `vrdataset\SHA256SUMS.txt` at 2022-08-25T03:32:34.
- Phase artifact names found under `vrdataset`: none.
- Interpretation: Phase 00-02 outputs are stored under `experiments`; no requested output artifact was found inside `vrdataset`.

### Phase 02 Scope Guard
- No suspicious modeling artifacts were found in Phase 02: [].
- `extraction_report.json` records that Phase 02 avoided final imputation, scaling, feature selection, four-class labeling, ML training, and HDC training.

### Questions Requiring Manual Confirmation
- No Phase 00-02 output artifact names were found under vrdataset; read-only status is supported by file placement, but only OS/file-history auditing can prove no metadata changes occurred.
- Confirm whether the absence of explicit control-input streams is expected, or whether another raw stream should be interpreted as controls.
- Confirm how Phase 03 should handle unknown torso accelerometer features: exclude, keep as unknown, or map after domain review.

### Recommended Next Step
- Proceed to Phase 03: multimodal dataset construction and four-class labeling.
- Phase 03 should create dataset versions with and without `performance_features`, decide how to handle `unknown_features`, keep imputation/scaling inside modeling pipelines, and avoid using performance features in leakage-sensitive primary models unless explicitly intended.


## Latest Update
- Phase 03 output check completed on 2026-07-02.
- Phase 03 is complete: notebook exists, is executed, and contains visible outputs.
- Primary dataset has 419 samples and 1,176 features; auxiliary with-performance dataset has 419 samples and 1,235 features.
- Class distribution is balanced across four classes: 104, 106, 104, and 105 samples.
- Subject-wise GroupKFold is feasible with 35 subjects; recommended splits: 5.
- Phase 03 avoided model training, HDC training, global scaling, and global imputation.
- Recommended next step: Phase 04 traditional ML four-class baseline.
