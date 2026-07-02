# Revised Experiment Plan

## Research Task
The final thesis task is four-class workload proxy classification for VR piloting difficulty levels 1, 2, 3, and 4. The later core method will be Hyperdimensional Computing (HDC), with traditional machine-learning baselines and multimodal ablations used for comparison.

## Data Source Positioning
`vrdataset/dataPackage` is the main source for the full multimodal experiment. It contains the raw and derived task files that must be audited for modality coverage, subject/session/run structure, difficulty levels, and quality before feature extraction.

`vrdataset/starterCode` is useful as a reference pipeline, but it is mainly an eye-tracking and oculomotor pilot workflow. It should not be treated as the final full multimodal dataset. Any use of starter code should be limited to understanding conventions, checking example processing patterns, or comparing against reference feature files.

## Data Safety Rule
Do not delete, rename, overwrite, or modify anything inside `vrdataset`. Store all generated notebooks, scripts, results, figures, tables, logs, and reports under `experiments`.

## Revised Phase Structure
| Phase | Folder | Purpose |
|---:|---|---|
| 00 | `phase_00_project_setup` | Project setup and data inventory |
| 01 | `phase_01_raw_data_modality_audit` | Raw data audit and modality inventory |
| 02 | `phase_02_full_multimodal_feature_extraction` | Full multimodal feature extraction |
| 03 | `phase_03_multimodal_dataset_labeling` | Multimodal dataset construction and four-class labeling |
| 04 | `phase_04_ml_baseline_four_class` | Traditional ML four-class baseline |
| 05 | `phase_05_basic_hdc_four_class` | Basic HDC four-class classification |
| 06 | `phase_06_hdc_classifier_screening` | HDC classifier screening |
| 07 | `phase_07_single_modality_contribution` | Single-modality contribution analysis |
| 08 | `phase_08_multimodal_fusion` | Multimodal fusion strategy comparison |
| 09 | `phase_09_robustness_missing_modality` | Robustness and missing-modality analysis |
| 10 | `phase_10_onlinehd_lsl_simulation` | OnlineHD and LSL simulation |

## Phase 00 Scope
Phase 00 only sets up the project, inventories files, and records the revised plan. It must not perform feature extraction, labeling, ML training, HDC training, or model evaluation.

## Next Phase
Phase 01 should audit raw modality coverage in `vrdataset/dataPackage`, identify available streams and sensors, verify task/run/subject/session metadata, and design the manifest that Phase 02 will use for feature extraction.
