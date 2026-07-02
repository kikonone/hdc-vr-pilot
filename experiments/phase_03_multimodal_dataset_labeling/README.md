# Phase 03: Multimodal Dataset Construction and Four-Class Labeling

## What this phase does
Phase 03 converts the Phase 02 run-level multimodal feature table into clean four-class modeling datasets. It parses task difficulty, keeps only difficulty levels 1, 2, 3, and 4, creates `target = difficulty_level - 1`, separates identifiers from input features, audits leakage risk, and saves dataset metadata for later modeling phases.

This phase does not train ML models, train HDC models, fit imputers, or fit scalers. Missing values remain as NaN so Phase 04 and Phase 05 can handle imputation and scaling inside training folds.

## Workload proxy labels
The labels are task-difficulty-induced workload proxy labels. They come from the experiment difficulty level rather than direct psychological stress annotation or questionnaire-based workload measurement.

| difficulty_level | target |
|---:|---:|
| 1 | 0 |
| 2 | 1 |
| 3 | 2 |
| 4 | 3 |

## Why performance metrics are separated
Performance metrics can be strongly tied to task difficulty and may let models learn shortcuts instead of workload-related multimodal patterns. For that reason, Phase 03 creates two dataset versions:

- `cleaned_multimodal_four_class_without_performance.csv`: primary leakage-controlled dataset for Phase 04 ML baselines and Phase 05 HDC.
- `cleaned_multimodal_four_class_with_performance.csv`: auxiliary upper-bound comparison dataset for testing shortcut-learning risk.

## Dataset summary
- Samples retained: 419
- Features without performance metrics: 1176
- Features with performance metrics: 1235
- High-missing features removed at 95% threshold: 12
- Non-numeric non-identifier columns dropped from feature lists: 0

## Subject-wise validation readiness
Subject-wise GroupKFold is feasible with 35 subjects; recommended splits: 5.

## Recommended next phase
Phase 04 should train classical ML baselines using `cleaned_multimodal_four_class_without_performance.csv` as the main dataset. Any imputation, scaling, feature selection, and model fitting should happen inside cross-validation folds. The with-performance dataset should only be used as an auxiliary comparison to estimate shortcut-learning risk and upper-bound performance.
