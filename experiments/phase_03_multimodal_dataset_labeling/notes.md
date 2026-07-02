# Notes: Phase 03 Multimodal Dataset Labeling

## Input Findings
- Phase 02 feature table exists at `experiments/phase_02_full_multimodal_feature_extraction/results/full_multimodal_run_level_features.csv`.
- The table has 487 rows and required identifiers: `subject_id`, `session_id`, `run_id`, `difficulty_level`, and `run_key`.
- `difficulty_level` is represented with strings such as `level-000`, `level-01B`, `level-02B`, `level-03B`, and `level-04B`.
- `run_key` has no duplicates in the Phase 02 table.
- `feature_groups.json` separates 59 `performance_features` from other multimodal groups.

## Construction Choices
- Parse difficulty into numeric values and keep the cleaned numeric `difficulty_level` column in final datasets.
- Create `target = difficulty_level - 1` for classes 0, 1, 2, and 3.
- Keep NaN values; do not impute or scale.
- Remove numeric features only when missing ratio is at least 95%.
- Use the without-performance dataset as the primary Phase 04/05 input.
