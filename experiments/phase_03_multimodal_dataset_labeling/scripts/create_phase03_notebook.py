import nbformat as nbf
from pathlib import Path

root = Path.cwd()
phase_dir = root / "experiments" / "phase_03_multimodal_dataset_labeling"
notebook_path = phase_dir / "notebooks" / "03_multimodal_dataset_labeling_four_class.ipynb"
notebook_path.parent.mkdir(parents=True, exist_ok=True)

cells = [
    nbf.v4.new_markdown_cell("""# Phase 03: Multimodal Dataset Labeling for Four-Class Workload Proxy Classification\n\nThis notebook constructs final run-level modeling datasets for task-difficulty-induced workload proxy classification. The target labels are difficulty levels 1, 2, 3, and 4 encoded as targets 0, 1, 2, and 3. These are workload proxy labels induced by task difficulty, not direct psychological stress or questionnaire-based workload labels. No ML model, HDC model, global scaling, or global imputation is performed in this phase."""),
    nbf.v4.new_code_cell("""from pathlib import Path\nimport sys\nfrom IPython.display import Markdown, display\n\nPROJECT_ROOT = Path.cwd()\nPHASE_DIR = PROJECT_ROOT / 'experiments' / 'phase_03_multimodal_dataset_labeling'\nsys.path.insert(0, str(PHASE_DIR / 'scripts'))\nfrom phase03_dataset import run_phase03\n\nartifacts = run_phase03(PROJECT_ROOT)\n\nprint(f\"Loaded Phase 02 feature table shape: {artifacts['phase02_shape']}\")\ndisplay(artifacts['phase02_head'])\nprint('Identifier column check:')\ndisplay(artifacts['identifier_check'])"""),
    nbf.v4.new_markdown_cell("""## Difficulty Labels and Target Encoding\n\nThe notebook parses difficulty labels, keeps only levels 1-4, removes duplicate `run_key` rows if present, and creates `target = difficulty_level - 1`."""),
    nbf.v4.new_code_cell("""print('Difficulty level distribution before filtering, as loaded from Phase 02:')\ndisplay(artifacts['raw_difficulty_distribution'])\nprint('Difficulty level distribution before filtering, after numeric parsing:')\ndisplay(artifacts['numeric_difficulty_distribution_before'])\nprint('Difficulty level distribution after keeping levels 1, 2, 3, and 4:')\ndisplay(artifacts['difficulty_distribution_after'])\nprint('Target mapping:')\ndisplay(artifacts['target_mapping_table'])"""),
    nbf.v4.new_markdown_cell("""## Dataset Construction and Feature Audits\n\nIdentifier and label columns are preserved for interpretation but excluded from input feature lists. Performance metrics are separated from the main dataset to reduce shortcut-learning risk."""),
    nbf.v4.new_code_cell("""print('Class distribution table:')\ndisplay(artifacts['class_distribution'])\nprint('Subject distribution table:')\ndisplay(artifacts['subject_distribution'])\nprint('Missing value summary, top 20 by missing ratio:')\ndisplay(artifacts['missing_value_summary'].head(20))\nprint('Missing value by feature group:')\ndisplay(artifacts['missing_by_group'])\nprint('Feature group summary without performance:')\ndisplay(artifacts['feature_count_by_group_without'])\nprint('Feature group summary with performance:')\ndisplay(artifacts['feature_count_by_group_with'])\nprint(f\"Number of features without performance metrics: {artifacts['feature_count_without_performance']}\")\nprint(f\"Number of features with performance metrics: {artifacts['feature_count_with_performance']}\")\nprint('Removed high-missing features:')\ndisplay(artifacts['removed_features_high_missing'])\nprint('Dropped non-numeric columns:')\ndisplay(artifacts['dropped_non_numeric_columns'])"""),
    nbf.v4.new_markdown_cell("""## Leakage Check, GroupKFold Readiness, and Final Comparison\n\nThe final outputs include two dataset versions, feature-column files, feature-group files, leakage and construction reports, summary tables, figures, logs, README updates, and experiment status updates."""),
    nbf.v4.new_code_cell("""print('Leakage check conclusion:')\ndisplay(Markdown(artifacts['leakage_check_report']['conclusion']))\nprint('GroupKFold readiness conclusion:')\ndisplay(Markdown(artifacts['groupkfold_conclusion']))\nprint('Final dataset version comparison:')\ndisplay(artifacts['dataset_version_comparison'])\nprint('Saved Phase 03 outputs under:')\ndisplay(Markdown(f\"`{artifacts['phase_dir']}`\"))"""),
]

nb = nbf.v4.new_notebook(cells=cells)
nb['metadata'] = {
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'pygments_lexer': 'ipython3'},
}
nbf.write(nb, notebook_path)
print(notebook_path)
