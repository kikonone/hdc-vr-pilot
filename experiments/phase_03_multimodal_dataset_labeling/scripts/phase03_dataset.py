from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_difficulty_value(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return int(value)
    text = str(value).strip().lower()
    if text.isdigit():
        return int(text)
    match = re.search(r"level[-_ ]*0*([0-9]+)", text)
    if match:
        return int(match.group(1))
    match = re.search(r"(?<!\d)([0-9]+)(?!\d)", text)
    if match:
        return int(match.group(1))
    return np.nan


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def make_dirs(phase_dir: Path) -> dict[str, Path]:
    dirs = {
        "results": phase_dir / "results",
        "tables": phase_dir / "tables",
        "figures": phase_dir / "figures",
        "logs": phase_dir / "logs",
        "notebooks": phase_dir / "notebooks",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def build_feature_group_lookup(raw_feature_groups: dict[str, list[str]]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for group_name, columns in raw_feature_groups.items():
        if group_name == "identifier_columns":
            continue
        for column in columns:
            lookup.setdefault(column, group_name)
    return lookup


def run_phase03(project_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(project_root) if project_root is not None else Path.cwd()
    phase_dir = root / "experiments" / "phase_03_multimodal_dataset_labeling"
    dirs = make_dirs(phase_dir)
    results_dir = dirs["results"]
    tables_dir = dirs["tables"]
    figures_dir = dirs["figures"]
    logs_dir = dirs["logs"]

    phase02_results = root / "experiments" / "phase_02_full_multimodal_feature_extraction" / "results"
    feature_table_path = phase02_results / "full_multimodal_run_level_features.csv"
    feature_groups_path = phase02_results / "feature_groups.json"

    phase02_table = pd.read_csv(feature_table_path)
    raw_feature_groups = json.loads(feature_groups_path.read_text(encoding="utf-8"))

    required_ids = ["subject_id", "session_id", "run_id", "difficulty_level", "run_key"]
    identifier_check = pd.DataFrame([{"column": col, "present": col in phase02_table.columns} for col in required_ids])

    working = phase02_table.copy()
    difficulty_source_column = None
    if "difficulty_level" in working.columns:
        difficulty_source_column = "difficulty_level"
    else:
        for candidate in ["run_key", "run_id"]:
            if candidate in working.columns:
                parsed = working[candidate].map(parse_difficulty_value)
                if parsed.notna().any():
                    working["difficulty_level"] = parsed
                    difficulty_source_column = candidate
                    break

    if "difficulty_level" not in working.columns:
        diagnostic = {
            "status": "failed",
            "reason": "difficulty_level missing and not parseable from run_key or run_id",
            "available_columns": list(working.columns),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        write_json(results_dir / "dataset_construction_diagnostic_report.json", diagnostic)
        raise RuntimeError("Difficulty labels unavailable; diagnostic report saved.")

    raw_difficulty_distribution = working["difficulty_level"].value_counts(dropna=False).rename_axis("raw_difficulty_level").reset_index(name="sample_count")
    working["difficulty_level"] = working["difficulty_level"].map(parse_difficulty_value)
    if working["difficulty_level"].isna().all():
        diagnostic = {
            "status": "failed",
            "reason": "difficulty_level values could not be parsed into numeric levels",
            "difficulty_source_column": difficulty_source_column,
            "raw_difficulty_distribution": raw_difficulty_distribution.astype(str).to_dict(orient="records"),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        write_json(results_dir / "dataset_construction_diagnostic_report.json", diagnostic)
        raise RuntimeError("Difficulty labels could not be parsed; diagnostic report saved.")

    numeric_difficulty_distribution_before = working["difficulty_level"].value_counts(dropna=False).sort_index().rename_axis("difficulty_level").reset_index(name="sample_count")
    duplicate_run_key_count = int(working["run_key"].duplicated().sum()) if "run_key" in working.columns else 0
    if "run_key" in working.columns:
        working = working.drop_duplicates(subset=["run_key"], keep="first").copy()

    four_class = working[working["difficulty_level"].isin([1, 2, 3, 4])].copy()
    four_class["difficulty_level"] = four_class["difficulty_level"].astype(int)
    four_class["target"] = four_class["difficulty_level"] - 1

    difficulty_distribution_after = four_class["difficulty_level"].value_counts().sort_index().rename_axis("difficulty_level").reset_index(name="sample_count")
    target_mapping = {
        "description": "Task-difficulty-induced workload proxy label mapping; not direct psychological stress or questionnaire workload labels.",
        "difficulty_level_to_target": {"1": 0, "2": 1, "3": 2, "4": 3},
        "target_to_difficulty_level": {"0": 1, "1": 2, "2": 3, "3": 4},
    }

    identifier_output_order = ["subject_id", "session_id", "run_id", "difficulty_level", "target", "run_key"]
    identifier_columns_available = [col for col in identifier_output_order if col in four_class.columns]
    forbidden_feature_names = {
        "target", "difficulty_level", "difficulty", "level", "task_difficulty",
        "subject_id", "subject", "participant", "session_id", "session",
        "run_id", "run", "trial_id", "trial", "run_key", "label",
    }
    leakage_identifier_or_label_columns = sorted([col for col in four_class.columns if col.lower() in forbidden_feature_names])

    performance_keywords = ["perfmetric", "performance", "error", "cumulative_error", "glideslope", "localizer", "airspeed_error"]
    performance_from_groups = set(raw_feature_groups.get("performance_features", []))
    performance_from_names = {col for col in four_class.columns if any(keyword in col.lower() for keyword in performance_keywords)}
    performance_columns_identified = sorted((performance_from_groups | performance_from_names).intersection(four_class.columns))

    base_excluded_columns = set(identifier_columns_available) | set(leakage_identifier_or_label_columns)
    candidate_feature_columns = [col for col in four_class.columns if col not in base_excluded_columns]
    numeric_candidate_columns = four_class[candidate_feature_columns].select_dtypes(include=[np.number]).columns.tolist()
    dropped_non_numeric_columns = sorted(set(candidate_feature_columns) - set(numeric_candidate_columns))

    missing_ratio = four_class[numeric_candidate_columns].isna().mean().sort_values(ascending=False)
    removed_high_missing_columns = missing_ratio[missing_ratio >= 0.95].index.tolist()
    valid_numeric_features = [col for col in numeric_candidate_columns if col not in set(removed_high_missing_columns)]
    feature_columns_with_performance = valid_numeric_features
    feature_columns_without_performance = [col for col in valid_numeric_features if col not in set(performance_columns_identified)]

    dataset_without_performance = four_class[identifier_columns_available + feature_columns_without_performance].copy()
    dataset_with_performance = four_class[identifier_columns_available + feature_columns_with_performance].copy()

    class_distribution = four_class.groupby(["difficulty_level", "target"], dropna=False).size().reset_index(name="sample_count").sort_values(["difficulty_level", "target"])
    class_distribution["proportion"] = class_distribution["sample_count"] / class_distribution["sample_count"].sum()

    if "subject_id" in four_class.columns:
        subject_distribution = four_class.groupby("subject_id", dropna=False).size().reset_index(name="sample_count")
        class_by_subject = pd.crosstab(four_class["subject_id"], four_class["target"]).reset_index()
        class_by_subject.columns = ["subject_id"] + [f"target_{col}_count" for col in class_by_subject.columns[1:]]
        subject_distribution = subject_distribution.merge(class_by_subject, on="subject_id", how="left")
    else:
        subject_distribution = pd.DataFrame(columns=["subject_id", "sample_count"])

    feature_to_group = build_feature_group_lookup(raw_feature_groups)

    def assign_feature_group(column: str) -> str:
        if column in feature_to_group:
            return feature_to_group[column]
        if column in performance_columns_identified:
            return "performance_features"
        return "unassigned_features"

    missing_value_summary = pd.DataFrame({
        "feature": numeric_candidate_columns,
        "feature_group": [assign_feature_group(col) for col in numeric_candidate_columns],
        "missing_count": [int(four_class[col].isna().sum()) for col in numeric_candidate_columns],
        "missing_ratio": [float(four_class[col].isna().mean()) for col in numeric_candidate_columns],
    })
    missing_value_summary["removed_high_missing"] = missing_value_summary["feature"].isin(removed_high_missing_columns)
    missing_value_summary["in_without_performance_dataset"] = missing_value_summary["feature"].isin(feature_columns_without_performance)
    missing_value_summary["in_with_performance_dataset"] = missing_value_summary["feature"].isin(feature_columns_with_performance)

    missing_by_group = missing_value_summary.groupby("feature_group", dropna=False).agg(
        feature_count=("feature", "count"),
        mean_missing_ratio=("missing_ratio", "mean"),
        max_missing_ratio=("missing_ratio", "max"),
        high_missing_removed=("removed_high_missing", "sum"),
    ).reset_index().sort_values("feature_group")

    def filtered_feature_groups(feature_columns: list[str]) -> dict[str, list[str]]:
        feature_set = set(feature_columns)
        groups = {"identifier_columns": identifier_columns_available}
        assigned: set[str] = set()
        for group_name, columns in raw_feature_groups.items():
            if group_name == "identifier_columns":
                continue
            kept = [col for col in columns if col in feature_set]
            groups[group_name] = kept
            assigned.update(kept)
        unassigned = [col for col in feature_columns if col not in assigned]
        if unassigned:
            groups["unassigned_features"] = unassigned
        return groups

    feature_groups_without_performance = filtered_feature_groups(feature_columns_without_performance)
    feature_groups_with_performance = filtered_feature_groups(feature_columns_with_performance)
    feature_count_by_group_without = pd.DataFrame([{"feature_group": g, "feature_count": len(c)} for g, c in feature_groups_without_performance.items() if g != "identifier_columns"]).sort_values("feature_group")
    feature_count_by_group_with = pd.DataFrame([{"feature_group": g, "feature_count": len(c)} for g, c in feature_groups_with_performance.items() if g != "identifier_columns"]).sort_values("feature_group")

    dataset_version_comparison = pd.DataFrame([
        {
            "dataset_version": "without_performance",
            "recommended_use": "Primary Phase 04 ML and Phase 05 HDC dataset",
            "sample_count": len(dataset_without_performance),
            "identifier_column_count": len(identifier_columns_available),
            "feature_count": len(feature_columns_without_performance),
            "performance_features_included": 0,
            "performance_features_excluded": len([col for col in performance_columns_identified if col in valid_numeric_features]),
        },
        {
            "dataset_version": "with_performance",
            "recommended_use": "Auxiliary upper-bound and shortcut-learning risk comparison",
            "sample_count": len(dataset_with_performance),
            "identifier_column_count": len(identifier_columns_available),
            "feature_count": len(feature_columns_with_performance),
            "performance_features_included": len([col for col in performance_columns_identified if col in feature_columns_with_performance]),
            "performance_features_excluded": 0,
        },
    ])

    removed_features_high_missing = pd.DataFrame({
        "feature": removed_high_missing_columns,
        "missing_ratio": [float(missing_ratio[col]) for col in removed_high_missing_columns],
        "feature_group": [assign_feature_group(col) for col in removed_high_missing_columns],
    })
    dropped_non_numeric_table = pd.DataFrame({
        "column": dropped_non_numeric_columns,
        "reason": "non-numeric non-identifier column excluded from feature lists",
    })

    dataset_without_performance.to_csv(results_dir / "cleaned_multimodal_four_class_without_performance.csv", index=False)
    dataset_with_performance.to_csv(results_dir / "cleaned_multimodal_four_class_with_performance.csv", index=False)
    write_json(results_dir / "feature_columns_without_performance.json", feature_columns_without_performance)
    write_json(results_dir / "feature_columns_with_performance.json", feature_columns_with_performance)
    write_json(results_dir / "feature_groups_without_performance.json", feature_groups_without_performance)
    write_json(results_dir / "feature_groups_with_performance.json", feature_groups_with_performance)
    write_json(results_dir / "target_mapping.json", target_mapping)
    removed_features_high_missing.to_csv(results_dir / "removed_features_high_missing.csv", index=False)
    dropped_non_numeric_table.to_csv(results_dir / "dropped_non_numeric_columns.csv", index=False)

    class_distribution.to_csv(tables_dir / "class_distribution.csv", index=False)
    subject_distribution.to_csv(tables_dir / "subject_distribution.csv", index=False)
    missing_value_summary.to_csv(tables_dir / "missing_value_summary.csv", index=False)
    feature_count_by_group_without.to_csv(tables_dir / "feature_count_by_group_without_performance.csv", index=False)
    feature_count_by_group_with.to_csv(tables_dir / "feature_count_by_group_with_performance.csv", index=False)
    dataset_version_comparison.to_csv(tables_dir / "dataset_version_comparison.csv", index=False)

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(class_distribution["difficulty_level"].astype(str), class_distribution["sample_count"], color="#3b82f6")
    ax.set_xlabel("Difficulty level")
    ax.set_ylabel("Run count")
    ax.set_title("Four-class difficulty distribution")
    for x, y in zip(class_distribution["difficulty_level"].astype(str), class_distribution["sample_count"]):
        ax.text(x, y, str(int(y)), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(figures_dir / "class_distribution.png", dpi=180)
    plt.close(fig)

    if not subject_distribution.empty:
        fig, ax = plt.subplots(figsize=(11, 5))
        subject_plot = subject_distribution.sort_values("subject_id")
        ax.bar(subject_plot["subject_id"].astype(str), subject_plot["sample_count"], color="#10b981")
        ax.set_xlabel("Subject ID")
        ax.set_ylabel("Run count")
        ax.set_title("Samples per subject")
        ax.tick_params(axis="x", rotation=90)
        fig.tight_layout()
        fig.savefig(figures_dir / "subject_distribution.png", dpi=180)
        plt.close(fig)
    else:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, "subject_id not available", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(figures_dir / "subject_distribution.png", dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    missing_plot = missing_by_group.sort_values("mean_missing_ratio", ascending=False)
    ax.bar(missing_plot["feature_group"], missing_plot["mean_missing_ratio"], color="#f59e0b")
    ax.set_xlabel("Feature group")
    ax.set_ylabel("Mean missing ratio")
    ax.set_title("Missing value ratio by feature group")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(figures_dir / "missing_value_by_feature_group.png", dpi=180)
    plt.close(fig)

    feature_count_plot = feature_count_by_group_with.merge(
        feature_count_by_group_without,
        on="feature_group",
        how="outer",
        suffixes=("_with_performance", "_without_performance"),
    ).fillna(0).sort_values("feature_group")
    x = np.arange(len(feature_count_plot))
    width = 0.38
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width / 2, feature_count_plot["feature_count_without_performance"], width, label="Without performance", color="#6366f1")
    ax.bar(x + width / 2, feature_count_plot["feature_count_with_performance"], width, label="With performance", color="#ef4444")
    ax.set_xticks(x)
    ax.set_xticklabels(feature_count_plot["feature_group"], rotation=45, ha="right")
    ax.set_ylabel("Feature count")
    ax.set_title("Feature count by group")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "feature_count_by_group.png", dpi=180)
    plt.close(fig)

    if "subject_id" in four_class.columns:
        unique_subjects = int(four_class["subject_id"].nunique())
        samples_per_subject = subject_distribution[["subject_id", "sample_count"]].to_dict(orient="records")
        groupkfold_feasible = unique_subjects >= 2
        recommended_splits = int(min(5, unique_subjects)) if groupkfold_feasible else 0
        groupkfold_conclusion = (
            f"Subject-wise GroupKFold is feasible with {unique_subjects} subjects; recommended splits: {recommended_splits}."
            if groupkfold_feasible else "Subject-wise GroupKFold is not feasible because fewer than two subjects are available."
        )
    else:
        unique_subjects = 0
        samples_per_subject = []
        groupkfold_feasible = False
        recommended_splits = 0
        fallback_group = "session_id" if "session_id" in four_class.columns else ("run_id" if "run_id" in four_class.columns else None)
        groupkfold_conclusion = f"subject_id is unavailable; consider grouping by {fallback_group} if scientifically appropriate." if fallback_group else "subject_id is unavailable and no clear alternative grouping column was found."

    leakage_check_report = {
        "status": "passed",
        "conclusion": "Identifier and label columns are excluded from feature lists. Performance metrics are excluded from the primary without-performance dataset and retained only for auxiliary upper-bound comparison.",
        "identifier_or_label_columns_excluded_from_features": leakage_identifier_or_label_columns,
        "forbidden_feature_names": sorted(forbidden_feature_names),
        "performance_keywords": performance_keywords,
        "performance_columns_identified": performance_columns_identified,
        "performance_columns_in_without_performance_features": sorted(set(performance_columns_identified).intersection(feature_columns_without_performance)),
        "performance_columns_in_with_performance_features": sorted(set(performance_columns_identified).intersection(feature_columns_with_performance)),
        "dropped_non_numeric_non_identifier_columns": dropped_non_numeric_columns,
        "removed_high_missing_features": removed_high_missing_columns,
    }
    construction_report = {
        "status": "complete",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_feature_table": str(feature_table_path),
        "input_feature_groups": str(feature_groups_path),
        "phase02_table_shape": list(phase02_table.shape),
        "difficulty_source_column": difficulty_source_column,
        "duplicate_run_key_rows_removed": duplicate_run_key_count,
        "rows_after_four_class_filtering": int(len(four_class)),
        "difficulty_levels_kept": [1, 2, 3, 4],
        "target_mapping": target_mapping["difficulty_level_to_target"],
        "class_distribution": class_distribution.to_dict(orient="records"),
        "identifier_columns_available": identifier_columns_available,
        "feature_count_without_performance": int(len(feature_columns_without_performance)),
        "feature_count_with_performance": int(len(feature_columns_with_performance)),
        "performance_feature_count_identified": int(len(performance_columns_identified)),
        "high_missing_threshold": 0.95,
        "removed_high_missing_feature_count": int(len(removed_high_missing_columns)),
        "dropped_non_numeric_column_count": int(len(dropped_non_numeric_columns)),
        "missing_values_imputed": False,
        "global_scaling_performed": False,
        "model_training_performed": False,
        "hdc_training_performed": False,
        "recommended_main_dataset": "cleaned_multimodal_four_class_without_performance.csv",
        "auxiliary_dataset": "cleaned_multimodal_four_class_with_performance.csv",
        "subject_validation_readiness": {
            "subject_id_exists": "subject_id" in four_class.columns,
            "unique_subjects": unique_subjects,
            "groupkfold_by_subject_feasible": groupkfold_feasible,
            "recommended_number_of_splits": recommended_splits,
            "conclusion": groupkfold_conclusion,
            "samples_per_subject": samples_per_subject,
        },
    }
    write_json(results_dir / "leakage_check_report.json", leakage_check_report)
    write_json(results_dir / "dataset_construction_report.json", construction_report)

    log_text = f"""Phase 03 multimodal dataset labeling log
Created at: {construction_report['created_at']}
Input table: {feature_table_path}
Rows loaded: {phase02_table.shape[0]}
Rows retained for four-class labels: {len(four_class)}
Duplicate run_key rows removed: {duplicate_run_key_count}
Features without performance: {len(feature_columns_without_performance)}
Features with performance: {len(feature_columns_with_performance)}
High-missing threshold: 0.95
High-missing features removed: {len(removed_high_missing_columns)}
Dropped non-numeric non-identifier columns: {len(dropped_non_numeric_columns)}
Leakage conclusion: {leakage_check_report['conclusion']}
GroupKFold conclusion: {groupkfold_conclusion}
No imputation, scaling, ML training, or HDC training was performed.
"""
    (logs_dir / "phase_03_log.txt").write_text(log_text, encoding="utf-8")

    readme_text = f"""# Phase 03: Multimodal Dataset Construction and Four-Class Labeling

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
- Samples retained: {len(four_class)}
- Features without performance metrics: {len(feature_columns_without_performance)}
- Features with performance metrics: {len(feature_columns_with_performance)}
- High-missing features removed at 95% threshold: {len(removed_high_missing_columns)}
- Non-numeric non-identifier columns dropped from feature lists: {len(dropped_non_numeric_columns)}

## Subject-wise validation readiness
{groupkfold_conclusion}

## Recommended next phase
Phase 04 should train classical ML baselines using `cleaned_multimodal_four_class_without_performance.csv` as the main dataset. Any imputation, scaling, feature selection, and model fitting should happen inside cross-validation folds. The with-performance dataset should only be used as an auxiliary comparison to estimate shortcut-learning risk and upper-bound performance.
"""
    (phase_dir / "README.md").write_text(readme_text, encoding="utf-8")

    status_path = root / "EXPERIMENT_STATUS.md"
    if status_path.exists():
        status_text = status_path.read_text(encoding="utf-8")
        status_text = re.sub(r"- Current completed scope: .*", "- Current completed scope: Phase 03 multimodal dataset construction and four-class labeling", status_text, count=1)
        status_text = re.sub(
            r"\| 03 \| `phase_03_multimodal_dataset_labeling` \| .*? \| .*? \|",
            "| 03 | `phase_03_multimodal_dataset_labeling` | Complete | Four-class workload proxy datasets created with and without performance metrics; no model training performed. |",
            status_text,
            count=1,
        )
        status_text = re.sub(r"\n## Phase 03 Outputs\n.*?(?=\n## Phase 00-02 Verification Audit|\n## Latest Update|\Z)", "\n", status_text, flags=re.S)
        phase03_block = f"""
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
- Four-class samples retained: {len(four_class)}
- Features without performance metrics: {len(feature_columns_without_performance)}
- Features with performance metrics: {len(feature_columns_with_performance)}
- Recommended main dataset for Phase 04 and Phase 05: `cleaned_multimodal_four_class_without_performance.csv`
- Auxiliary upper-bound dataset: `cleaned_multimodal_four_class_with_performance.csv`
- {groupkfold_conclusion}
- No global imputation, global scaling, ML training, or HDC training was performed.
"""
        marker = "\n## Phase 00-02 Verification Audit"
        if marker in status_text:
            status_text = status_text.replace(marker, "\n" + phase03_block + marker, 1)
        else:
            status_text = status_text.rstrip() + "\n" + phase03_block
        latest_update = f"""
## Latest Update
- Phase 03 completed on {construction_report['created_at']}.
- Constructed leakage-controlled four-class workload proxy datasets from Phase 02 features.
- Primary Phase 04/05 input: `cleaned_multimodal_four_class_without_performance.csv`.
- Auxiliary upper-bound comparison input: `cleaned_multimodal_four_class_with_performance.csv`.
- {groupkfold_conclusion}
"""
        status_text = re.sub(r"\n## Latest Update\n.*\Z", "\n" + latest_update, status_text, flags=re.S)
        status_path.write_text(status_text, encoding="utf-8")

    return {
        "phase02_shape": phase02_table.shape,
        "phase02_head": phase02_table.head(),
        "identifier_check": identifier_check,
        "raw_difficulty_distribution": raw_difficulty_distribution,
        "numeric_difficulty_distribution_before": numeric_difficulty_distribution_before,
        "difficulty_distribution_after": difficulty_distribution_after,
        "target_mapping_table": pd.DataFrame([{"difficulty_level": k, "target": v} for k, v in target_mapping["difficulty_level_to_target"].items()]),
        "class_distribution": class_distribution,
        "subject_distribution": subject_distribution,
        "missing_value_summary": missing_value_summary.sort_values("missing_ratio", ascending=False),
        "missing_by_group": missing_by_group,
        "feature_count_by_group_without": feature_count_by_group_without,
        "feature_count_by_group_with": feature_count_by_group_with,
        "feature_count_without_performance": len(feature_columns_without_performance),
        "feature_count_with_performance": len(feature_columns_with_performance),
        "removed_features_high_missing": removed_features_high_missing,
        "dropped_non_numeric_columns": dropped_non_numeric_table,
        "leakage_check_report": leakage_check_report,
        "groupkfold_conclusion": groupkfold_conclusion,
        "dataset_version_comparison": dataset_version_comparison,
        "construction_report": construction_report,
        "phase_dir": str(phase_dir),
    }
