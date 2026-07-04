
"""Small HDC helpers for P1 demonstration notebooks.

The implementations are intentionally compact and transparent. They are for
P1 feasibility demonstration, not the final optimized HDC experiment suite.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

IDENTIFIER_COLUMNS = {"subject_id", "session_id", "run_id", "run_key", "difficulty_level", "target"}


def load_phase03_dataset(path: str | None = None) -> pd.DataFrame:
    if path is None:
        path = (
            "E:/hdc-vr-pilot/experiments/phase_03_multimodal_dataset_labeling/"
            "results/cleaned_multimodal_four_class_without_performance.csv"
        )
    return pd.read_csv(path)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in IDENTIFIER_COLUMNS and pd.api.types.is_numeric_dtype(df[c])]


def select_top_variance_features(X: pd.DataFrame, max_features: int = 300) -> list[str]:
    variances = X.var(axis=0, skipna=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return variances.sort_values(ascending=False).head(min(max_features, X.shape[1])).index.tolist()


def metric_row(model_name: str, fold: int, y_true: Iterable[int], y_pred: Iterable[int], seconds: float) -> dict[str, float | int | str]:
    return {
        "model": model_name,
        "fold": fold,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "fit_predict_seconds": float(seconds),
    }


def summarize_cv(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    metric_cols = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "fit_predict_seconds"]
    summary = df.groupby("model")[metric_cols].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join([str(x) for x in col if x]) for col in summary.columns.to_flat_index()]
    return summary.sort_values("macro_f1_mean", ascending=False)


def evaluate_sklearn_models(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, models: dict, n_splits: int = 5) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    rows: list[dict] = []
    confusion: dict[str, np.ndarray] = {name: np.zeros((4, 4), dtype=int) for name in models}
    splitter = GroupKFold(n_splits=n_splits)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        for name, model in models.items():
            estimator = clone(model)
            start = perf_counter()
            estimator.fit(X_train, y_train)
            pred = estimator.predict(X_test)
            seconds = perf_counter() - start
            rows.append(metric_row(name, fold, y_test, pred, seconds))
            confusion[name] += confusion_matrix(y_test, pred, labels=[0, 1, 2, 3])
    return pd.DataFrame(rows), summarize_cv(rows), confusion


@dataclass
class HDCConfig:
    dimensions: int = 1000
    n_levels: int = 21
    random_state: int = 42


class HDCEncoder:
    def __init__(self, n_features: int, config: HDCConfig):
        self.n_features = n_features
        self.config = config
        rng = np.random.default_rng(config.random_state)
        self.feature_hv = rng.choice(np.array([-1, 1], dtype=np.int8), size=(n_features, config.dimensions))
        self.level_hv = rng.choice(np.array([-1, 1], dtype=np.int8), size=(config.n_levels, config.dimensions))

    def _quantize(self, X: np.ndarray) -> np.ndarray:
        clipped = np.clip(X, -3.0, 3.0)
        bins = np.linspace(-3.0, 3.0, self.config.n_levels + 1)[1:-1]
        return np.digitize(clipped, bins).astype(np.int16)

    def encode(self, X: np.ndarray) -> np.ndarray:
        idx = self._quantize(X)
        encoded = np.empty((X.shape[0], self.config.dimensions), dtype=np.int8)
        for i in range(X.shape[0]):
            bound = self.feature_hv * self.level_hv[idx[i]]
            summed = bound.sum(axis=0)
            encoded[i] = np.where(summed >= 0, 1, -1).astype(np.int8)
        return encoded


class VanillaPrototypeHDC:
    def __init__(self, config: HDCConfig | None = None):
        self.config = config or HDCConfig()

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.encoder = HDCEncoder(X.shape[1], self.config)
        Z = self.encoder.encode(X)
        self.classes_ = np.array(sorted(np.unique(y)))
        self.prototypes_ = []
        for cls in self.classes_:
            proto = Z[y == cls].sum(axis=0)
            self.prototypes_.append(np.where(proto >= 0, 1.0, -1.0))
        self.prototypes_ = np.vstack(self.prototypes_)
        return self

    def predict_encoded(self, Z: np.ndarray) -> np.ndarray:
        scores = Z @ self.prototypes_.T / self.config.dimensions
        return self.classes_[scores.argmax(axis=1)]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_encoded(self.encoder.encode(X))


class OnlineHDStyleHDC(VanillaPrototypeHDC):
    def __init__(self, config: HDCConfig | None = None, epochs: int = 1, learning_rate: float = 1.0):
        super().__init__(config)
        self.epochs = epochs
        self.learning_rate = learning_rate

    def fit(self, X: np.ndarray, y: np.ndarray):
        super().fit(X, y)
        Z = self.encoder.encode(X).astype(float)
        class_to_idx = {cls: i for i, cls in enumerate(self.classes_)}
        for _ in range(self.epochs):
            for z, true in zip(Z, y):
                pred = self.predict_encoded(z.reshape(1, -1))[0]
                if pred != true:
                    self.prototypes_[class_to_idx[true]] += self.learning_rate * z
                    self.prototypes_[class_to_idx[pred]] -= self.learning_rate * z
        norms = np.linalg.norm(self.prototypes_, axis=1, keepdims=True) + 1e-9
        self.prototypes_ = self.prototypes_ / norms
        return self


class MultiCentroidHDC(VanillaPrototypeHDC):
    def __init__(self, config: HDCConfig | None = None, centroids_per_class: int = 2):
        super().__init__(config)
        self.centroids_per_class = centroids_per_class

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.encoder = HDCEncoder(X.shape[1], self.config)
        Z = self.encoder.encode(X).astype(float)
        labels = []
        centroids = []
        for cls in sorted(np.unique(y)):
            class_z = Z[y == cls]
            k = min(self.centroids_per_class, len(class_z))
            if k == 1:
                cls_centroids = class_z.mean(axis=0, keepdims=True)
            else:
                km = KMeans(n_clusters=k, random_state=self.config.random_state, n_init=10)
                km.fit(class_z)
                cls_centroids = km.cluster_centers_
            centroids.append(cls_centroids)
            labels.extend([cls] * cls_centroids.shape[0])
        self.centroid_labels_ = np.array(labels)
        self.centroids_ = np.vstack(centroids)
        norms = np.linalg.norm(self.centroids_, axis=1, keepdims=True) + 1e-9
        self.centroids_ = self.centroids_ / norms
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Z = self.encoder.encode(X).astype(float)
        Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
        scores = Z @ self.centroids_.T
        return self.centroid_labels_[scores.argmax(axis=1)]


def evaluate_hdc_models(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, models: dict, n_splits: int = 5) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    rows: list[dict] = []
    confusion: dict[str, np.ndarray] = {name: np.zeros((4, 4), dtype=int) for name in models}
    splitter = GroupKFold(n_splits=n_splits)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups), start=1):
        X_train_raw, X_test_raw = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        prep = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        X_train = prep.fit_transform(X_train_raw)
        X_test = prep.transform(X_test_raw)
        for name, model in models.items():
            estimator = clone_like_hdc(model)
            start = perf_counter()
            estimator.fit(X_train, y_train)
            pred = estimator.predict(X_test)
            seconds = perf_counter() - start
            rows.append(metric_row(name, fold, y_test, pred, seconds))
            confusion[name] += confusion_matrix(y_test, pred, labels=[0, 1, 2, 3])
    return pd.DataFrame(rows), summarize_cv(rows), confusion


def clone_like_hdc(model):
    if isinstance(model, OnlineHDStyleHDC):
        return OnlineHDStyleHDC(config=model.config, epochs=model.epochs, learning_rate=model.learning_rate)
    if isinstance(model, MultiCentroidHDC):
        return MultiCentroidHDC(config=model.config, centroids_per_class=model.centroids_per_class)
    if isinstance(model, VanillaPrototypeHDC):
        return VanillaPrototypeHDC(config=model.config)
    raise TypeError(f"Unsupported HDC model type: {type(model)!r}")
