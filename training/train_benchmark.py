"""Train the benchmark phishing classifier without the dataset row identifier."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "dataset" / "raw" / "phishing.csv"
MODEL_PATH = ROOT / "models" / "benchmark_model.pkl"
FEATURES_PATH = ROOT / "models" / "benchmark_features.json"
METRICS_PATH = ROOT / "results" / "benchmark_model_metrics.json"

FEATURES = [
    "HTTPS", "AnchorURL", "WebsiteTraffic", "SubDomains", "PrefixSuffix-",
    "LinksInScriptTags", "ServerFormHandler", "RequestURL",
    "LinksPointingToPage", "DomainRegLen", "AgeofDomain", "GoogleIndex",
    "DNSRecording", "UsingIP", "PageRank", "LongURL", "HTTPSDomainURL",
    "InfoEmail", "ShortURL",
]


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Load the verified benchmark dataset and convert labels to one convention."""
    dataset = pd.read_csv(DATA_PATH)
    required = set(FEATURES) | {"Index", "class"}
    missing = required - set(dataset.columns)
    if missing:
        raise ValueError(f"Benchmark dataset is missing columns: {sorted(missing)}")
    if dataset["Index"].nunique() != len(dataset):
        raise ValueError("Index is expected to be a unique dataset row identifier.")
    if dataset[FEATURES].isna().any().any() or dataset["class"].isna().any():
        raise ValueError("Benchmark dataset contains missing values.")
    if dataset.duplicated().any():
        raise ValueError("Benchmark dataset contains duplicate rows.")
    labels = (dataset["class"] == -1).astype(int)
    if not labels.isin([0, 1]).all():
        raise ValueError("Benchmark labels must be -1 (phishing) or 1 (legitimate).")
    return dataset[FEATURES], labels


def train() -> dict[str, object]:
    """Train, evaluate, and save the benchmark Random Forest model."""
    features, labels = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.30,
        random_state=42,
        stratify=labels,
    )
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
    metrics = {
        "dataset_rows": len(features),
        "feature_count": len(FEATURES),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "random_state": 42,
        "test_size": 0.30,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "label_mapping": {"0": "Legitimate", "1": "Phishing"},
        "index_feature_removed": True,
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    FEATURES_PATH.write_text(json.dumps(FEATURES, indent=2) + "\n", encoding="utf-8")
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


if __name__ == "__main__":
    result = train()
    for name in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        print(f"{name}: {result[name]:.4f}")