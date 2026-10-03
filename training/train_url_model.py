"""Train the static raw-URL phishing classifier with domain-aware evaluation."""

from __future__ import annotations

import json
import sys
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
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.url_dataset_helper import engineer_url_dataset
from src.url_feature_extraction import URL_ONLY_FEATURE_ORDER

DATA_PATH = ROOT / "dataset" / "raw" / "PhiUSIIL_Phishing_URL_Dataset.csv"
MODEL_PATH = ROOT / "models" / "url_model.pkl"
FEATURES_PATH = ROOT / "models" / "url_features.json"
METRICS_PATH = ROOT / "results" / "url_model_metrics.json"


def train() -> dict[str, object]:
    """Extract URL features, split by domain, train, evaluate, and save the model."""
    raw = pd.read_csv(DATA_PATH, usecols=["URL", "label"])
    engineered, cleaning = engineer_url_dataset(raw)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_indices, test_indices = next(
        splitter.split(engineered, engineered["label"], groups=engineered["domain_group"])
    )
    X_train = engineered.iloc[train_indices][URL_ONLY_FEATURE_ORDER]
    X_test = engineered.iloc[test_indices][URL_ONLY_FEATURE_ORDER]
    # Source labels are 0=phishing and 1=legitimate; the application uses 1=phishing.
    y_train = (engineered.iloc[train_indices]["label"] == 0).astype(int)
    y_test = (engineered.iloc[test_indices]["label"] == 0).astype(int)
    model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
    metrics = {
        **cleaning,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "feature_count": len(URL_ONLY_FEATURE_ORDER),
        "domain_aware_split": True,
        "random_state": 42,
        "test_size": 0.30,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "label_mapping": {"0": "Legitimate", "1": "Phishing"},
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    FEATURES_PATH.write_text(json.dumps(URL_ONLY_FEATURE_ORDER, indent=2) + "\n", encoding="utf-8")
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


if __name__ == "__main__":
    result = train()
    for name in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        print(f"{name}: {result[name]:.4f}")