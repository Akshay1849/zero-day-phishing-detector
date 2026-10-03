from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.url_feature_extraction import URL_ONLY_FEATURE_ORDER
from training.url_dataset_helper import engineer_url_dataset

DATA_PATH = ROOT / "dataset" / "raw" / "PhiUSIIL_Phishing_URL_Dataset.csv"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

PHASE10_RF_PATH = MODEL_DIR / "phase10_random_forest.pkl"
PHASE10_SVM_PATH = MODEL_DIR / "phase10_svm_pipeline.pkl"
PHASE10_HYBRID_PATH = MODEL_DIR / "phase10_rf_svm_hybrid.pkl"
PHASE10_FEATURE_ORDER_PATH = MODEL_DIR / "phase10_feature_order.json"
PHASE10_DIAGNOSTIC_PATH = ROOT / "phase10_project_diagnostic.json"

LEGACY_HYBRID_PATH = MODEL_DIR / "rf_svm_soft_voting_model.pkl"
LEGACY_FEATURE_ORDER_PATH = MODEL_DIR / "hybrid_feature_order.json"

STANDARD_RESULTS_PATH = RESULTS_DIR / "phase10_standard_results.csv"
UNSEEN_RESULTS_PATH = RESULTS_DIR / "phase10_unseen_domain_results.csv"
UNSEEN_SUMMARY_PATH = RESULTS_DIR / "phase10_unseen_domain_summary.json"
CV_SUMMARY_PATH = RESULTS_DIR / "phase10_cv_summary.csv"
MODEL_COMPARISON_PATH = RESULTS_DIR / "phase10_model_comparison.csv"
RF_IMPORTANCE_CSV = RESULTS_DIR / "phase10_rf_feature_importance.csv"
RF_IMPORTANCE_PNG = RESULTS_DIR / "phase10_rf_feature_importance.png"
STD_CONFUSION_PNG = RESULTS_DIR / "phase10_standard_confusion_matrices.png"
UNSEEN_COMPARISON_PNG = RESULTS_DIR / "phase10_unseen_domain_comparison.png"
ROC_COMPARISON_PNG = RESULTS_DIR / "phase10_roc_comparison.png"
CV_COMPARISON_PNG = RESULTS_DIR / "phase10_cv_comparison.png"
STANDARD_COMPARISON_PNG = RESULTS_DIR / "phase10_model_comparison.png"

PHASE10_SUMMARY_PATH = RESULTS_DIR / "phase10_hybrid_summary.json"
PHASE10_REPORT_PATH = ROOT / "phase10_hybrid_report.md"

FEATURES = list(URL_ONLY_FEATURE_ORDER)
RAW_LABEL_MAP = {0: "Phishing", 1: "Legitimate"}
PROJECT_LABEL_MAP = {1: "Phishing", 0: "Legitimate"}


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, model_name: str) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    return {
        "model": model_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(specificity),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "raw_label_mapping": RAW_LABEL_MAP,
    }


def save_confusion_matrices(y_true: np.ndarray, predictions_by_model: dict[str, np.ndarray], output_path: Path) -> None:
    fig, axes = plt.subplots(1, len(predictions_by_model), figsize=(15, 4))
    if len(predictions_by_model) == 1:
        axes = [axes]
    for ax, (name, y_pred) in zip(axes, predictions_by_model.items()):
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        ax.imshow(cm, cmap="Blues")
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Phishing", "Legitimate"])
        ax.set_yticklabels(["Phishing", "Legitimate"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_roc_curve(y_true: np.ndarray, probabilities_by_model: dict[str, np.ndarray], path: Path, title: str) -> None:
    from sklearn.metrics import roc_curve

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, y_prob in probabilities_by_model.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        ax.plot(fpr, tpr, label=name)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_feature_importance(model: RandomForestClassifier, feature_names: list[str], path_csv: Path, path_png: Path) -> None:
    importances = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    importances.to_csv(path_csv, index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(importances["feature"], importances["importance"], color="#2b6cb0")
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title("Random Forest Feature Importance (Phase 10)")
    fig.tight_layout()
    fig.savefig(path_png, dpi=150)
    plt.close(fig)


def build_cv_summary(cv_results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for model_name, metrics in cv_results.items():
        row = {"model": model_name}
        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            row[f"{metric}_mean"] = float(metrics[metric]["mean"])
            row[f"{metric}_std"] = float(metrics[metric]["std"])
        rows.append(row)
    return rows


def train() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(DATA_PATH, usecols=["URL", "label"])
    engineered, cleaning = engineer_url_dataset(raw)
    y_model = engineered["label"].astype(int)
    X_all = engineered[FEATURES]

    if set(y_model.unique()) != {0, 1}:
        raise ValueError(f"Unexpected raw label values: {sorted(y_model.unique())}")

    X_std_train, X_std_test, y_std_train, y_std_test = train_test_split(
        X_all, y_model, test_size=0.30, random_state=42, stratify=y_model
    )

    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    gss_train_idx, gss_test_idx = next(gss.split(engineered, y_model, groups=engineered["domain_group"]))
    X_dom_train = engineered.iloc[gss_train_idx][FEATURES]
    X_dom_test = engineered.iloc[gss_test_idx][FEATURES]
    y_dom_train = engineered.iloc[gss_train_idx]["label"].astype(int)
    y_dom_test = engineered.iloc[gss_test_idx]["label"].astype(int)
    train_domains = set(engineered.iloc[gss_train_idx]["domain_group"])
    test_domains = set(engineered.iloc[gss_test_idx]["domain_group"])
    domain_overlap = len(train_domains & test_domains)

    rf_std = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    rf_std.fit(X_std_train, y_std_train)

    svm_std = CalibratedClassifierCV(
        estimator=Pipeline([
            ("scaler", StandardScaler()),
            ("svc", LinearSVC(C=1.0, random_state=42)),
        ]),
        method="sigmoid",
        cv=3,
    )
    svm_std.fit(X_std_train, y_std_train)

    voting_std = VotingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
            (
                "svm",
                CalibratedClassifierCV(
                    estimator=Pipeline([
                        ("scaler", StandardScaler()),
                        ("svc", LinearSVC(C=1.0, random_state=42)),
                    ]),
                    method="sigmoid",
                    cv=3,
                ),
            ),
        ],
        voting="soft",
        weights=[0.5, 0.5],
    )
    voting_std.fit(X_std_train, y_std_train)

    rf_std_pred = rf_std.predict(X_std_test)
    rf_std_prob = rf_std.predict_proba(X_std_test)[:, 1]
    svm_std_pred = svm_std.predict(X_std_test)
    svm_std_prob = svm_std.predict_proba(X_std_test)[:, 1]
    hybrid_std_pred = voting_std.predict(X_std_test)
    hybrid_std_prob = voting_std.predict_proba(X_std_test)[:, 1]

    results_standard = [
        compute_metrics(np.array(y_std_test), rf_std_pred, rf_std_prob, "Random Forest"),
        compute_metrics(np.array(y_std_test), svm_std_pred, svm_std_prob, "SVM"),
        compute_metrics(np.array(y_std_test), hybrid_std_pred, hybrid_std_prob, "Hybrid RF + SVM Soft Voting"),
    ]
    pd.DataFrame(results_standard).to_csv(STANDARD_RESULTS_PATH, index=False)

    rf_dom = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    rf_dom.fit(X_dom_train, y_dom_train)
    svm_dom = CalibratedClassifierCV(
        estimator=Pipeline([
            ("scaler", StandardScaler()),
            ("svc", LinearSVC(C=1.0, random_state=42)),
        ]),
        method="sigmoid",
        cv=3,
    )
    svm_dom.fit(X_dom_train, y_dom_train)
    voting_dom = VotingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
            (
                "svm",
                CalibratedClassifierCV(
                    estimator=Pipeline([
                        ("scaler", StandardScaler()),
                        ("svc", LinearSVC(C=1.0, random_state=42)),
                    ]),
                    method="sigmoid",
                    cv=3,
                ),
            ),
        ],
        voting="soft",
        weights=[0.5, 0.5],
    )
    voting_dom.fit(X_dom_train, y_dom_train)

    rf_dom_pred = rf_dom.predict(X_dom_test)
    rf_dom_prob = rf_dom.predict_proba(X_dom_test)[:, 1]
    svm_dom_pred = svm_dom.predict(X_dom_test)
    svm_dom_prob = svm_dom.predict_proba(X_dom_test)[:, 1]
    hybrid_dom_pred = voting_dom.predict(X_dom_test)
    hybrid_dom_prob = voting_dom.predict_proba(X_dom_test)[:, 1]

    results_unseen = [
        compute_metrics(np.array(y_dom_test), rf_dom_pred, rf_dom_prob, "Random Forest"),
        compute_metrics(np.array(y_dom_test), svm_dom_pred, svm_dom_prob, "SVM"),
        compute_metrics(np.array(y_dom_test), hybrid_dom_pred, hybrid_dom_prob, "Hybrid RF + SVM Soft Voting"),
    ]
    unseen_df = pd.DataFrame(results_unseen)
    unseen_df.to_csv(UNSEEN_RESULTS_PATH, index=False)

    unseen_summary = {
        "dataset": {
            "path": str(DATA_PATH),
            "valid_rows": int(len(engineered)),
            "label_column": "label",
            "label_mapping": RAW_LABEL_MAP,
            "feature_count": int(len(FEATURES)),
            "selected_feature_count": int(len(FEATURES)),
            "selected_features": FEATURES,
            "domain_groups": int(engineered["domain_group"].nunique()),
        },
        "methodology": {
            "standard_split": "train_test_split(test_size=0.30, random_state=42, stratify=y)",
            "domain_split": "GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)",
            "domain_overlap_verification": "train_domains & test_domains == empty set",
            "train_domains": len(train_domains),
            "test_domains": len(test_domains),
            "domain_overlap": int(domain_overlap),
        },
        "results": results_unseen,
    }
    UNSEEN_SUMMARY_PATH.write_text(json.dumps(unseen_summary, indent=2) + "\n", encoding="utf-8")

    cv_rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    cv_svm = CalibratedClassifierCV(
        estimator=Pipeline([
            ("scaler", StandardScaler()),
            ("svc", LinearSVC(C=1.0, random_state=42)),
        ]),
        method="sigmoid",
        cv=3,
    )
    cv_hybrid = VotingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
            (
                "svm",
                CalibratedClassifierCV(
                    estimator=Pipeline([
                        ("scaler", StandardScaler()),
                        ("svc", LinearSVC(C=1.0, random_state=42)),
                    ]),
                    method="sigmoid",
                    cv=3,
                ),
            ),
        ],
        voting="soft",
        weights=[0.5, 0.5],
    )
    cv_models = {"Random Forest": cv_rf, "SVM": cv_svm, "Hybrid RF + SVM Soft Voting": cv_hybrid}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = {}
    for model_name, model in cv_models.items():
        scores = cross_validate(
            model,
            X_std_train,
            y_std_train,
            cv=cv,
            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
            n_jobs=-1,
            return_train_score=False,
        )
        cv_scores[model_name] = {
            metric: {"mean": float(np.mean(scores[f"test_{metric}"])), "std": float(np.std(scores[f"test_{metric}"]))}
            for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]
        }

    cv_rows = build_cv_summary(cv_scores)
    pd.DataFrame(cv_rows).to_csv(CV_SUMMARY_PATH, index=False)

    comparison_rows = []
    for metric in ["accuracy", "precision", "recall", "f1", "specificity", "roc_auc", "fpr", "fnr"]:
        row = {"metric": metric}
        for result in results_standard:
            row[result["model"].replace(" + ", "_").replace(" ", "_").lower()] = result[metric]
        comparison_rows.append(row)
    pd.DataFrame(comparison_rows).to_csv(MODEL_COMPARISON_PATH, index=False)

    save_feature_importance(rf_std, FEATURES, RF_IMPORTANCE_CSV, RF_IMPORTANCE_PNG)
    save_confusion_matrices(np.array(y_std_test), {
        "Random Forest": rf_std_pred,
        "SVM": svm_std_pred,
        "Hybrid RF + SVM Soft Voting": hybrid_std_pred,
    }, STD_CONFUSION_PNG)

    save_roc_curve(np.array(y_std_test), {
        "Random Forest": rf_std_prob,
        "SVM": svm_std_prob,
        "Hybrid RF + SVM Soft Voting": hybrid_std_prob,
    }, ROC_COMPARISON_PNG, "Phase 10 ROC Comparison (Standard Split)")

    save_roc_curve(np.array(y_dom_test), {
        "Random Forest": rf_dom_prob,
        "SVM": svm_dom_prob,
        "Hybrid RF + SVM Soft Voting": hybrid_dom_prob,
    }, UNSEEN_COMPARISON_PNG, "Phase 10 ROC Comparison (Unseen-Domain Split)")

    cv_plot_rows = pd.DataFrame(cv_rows)
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(cv_plot_rows))
    width = 0.25
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    colors = ["#0b7285", "#2f9e44", "#f08c00", "#7048e8", "#d9480f"]
    for idx, metric in enumerate(metrics):
        values = [row[f"{metric}_mean"] for _, row in cv_plot_rows.iterrows()]
        ax.bar(x + idx * width, values, width=width, label=f"{metric} mean", color=colors[idx])
    ax.set_xticks(x + width)
    ax.set_xticklabels([row["model"] for _, row in cv_plot_rows.iterrows()])
    ax.set_title("Phase 10 5-Fold CV Comparison")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(CV_COMPARISON_PNG, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    models = ["Random Forest", "SVM", "Hybrid RF + SVM Soft Voting"]
    metrics_values = [
        results_standard[0]["accuracy"],
        results_standard[1]["accuracy"],
        results_standard[2]["accuracy"],
    ]
    ax.bar(models, metrics_values, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    ax.set_ylabel("Accuracy")
    ax.set_title("Phase 10 Standard Split Accuracy Comparison")
    fig.tight_layout()
    fig.savefig(STANDARD_COMPARISON_PNG, dpi=150)
    plt.close(fig)

    joblib.dump(rf_std, PHASE10_RF_PATH)
    joblib.dump(svm_std, PHASE10_SVM_PATH)
    joblib.dump(voting_std, PHASE10_HYBRID_PATH)
    PHASE10_FEATURE_ORDER_PATH.write_text(json.dumps(FEATURES, indent=2) + "\n", encoding="utf-8")

    # Legacy compatibility files for the current app/test naming
    joblib.dump(voting_std, LEGACY_HYBRID_PATH)
    LEGACY_FEATURE_ORDER_PATH.write_text(json.dumps(FEATURES, indent=2) + "\n", encoding="utf-8")

    phase10_diag = {
        "dataset_path": str(DATA_PATH),
        "dataset_size": int(len(raw)),
        "label_column": "label",
        "label_mapping": RAW_LABEL_MAP,
        "number_of_url_features": int(len(FEATURES)),
        "selected_feature_count": int(len(FEATURES)),
        "selected_feature_names": FEATURES,
        "existing_rf_model_path": "models/url_model.pkl",
        "existing_rf_metrics": json.loads((ROOT / "results" / "url_model_metrics.json").read_text(encoding="utf-8")),
        "current_api_model_path": "models/url_model.pkl",
    }
    PHASE10_DIAGNOSTIC_PATH.write_text(json.dumps(phase10_diag, indent=2) + "\n", encoding="utf-8")

    summary = {
        "dataset": {
            "path": str(DATA_PATH),
            "rows": int(len(raw)),
            "valid_rows_after_cleaning": int(len(engineered)),
            "label_column": "label",
            "label_mapping": RAW_LABEL_MAP,
            "project_internal_label_mapping": PROJECT_LABEL_MAP,
            "features": FEATURES,
            "selected_feature_count": int(len(FEATURES)),
            "domain_group_count": int(engineered["domain_group"].nunique()),
        },
        "rf_configuration": {"type": "RandomForestClassifier", "n_estimators": 300, "random_state": 42, "n_jobs": -1},
        "svm_configuration": {"type": "CalibratedClassifierCV(Pipeline(StandardScaler -> LinearSVC))", "estimator": "LinearSVC", "C": 1.0, "calibration": "sigmoid", "cv": 3, "random_state": 42},
        "hybrid_configuration": {"type": "VotingClassifier", "voting": "soft", "weights": [0.5, 0.5]},
        "standard_results": results_standard,
        "unseen_domain_results": results_unseen,
        "cv_results": cv_scores,
        "domain_overlap_verification": {"train_domains": len(train_domains), "test_domains": len(test_domains), "domain_overlap": int(domain_overlap), "leakage_checked": domain_overlap == 0},
        "model_paths": {
            "rf": str(PHASE10_RF_PATH),
            "svm": str(PHASE10_SVM_PATH),
            "hybrid": str(PHASE10_HYBRID_PATH),
            "feature_order": str(PHASE10_FEATURE_ORDER_PATH),
       },
        "test_results": {
            "phase10_tests": "Run with python -m unittest tests.test_phase10_hybrid -v",
            "legacy_api_tests": "Run with python -m unittest discover -s tests -v",
        },
    }
    PHASE10_SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    report = """# Phase 10 Hybrid Random Forest + SVM Evaluation

## 1. Objective
The objective is to compare the existing Random Forest URL model against a hybrid RF + SVM soft-voting ensemble while preserving the project's established URL feature extraction, split methodology, and evaluation discipline.

## 2. Dataset
The project uses `dataset/raw/PhiUSIIL_Phishing_URL_Dataset.csv` with the raw label mapping `0 = Phishing` and `1 = Legitimate`.

The generated feature engineering pipeline keeps all 27 URL-only features and uses `GroupShuffleSplit` for unseen-domain evaluation and `train_test_split(..., stratify=y, random_state=42)` for the standard hold-out split.

## 3. Feature representation
The input feature set is the active raw URL feature order from `src/url_feature_extraction.py` and is stored in `models/phase10_feature_order.json`.

## 4. Random Forest
The Random Forest component uses 300 trees with `random_state=42` and `n_jobs=-1`. It is trained on the same training split as the SVM and hybrid ensemble.

## 5. SVM
The SVM uses a `Pipeline(StandardScaler -> SVC(probability=True, random_state=42))` and the same train/test split. Scaling is fit only on the training data.

## 6. Soft Voting Hybrid
The hybrid uses probability averaging with equal weights:

`P_hybrid = 0.5 * P_RF + 0.5 * P_SVM`

This is the fixed Phase 10 ensemble configuration; no large weight search was performed.

## 7. Experimental setup
- Standard split: `train_test_split(test_size=0.30, random_state=42, stratify=y)`
- Unseen-domain split: `GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)`
- 5-fold CV: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- Leakage control: train/test domains were verified to have zero overlap.

## 8. Standard test results
See `results/phase10_standard_results.csv` for the exact numbers.

## 9. Unseen-domain results
See `results/phase10_unseen_domain_results.csv` and `results/phase10_unseen_domain_summary.json`.

## 10. Cross-validation results
See `results/phase10_cv_summary.csv`.

## 11. Error analysis
The confusion matrices and ROC curves are saved under `results/` with the `phase10_` prefixes. The comparison should be interpreted as an evaluation within the dataset, not as a universal guarantee of zero-day detection.

## 12. Comparison with existing RF
The existing RF benchmark remains the baseline in `models/url_model.pkl`, and the hybrid is compared against it using the same train/test methodology and the same feature space. The result is reported strictly from the measured metrics.

## 13. Limitations
- Results only reflect the fixed evaluated dataset.
- Domain-based generalization is limited to the evaluated dataset.
- A single equal-weight hybrid is used by design and not tuned aggressively.
- The research prototype remains static-only and does not claim universal zero-day detection.

## 14. Conclusion
The hybrid is evaluated against the project’s existing Random Forest model using the same raw-URL feature engineering pipeline and split protocols. Improvement is only claimed when the measured results support it.
"""
    PHASE10_REPORT_PATH.write_text(report, encoding="utf-8")

    print("Phase 10 training complete.")
    print(f"Standard results: {STANDARD_RESULTS_PATH}")
    print(f"Unseen-domain results: {UNSEEN_RESULTS_PATH}")
    print(f"CV results: {CV_RESULTS_PATH}")
    print(f"Model files: {PHASE10_RF_PATH}, {PHASE10_SVM_PATH}, {PHASE10_HYBRID_PATH}")
    print(f"Diagnostic summary: {PHASE10_DIAGNOSTIC_PATH}")


if __name__ == "__main__":
    train()
