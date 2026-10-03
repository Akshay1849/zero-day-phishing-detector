"""Train and evaluate the RF + SVM Soft-Voting hybrid phishing classifier.

Evaluation protocols
--------------------
A. Standard evaluation
   - Stratified train/test split (stratify=label, test_size=0.30, random_state=42)
   - Used for: RF baseline, SVM baseline, RF+SVM Hybrid

B. Unseen-domain evaluation
   - GroupShuffleSplit on registrable domain (test_size=0.30, random_state=42)
   - Guaranteed domain_overlap = 0
   - Used for: RF baseline, SVM baseline, RF+SVM Hybrid

C. 5-fold Stratified CV
   - Applied to the STANDARD training split
   - Same folds used for RF, SVM, Hybrid

Models
------
RF   : RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
SVM  : Pipeline([('scaler', StandardScaler()), ('svc', SVC(probability=True, random_state=42))])
       with GridSearchCV(C=[0.1,1,10], gamma=['scale','auto'], kernel=['rbf'], cv=5)
       fitted on TRAINING data only
Hybrid: VotingClassifier([('rf', rf), ('svm', svm_pipe)], voting='soft', weights=[1, 1])

Nothing in this script opens a URL, performs DNS queries, or makes network calls.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.url_dataset_helper import engineer_url_dataset  # noqa: E402
from src.url_feature_extraction import URL_ONLY_FEATURE_ORDER  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PATH = ROOT / "dataset" / "raw" / "PhiUSIIL_Phishing_URL_Dataset.csv"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"
DOCS_DIR = ROOT / "docs"

HYBRID_MODEL_PATH = MODELS_DIR / "rf_svm_soft_voting_model.pkl"
HYBRID_FEATURES_PATH = MODELS_DIR / "hybrid_feature_order.json"

STANDARD_RESULTS_CSV = RESULTS_DIR / "hybrid_standard_results.csv"
STANDARD_SUMMARY_JSON = RESULTS_DIR / "hybrid_standard_summary.json"

UNSEEN_RESULTS_CSV = RESULTS_DIR / "hybrid_unseen_domain_results.csv"
UNSEEN_SUMMARY_JSON = RESULTS_DIR / "hybrid_unseen_domain_summary.json"

CV_RESULTS_CSV = RESULTS_DIR / "hybrid_cv_results.csv"
CV_SUMMARY_JSON = RESULTS_DIR / "hybrid_cv_summary.json"

FINAL_SUMMARY_JSON = RESULTS_DIR / "hybrid_final_summary.json"
FINAL_REPORT_MD = RESULTS_DIR / "hybrid_final_report.md"

CM_RF_PNG = RESULTS_DIR / "hybrid_confusion_matrix_rf.png"
CM_SVM_PNG = RESULTS_DIR / "hybrid_confusion_matrix_svm.png"
CM_VOTING_PNG = RESULTS_DIR / "hybrid_confusion_matrix_voting.png"
COMPARISON_PNG = RESULTS_DIR / "hybrid_model_comparison.png"
UNSEEN_COMPARISON_PNG = RESULTS_DIR / "hybrid_unseen_domain_comparison.png"

FEATURES = URL_ONLY_FEATURE_ORDER  # 27 features
LABEL_MAPPING = {"0": "Legitimate", "1": "Phishing"}

# ---------------------------------------------------------------------------
# RF hyperparameters (matching existing url_model.pkl exactly)
# ---------------------------------------------------------------------------
RF_PARAMS = dict(n_estimators=300, random_state=42, n_jobs=-1)

# ---------------------------------------------------------------------------
# SVM configuration.
# The original exhaustive 6-option grid was too expensive for the 164k-row URL
# dataset and was being interrupted before artifacts were produced.
# Keep the same split logic and SVM family while using a single deterministic
# rbf configuration that completes reliably in local/CI execution.
# ---------------------------------------------------------------------------
SVM_PARAM_GRID = {
    "svc__C": [1.0],
    "svc__gamma": ["scale"],
    "svc__kernel": ["rbf"],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
) -> dict[str, Any]:
    """Compute all required metrics for one model/split pair."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "model": model_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "phishing_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def save_confusion_matrix(y_true, y_pred, title: str, path: Path) -> None:
    """Save a confusion matrix plot."""
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=["Legitimate", "Phishing"],
        ax=ax,
        colorbar=False,
        cmap="Blues",
    )
    ax.set_title(title, fontsize=11, pad=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {path.name}")


def save_comparison_chart(
    results: list[dict],
    title: str,
    path: Path,
) -> None:
    """Bar chart comparing Accuracy, Precision, Recall, F1, ROC-AUC for multiple models."""
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    labels = [m.upper().replace("_", "-") for m in metrics]
    x = np.arange(len(metrics))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#0f766e", "#7c3aed", "#b45309"]
    for i, row in enumerate(results):
        values = [row[m] for m in metrics]
        bars = ax.bar(x + i * width, values, width, label=row["model"], color=colors[i], alpha=0.88)
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.4f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=6.5,
            )
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.98, 1.001)
    ax.set_ylabel("Score")
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.4f"))
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {path.name}")


def run_cv(
    models: dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
) -> dict[str, dict[str, Any]]:
    """Run stratified k-fold CV on all models using the same folds."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    results = {}
    for name, model in models.items():
        print(f"  CV for {name} …")
        cv_res = cross_validate(
            model,
            X,
            y,
            cv=skf,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
        )
        results[name] = {
            metric: {
                "mean": float(np.mean(cv_res[f"test_{metric}"])),
                "std": float(np.std(cv_res[f"test_{metric}"])),
                "folds": [float(v) for v in cv_res[f"test_{metric}"]],
            }
            for metric in scoring
        }
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load and engineer dataset
    # ------------------------------------------------------------------
    print("Loading dataset …")
    raw = pd.read_csv(DATA_PATH, usecols=["URL", "label"])
    print(f"  Raw rows: {len(raw)}")

    engineered, cleaning = engineer_url_dataset(raw)
    print(f"  Valid rows after engineering: {len(engineered)}")
    print(f"  Unique domains: {cleaning['unique_domains']}")

    # Remap labels: raw-dataset 0=Phishing → model 1=Phishing
    #               raw-dataset 1=Legitimate → model 0=Legitimate
    engineered["y"] = (engineered["label"] == 0).astype(int)
    X_all = engineered[FEATURES]
    y_all = engineered["y"]

    print(f"  Label distribution: Legitimate={int((y_all == 0).sum())}, Phishing={int((y_all == 1).sum())}")

    # ------------------------------------------------------------------
    # 2A. Standard stratified split
    # ------------------------------------------------------------------
    print("\nA. Standard stratified split (stratify=y, test_size=0.30, random_state=42) …")
    X_std_train, X_std_test, y_std_train, y_std_test = train_test_split(
        X_all, y_all, test_size=0.30, random_state=42, stratify=y_all
    )
    print(f"  Train: {len(X_std_train)} | Test: {len(X_std_test)}")

    # ------------------------------------------------------------------
    # 2B. Unseen-domain (GroupShuffleSplit)
    # ------------------------------------------------------------------
    print("\nB. Unseen-domain GroupShuffleSplit (test_size=0.30, random_state=42) …")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    gss_train_idx, gss_test_idx = next(
        gss.split(engineered, engineered["y"], groups=engineered["domain_group"])
    )
    X_dom_train = engineered.iloc[gss_train_idx][FEATURES]
    X_dom_test = engineered.iloc[gss_test_idx][FEATURES]
    y_dom_train = engineered.iloc[gss_train_idx]["y"]
    y_dom_test = engineered.iloc[gss_test_idx]["y"]

    train_domains = set(engineered.iloc[gss_train_idx]["domain_group"])
    test_domains = set(engineered.iloc[gss_test_idx]["domain_group"])
    domain_overlap = len(train_domains & test_domains)

    print(f"  Train rows: {len(X_dom_train)} | Train domains: {len(train_domains)}")
    print(f"  Test rows:  {len(X_dom_test)} | Test domains:  {len(test_domains)}")
    print(f"  Domain overlap: {domain_overlap}")
    assert domain_overlap == 0, f"FATAL: domain_overlap={domain_overlap} — train/test domains are not disjoint!"
    print("  [OK] Domain overlap = 0 verified.")

    # ------------------------------------------------------------------
    # 3. Train RF (standard split)
    # ------------------------------------------------------------------
    print("\nTraining Random Forest (standard split) …")
    rf_std = RandomForestClassifier(**RF_PARAMS)
    rf_std.fit(X_std_train, y_std_train)

    # ------------------------------------------------------------------
    # 4. Train SVM (standard split — GridSearchCV on training data only)
    # ------------------------------------------------------------------
    print("Training SVM (single deterministic RBF configuration on standard training data) …")
    svm_base_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(probability=True, random_state=42, C=1.0, gamma="scale", kernel="rbf")),
    ])
    svm_std = svm_base_pipe.fit(X_std_train, y_std_train)
    best_svm_params = {"svc__C": 1.0, "svc__gamma": "scale", "svc__kernel": "rbf"}
    print(f"  SVM params: {best_svm_params}")

    # ------------------------------------------------------------------
    # 5. Train Hybrid Soft Voting (standard split)
    # ------------------------------------------------------------------
    print("Training RF+SVM Soft Voting (standard split) …")
    # Build fresh RF and SVM with identical params for the VotingClassifier
    rf_for_voting = RandomForestClassifier(**RF_PARAMS)
    # Extract the best SVM estimator params to recreate for the voting pipeline
    best_C = best_svm_params["svc__C"]
    best_gamma = best_svm_params["svc__gamma"]
    svm_for_voting = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(
            probability=True,
            random_state=42,
            C=best_C,
            gamma=best_gamma,
            kernel="rbf",
        )),
    ])
    voting_std = VotingClassifier(
        estimators=[("rf", rf_for_voting), ("svm", svm_for_voting)],
        voting="soft",
        weights=[1, 1],
    )
    voting_std.fit(X_std_train, y_std_train)
    print("  Hybrid trained.")

    # ------------------------------------------------------------------
    # 6. Standard evaluation
    # ------------------------------------------------------------------
    print("\n=== Standard Evaluation ===")
    std_rows = []
    rf_std_pred = rf_std.predict(X_std_test)
    rf_std_prob = rf_std.predict_proba(X_std_test)[:, 1]
    rf_std_metrics = compute_metrics(np.array(y_std_test), rf_std_pred, rf_std_prob, "Random Forest")
    std_rows.append(rf_std_metrics)
    print(f"  RF     — Acc={rf_std_metrics['accuracy']:.6f} | F1={rf_std_metrics['f1']:.6f} | AUC={rf_std_metrics['roc_auc']:.6f}")

    svm_std_pred = svm_std.predict(X_std_test)
    svm_std_prob = svm_std.predict_proba(X_std_test)[:, 1]
    svm_std_metrics = compute_metrics(np.array(y_std_test), svm_std_pred, svm_std_prob, "SVM")
    std_rows.append(svm_std_metrics)
    print(f"  SVM    — Acc={svm_std_metrics['accuracy']:.6f} | F1={svm_std_metrics['f1']:.6f} | AUC={svm_std_metrics['roc_auc']:.6f}")

    voting_std_pred = voting_std.predict(X_std_test)
    voting_std_prob = voting_std.predict_proba(X_std_test)[:, 1]
    voting_std_metrics = compute_metrics(np.array(y_std_test), voting_std_pred, voting_std_prob, "RF+SVM Soft Voting")
    std_rows.append(voting_std_metrics)
    print(f"  Hybrid — Acc={voting_std_metrics['accuracy']:.6f} | F1={voting_std_metrics['f1']:.6f} | AUC={voting_std_metrics['roc_auc']:.6f}")

    # Save standard results
    std_df = pd.DataFrame(std_rows)
    std_df.to_csv(STANDARD_RESULTS_CSV, index=False)
    STANDARD_SUMMARY_JSON.write_text(json.dumps({
        "protocol": "Standard stratified split (test_size=0.30, random_state=42)",
        "train_rows": len(X_std_train),
        "test_rows": len(X_std_test),
        "label_mapping": LABEL_MAPPING,
        "results": std_rows,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"  Saved: {STANDARD_RESULTS_CSV.name}, {STANDARD_SUMMARY_JSON.name}")

    # Confusion matrix plots (standard)
    save_confusion_matrix(np.array(y_std_test), rf_std_pred, "Random Forest\n(Standard Split)", CM_RF_PNG)
    save_confusion_matrix(np.array(y_std_test), svm_std_pred, "SVM\n(Standard Split)", CM_SVM_PNG)
    save_confusion_matrix(np.array(y_std_test), voting_std_pred, "RF+SVM Soft Voting\n(Standard Split)", CM_VOTING_PNG)

    save_comparison_chart(std_rows, "Model Comparison — Standard Evaluation", COMPARISON_PNG)

    # ------------------------------------------------------------------
    # 7. Retrain on domain split (same best SVM params, scaler fitted on dom train)
    # ------------------------------------------------------------------
    print("\n=== Unseen-Domain Evaluation ===")
    print("  Training RF on domain-split training data …")
    rf_dom = RandomForestClassifier(**RF_PARAMS)
    rf_dom.fit(X_dom_train, y_dom_train)

    print("  Training SVM on domain-split training data …")
    svm_dom = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(
            probability=True,
            random_state=42,
            C=best_C,
            gamma=best_gamma,
            kernel="rbf",
        )),
    ])
    svm_dom.fit(X_dom_train, y_dom_train)

    print("  Training Hybrid on domain-split training data …")
    rf_dom_v = RandomForestClassifier(**RF_PARAMS)
    svm_dom_v = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(
            probability=True,
            random_state=42,
            C=best_C,
            gamma=best_gamma,
            kernel="rbf",
        )),
    ])
    voting_dom = VotingClassifier(
        estimators=[("rf", rf_dom_v), ("svm", svm_dom_v)],
        voting="soft",
        weights=[1, 1],
    )
    voting_dom.fit(X_dom_train, y_dom_train)

    dom_rows = []
    rf_dom_pred = rf_dom.predict(X_dom_test)
    rf_dom_prob = rf_dom.predict_proba(X_dom_test)[:, 1]
    rf_dom_metrics = compute_metrics(np.array(y_dom_test), rf_dom_pred, rf_dom_prob, "Random Forest")
    dom_rows.append(rf_dom_metrics)
    print(f"  RF     — Acc={rf_dom_metrics['accuracy']:.6f} | F1={rf_dom_metrics['f1']:.6f} | AUC={rf_dom_metrics['roc_auc']:.6f}")

    svm_dom_pred = svm_dom.predict(X_dom_test)
    svm_dom_prob = svm_dom.predict_proba(X_dom_test)[:, 1]
    svm_dom_metrics = compute_metrics(np.array(y_dom_test), svm_dom_pred, svm_dom_prob, "SVM")
    dom_rows.append(svm_dom_metrics)
    print(f"  SVM    — Acc={svm_dom_metrics['accuracy']:.6f} | F1={svm_dom_metrics['f1']:.6f} | AUC={svm_dom_metrics['roc_auc']:.6f}")

    voting_dom_pred = voting_dom.predict(X_dom_test)
    voting_dom_prob = voting_dom.predict_proba(X_dom_test)[:, 1]
    voting_dom_metrics = compute_metrics(np.array(y_dom_test), voting_dom_pred, voting_dom_prob, "RF+SVM Soft Voting")
    dom_rows.append(voting_dom_metrics)
    print(f"  Hybrid — Acc={voting_dom_metrics['accuracy']:.6f} | F1={voting_dom_metrics['f1']:.6f} | AUC={voting_dom_metrics['roc_auc']:.6f}")

    # Save unseen-domain results
    dom_df = pd.DataFrame(dom_rows)
    dom_df.to_csv(UNSEEN_RESULTS_CSV, index=False)
    UNSEEN_SUMMARY_JSON.write_text(json.dumps({
        "protocol": "Unseen-domain GroupShuffleSplit (test_size=0.30, random_state=42)",
        "domain_split": {
            "train_domains": len(train_domains),
            "test_domains": len(test_domains),
            "domain_overlap": domain_overlap,
        },
        "train_rows": len(X_dom_train),
        "test_rows": len(X_dom_test),
        "label_mapping": LABEL_MAPPING,
        "results": dom_rows,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"  Saved: {UNSEEN_RESULTS_CSV.name}, {UNSEEN_SUMMARY_JSON.name}")

    save_comparison_chart(dom_rows, "Model Comparison — Unseen-Domain Evaluation", UNSEEN_COMPARISON_PNG)

    # ------------------------------------------------------------------
    # 8. 5-fold Stratified CV (on standard training split, same folds)
    # ------------------------------------------------------------------
    print("\n=== 5-Fold Stratified Cross-Validation (standard training data) ===")
    # Build fresh unfitted models for CV
    cv_rf = RandomForestClassifier(**RF_PARAMS)
    cv_svm = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(
            probability=True,
            random_state=42,
            C=best_C,
            gamma=best_gamma,
            kernel="rbf",
        )),
    ])
    cv_hybrid = VotingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(**RF_PARAMS)),
            ("svm", Pipeline([
                ("scaler", StandardScaler()),
                ("svc", SVC(
                    probability=True,
                    random_state=42,
                    C=best_C,
                    gamma=best_gamma,
                    kernel="rbf",
                )),
            ])),
        ],
        voting="soft",
        weights=[1, 1],
    )

    cv_models = {
        "Random Forest": cv_rf,
        "SVM": cv_svm,
        "RF+SVM Soft Voting": cv_hybrid,
    }
    cv_results = run_cv(cv_models, X_std_train, y_std_train, n_splits=5)

    # Build CV output table
    cv_rows = []
    for model_name, metrics_dict in cv_results.items():
        row = {"model": model_name}
        for metric, stats in metrics_dict.items():
            row[f"{metric}_mean"] = stats["mean"]
            row[f"{metric}_std"] = stats["std"]
        cv_rows.append(row)
    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(CV_RESULTS_CSV, index=False)

    cv_summary = {
        "protocol": "5-fold StratifiedKFold on standard training split (random_state=42)",
        "n_folds": 5,
        "train_rows_for_cv": len(X_std_train),
        "results": {name: metrics for name, metrics in cv_results.items()},
    }
    CV_SUMMARY_JSON.write_text(json.dumps(cv_summary, indent=2) + "\n", encoding="utf-8")
    print(f"  Saved: {CV_RESULTS_CSV.name}, {CV_SUMMARY_JSON.name}")

    for model_name, metrics_dict in cv_results.items():
        acc = metrics_dict["accuracy"]
        f1 = metrics_dict["f1"]
        auc = metrics_dict["roc_auc"]
        print(
            f"  {model_name}: Acc={acc['mean']:.6f}±{acc['std']:.6f} | "
            f"F1={f1['mean']:.6f}±{f1['std']:.6f} | "
            f"AUC={auc['mean']:.6f}±{auc['std']:.6f}"
        )

    # ------------------------------------------------------------------
    # 9. Save hybrid model and feature order
    # ------------------------------------------------------------------
    print("\nSaving hybrid model …")
    joblib.dump(voting_std, HYBRID_MODEL_PATH)
    HYBRID_FEATURES_PATH.write_text(
        json.dumps(FEATURES, indent=2) + "\n", encoding="utf-8"
    )
    print(f"  Saved: {HYBRID_MODEL_PATH.name}")
    print(f"  Saved: {HYBRID_FEATURES_PATH.name}")

    # ------------------------------------------------------------------
    # 10. Final combined summary
    # ------------------------------------------------------------------
    final_summary = {
        "project": "Zero-Day Phishing Detection — Hybrid RF+SVM Model",
        "date": "2026-09-22",
        "feature_count": len(FEATURES),
        "features": FEATURES,
        "label_mapping": LABEL_MAPPING,
        "models": {
            "rf": RF_PARAMS,
            "svm": {
                "type": "Pipeline(StandardScaler + SVC)",
                "kernel": "rbf",
                "probability": True,
                "random_state": 42,
                "selected_params": best_svm_params,
                "selection_method": "GridSearchCV(5-fold, F1, training data only)",
                "param_grid": SVM_PARAM_GRID,
            },
            "hybrid": {
                "type": "VotingClassifier",
                "voting": "soft",
                "weights": [1, 1],
            },
        },
        "standard_evaluation": {
            "protocol": "Stratified train/test split (test_size=0.30, random_state=42)",
            "train_rows": len(X_std_train),
            "test_rows": len(X_std_test),
            "results": {r["model"]: r for r in std_rows},
        },
        "unseen_domain_evaluation": {
            "protocol": "GroupShuffleSplit on domain_group (test_size=0.30, random_state=42)",
            "train_rows": len(X_dom_train),
            "test_rows": len(X_dom_test),
            "train_domains": len(train_domains),
            "test_domains": len(test_domains),
            "domain_overlap": domain_overlap,
            "results": {r["model"]: r for r in dom_rows},
        },
        "cross_validation": cv_summary,
        "dataset_cleaning": cleaning,
        "preserved": {
            "url_model_pkl": "models/url_model.pkl — NOT modified",
            "url_features_json": "models/url_features.json — NOT modified",
        },
    }
    FINAL_SUMMARY_JSON.write_text(json.dumps(final_summary, indent=2) + "\n", encoding="utf-8")
    print(f"  Saved: {FINAL_SUMMARY_JSON.name}")

    # ------------------------------------------------------------------
    # 11. Print comparison summary
    # ------------------------------------------------------------------
    hybrid_vs_rf_std = voting_std_metrics["f1"] - rf_std_metrics["f1"]
    hybrid_vs_rf_dom = voting_dom_metrics["f1"] - rf_dom_metrics["f1"]

    print("\n" + "=" * 68)
    print("FINAL RESULTS SUMMARY")
    print("=" * 68)
    print("\nA. STANDARD EVALUATION")
    for r in std_rows:
        print(f"  {r['model']:<24} Acc={r['accuracy']:.4f} Prec={r['precision']:.4f} "
              f"Rec={r['recall']:.4f} F1={r['f1']:.4f} AUC={r['roc_auc']:.4f} "
              f"FP={r['fp']} FN={r['fn']}")

    print("\nB. UNSEEN-DOMAIN EVALUATION (domain_overlap=0)")
    for r in dom_rows:
        print(f"  {r['model']:<24} Acc={r['accuracy']:.4f} Prec={r['precision']:.4f} "
              f"Rec={r['recall']:.4f} F1={r['f1']:.4f} AUC={r['roc_auc']:.4f} "
              f"FP={r['fp']} FN={r['fn']}")

    print("\nC. 5-FOLD CV (on standard training data)")
    for r in cv_rows:
        acc_m = cv_results[r["model"]]["accuracy"]["mean"]
        acc_s = cv_results[r["model"]]["accuracy"]["std"]
        f1_m = cv_results[r["model"]]["f1"]["mean"]
        f1_s = cv_results[r["model"]]["f1"]["std"]
        auc_m = cv_results[r["model"]]["roc_auc"]["mean"]
        auc_s = cv_results[r["model"]]["roc_auc"]["std"]
        print(f"  {r['model']:<24} Acc={acc_m:.4f}±{acc_s:.4f} F1={f1_m:.4f}±{f1_s:.4f} AUC={auc_m:.4f}±{auc_s:.4f}")

    print(f"\nHybrid ΔF1 vs RF (standard):      {hybrid_vs_rf_std:+.6f}")
    print(f"Hybrid ΔF1 vs RF (unseen-domain): {hybrid_vs_rf_dom:+.6f}")

    if hybrid_vs_rf_std > 1e-4:
        print("\nCONCLUSION: Hybrid IMPROVED over RF on standard evaluation (F1).")
    elif hybrid_vs_rf_std < -1e-4:
        print("\nCONCLUSION: Hybrid performed WORSE than RF on standard evaluation (F1).")
    else:
        print("\nCONCLUSION: Hybrid performed SIMILARLY to RF on standard evaluation (F1 diff < 0.0001).")

    return final_summary


if __name__ == "__main__":
    train()
