# Phase 10 Hybrid Random Forest + SVM Evaluation

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
