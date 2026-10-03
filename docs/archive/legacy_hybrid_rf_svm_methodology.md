# Hybrid RF + SVM Soft-Voting Phishing Classifier

**Project:** Zero-Day Phishing Detection Using Machine Learning-Based Feature Analysis  
**Date:** 2026-09-22  
**Status:** Experimental — see Section 10 for honest comparison with the existing RF baseline.

---

## 1. Why the Hybrid Model Was Introduced

The existing project already has a strong Random Forest classifier (27 URL-only features, n_estimators=300) that achieves very high accuracy on both standard and domain-disjoint evaluations. However, Random Forest and SVM have fundamentally different learning mechanisms:

- **Random Forest** partitions the feature space into regions through an ensemble of decision trees. It handles nonlinear, high-dimensional interactions well and is robust to irrelevant features.
- **SVM** finds a maximum-margin separating hyperplane in feature space (or in a kernel-induced space). With an RBF kernel, SVM learns nonlinear boundaries using the geometry of the training data.

When two models make errors on _different_ examples, combining their probability estimates through soft voting can produce a more stable and sometimes more accurate classifier. The hybrid model is therefore introduced as an experimental extension to evaluate whether combining these two learning paradigms provides measurable improvement, at the same performance, or performs worse than the RF baseline alone.

**No improvement is assumed in advance. The conclusion is drawn from experimental results only.**

---

## 2. Random Forest Component

| Parameter | Value |
|---|---|
| `n_estimators` | 300 |
| `random_state` | 42 |
| `max_depth` | None (unlimited) |
| `class_weight` | None |
| `n_jobs` | -1 |

The RF in the hybrid uses **identical hyperparameters** to the existing `url_model.pkl` baseline. This ensures that the RF's contribution to the ensemble is the same model that has already been validated independently.

**Note:** The existing `models/url_model.pkl` is NOT modified. The RF inside the VotingClassifier is trained fresh on the same training data with the same configuration.

---

## 3. SVM Component

The SVM is wrapped in a scikit-learn `Pipeline`:

```
StandardScaler → SVC(probability=True, kernel='rbf', random_state=42)
```

### Why Scaling Is Needed

SVM decision boundaries are sensitive to feature scale. The 27 URL features include:
- Raw character counts (url_length up to ~500, letter_count similar)
- Binary flags (uses_https ∈ {0, 1})
- Entropy values (0 to ~5)
- Ratios (0 to 1)

Without scaling, features with large ranges would dominate the kernel computation. `StandardScaler` normalises each feature to zero mean and unit variance on the **training data only**. The scaler is fitted on training data and applied to test data — no information leakage.

### Hyperparameter Selection

A small, reproducible grid search is performed on the **training data only** using 5-fold cross-validation:

```python
param_grid = {
    "svc__C": [0.1, 1, 10],
    "svc__gamma": ["scale", "auto"],
    "svc__kernel": ["rbf"],
}
GridSearchCV(cv=5, scoring="f1", n_jobs=-1)
```

The best parameters selected are recorded in `results/hybrid_final_summary.json`.

### Scaler Saved Inside the Pipeline

The `StandardScaler` is saved inside the `Pipeline` object, which is saved inside the `VotingClassifier` as `models/rf_svm_soft_voting_model.pkl`. There is no standalone scaler file. This prevents any scaler-model mismatch at inference time.

---

## 4. Soft Voting

### Architecture

```
Raw URL
   ↓
Existing URL feature extraction (src/url_feature_extraction.py)
   ↓
27 URL-only features
   ↓
       ┌──────────────────────┐
       │                      │
       ↓                      ↓
  Random Forest          SVM Pipeline
  (n_estimators=300)     (StandardScaler → SVC)
       ↓                      ↓
  P(Phishing)           P(Phishing)
       └──────────┬───────────┘
                  ↓
           Soft Voting
           (weights=[1, 1])
           average of probabilities
                  ↓
         Final: Phishing / Legitimate
```

### Why Soft Voting (Not Hard Voting)

Hard voting counts majority class labels — it discards confidence information. Soft voting averages the predicted class probabilities:

$$P_{ensemble}(y=1) = \frac{P_{RF}(y=1) + P_{SVM}(y=1)}{2}$$

This is more informative because a model that is 99% confident should outweigh one that is 51% confident. Soft voting requires `predict_proba()` from both models. SVC with `probability=True` uses Platt scaling to produce calibrated probabilities.

### Weights

Equal weights `[1, 1]` are used. Weight optimisation is not performed as there is no a priori reason to prefer one model's probabilities over the other's.

---

## 5. Training Procedure

### Evaluation Protocol A — Standard Stratified Split

```python
train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
```

- All three models (RF, SVM, Hybrid) use **exactly the same** training and test sets.
- SVM GridSearchCV is performed on the training set only.
- StandardScaler is fitted on the training set and applied to the test set.

### Evaluation Protocol B — Unseen-Domain Split

```python
GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
# groups = registrable domain (e.g., "example.com")
```

- Domain groups are extracted by `training/url_dataset_helper.py::registrable_domain()`.
- A domain is entirely in train or entirely in test — never split across both sets.
- `domain_overlap = 0` is verified programmatically with `assert`.
- SVM and Hybrid are retrained on the domain-split training set (with scaler fitted on that set).
- Best SVM parameters from the standard-split grid search are reused for the domain split.

### 5-fold Stratified CV

Applied to the **standard training split** only. Same `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` used for all three models.

---

## 6. Feature Selection

The hybrid uses **all 27 URL-only features** extracted by `src/url_feature_extraction.py`:

| Category | Features |
|---|---|
| Length | url_length, hostname_length, path_length, query_length, fragment_length |
| Counts | dot_count, hyphen_count, underscore_count, digit_count, special_character_count, slash_count, query_parameter_count, letter_count, numeric_count |
| Structural | url_depth, subdomain_count, uses_https, uses_ip_address, contains_at_symbol, looks_like_shortener |
| Ratios | digit_ratio, special_character_ratio |
| Entropy | url_entropy, hostname_entropy, path_entropy |
| Encoded | encoded_character_count, suspicious_character_count |

This matches the 27 features used by the existing `url_model.pkl`.

**No feature selection is applied.** Using the same feature set for RF and SVM ensures a scientifically fair comparison: any performance difference is attributable to the model, not the feature set.

---

## 7. Standard Evaluation

All metrics are computed on the held-out test set using scikit-learn:

- **Accuracy** = (TP + TN) / total
- **Precision** = TP / (TP + FP) — of all predicted phishing, how many actually are
- **Recall** = TP / (TP + FN) — of all actual phishing, how many were caught
- **F1** = harmonic mean of Precision and Recall
- **ROC-AUC** = area under the ROC curve (uses predicted probabilities)
- **Confusion matrix** — TP, FP, TN, FN

Label convention: `0 = Legitimate, 1 = Phishing`.

Results are saved to `results/hybrid_standard_results.csv` and `results/hybrid_standard_summary.json`.

---

## 8. Unseen-Domain Evaluation

This evaluation measures the model's ability to generalise to URLs from domains it has never seen during training — the core "zero-day" objective of the project.

Results are saved to `results/hybrid_unseen_domain_results.csv` and `results/hybrid_unseen_domain_summary.json`.

---

## 9. Cross-Validation

5-fold Stratified CV is applied to the training split (not the entire dataset) to avoid leaking test-set information. The same folds are used for all three models.

Results are saved to `results/hybrid_cv_results.csv` and `results/hybrid_cv_summary.json`.

---

## 10. Comparison with Existing RF

See `results/hybrid_final_report.md` for the actual experimental comparison.

The existing RF baseline (`models/url_model.pkl`) achieved (using GroupShuffleSplit, domain_overlap=0):

| Metric | Value |
|---|---|
| Accuracy | 0.9957 |
| Precision | 0.9968 |
| Recall | 0.9932 |
| F1 | 0.9950 |
| ROC-AUC | 0.9975 |

Whether the hybrid improves on these numbers is determined by the experimental results alone. See `results/hybrid_final_report.md` for the honest comparison.

---

## 11. Limitations

1. **Near-ceiling performance:** The existing RF already achieves >99.5% accuracy. Improvements from the hybrid will necessarily be small in absolute terms.

2. **SVM inference time:** SVM prediction is slower than RF at inference time due to support vector evaluation. For a web endpoint with single-URL requests this is negligible, but it would matter at high throughput.

3. **Static features only:** All 27 features are derived purely from the URL string. Features such as domain age, DNS TTL, page content, and certificate validity would likely improve detection of sophisticated phishing but require external calls.

4. **Dataset coverage:** The PhiUSIIL dataset is a fixed snapshot. Novel phishing patterns not represented in the dataset will not be reflected in these metrics.

5. **Platt scaling calibration:** SVM probabilities are calibrated via Platt scaling, which can be inaccurate when the training class ratio changes significantly. The scaler is fitted on the training split.

6. **Weight optimisation not explored:** Equal weights [1, 1] are used. Optimising weights on a validation set might improve results but is excluded to avoid overfitting to the specific split.
