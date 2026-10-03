# CURRENT ACTIVE PIPELINE AUDIT
**Date:** 2026-09-22  
**Auditor:** Pre-modification inspection — no files were changed during this audit.

---

## 1. PROJECT STRUCTURE OVERVIEW

```
zero-day-phishing-detector/
├── app.py                          ← Active Flask application (single file)
├── src/
│   ├── url_feature_extraction.py   ← ACTIVE URL feature extractor (27 features)
│   └── feature_extraction.py       ← Benchmark feature validator (NOT URL model)
├── models/
│   ├── url_model.pkl               ← ACTIVE URL model (RF, 27 features)
│   ├── url_features.json           ← ACTIVE feature order (27 features)
│   ├── benchmark_model.pkl         ← Benchmark model (19 features, research only)
│   └── benchmark_features.json     ← Benchmark feature order
├── training/
│   ├── train_url_model.py          ← ACTIVE URL model training script
│   ├── url_dataset_helper.py       ← Dataset preparation helper
│   └── train_benchmark.py          ← Benchmark model training script
├── dataset/raw/
│   ├── PhiUSIIL_Phishing_URL_Dataset.csv  ← ACTIVE URL dataset (235,795 rows)
│   └── phishing.csv                        ← Benchmark dataset (11,054 rows)
├── results/
│   ├── url_model_metrics.json      ← ACTIVE URL model metrics
│   ├── final_results_summary.json  ← Comprehensive summary (both tracks)
│   └── final_results_report.md     ← Detailed report
├── tests/
│   ├── test_url_feature_extraction.py  ← PASSING (6/6 tests pass)
│   ├── test_url_api.py                 ← BROKEN: imports from app.app (not a package)
│   ├── test_url_model.py               ← BROKEN: references url_final_model.pkl (doesn't exist)
│   ├── test_api.py                     ← BROKEN: imports from app.app; references processed/
│   └── test_model_inference.py         ← BROKEN: references final_random_forest_20_features.pkl
└── templates/index.html            ← Single-page frontend
```

---

## 2. ACTIVE URL PIPELINE (TRACK B — THE CURRENT MODEL)

### 2.1 Dataset
- **File:** `dataset/raw/PhiUSIIL_Phishing_URL_Dataset.csv`
- **Original rows:** 235,795
- **After cleaning:** 235,360 valid rows
  - 10 rows removed (invalid/unparseable URLs)
  - 425 duplicate URLs removed
- **Unique domains:** 167,154 (used for domain-aware splitting)
- **Label mapping in raw dataset:** `0 = Phishing`, `1 = Legitimate`
- **Label mapping in trained model:** `0 = Legitimate`, `1 = Phishing`  
  ⚠️ **CRITICAL NOTE:** `train_url_model.py` line 46–47 remaps labels: `y = (label == 0).astype(int)` — converts raw-dataset Phishing (0) → model label 1, and raw-dataset Legitimate (1) → model label 0.

### 2.2 Feature Extraction
- **File:** `src/url_feature_extraction.py`
- **Feature count:** **27 features** (all from static URL string analysis only — no network/DNS)
- **Feature list (in order):** `URL_ONLY_FEATURE_ORDER`

```python
URL_ONLY_FEATURE_ORDER = [
    "url_length", "hostname_length", "path_length", "query_length", "fragment_length",
    "dot_count", "hyphen_count", "underscore_count", "digit_count", "special_character_count",
    "slash_count", "query_parameter_count", "url_depth", "subdomain_count", "uses_https",
    "uses_ip_address", "contains_at_symbol", "encoded_character_count", "suspicious_character_count",
    "url_entropy", "hostname_entropy", "path_entropy", "letter_count", "numeric_count",
    "digit_ratio", "special_character_ratio", "looks_like_shortener",
]
```

**Feature types:**
- Continuous (floats): url_entropy, hostname_entropy, path_entropy, digit_ratio, special_character_ratio
- Integer counts: url_length, hostname_length, path_length, query_length, fragment_length, dot_count, hyphen_count, underscore_count, digit_count, special_character_count, slash_count, query_parameter_count, url_depth, subdomain_count, encoded_character_count, suspicious_character_count, letter_count, numeric_count
- Binary (0/1): uses_https, uses_ip_address, contains_at_symbol, looks_like_shortener

### 2.3 Random Forest Model Configuration
- **File:** `models/url_model.pkl`
- **Type:** `sklearn.ensemble.RandomForestClassifier`
- **n_features_in_:** 27 (confirmed by `model.n_features_in_`)
- **n_estimators:** 300 (confirmed by `model.n_estimators`)
- **random_state:** 42 (confirmed by `model.random_state`)
- **max_depth:** None (unlimited)
- **class_weight:** None
- **n_jobs:** -1 (used during training, not stored as hyperparameter in same sense)
- **classes_:** [0, 1] (0=Legitimate, 1=Phishing)

### 2.4 Train/Test Split Protocol
- **Method:** `GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)`
- **Groups:** registrable domain (e.g., "example.com") — computed by `training/url_dataset_helper.py::registrable_domain()`
- **Train rows:** 164,195
- **Test rows:** 71,165
- **Domain overlap:** 0 (enforced by GroupShuffleSplit — no domain appears in both sets)

### 2.5 Unseen-Domain Split Protocol
- The GroupShuffleSplit IS the unseen-domain split. The test set contains domains that never appear in training.
- **Training domains:** (varies per split — ~70% of 167,154 = ~116,000-117,000 domains)
- **Testing domains:** (~30% of 167,154 = ~50,000 domains)
- **domain_overlap:** 0 (guaranteed by GroupShuffleSplit)

### 2.6 Preprocessing / Scaling
- **None.** The active Random Forest model uses raw/unscaled features directly.
- No `StandardScaler` or any other scaler is applied.

### 2.7 Feature Order File
- **File:** `models/url_features.json`
- **Contents:** same as `URL_ONLY_FEATURE_ORDER` in `src/url_feature_extraction.py` (27 features)

---

## 3. RESULTS DISCREPANCY — IMPORTANT FINDING

The `results/final_results_summary.json` (Track B) reports:

```json
"feature_engineering": {
    "total_url_only_features_engineered": 27,
    "selected_features_count": 15,
    "selected_features": ["uses_https", "path_length", "slash_count", ...]
},
"final_model": {
    "n_estimators": 100,
    "model_file": "models/url_final_model.pkl"
}
```

**BUT the currently active model (`models/url_model.pkl`):**
- Uses **all 27 features** (not 15 selected features)
- Has **n_estimators=300** (not 100)
- Is trained by `training/train_url_model.py` which uses all `URL_ONLY_FEATURE_ORDER` (27 features)

This means there are TWO generations of the URL model:
1. **Old/notebook model:** `url_final_model.pkl` — 15 features, n_estimators=100 (referenced in `final_results_summary.json`, does NOT exist as a file)
2. **Current/active model:** `url_model.pkl` — 27 features, n_estimators=300, loaded by `app.py`

The `test_url_model.py` test references `url_final_model.pkl` and `url_feature_order.json` (which do not exist) — meaning those tests were written against the OLD model version.

The `test_url_api.py` imports `from app.app import app` which does not work because `app.py` is NOT in an `app/` package.

**The currently active, running pipeline is: `url_model.pkl` + `url_features.json` + `app.py` + `src/url_feature_extraction.py`**

---

## 4. ACTIVE FLASK API (`app.py`)

### Endpoints:
| Endpoint | Method | Purpose | Status |
|---|---|---|---|
| `/` | GET | Serve frontend UI | Active |
| `/api/health` | GET | Model availability check | Active |
| `/api/predict-url` | POST | URL prediction with 27-feature RF | **Active — UI uses this** |
| `/api/predict` | POST | Benchmark model (19 features) | Research only |

### Active endpoint `/api/predict-url`:
- Input: `{"url": "https://example.com"}`
- Output: `{"success": true, "prediction": 0|1, "label": "Legitimate"|"Phishing", "confidence": float, "risk_level": "Low"|"Medium"|"High", "indicators": [...]}`
- Label mapping: `0 = Legitimate`, `1 = Phishing`

---

## 5. BENCHMARK / TRACK A (Research Only — NOT Active UI)

- **File:** `models/benchmark_model.pkl`
- **Features:** 19 selected from 30 benchmark features
- **Training data:** `dataset/raw/phishing.csv` (11,054 rows)
- **Training script:** `training/train_benchmark.py`
- **Label convention:** 0 = Legitimate, 1 = Phishing (remapped from dataset's -1/1)
- **Status:** Accessible only via `/api/predict` (research endpoint, not UI)

---

## 6. TEST STATUS (BEFORE ANY MODIFICATION)

| Test File | Status | Issue |
|---|---|---|
| `test_url_feature_extraction.py` | ✅ PASSING (6/6) | None |
| `test_url_api.py` | ❌ BROKEN | `from app.app import app` — `app` is not a package |
| `test_url_model.py` | ❌ BROKEN | References `url_final_model.pkl` which does not exist |
| `test_api.py` | ❌ BROKEN | `from app.app import app`; references `dataset/processed/` which doesn't exist |
| `test_model_inference.py` | ❌ BROKEN | References `final_random_forest_20_features.pkl` which doesn't exist |

**Only `test_url_feature_extraction.py` passes.** The other 4 test files contain stale imports referencing an older project structure.

---

## 7. STORED METRICS (EXISTING BASELINE — DO NOT OVERWRITE)

From `results/url_model_metrics.json` (the current active model — 27 features):

| Metric | Value |
|---|---|
| Accuracy | 0.9957142 |
| Precision | 0.9967528 |
| Recall (Phishing) | 0.9931658 |
| F1 | 0.9949561 |
| ROC-AUC | 0.9974794 |
| TN | 40,778 |
| FP | 98 |
| FN | 207 |
| TP | 30,082 |

**Note:** This uses GroupShuffleSplit (the domain-aware/unseen-domain split). There is no separate "standard random" split stored for this model.

---

## 8. SUMMARY FOR HYBRID IMPLEMENTATION

| Decision Point | Answer |
|---|---|
| Active feature count | **27** (all URL_ONLY_FEATURE_ORDER) |
| Active feature file | `models/url_features.json` |
| Active model | `models/url_model.pkl` |
| Active RF n_estimators | 300 |
| Active RF random_state | 42 |
| Active RF max_depth | None |
| Active RF class_weight | None |
| Label mapping | 0=Legitimate, 1=Phishing |
| Split protocol | GroupShuffleSplit(test_size=0.30, random_state=42) on domain groups |
| Domain-aware? | YES — GroupShuffleSplit guarantees zero domain overlap |
| Preprocessing required for RF | None (trees are invariant to scale) |
| Preprocessing required for SVM | StandardScaler (fitted on training data only) |
| Existing test files that pass | `test_url_feature_extraction.py` (6/6) |
| Broken test files (pre-existing) | test_url_api.py, test_url_model.py, test_api.py, test_model_inference.py |
