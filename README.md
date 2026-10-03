# Zero-Day Phishing URL Detection Using Machine Learning-Based Feature Analysis

A college mini-project based on URL-related feature analysis and machine-learning classification. The project studies phishing versus legitimate samples using the benchmark's encoded feature representation and evaluates whether a reduced feature set preserves standard classification performance.

The phrase "zero-day" is treated as a research motivation only. The current benchmark has no raw URL or domain identifier, so it does not support rigorous unseen-domain or zero-day evaluation.

## Problem Statement

Phishing URLs and websites can imitate legitimate services and exploit users through suspicious URL, domain, and web-page characteristics. A machine-learning classifier can assist with consistent analysis of encoded URL-related indicators, but its reliability depends on representative data, valid feature definitions, and evaluation that avoids leakage.

## Aim

Develop and evaluate a feature-based machine-learning system that classifies benchmark samples as Phishing or Legitimate, while investigating whether feature selection can reduce the model input space without materially reducing standard classification performance.

## Objectives

- Inspect and validate the supplied benchmark dataset.
- Analyze URL, structural/domain, and security/web feature groups.
- Train baseline Decision Tree, Random Forest, SVM, and Gradient Boosting models.
- Compare all 31 features with selected feature subsets.
- Evaluate the selected model with fixed-test metrics, stratified cross-validation, and error analysis.
- Provide a Flask demonstration for safe static URL analysis while preserving the encoded-feature benchmark API.

## Dataset

The verified dataset is `dataset/raw/phishing.csv`:

- 11,054 samples
- 32 columns
- 31 numeric input features
- Target column: `class`
- `-1`: Phishing
- `1`: Legitimate
- No missing values or duplicate rows
- No raw URL or domain identifier

Phase 3 uses the reproducible stratified split with `test_size=0.30` and `random_state=42`:

- Training: 7,737 samples
- Testing: 3,317 samples

## Feature Engineering and Selection

The dataset contains pre-encoded indicators such as `HTTPS`, `AnchorURL`, `WebsiteTraffic`, `SubDomains`, `DomainRegLen`, and related security/web features. Phase 5 ranked features using Random Forest importance fitted on training data only. The final selected configuration contains 20 features:

`HTTPS`, `AnchorURL`, `WebsiteTraffic`, `Index`, `SubDomains`, `PrefixSuffix-`, `LinksInScriptTags`, `ServerFormHandler`, `RequestURL`, `LinksPointingToPage`, `DomainRegLen`, `AgeofDomain`, `GoogleIndex`, `DNSRecording`, `UsingIP`, `PageRank`, `LongURL`, `HTTPSDomainURL`, `InfoEmail`, `ShortURL`.

The `Index` value is retained because it is part of the verified trained representation, although it is treated as a dataset identifier rather than a URL characteristic. A future production dataset should resolve this representation issue before deployment.

## Models Evaluated

Phase 4 evaluated:

- Decision Tree
- Random Forest
- Support Vector Machine with training-only scaling
- Gradient Boosting

Phase 5 additionally evaluated Random Forest and SVM with Top-K and SelectKBest feature selection. Phase 6 performed 5-fold Stratified Cross-Validation and fixed-test error analysis.

## Final Model and Results

Final model: **Random Forest with 20 selected features**.

| Metric | Result |
| --- | ---: |
| Accuracy | 97.02% |
| F1 | 96.62% |
| Phishing Recall | 96.32% |
| ROC-AUC | 99.58% |
| CV F1 | 95.94% ± 0.38% |
| CV ROC-AUC | 99.45% ± 0.12% |

Error analysis on the fixed test set:

- TP = 1,415
- TN = 1,803
- FP = 45
- FN = 54

These are results from this project dataset and test procedure, not copied from the base paper.

## Web Application

The Flask application has two deliberately separate paths:

1. `POST /api/predict` remains the benchmark inference endpoint. It accepts the complete 31-feature encoded representation, selects the exact 20 final features, and runs the saved Random Forest.
2. `POST /api/predict-url` uses the independently trained raw-URL model described below.

The homepage accepts a URL and displays deterministic measurements such as URL length, HTTPS usage, IP-host detection, subdomain count, query count, and encoded-character count. It never opens the URL, performs DNS or HTTP requests, follows redirects, downloads content, or executes JavaScript.

The URL analyzer is not connected to the benchmark model. The benchmark's encoded features cannot be faithfully reconstructed from a raw URL alone.

## Raw URL Detection Model

This independent experiment uses `dataset/raw/PhiUSIIL_Phishing_URL_Dataset.csv` from UCI. It retained 235,360 valid URL rows and 167,154 derived registrable-domain groups. It uses 27 deterministic URL-string features and excludes page-content, HTML, DNS, reputation, traffic, and probability-like UCI fields. Labels are `0 = Phishing` and `1 = Legitimate`.

The selected model is a Random Forest with 15 URL-only features. Domain groups are used only for evaluation splits, never as model inputs.

| Evaluation | Accuracy | F1 | Phishing Recall | ROC-AUC |
| --- | ---: | ---: | ---: | ---: |
| Standard split | 99.55% | 99.47% | 99.19% | 99.73% |
| Unseen-domain split | 99.57% | 99.53% | 99.32% | 99.75% |

Standard 5-fold cross-validation:

- Accuracy: 99.49% +/- 0.05%
- Precision: 99.76% +/- 0.05%
- Recall: 99.05% +/- 0.12%
- F1: 99.40% +/- 0.06%
- ROC-AUC: 99.71% +/- 0.04%

The 15-feature metrics above describe the selected-feature research experiment; that experiment's model artifact is not included. The Flask application instead loads the 27-feature model at `models/url_model.pkl`, with feature order in `models/url_features.json`. Its separate metrics are recorded in `results/url_model_metrics.json`.

## Run Locally

From the repository root:

```powershell
.\venv\Scripts\Activate.ps1
python -m flask --app app run
```

Open `http://127.0.0.1:5000/`.

The dashboard's URL path maps `0` to Phishing and `1` to Legitimate using the independent URL model.

## API Usage

Endpoint: `POST /api/predict`

Example request shape:

```json
{
  "features": {
    "Index": 0,
    "UsingIP": 1,
    "LongURL": 1,
    "ShortURL": 1,
    "Symbol@": 1,
    "Redirecting//": 1,
    "PrefixSuffix-": -1,
    "SubDomains": 0,
    "HTTPS": 1,
    "DomainRegLen": -1,
    "Favicon": 1,
    "NonStdPort": 1,
    "HTTPSDomainURL": -1,
    "RequestURL": 1,
    "AnchorURL": 0,
    "LinksInScriptTags": -1,
    "ServerFormHandler": -1,
    "InfoEmail": 1,
    "AbnormalURL": 1,
    "WebsiteForwarding": 0,
    "StatusBarCust": 1,
    "DisableRightClick": 1,
    "UsingPopupWindow": 1,
    "IframeRedirection": 1,
    "AgeofDomain": -1,
    "DNSRecording": -1,
    "WebsiteTraffic": 0,
    "PageRank": -1,
    "GoogleIndex": 1,
    "LinksPointingToPage": 1,
    "StatsReport": 1
  }
}
```

Successful responses have this form:

```json
{
  "success": true,
  "prediction": -1,
  "label": "Phishing",
  "confidence": 0.98
}
```

### Static URL analysis

Endpoint: `POST /api/predict-url`

Request:

```json
{
  "url": "https://example.com/login"
}
```

Valid URLs return a prediction and confidence from the independent URL model. Invalid URLs return HTTP 400. This endpoint never visits the submitted URL.

## Testing

Run the application tests:

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```

The tests cover model loading, exact feature order and count, known test-example inference, label mapping, valid API requests, missing features, invalid values, invalid JSON shape, and JSON responses.

## Limitations

- The current benchmark does not support rigorous unseen-domain or zero-day evaluation because URL/domain identifiers are unavailable.
- The application does not claim to detect every true zero-day phishing attack.
- Raw URL extraction is not implemented because the complete 31-feature mapping cannot be faithfully reproduced from the available information.
- The URL model is trained on the UCI raw-URL dataset and remains separate from the benchmark model.
- The URL-only features are descriptive measurements and must not be treated as substitutes for the benchmark feature definitions.
- Several features require external domain, DNS, page, traffic, search, or reputation information.
- The benchmark's `Index` column is a dataset identifier retained in the trained representation; it should be removed or redefined in a future production-grade dataset.
- The Flask app is a research prototype, not a guaranteed security verdict.

## Future Work

- Collect raw URLs with stable domain identifiers and provenance.
- The future raw-URL dataset must contain at least a raw `url` column and a trusted binary `class`/label column, with documented label meaning. Domain identifiers and collection timestamps are strongly recommended for unseen-domain and temporal evaluation.
- Define and validate URL, page, DNS, and domain feature extraction against the source dataset specification.
- Evaluate domain-level, temporal, and unseen-domain splits.
- Calibrate probabilities and assess operational thresholds.
- Add authentication, rate limiting, logging, monitoring, and deployment hardening before any real-world use.
