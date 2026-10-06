"""Flask web application for phishing URL detection using machine learning.

Entry point: python app.py   OR   python -m flask --app app run

Models loaded at startup:
  - phase10_hybrid_wrapper: RF + calibrated Linear SVM soft-voting model (27 features).
    Used by the user-facing /api/predict-url route.
  - url_model: separate Random Forest retained for research compatibility.
  - hybrid_model: legacy RF+SVM endpoint model used by /api/predict-hybrid.
  - benchmark_model: Random Forest on pre-encoded dataset (19 features). Research API only.

No model visits, downloads, or executes the submitted URL.
All analysis is purely static (text parsing only).
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException

from src.feature_extraction import (
    ALL_FEATURE_ORDER,
    FINAL_FEATURE_ORDER,
    select_final_features,
    validate_feature_vector,
)
from src.phase10_hybrid import Phase10HybridModel
from src.url_feature_extraction import InvalidURL, extract_url_features, extract_url_only_row

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent

BENCHMARK_MODEL_PATH = ROOT / "models" / "benchmark_model.pkl"
BENCHMARK_FEATURES_PATH = ROOT / "models" / "benchmark_features.json"
URL_MODEL_PATH = ROOT / "models" / "url_model.pkl"
URL_FEATURES_PATH = ROOT / "models" / "url_features.json"
HYBRID_MODEL_PATH = ROOT / "models" / "rf_svm_soft_voting_model.pkl"
HYBRID_FEATURES_PATH = ROOT / "models" / "hybrid_feature_order.json"
PHASE10_RF_PATH = ROOT / "models" / "phase10_random_forest.pkl"
PHASE10_SVM_PATH = ROOT / "models" / "phase10_svm_pipeline.pkl"
PHASE10_HYBRID_PATH = ROOT / "models" / "phase10_rf_svm_hybrid.pkl"
PHASE10_FEATURE_ORDER_PATH = ROOT / "models" / "phase10_feature_order.json"

# ---------------------------------------------------------------------------
# Load models at startup — fail loudly if anything is wrong
# ---------------------------------------------------------------------------
benchmark_model = joblib.load(BENCHMARK_MODEL_PATH)
benchmark_features = json.loads(BENCHMARK_FEATURES_PATH.read_text(encoding="utf-8"))

url_model = joblib.load(URL_MODEL_PATH)
url_features = json.loads(URL_FEATURES_PATH.read_text(encoding="utf-8"))

hybrid_model = joblib.load(HYBRID_MODEL_PATH)
hybrid_features = json.loads(HYBRID_FEATURES_PATH.read_text(encoding="utf-8"))

phase10_rf_model = joblib.load(PHASE10_RF_PATH) if PHASE10_RF_PATH.exists() else None
phase10_svm_model = joblib.load(PHASE10_SVM_PATH) if PHASE10_SVM_PATH.exists() else None
phase10_hybrid_wrapper = None
phase10_feature_order = json.loads(PHASE10_FEATURE_ORDER_PATH.read_text(encoding="utf-8")) if PHASE10_FEATURE_ORDER_PATH.exists() else list(url_features)

if phase10_rf_model is not None and phase10_svm_model is not None:
    phase10_hybrid_wrapper = Phase10HybridModel(phase10_rf_model, phase10_svm_model, phase10_feature_order)

# Sanity checks — catch mismatches between saved model and feature-order file
if benchmark_features != FINAL_FEATURE_ORDER:
    raise RuntimeError(
        "benchmark_features.json does not match FINAL_FEATURE_ORDER in src/feature_extraction.py."
    )
if benchmark_model.n_features_in_ != len(benchmark_features):
    raise RuntimeError(
        f"benchmark_model expects {benchmark_model.n_features_in_} features "
        f"but benchmark_features.json has {len(benchmark_features)}."
    )
if url_model.n_features_in_ != len(url_features):
    raise RuntimeError(
        f"url_model expects {url_model.n_features_in_} features "
        f"but url_features.json has {len(url_features)}."
    )
if hybrid_model.n_features_in_ != len(hybrid_features):
    raise RuntimeError(
        f"hybrid_model expects {hybrid_model.n_features_in_} features "
        f"but hybrid_feature_order.json has {len(hybrid_features)}."
    )
if hybrid_features != url_features:
    raise RuntimeError(
        "hybrid_feature_order.json does not match url_features.json — feature mismatch."
    )

app = Flask(__name__)


@app.errorhandler(HTTPException)
def handle_http_exception(error: HTTPException):
    if request.path.startswith("/api/"):
        return jsonify({"success": False, "error": error.description}), error.code
    return error.get_response()


@app.errorhandler(500)
def handle_internal_server_error(error):
    if request.path.startswith("/api/"):
        original_exception = getattr(error, "original_exception", None)
        if original_exception is not None:
            app.logger.error(
                "Unhandled exception during API request",
                exc_info=(
                    type(original_exception),
                    original_exception,
                    original_exception.__traceback__,
                ),
            )
        else:
            app.logger.error("Unhandled server error during API request: %s", error)
        return jsonify({"success": False, "error": "Prediction failed. Please try again."}), 500
    return error.get_response()


def load_phase10_rf_model():
    if not PHASE10_RF_PATH.exists():
        raise FileNotFoundError(f"Phase 10 RF model not found: {PHASE10_RF_PATH}")
    return joblib.load(PHASE10_RF_PATH)


def load_phase10_svm_model():
    if not PHASE10_SVM_PATH.exists():
        raise FileNotFoundError(f"Phase 10 SVM model not found: {PHASE10_SVM_PATH}")
    return joblib.load(PHASE10_SVM_PATH)


def load_phase10_hybrid_model():
    rf_model = load_phase10_rf_model()
    svm_model = load_phase10_svm_model()
    feature_order = json.loads(PHASE10_FEATURE_ORDER_PATH.read_text(encoding="utf-8"))
    return Phase10HybridModel(rf_model, svm_model, feature_order)


# ---------------------------------------------------------------------------
# Label convention used by the benchmark and legacy hybrid endpoints:
#   0 = Legitimate
#   1 = Phishing
# ---------------------------------------------------------------------------
def _label(prediction: int) -> str:
    return "Phishing" if prediction == 1 else "Legitimate"


def _risk_level(phishing_probability: float) -> str:
    """Map phishing probability to a human-readable risk level.

    Thresholds are conservative to reduce false positives in a demo setting.
    Do not treat these as scientifically calibrated production thresholds.
    """
    if phishing_probability >= 0.75:
        return "High"
    if phishing_probability >= 0.40:
        return "Medium"
    return "Low"


def _run_model(model, values: list[float | int], feature_order: list[str]) -> tuple[int, float]:
    """Run a scikit-learn classifier and return (prediction_class, confidence).

    confidence is the predicted probability of the returned class.
    """
    frame = pd.DataFrame([values], columns=feature_order)
    prediction = int(model.predict(frame)[0])
    classes = list(model.classes_)
    confidence = float(model.predict_proba(frame)[0][classes.index(prediction)])
    return prediction, confidence


def _url_indicators(features: dict) -> list[dict[str, str]]:
    """Describe observable URL characteristics detected in the submitted URL.

    Each indicator has:
      status: 'ok'  — characteristic looks normal
              'warn' — characteristic is often associated with phishing
      text:   human-readable description
    """
    indicators = []

    # HTTPS
    if features["uses_https"]:
        indicators.append({"status": "ok", "text": "HTTPS is enabled"})
    else:
        indicators.append({"status": "warn", "text": "URL does not use HTTPS"})

    # IP address as host
    if features["uses_ip_address"]:
        indicators.append({"status": "warn", "text": "Hostname is an IP address (unusual for legitimate sites)"})
    else:
        indicators.append({"status": "ok", "text": "Hostname is a domain name"})

    # URL length
    url_len = features["url_length"]
    if url_len > 75:
        indicators.append({"status": "warn", "text": f"URL is long ({url_len} characters)"})
    else:
        indicators.append({"status": "ok", "text": f"URL length is normal ({url_len} characters)"})

    # Subdomains
    sub_count = features["subdomain_count"]
    if sub_count >= 3:
        indicators.append({"status": "warn", "text": f"High subdomain count ({sub_count})"})
    elif sub_count > 0:
        indicators.append({"status": "ok", "text": f"Subdomain count is normal ({sub_count})"})

    # @ symbol
    if features["contains_at_symbol"]:
        indicators.append({"status": "warn", "text": "URL contains an @ symbol"})

    # Encoded characters
    enc = features["encoded_character_count"]
    if enc > 0:
        indicators.append({"status": "warn", "text": f"URL contains {enc} percent-encoded character(s)"})

    # Suspicious characters
    if features["suspicious_character_count"] > 0:
        indicators.append({"status": "warn", "text": "URL contains suspicious characters"})

    # Known URL shortener
    if features["looks_like_shortener"]:
        indicators.append({"status": "warn", "text": "Hostname is a known URL shortener"})
    else:
        indicators.append({"status": "ok", "text": "Hostname is not a known shortener"})

    # Hyphen count (more than 4 hyphens in the full URL is unusual)
    hyphen = features["hyphen_count"]
    if hyphen > 4:
        indicators.append({"status": "warn", "text": f"URL contains many hyphens ({hyphen})"})

    return indicators


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    """Returns model availability and feature counts for sanity checking."""
    return jsonify({
        "status": "ok",
        "url_model_features": len(url_features),
        "benchmark_model_features": len(benchmark_features),
        "hybrid_model_features": len(hybrid_features),
    })


@app.post("/api/predict")
def predict_benchmark():
    """Classify using the benchmark model (research/demo use only).

    Accepts the 30-feature encoded benchmark representation. This endpoint is
    NOT connected to the UI — it exists for research reproducibility and viva
    demonstration. It accepts all 30 features but uses only the 19 selected ones.

    Request body:
        {"features": {"HTTPS": 1, "UsingIP": -1, ... (all 30 benchmark features)}}

    Label mapping: 0 = Legitimate, 1 = Phishing
    """
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("features"), dict):
        return jsonify({"success": False, "error": "Request body must be JSON with a 'features' object."}), 400

    try:
        validated = validate_feature_vector(payload["features"], ALL_FEATURE_ORDER)
        values = select_final_features(validated)
    except (TypeError, ValueError) as error:
        return jsonify({"success": False, "error": str(error)}), 400

    prediction, confidence = _run_model(benchmark_model, values, benchmark_features)
    return jsonify({
        "success": True,
        "prediction": prediction,
        "label": _label(prediction),
        "confidence": confidence,
    })


@app.post("/api/predict-hybrid")
def predict_hybrid():
    """Classify a URL using the RF+SVM Soft-Voting hybrid model. No network request is made.

    Request body:
        {"url": "https://example.com/login"}

    Response:
        {
            "success": true,
            "url": "https://example.com/login",
            "prediction": 1,
            "label": "Phishing",
            "confidence": 0.88,
            "phishing_probability": 0.88,
            "legitimate_probability": 0.12,
            "model": "RF+SVM Soft Voting",
            "risk_level": "High"
        }

    Label mapping: 0 = Legitimate, 1 = Phishing
    The URL is never opened. Only static string features are extracted.
    """
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("url"), str):
        return jsonify({"success": False, "error": "Request body must be JSON with a 'url' string."}), 400

    try:
        url_feature_dict = extract_url_features(payload["url"])
        model_input = extract_url_only_row(payload["url"])
    except InvalidURL as error:
        return jsonify({"success": False, "error": str(error)}), 400

    ordered_values = [model_input[name] for name in hybrid_features]
    frame = pd.DataFrame([ordered_values], columns=hybrid_features)

    prediction = int(hybrid_model.predict(frame)[0])
    probas = hybrid_model.predict_proba(frame)[0]
    # hybrid_model.classes_ is [0, 1] — index 0=Legitimate, index 1=Phishing
    classes = list(hybrid_model.classes_)
    phishing_prob = float(probas[classes.index(1)])
    legitimate_prob = float(probas[classes.index(0)])
    confidence = float(probas[classes.index(prediction)])

    return jsonify({
        "success": True,
        "url": url_feature_dict["url"],
        "prediction": prediction,
        "label": _label(prediction),
        "confidence": confidence,
        "phishing_probability": phishing_prob,
        "legitimate_probability": legitimate_prob,
        "model": "RF+SVM Soft Voting",
        "risk_level": _risk_level(phishing_prob),
        "indicators": _url_indicators(url_feature_dict),
    })


@app.post("/api/predict-url")
@app.post("/api/predict-url-hybrid")
def predict_url_hybrid():
    """Classify a URL using the hybrid model, with no external URL access.

    The submitted URL is validated and converted to the existing 27 ordered
    URL-string features before the RF and calibrated SVM probabilities are
    combined by equal-weight soft voting.
    """
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("url"), str):
        return jsonify({"success": False, "error": "Request body must be JSON with a 'url' string."}), 400

    url_text = payload["url"].strip()
    if not url_text:
        return jsonify({"success": False, "error": "URL cannot be empty."}), 400

    try:
        url_feature_dict = extract_url_features(url_text)
        model_input = extract_url_only_row(url_text)
    except InvalidURL as error:
        return jsonify({"success": False, "error": str(error)}), 400

    ordered_values = [model_input[name] for name in phase10_feature_order]
    frame = pd.DataFrame([ordered_values], columns=phase10_feature_order)
    if phase10_hybrid_wrapper is None:
        return jsonify({"success": False, "error": "Phase 10 hybrid model is unavailable."}), 503

    prediction, confidence, _, _, hybrid_probability = (
        phase10_hybrid_wrapper.predict_with_details(frame)
    )
    legitimate_probability = 1.0 - hybrid_probability

    return jsonify({
        "success": True,
        "model": "Hybrid RF + SVM",
        "url": url_feature_dict["url"],
        "prediction": int(prediction),
        "label": "Phishing" if prediction == 0 else "Legitimate",
        "confidence": confidence,
        "phishing_probability": hybrid_probability,
        "legitimate_probability": legitimate_probability,
        "risk_level": _risk_level(hybrid_probability),
        "indicators": _url_indicators(url_feature_dict),
    })


if __name__ == "__main__":
    app.run(debug=True)
