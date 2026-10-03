"""Tests for the RF+SVM Soft-Voting hybrid model endpoint and artifacts.

Pre-existing test failures (documented before this file was created):
  - tests/test_url_api.py     : ImportError — 'from app.app import app' (app.py is not a package)
  - tests/test_url_model.py   : FileNotFoundError — models/url_final_model.pkl does not exist
  - tests/test_api.py         : ImportError — 'from app.app import app'; also references dataset/processed/
  - tests/test_model_inference.py: FileNotFoundError — models/final_random_forest_20_features.pkl

These pre-existing failures are NOT caused by the hybrid implementation.
They reference an older project layout that predates the current url_model.pkl pipeline.

This file tests only the CURRENT active pipeline + the new hybrid extension.
"""

import json
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Lazy import of the Flask app so the test can be collected even if the
# hybrid model pkl does not yet exist (it is created by train_hybrid_model.py).
# Tests that require the model will fail cleanly with a descriptive error.
# ---------------------------------------------------------------------------
def _get_app():
    import app as app_module
    return app_module.app


class HybridArtifactTests(unittest.TestCase):
    """Tests 1–2: Model file and feature-order file exist."""

    def test_01_hybrid_model_file_exists(self):
        model_path = ROOT / "models" / "rf_svm_soft_voting_model.pkl"
        self.assertTrue(
            model_path.exists(),
            f"Hybrid model file not found: {model_path}\n"
            "Run: python training/train_hybrid_model.py",
        )

    def test_02_hybrid_feature_order_file_exists(self):
        features_path = ROOT / "models" / "hybrid_feature_order.json"
        self.assertTrue(
            features_path.exists(),
            f"Hybrid feature order file not found: {features_path}\n"
            "Run: python training/train_hybrid_model.py",
        )

    def test_02b_hybrid_feature_order_matches_url_features(self):
        hybrid_path = ROOT / "models" / "hybrid_feature_order.json"
        url_path = ROOT / "models" / "url_features.json"
        if not hybrid_path.exists() or not url_path.exists():
            self.skipTest("Model files not yet created — run training script first.")
        hybrid_features = json.loads(hybrid_path.read_text(encoding="utf-8"))
        url_features = json.loads(url_path.read_text(encoding="utf-8"))
        self.assertEqual(
            hybrid_features,
            url_features,
            "hybrid_feature_order.json must match url_features.json (both are 27 features).",
        )

    def test_02c_hybrid_feature_count_is_27(self):
        features_path = ROOT / "models" / "hybrid_feature_order.json"
        if not features_path.exists():
            self.skipTest("Feature order file not yet created — run training script first.")
        features = json.loads(features_path.read_text(encoding="utf-8"))
        self.assertEqual(len(features), 27, f"Expected 27 features, got {len(features)}.")


class HybridAPITests(unittest.TestCase):
    """Tests 3–13: Hybrid API endpoint behaviour."""

    @classmethod
    def setUpClass(cls):
        model_path = ROOT / "models" / "rf_svm_soft_voting_model.pkl"
        if not model_path.exists():
            raise unittest.SkipTest(
                "Hybrid model not found. Run: python training/train_hybrid_model.py"
            )
        flask_app = _get_app()
        cls.client = flask_app.test_client()

    def test_03_valid_url_prediction(self):
        """Test 3: Valid URL returns 200 with success=True and required fields."""
        response = self.client.post(
            "/api/predict-hybrid", json={"url": "https://example.com/login"}
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["success"])
        self.assertIn("prediction", payload)
        self.assertIn("label", payload)
        self.assertIn("confidence", payload)
        self.assertIn("phishing_probability", payload)
        self.assertIn("legitimate_probability", payload)
        self.assertIn("model", payload)

    def test_04_empty_url_rejected(self):
        """Test 4: Empty URL string returns 400."""
        response = self.client.post("/api/predict-hybrid", json={"url": ""})
        self.assertEqual(response.status_code, 400)
        self.assertFalse(response.get_json()["success"])

    def test_05_invalid_url_rejected(self):
        """Test 5: Non-HTTP URL returns 400."""
        for value in ("example.com", "javascript:alert(1)", "ftp://example.com"):
            with self.subTest(value=value):
                response = self.client.post("/api/predict-hybrid", json={"url": value})
                self.assertEqual(response.status_code, 400)
                self.assertFalse(response.get_json()["success"])

    def test_06_prediction_class_is_valid(self):
        """Test 6: Prediction is 0 (Legitimate) or 1 (Phishing)."""
        response = self.client.post(
            "/api/predict-hybrid", json={"url": "https://example.com/login"}
        )
        payload = response.get_json()
        self.assertIn(payload["prediction"], {0, 1})

    def test_07_confidence_between_0_and_1(self):
        """Test 7: Confidence is a float in [0, 1]."""
        response = self.client.post(
            "/api/predict-hybrid", json={"url": "https://example.com/login"}
        )
        payload = response.get_json()
        conf = payload["confidence"]
        self.assertIsInstance(conf, float)
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)

    def test_08_phishing_probability_between_0_and_1(self):
        """Test 8: phishing_probability is a float in [0, 1]."""
        response = self.client.post(
            "/api/predict-hybrid", json={"url": "https://example.com/login"}
        )
        payload = response.get_json()
        prob = payload["phishing_probability"]
        self.assertIsInstance(prob, float)
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)

    def test_09_legitimate_probability_between_0_and_1(self):
        """Test 9: legitimate_probability is a float in [0, 1]."""
        response = self.client.post(
            "/api/predict-hybrid", json={"url": "https://example.com/login"}
        )
        payload = response.get_json()
        prob = payload["legitimate_probability"]
        self.assertIsInstance(prob, float)
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)

    def test_10_probabilities_sum_to_one(self):
        """Test 10: phishing_probability + legitimate_probability ≈ 1.0."""
        response = self.client.post(
            "/api/predict-hybrid", json={"url": "https://example.com/login"}
        )
        payload = response.get_json()
        total = payload["phishing_probability"] + payload["legitimate_probability"]
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_11_no_network_access_during_prediction(self):
        """Test 11: No DNS lookups or network requests during prediction."""
        with patch("socket.getaddrinfo", side_effect=AssertionError("DNS access forbidden")) as dns_mock:
            with patch("urllib.request.urlopen", side_effect=AssertionError("HTTP access forbidden")) as net_mock:
                response = self.client.post(
                    "/api/predict-hybrid", json={"url": "https://example.com/login"}
                )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_json()["success"])
        dns_mock.assert_not_called()
        net_mock.assert_not_called()

    def test_12_existing_predict_url_still_works(self):
        """Test 12: Original /api/predict-url endpoint still responds correctly."""
        response = self.client.post(
            "/api/predict-url", json={"url": "https://example.com/login"}
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["success"])
        self.assertIn(payload["prediction"], {0, 1})
        self.assertIn(payload["label"], {"Phishing", "Legitimate"})

    def test_13_existing_rf_endpoint_returns_valid_prediction(self):
        """Test 13: /api/predict-url returns a numeric prediction and float confidence."""
        response = self.client.post(
            "/api/predict-url",
            json={"url": "https://a.b.c.example.com/verify?token=12345abcde"},
        )
        payload = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(payload["success"])
        self.assertIn(payload["prediction"], {0, 1})
        self.assertIsInstance(payload["confidence"], float)

    def test_model_label_in_response(self):
        """Hybrid response must include 'model' key identifying the ensemble."""
        response = self.client.post(
            "/api/predict-hybrid", json={"url": "https://example.com/login"}
        )
        payload = response.get_json()
        self.assertEqual(payload.get("model"), "RF+SVM Soft Voting")

    def test_missing_url_field_rejected(self):
        """Request with no 'url' field must be rejected with 400."""
        response = self.client.post("/api/predict-hybrid", json={})
        self.assertEqual(response.status_code, 400)
        self.assertFalse(response.get_json()["success"])

    def test_malformed_json_rejected(self):
        """Malformed JSON must be rejected with 400."""
        response = self.client.post(
            "/api/predict-hybrid", data=b"{invalid", content_type="application/json"
        )
        self.assertEqual(response.status_code, 400)
        self.assertFalse(response.get_json()["success"])

    def test_http_url_also_accepted(self):
        """HTTP (non-HTTPS) URLs are valid inputs for the hybrid endpoint."""
        response = self.client.post(
            "/api/predict-hybrid", json={"url": "http://192.0.2.10/login"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_json()["success"])


class ExistingURLFeatureExtractionTests(unittest.TestCase):
    """Sanity-check that the existing feature extraction contract is preserved."""

    def test_static_analysis_only_flag(self):
        from src.url_feature_extraction import extract_url_features
        features = extract_url_features("https://example.com/login")
        self.assertTrue(features["safe_static_analysis_only"])

    def test_feature_count_is_27(self):
        from src.url_feature_extraction import URL_ONLY_FEATURE_ORDER
        self.assertEqual(len(URL_ONLY_FEATURE_ORDER), 27)


if __name__ == "__main__":
    unittest.main()
