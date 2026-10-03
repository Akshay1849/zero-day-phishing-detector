import json
import unittest
from pathlib import Path
from unittest.mock import patch

import app as app_module
from src.url_feature_extraction import URL_ONLY_FEATURE_ORDER, extract_url_only_row

ROOT = Path(__file__).resolve().parents[1]


class Phase10HybridModelTests(unittest.TestCase):
    def test_01_model_files_exist(self):
        for rel in [
            "models/phase10_random_forest.pkl",
            "models/phase10_svm_pipeline.pkl",
            "models/phase10_rf_svm_hybrid.pkl",
            "models/phase10_feature_order.json",
        ]:
            self.assertTrue((ROOT / rel).exists(), f"Missing {rel}")

    def test_02_feature_order_and_count(self):
        feature_order = json.loads((ROOT / "models" / "phase10_feature_order.json").read_text(encoding="utf-8"))
        self.assertGreater(len(feature_order), 0)
        self.assertEqual(len(feature_order), 27)
        self.assertEqual(feature_order, URL_ONLY_FEATURE_ORDER)
        self.assertEqual(feature_order, json.loads((ROOT / "models" / "url_features.json").read_text(encoding="utf-8")))

    def test_03_rf_model_loads(self):
        model = app_module.load_phase10_rf_model()
        self.assertTrue(hasattr(model, "predict"))

    def test_04_svm_pipeline_loads(self):
        model = app_module.load_phase10_svm_model()
        self.assertTrue(hasattr(model, "predict"))

    def test_05_hybrid_wrapper_loads(self):
        model = app_module.load_phase10_hybrid_model()
        self.assertTrue(hasattr(model, "predict"))
        self.assertTrue(hasattr(model, "predict_proba"))

    def test_06_valid_url_produces_prediction(self):
        response = app_module.app.test_client().post("/api/predict-url-hybrid", json={"url": "https://example.com/login"})
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["success"])
        self.assertEqual(payload["model"], "Hybrid RF + SVM")
        self.assertIn(payload["prediction"], {0, 1})

    def test_07_prediction_is_either_0_or_1(self):
        response = app_module.app.test_client().post("/api/predict-url-hybrid", json={"url": "https://example.com/login"})
        self.assertIn(response.get_json()["prediction"], {0, 1})

    def test_08_confidence_is_valid(self):
        response = app_module.app.test_client().post("/api/predict-url-hybrid", json={"url": "https://example.com/login"})
        conf = response.get_json()["confidence"]
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)

    def test_09_rf_probability_is_valid(self):
        response = app_module.app.test_client().post("/api/predict-url-hybrid", json={"url": "https://example.com/login"})
        self.assertGreaterEqual(response.get_json()["rf_probability"], 0.0)
        self.assertLessEqual(response.get_json()["rf_probability"], 1.0)

    def test_10_svm_probability_is_valid(self):
        response = app_module.app.test_client().post("/api/predict-url-hybrid", json={"url": "https://example.com/login"})
        self.assertGreaterEqual(response.get_json()["svm_probability"], 0.0)
        self.assertLessEqual(response.get_json()["svm_probability"], 1.0)

    def test_11_hybrid_probability_is_valid(self):
        response = app_module.app.test_client().post("/api/predict-url-hybrid", json={"url": "https://example.com/login"})
        self.assertGreaterEqual(response.get_json()["hybrid_probability"], 0.0)
        self.assertLessEqual(response.get_json()["hybrid_probability"], 1.0)

    def test_phase10_api_preserves_raw_label_mapping_for_synthetic_urls(self):
        urls = [
            "http://secure-login-example.invalid/verify-account",
            "http://paypal-security-check.invalid/login",
            "http://bank-account-verification.invalid/update",
            "https://www.google.com/",
            "https://www.wikipedia.org/",
        ]
        client = app_module.app.test_client()
        model = app_module.load_phase10_hybrid_model()

        for url in urls:
            with self.subTest(url=url):
                feature_row = extract_url_only_row(url)
                ordered_values = [feature_row[name] for name in app_module.phase10_feature_order]
                expected_prediction, _, rf_probability, svm_probability, hybrid_probability = model.predict_with_details(ordered_values)
                response = client.post("/api/predict-url-hybrid", json={"url": url})
                payload = response.get_json()

                self.assertEqual(response.status_code, 200)
                self.assertEqual(payload["prediction"], expected_prediction)
                self.assertEqual(payload["prediction"], 0 if hybrid_probability >= 0.5 else 1)
                self.assertEqual(payload["label"], "Phishing" if expected_prediction == 0 else "Legitimate")
                self.assertAlmostEqual(payload["confidence"], max(hybrid_probability, 1.0 - hybrid_probability))
                self.assertAlmostEqual(payload["rf_probability"], rf_probability)
                self.assertAlmostEqual(payload["svm_probability"], svm_probability)
                self.assertAlmostEqual(payload["phishing_probability"], hybrid_probability)
                self.assertAlmostEqual(payload["legitimate_probability"], 1.0 - hybrid_probability)

    def test_phase10_api_returns_both_raw_classes(self):
        client = app_module.app.test_client()
        examples = {
            "http://pdllfsstp-fepm.web.app/": (0, "Phishing"),
            "https://www.islrn.org": (1, "Legitimate"),
        }
        for url, (prediction, label) in examples.items():
            with self.subTest(url=url):
                response = client.post("/api/predict-url-hybrid", json={"url": url})
                payload = response.get_json()
                self.assertEqual(response.status_code, 200)
                self.assertEqual(payload["prediction"], prediction)
                self.assertEqual(payload["label"], label)

    def test_12_empty_url_rejected(self):
        response = app_module.app.test_client().post("/api/predict-url-hybrid", json={"url": ""})
        self.assertEqual(response.status_code, 400)

    def test_13_invalid_json_rejected(self):
        response = app_module.app.test_client().post("/api/predict-url-hybrid", data=b"{invalid", content_type="application/json")
        self.assertEqual(response.status_code, 400)

    def test_14_missing_url_rejected(self):
        response = app_module.app.test_client().post("/api/predict-url-hybrid", json={})
        self.assertEqual(response.status_code, 400)

    def test_15_no_network_requests_are_made(self):
        with patch("socket.getaddrinfo", side_effect=AssertionError("DNS access forbidden")) as dns_mock:
            with patch("urllib.request.urlopen", side_effect=AssertionError("HTTP access forbidden")) as net_mock:
                response = app_module.app.test_client().post("/api/predict-url-hybrid", json={"url": "https://example.com/login"})
        self.assertEqual(response.status_code, 200)
        dns_mock.assert_not_called()
        net_mock.assert_not_called()

    def test_16_existing_predict_url_still_works(self):
        response = app_module.app.test_client().post("/api/predict-url", json={"url": "https://example.com/login"})
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["success"])
        self.assertEqual(payload["model"], "Random Forest")

    def test_17_existing_predict_still_works(self):
        response = app_module.app.test_client().post("/api/predict", json={"features": {"HTTPS": 1, "UsingIP": -1, "LongURL": 1, "ShortURL": 1, "Symbol@": 1, "Redirecting//": 1, "PrefixSuffix-": -1, "SubDomains": 0, "DomainRegLen": -1, "Favicon": 1, "NonStdPort": 1, "HTTPSDomainURL": -1, "RequestURL": 1, "AnchorURL": 0, "LinksInScriptTags": -1, "ServerFormHandler": -1, "InfoEmail": 1, "AbnormalURL": 1, "WebsiteForwarding": 0, "StatusBarCust": 1, "DisableRightClick": 1, "UsingPopupWindow": 1, "IframeRedirection": 1, "AgeofDomain": -1, "DNSRecording": -1, "WebsiteTraffic": 0, "PageRank": -1, "GoogleIndex": 1, "LinksPointingToPage": 1, "StatsReport": 1, "Index": 0}})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_json()["success"])


if __name__ == "__main__":
    unittest.main()
