import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "static" / "script.js"


class FrontendModelSelectionTests(unittest.TestCase):
    def setUp(self):
        self.selected_model = "rf"
        self.responses = {
            "/api/predict-url": {
                "success": True,
                "model": "Random Forest",
                "prediction": 1,
                "label": "Legitimate",
                "confidence": 0.91,
                "phishing_probability": 0.09,
                "legitimate_probability": 0.91,
                "risk_level": "Low",
            },
            "/api/predict-url-hybrid": {
                "success": True,
                "model": "Hybrid RF + SVM",
                "prediction": 0,
                "label": "Phishing",
                "confidence": 0.82,
                "phishing_probability": 0.82,
                "legitimate_probability": 0.18,
            },
        }

    def _run_frontend(self, submitted_models=None, responses=None):
        import json

        # Build a self-contained harness in JavaScript; Python supplies only mock responses.
        harness = r"""
const vm = require('node:vm');
const fs = require('node:fs');
class Element {
  constructor() { this.listeners = {}; this.children = []; this.hidden = false; this.disabled = false; this.value = ''; this.textContent = ''; this.className = ''; }
  addEventListener(name, callback) { this.listeners[name] = callback; }
  replaceChildren() { this.children = []; }
  appendChild(child) { this.children.push(child); }
  focus() {}
}
const ids = ['prediction-form','url-input','analyze-button','button-label','button-loading','result-card','error-card','result-title','result-confidence','result-risk','result-model','indicator-list','clear-button','result-badge'];
const elements = Object.fromEntries(ids.map(id => [id, new Element()]));
elements['url-input'].value = 'https://example.invalid/login';
const document = {
  addEventListener: (_name, callback) => callback(),
  getElementById: id => elements[id],
  querySelector: () => ({value: selectedModel}),
  createElement: () => new Element(),
};
const calls = [];
const fetch = async endpoint => {
  calls.push(endpoint);
  const response = responses[endpoint];
  if (response.networkError) throw new TypeError('network failure');
  return {
    ok: response.status >= 200 && response.status < 300,
    status: response.status,
    text: async () => response.body
  };
};
vm.runInNewContext(frontendScript, {document, fetch, JSON, Error});
(async () => {
  for (const model of submittedModels) {
    selectedModel = model;
    await elements['prediction-form'].listeners.submit({preventDefault(){}});
  }
  process.stdout.write(JSON.stringify({calls, title:elements['result-title'].textContent, confidence:elements['result-confidence'].textContent, risk:elements['result-risk'].textContent, model:elements['result-model'].textContent, error:elements['error-card'].textContent, resultHidden:elements['result-card'].hidden}));
})();
"""
        response_configs = responses or {
            endpoint: {"status": 200, "body": json.dumps(payload)}
            for endpoint, payload in self.responses.items()
        }
        env_payload = {
            "script": SCRIPT_PATH.read_text(encoding="utf-8"),
            "responses": response_configs,
            "models": submitted_models or [self.selected_model],
        }
        executable = (
            "const frontendScript=JSON.parse(process.argv[1]).script;\n"
            "const responses=JSON.parse(process.argv[1]).responses;\n"
            "const submittedModels=JSON.parse(process.argv[1]).models;\n"
            "let selectedModel;\n"
            + harness
        )
        result = subprocess.run(
            ["node", "-e", executable, json.dumps(env_payload)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return json.loads(result.stdout)

    def test_rf_selection_calls_only_rf_endpoint_and_displays_rf_identity(self):
        result = self._run_frontend()
        self.assertEqual(result["calls"], ["/api/predict-url"])
        self.assertEqual(result["title"], "LEGITIMATE URL")
        self.assertEqual(result["model"], "Model: Random Forest")
        self.assertEqual(result["confidence"], "Confidence: 91.00%")
        self.assertIn("Phishing prob: 9.0%", result["risk"])
        self.assertIn("Legitimate prob: 91.0%", result["risk"])

    def test_hybrid_selection_calls_only_phase10_endpoint_and_displays_hybrid_fields(self):
        self.selected_model = "hybrid"
        result = self._run_frontend()
        self.assertEqual(result["calls"], ["/api/predict-url-hybrid"])
        self.assertEqual(result["title"], "PHISHING URL")
        self.assertEqual(result["model"], "Model: Hybrid RF + SVM")
        self.assertEqual(result["confidence"], "Confidence: 82.00%")
        self.assertIn("Phishing prob: 82.0%", result["risk"])
        self.assertIn("Legitimate prob: 18.0%", result["risk"])

    def test_hybrid_result_replaces_prior_rf_result(self):
        result_hybrid = self._run_frontend(["rf", "hybrid"])
        self.assertEqual(result_hybrid["calls"], ["/api/predict-url", "/api/predict-url-hybrid"])
        self.assertEqual(result_hybrid["model"], "Model: Hybrid RF + SVM")
        self.assertEqual(result_hybrid["title"], "PHISHING URL")
        self.assertEqual(result_hybrid["confidence"], "Confidence: 82.00%")
        self.assertIn("Phishing prob: 82.0%", result_hybrid["risk"])

    def test_http_json_error_is_displayed_to_user(self):
        result = self._run_frontend(responses={
            "/api/predict-url": {
                "status": 500,
                "body": '{"success":false,"error":"Prediction failed. Please try again."}',
            },
        })
        self.assertEqual(result["error"], "Prediction failed. Please try again.")
        self.assertTrue(result["resultHidden"])

    def test_http_500_non_json_response_displays_generic_server_error(self):
        result = self._run_frontend(responses={
            "/api/predict-url": {"status": 500, "body": "<html>server error</html>"},
        })
        self.assertEqual(result["error"], "The prediction server encountered an error. Please try again.")

    def test_invalid_json_response_displays_useful_error(self):
        result = self._run_frontend(responses={
            "/api/predict-url": {"status": 200, "body": "{invalid"},
        })
        self.assertEqual(result["error"], "The prediction server returned an invalid response. Please try again.")

    def test_empty_response_displays_retry_message(self):
        result = self._run_frontend(responses={
            "/api/predict-url": {"status": 200, "body": ""},
        })
        self.assertEqual(result["error"], "Server returned an empty response. Please try again.")

    def test_network_failure_displays_connection_message(self):
        result = self._run_frontend(responses={
            "/api/predict-url": {"networkError": True},
        })
        self.assertEqual(result["error"], "Unable to connect to the prediction server. Please try again.")

    def test_http_client_error_displays_backend_error(self):
        result = self._run_frontend(responses={
            "/api/predict-url": {
                "status": 400,
                "body": '{"success":false,"error":"URL must use HTTP or HTTPS."}',
            },
        })
        self.assertEqual(result["error"], "URL must use HTTP or HTTPS.")


if __name__ == "__main__":
    unittest.main()