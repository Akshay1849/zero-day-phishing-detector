import unittest

import pandas as pd

from app import url_features, url_model
from src.url_feature_extraction import URL_ONLY_FEATURE_ORDER, extract_url_only_row


class URLModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = url_model
        cls.order = url_features
        cls.summary = pd.read_json('results/url_model_metrics.json', typ='series').to_dict()

    def test_model_and_feature_order(self):
        self.assertEqual(self.model.n_features_in_, len(self.order))
        self.assertEqual(self.order, URL_ONLY_FEATURE_ORDER)
        self.assertEqual(self.summary['feature_count'], len(self.order))
        self.assertEqual(self.summary['label_mapping'], {'0': 'Legitimate', '1': 'Phishing'})

    def test_raw_url_inference(self):
        features = extract_url_only_row('https://example.com/login?next=%2Fhome')
        frame = pd.DataFrame([[features[name] for name in self.order]], columns=self.order)
        prediction = int(self.model.predict(frame)[0])
        self.assertIn(prediction, {0, 1})
        self.assertEqual(len(self.model.predict_proba(frame)[0]), 2)

    def test_url_feature_contract_is_deterministic(self):
        first = extract_url_only_row('http://192.0.2.10/a/b?x=1')
        second = extract_url_only_row('http://192.0.2.10/a/b?x=1')
        self.assertEqual(first, second)


if __name__ == '__main__':
    unittest.main()
