import unittest

import pandas as pd

from app import _label, benchmark_features, benchmark_model
from src.feature_extraction import ALL_FEATURE_ORDER, FINAL_FEATURE_ORDER, select_final_features, validate_feature_vector


class ModelInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = benchmark_model
        cls.feature_order = benchmark_features
        cls.test_features = {feature: 1 for feature in ALL_FEATURE_ORDER}

    def test_final_model_and_feature_order(self):
        self.assertEqual(len(self.feature_order), 19)
        self.assertEqual(self.feature_order, FINAL_FEATURE_ORDER)
        self.assertEqual(self.model.n_features_in_, len(FINAL_FEATURE_ORDER))

    def test_synthetic_valid_feature_vector_inference(self):
        validated = validate_feature_vector(self.test_features, ALL_FEATURE_ORDER)
        values = pd.DataFrame([select_final_features(validated)], columns=self.feature_order)
        prediction = int(self.model.predict(values)[0])
        self.assertIn(prediction, {0, 1})

    def test_label_mapping_is_explicit(self):
        self.assertEqual(_label(0), 'Legitimate')
        self.assertEqual(_label(1), 'Phishing')


if __name__ == '__main__':
    unittest.main()
