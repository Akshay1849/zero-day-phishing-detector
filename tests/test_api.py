import unittest

from app import app
from src.feature_extraction import ALL_FEATURE_ORDER


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()
        cls.example = {feature: 1 for feature in ALL_FEATURE_ORDER}

    def test_health(self):
        response = self.client.get('/api/health')
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload['status'], 'ok')
        self.assertEqual(payload['benchmark_model_features'], 19)
        self.assertEqual(payload['url_model_features'], 27)

    def test_homepage(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn('Phishing URL Detection', response.text)
        self.assertIn('Analyze URL', response.text)

    def test_valid_prediction(self):
        response = self.client.post('/api/predict', json={'features': self.example})
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload['success'])
        self.assertIn(payload['prediction'], {0, 1})
        self.assertIn(payload['label'], {'Phishing', 'Legitimate'})
        self.assertIsInstance(payload['confidence'], float)

    def test_missing_feature_is_rejected(self):
        invalid = dict(self.example)
        invalid.pop(ALL_FEATURE_ORDER[0])
        response = self.client.post('/api/predict', json={'features': invalid})
        self.assertEqual(response.status_code, 400)
        self.assertFalse(response.get_json()['success'])

    def test_invalid_feature_value_is_rejected(self):
        invalid = dict(self.example)
        invalid['HTTPS'] = 7
        response = self.client.post('/api/predict', json={'features': invalid})
        self.assertEqual(response.status_code, 400)
        self.assertFalse(response.get_json()['success'])

    def test_invalid_json_shape_is_rejected(self):
        response = self.client.post('/api/predict', json={'url': 'https://example.com'})
        self.assertEqual(response.status_code, 400)
        self.assertFalse(response.get_json()['success'])

    def test_benchmark_invalid_json_is_rejected(self):
        response = self.client.post('/api/predict', data=b'{invalid', content_type='application/json')
        self.assertEqual(response.status_code, 400)
        self.assertFalse(response.get_json()['success'])


if __name__ == '__main__':
    unittest.main()
