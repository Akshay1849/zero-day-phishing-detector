import unittest
from unittest.mock import patch

import app as _app_module

app = _app_module.app


class URLApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()

    def test_valid_url_returns_hybrid_prediction_without_component_outputs(self):
        response = self.client.post('/api/predict-url', json={'url': 'https://example.com/login'})
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload['success'])
        self.assertEqual(payload['model'], 'Hybrid RF + SVM')
        self.assertIn(payload['prediction'], {0, 1})
        self.assertIn(payload['label'], {'Phishing', 'Legitimate'})
        self.assertIsInstance(payload['confidence'], float)
        self.assertAlmostEqual(payload['phishing_probability'] + payload['legitimate_probability'], 1.0)
        self.assertEqual(payload['prediction'], 0 if payload['label'] == 'Phishing' else 1)
        self.assertEqual(
            payload['prediction'],
            0 if payload['phishing_probability'] >= 0.5 else 1,
        )
        self.assertNotIn('rf_probability', payload)
        self.assertNotIn('svm_probability', payload)

    def test_https_http_ip_and_query_urls_are_accepted_for_analysis(self):
        for url in ('https://example.com', 'http://192.0.2.10/login', 'https://a.example.com/?x=1'):
            with self.subTest(url=url):
                response = self.client.post('/api/predict-url', json={'url': url})
                self.assertEqual(response.status_code, 200)
                self.assertIn(response.get_json()['prediction'], {0, 1})

    def test_empty_or_invalid_url_is_rejected(self):
        for value in ('', 'example.com', 'javascript:alert(1)'):
            with self.subTest(value=value):
                response = self.client.post('/api/predict-url', json={'url': value})
                self.assertEqual(response.status_code, 400)
                self.assertFalse(response.get_json()['success'])

    def test_missing_url_field_is_rejected(self):
        response = self.client.post('/api/predict-url', json={})
        self.assertEqual(response.status_code, 400)
        self.assertFalse(response.get_json()['success'])

    def test_malformed_json_is_rejected(self):
        response = self.client.post('/api/predict-url', data=b'{invalid', content_type='application/json')
        self.assertEqual(response.status_code, 400)
        self.assertFalse(response.get_json()['success'])

    def test_prediction_does_not_make_network_or_dns_requests(self):
        with patch('socket.getaddrinfo', side_effect=AssertionError('DNS access is forbidden')) as dns:
            with patch('urllib.request.urlopen', side_effect=AssertionError('Network access is forbidden')) as network:
                response = self.client.post('/api/predict-url', json={'url': 'https://example.com/login'})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_json()['success'])
        dns.assert_not_called()
        network.assert_not_called()

    def test_url_request_shape_is_rejected(self):
        response = self.client.post('/api/predict-url', json={'features': {}})
        self.assertEqual(response.status_code, 400)
        self.assertFalse(response.get_json()['success'])

    def test_unexpected_prediction_error_returns_safe_json_and_is_logged(self):
        with patch.object(
            _app_module.phase10_hybrid_wrapper,
            'predict_with_details',
            side_effect=RuntimeError('model inference exploded'),
        ):
            with self.assertLogs(app.logger.name, level='ERROR') as captured:
                response = self.client.post('/api/predict-url', json={'url': 'https://example.com/login'})

        self.assertEqual(response.status_code, 500)
        self.assertTrue(response.is_json)
        self.assertEqual(
            response.get_json(),
            {'success': False, 'error': 'Prediction failed. Please try again.'},
        )
        self.assertIn('model inference exploded', '\n'.join(captured.output))


if __name__ == '__main__':
    unittest.main()
