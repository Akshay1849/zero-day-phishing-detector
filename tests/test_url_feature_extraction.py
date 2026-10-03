import unittest

from src.url_feature_extraction import InvalidURL, extract_url_features


class URLFeatureExtractionTests(unittest.TestCase):
    def test_https_url(self):
        features = extract_url_features('https://example.com/login')
        self.assertTrue(features['uses_https'])
        self.assertFalse(features['uses_ip_address'])
        self.assertEqual(features['hostname'], 'example.com')

    def test_http_ip_address(self):
        features = extract_url_features('http://192.0.2.10/login')
        self.assertFalse(features['uses_https'])
        self.assertTrue(features['uses_ip_address'])

    def test_long_url(self):
        url = 'https://example.com/' + ('a' * 180)
        features = extract_url_features(url)
        self.assertGreater(features['url_length'], 180)

    def test_at_symbol_subdomains_and_query(self):
        features = extract_url_features('https://a.b.example.com/login?next=%2Fhome&user=test@example.com')
        self.assertTrue(features['contains_at_symbol'])
        self.assertEqual(features['subdomain_count'], 2)
        self.assertEqual(features['query_parameter_count'], 2)
        self.assertGreater(features['encoded_character_count'], 0)

    def test_empty_and_invalid_urls(self):
        for value in ['', 'example.com', 'ftp://example.com', 'https://']:
            with self.subTest(value=value), self.assertRaises(InvalidURL):
                extract_url_features(value)

    def test_static_only_marker(self):
        features = extract_url_features('https://example.com')
        self.assertTrue(features['safe_static_analysis_only'])
        self.assertNotIn('prediction', features)


if __name__ == '__main__':
    unittest.main()
