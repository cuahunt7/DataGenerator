import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime
import sys

# Mock the get_temp_credentials function to avoid importing user_auth
def mock_get_temp_credentials(id_token):
    return {
        'AccessKeyId': 'mock_access_key',
        'SecretKey': 'mock_secret_key',
        'SessionToken': 'mock_session_token'
    }

# Mock the boto3 client to avoid actual AWS calls
class MockBoto3Client:
    def __init__(self, *args, **kwargs):
        self.put_object = MagicMock()
        self.head_object = MagicMock()

    class exceptions:
        class ClientError(Exception):
            pass

# Replace the actual import with the mock
sys.modules['user_auth'] = MagicMock(get_temp_credentials=mock_get_temp_credentials)
sys.modules['boto3'] = MagicMock(client=lambda *args, **kwargs: MockBoto3Client())

# Now import the dataset_gen module
import dataset_gen

class TestDatasetGenFunctions(unittest.TestCase):

    def test_linear_regression_dataset(self):
        df = dataset_gen.linear_regression_dataset()
        self.assertEqual(df.shape[1], 9)  # 8 features + 1 target
        self.assertEqual(df.shape[0], 1000)  # 1000 rows

    def test_random_forest_dataset(self):
        df = dataset_gen.random_forest_dataset()
        self.assertEqual(df.shape[1], 9)  # 8 features + 1 target
        self.assertEqual(df.shape[0], 1000)  # 1000 rows

    def test_knn_dataset(self):
        df = dataset_gen.knn_dataset()
        self.assertEqual(df.shape[1], 9)  # 8 features + 1 target
        self.assertEqual(df.shape[0], 1000)  # 1000 rows

    @patch('dataset_gen.upload_csv_to_s3', return_value='mock_s3_key')
    @patch('dataset_gen.validator.validator', return_value=(pd.DataFrame(np.random.randn(1000, 9), columns=[f'Feature_{i}' for i in range(8)] + ['Target']), True))
    def test_main(self, mock_validator, mock_upload_csv_to_s3):
        result = dataset_gen.main('Linear Regression', 'less than 500', 'less than 10', 'mock_id_token')
        self.assertEqual(result, 'mock_s3_key')

if __name__ == '__main__':
    unittest.main()
