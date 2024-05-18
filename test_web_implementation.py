import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import re
import botocore
from dotenv import load_dotenv

# Load environment variables from 'env 1'
load_dotenv('env 1')

# Import the functions from your script
from web_implementation import generate_presigned_url, make_dataset_unclean, fetch_dataset_metadata, password_requirements

class TestWebImplementation(unittest.TestCase):

    @patch('web_implementation.s3_client')
    def test_generate_presigned_url_success(self, mock_s3_client):
        mock_s3_client.generate_presigned_url.return_value = 'http://example.com'
        url = generate_presigned_url('bucket_name', 'object_key')
        self.assertEqual(url, 'http://example.com')

    @patch('web_implementation.s3_client')
    def test_generate_presigned_url_failure(self, mock_s3_client):
        error_response = {'Error': {'Code': '403', 'Message': 'Forbidden'}}
        mock_s3_client.generate_presigned_url.side_effect = botocore.exceptions.ClientError(error_response, 'get_object')
        url = generate_presigned_url('bucket_name', 'object_key')
        self.assertIsNone(url)

    def test_make_dataset_unclean(self):
        # Create a sample dataframe
        df = pd.DataFrame({
            'A': range(10),
            'B': range(10, 20),
            'C': range(20, 30)
        })
        # Introduce errors into the dataframe
        unclean_df = make_dataset_unclean(df.copy(), error_rate=0.1)
        self.assertTrue(unclean_df.isnull().values.any() or not unclean_df.equals(df))

    @patch('web_implementation.dynamodb')
    def test_fetch_dataset_metadata(self, mock_dynamodb):
        mock_table = MagicMock()
        mock_table.scan.return_value = {'Items': [{'DatasetName': 'TestDataset'}]}
        mock_dynamodb.Table.return_value = mock_table

        result = fetch_dataset_metadata('classification', '10 or more', '500 or more', 'Finance', 'Clean')
        self.assertEqual(result, [{'DatasetName': 'TestDataset'}])

    def test_password_requirements(self):
        self.assertTrue(password_requirements('Aa1!aaaa'))
        self.assertFalse(password_requirements('short1!'))
        self.assertFalse(password_requirements('NoDigits!'))
        self.assertFalse(password_requirements('NOLOWERCASE1!'))
        self.assertFalse(password_requirements('nouppercase1!'))
        self.assertFalse(password_requirements('NoSpecialChar1'))

if __name__ == '__main__':
    unittest.main()
