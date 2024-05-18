import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from io import StringIO

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
import sys
sys.modules['user_auth'] = MagicMock(get_temp_credentials=mock_get_temp_credentials)

# Now import the database module
import database

class TestDatabaseFunctions(unittest.TestCase):

    @patch('database.boto3.client', new_callable=lambda: MockBoto3Client)
    @patch('database.get_temp_credentials', new=mock_get_temp_credentials)
    @patch('database.os.getenv')
    def test_upload_csv_to_s3_synthetic(self, mock_getenv, mock_boto3_client):
        # Mock environment variables and credentials
        mock_getenv.side_effect = lambda key: 'mock_value'

        # Sample DataFrame
        df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})

        # Call the function
        result = database.upload_csv_to_s3(df, 'mock_bucket', 'mock_folder/', 'mock_file.csv', is_synthetic=True, id_token='mock_token')

        # Assertions
        self.assertIsNotNone(result)
        self.assertTrue(result.startswith("synthetic/"))

    @patch('database.boto3.client', new_callable=lambda: MockBoto3Client)
    @patch('database.os.getenv')
    def test_upload_csv_to_s3_non_synthetic(self, mock_getenv, mock_boto3_client):
        # Mock environment variables
        mock_getenv.side_effect = lambda key: 'mock_value'

        # Sample DataFrame
        df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})

        # Call the function
        result = database.upload_csv_to_s3(df, 'mock_bucket', 'mock_folder/', 'mock_file.csv', is_synthetic=False)

        # Assertions
        self.assertIsNotNone(result)
        self.assertFalse(result.startswith("synthetic/"))

    @patch('database.boto3.client')
    def test_check_dataset_exists(self, mock_boto3_client):
        # Mock S3 client
        mock_s3 = MockBoto3Client()
        mock_boto3_client.return_value = mock_s3

        # Test when the object exists
        mock_s3.head_object.return_value = {}
        result = database.check_dataset_exists('mock_bucket', 'mock_folder/', 'mock_file.csv')
        self.assertTrue(result)

        # Test when the object does not exist
        mock_s3.head_object.side_effect = mock_s3.exceptions.ClientError({}, 'head_object')
        result = database.check_dataset_exists('mock_bucket', 'mock_folder/', 'mock_file.csv')
        self.assertFalse(result)

if __name__ == '__main__':
    unittest.main()
