from unittest.mock import patch, MagicMock
import unittest
import pandas as pd  # Import Pandas for creating a DataFrame

from metadata import extract_and_upload_metadata

class TestExtractAndUploadMetadata(unittest.TestCase):

    @patch('builtins.input', side_effect=['Dataset Name', '1', 'http://example.com', 'object_key'])
    @patch('os.path.getsize', return_value=1024)
    @patch('boto3.resource')
    def test_extract_and_upload_metadata_success(self, mock_boto3_resource, mock_os_path_getsize, mock_input):
        mock_table = MagicMock()
        mock_boto3_resource.return_value.Table.return_value = mock_table
        
        # Create a mock DataFrame
        mock_data = pd.DataFrame({
            'feature1': range(10),
            'feature2': range(10)
        })

        # Mocking environment variables
        with patch.dict('os.environ', {'AWS_ACCESS_KEY_ID': 'your_access_key', 'AWS_SECRET_ACCESS_KEY': 'your_secret_key', 'AWS_DEFAULT_REGION': 'your_region'}):
            extract_and_upload_metadata(data=mock_data, algorithm_name='algorithm', bucket_name='bucket', file_path='/path/to/file', object_key='object_key', target_variable='target')

        mock_table.put_item.assert_called_once_with(
            Item={
                'Dataset Name': 'Dataset Name',
                'Machine Learning Task': 'algorithm',
                'Topic': 'Health',
                'Number of Instances': 10,
                'Number of Features': 2,
                'Size in KB': 1,
                'Source Link': 'http://example.com',
                'S3ObjectKey': 'object_key',
                'Target Variable': 'target'
            }
        )

if __name__ == '__main__':
    unittest.main()
