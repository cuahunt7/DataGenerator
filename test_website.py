import unittest
from unittest.mock import patch, MagicMock
import sys
import pandas as pd

# Mock the crypt module since it's not available on Windows
sys.modules['crypt'] = MagicMock()

# Mock the user_auth module to avoid import errors
sys.modules['user_auth'] = MagicMock()
from user_auth import signup_user, authenticate_user

class TestWebsite(unittest.TestCase):
    @patch('streamlit.session_state', new_callable=lambda: {'logged_in': True, 'email': 'test@email.com', 'id_token': 'token123', 'menu': 'Home'})
    @patch('streamlit.columns', return_value=(MagicMock(), MagicMock(), MagicMock()))
    @patch('streamlit.form_submit_button', return_value=True)
    @patch('streamlit.selectbox', side_effect=['Random Forest', '10 or more', '500 or more', 'Health', 'Clean'])
    @patch('streamlit.markdown')
    @patch('website.generate_presigned_url', return_value="https://example.com/dataset.csv")
    @patch('website.fetch_dataset_metadata', return_value=[{
        'Dataset Name': 'Test Dataset',
        'Machine Learning Task': 'Random Forest',
        'S3ObjectKey': 'test_dataset.csv',
        'Number of Features': '10',
        'Number of Instances': '500',
        'Size in KB': '100',
        'Source Link': 'https://example.com',
        'Target Variable': 'target',
        'Topic': 'Health'
    }])
    @patch('website.make_dataset_unclean', return_value=pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]}))
    @patch('website.generate_synthetic_dataset', return_value='synthetic_dataset.csv')
    @patch('pandas.read_csv', return_value=pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]}))  # Mocking pd.read_csv
    @patch('boto3.resource')  # Mocking boto3 resource
    @patch('boto3.client')    # Mocking boto3 client
    def test_dataset_form_submission(self, mock_client, mock_resource, mock_read_csv, mock_generate_synthetic_dataset, mock_make_dataset_unclean, mock_fetch_dataset_metadata, mock_generate_presigned_url, mock_markdown, mock_selectbox, mock_button, mock_columns, mock_session_state):
        # Mock the resource and client to return a MagicMock object, avoiding real AWS calls
        mock_resource.return_value = MagicMock()
        mock_client.return_value = MagicMock()

        # Mock the signup_user and authenticate_user functions
        signup_user.return_value = True
        authenticate_user.return_value = {'AuthenticationResult': {'IdToken': 'test_token'}}

        # Import and call the main function from your Streamlit app
        from website import main
        main()

        # Print Debug Information
        print(mock_markdown.call_args_list)  # This will print all calls to mock_markdown to the console

        # Check if the dataset download link is present
        expected_call_1 = '**Download Link:** [Download Dataset](https://example.com/dataset.csv)'
        found_expected_call_1 = any(expected_call_1 in str(call) for call in mock_markdown.call_args_list)

        # Check for the welcome message
        expected_call_2 = '**Welcome, test@email.com!**'
        found_expected_call_2 = any(expected_call_2 in str(call) for call in mock_markdown.call_args_list)

        self.assertTrue(found_expected_call_1, f"Expected markdown call with '{expected_call_1}' not found.")
        self.assertTrue(found_expected_call_2, f"Expected markdown call with '{expected_call_2}' not found.")

if __name__ == '__main__':
    unittest.main()
