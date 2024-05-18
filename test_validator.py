import unittest
from unittest.mock import patch
import pandas as pd
from validator import user_select_target, select_algorithm, validate_algorithm_suitability, validator

class TestValidatorFunctions(unittest.TestCase):

    def setUp(self):
        # Sample DataFrame for testing
        self.data = pd.DataFrame({
            'Age': [25, 30, 35],
            'Salary': [50000, 60000, 65000],
            'Department': ['HR', 'IT', 'Finance']
        })

    @patch('builtins.input', side_effect=['1'])  # Assume user selects the first column
    def test_user_select_target(self, mock_input):
        result = user_select_target(self.data)
        self.assertEqual(result, 'Age', "Should return 'Age' as the selected target variable")

    @patch('builtins.input', side_effect=['1'])  # Assume user selects Linear Regression
    def test_select_algorithm(self, mock_input):
        result = select_algorithm()
        self.assertEqual(result, 1, "Should return 1 for Linear Regression")

    @patch('builtins.print')
    def test_validate_algorithm_suitability_for_random_forest(self, mock_print):
        # Create a test DataFrame with a suitable length for 'Target' column
        # Ensure the DataFrame has more rows to avoid the length mismatch error
        self.data = pd.DataFrame({
            'Feature1': range(15),
            'Feature2': range(15, 30),
            'Target': range(15)
        })
        print("Testing validate_algorithm_suitability for Random Forest")
        print(f"Data types:\n{self.data.dtypes}")
        print(f"Unique values in 'Target': {self.data['Target'].nunique()}")
        result = validate_algorithm_suitability(self.data, 'Target', 'random-forest')
        print(f"Validation result: {result}")
        self.assertFalse(result, "Random Forest should not be suitable for a continuous target with more than 10 unique values")

    def test_validator_linear_regression(self):
        # Test with a larger dataset for linear regression
        self.data = pd.DataFrame({
            'Age': [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
            'Salary': [50000, 60000, 65000, 70000, 75000, 80000, 85000, 90000, 95000, 100000],
            'Department': ['HR', 'IT', 'Finance', 'HR', 'IT', 'Finance', 'HR', 'IT', 'Finance', 'HR'],
            'Target': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        })
        print("Testing validator for Linear Regression")

        # Adding debug prints to trace the flow
        print(f"Original DataFrame:\n{self.data}")

        # Call validator and capture intermediate results
        try:
            clean_df, result = validator(self.data, 1, 'Target')  # Linear Regression index
            print(f"Processed DataFrame Shape: {clean_df.shape}")
            print(f"Processed DataFrame Head:\n{clean_df.head()}")
            print(f"Validation result: {result}")
        except Exception as e:
            print(f"Exception during validation: {e}")
            self.fail("Exception occurred during validation")

        self.assertIsInstance(clean_df, pd.DataFrame, "Should return a DataFrame")
        if result is not None:  # Check if result is not None before assertion
            self.assertTrue(result, "Linear Regression validation should pass for continuous target")
        else:
            self.fail("Linear Regression validation result is None")

if __name__ == '__main__':
    unittest.main()
