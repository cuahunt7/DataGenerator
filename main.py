import pandas as pd
import os
import validator
from database import upload_csv_to_s3, check_dataset_exists
from metadata import extract_and_upload_metadata


def main():
    print("Welcome to the Dataset Validator!")
    while True:
        path = input("Please enter the path to your dataset or type 'exit' to quit: ")
        if path.lower() == 'exit':
            break

        try:
            data = pd.read_csv(path)
        except Exception as e:
            print(f"An error occurred while loading the dataset: {e}")
            continue

        print("\n" + "*" * 20 + "\n")
        target_variable = validator.user_select_target(data)

        print("\n" + "*" * 20 + "\n")
        algorithm_index = validator.select_algorithm()
        algorithms = {1: "linear-regression", 2: "random-forest", 3: "k-nearest-neighbors"}
        algorithm_name = algorithms[algorithm_index]

        # Algorithm suitability checks
        if algorithm_name == "random-forest":
            # Check if the target variable is continuous for random forest
            if pd.api.types.is_numeric_dtype(data[target_variable]):
                unique_values = data[target_variable].nunique()
                if unique_values > 10:
                    print("\n" + "*" * 20 + "\n")
                    print(f"*** The selected target variable '{target_variable}' is continuous and not suitable for random forest classification. ***")
                    print("*** Please select a different algorithm or choose a categorical target variable. ***")
                    continue

        elif algorithm_name == "linear-regression":
            # Ensure the target variable is continuous for linear regression
            if not pd.api.types.is_numeric_dtype(data[target_variable]):
                print("\n" + "*" * 20 + "\n")
                print(f"*** The selected target variable '{target_variable}' is not continuous and not suitable for linear regression. ***")
                print("*** Please select a different algorithm or choose a continuous target variable. ***")
                continue

        elif algorithm_name == "k-nearest-neighbors":
            # Ensure the target variable is categorical for KNN classification
            if pd.api.types.is_numeric_dtype(data[target_variable]):
                unique_values = data[target_variable].nunique()
                if unique_values > 10:
                    print("\n" + "*" * 20 + "\n")
                    print(f"*** The selected target variable '{target_variable}' is continuous and not suitable for k-nearest neighbors classification. ***")
                    print("*** Please select a different algorithm or choose a categorical target variable. ***")
                    continue

        folder_route = algorithm_name + "/"

        # Check for duplicate datasets in S3
        if check_dataset_exists("capstonedatasets", folder_route, path):
            print("A dataset with this name already exists in the selected category. Please rename your dataset.")
            continue

        print("\n" + "*" * 20 + "\n")
        clean_df, valid = validator.validator(data, algorithm_index, target_variable)
        if valid:
            object_key = upload_csv_to_s3(clean_df, "capstonedatasets", folder_route, path)
            if object_key:
                print(f"*** Dataset uploaded successfully. ***")
                print("*** Extracting metadata and uploading to DynamoDB ***")
                extract_and_upload_metadata(clean_df, algorithm_name, "capstonedatasets", path, object_key, target_variable)
                break  # Exit the loop since validation passed and file is uploaded
            else:
                print("*** Failed to upload file to S3 ***")
        else:
            print("*** Dataset validation failed ***")

if __name__ == "__main__":
    main()

