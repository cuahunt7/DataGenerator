import pandas as pd
import os
import validator
from database import upload_csv_to_s3, generate_presigned_url, check_dataset_exists
from metadata import extract_and_upload_metadata


def main():
    print("Welcome to the Dataset Validator!")
    while True:
        path = input("Please enter the path to your dataset or type 'exit' to quit: ")
        if path.lower() == 'exit':
            break

        csv_file_name = os.path.basename(path)

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
        folder_route = algorithm_name + "/"

        # Check for duplicate datasets in S3
        if check_dataset_exists("capstonedatasets", folder_route, path):
            print("A dataset with this name already exists in the selected category. Please rename your dataset.")
            continue

        print("\n" + "*" * 20 + "\n")
        clean_df, valid = validator.validator(data, algorithm_index, target_variable)
        if valid:
            print("\n" + "*" * 20 + "\n")
            print("***This dataset has passed the checks***")
            print("***Uploading it to S3***")
            object_key = upload_csv_to_s3(clean_df, "capstonedatasets", folder_route, path)
            if object_key:
                s3_link = generate_presigned_url("capstonedatasets", object_key)
                print("\n" + "*" * 20 + "\n")
                print(f"*** Dataset uploaded successfully. ***")
                print("***Extracting metadata and uploading to DynamoDB and S3***")
                extract_and_upload_metadata(clean_df, algorithm_name, "capstonedatasets", path, s3_link, target_variable)  # Pass target_variable here
                break  # Exit the loop since validation passed and file is uploaded
            else:
                print("*** Failed to generate a valid presigned URL ***")
        else:
            print("*** Failed to upload file to S3 ***")

if __name__ == "__main__":
    main()
