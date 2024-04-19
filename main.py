import pandas as pd
import os
import validator
import boto3
from io import StringIO
from dotenv import load_dotenv

load_dotenv()

def upload_csv_to_s3(df, bucket_name, folder_route, csv_filename):
    """Upload a DataFrame to S3 as a CSV file."""
    s3 = boto3.client(
        's3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name=os.getenv('AWS_DEFAULT_REGION')
    )

    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    full_s3_path = f"{folder_route}{csv_filename}"
    try:
        s3.put_object(Bucket=bucket_name, Key=full_s3_path, Body=csv_buffer.getvalue())
        print(f"File uploaded successfully")
    except Exception as e:
        print(f"Failed to upload file: {e}")

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
            continue  # Go back to the start of the loop

        print("\n" + "*" * 20 + "\n")
        target_variable = validator.user_select_target(data)
            
        print("\n" + "*" * 20 + "\n")
        algorithm_index = validator.select_algorithm()

        bucket_name = "capstonedatasets"

        algorithms = {1: "linear-regression", 2: "random-forest", 3: "knn"}
        folder_route = algorithms[algorithm_index] + "/"
        
        print("\n" + "*" * 20 + "\n")
        valid = validator.validator(data, algorithm_index, target_variable)
        if valid:
            print("\n" + "*" * 20 + "\n")
            print("***This dataset has passed the checks***")
            print("***Uploading it to S3***")
            upload_csv_to_s3(data, bucket_name, folder_route, csv_file_name)
            break  # Exit the loop since validation passed and file is uploaded
        print("")

if __name__ == "__main__":
    main()



