import pandas as pd
import os
import validator
import boto3
from io import StringIO
from dotenv import load_dotenv

load_dotenv()

def upload_csv_to_s3(df, bucket_name, folder_route, file_path):
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
    csv_filename = os.path.basename(file_path)
    full_s3_path = f"{folder_route}{csv_filename}"
    try:
        s3.put_object(Bucket=bucket_name, Key=full_s3_path, Body=csv_buffer.getvalue())
        print("File uploaded successfully to S3.")
        return f"https://{bucket_name}.s3.{os.getenv('AWS_DEFAULT_REGION')}.amazonaws.com/{full_s3_path}"
    except Exception as e:
        print(f"Failed to upload file: {e}")
        return None

def extract_and_upload_metadata(data, algorithm_name, bucket_name, file_path, s3_link):
    """Extract metadata from the dataset and upload it to DynamoDB."""
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table('DatasetMetadata')
    dataset_name = input("What is the dataset name? ")
    topic = input("Enter the topic (1 for Health, 2 for Finance, 3 for Environment, 4 for Technology): ")
    topics = {1: "Health", 2: "Finance", 3: "Environment", 4: "Technology"}
    source_link = input("Enter the source link: ")
    size_in_mb = os.path.getsize(file_path) / (1024 * 1024)  # size in megabytes

    try:
        response = table.put_item(
            Item={
                'Dataset Name': dataset_name,
                'Machine Learning Task': algorithm_name,
                'Topic': topics[int(topic)],
                'Number of Instances': str(data.shape[0]),
                'Number of Features': str(data.shape[1]),
                'Size in MB': f"{size_in_mb:.2f} MB",
                'Source Link': source_link,
                'Download Link': s3_link
            }
        )
        print("Metadata uploaded successfully to DynamoDB.")
    except Exception as e:
        print(f"Failed to upload metadata: {e}")

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
            continue  # Go back to the start of the loop

        target_variable = validator.user_select_target(data)
        algorithm_index = validator.select_algorithm()
        algorithms = {1: "Linear Regression", 2: "Random Forest", 3: "K-nearest Neighbors"}
        algorithm_name = algorithms[algorithm_index]
        bucket_name = "capstonedatasets"
        folder_route = f"{algorithm_name.lower().replace(' ', '-')}/"
        
        valid = validator.validator(data, algorithm_index, target_variable)
        if valid:
            s3_link = upload_csv_to_s3(data, bucket_name, folder_route, path)
            if s3_link:
                print("\n" + "*" * 20 + "\n")
                print("***This dataset has passed the checks***")
                print("***Extracting metadata and uploading to DynamoDB and S3***")
                extract_and_upload_metadata(data, algorithm_name, bucket_name, path, s3_link)
                break  # Exit the loop since validation passed and file is uploaded
        else:
            print("The dataset did not pass validation. Please review and try again.")

if __name__ == "__main__":
    main()



