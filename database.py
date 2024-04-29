import boto3
from io import StringIO
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def upload_csv_to_s3(df, bucket_name, folder_route, file_path):
    """Upload a DataFrame to S3 as a CSV file, matching old function's structure and feedback."""
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
        return full_s3_path
    except Exception as e:
        print(f"Failed to upload file: {e}")
        return None

def check_dataset_exists(bucket_name, folder_route, file_path):
    s3 = boto3.client('s3')
    csv_filename = os.path.basename(file_path)
    full_s3_path = f"{folder_route}{csv_filename}"
    try:
        s3.head_object(Bucket=bucket_name, Key=full_s3_path)
        return True  # Object exists
    except s3.exceptions.ClientError:
        return False  # Object does not exist


def generate_presigned_url(bucket_name, object_key, expiration=3600):
    """Generate a presigned URL for temporary access to an S3 object."""
    s3 = boto3.client('s3')
    try:
        response = s3.generate_presigned_url('get_object',
                                             Params={'Bucket': bucket_name, 'Key': object_key},
                                             ExpiresIn=expiration)
        return response
    except Exception as e:
        print(f"Error generating presigned URL: {e}")
        return None
