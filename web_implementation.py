import re
import boto3
from boto3.dynamodb.conditions import Attr
import pandas as pd
import numpy as np
from botocore.exceptions import NoCredentialsError, ClientError
from dotenv import load_dotenv
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Setup AWS clients
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_DEFAULT_REGION')
)
dynamodb = boto3.resource(
    'dynamodb',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_DEFAULT_REGION')
)
AWS_S3_BUCKET = os.getenv('AWS_S3_BUCKET')

# Presigned URL function
def generate_presigned_url(bucket_name, object_key, expiration=3600):
    try:
        response = s3_client.generate_presigned_url('get_object',
                                                    Params={'Bucket': bucket_name, 'Key': object_key},
                                                    ExpiresIn=expiration)
        return response
    except ClientError as e:
        logging.error(f"Error generating presigned URL: {e}")
        return None

def make_dataset_unclean(dataframe, error_rate=0.05):
    """
    Introduce noise/errors into the dataset to simulate unclean data.
    
    Parameters:
    dataframe (pd.DataFrame): The DataFrame to be modified.
    error_rate (float): The rate at which to introduce errors.
    
    Returns:
    pd.DataFrame: The modified DataFrame with introduced errors.
    """
    # Calculate the number of changes to be made
    n_changes = int(np.ceil(error_rate * dataframe.size))

    for _ in range(n_changes):
        # Select a random cell in the DataFrame
        row_idx = np.random.randint(0, dataframe.shape[0])
        col_idx = np.random.randint(0, dataframe.shape[1])

        # Introduce an error or NaN into the selected cell
        if np.random.rand() > 0.5:
            dataframe.iat[row_idx, col_idx] = np.nan  # Insert NaN
        else:
            # Introduce random noise by selecting a random value from the column
            col = dataframe.iloc[:, col_idx]
            dataframe.iat[row_idx, col_idx] = np.random.choice(col.dropna().values) if col.dropna().values.size > 0 else np.nan

    return dataframe

def fetch_dataset_metadata(input_algorithm, features, instances, topic, cleanliness):
    table = dynamodb.Table('DatasetMetadata')
    
    # Normalize inputs
    input_algorithm = input_algorithm.replace(' ', '-').lower()
    topic = topic.capitalize()
    
    # Log inputs for debugging
    logging.info(f"Inputs - Algorithm: {input_algorithm}, Features: {features}, Instances: {instances}, Topic: {topic}, Cleanliness: {cleanliness}")

    # Building the filter expression based on user inputs
    filter_expression = Attr('Machine Learning Task').eq(input_algorithm)

    if features == 'less than 10':
        filter_expression &= Attr('Number of Features').lt(10)
    elif features == '10 or more':
        filter_expression &= Attr('Number of Features').gte(10)
        
    if instances == 'less than 500':
        filter_expression &= Attr('Number of Instances').lt(500)
    elif instances == '500 or more':
        filter_expression &= Attr('Number of Instances').gte(500)
    
    filter_expression &= Attr('Topic').eq(topic)
    
    # Log the filter expression for debugging
    logging.info(f"Combined Filter Expression: {filter_expression}")

    try:
        response = table.scan(
            FilterExpression=filter_expression
        )
        # Log the response for debugging
        logging.info(f"Combined Filter Response: {response}")
        return response['Items'] if 'Items' in response else []
    except Exception as e:
        logging.error(f"Error fetching data from DynamoDB: {e}")
        return []

def password_requirements(password):
    if (len(password) < 8 or
        not re.search(r"\d", password) or
        not re.search(r"[A-Z]", password) or
        not re.search(r"[a-z]", password) or
        not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password)):
        return False
    return True