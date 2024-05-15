import boto3
import logging
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load AWS Cognito configuration from environment variables
COGNITO_USER_POOL_ID = os.getenv("COGNITO_USER_POOL_ID")
COGNITO_APP_CLIENT_ID = os.getenv("COGNITO_APP_CLIENT_ID")
COGNITO_IDENTITY_POOL_ID = os.getenv("COGNITO_IDENTITY_POOL_ID")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION")

cognito_client = boto3.client('cognito-idp', region_name=AWS_DEFAULT_REGION)
identity_client = boto3.client('cognito-identity', region_name=AWS_DEFAULT_REGION)

# Cognito signup and signin functions
def signup_user(email, password):
    try:
        response = cognito_client.sign_up(
            ClientId=COGNITO_APP_CLIENT_ID,
            Username=email,
            Password=password,
            UserAttributes=[
                {'Name': 'email', 'Value': email}
            ]
        )
        cognito_client.admin_confirm_sign_up(
            UserPoolId=COGNITO_USER_POOL_ID,
            Username=email
        )
        return response
    except Exception as e:
        logging.error(f"Error signing up: {e}")
        return None

def authenticate_user(email, password):
    try:
        response = cognito_client.initiate_auth(
            ClientId=COGNITO_APP_CLIENT_ID,
            AuthFlow='USER_PASSWORD_AUTH',
            AuthParameters={
                'USERNAME': email,
                'PASSWORD': password
            }
        )
        logging.info(f"Authentication response: {response}")
        return response
    except cognito_client.exceptions.NotAuthorizedException:
        logging.error("The username or password is incorrect")
        return None
    except cognito_client.exceptions.UserNotConfirmedException:
        logging.error("User is not confirmed")
        return None
    except Exception as e:
        logging.error(f"Error authenticating: {e}")
        return None