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

# Initialize Cognito and Identity clients
cognito_client = boto3.client('cognito-idp', region_name=AWS_DEFAULT_REGION)
identity_client = boto3.client('cognito-identity', region_name=AWS_DEFAULT_REGION)

# Function to sign up a new user
def signup_user(email, password):
    try:
        # Sign up user in Cognito
        response = cognito_client.sign_up(
            ClientId=COGNITO_APP_CLIENT_ID,
            Username=email,
            Password=password,
            UserAttributes=[
                {'Name': 'email', 'Value': email}
            ]
        )
        # Confirm the user sign up
        cognito_client.admin_confirm_sign_up(
            UserPoolId=COGNITO_USER_POOL_ID,
            Username=email
        )
        return response
    except Exception as e:
        logging.error(f"Error signing up: {e}")
        return None

# Function to authenticate a user and get tokens
def authenticate_user(email, password):
    try:
        # Initiate auth to get tokens
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

# Function to get temporary credentials using an ID token
def get_temp_credentials(id_token):
    try:
        # Get identity ID using the ID token
        identity_id_response = identity_client.get_id(
            IdentityPoolId=COGNITO_IDENTITY_POOL_ID,
            Logins={
                f'cognito-idp.{AWS_DEFAULT_REGION}.amazonaws.com/{COGNITO_USER_POOL_ID}': id_token
            }
        )
        identity_id = identity_id_response['IdentityId']

        # Get temporary credentials for the identity
        credentials_response = identity_client.get_credentials_for_identity(
            IdentityId=identity_id,
            Logins={
                f'cognito-idp.{AWS_DEFAULT_REGION}.amazonaws.com/{COGNITO_USER_POOL_ID}': id_token
            }
        )
        return credentials_response['Credentials']
    except Exception as e:
        logging.error(f"Error getting temporary credentials: {e}")
        return None
