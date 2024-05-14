import streamlit as st
import pandas as pd
import boto3
import logging
from web_implementation import generate_presigned_url, fetch_dataset_metadata, make_dataset_unclean
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

# Custom styles
st.markdown("""
    <style>
    body {
        font-family: 'Arial', sans-serif;
    }
    .stApp {
        background-color: #FFFFFF;
        background-image: linear-gradient(180deg, #f8fbfd, #FFFFFF 90%);
    }
    h1, h2, h3, h4, h5, h6, .stMarkdown, .stLabel, .stSelectbox label, .stNumberInput label {
        color: #0055A2;
        font-weight: bold;
    }
    .st-cb, .st-ci, .st-ck, .st-cl, .stMarkdown, h1, h2 {
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
    }
    .stSelectbox, .stNumberInput, .stButton {
        margin: auto;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        transition: box-shadow 0.2s ease-in-out;
    }
    .stSelectbox > div > div, .stNumberInput > input {
        color: #013369;
        border: 1px solid #ced4da;
        background-color: white;
    }
    .stSelectbox > label, .stNumberInput > label, .stTextInput label {
        display: block;
        text-align: center;
        width: 100%;
        color: #0055A2;
    }
    .stButton > button {
        width: 100%;
        border-radius: 20px;
        background-color: #00529B;
        color: white;
        padding: 10px 24px;
        border: none;
        font-size: 16px;
    }
    .stButton > button:hover {
        background-color: #003366;
        box-shadow: 0 6px 12px rgba(0,0,0,0.2);
    }
    #title, #about-header {
        border: 2px solid #013369;
        padding: 10px;
        border-radius: 10px;
        background-color: #0055A2;
        color: #FFFFFF;
    }

    .stDataFrame {
        width: 100%;
        margin: auto;
    }
    .stTextInput > div {
        flex-direction: column;
    }
    .stTextInput label {
        color: #0055A2;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'email' not in st.session_state:
    st.session_state['email'] = ''
if 'id_token' not in st.session_state:
    st.session_state['id_token'] = ''
if 'menu' not in st.session_state:
    st.session_state['menu'] = 'Home'

def main():
    st.title('Dataset Generator')

    st.markdown('<div style="margin-top: 20px;"></div>', unsafe_allow_html=True)

    # Login/Sign Off and Signup section
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.session_state['logged_in']:
            st.markdown(f"**Welcome, {st.session_state['email']}!**", unsafe_allow_html=True)
            if st.button("Sign Off", key="sign_off"):
                st.session_state['logged_in'] = False
                st.session_state['email'] = ''
                st.session_state['id_token'] = ''
                st.session_state['menu'] = 'Home'
        else:
            col_login, col_signup = st.columns(2)
            with col_login:
                if st.button("Login", key="homepage_login"):
                    st.session_state['menu'] = "Login"
            with col_signup:
                if st.button("Signup", key="homepage_signup"):
                    st.session_state['menu'] = "Signup"

    # Handle the login and signup processes
    if not st.session_state['logged_in']:
        if st.session_state['menu'] == "Login":
            st.subheader("Login")
            email = st.text_input("Email")
            password = st.text_input("Password", type='password')
            if st.button("Login", key="login_page_login"):
                response = authenticate_user(email, password)
                if response:
                    st.session_state['logged_in'] = True
                    st.session_state['email'] = email
                    st.session_state['id_token'] = response['AuthenticationResult']['IdToken']
                    st.success(f'Welcome, {email}!')
                else:
                    st.error("Invalid email or password")
        elif st.session_state['menu'] == "Signup":
            st.subheader("Create New Account")
            new_email = st.text_input("Email", key="new_email")
            new_password = st.text_input("Choose Password", type='password', key="new_password")
            confirm_password = st.text_input("Confirm Password", type='password', key="confirm_password")
            if st.button("Signup", key="signup_page_signup"):
                if new_password != confirm_password:
                    st.error("Passwords do not match")
                else:
                    response = signup_user(new_email, new_password)
                    if response:
                        st.success(f"Account created successfully for {new_email}")
                    else:
                        st.error("Failed to create account")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.header('Fill Out the Form')
        st.write('Answer five inputs below to specify your dataset needs.')
    with col2:
        st.header('Search the Dataset')
        st.write('Our program then identifies or create your dataset.')
    with col3:
        st.header('Download Dataset')
        st.write('You will receive a downloadable custom dataset link.')

    with st.form(key='dataset_form'):
        algorithm = st.selectbox('Machine Learning Task:', ['', 'Random Forest', 'Linear Regression', 'K-nearest Neighbours'])
        features = st.selectbox('Number of Features:', ['', 'less than 10', '10 or more'])
        instances = st.selectbox('Number of Instances:', ['', 'less than 500', '500 or more'])
        topic = st.selectbox('Dataset Topic:', ['', 'Health', 'Finance', 'Education', 'Technology', 'Entertainment'])
        cleanliness = st.selectbox('Data Cleanliness:', ['', 'Clean', 'Unclean'])
        submit_button = st.form_submit_button('Generate Dataset')

    if submit_button:
        if algorithm and features and instances and topic:
            metadata = fetch_dataset_metadata(algorithm, features, instances, topic, cleanliness)
            if metadata:
                selected_metadata = metadata[0]  # Assuming one matching dataset is found
                dataset_link = generate_presigned_url('capstonedatasets', selected_metadata['S3ObjectKey'])
                if dataset_link:
                    # Display metadata
                    st.markdown(f"**Dataset Name:** {selected_metadata['Dataset Name']}")
                    st.markdown(f"**Machine Learning Task:** {selected_metadata['Machine Learning Task']}")
                    st.markdown(f"**Download Link:** [Download Dataset]({dataset_link})")
                    st.markdown(f"**Number of Features:** {selected_metadata['Number of Features']}")
                    st.markdown(f"**Number of Instances:** {selected_metadata['Number of Instances']}")
                    st.markdown(f"**Size in KB:** {selected_metadata['Size in KB']}")
                    st.markdown(f"**Source Link:** [Source]({selected_metadata['Source Link']})")
                    st.markdown(f"**Target Variable:** {selected_metadata['Target Variable']}")
                    st.markdown(f"**Topic:** {selected_metadata['Topic']}")

                    # Display dataset preview (first 50 rows)
                    dataset_preview = pd.read_csv(dataset_link)
                    if cleanliness == 'Unclean':
                        dataset_preview = make_dataset_unclean(dataset_preview)
                    st.dataframe(dataset_preview.head(50))
                else:
                    st.error("Failed to generate a download link. Please try again.")
            else:
                st.error("No matching dataset found. Please adjust your selection criteria.")
        else:
            st.error("Please fill in all the fields.")

    st.subheader('About')
    st.write('Our platform is designed to assist students and researchers in finding datasets for their machine learning experiments. By streamlining the process with powerful search functionalities, we aim to provide accurate and meaningful results.')

if __name__ == '__main__':
    main()
