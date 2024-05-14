import streamlit as st
import pandas as pd
import hashlib

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


def send_to_aws_and_fetch_link(data):
    return "https://aws.s3/download/dataset_link_based_on_criteria"

def generate_fake_dataset(size):
    import numpy as np
    np.random.seed(0)
    data = np.random.rand(size, 5)
    return pd.DataFrame(data, columns=['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5'])

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def check_username(username, df):
    return username in df['username'].values

def verify_login(username, password, df):
    user = df[df['username'] == username]
    if not user.empty:
        return user.iloc[0]['password'] == hash_password(password)
    return False

def register_user(username, password, df):
    new_user = pd.DataFrame({'username': [username], 'password': [hash_password(password)]})
    df = pd.concat([df, new_user], ignore_index=True)
    df.to_csv('user_credentials.csv', index=False)

try:
    user_credentials = pd.read_csv('user_credentials.csv')
except FileNotFoundError:
    user_credentials = pd.DataFrame(columns=['username', 'password'])

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = ''
if 'menu' not in st.session_state:
    st.session_state['menu'] = 'Home'

def main():
    st.title('Dataset Generator')

    st.markdown('<div style="margin-top: 20px;"></div>', unsafe_allow_html=True)

    # Login/Sign Off and Signup section
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.session_state['logged_in']:
            st.markdown(f"**Welcome, {st.session_state['username']}!**", unsafe_allow_html=True)
            if st.button("Sign Off", key="sign_off"):
                st.session_state['logged_in'] = False
                st.session_state['username'] = ''
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
            username = st.text_input("Username")
            password = st.text_input("Password", type='password')
            if st.button("Login", key="login_page_login"):
                if verify_login(username, password, user_credentials):
                    st.session_state['logged_in'] = True
                    st.session_state['username'] = username
                    st.success(f'Welcome, {username}!')
                else:
                    st.error("Invalid username or password")
        elif st.session_state['menu'] == "Signup":
            st.subheader("Create New Account")
            new_username = st.text_input("Choose Username", key="new_username")
            new_password = st.text_input("Choose Password", type='password', key="new_password")
            confirm_password = st.text_input("Confirm Password", type='password', key="confirm_password")
            if st.button("Signup", key="signup_page_signup"):
                if new_password != confirm_password:
                    st.error("Passwords do not match")
                elif check_username(new_username, user_credentials):
                    st.error("Username already exists")
                else:
                    register_user(new_username, new_password, user_credentials)
                    st.success(f"Account created successfully for {new_username}")


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
        algorithm = st.selectbox('Algorithm type:', ['', 'Random Forest', 'Linear Regression', 'K-nearest Neighbours',])
        size = st.selectbox('Dataset size:', ['', 'less than 500', '500 or more',])
        features = st.selectbox('Number of features:', ['', 'less than 10', '10 or more',])
        topic = st.selectbox('Dataset topic:', ['', 'Health', 'Finance', 'Education', 'Technology', 'Entertainment'])
        cleanliness = st.selectbox('Cleanliness:', ['', 'Clean', 'Unclean'])

        # Disable the submit button if the user is not logged in
        submit_button = st.form_submit_button('Generate', disabled=not st.session_state['logged_in'])

    if submit_button:
        data = {'algorithm': algorithm, 'size': size, 'features': features, 'topic': topic, 'cleanliness': cleanliness}
        link = send_to_aws_and_fetch_link(data)
        st.markdown(f"Download your dataset [here]({link})")

        st.write('### Dataset Preview')
        dataset_preview = generate_fake_dataset(10)
        col1, col2, col3 = st.columns([1, 6, 1])
        with col2:
            st.dataframe(dataset_preview)

    st.subheader('About')
    st.write('Our platform is designed to assist students and researchers in finding datasets for their machine learning experiments. By streamlining the process with powerful search functionalities, we aim to provide accurate and meaningful results.')

if __name__ == '__main__':
    main()