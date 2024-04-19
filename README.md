# DataGenerator
A web application that generates a customised dataset based on the user needs

<!-- **Introduction** -->

Welcome to Dataset Generator! Before you run the application, you need to set up a few things. This document will guide you through the process of configuring your environment with necessary credentials and settings.
Prerequisites

    Python 3.12 installed
    Git (for cloning the repository)

Steps to Set Up Your Environment
1. Clone the Repository

First, clone the repository to your local machine:
```bash
git clone https://github.com/cuahunt7/DatasetGenerator.git
cd DatasetGenerator
```

2. Set Up Your Credentials

Since sensitive data such as API keys and credentials should not be tracked by Git, you need to set up your `.env` file manually:

**Create Your .env File**

* Copy the Sample File: A sample environment file named `.env.sample` is provided in the repository. Start by copying this file and renaming it to `.env`:
    ```bash
    cp .env.sample .env
    ```

* Edit the .`env` File: Open the `.env` file in a text editor of your choice:
    ```bash
    code .env
    ```

* Replace the placeholder values with your actual credentials.
    * For example:
        AWS_ACCESS_KEY_ID=your_aws_access_key_id_here
        AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key_here
        AWS_DEFAULT_REGION=your_aws_region_here
* Save and close the file.

3. Install Dependencies

Install the required Python dependencies:
```bash
pip3 install -r requirements.txt
```

4. Run the Application

Now, you are ready to run the application:
```bash
python3 main.py
```
