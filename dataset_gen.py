# Import necessary modules
import numpy as np
import pandas as pd
from datetime import datetime
from database import upload_csv_to_s3
import validator


def linear_regression_dataset(column_counts=9, row_counts=1000):
    features = []
    
    # Generate features dynamically based on column_counts
    for i in range(column_counts - 1):  # Generate one less than column count for the features
        if i < 3:
            # First three are important features
            if i == 0:
                features.append(np.random.normal(20, 5, row_counts))  # Feature 1
            elif i == 1:
                features.append(np.random.normal(50, 10, row_counts))  # Feature 2
            else:
                features.append(np.random.uniform(0, 1, row_counts))  # Feature 3
        else:
            # Additional noise features
            features.append(np.random.normal(0, 0.1, row_counts))  # Noise features

    # Create a target variable with a linear relationship
    y = 2 * features[0] + 0.5 * features[1] + 10 * features[2] + np.random.normal(0, 2, row_counts)
    
    # Combine all features into a single DataFrame and include the target
    column_names = [f'Feature_{i+1}' for i in range(column_counts - 1)] + ['Target']
    X = pd.DataFrame(np.column_stack(features + [y]), columns=column_names)

    return X

# ***************************************************************
def random_forest_dataset(column_counts=9, row_counts=1000):
    features = []
    
    # Generate features dynamically based on column_counts
    for i in range(column_counts - 1):  # Reserve the last column for the target
        if i == 0:
            features.append(np.random.normal(0, 1, row_counts))  # Continuous feature
        elif i == 1:
            features.append(np.random.normal(5, 2, row_counts))  # Continuous feature
        elif i == 2:
            features.append(np.random.randint(0, 3, row_counts))  # Categorical feature
        elif i == 3:
            features.append(features[0] * features[1])  # Interaction feature
        elif i == 4:
            features.append(np.random.choice([0, 1], size=row_counts, p=[0.7, 0.3]))  # Binary feature
        else:
            features.append(np.random.normal(0, 1, row_counts))  # Additional noise features

    # Create a target variable with non-linear relationship
    y = (features[0]**2 + np.sin(features[1]) + features[2] > np.median(features[0]**2 + np.sin(features[1]) + features[2])).astype(int)
    feature_names = [f'Feature_{i+1}' for i in range(column_counts - 1)]
    X = pd.DataFrame(np.column_stack(features), columns=feature_names)
    X['Target'] = y

    return X
# **************************

def knn_dataset(column_counts=9, row_counts=1000):
    features = []
    
    # Generate features dynamically based on column_counts
    for i in range(column_counts - 1):  # Reserve the last column for the target
        if i == 0:
            features.append(np.random.normal(0, 1, row_counts))  # Continuous feature
        elif i == 1:
            features.append(np.sin(features[0]) + np.random.normal(0, 0.1, row_counts))  # Derived feature
        elif i == 2:
            features.append(features[0]**2 + np.random.normal(0, 1, row_counts))  # Non-linear feature
        elif i == 3:
            features.append(np.random.choice([0, 1, 2], size=row_counts))  # Categorical feature
        else:
            features.append(np.random.normal(0, 1, row_counts))  # Additional noise features

    # Create a target variable influenced by non-linear combinations
    y = (2*features[0] - features[1] + features[2] + 5*np.where(features[3] == 2, 1, 0) > 1).astype(int)

    # Combine all features into a DataFrame
    column_names = [f'Feature_{i+1}' for i in range(column_counts - 1)]
    X = pd.DataFrame(np.column_stack(features), columns=column_names)
    X['Target'] = y

    return X
def dataset_generator(algorithm, size, features):
    # Determine the number of rows based on size
    row_counts = 499 if size == "less than 500" else 501
    
    # Determine the number of columns based on features
    column_counts = 9 if features == 'less than 10' else 10

    # Generate dataset based on the specified algorithm
    if algorithm == 'Linear Regression':
        return linear_regression_dataset(column_counts=column_counts, row_counts=row_counts)
    elif algorithm == 'Random Forest':
        return random_forest_dataset(column_counts=column_counts, row_counts=row_counts)
    elif algorithm == 'KNN':
        return knn_dataset(column_counts=column_counts, row_counts=row_counts)
    else:
        raise ValueError(f"Unsupported algorithm specified: {algorithm}")


def main(algorithm, size, features):
    print("Validating the Synthetic Dataset")
    while True:
        data = dataset_generator(algorithm, size, features)
        mode = "auto"
        path = f"Synthetic/Data/Synthetic_Dataset_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        target_variable = 'Target'
        algorithm_index = data.columns.get_loc(target_variable)
        algorithm_name = algorithm.lower().replace(" ", "-")
        algorithm = {1: "linear-regression", 2: "random-forest", 3: "k-nearest-neighbors"}
        algorithm_index = next((key for key, value in algorithm.items() if value == algorithm_name), None)
        folder_route = algorithm_name + "/"
        # clean_df, valid = validator(data, algorithm_index, target_variable)
        clean_df, valid = validator.validator(data, algorithm_index, target_variable)
        if valid:
            object_key = upload_csv_to_s3(clean_df, "capstonedatasets", folder_route, path)
            if object_key:
                print(f"*** Dataset uploaded successfully. ***")
                return object_key  # Return the object key
            else:
                print("*** Generating New Synthetic Dataset ***")
        else:
            print("*** Dataset validation failed ***")
