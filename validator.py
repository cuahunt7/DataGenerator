from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score,confusion_matrix, classification_report
import pandas as pd
import numpy as np

def user_select_target(data):
    print("Select the number of the target variable:")
    for i, column in enumerate(data.columns, start=1):
        print(f"{i}. {column}")
    
    while True:
        try:
            target_index = int(input("Enter the number corresponding to the target variable: "))
            if 1 <= target_index <= len(data.columns):
                return data.columns[target_index - 1]
            else:
                print("Error: Please enter a valid number from the list.")
        except ValueError:
            print("Error: Please enter a valid integer.")

def select_algorithm():
    print("Select the Machine Learning Algorithm Suitable for this dataset:")
    print("1. Linear Regression")
    print("2. Random Forest")
    print("3. K-nearest neighbors (KNN)")

    while True:
        try:
            algo_index = int(input("Enter the number corresponding to the algorithm: "))
            algorithms = {1: "Linear Regression", 2: "Random Forest", 3: "K-nearest neighbors (KNN)"}
            if algo_index in algorithms:
                print(f"You have selected {algorithms[algo_index]}")
                return algo_index
            else:
                print("Error: Please only enter a valid choice.")
        except ValueError:
            print("Error: Please enter a valid integer.")

def validate_linear_regression(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    normalised_rmse = rmse / (max(y_test) - min(y_test))

    print("Linear Regression Metrics:")
    print(f"RMSE: {round(rmse, 2)}")
    print(f"Normalised RMSE: {round(normalised_rmse, 2)}")
    print(f"R-squared: {round(r2, 2)}")

    if normalised_rmse < 0.15 and r2 > 0.7:
        print("Dataset provided is suitable for Linear Regression.")
        return True
    else:
        print("Dataset may not be suitable for Linear Regression. Consider reviewing the data.")

def validate_random_forest(X_train, X_test, y_train, y_test):
    # Define the parameter grid for hyperparameter tuning
    param_grid = {
        'n_estimators': [10, 50, 100],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Initialize the RandomForestClassifier
    rf = RandomForestClassifier(random_state=42)
    
    # Perform GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', verbose=0)
    grid_search.fit(X_train, y_train)
    
    # Retrieve the best model
    best_rf = grid_search.best_estimator_
    
    # Print the best parameters
    print("Best parameters found: ", grid_search.best_params_)
    
    # Perform cross-validation and calculate the average performance metrics
    cv_scores = cross_val_score(best_rf, X_train, y_train, cv=5)
    print("Average CV Accuracy: {:.2f}".format(cv_scores.mean()))

    # Train the model on the entire training data with the best parameters
    best_rf.fit(X_train, y_train)
    
    # Predict on the test data
    y_pred = best_rf.predict(X_test)
    
    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("Random Forest Metrics:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    
    # Print classification report and confusion matrix
    # print(classification_report(y_test, y_pred))
    # print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    
    # Check the performance against the thresholds
    if accuracy > 0.80 and precision > 0.80 and recall > 0.80 and f1 > 0.80:
        print("Dataset provided is suitable for Random Forest.")
        return True
    else:
        print("Dataset may not be suitable for Random Forest. Consider reviewing the data.")
        return False



def validator(data, algorithm_index, target_variable):
    constant_columns = [col for col in data.columns if data[col].nunique() == 1]
    data.drop(constant_columns, axis=1, inplace=True)

    data_processed = data

    X = data_processed.drop(target_variable, axis=1)
    y = data_processed[target_variable]
 
    stratify_option = y if data_processed[target_variable].nunique() == 2 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=stratify_option, random_state=42)

    if algorithm_index == 1:
        return validate_linear_regression(X_train, X_test, y_train, y_test)
    elif algorithm_index == 2:
        return validate_random_forest(X_train, X_test, y_train, y_test)

