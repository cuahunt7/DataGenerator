import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def user_select_target(data):
    """Prompt user to select the target variable from dataset columns."""
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
    """Prompt user to select a machine learning algorithm."""
    print("Select the Machine Learning Algorithm Suitable for this dataset:")
    print("1. Linear Regression")
    print("2. Random Forest")
    print("3. KNN")

    while True:
        try:
            algo_index = int(input("Enter the number corresponding to the algorithm: "))
            if 1 <= algo_index <= 3:
                algorithms = {1: "Linear Regression", 2: "Random Forest", 3: "KNN"}
                print(f"You have selected {algorithms[algo_index]}")
                return algo_index
            else:
                print("Error: Please only enter a number between 1 to 3")
        except ValueError:
            print("Error: Please enter a valid integer.")

def validator(file_path, algorithm_index, target_variable):
    """Validate dataset suitability for selected algorithm using specified metrics."""
    data = pd.read_csv(file_path)
    
    # Encode categorical variables
    X = pd.get_dummies(data.drop(target_variable, axis=1), drop_first=True)
    y = data[target_variable]

    # Ensure target variable is numeric
    try:
        y = pd.to_numeric(y)
    except ValueError:
        print("Error: Target variable contains non-numeric data that cannot be converted.")
        return
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if algorithm_index == 1:  # Linear Regression
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        normalised_rmse = rmse / (max(y) - min(y))

        print("--Linear Regression Metrics--")
        print(f"RMSE: {round(rmse, 2)}")
        print(f"Normalised RMSE: {round(normalised_rmse, 2)}")
        print(f"R-squared: {round(r2, 2)}")
        
        if normalised_rmse < 0.20 and r2 > 0.6:
            print("Dataset provided is suitable for Linear Regression.")
        else:
            print("Dataset may not be suitable for Linear Regression. Consider reviewing the data or choosing a different model.")

def main():
    """Main function to orchestrate the workflow for dataset validation."""
    print("Welcome to the Dataset Validator!")
    path = input("Please enter the path to your dataset: ")

    try:
        data = pd.read_csv(path)
    except Exception as e:
        print(f"An error occurred while loading the dataset: {e}")
        return

    print("\n" + "*" * 20 + "\n")
    target_variable = user_select_target(data)
    
    if target_variable is None:
        print("No valid target variable selected, exiting.")
        return

    print("\n" + "*" * 20 + "\n")
    algorithm_index = select_algorithm()

    print("\n" + "*" * 20 + "\n")
    validator(data, algorithm_index, target_variable)

if __name__ == "__main__":
    main()

