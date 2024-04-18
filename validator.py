from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score,confusion_matrix, classification_report
from sklearn.impute import SimpleImputer, SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
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




def validate_knn(X_train, X_test, y_train, y_test):
  # Apply PCA to reduce the dimensions to X principal components
    components = len(X_train.columns)
    pca = PCA(n_components=3 if components > 2 else components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    # print("Explained variance ratio by top 3 components:", pca.explained_variance_ratio_)

    # Hyperparameter tuning for KNN on PCA-reduced dataset
    params = {
        'n_neighbors': range(1, 11, 2), 
        'weights': ['uniform', 'distance'], 
        'metric': ['euclidean', 'manhattan']
    }
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn, params, cv=10, scoring='accuracy')
    grid_search.fit(X_train_pca, y_train)
    
    # Best KNN model
    best_knn = grid_search.best_estimator_
    # print("Best KNN Parameters:", grid_search.best_params_)

    # Cross-validation to evaluate model
    cv_scores = cross_val_score(best_knn, X_train_pca, y_train, cv=10, scoring='accuracy')
    print("Average CV Accuracy with PCA:", np.mean(cv_scores))

    # Final evaluation on the test set
    y_pred = best_knn.predict(X_test_pca)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    # Print classification report and confusion matrix
    # print(classification_report(y_test, y_pred))
    # print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    print("Test Set Metrics with PCA:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

    if accuracy > 0.80 and precision > 0.80 and recall > 0.80 and f1 > 0.80:
        print("Dataset provided is suitable for KNN")
        return True
    else:
        print("Dataset may not be suitable for KNN. Consider reviewing the data.")
        return False
    

def dynamic_preprocess(data, target_variable=None,correlation_threshold=0.85):
    # Identifying numeric and categorical columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()

    # Remove target variable from categorical if specified and binary
    if target_variable and target_variable in categorical_cols and data[target_variable].nunique() == 2:
        categorical_cols.remove(target_variable)

    if target_variable and target_variable in numeric_cols:
        numeric_cols.remove(target_variable)

    # Initial processing lists
    processed_columns = []
    data_processed_list = []

    # Processing numeric columns
    for col in numeric_cols:
        skewness = data[col].skew()
        strategy = 'mean' if abs(skewness) < 1 else 'median'
        # print(f"Column: {col}, Skewness: {skewness:.2f}, Imputation strategy: {strategy}")

        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy=strategy)),
            ('scaler', MinMaxScaler())
        ])

        processed_data = numeric_transformer.fit_transform(data[[col]])
        data_processed_list.append(processed_data)
        processed_columns.append(col)

    # Processing categorical columns with imputation and one-hot encoding
    if categorical_cols:
        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        cat_data = data[categorical_cols]
        processed_data = categorical_transformer.fit_transform(cat_data)
        data_processed_list.append(processed_data)
        processed_columns.extend(categorical_transformer.named_steps['onehot'].get_feature_names_out(categorical_cols))

    # Concatenate all processed columns
    data_processed = np.hstack(data_processed_list)
    processed_df = pd.DataFrame(data_processed, columns=processed_columns)

    # Remove highly correlated columns
    corr_matrix = processed_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))
    to_drop = [column for column in upper.columns if any(upper[column] > correlation_threshold)]
    processed_df.drop(columns=to_drop, inplace=True)

    if target_variable and data[target_variable].nunique() == 2:
        le = LabelEncoder()
        processed_df[target_variable] = le.fit_transform(data[target_variable])
    else:
        processed_df[target_variable] = data[target_variable]

    return processed_df

def validator(data, algorithm_index, target_variable):
    constant_columns = [col for col in data.columns if data[col].nunique() == 1]
    data.drop(constant_columns, axis=1, inplace=True)
    data_processed = dynamic_preprocess(data, target_variable)

    X = data_processed.drop(target_variable, axis=1)
    y = data_processed[target_variable]
 
    stratify_option = y if data_processed[target_variable].nunique() == 2 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=stratify_option, random_state=42)

    if algorithm_index == 1:
        return validate_linear_regression(X_train, X_test, y_train, y_test)
    elif algorithm_index == 2:
        return validate_random_forest(X_train, X_test, y_train, y_test)
    elif algorithm_index == 3:
        return validate_knn(X_train, X_test, y_train, y_test)