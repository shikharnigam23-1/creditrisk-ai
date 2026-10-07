"""
Data loading, cleaning, splitting, and preprocessing
for the Credit Risk AI project.
"""

from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------------------------------------------------------
# 1. Define the location of the dataset
# ---------------------------------------------------------

# Path(__file__) gives us the location of this file.
# .parent gives us the src folder.
# .parent.parent gives us the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Build the complete path to the CSV dataset.
DATA_PATH = PROJECT_ROOT / "data" / "credit_risk_dataset.csv"


# ---------------------------------------------------------
# 2. Load the dataset
# ---------------------------------------------------------

def load_data():
    """
    Load the credit risk dataset from the data directory.
    """

    # Read the CSV file into a pandas DataFrame.
    df = pd.read_csv(DATA_PATH)

    return df


# ---------------------------------------------------------
# 3. Clean obvious outliers
# ---------------------------------------------------------

def clean_data(df):
    """
    Remove clearly invalid age and employment-length values.
    """

    # Remove records where age is greater than 100.
    df = df[df["person_age"] <= 100].copy()

    # Remove records where employment length is greater than 60 years.
    df = df[df["person_emp_length"] <= 60].copy()

    return df


# ---------------------------------------------------------
# 4. Separate features and target
# ---------------------------------------------------------

def split_features_target(df):
    """
    Separate the input features from the target variable.
    """

    # loan_status is the target we want to predict.
    X = df.drop(columns=["loan_status"])

    # y contains the target variable.
    y = df["loan_status"]

    return X, y


# ---------------------------------------------------------
# 5. Create the preprocessing pipeline
# ---------------------------------------------------------

def create_preprocessor(X):
    """
    Create preprocessing pipelines for numeric and categorical columns.
    """

    # Identify numeric columns automatically.
    numeric_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    # Identify categorical columns automatically.
    categorical_features = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    # Numeric preprocessing:
    # 1. Replace missing values with the median.
    # 2. Standardize the numeric values.
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Categorical preprocessing:
    # 1. Replace missing values with the most frequent value.
    # 2. Convert categories into numerical one-hot encoded columns.
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    # Combine the numeric and categorical pipelines.
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )

    return preprocessor


# ---------------------------------------------------------
# 6. Main execution
# ---------------------------------------------------------

def main():
    """
    Load, clean, split, and create the preprocessing pipeline.
    """

    # Load the raw dataset.
    df = load_data()

    print(f"Original dataset shape: {df.shape}")

    # Clean invalid outlier values.
    df = clean_data(df)

    print(f"Cleaned dataset shape: {df.shape}")

    # Separate features and target.
    X, y = split_features_target(df)

    # Create a stratified 80/20 train-test split.
    # Stratification ensures that the proportion of loan_status
    # classes remains approximately the same in both datasets.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # Create the preprocessing pipeline using the training features.
    preprocessor = create_preprocessor(X_train)

    # Print the resulting shapes so we can verify everything works.
    print(f"Train features shape: {X_train.shape}")
    print(f"Test features shape: {X_test.shape}")
    print(f"Train target shape: {y_train.shape}")
    print(f"Test target shape: {y_test.shape}")

    # Print the preprocessing pipeline.
    print("\nPreprocessing pipeline:")
    print(preprocessor)


# Run main() when this file is executed directly.
if __name__ == "__main__":
    main()