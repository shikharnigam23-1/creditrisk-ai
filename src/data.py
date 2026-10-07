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

# Settings used for the train-test split (kept fixed so results are reproducible).
TEST_SIZE = 0.20
RANDOM_STATE = 42


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
    Rows with MISSING employment length are kept; they are
    filled later by the median imputer in the pipeline.
    """

    # Remove records where age is greater than 100.
    df = df[df["person_age"] <= 100].copy()

    # Remove records where employment length is greater than 60 years,
    # but keep rows where employment length is missing (NaN).
    # Note: in pandas, NaN <= 60 is False, so without .isna()
    # these rows would be silently deleted.
    df = df[
        df["person_emp_length"].isna() | (df["person_emp_length"] <= 60)
    ].copy()

    return df


# ---------------------------------------------------------
# 4. Separate features and target
# ---------------------------------------------------------

def split_features_target(df):
    """
    Separate the input features from the target variable.
    """

    # loan_status is the target we want to predict (1 = default, 0 = repaid).
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

    # Identify numeric columns automatically (all number types).
    numeric_features = X.select_dtypes(include="number").columns.tolist()

    # Identify categorical columns automatically.
    categorical_features = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    # Numeric preprocessing:
    # 1. Replace missing values with the median.
    # 2. Standardize the numeric values (mean 0, standard deviation 1).
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
# 6. Helper functions used by model.py and app.py
# ---------------------------------------------------------

def get_data():
    """
    Load -> clean -> split.
    Returns X_train, X_test, y_train, y_test.
    The same random_state is used every time, so the test set never changes.
    """

    df = clean_data(load_data())
    X, y = split_features_target(df)

    # Stratified 80/20 split: keeps the default rate (~22%)
    # the same in both the training and the test set.
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def build_preprocessor():
    """
    Return a NEW, unfitted preprocessor.
    Each model gets its own fresh copy, so nothing leaks between models.
    """

    X_train, _, _, _ = get_data()
    return create_preprocessor(X_train)


# ---------------------------------------------------------
# 7. Main execution (quick check that everything works)
# ---------------------------------------------------------

def main():
    """
    Load, clean, split, and create the preprocessing pipeline.
    """

    # Load the raw dataset.
    df = load_data()
    print(f"Original dataset shape: {df.shape}")

    # Clean invalid outlier values.
    cleaned = clean_data(df)
    print(f"Cleaned dataset shape: {cleaned.shape}")
    print(f"Rows removed as outliers: {len(df) - len(cleaned)}")

    # Missing values that the pipeline will fill.
    print("\nMissing values per column (filled later by the imputer):")
    print(cleaned.isna().sum()[cleaned.isna().sum() > 0])

    # Create the stratified train-test split.
    X_train, X_test, y_train, y_test = get_data()

    print(f"\nTrain features shape: {X_train.shape}")
    print(f"Test features shape: {X_test.shape}")
    print(f"Default rate - train: {y_train.mean():.3f} | test: {y_test.mean():.3f}")

    # Create the preprocessing pipeline.
    preprocessor = build_preprocessor()

    # Fit on training data only, to show the final number of model inputs.
    X_train_processed = preprocessor.fit_transform(X_train)
    print(f"Processed train shape (after one-hot encoding): {X_train_processed.shape}")

    print("\nPreprocessing pipeline:")
    print(preprocessor)


# Run main() when this file is executed directly.
if __name__ == "__main__":
    main()