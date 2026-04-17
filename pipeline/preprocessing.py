import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


def preprocess_data(base_directory):
    """
    Load data, split it, preprocess features, and save processed splits.
    """
    data = _read_data(base_directory)

    df_train, df_val, df_test = _split_data(data)

    X_train, y_train = df_train.iloc[:, :-1], df_train.iloc[:, -1]
    X_val, y_val = df_val.iloc[:, :-1], df_val.iloc[:, -1]
    X_test, y_test = df_test.iloc[:, :-1], df_test.iloc[:, -1]

    feature_preprocessor = _build_preprocessor()
    target_preprocessor_pipeline = target_preprocessor()

    X_train = feature_preprocessor.fit_transform(X_train)
    X_val = feature_preprocessor.transform(X_val)
    X_test = feature_preprocessor.transform(X_test)

    y_train = target_preprocessor_pipeline.fit_transform(y_train)
    y_val = target_preprocessor_pipeline.transform(y_val)
    y_test = target_preprocessor_pipeline.transform(y_test)

    _save_split_data(X_train, y_train, X_val, y_val, X_test, y_test, base_directory)


def _build_preprocessor():
    """
    Build preprocessing pipeline for numerical and categorical features.
    """
    numerical_features = make_pipeline(
        SimpleImputer(strategy="mean"),
        StandardScaler()
    )

    categorical_features = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_features, make_column_selector(dtype_include=np.number)),
            ("cat", categorical_features, make_column_selector(dtype_include=object))
        ]
    )

    return preprocessor


def _read_data(base_directory):
    """
    Read CSV files from input directory and concatenate them.
    """
    data_path = Path(base_directory) / "input"
    files = list(data_path.glob("*.csv"))

    assert len(files) > 0, "No CSV files found in the input directory."

    dataframes = [pd.read_csv(file) for file in files]
    data = pd.concat(dataframes, ignore_index=True)
    return data


def _split_data(data):
    """
    Split data into train, validation, and test sets.
    """
    df_train, temp = train_test_split(data, test_size=0.3, random_state=43)
    df_val, df_test = train_test_split(temp, test_size=0.5, random_state=43)

    return df_train, df_val, df_test


def _save_split_data(X_train, y_train, X_val, y_val, X_test, y_test, base_directory):
    """
    Concatenate features and target and save them as CSV files.
    """
    train_data = np.concatenate([X_train, y_train], axis=1)
    val_data = np.concatenate([X_val, y_val], axis=1)
    test_data = np.concatenate([X_test, y_test], axis=1)

    output_dir = Path(base_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(train_data).to_csv(output_dir / "train.csv", index=False)
    pd.DataFrame(val_data).to_csv(output_dir / "val.csv", index=False)
    pd.DataFrame(test_data).to_csv(output_dir / "test.csv", index=False)


def target_preprocessor():
    """
    Build a separate preprocessor for the target variable.
    """
    return make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    )    