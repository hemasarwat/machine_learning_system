import tarfile
import tempfile
import joblib

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
    _save_model(base_directory, target_preprocessor_pipeline, feature_preprocessor) # Save the preprocessing pipelines for later use.


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
            ("num", numerical_features, make_column_selector(dtype_include=np.number)), # type: ignore
            ("cat", categorical_features, make_column_selector(dtype_include=object)) # type: ignore
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
    train_data = np.concatenate((X_train, y_train.reshape(-1, 1)), axis=1)
    val_data = np.concatenate((X_val, y_val.reshape(-1, 1)), axis=1)
    test_data = np.concatenate((X_test, y_test.reshape(-1, 1)), axis=1)

    train_path = Path(base_directory) / "train"
    val_path = Path(base_directory) / "val"
    test_path = Path(base_directory) / "test"

    train_path.mkdir(parents=True, exist_ok=True)
    val_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(train_data).to_csv(train_path / "train.csv", index=False)
    pd.DataFrame(val_data).to_csv(val_path / "val.csv", index=False)
    pd.DataFrame(test_data).to_csv(test_path / "test.csv", index=False)


def target_preprocessor():
    """
    Build a separate preprocessor for the target variable.
    """
    return make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    )    


def _save_model(base_directory, target_preprocessor, feature_preprocessor):
    """
    Save the scikit-learn pipelines transforming,
    in the disk. This function turn it into tar.gz file
    and save it in the output directory.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        joblib.dump(target_preprocessor, Path(temp_dir) / "target_preprocessor.joblib")
        joblib.dump(feature_preprocessor, Path(temp_dir) / "feature_preprocessor.joblib")

        model_path = Path(base_directory) / "model"
        model_path.mkdir(parents=True, exist_ok=True)

        with tarfile.open(f"{(model_path / 'model.tar.gz').as_posix()}", "w:gz") as tar:
            tar.add(Path(temp_dir) / "target_preprocessor.joblib",  arcname="target_preprocessor.joblib")
            tar.add(Path(temp_dir) / "feature_preprocessor.joblib", arcname="feature_preprocessor.joblib")
        
                    
                    
        