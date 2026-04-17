import os
import tarfile
import tempfile
import numpy as np
import joblib
from pathlib import Path
import pandas as pd

from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

def preprocess_data(base_directory):
    """
    This functions Load and applies
    the preprocessing pipeline to the data
    and saves the preprocessed data to disk.
    """
    # Load the data
    data = _read_data(base_directory)

    # Transfoming Data 
    # data = _transform_data(data)



def _transform_data(data):
    """
    This function transforms the data
    by applying the feature engineering steps,
    will return the transformed data.
    """
    numerical_features = make_pipeline(SimpleImputer(strategy = "median"), 
                                       StandardScaler())
    categorical_features = make_pipeline(SimpleImputer(strategy = "most_frequent"),
                                         OneHotEncoder(handle_unknown = "ignore"))
    preprocessor = ColumnTransformer(transformers = [
        ("num", numerical_features, make_column_selector(dtype_include = np.number)),
        ("cat", categorical_features, make_column_selector(dtype_include = np.object))
    ])
    transformed_data = preprocessor.fit_transform(data)
    return transformed_data


def _read_data(base_directory):
    """
    This function reads the data
    and then concatenates them into a single DataFrame.
    """
    # Define the path
    data_path = Path(base_directory) / "input"

    # Read the csv files
    files = list(data_path.glob("*.csv"))
    dataframes = [pd.read_csv(file) for file in files]

    assert len(dataframes) > 0, "No CSV files found in the input directory."
    # Concatenate the dataframes
    data = pd.concat(dataframes, ignore_index = True)
    return data