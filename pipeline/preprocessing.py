import os
import numpy as np
import joblib
from pathlib import Path
import pandas as pd

from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

def preprocess_data(file_path):
    """
    This function loads the data,
    splits it and transform it
    """
    df = _read_data_from_csv(file_path)
    target_transformer = ColumnTransformer(
        transformers= [('species', OrdinalEncoder(), [0])]
    )
    numeric_transformer = make_pipeline(
        SimpleImputer(strategy='mean'),
        StandardScaler()
    )
    categorical_transformer = make_pipeline(
        SimpleImputer(strategy='most_frequent'),
        OneHotEncoder(handle_unknown='ignore')
    )

    feature_transformer = ColumnTransformer(
        transformers=[
            (
                "numeric",
                 numeric_transformer,
                 make_column_selector(dtype_exclude="object")
            ),
                ("categorical",
                  categorical_transformer,
                  ["island"])
        ]
    )


def _read_data_from_csv(file_path):
    """
    This function reads csv file and concatenates all of them into a single dataframe.
    """
    input_files = Path(file_path) / "input"
    file_samples = [file for file in input_files.glob("*.csv")]

    if len(file_samples) == 0:
        raise ValueError(f"No csv files found in the directory: {file_path}")
     
    raw_dataframes = [pd.read_csv(file) for file in file_samples] # Read each csv file into a dataframe and store them in a list
    dataframe = pd.concat(raw_dataframes, ignore_index=True) # Concatenate all dataframes into a single dataframe, ignoring the index to create a new one
    

    return dataframe.sample(frac=1, random_state=42) # Shuffle the dataframe 


 


