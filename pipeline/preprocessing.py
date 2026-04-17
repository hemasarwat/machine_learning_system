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
    df_train, df_val, df_test = _split_data(df)

    # _save_train_baseline(df_train, file_path)
    # _save_test_baseline(df_test, file_path)

    y_train = target_transformer.fit_transform(
        np.array(df_train.species.values).reshape(-1, 1)
    )
    y_val = target_transformer.transform(
        np.array(df_val.species.values).reshape(-1, 1)
    )
    y_test = target_transformer.transform(
        np.array(df_test.species.values).reshape(-1, 1)
    )

    df_train = df_train.drop("species", axis=1)
    df_val = df_val.drop("species", axis=1)
    df_test = df_test.drop("species", axis=1)

    X_train = feature_transformer.fit_transform(df_train)
    X_val = feature_transformer.transform(df_val)
    X_test = feature_transformer.transform(df_test)

    _save_split_data(
        file_path,
        X_train, 
        y_train, 
        X_val, 
        y_val, 
        X_test, 
        y_test)
    
    _save_model(file_path, feature_transformer, target_transformer)


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

def _split_data(df,random_state=42):
    """
    This function splits the data into train, validation and test sets.
    """
    df_train, temp = train_test_split(df, test_size=0.2, random_state=random_state)
    df_val, df_test = train_test_split(temp, test_size=0.5, random_state=random_state)

    return df_train, df_val, df_test

def _save_split_data(
        base_directory, X_train, y_train, X_val, y_val, X_test, y_test
    ):
    """
    This function concatenates the transformed features and target variable
    for each split and saves them as csv files
    """
    train = np.concatenate((X_train, y_train), axis=1)
    val = np.concatenate((X_val, y_val), axis=1)
    test = np.concatenate((X_test, y_test), axis=1)

    train_path = Path(base_directory) / "train"   # Here we create the path that will contain the train, val and test csv files.
    val_path = Path(base_directory) / "val"
    test_path = Path(base_directory) / "test"

    train_path.mkdir(parents=True, exist_ok=True) # Here we create the dir that will contain the train, val and test csv files.
    val_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(train).to_csv(train_path / "train.csv", index=False) # Here we save the train, val and test csv files in the corresponding directories.
    pd.DataFrame(val).to_csv(val_path / "val.csv", index=False)
    pd.DataFrame(test).to_csv(test_path / "test.csv", index=False)
 

def _save_model(base_directory, feature_transformer, target_transformer):
    """
    This functions creates a model.tar.gz file that
    contains the feature and target transformers of the data
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        joblib.dump(feature_transformer, os.path.join(temp_dir, "feature_transformer.joblib"))
        joblib.dump(target_transformer, os.path.join(temp_dir, "target_transformer.joblib"))

        model_path = Path(base_directory) / "model"
        model_path.mkdir(parents= True, exist_ok = True)

        with tarfile.open(model_path / "model.tar.gz", "w:gz") as tar:
            tar.add(os.path.join(temp_dir, "feature_transformer.joblib"), arcname="feature_transformer.joblib")
            tar.add(os.path.join(temp_dir, "target_transformer.joblib"), arcname="target_transformer.joblib")


if __name__ == "__main__":
    preprocess_data("data")