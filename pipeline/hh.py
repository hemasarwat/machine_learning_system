import pandas as pd
import numpy as np
from metaflow.flowspec import FlowSpec
from metaflow.decorators import step
import mlflow
class HhFlow(FlowSpec):
    @step
    def start(self):
        self.data = pd.read_csv('/home/ibrahim/Desktop/python/Machine Learning projects/machine_learning_system/data/penguins.csv')
        self.next(self.eda)

    @step
    def eda(self):
        print(self.data.head())
        print(self.data.info())
        print(self.data.describe())
        self.next(self.log_metadata)
    @step
    def log_metadata(self):
        with mlflow.start_run():
            mlflow.log_param("rows", len(self.data))
            mlflow.log_metric("columns", len(self.data.columns))
        self.next(self.end)    

    @step
    def end(self):
        print("Flow completed.")

if __name__ == '__main__':
    HhFlow()