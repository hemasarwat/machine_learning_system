import pandas as pd
import numpy as np
from metaflow.flowspec import FlowSpec
from metaflow.decorators import step

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
        self.next(self.end)

    @step
    def end(self):
        print("Flow completed.")

if __name__ == '__main__':
    HhFlow()