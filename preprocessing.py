from metaflow.flowspec import FlowSpec
from metaflow import step
import time

class ForeachFlow(FlowSpec):
    @step
    def start(self):
        self.data = ["Apple", "Orange"]
        self.next(self.process, foreach="data")

    @step
    def process(self):
        print("Processing:", self.input)
        self.fruit = self.input
        self.score = len(self.fruit)
        self.next(self.join)

    @step
    def join(self, inputs):
        print("Choosing the best fruit")
        self.best = max(inputs, key=lambda x: x.score).fruit
        print("Best fruit:", self.best)
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    ForeachFlow()
