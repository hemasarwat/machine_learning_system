from metaflow.flowspec import FlowSpec
from metaflow import step
class preprocess(FlowSpec):
    @step
    def start(self):
        print("hello")
        self.next(self.end)

    @step
    def end(self):
        print("hellllllooooo")


if __name__ == "__main__":
    preprocess()