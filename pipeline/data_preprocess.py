from sagemaker.sklearn.processing import SKLearnProcessor  # type: ignore[import]
from sagemaker.processing import ProcessingInput, ProcessingOutput  # type: ignore[import]
from sagemaker.workflow.steps import ProcessingStep #type: ignore[import]
from pipeline import preprocessing  
from pipeline.pipeline_conf import dataset_location
ROLE = "arn:aws:iam::017117988361:role/service-role/AmazonSageMaker-ExecutionRole-20260805T041935 "   
preprocessor = SKLearnProcessor(
    base_job_name="sklearn-preprocess",
    framework_version="1.2-1",
    instance_type="ml.m5.xlarge",
    instance_count=1,
    role=ROLE
    )


preprocessing_step = ProcessingStep(
    name="Preprocessing_data",
    step_args=preprocessor.run(
        code=preprocessing.preprocess_data,
        inputs=[
            ProcessingInput(
                source=dataset_location,  # where we will get the data
                destination="/opt/ml/processing/input"  # where we will locate it in s3
            )
        ],
        # we need to make outputs destination for each split of the data,
        # so we will use the dataset_location parameter to specify the location of the data we want to process.
        outputs=[
            ProcessingOutput(
                output_name="train",
                source="/opt/ml/processing/train",
                destination=dataset_location + "train"
            ),
            ProcessingOutput(
                output_name="validation",
                source="/opt/ml/processing/val",
                destination=dataset_location + "val"
            ),
            ProcessingOutput(
                output_name="test",
                source="/opt/ml/processing/test",
                destination=dataset_location + "test"
            ),
            ProcessingOutput(
                output_name="model",
                source="/opt/ml/processing/model",
                destination=dataset_location + "model"
            ),
            ProcessingOutput(
                output_name="train_baseline",
                source="/opt/ml/processing/train_baseline",
                destination=dataset_location + "train_baseline"
            ),
            ProcessingOutput(
                output_name="val_baseline",
                source="/opt/ml/processing/val_baseline",
                destination=dataset_location + "val_baseline"
            ),
            ProcessingOutput(
                output_name="test_baseline",
                source="/opt/ml/processing/test_baseline",
                destination=dataset_location + "test_baseline"
            )
        ]
    ),
    cache_config=cache_config,
)