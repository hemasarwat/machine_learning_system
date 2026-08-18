from sagemaker.sklearn.processing import SKLearnProcessor  # type: ignore[import]

preprocessor = SKLearnProcessor(
    base_job_name="sklearn-preprocess",
    framework_version="1.2-1",
    instance_type="ml.m5.xlarge",
    instance_count=1,
    role=role,
    sagemaker_session=config["session"]
)