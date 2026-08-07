"""
we can parametize a SageMaker pipeline to make it more flexible,
in this we will use a parameter to specify the location of the data we want to process.
"""

from sagemaker.workflow.parameters import ParameterString  # type: ignore[import]
from sagemaker.workflow.pipeline_definition_config import PipelineDefinitionConfig  # type: ignore[import]

