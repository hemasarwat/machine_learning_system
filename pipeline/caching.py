"""
Helpers for configuring SageMaker step caching.
"""

from sagemaker.workflow.steps import CacheConfig  # type: ignore[import]

# this caching is being used in the pipeline to avoid re-running steps,
# that have already been executed with the same input parameters, and this when the input data is not changed
cache_config = CacheConfig(enable_caching=True, expire_after="15d") 

                                                                    