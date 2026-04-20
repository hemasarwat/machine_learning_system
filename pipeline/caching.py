"""
Helpers for configuring SageMaker step caching.
"""

from importlib import import_module

try:
    CacheConfig = import_module("sagemaker.mlops.workflow.steps").CacheConfig
except ModuleNotFoundError:
    try:
        CacheConfig = import_module("sagemaker.workflow.cache_config").CacheConfig
    except ModuleNotFoundError:
        class CacheConfig:
            """
            Minimal fallback shim used when SageMaker is not installed locally.
            """

            def __init__(self, enable_caching=False, expire_after=None, **kwargs):
                self.enable_caching = enable_caching
                self.expire_after = expire_after

            def __repr__(self):
                return (
                    "CacheConfig("
                    f"enable_caching={self.enable_caching!r}, "
                    f"expire_after={self.expire_after!r})"
                )


def get_cache_config():
    """
    Return the cache configuration used by the preprocessing step.
    """
    return CacheConfig(enable_caching=True, expire_after="15d")


def _get_cache_config():
    """
    Backward-compatible alias for older callers.
    """
    return get_cache_config()


__all__ = ["CacheConfig", "get_cache_config", "_get_cache_config"]
