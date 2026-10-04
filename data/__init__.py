"""Dataset configuration and lazy loader exports."""

from .config import DataConfig

__all__ = ["DataConfig", "data_provider"]


def __getattr__(name):
    if name == "data_provider":
        from .provider import data_provider
        return data_provider
    raise AttributeError(name)
