"""Classification metadata and lazy data-loader exports."""

from .protocol import DATASET_SPECS

__all__ = ["DATASET_SPECS", "ClassificationRecord", "ClassificationWrapperDataset", "ClassificationDataBundle", "classification_data_provider"]


def __getattr__(name):
    if name in {"ClassificationRecord", "ClassificationWrapperDataset"}:
        from .dataset import ClassificationRecord, ClassificationWrapperDataset
        return {"ClassificationRecord": ClassificationRecord, "ClassificationWrapperDataset": ClassificationWrapperDataset}[name]
    if name in {"ClassificationDataBundle", "classification_data_provider"}:
        from .provider import ClassificationDataBundle, classification_data_provider
        return {"ClassificationDataBundle": ClassificationDataBundle, "classification_data_provider": classification_data_provider}[name]
    raise AttributeError(name)
