from __future__ import annotations

"""分类 DataLoader 入口。"""

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset

from .collate import ClassificationNormalizer, build_classification_collate
from .dataset import ClassificationWrapperDataset
from .protocol import DATASET_SPECS
from ..irregular_perturbation import apply_observation_perturbation


@dataclass
class ClassificationDataBundle:
    """一次分类实验需要的数据侧信息。"""

    dataset_name: str
    split: int
    n_variables: int
    n_classes: int
    seq_len_max_irr: int
    train_dataset: Dataset
    val_dataset: Dataset
    test_dataset: Dataset
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader


def _apply_normalizer(
    dataset: ClassificationWrapperDataset,
    normalizer: ClassificationNormalizer,
) -> ClassificationWrapperDataset:
    dataset.records = [normalizer.transform(record) for record in dataset.records]
    dataset.max_seq_len = max((item.values.shape[0] for item in dataset.records), default=1)
    return dataset


def classification_data_provider(
    dataset_name: str,
    split: int,
    batch_size: int,
    num_workers: int,
    data_root: str | None = None,
    perturbation_mode: str = "none",
    perturbation_rate: float = 0.0,
    perturbation_seed: int = 2026,
    perturbation_split: str = "test",
    perturbation_min_observations: int = 1,
    perturbation_local_scale_contrast: float = 0.75,
) -> ClassificationDataBundle:
    """构造官方 split 的 train/val/test loader。"""

    name = dataset_name.upper()
    if name not in DATASET_SPECS:
        raise ValueError(f"Unsupported classification dataset: {dataset_name}")

    train_dataset = ClassificationWrapperDataset(name, split, "train", data_root=data_root)
    val_dataset = ClassificationWrapperDataset(name, split, "val", data_root=data_root)
    test_dataset = ClassificationWrapperDataset(name, split, "test", data_root=data_root)
    # Fit preprocessing statistics on the training split only to avoid
    # leaking validation distribution information into model selection.
    normalizer = ClassificationNormalizer(train_dataset.records)
    train_dataset = _apply_normalizer(train_dataset, normalizer)
    val_dataset = _apply_normalizer(val_dataset, normalizer)
    test_dataset = _apply_normalizer(test_dataset, normalizer)

    seq_len_max_irr = max(train_dataset.max_seq_len, val_dataset.max_seq_len, test_dataset.max_seq_len)
    base_collate_fn = build_classification_collate(seq_len_max_irr)

    loader_common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )
    if torch.cuda.is_available():
        loader_common["pin_memory"] = True
    if num_workers > 0:
        loader_common["persistent_workers"] = True
        loader_common["prefetch_factor"] = 2

    def loader_collate(split_name: str):
        def _collate(batch):
            output = base_collate_fn(batch)
            if (
                perturbation_mode != "none"
                and perturbation_rate > 0
                and perturbation_split in {"all", split_name}
            ):
                output = apply_observation_perturbation(
                    output,
                    mode=perturbation_mode,
                    rate=perturbation_rate,
                    seed=perturbation_seed,
                    min_observations=perturbation_min_observations,
                    local_scale_contrast=perturbation_local_scale_contrast,
                )
            return output

        return _collate

    train_loader = DataLoader(train_dataset, shuffle=True, collate_fn=loader_collate("train"), **{k: v for k, v in loader_common.items() if k != "collate_fn"})
    val_loader = DataLoader(val_dataset, shuffle=False, collate_fn=loader_collate("val"), **{k: v for k, v in loader_common.items() if k != "collate_fn"})
    test_loader = DataLoader(test_dataset, shuffle=False, collate_fn=loader_collate("test"), **{k: v for k, v in loader_common.items() if k != "collate_fn"})
    return ClassificationDataBundle(
        dataset_name=name,
        split=split,
        n_variables=train_dataset.n_variables,
        n_classes=train_dataset.n_classes,
        seq_len_max_irr=seq_len_max_irr,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )
