from __future__ import annotations

"""数据集与 DataLoader 构造入口。"""

from collections.abc import Callable

import torch
from torch.utils.data import DataLoader, Dataset

from .config import DataConfig
from .irregular_perturbation import apply_observation_perturbation
from .human_activity.collate import COLLATE_BUILDERS as HUMAN_ACTIVITY_COLLATE_BUILDERS
from .human_activity.wrapper import HumanActivityWrapperDataset
from .mimic_iii.wrapper import MIMICIIIWrapperDataset
from .p12.collate import COLLATE_BUILDERS as P12_COLLATE_BUILDERS
from .p12.wrapper import P12WrapperDataset
from .ushcn.collate import COLLATE_BUILDERS as USHCN_COLLATE_BUILDERS
from .ushcn.wrapper import USHCNWrapperDataset


DATASET_REGISTRY: dict[str, tuple[type[Dataset], dict[str, Callable]]] = {
    "USHCN": (USHCNWrapperDataset, USHCN_COLLATE_BUILDERS),
    "P12": (P12WrapperDataset, P12_COLLATE_BUILDERS),
    "HUMANACTIVITY": (HumanActivityWrapperDataset, HUMAN_ACTIVITY_COLLATE_BUILDERS),
    "MIMIC_III": (MIMICIIIWrapperDataset, P12_COLLATE_BUILDERS),
}


def data_provider(
    config: DataConfig,
    flag: str,
) -> tuple[Dataset, DataLoader]:
    """为指定数据集和 split 构造 Dataset / DataLoader。"""

    if flag not in {"train", "val", "test", "test_all"}:
        raise ValueError(f"Unsupported split: {flag}")

    try:
        dataset_cls, collate_builders = DATASET_REGISTRY[config.dataset_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported dataset_name: {config.dataset_name}") from exc

    dataset = dataset_cls(config=config, flag=flag)

    try:
        collate_builder = collate_builders[config.collate_fn]
    except KeyError as exc:
        raise ValueError(f"Unknown collate_fn: {config.collate_fn}") from exc

    if flag in {"test", "test_all"}:
        shuffle = False
        drop_last = False
    else:
        shuffle = True if config.train_val_loader_shuffle is None else config.train_val_loader_shuffle
        drop_last = True if config.train_val_loader_drop_last is None else config.train_val_loader_drop_last

    base_collate_fn = collate_builder(config)

    def collate_fn(batch):
        output = base_collate_fn(batch)
        if (
            config.perturbation_mode != "none"
            and config.perturbation_rate > 0
            and config.perturbation_split in {"all", flag}
        ):
            output = apply_observation_perturbation(
                output,
                mode=config.perturbation_mode,
                rate=config.perturbation_rate,
                seed=config.perturbation_seed,
                min_observations=config.perturbation_min_observations,
                local_scale_contrast=config.perturbation_local_scale_contrast,
            )
        return output

    loader_kwargs = dict(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        drop_last=drop_last,
        collate_fn=collate_fn,
    )
    if torch.cuda.is_available():
        loader_kwargs["pin_memory"] = True
    if config.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    loader = DataLoader(**loader_kwargs)
    return dataset, loader
