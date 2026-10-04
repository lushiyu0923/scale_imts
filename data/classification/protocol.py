from __future__ import annotations

"""Paths and metadata for official IMTS classification datasets.

The repository never stores these datasets.  Pass ``--data-root`` or set
``SCALEIMTS_DATA_ROOT`` to a directory containing the downloaded files.
"""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ClassificationDatasetSpec:
    name: str
    n_classes: int
    n_variables: int
    split_glob: str
    dict_names: tuple[str, ...]
    outcome_names: tuple[str, ...]
    figshare_url: str
    time_unit: str
    patch_len: int
    default_batch_size: int
    default_lr: float
    max_time_horizon: float


DATASET_SPECS = {
    "P12": ClassificationDatasetSpec("P12", 2, 36, "splits/phy12_split{split}.npy", ("processed_data/PTdict_list.npy",), ("processed_data/arr_outcomes.npy",), "https://doi.org/10.6084/m9.figshare.19514341.v1", "minutes", 12, 8, 1e-3, 48.0 * 60.0),
    "P19": ClassificationDatasetSpec("P19", 2, 34, "splits/phy19_split{split}_new.npy", ("processed_data/PT_dict_list_6.npy",), ("processed_data/arr_outcomes_6.npy",), "https://doi.org/10.6084/m9.figshare.19514338.v1", "hours", 12, 8, 1e-3, 60.0),
    "PAM": ClassificationDatasetSpec("PAM", 8, 17, "splits/PAMAP2_split_{split}.npy", ("processed_data/PTdict_list.npy",), ("processed_data/arr_outcomes.npy",), "https://doi.org/10.6084/m9.figshare.19514347.v1", "seconds", 20, 8, 1e-3, 60.0),
    "MIMIC_III": ClassificationDatasetSpec("MIMIC_III", 2, 16, "", (), (), "", "hours", 12, 8, 1e-3, 48.0),
}


def candidate_roots(dataset_name: str, data_root: str | Path | None = None) -> list[Path]:
    name = dataset_name.upper()
    if name not in DATASET_SPECS:
        raise ValueError(f"Unsupported classification dataset: {dataset_name}")
    if data_root is not None:
        base = Path(data_root).expanduser()
        return [base / name, base]

    configured = os.environ.get("SCALEIMTS_DATA_ROOT")
    if configured:
        base = Path(configured).expanduser()
        return [base / name, base]

    repo_root = Path(__file__).resolve().parents[3]
    return [repo_root / "data" / "classification" / name]


def split_file_candidates(dataset_name: str, split: int, data_root: str | Path | None = None) -> list[Path]:
    name = dataset_name.upper()
    spec = DATASET_SPECS[name]
    if not spec.split_glob:
        return []
    paths = [root / spec.split_glob.format(split=split) for root in candidate_roots(name, data_root)]
    if name == "PAM":
        paths.extend(root / f"splits/PAM_split_{split}.npy" for root in candidate_roots(name, data_root))
    return paths


def required_processed_files(dataset_name: str, split: int = 1) -> list[str]:
    spec = DATASET_SPECS[dataset_name.upper()]
    names = list(spec.dict_names) + list(spec.outcome_names)
    if spec.split_glob:
        names.append(spec.split_glob.format(split=split))
    return names


def resolve_existing_file(candidates: list[Path]) -> Path | None:
    return next((path for path in candidates if path.exists()), None)


def missing_file_message(dataset_name: str, data_root: str | Path | None = None) -> str:
    name = dataset_name.upper()
    spec = DATASET_SPECS[name]
    roots = "\n".join(f"  - {path}" for path in candidate_roots(name, data_root))
    files = "\n".join(f"  - {file_name}" for file_name in required_processed_files(name))
    download = f" Download the official files from {spec.figshare_url}." if spec.figshare_url else ""
    return f"Processed files for {name} were not found.\nLooked in:\n{roots}\nRequired files:\n{files}\n{download}"
