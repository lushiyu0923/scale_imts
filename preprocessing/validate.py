from __future__ import annotations

"""Dataset validation helpers used before a training run."""

from pathlib import Path

from scaleimts.data.classification.protocol import DATASET_SPECS, candidate_roots, required_processed_files, resolve_existing_file, split_file_candidates


FORECASTING_FILES = {
    "USHCN": "small_chunked_sporadic.csv",
    "P12": "set-a.tar.gz",
    "HUMANACTIVITY": "ConfLongDemo_JSI.txt",
    "MIMIC_III": "complete_tensor.csv",
}


def validate_forecasting_root(dataset_name: str, root: str | Path) -> list[Path]:
    name = dataset_name.upper()
    if name not in FORECASTING_FILES:
        raise ValueError(f"Unsupported forecasting dataset: {dataset_name}")
    path = Path(root).expanduser()
    expected = [path / FORECASTING_FILES[name]]
    if name == "P12":
        expected.extend(path / filename for filename in ("set-b.tar.gz", "set-c.tar.gz"))
    missing = [item for item in expected if not item.exists()]
    if missing:
        names = "\n".join(f"  - {item}" for item in missing)
        raise FileNotFoundError(f"Missing {name} files:\n{names}")
    return expected


def validate_classification_root(dataset_name: str, root: str | Path, split: int = 1) -> list[Path]:
    name = dataset_name.upper()
    roots = candidate_roots(name, root)
    spec = DATASET_SPECS[name]
    required: list[Path] = []
    for relative in (*spec.dict_names, *spec.outcome_names):
        resolved = resolve_existing_file([candidate / relative for candidate in roots])
        required.append(resolved or roots[0] / relative)

    split_candidates = split_file_candidates(name, split, root)
    if split_candidates:
        required.append(resolve_existing_file(split_candidates) or split_candidates[0])
    elif name == "MIMIC_III":
        required.extend(
            roots[0] / f"mimic3_{flag}_{suffix}.npy"
            for flag in ("train", "val", "test")
            for suffix in ("x", "y")
        )

    missing = [item for item in required if not item.exists()]
    if missing:
        names = "\n".join(f"  - {item}" for item in missing)
        raise FileNotFoundError(f"Missing processed {name} files:\n{names}")
    return required