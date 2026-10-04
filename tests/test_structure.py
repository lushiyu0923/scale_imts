from __future__ import annotations

from pathlib import Path

from scaleimts.preprocessing.validate import validate_classification_root, validate_forecasting_root
from scaleimts.training.config import TrainConfig


def test_forecasting_config_defaults():
    config = TrainConfig.for_dataset("USHCN", "D:/datasets/USHCN")
    assert config.dataset_name == "USHCN"
    assert config.dataset_root_path == str(Path("D:/datasets/USHCN"))
    assert config.enc_in == config.c_out == 5


def test_external_validation_is_path_only():
    assert callable(validate_forecasting_root)
    assert callable(validate_classification_root)
