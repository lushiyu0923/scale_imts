from __future__ import annotations

"""Experiment configuration shared by forecasting and classification."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TrainConfig:
    """All runtime options required by the ScaleIMTS pipeline.

    Dataset files are intentionally referenced by path and are never copied into
    the source tree.  Use :meth:`for_dataset` for sensible dataset defaults and
    override individual values from the command line.
    """

    dataset_name: str
    dataset_root_path: str
    raw_csv_name: str = "small_chunked_sporadic.csv"
    raw_data_files: list[str] = field(default_factory=list)
    seq_len: int = 150
    pred_len: int = 3
    batch_size: int = 8
    num_workers: int = 0
    collate_fn: str = "collate_fn"
    missing_rate: float = 0.0
    patch_len: int = 12
    max_time_steps: int = 200
    enc_in: int = 5
    c_out: int = 5
    features: str = "M"
    seed: int = 2024
    num_runs: int = 1
    learning_rate: float = 1e-3
    train_epochs: int = 10
    patience: int = 5
    dropout: float = 0.1
    hidden_dim: int = 768
    num_hidden_layers: int = 2
    use_time_features: bool = True
    device: str = "cuda"
    save_dir: str = "scaleimts/checkpoints"
    optimizer_name: str = "adamw"
    use_scheduler: bool = False
    scheduler_name: str = "constant"
    scheduler_gamma: float = 0.5
    scheduler_step_size: int = 10
    use_early_stopping: bool = True
    branch_stats_interval: int = 0

    foundation_te_model: str = "gpt"
    foundation_st_model: str = "bert"
    foundation_n_te_layers: int = 2
    foundation_n_st_layers: int = 2
    foundation_semi_freeze: bool = False
    foundation_pretrained_root: str = ""
    foundation_train_last_n_layers: int = 0
    foundation_scale_mode: str = "none"
    foundation_scale_num_segments: int = 4
    foundation_scale_num_experts: int = 3
    foundation_scale_router_hidden_dim: int = 256
    foundation_scale_expert_hidden_dim: int = 256
    foundation_scale_temperature: float = 1.0
    foundation_scale_use_future_context: bool = True
    foundation_scale_use_irregular_context: bool = True
    ablation_variant: str = "full"

    irregular_hidden_dim: int = 64
    irregular_time_dim: int = 16
    irregular_node_dim: int = 16
    foundation_reduced_dim: int = 128
    foundation_gate_hidden_dim: int = 256
    fusion_type: str = "film_i2f_var"
    fusion_hidden_dim: int = 128
    film_hidden_dim: int = 256

    perturbation_mode: str = "none"
    perturbation_rate: float = 0.0
    perturbation_seed: int = 2026
    perturbation_split: str = "test"
    perturbation_min_observations: int = 1
    perturbation_local_scale_contrast: float = 0.75

    split: int = 1
    classification_data_root: str | None = None

    @classmethod
    def for_dataset(cls, dataset_name: str, dataset_root_path: str, **overrides: Any) -> "TrainConfig":
        name = dataset_name.upper()
        defaults: dict[str, dict[str, Any]] = {
            "USHCN": dict(raw_csv_name="small_chunked_sporadic.csv", seq_len=150, pred_len=3, batch_size=32, learning_rate=1e-3, train_epochs=200, patch_len=12, max_time_steps=200, enc_in=5, c_out=5),
            "P12": dict(raw_data_files=["set-a.tar.gz", "set-b.tar.gz", "set-c.tar.gz"], seq_len=36, pred_len=3, batch_size=32, learning_rate=1e-3, train_epochs=200, patch_len=12, max_time_steps=48, enc_in=36, c_out=36),
            "HUMANACTIVITY": dict(raw_data_files=["ConfLongDemo_JSI.txt"], seq_len=3000, pred_len=300, batch_size=16, num_workers=0, learning_rate=1e-3, train_epochs=200, patch_len=100, max_time_steps=4000, enc_in=12, c_out=12),
            "MIMIC_III": dict(raw_csv_name="complete_tensor.csv", seq_len=72, pred_len=3, batch_size=8, learning_rate=1e-3, train_epochs=300, patch_len=12, max_time_steps=96, enc_in=96, c_out=96),
        }
        if name not in defaults:
            raise ValueError(f"Unsupported forecasting dataset: {dataset_name}")
        values: dict[str, Any] = dict(defaults[name])
        values.update(dataset_name=name, dataset_root_path=str(Path(dataset_root_path)))
        values.update(overrides)
        return cls(**values)

    def validate(self) -> None:
        if self.batch_size <= 0 or self.train_epochs <= 0:
            raise ValueError("batch_size and train_epochs must be positive")
        if self.seq_len <= 0 or self.pred_len <= 0 or self.patch_len <= 0:
            raise ValueError("seq_len, pred_len and patch_len must be positive")
        if self.hidden_dim < 2:
            raise ValueError("hidden_dim must be at least 2")
        if self.foundation_scale_temperature <= 0:
            raise ValueError("foundation_scale_temperature must be positive")
