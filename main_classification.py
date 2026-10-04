from __future__ import annotations

"""Classification entry point, reorganized from ``work/main_classification.py``."""

import argparse
from pathlib import Path

import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scaleimts.data.classification import DATASET_SPECS
from scaleimts.training.config import TrainConfig
from scaleimts.utils.metrics import round_metric_dict, round_run_summary, summarize_runs


ABLATIONS = ["full", "irregular_only", "wo_local_scale", "wo_irregular_dynamics", "wo_foundation_refinement", "concat_fusion", "global_routing", "wo_scale_statistics"]


def _bool(value: str) -> bool:
    return value.lower() == "true"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate ScaleIMTS on irregular time-series classification.")
    parser.add_argument("--dataset", choices=sorted(DATASET_SPECS), default="P12")
    parser.add_argument("--data-root", default=None, help="Directory containing P12/P19/PAM/MIMIC_III; kept outside this repository.")
    parser.add_argument("--split", type=int, default=1, help="Official split id, usually 1-5.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num-runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--hidden-dim", type=int, default=768)
    parser.add_argument("--num-hidden-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patch-len", type=int, default=None)
    parser.add_argument("--foundation-pretrained-root", default=None)
    parser.add_argument("--foundation-te-model", choices=["gpt", "bert"], default="gpt")
    parser.add_argument("--foundation-st-model", choices=["gpt", "bert"], default="bert")
    parser.add_argument("--foundation-n-te-layers", type=int, default=2)
    parser.add_argument("--foundation-n-st-layers", type=int, default=2)
    parser.add_argument("--foundation-semi-freeze", choices=["true", "false"], default="false")
    parser.add_argument("--foundation-train-last-n-layers", type=int, default=0)
    parser.add_argument("--foundation-scale-mode", choices=["none", "expert_routing", "expert_routing_c2lite"], default="expert_routing_c2lite")
    parser.add_argument("--foundation-scale-num-segments", type=int, default=4)
    parser.add_argument("--foundation-scale-num-experts", type=int, default=3)
    parser.add_argument("--foundation-scale-temperature", type=float, default=1.0)
    parser.add_argument("--fusion-type", choices=["concat", "film_f2i", "film_i2f", "film_f2i_var", "film_i2f_var"], default="film_i2f_var")
    parser.add_argument("--ablation-variant", choices=ABLATIONS, default="full")
    parser.add_argument("--perturbation-mode", choices=["none", "random", "local_scale", "block_missing"], default="none")
    parser.add_argument("--perturbation-rate", type=float, default=0.0)
    parser.add_argument("--perturbation-split", choices=["all", "train", "val", "test"], default="test")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    from scaleimts.training import ClassificationTrainer, set_global_seed
    repo_root = Path(__file__).resolve().parents[1]
    data_root = Path(args.data_root).expanduser() if args.data_root else repo_root / "data" / "classification"
    config = TrainConfig(
        dataset_name=args.dataset.upper(),
        dataset_root_path=str(data_root),
        batch_size=args.batch_size or DATASET_SPECS[args.dataset.upper()].default_batch_size,
        train_epochs=args.epochs,
        learning_rate=args.lr or DATASET_SPECS[args.dataset.upper()].default_lr,
        num_runs=args.num_runs,
        seed=args.seed,
        num_workers=args.num_workers,
        patience=args.patience,
        device=args.device,
        save_dir=args.save_dir or str(repo_root / "scaleimts" / "checkpoints"),
        hidden_dim=args.hidden_dim,
        num_hidden_layers=args.num_hidden_layers,
        dropout=args.dropout,
        patch_len=args.patch_len or DATASET_SPECS[args.dataset.upper()].patch_len,
        foundation_pretrained_root=args.foundation_pretrained_root or "",
        foundation_te_model=args.foundation_te_model,
        foundation_st_model=args.foundation_st_model,
        foundation_n_te_layers=args.foundation_n_te_layers,
        foundation_n_st_layers=args.foundation_n_st_layers,
        foundation_semi_freeze=_bool(args.foundation_semi_freeze),
        foundation_train_last_n_layers=args.foundation_train_last_n_layers,
        foundation_scale_mode=args.foundation_scale_mode,
        foundation_scale_num_segments=args.foundation_scale_num_segments,
        foundation_scale_num_experts=args.foundation_scale_num_experts,
        foundation_scale_temperature=args.foundation_scale_temperature,
        fusion_type=args.fusion_type,
        ablation_variant=args.ablation_variant,
        perturbation_mode=args.perturbation_mode,
        perturbation_rate=args.perturbation_rate,
        perturbation_split=args.perturbation_split,
        split=args.split,
        classification_data_root=str(data_root),
    )
    config.validate()
    all_metrics = []
    for run_index in range(config.num_runs):
        run_seed = config.seed + run_index
        set_global_seed(run_seed)
        config.seed = run_seed
        print(f"[classification run {run_index + 1}/{config.num_runs}] dataset={config.dataset_name} split={config.split} seed={run_seed}")
        metrics = ClassificationTrainer(config).train()
        all_metrics.append(metrics)
        print("metrics:", round_metric_dict(metrics))
    if len(all_metrics) > 1:
        print("summary:", round_run_summary(summarize_runs(all_metrics)))


if __name__ == "__main__":
    main()
