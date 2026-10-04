from __future__ import annotations

"""Forecasting entry point, reorganized from ``work/main.py``."""

import argparse
from pathlib import Path

import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scaleimts.training.config import TrainConfig
from scaleimts.utils.metrics import round_metric_dict, round_run_summary, summarize_runs


ABLATIONS = ["full", "irregular_only", "wo_local_scale", "wo_irregular_dynamics", "wo_foundation_refinement", "concat_fusion", "global_routing", "wo_scale_statistics"]


def _bool(value: str) -> bool:
    return value.lower() == "true"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate ScaleIMTS for irregular time-series forecasting.")
    parser.add_argument("--dataset", choices=["USHCN", "P12", "HUMANACTIVITY", "MIMIC_III"], default="USHCN")
    parser.add_argument("--data-root", default=None, help="Directory containing the selected dataset; kept outside this repository.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num-runs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--hidden-dim", type=int, default=None, help="Foundation hidden size; use 768 for standard local GPT-2/BERT weights.")
    parser.add_argument("--num-hidden-layers", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--patch-len", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--pred-len", type=int, default=None)
    parser.add_argument("--foundation-pretrained-root", default=None, help="Optional directory containing gpt2/ and bert-base-uncased/.")
    parser.add_argument("--foundation-te-model", choices=["gpt", "bert"], default=None)
    parser.add_argument("--foundation-st-model", choices=["gpt", "bert"], default=None)
    parser.add_argument("--foundation-n-te-layers", type=int, default=None)
    parser.add_argument("--foundation-n-st-layers", type=int, default=None)
    parser.add_argument("--foundation-semi-freeze", choices=["true", "false"], default=None)
    parser.add_argument("--foundation-train-last-n-layers", type=int, default=None)
    parser.add_argument("--foundation-scale-mode", choices=["none", "expert_routing", "expert_routing_c2lite"], default=None)
    parser.add_argument("--foundation-scale-num-segments", type=int, default=None)
    parser.add_argument("--foundation-scale-num-experts", type=int, default=None)
    parser.add_argument("--foundation-scale-temperature", type=float, default=None)
    parser.add_argument("--fusion-type", choices=["concat", "film_f2i", "film_i2f", "film_f2i_var", "film_i2f_var"], default=None)
    parser.add_argument("--ablation-variant", choices=ABLATIONS, default=None)
    parser.add_argument("--optimizer", choices=["adam", "adamw", "sgd"], default=None)
    parser.add_argument("--use-scheduler", choices=["true", "false"], default=None)
    parser.add_argument("--use-early-stopping", choices=["true", "false"], default=None)
    parser.add_argument("--perturbation-mode", choices=["none", "random", "local_scale", "block_missing"], default=None)
    parser.add_argument("--perturbation-rate", type=float, default=None)
    parser.add_argument("--perturbation-split", choices=["all", "train", "val", "test"], default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    from scaleimts.training import ForecastingTrainer, set_global_seed
    repo_root = Path(__file__).resolve().parents[1]
    dataset_root = Path(args.data_root).expanduser() if args.data_root else repo_root / "data" / args.dataset
    pretrained_root = args.foundation_pretrained_root or ""
    overrides = {
        "batch_size": args.batch_size,
        "train_epochs": args.epochs,
        "learning_rate": args.lr,
        "num_runs": args.num_runs,
        "seed": args.seed,
        "num_workers": args.num_workers,
        "patience": args.patience,
        "device": args.device,
        "save_dir": args.save_dir,
        "hidden_dim": args.hidden_dim,
        "num_hidden_layers": args.num_hidden_layers,
        "dropout": args.dropout,
        "patch_len": args.patch_len,
        "seq_len": args.seq_len,
        "pred_len": args.pred_len,
        "foundation_pretrained_root": pretrained_root,
        "foundation_te_model": args.foundation_te_model,
        "foundation_st_model": args.foundation_st_model,
        "foundation_n_te_layers": args.foundation_n_te_layers,
        "foundation_n_st_layers": args.foundation_n_st_layers,
        "foundation_semi_freeze": _bool(args.foundation_semi_freeze) if args.foundation_semi_freeze else None,
        "foundation_train_last_n_layers": args.foundation_train_last_n_layers,
        "foundation_scale_mode": args.foundation_scale_mode,
        "foundation_scale_num_segments": args.foundation_scale_num_segments,
        "foundation_scale_num_experts": args.foundation_scale_num_experts,
        "foundation_scale_temperature": args.foundation_scale_temperature,
        "fusion_type": args.fusion_type,
        "ablation_variant": args.ablation_variant,
        "optimizer_name": args.optimizer,
        "use_scheduler": _bool(args.use_scheduler) if args.use_scheduler else None,
        "use_early_stopping": _bool(args.use_early_stopping) if args.use_early_stopping else None,
        "perturbation_mode": args.perturbation_mode,
        "perturbation_rate": args.perturbation_rate,
        "perturbation_split": args.perturbation_split,
    }
    overrides = {key: value for key, value in overrides.items() if value is not None}
    config = TrainConfig.for_dataset(args.dataset, str(dataset_root), **overrides)
    config.validate()
    run_metrics = []
    for run_index in range(config.num_runs):
        run_seed = config.seed + run_index
        set_global_seed(run_seed)
        config.seed = run_seed
        print(f"[run {run_index + 1}/{config.num_runs}] dataset={config.dataset_name} seed={run_seed}")
        metrics = ForecastingTrainer(config).train()
        run_metrics.append(metrics)
        print("metrics:", round_metric_dict(metrics))
    if len(run_metrics) > 1:
        print("summary:", round_run_summary(summarize_runs(run_metrics)))


if __name__ == "__main__":
    main()
