from __future__ import annotations

"""Forecasting training, validation and test loop."""

from pathlib import Path

import numpy as np
import torch

from scaleimts.data import DataConfig, data_provider
from scaleimts.models.core import Model, ScaleIMTSConfig
from scaleimts.training.base import TrainerBase
from scaleimts.utils.globals import logger
from scaleimts.utils.losses import MaskedMSELoss
from scaleimts.utils.metrics import metric, round_metric_dict
from scaleimts.utils.tools import EarlyStopping, move_batch_to_device


class ForecastingTrainer(TrainerBase):
    def _data_config(self) -> DataConfig:
        return DataConfig(
            dataset_name=self.config.dataset_name,
            dataset_root_path=self.config.dataset_root_path,
            raw_csv_name=self.config.raw_csv_name,
            raw_data_files=self.config.raw_data_files,
            seq_len=self.config.seq_len,
            pred_len=self.config.pred_len,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            collate_fn=self.config.collate_fn,
            missing_rate=self.config.missing_rate,
            patch_len=self.config.patch_len,
            random_seed=self.config.seed,
            max_time_steps=self.config.max_time_steps,
            enc_in=self.config.enc_in,
            c_out=self.config.c_out,
            features=self.config.features,
            perturbation_mode=self.config.perturbation_mode,
            perturbation_rate=self.config.perturbation_rate,
            perturbation_seed=self.config.perturbation_seed,
            perturbation_split=self.config.perturbation_split,
            perturbation_min_observations=self.config.perturbation_min_observations,
            perturbation_local_scale_contrast=self.config.perturbation_local_scale_contrast,
        )

    def _model(self, data_config: DataConfig) -> Model:
        model_config = ScaleIMTSConfig(
            enc_in=data_config.enc_in,
            c_out=data_config.c_out,
            seq_len_max_irr=data_config.seq_len_max_irr or self.config.seq_len,
            pred_len_max_irr=data_config.pred_len_max_irr or self.config.pred_len,
            features=data_config.features,
            hidden_dim=self.config.hidden_dim,
            num_hidden_layers=self.config.num_hidden_layers,
            dropout=self.config.dropout,
            use_time_features=self.config.use_time_features,
            foundation_te_model=self.config.foundation_te_model,
            foundation_st_model=self.config.foundation_st_model,
            foundation_n_te_layers=self.config.foundation_n_te_layers,
            foundation_n_st_layers=self.config.foundation_n_st_layers,
            foundation_semi_freeze=self.config.foundation_semi_freeze,
            foundation_pretrained_root=self.config.foundation_pretrained_root,
            foundation_train_last_n_layers=self.config.foundation_train_last_n_layers,
            foundation_scale_mode=self.config.foundation_scale_mode,
            foundation_scale_num_segments=self.config.foundation_scale_num_segments,
            foundation_scale_num_experts=self.config.foundation_scale_num_experts,
            foundation_scale_router_hidden_dim=self.config.foundation_scale_router_hidden_dim,
            foundation_scale_expert_hidden_dim=self.config.foundation_scale_expert_hidden_dim,
            foundation_scale_temperature=self.config.foundation_scale_temperature,
            foundation_scale_use_future_context=self.config.foundation_scale_use_future_context,
            foundation_scale_use_irregular_context=self.config.foundation_scale_use_irregular_context,
            ablation_variant=self.config.ablation_variant,
            patch_len=self.config.patch_len,
            max_time_steps=self.config.max_time_steps,
            irregular_hidden_dim=self.config.irregular_hidden_dim,
            irregular_time_dim=self.config.irregular_time_dim,
            irregular_node_dim=self.config.irregular_node_dim,
            foundation_reduced_dim=self.config.foundation_reduced_dim,
            foundation_gate_hidden_dim=self.config.foundation_gate_hidden_dim,
            fusion_type=self.config.fusion_type,
            fusion_hidden_dim=self.config.fusion_hidden_dim,
            film_hidden_dim=self.config.film_hidden_dim,
        )
        return Model(model_config).to(self.device)

    def _iterate(self, model, loader, criterion, optimizer=None):
        training = optimizer is not None
        model.train(training)
        losses: list[float] = []
        predictions: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        for batch in loader:
            batch = move_batch_to_device(batch, self.device)
            with torch.set_grad_enabled(training):
                outputs = model(**batch)
                loss = criterion(**outputs)["loss"]
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if outputs["true"] is not None:
                predictions.append(outputs["pred"].detach().cpu().numpy())
                targets.append(outputs["true"].detach().cpu().numpy())
                masks.append(outputs["mask"].detach().cpu().numpy() if outputs["mask"] is not None else np.ones_like(targets[-1]))
        if not predictions:
            return float(np.mean(losses)) if losses else 0.0, {}
        metrics = metric(np.concatenate(predictions), np.concatenate(targets), np.concatenate(masks))
        return float(np.mean(losses)), metrics

    def train(self) -> dict[str, float]:
        data_config = self._data_config()
        train_dataset, train_loader = data_provider(data_config, "train")
        _, val_loader = data_provider(data_config, "val")
        _, test_loader = data_provider(data_config, "test")
        model = self._model(data_config)
        criterion = MaskedMSELoss()
        optimizer = self.build_optimizer(model)
        scheduler = self.build_scheduler(optimizer)
        checkpoint = self.checkpoint_dir("forecasting") / "best_model.pt"
        stopper = EarlyStopping(self.config.patience) if self.config.use_early_stopping else None
        saved = False
        logger.info("Forecasting: dataset=%s train=%d val=%d test=%d device=%s", self.config.dataset_name, len(train_dataset), len(val_loader.dataset), len(test_loader.dataset), self.device)
        for epoch in range(1, self.config.train_epochs + 1):
            train_loss, train_metrics = self._iterate(model, train_loader, criterion, optimizer)
            val_loss, val_metrics = self._iterate(model, val_loader, criterion)
            if scheduler is not None:
                scheduler.step()
            logger.info("Epoch %d/%d train_loss=%.5f val_loss=%.5f train=%s val=%s", epoch, self.config.train_epochs, train_loss, val_loss, round_metric_dict(train_metrics), round_metric_dict(val_metrics))
            if stopper is not None:
                saved = stopper(val_loss, model, checkpoint) or saved
                if stopper.early_stop:
                    break
        if stopper is None:
            self.save_best(model, checkpoint)
            saved = True
        if saved:
            self.load_best(model, checkpoint, self.device)
        _, test_metrics = self._iterate(model, test_loader, criterion)
        logger.info("Test metrics: %s", round_metric_dict(test_metrics))
        return test_metrics
