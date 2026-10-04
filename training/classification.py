from __future__ import annotations

"""Classification training and evaluation for official IMTS splits."""

import numpy as np
import torch

from scaleimts.data.classification import DATASET_SPECS, classification_data_provider
from scaleimts.models.core import Model, ScaleIMTSClassifier, ScaleIMTSConfig
from scaleimts.training.base import TrainerBase
from scaleimts.utils.globals import logger
from scaleimts.utils.losses import ClassificationCELoss
from scaleimts.utils.metrics import classification_metric, round_metric_dict
from scaleimts.utils.tools import EarlyStopping, move_batch_to_device


class ClassificationTrainer(TrainerBase):
    def _model(self, n_variables: int, seq_len: int, n_classes: int) -> ScaleIMTSClassifier:
        spec = DATASET_SPECS[self.config.dataset_name.upper()]
        model_config = ScaleIMTSConfig(
            enc_in=n_variables,
            c_out=n_variables,
            seq_len_max_irr=seq_len,
            pred_len_max_irr=1,
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
            patch_len=self.config.patch_len or spec.patch_len,
            max_time_steps=max(seq_len, 1),
            irregular_hidden_dim=self.config.irregular_hidden_dim,
            irregular_time_dim=self.config.irregular_time_dim,
            irregular_node_dim=self.config.irregular_node_dim,
            foundation_reduced_dim=self.config.foundation_reduced_dim,
            foundation_gate_hidden_dim=self.config.foundation_gate_hidden_dim,
            fusion_type=self.config.fusion_type,
            fusion_hidden_dim=self.config.fusion_hidden_dim,
            film_hidden_dim=self.config.film_hidden_dim,
        )
        return ScaleIMTSClassifier(Model(model_config).to(self.device), n_classes=n_classes, dropout=self.config.dropout).to(self.device)

    def _iterate(self, model, loader, criterion, n_classes: int, optimizer=None):
        training = optimizer is not None
        model.train(training)
        losses: list[float] = []
        logits: list[np.ndarray] = []
        labels: list[np.ndarray] = []
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
            logits.append(outputs["pred"].detach().cpu().numpy())
            labels.append(outputs["true"].detach().cpu().numpy().reshape(-1))
        if not logits:
            return 0.0, {}
        metrics = classification_metric(np.concatenate(logits), np.concatenate(labels), n_classes=n_classes)
        return float(np.mean(losses)), metrics

    def train(self) -> dict[str, float]:
        data = classification_data_provider(
            dataset_name=self.config.dataset_name,
            split=self.config.split,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            data_root=self.config.classification_data_root,
            perturbation_mode=self.config.perturbation_mode,
            perturbation_rate=self.config.perturbation_rate,
            perturbation_seed=self.config.perturbation_seed,
            perturbation_split=self.config.perturbation_split,
            perturbation_min_observations=self.config.perturbation_min_observations,
            perturbation_local_scale_contrast=self.config.perturbation_local_scale_contrast,
        )
        model = self._model(data.n_variables, data.seq_len_max_irr, data.n_classes)
        criterion = ClassificationCELoss()
        optimizer = self.build_optimizer(model)
        checkpoint = self.checkpoint_dir("classification") / "best_model.pt"
        stopper = EarlyStopping(self.config.patience) if self.config.use_early_stopping else None
        saved = False
        logger.info("Classification: dataset=%s split=%d train=%d val=%d test=%d device=%s", self.config.dataset_name, self.config.split, len(data.train_dataset), len(data.val_dataset), len(data.test_dataset), self.device)
        for epoch in range(1, self.config.train_epochs + 1):
            train_loss, train_metrics = self._iterate(model, data.train_loader, criterion, data.n_classes, optimizer)
            val_loss, val_metrics = self._iterate(model, data.val_loader, criterion, data.n_classes)
            logger.info("Epoch %d/%d train_loss=%.5f val_loss=%.5f train=%s val=%s", epoch, self.config.train_epochs, train_loss, val_loss, round_metric_dict(train_metrics), round_metric_dict(val_metrics))
            if stopper is not None:
                saved = stopper(-float(val_metrics.get("AUROC", 0.5)), model, checkpoint) or saved
                if stopper.early_stop:
                    break
        if stopper is None:
            self.save_best(model, checkpoint)
            saved = True
        if saved:
            self.load_best(model, checkpoint, self.device)
        _, test_metrics = self._iterate(model, data.test_loader, criterion, data.n_classes)
        logger.info("Test metrics: %s", round_metric_dict(test_metrics))
        return test_metrics
