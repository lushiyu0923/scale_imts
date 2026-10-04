from __future__ import annotations

"""Patch-based irregular alignment and encoding blocks for ScaleIMTS."""

import math

import torch
import torch.nn as nn
from torch import Tensor

from .config import BranchAlignedBatch
from .time_encoders import TimeFeatureEncoder


class IrregularAligner(nn.Module):
    """Align irregular observations into fixed patch buckets."""

    def __init__(self, patch_len: int, max_time_steps: int):
        super().__init__()
        if patch_len <= 0:
            raise ValueError("patch_len must be positive for IrregularAligner.")
        if max_time_steps <= 0:
            raise ValueError("max_time_steps must be positive for IrregularAligner.")
        self.patch_len = patch_len
        self.max_time_steps = max_time_steps

    def _build_patch_metadata(self, seq_len: int) -> tuple[float, int]:
        n_patch = int((seq_len + self.patch_len - 1) // self.patch_len)
        patch_size_norm = self.patch_len / float(self.max_time_steps)
        return patch_size_norm, n_patch

    def forward(self, x: Tensor, x_mark: Tensor, x_mask: Tensor) -> BranchAlignedBatch:
        batch_size, seq_len, n_var = x.shape
        device = x.device
        patch_size_norm, n_patch = self._build_patch_metadata(seq_len=seq_len)
        time_per_variable = x_mark.expand(-1, -1, n_var) if x_mark.shape[-1] == 1 else x_mark
        if time_per_variable.shape != x.shape:
            raise ValueError(
                f"Expected x_mark to broadcast to shape {tuple(x.shape)}, got {tuple(time_per_variable.shape)}."
            )

        valid_mask = x_mask > 0
        valid_indices = valid_mask.nonzero(as_tuple=False)

        if valid_indices.numel() == 0:
            patch_values = torch.zeros(batch_size, n_var, n_patch, 1, 1, device=device, dtype=x.dtype)
            patch_times = torch.zeros(batch_size, n_var, n_patch, 1, 1, device=device, dtype=x_mark.dtype)
            patch_masks = torch.zeros(batch_size, n_var, n_patch, 1, 1, device=device, dtype=x_mask.dtype)
            patch_valid = torch.zeros(batch_size, n_var, n_patch, 1, device=device, dtype=x_mask.dtype)
            return BranchAlignedBatch(
                values=x,
                time=x_mark,
                mask=x_mask,
                patch_values=patch_values,
                patch_times=patch_times,
                patch_masks=patch_masks,
                patch_valid=patch_valid,
            )

        valid_batch_indices = valid_indices[:, 0]
        valid_time_indices = valid_indices[:, 1]
        valid_variable_indices = valid_indices[:, 2]
        valid_values = x[valid_mask]
        valid_times = time_per_variable[valid_mask]

        patch_indices = torch.floor(valid_times / patch_size_norm).to(torch.long)
        patch_indices.clamp_(min=0, max=n_patch - 1)

        # Group by (batch, variable, patch), then keep events in time order within each bucket.
        group_keys = ((valid_batch_indices * n_var) + valid_variable_indices) * n_patch + patch_indices
        sort_keys = group_keys * (seq_len + 1) + valid_time_indices
        sort_order = torch.argsort(sort_keys)

        sorted_group_keys = group_keys[sort_order]
        sorted_batch_indices = valid_batch_indices[sort_order]
        sorted_variable_indices = valid_variable_indices[sort_order]
        sorted_patch_indices = patch_indices[sort_order]
        sorted_values = valid_values[sort_order]
        sorted_times = valid_times[sort_order]

        group_start_mask = torch.ones_like(sorted_group_keys, dtype=torch.bool)
        group_start_mask[1:] = sorted_group_keys[1:] != sorted_group_keys[:-1]
        group_starts = group_start_mask.nonzero(as_tuple=False).squeeze(-1)
        group_ends = torch.cat([group_starts[1:], group_starts.new_tensor([sorted_group_keys.numel()])])
        group_counts = group_ends - group_starts
        max_patch_events = max(1, int(group_counts.max().item()))

        event_positions = torch.arange(sorted_group_keys.numel(), device=device)
        event_ranks = event_positions - group_starts.repeat_interleave(group_counts)

        patch_values = torch.zeros(batch_size, n_var, n_patch, max_patch_events, 1, device=device, dtype=x.dtype)
        patch_times = torch.zeros(batch_size, n_var, n_patch, max_patch_events, 1, device=device, dtype=x_mark.dtype)
        patch_masks = torch.zeros(batch_size, n_var, n_patch, max_patch_events, 1, device=device, dtype=x_mask.dtype)
        patch_valid = torch.zeros(batch_size, n_var, n_patch, 1, device=device, dtype=x_mask.dtype)

        patch_values[sorted_batch_indices, sorted_variable_indices, sorted_patch_indices, event_ranks, 0] = sorted_values
        patch_times[sorted_batch_indices, sorted_variable_indices, sorted_patch_indices, event_ranks, 0] = sorted_times
        patch_masks[sorted_batch_indices, sorted_variable_indices, sorted_patch_indices, event_ranks, 0] = 1
        patch_valid[
            sorted_batch_indices[group_starts],
            sorted_variable_indices[group_starts],
            sorted_patch_indices[group_starts],
            0,
        ] = 1

        return BranchAlignedBatch(
            values=x,
            time=x_mark,
            mask=x_mask,
            patch_values=patch_values,
            patch_times=patch_times,
            patch_masks=patch_masks,
            patch_valid=patch_valid,
        )


class PatchPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for patch sequences."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe[:, : x.size(1), :]


class PatchIntraEncoder(nn.Module):
    """Aggregate irregular events within each patch."""

    def __init__(self, hidden_dim: int, time_dim: int, dropout: float):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.time_dim = time_dim
        self.time_encoder = TimeFeatureEncoder(time_dim)
        self.delta_encoder = TimeFeatureEncoder(time_dim)
        self.input_dim = 1 + time_dim + time_dim
        self.ttcn_dim = hidden_dim
        self.filter_generators = nn.Sequential(
            nn.Linear(self.input_dim, self.ttcn_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(self.ttcn_dim, self.ttcn_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(self.ttcn_dim, self.input_dim * self.ttcn_dim, bias=True),
        )
        self.t_bias = nn.Parameter(torch.randn(1, self.ttcn_dim))
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self._init_stable_parameters()

    def _init_stable_parameters(self) -> None:
        """Use a small bias scale so patch aggregation does not start overly aggressive."""
        nn.init.normal_(self.t_bias, mean=0.0, std=0.02)

    def forward(self, patch_values: Tensor, patch_times: Tensor, patch_masks: Tensor) -> Tensor:
        _, _, _, patch_len_max, _ = patch_values.shape

        time_emb = self.time_encoder(patch_times)

        if patch_len_max > 1:
            deltas = torch.zeros_like(patch_times)
            deltas[..., 1:, :] = patch_times[..., 1:, :] - patch_times[..., :-1, :]
        else:
            deltas = torch.zeros_like(patch_times)
        delta_emb = self.delta_encoder(deltas)

        event_features = torch.cat([patch_values, time_emb, delta_emb], dim=-1)
        feature_mask = patch_masks.expand_as(event_features)

        batch_size, n_var, n_patch, patch_len_max, _ = event_features.shape
        flattened_features = event_features.reshape(batch_size * n_var * n_patch, patch_len_max, self.input_dim)
        flattened_mask = feature_mask.reshape(batch_size * n_var * n_patch, patch_len_max, self.input_dim)
        patch_presence = patch_masks.reshape(batch_size * n_var * n_patch, patch_len_max, 1).sum(dim=1) > 0

        filter_logits = self.filter_generators(flattened_features)
        filter_mask = (
            flattened_mask.unsqueeze(dim=-2)
            .expand(-1, -1, self.ttcn_dim, -1)
            .reshape(batch_size * n_var * n_patch, patch_len_max, self.input_dim * self.ttcn_dim)
        )
        filter_logits = filter_logits * filter_mask + (1 - filter_mask) * (-1e8)
        filter_seqnorm = torch.softmax(filter_logits, dim=-2)
        filter_seqnorm = filter_seqnorm.view(batch_size * n_var * n_patch, patch_len_max, self.ttcn_dim, self.input_dim)

        expanded_features = flattened_features.unsqueeze(dim=-2).repeat(1, 1, self.ttcn_dim, 1)
        ttcn_out = torch.sum(torch.sum(expanded_features * filter_seqnorm, dim=-3), dim=-1)
        patch_repr = torch.relu(ttcn_out + self.t_bias)
        patch_repr = self.norm(self.dropout(patch_repr))

        patch_repr = patch_repr * patch_presence.float()
        return patch_repr.view(batch_size, n_var, n_patch, self.hidden_dim)


class PatchTemporalEncoder(nn.Module):
    """Temporal modeling over per-variable patch sequences."""

    def __init__(self, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=max(1, min(8, hidden_dim // max(1, hidden_dim // 8))),
            batch_first=True,
            dropout=dropout,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=max(1, num_layers))
        self.position_encoding = PatchPositionalEncoding(hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, patch_repr: Tensor, patch_valid: Tensor) -> Tensor:
        batch_size, n_var, n_patch, hidden_dim = patch_repr.shape
        sequence = patch_repr.reshape(batch_size * n_var, n_patch, hidden_dim)
        sequence = self.position_encoding(sequence)

        valid_mask = patch_valid.squeeze(-1).reshape(batch_size * n_var, n_patch).bool()
        key_padding_mask = ~valid_mask
        empty_rows = valid_mask.sum(dim=1) == 0
        if empty_rows.any():
            key_padding_mask[empty_rows, 0] = False

        encoded = self.encoder(sequence, src_key_padding_mask=key_padding_mask)
        encoded = self.norm(encoded)
        return encoded.reshape(batch_size, n_var, n_patch, hidden_dim)


class DynamicGraphBlock(nn.Module):
    """Time-adaptive cross-variable graph interaction on patch features."""

    def __init__(self, hidden_dim: int, node_dim: int, dropout: float):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.node_dim = node_dim
        self.base_nodevec1 = nn.Parameter(torch.randn(1, 1, 1, node_dim))
        self.base_nodevec2 = nn.Parameter(torch.randn(1, 1, 1, node_dim))
        self.node_proj1 = nn.Linear(hidden_dim, node_dim)
        self.node_proj2 = nn.Linear(hidden_dim, node_dim)
        self.update = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self._init_stable_parameters()

    def _init_stable_parameters(self) -> None:
        """Keep initial graph biases small so adjacency starts near the learned projections."""
        nn.init.normal_(self.base_nodevec1, mean=0.0, std=0.02)
        nn.init.normal_(self.base_nodevec2, mean=0.0, std=0.02)

    def forward(self, patch_repr: Tensor, patch_valid: Tensor) -> Tensor:
        features = patch_repr.permute(0, 2, 1, 3)
        batch_size, n_patch, n_var, _ = features.shape

        proj1 = self.node_proj1(features) + self.base_nodevec1.expand(batch_size, n_patch, n_var, -1)
        proj2 = self.node_proj2(features) + self.base_nodevec2.expand(batch_size, n_patch, n_var, -1)
        adjacency = torch.matmul(proj1, proj2.transpose(-1, -2))
        adjacency = torch.softmax(torch.relu(adjacency), dim=-1)

        aggregated = torch.einsum("bmdn,bmnh->bmdh", adjacency, features)
        updated = self.update(torch.cat([features, aggregated], dim=-1))
        updated = self.norm(features + self.dropout(updated))

        patch_valid_expanded = patch_valid.permute(0, 2, 1, 3)
        updated = updated * patch_valid_expanded
        return updated.permute(0, 2, 1, 3)
