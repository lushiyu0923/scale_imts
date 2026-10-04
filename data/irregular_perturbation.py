"""Controlled perturbations for robustness experiments on irregular inputs.

The functions in this module operate on a collated batch. They only modify
historical observations (``x`` and ``x_mask``); timestamps, targets, labels,
and dataset files are left unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor


PERTURBATION_MODES = ("none", "random", "local_scale", "block_missing")
PERTURBATION_SPLITS = ("all", "train", "val", "test")


def _sample_seed(base_seed: int, sample_id: Tensor | None, index: int) -> int:
    """Create a stable per-sample seed independent of batch order."""
    if sample_id is None:
        identity = index
    else:
        identity = int(round(float(sample_id[index].item())))
    return (int(base_seed) + 1000003 * identity) % (2**63 - 1)


def _generator(seed: int, device: torch.device) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def _random_scores(shape: tuple[int, ...], seed: int, device: torch.device) -> Tensor:
    return torch.rand(shape, device=device, generator=_generator(seed, device))


def _drop_flat_indices(mask: Tensor, scores: Tensor, rate: float, min_observations: int) -> Tensor:
    valid_indices = torch.where(mask.reshape(-1))[0]
    n_valid = int(valid_indices.numel())
    n_drop = min(int(rate * n_valid), max(0, n_valid - min_observations))
    if n_drop <= 0:
        return valid_indices[:0]
    selected = torch.topk(scores.reshape(-1)[valid_indices], k=n_drop, largest=True).indices
    return valid_indices[selected]


def _apply_random(
    x: Tensor,
    mask: Tensor,
    rate: float,
    seed: int,
    min_observations: int,
) -> None:
    scores = _random_scores(tuple(mask.shape), seed, mask.device)
    selected = _drop_flat_indices(mask, scores, rate, min_observations)
    mask.reshape(-1)[selected] = 0
    x.reshape(-1)[selected] = 0


def _apply_local_scale(
    x: Tensor,
    mask: Tensor,
    times: Tensor,
    rate: float,
    seed: int,
    min_observations: int,
    contrast: float,
) -> None:
    valid_rows = mask.any(dim=-1)
    valid_times = times[valid_rows]
    if valid_times.numel() == 0:
        return

    time_min = valid_times.min()
    time_max = valid_times.max()
    span = (time_max - time_min).clamp_min(torch.finfo(times.dtype).eps)
    random_value = _random_scores((1,), seed, times.device)[0]
    boundary = time_min + span * (0.35 + 0.30 * random_value)
    sparse_left = bool(_random_scores((1,), seed + 1, times.device)[0] > 0.5)
    sparse_region = times < boundary if sparse_left else times >= boundary

    # Add a deterministic priority to one temporal region. Selecting an exact
    # number of values keeps the effective removal rate comparable to random.
    scores = _random_scores(tuple(mask.shape), seed + 2, mask.device)
    scores = scores + contrast * sparse_region.unsqueeze(-1).to(scores.dtype)
    selected = _drop_flat_indices(mask, scores, rate, min_observations)
    mask.reshape(-1)[selected] = 0
    x.reshape(-1)[selected] = 0


def _apply_block_missing(
    x: Tensor,
    mask: Tensor,
    times: Tensor,
    rate: float,
    seed: int,
    min_observations: int,
) -> None:
    valid_rows = mask.any(dim=-1)
    valid_times = times[valid_rows]
    if valid_times.numel() == 0:
        return

    time_min = valid_times.min()
    time_max = valid_times.max()
    span = (time_max - time_min).clamp_min(torch.finfo(times.dtype).eps)
    block_width = span * rate
    start_max = (time_max - block_width).clamp_min(time_min)
    random_value = _random_scores((1,), seed, times.device)[0]
    start = time_min + (start_max - time_min) * random_value
    end = start + block_width

    candidate = mask & ((times[:, None] >= start) & (times[:, None] <= end))
    candidate_indices = torch.where(candidate.reshape(-1))[0]
    n_valid = int(mask.sum().item())
    max_drop = max(0, n_valid - min_observations)
    if candidate_indices.numel() > max_drop:
        scores = _random_scores(tuple(candidate.shape), seed + 1, mask.device)
        candidate_indices = candidate_indices[
            torch.topk(scores.reshape(-1)[candidate_indices], k=max_drop, largest=True).indices
        ]
    mask.reshape(-1)[candidate_indices] = 0
    x.reshape(-1)[candidate_indices] = 0


def apply_observation_perturbation(
    batch: Mapping[str, Tensor],
    *,
    mode: str = "none",
    rate: float = 0.0,
    seed: int = 2026,
    min_observations: int = 1,
    local_scale_contrast: float = 0.75,
) -> dict[str, Tensor]:
    """Apply a deterministic perturbation to a collated input batch.

    Parameters
    ----------
    batch:
        Collated batch containing ``x``, ``x_mark`` and ``x_mask``.
    mode:
        ``none``, ``random``, ``local_scale`` or ``block_missing``.
    rate:
        Fraction of valid variable observations to remove. It must be in
        ``[0, 1)``; a minimum number of observations is always retained.
    seed:
        Base seed. ``sample_ID`` is used when available to make perturbations
        invariant to DataLoader batch order.
    min_observations:
        Minimum number of valid variable observations retained per sample.
    local_scale_contrast:
        Priority added to the sparse temporal region in ``local_scale`` mode.
    """
    normalized_mode = str(mode).lower()
    if normalized_mode not in PERTURBATION_MODES:
        raise ValueError(f"Unsupported perturbation mode: {mode}. Choose from {PERTURBATION_MODES}.")
    if not 0.0 <= float(rate) < 1.0:
        raise ValueError(f"perturbation rate must be in [0, 1), got {rate}.")
    if min_observations < 0:
        raise ValueError("min_observations must be non-negative.")
    if normalized_mode == "none" or rate == 0.0:
        return dict(batch)

    required = ("x", "x_mark", "x_mask")
    missing = [key for key in required if key not in batch]
    if missing:
        raise KeyError(f"Batch is missing perturbation fields: {missing}")

    x = batch["x"].clone()
    mask = batch["x_mask"].clone()
    x_mark = batch["x_mark"]
    if x.ndim != 3 or mask.ndim != 3 or x_mark.ndim < 2:
        raise ValueError(
            f"Expected x/x_mask [B,L,V] and x_mark [B,L,...], got "
            f"{tuple(x.shape)}, {tuple(mask.shape)}, {tuple(x_mark.shape)}"
        )
    if x.shape != mask.shape or x.shape[:2] != x_mark.shape[:2]:
        raise ValueError("x, x_mask and x_mark have incompatible shapes.")

    sample_ids = batch.get("sample_ID")
    if sample_ids is not None:
        sample_ids = sample_ids.reshape(-1)
        if sample_ids.numel() != x.shape[0]:
            sample_ids = None

    for index in range(x.shape[0]):
        sample_seed = _sample_seed(seed, sample_ids, index)
        sample_x = x[index]
        sample_mask = mask[index].bool()
        sample_times = x_mark[index, :, 0] if x_mark.ndim == 3 else x_mark[index]
        if normalized_mode == "random":
            _apply_random(sample_x, sample_mask, rate, sample_seed, min_observations)
        elif normalized_mode == "local_scale":
            _apply_local_scale(
                sample_x,
                sample_mask,
                sample_times,
                rate,
                sample_seed,
                min_observations,
                local_scale_contrast,
            )
        else:
            _apply_block_missing(sample_x, sample_mask, sample_times, rate, sample_seed, min_observations)
        mask[index] = sample_mask.to(dtype=mask.dtype)

    perturbed = dict(batch)
    perturbed["x"] = x
    perturbed["x_mask"] = mask
    return perturbed


def perturbation_summary(before: Mapping[str, Tensor], after: Mapping[str, Tensor]) -> dict[str, float]:
    """Return aggregate observation statistics for a before/after batch pair."""
    before_count = before["x_mask"].float().sum()
    after_count = after["x_mask"].float().sum()
    removed = before_count - after_count
    return {
        "before_observations": float(before_count.item()),
        "after_observations": float(after_count.item()),
        "removed_observations": float(removed.item()),
        "removal_rate": float((removed / before_count.clamp_min(1.0)).item()),
    }
