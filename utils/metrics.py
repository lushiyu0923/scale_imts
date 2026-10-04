from __future__ import annotations

"""评估指标与多次重复实验结果汇总。"""

import numpy as np


def mae(pred: np.ndarray, true: np.ndarray, mask: np.ndarray | None = None) -> float:
    """计算 MAE；如果提供 mask，只统计有效位置。"""
    if mask is None:
        return float(np.mean(np.abs(pred - true)))
    residual = (pred - true) * mask
    num_eval = float(np.sum(mask))
    return float(np.sum(np.abs(residual)) / (num_eval if num_eval > 0 else 1.0))


def mse(pred: np.ndarray, true: np.ndarray, mask: np.ndarray | None = None) -> float:
    """计算 MSE；如果提供 mask，只统计有效位置。"""
    if mask is None:
        return float(np.mean((pred - true) ** 2))
    residual = (pred - true) * mask
    num_eval = float(np.sum(mask))
    return float(np.sum(residual ** 2) / (num_eval if num_eval > 0 else 1.0))


def metric(pred: np.ndarray, true: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    """统一返回当前实验里常用的基础指标。"""
    return {
        "MAE": mae(pred, true, mask),
        "MSE": mse(pred, true, mask),
    }


def round_metric_dict(metrics: dict[str, float] | None, ndigits: int = 4) -> dict[str, float] | None:
    """把 metric 字典格式化到固定小数位，仅用于展示与日志输出。"""
    if metrics is None:
        return None
    return {str(name): round(float(value), ndigits) for name, value in metrics.items()}


def round_run_summary(summary: dict[str, dict[str, float]], ndigits: int = 4) -> dict[str, dict[str, float]]:
    """把多次重复实验的汇总结果格式化到固定小数位，仅用于展示。"""
    rounded: dict[str, dict[str, float]] = {}
    for metric_name, values in summary.items():
        rounded[metric_name] = {
            str(key): round(float(value), ndigits)
            for key, value in values.items()
        }
    return rounded


def _one_hot(labels: np.ndarray, n_classes: int) -> np.ndarray:
    eye = np.eye(n_classes, dtype=np.float64)
    return eye[np.clip(labels.astype(np.int64), 0, n_classes - 1)]


def classification_metric(logits: np.ndarray, labels: np.ndarray, n_classes: int) -> dict[str, float]:
    """计算 Acc / AUROC / AUPRC / Precision / Recall / F1。"""

    from sklearn import metrics as sk_metrics

    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels).reshape(-1).astype(np.int64)
    if logits.ndim == 1:
        logits = logits.reshape(-1, 1)
    logits = np.nan_to_num(logits)
    y_pred = np.argmax(logits, axis=1)
    max_logit = np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits - max_logit)
    probs = exp_logits / np.clip(exp_logits.sum(axis=1, keepdims=True), 1e-8, None)

    acc = float(np.mean(y_pred == labels)) if labels.size else 0.0
    if n_classes == 2:
        pos_scores = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
        precision = float(sk_metrics.precision_score(labels, y_pred, zero_division=0))
        recall = float(sk_metrics.recall_score(labels, y_pred, zero_division=0))
        f1 = float(sk_metrics.f1_score(labels, y_pred, zero_division=0))
        try:
            auroc = float(sk_metrics.roc_auc_score(labels, pos_scores))
        except ValueError:
            auroc = 0.5
        try:
            auprc = float(sk_metrics.average_precision_score(labels, pos_scores))
        except ValueError:
            auprc = float(np.mean(labels))
    else:
        precision = float(sk_metrics.precision_score(labels, y_pred, average="macro", zero_division=0))
        recall = float(sk_metrics.recall_score(labels, y_pred, average="macro", zero_division=0))
        f1 = float(sk_metrics.f1_score(labels, y_pred, average="macro", zero_division=0))
        one_hot = _one_hot(labels, n_classes)
        try:
            auroc = float(sk_metrics.roc_auc_score(one_hot, probs, average="macro", multi_class="ovr"))
        except ValueError:
            auroc = 0.5
        try:
            auprc = float(sk_metrics.average_precision_score(one_hot, probs, average="macro"))
        except ValueError:
            auprc = 1.0 / max(n_classes, 1)
    return {
        "Acc": acc,
        "AUROC": auroc,
        "AUPRC": auprc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
    }


def summarize_runs(run_metrics: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """汇总多次重复实验的均值、标准差、最小值和最大值。"""
    if not run_metrics:
        return {}

    summary: dict[str, dict[str, float]] = {}
    metric_names = run_metrics[0].keys()
    for name in metric_names:
        values = np.array([metrics[name] for metrics in run_metrics], dtype=np.float64)
        summary[name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return summary
