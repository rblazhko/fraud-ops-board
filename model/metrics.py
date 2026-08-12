"""Eval helpers: PR-AUC, ROC-AUC, precision@k."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def precision_at_k(
    y_true: np.ndarray | pd.Series,
    scores: np.ndarray | pd.Series,
    k: int,
) -> float:
    """Label precision among the top-k by score."""
    y = np.asarray(y_true)
    s = np.asarray(scores)
    if k <= 0 or len(y) == 0:
        return float("nan")
    k_eff = min(k, len(y))
    idx = np.argpartition(-s, k_eff - 1)[:k_eff]
    return float(np.mean(y[idx]))


def classification_report_dict(
    y_true: np.ndarray | pd.Series,
    scores: np.ndarray | pd.Series,
    ks: tuple[int, ...] = (50, 100),
) -> dict[str, Any]:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    out: dict[str, Any] = {
        "n": int(len(y)),
        "fraud_rate": float(y.mean()) if len(y) else float("nan"),
        "pr_auc": float("nan"),
        "roc_auc": float("nan"),
    }
    if len(y) and y.min() != y.max():
        out["pr_auc"] = float(average_precision_score(y, s))
        out["roc_auc"] = float(roc_auc_score(y, s))
    for k in ks:
        out[f"precision_at_{k}"] = precision_at_k(y, s, k)
    return out
