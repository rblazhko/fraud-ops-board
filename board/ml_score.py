"""Board helpers for ML scores (wraps model.predict)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from model.predict import DEFAULT_MODEL, load_bundle, score_frame

__all__ = ["DEFAULT_MODEL", "attach_ml_scores", "model_available"]


def model_available(path: Path | None = None) -> bool:
    return (path or DEFAULT_MODEL).exists()


def attach_ml_scores(df: pd.DataFrame, path: Path | None = None) -> pd.DataFrame:
    bundle = load_bundle(path or DEFAULT_MODEL)
    return score_frame(df, bundle)
