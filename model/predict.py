"""Load a trained bundle and attach ML scores to a feature frame."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from model.features import FEATURE_COLS, build_matrix

DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "artifacts" / "model.joblib"


def load_bundle(path: Path | str = DEFAULT_MODEL) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run `make train` after `make features`."
        )
    bundle = joblib.load(path)
    if "model" not in bundle or "feature_cols" not in bundle:
        raise ValueError(f"invalid model bundle at {path}")
    return bundle


def predict_proba(df: pd.DataFrame, bundle: dict[str, Any]) -> pd.Series:
    cols = list(bundle["feature_cols"])
    if cols != FEATURE_COLS:
        # Allow older bundles only if they match current FEATURE_COLS order/set.
        missing = set(FEATURE_COLS) - set(cols)
        if missing:
            raise ValueError(f"bundle missing features: {sorted(missing)}")
    X = build_matrix(df)[FEATURE_COLS]
    proba = bundle["model"].predict_proba(X)[:, 1]
    return pd.Series(proba, index=df.index, name="ml_score")


def score_frame(df: pd.DataFrame, bundle: dict[str, Any] | None = None) -> pd.DataFrame:
    """Return a copy with `ml_score` (P(fraud)). Loads default artifact if needed."""
    if bundle is None:
        bundle = load_bundle()
    out = df.copy()
    out["ml_score"] = predict_proba(out, bundle)
    return out
