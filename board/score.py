"""Rule-based risk score from SQL features (not ML). Clipped to [0, 1].

Components (~[0, 1] with soft caps): velocity, device reuse, amount z, geo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Fixed weights — sum to 1.0. Device reuse carries more weight because
# rings are a primary synth fraud story; velocity is weaker in this seed.
WEIGHTS = {
    "velocity": 0.22,
    "device": 0.38,
    "amount": 0.25,
    "geo": 0.15,
}


def _clip01(x: pd.Series | np.ndarray) -> pd.Series:
    return pd.Series(x, index=getattr(x, "index", None)).clip(0.0, 1.0)


def _velocity_component(df: pd.DataFrame) -> pd.Series:
    # Soft caps: 5 txs/1h or 12/24h already look like an ops spike.
    dense_1h = _clip01(df["tx_cnt_1h"].fillna(0) / 5.0)
    dense_24h = _clip01(df["tx_cnt_24h"].fillna(0) / 12.0)
    burst = df["burst_flag_1h"].fillna(0).astype(float)
    return _clip01(0.45 * dense_1h + 0.35 * dense_24h + 0.20 * burst)


def _device_component(df: pd.DataFrame) -> pd.Series:
    shared = df["shared_flag"].fillna(0).astype(float)
    degree = _clip01((df["device_degree_to_date"].fillna(1) - 1) / 8.0)
    multi = _clip01(df["device_users_24h"].fillna(0) / 4.0)
    return _clip01(0.40 * shared + 0.35 * degree + 0.25 * multi)


def _amount_component(df: pd.DataFrame) -> pd.Series:
    z = df["amount_z_user"]
    # |z| >= 4 is already extreme vs user history
    from_z = _clip01(z.abs().fillna(0) / 4.0)
    cold = df["cold_start_flag"].fillna(0).astype(float)
    # Cold-start: no z — mild uncertainty, not a hard flag
    return _clip01(np.where(z.isna(), 0.12 * cold, from_z))


def _geo_component(df: pd.DataFrame) -> pd.Series:
    return df["geo_mismatch"].fillna(0).astype(float)


def score_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with component columns and `risk_score` in [0, 1]."""
    out = df.copy()
    out["comp_velocity"] = _velocity_component(out)
    out["comp_device"] = _device_component(out)
    out["comp_amount"] = _amount_component(out)
    out["comp_geo"] = _geo_component(out)
    out["risk_score"] = (
        WEIGHTS["velocity"] * out["comp_velocity"]
        + WEIGHTS["device"] * out["comp_device"]
        + WEIGHTS["amount"] * out["comp_amount"]
        + WEIGHTS["geo"] * out["comp_geo"]
    ).clip(0.0, 1.0)
    return out


def precision_at_k(df: pd.DataFrame, k: int = 50, score_col: str = "risk_score") -> float:
    """Label precision among the top-k rows by score (ops alert quality proxy)."""
    if k <= 0 or df.empty or "label_fraud" not in df.columns:
        return float("nan")
    top = df.nlargest(min(k, len(df)), score_col)
    return float(top["label_fraud"].mean())
