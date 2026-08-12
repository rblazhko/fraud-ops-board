"""Explicit feature list for the fraud classifier."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Keep IDs / labels out. amount_z_user may be NaN (cold start) — HGB handles that.
FEATURE_COLS = [
    "amount",
    "tx_cnt_1h",
    "tx_cnt_24h",
    "amt_sum_24h",
    "burst_flag_1h",
    "shared_flag",
    "device_degree_to_date",
    "device_tx_cnt_24h",
    "device_users_24h",
    "amount_z_user",
    "cold_start_flag",
    "geo_mismatch",
    "segment_vip",
]


def build_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return model matrix with FEATURE_COLS only (copy)."""
    out = pd.DataFrame(index=df.index)
    for col in FEATURE_COLS:
        if col == "segment_vip":
            out[col] = (df["segment"].astype(str) == "vip").astype(np.float64)
            continue
        if col not in df.columns:
            raise KeyError(f"missing feature column: {col}")
        out[col] = pd.to_numeric(df[col], errors="coerce")
    return out
