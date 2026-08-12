"""Fraud injection and late chargeback labels."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd


def assign_fraud_flags(
    rng: np.random.Generator,
    tx: pd.DataFrame,
    users: pd.DataFrame,
    *,
    geo_mismatch_fraud_share: float,
    late_chargeback_share: float,
    chargeback_lag_days_min: int,
    chargeback_lag_days_max: int,
) -> pd.DataFrame:
    """Flip labels using segment risk_prior; add geo / late chargeback flavor."""
    out = tx.copy()
    prior = users.set_index("user_id")["risk_prior"]
    p = out["user_id"].map(prior).to_numpy(dtype=float)

    # ring txs already marked with ring_id get a bump
    ring = out.get("ring_id", pd.Series(-1, index=out.index)).fillna(-1).to_numpy()
    p = np.where(ring >= 0, np.clip(p * 1.8, 0.0, 0.12), p)

    draw = rng.random(len(out))
    fraud = draw < p

    # force a chunk of ring txs to be fraud so rings show up in features
    ring_mask = ring >= 0
    if ring_mask.any():
        force_n = max(1, int(ring_mask.sum() * 0.08))
        ring_idx = np.flatnonzero(ring_mask)
        pick = rng.choice(ring_idx, size=min(force_n, len(ring_idx)), replace=False)
        fraud[pick] = True

    out["label_fraud"] = fraud.astype(int)
    out["label_source"] = np.where(fraud, "rule_inject", "none")

    # geo mismatch on a share of fraud
    user_country = users.set_index("user_id")["country"]
    uc = out["user_id"].map(user_country)
    fraud_idx = np.flatnonzero(fraud)
    if len(fraud_idx):
        n_geo = int(round(len(fraud_idx) * geo_mismatch_fraud_share))
        geo_pick = rng.choice(fraud_idx, size=min(n_geo, len(fraud_idx)), replace=False)
        for i in geo_pick:
            alt = [c for c in ("DE", "FR", "NL", "PL", "ES", "IT", "GB", "US") if c != uc.iloc[i]]
            out.at[out.index[i], "country"] = rng.choice(alt)

    # late chargebacks
    chargeback_ts = [pd.NaT] * len(out)
    if len(fraud_idx):
        n_late = int(round(len(fraud_idx) * late_chargeback_share))
        late_pick = rng.choice(fraud_idx, size=min(n_late, len(fraud_idx)), replace=False)
        for i in late_pick:
            lag = int(rng.integers(chargeback_lag_days_min, chargeback_lag_days_max + 1))
            ts = pd.Timestamp(out.at[out.index[i], "ts"])
            chargeback_ts[i] = ts + timedelta(days=lag)
            out.at[out.index[i], "label_source"] = "chargeback"

    out["chargeback_ts"] = chargeback_ts
    # drop helper col if present
    if "ring_id" in out.columns:
        out = out.drop(columns=["ring_id"])
    return out
