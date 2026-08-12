"""Smoke checks for the transparent rule score."""

from __future__ import annotations

import pandas as pd

from board.score import WEIGHTS, precision_at_k, score_frame


def _tiny_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tx_cnt_1h": [0, 4],
            "tx_cnt_24h": [1, 10],
            "burst_flag_1h": [0, 1],
            "shared_flag": [0, 1],
            "device_degree_to_date": [1, 12],
            "device_users_24h": [0, 5],
            "amount_z_user": [0.1, 5.0],
            "cold_start_flag": [0, 0],
            "geo_mismatch": [0, 1],
            "label_fraud": [0, 1],
        }
    )


def test_score_in_unit_interval():
    out = score_frame(_tiny_frame())
    assert out["risk_score"].between(0.0, 1.0).all()
    assert out.loc[1, "risk_score"] > out.loc[0, "risk_score"]


def test_weights_sum_to_one():
    assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9


def test_precision_at_k_prefers_high_score_fraud():
    out = score_frame(_tiny_frame())
    assert precision_at_k(out, k=1) == 1.0
