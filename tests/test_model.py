"""Tests for model feature matrix, metrics, and tiny-data training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from model.features import FEATURE_COLS, build_matrix
from model.metrics import classification_report_dict, precision_at_k
from model.predict import predict_proba, score_frame
from model.train import fit_model, time_split, train_and_persist


def _tiny_labeled(n: int = 80) -> pd.DataFrame:
    ts0 = pd.Timestamp("2025-01-01")
    rows = []
    for i in range(n):
        fraud = int(i % 10 == 0)
        rows.append(
            {
                "tx_id": f"T{i:04d}",
                "ts": ts0 + pd.Timedelta(hours=i),
                "segment": "vip" if i % 5 == 0 else "mass",
                "amount": float(10 + i + 40 * fraud),
                "tx_cnt_1h": int(1 + 3 * fraud),
                "tx_cnt_24h": int(2 + 5 * fraud),
                "amt_sum_24h": float(20 + 80 * fraud),
                "burst_flag_1h": fraud,
                "shared_flag": fraud,
                "device_degree_to_date": int(1 + 4 * fraud),
                "device_tx_cnt_24h": int(1 + 3 * fraud),
                "device_users_24h": int(fraud * 2),
                "amount_z_user": np.nan if i % 17 == 0 else float(0.2 + 2.5 * fraud),
                "cold_start_flag": int(i % 17 == 0),
                "geo_mismatch": fraud,
                "label_fraud": fraud,
            }
        )
    return pd.DataFrame(rows)


def test_build_matrix_shape_and_segment():
    df = _tiny_labeled(12)
    X = build_matrix(df)
    assert list(X.columns) == FEATURE_COLS
    assert len(X) == 12
    assert set(X["segment_vip"].unique()).issubset({0.0, 1.0})


def test_precision_at_k_and_report():
    y = np.array([0, 1, 0, 1, 0])
    s = np.array([0.1, 0.9, 0.2, 0.8, 0.05])
    assert precision_at_k(y, s, k=2) == 1.0
    report = classification_report_dict(y, s, ks=(2,))
    assert report["precision_at_2"] == 1.0
    assert 0.0 <= report["pr_auc"] <= 1.0


def test_time_split_no_future_leak():
    df = _tiny_labeled(40)
    train, eval_df, cutoff = time_split(df, train_frac=0.7)
    assert train["ts"].max() <= cutoff
    assert eval_df["ts"].min() > cutoff
    assert len(train) + len(eval_df) == len(df)


def test_fit_and_predict_shape(tmp_path: Path):
    df = _tiny_labeled(100)
    train, eval_df, _ = time_split(df, train_frac=0.7)
    model = fit_model(train, random_state=0)
    bundle = {"model": model, "feature_cols": list(FEATURE_COLS)}
    proba = predict_proba(eval_df, bundle)
    assert len(proba) == len(eval_df)
    assert proba.between(0.0, 1.0).all()

    scored = score_frame(eval_df, bundle)
    assert "ml_score" in scored.columns

    feat = tmp_path / "features.parquet"
    art = tmp_path / "model.joblib"
    metrics = tmp_path / "metrics.json"
    df.to_parquet(feat)
    report = train_and_persist(
        features_path=feat,
        artifact_path=art,
        metrics_path=metrics,
        train_frac=0.7,
        random_state=0,
    )
    assert art.exists()
    assert metrics.exists()
    assert "pr_auc" in report["eval"]
    assert report["eval"]["n"] > 0
