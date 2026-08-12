"""Train HistGradientBoostingClassifier on SQL features with a time split.

No future leak: train on txs with ts <= cutoff, evaluate on ts > cutoff.
Imbalance: class_weight='balanced' (sklearn scales minority upward).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight

from board.score import score_frame as rule_score_frame
from model.features import FEATURE_COLS, build_matrix
from model.metrics import classification_report_dict

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURES = ROOT / "data" / "features_wide.parquet"
DEFAULT_ARTIFACT = ROOT / "artifacts" / "model.joblib"
DEFAULT_METRICS = ROOT / "reports" / "metrics.json"


def time_split(
    df: pd.DataFrame, train_frac: float = 0.7
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Split by timestamp quantile — train past, eval future."""
    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac must be in (0, 1)")
    ts = pd.to_datetime(df["ts"])
    cutoff = ts.quantile(train_frac)
    # Tie-break at exact cutoff: earlier rows train, later eval
    train = df.loc[ts <= cutoff].copy()
    eval_df = df.loc[ts > cutoff].copy()
    if train.empty or eval_df.empty:
        raise ValueError("time split produced empty train or eval")
    return train, eval_df, pd.Timestamp(cutoff)


def fit_model(train: pd.DataFrame, random_state: int = 42) -> HistGradientBoostingClassifier:
    X = build_matrix(train)[FEATURE_COLS]
    y = train["label_fraud"].astype(int).to_numpy()
    # HGB has no class_weight kwarg; balanced sample weights are equivalent here.
    sw = compute_sample_weight("balanced", y)
    clf = HistGradientBoostingClassifier(
        max_depth=4,
        max_iter=120,
        learning_rate=0.08,
        min_samples_leaf=40,
        l2_regularization=0.1,
        random_state=random_state,
    )
    clf.fit(X, y, sample_weight=sw)
    return clf


def evaluate(model: HistGradientBoostingClassifier, frame: pd.DataFrame) -> dict[str, Any]:
    X = build_matrix(frame)[FEATURE_COLS]
    scores = model.predict_proba(X)[:, 1]
    return classification_report_dict(frame["label_fraud"], scores, ks=(50, 100))


def evaluate_rule_baseline(frame: pd.DataFrame) -> dict[str, Any]:
    """Same eval slice, scored by the transparent rule baseline (board/score.py)."""
    scored = rule_score_frame(frame)
    return classification_report_dict(scored["label_fraud"], scored["risk_score"], ks=(50, 100))


def train_and_persist(
    features_path: Path = DEFAULT_FEATURES,
    artifact_path: Path = DEFAULT_ARTIFACT,
    metrics_path: Path = DEFAULT_METRICS,
    train_frac: float = 0.7,
    random_state: int = 42,
) -> dict[str, Any]:
    if not features_path.exists():
        raise FileNotFoundError(
            f"Missing {features_path}. Run `make data && make features` first."
        )
    df = pd.read_parquet(features_path)
    df["ts"] = pd.to_datetime(df["ts"])
    train, eval_df, cutoff = time_split(df, train_frac=train_frac)

    model = fit_model(train, random_state=random_state)
    train_metrics = evaluate(model, train)
    eval_metrics = evaluate(model, eval_df)
    rule_metrics = evaluate_rule_baseline(eval_df)

    bundle = {
        "model": model,
        "feature_cols": list(FEATURE_COLS),
        "train_frac": train_frac,
        "cutoff_ts": cutoff.isoformat(),
        "random_state": random_state,
        "n_train": int(len(train)),
        "n_eval": int(len(eval_df)),
    }
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, artifact_path)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "features_path": str(features_path),
        "artifact_path": str(artifact_path),
        "model": "HistGradientBoostingClassifier",
        "imbalance": "balanced sample_weight",
        "split": {
            "kind": "time_quantile",
            "train_frac": train_frac,
            "cutoff_ts": cutoff.isoformat(),
            "n_train": int(len(train)),
            "n_eval": int(len(eval_df)),
            "train_ts_min": train["ts"].min().isoformat(),
            "train_ts_max": train["ts"].max().isoformat(),
            "eval_ts_min": eval_df["ts"].min().isoformat(),
            "eval_ts_max": eval_df["ts"].max().isoformat(),
        },
        "train": train_metrics,
        "eval": eval_metrics,
        "rule_baseline_eval": rule_metrics,
        "feature_cols": list(FEATURE_COLS),
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(report, indent=2) + "\n")

    md_path = metrics_path.with_suffix(".md")
    md_path.write_text(_metrics_markdown(report))
    return report


def _metrics_markdown(report: dict[str, Any]) -> str:
    ev = report["eval"]
    tr = report["train"]
    ru = report["rule_baseline_eval"]
    sp = report["split"]
    lines = [
        "# Fraud model metrics",
        "",
        f"- Model: `{report['model']}`",
        f"- Imbalance: {report['imbalance']}",
        f"- Time split: train_frac={sp['train_frac']}, cutoff=`{sp['cutoff_ts']}`",
        f"- Rows: train={sp['n_train']}, eval={sp['n_eval']}",
        "",
        "## ML vs rule baseline, same eval window",
        "",
        "| Scorer | PR-AUC | ROC-AUC | P@50 | P@100 |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| ML ({report['model']}) | **{ev['pr_auc']:.4f}** | {ev['roc_auc']:.4f} "
        f"| {ev['precision_at_50']:.4f} | {ev['precision_at_100']:.4f} |",
        f"| Rule baseline | {ru['pr_auc']:.4f} | {ru['roc_auc']:.4f} "
        f"| {ru['precision_at_50']:.4f} | {ru['precision_at_100']:.4f} |",
        "",
        f"Fraud rate on eval: {ev['fraud_rate']:.4f} (n={ev['n']}). The rule score "
        "is a fixed weighted blend (`board/score.py`), kept as a transparent, "
        "non-learned baseline — the model earns its place by beating it on the "
        "same future slice, not by construction.",
        "",
        "## Train (reference, ML only)",
        f"- PR-AUC: {tr['pr_auc']:.4f}",
        f"- ROC-AUC: {tr['roc_auc']:.4f}",
        f"- Precision@50: {tr['precision_at_50']:.4f}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Train fraud classifier on feature parquet")
    p.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    p.add_argument("--out", type=Path, default=DEFAULT_ARTIFACT)
    p.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    report = train_and_persist(
        features_path=args.features,
        artifact_path=args.out,
        metrics_path=args.metrics,
        train_frac=args.train_frac,
        random_state=args.seed,
    )
    ev = report["eval"]
    print(
        f"wrote {args.out}\n"
        f"eval PR-AUC={ev['pr_auc']:.4f} ROC-AUC={ev['roc_auc']:.4f} "
        f"P@50={ev['precision_at_50']:.4f} P@100={ev['precision_at_100']:.4f}\n"
        f"metrics → {args.metrics}"
    )


if __name__ == "__main__":
    main()
