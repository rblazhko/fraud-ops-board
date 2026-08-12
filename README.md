# Fraud Ops Board

Portfolio skeleton: **synthetic** antifraud ops data + SQL feature views + a
classical sklearn fraud score on the Streamlit desk.

Not a production risk system. Not trained on real cardholder or merchant data.
Everything under `synth/` and `data/` is fake — built so recruiters can see Python + SQL
(+ a boring ML baseline) working together on an imbalanced fraud-ops style problem.

## What this is

- Reproducible event generation (users, devices, transactions) from `seed.yaml`
- DuckDB SQL feature views: velocity, device reuse, amount z-score
- A thin Python CLI to materialize those features as parquet
- `HistGradientBoostingClassifier` trained on those features with a **time-based**
  train/eval split (no future leak); metrics under `reports/`
- Local Streamlit ops board: ML queue + rule-score baseline for comparison
- Tests for fraud-rate band, VIP vs mass priors, seed reproducibility, and model smoke

## What this is not

- A production monitoring stack or deep-learning fraud model
- A copy of any employer pipeline or NDA dataset

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

make data          # write CSVs under data/
make features      # DuckDB: load tables, run sql/*.sql, write feature parquet
make train         # fit sklearn model → artifacts/model.joblib + reports/metrics.*
make test
make board         # Streamlit ops board (opens locally)
```

### Model

```bash
make train
# or: python -m model.train --features data/features_wide.parquet
```

Trains on the earlier ~70% of the timeline, evaluates on the later 30%.
Imbalance handling: balanced `sample_weight` (minority class up-weighted).
Rule score in `board/score.py` stays as a transparent baseline — not replaced.

### Ops board

```bash
make board
# or: streamlit run board/app.py
```

Requires `make train` first. Desk view: headline KPIs on **ML score**, ML vs rule
precision@k on the eval window, VIP vs mass, trends, and a queue ranked by ML
score (rule score shown beside it).

Filters: date range, segment (vip / mass / all), min ML score.

Optional PRNG check (reproducibility smoke, or [rng-diagnostics](https://github.com/rblazhko/rng-diagnostics) if installed):

```bash
make validate-rng
```

## Layout

| Path | Role |
|------|------|
| `synth/` | Synthetic generator |
| `sql/` | DuckDB feature views |
| `scripts/` | Feature runner + RNG check |
| `model/` | Train / predict / metrics (sklearn) |
| `board/` | Streamlit ops board + rule baseline |
| `artifacts/` | `model.joblib` (gitignored; `make train`) |
| `reports/` | Eval metrics JSON/MD from training |
| `DATA.md` | Schema, seeds, generation rules |

## Status

Data + SQL features + sklearn fraud score + local ops board.
