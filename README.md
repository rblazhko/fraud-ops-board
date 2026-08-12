# Fraud Ops Board

Portfolio skeleton: **synthetic** antifraud ops data + SQL feature views.

Not a production risk system. Not trained on real cardholder or merchant data.
Everything under `synth/` and `data/` is fake — built so recruiters can see Python + SQL
working together on an imbalanced fraud-ops style problem.

## What this is

- Reproducible event generation (users, devices, transactions) from `seed.yaml`
- DuckDB SQL feature views: velocity, device reuse, amount z-score
- A thin Python CLI to materialize those features as parquet
- Local Streamlit ops board (rule score + VIP/mass queue) over `features_wide.parquet`
- Tests for fraud-rate band, VIP vs mass priors, and seed reproducibility

## What this is not

- A production monitoring stack or trained fraud model
- A copy of any employer pipeline or NDA dataset

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

make data          # write CSVs under data/
make features      # DuckDB: load tables, run sql/*.sql, write feature parquet
make test
make board         # Streamlit ops board (opens locally)
```

### Ops board

```bash
make board
# or: streamlit run board/app.py
```

Opens a single desk view: headline metrics, VIP vs mass, daily volume/rates,
score histogram, and a sortable alert queue. Risk score is a documented
weighted blend of SQL features (`board/score.py`) — not ML.

Filters: date range, segment (vip / mass / all), min score.

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
| `board/` | Streamlit ops board + rule score |
| `DATA.md` | Schema, seeds, generation rules |

## Status

Data + SQL features + local ops board. Trained model scoring is the natural next step.
