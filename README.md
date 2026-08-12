# Fraud Ops Board

Portfolio skeleton: **synthetic** antifraud ops data + SQL feature views.

Not a production risk system. Not trained on real cardholder or merchant data.
Everything under `synth/` and `data/` is fake — built so recruiters can see Python + SQL
working together on an imbalanced fraud-ops style problem.

## What this is

- Reproducible event generation (users, devices, transactions) from `seed.yaml`
- DuckDB SQL feature views: velocity, device reuse, amount z-score
- A thin Python CLI to materialize those features as parquet
- Tests for fraud-rate band, VIP vs mass priors, and seed reproducibility

## What this is not

- A live monitoring UI (Streamlit board comes later)
- An ML model package (next step after this skeleton)
- A copy of any employer pipeline or NDA dataset

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

make data          # write CSVs under data/
make features      # DuckDB: load tables, run sql/*.sql, write feature parquet
make test
```

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
| `DATA.md` | Schema, seeds, generation rules |

## Status

First commit is data + SQL only. Model scoring and a Streamlit ops board are intentional follow-ups.
