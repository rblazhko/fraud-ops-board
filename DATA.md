# Data notes (synthetic)

All tables are **fake**. No production IDs, no real PSPs, no employer schemas.

## Seed

`seed.yaml` holds `master_seed` and a small scenario block (n_users, date window, fraud target band).

Generation uses:

```python
rng = np.random.Generator(np.random.PCG64(master_seed))
```

Same seed → same CSVs (asserted in `tests/test_synth_invariants.py`).

Optional PRNG validation is out of band: see [rng-diagnostics](https://github.com/rblazhko/rng-diagnostics).
This repo only checks reproducibility and documents how to call that CLI.

## Tables

### users
| column | notes |
|--------|--------|
| user_id | `U000001` … |
| segment | `vip` \| `mass` |
| tenure_days | how long the account has existed |
| country | ISO-ish codes from a small fixed list |
| risk_prior | soft prior used when injecting labels (VIP lower than mass) |

### devices
| column | notes |
|--------|--------|
| device_id | `D000001` … |
| first_seen | first appearance in the window |
| shared_flag | 1 if device is part of a reuse ring |

### transactions
| column | notes |
|--------|--------|
| tx_id | `T00000001` … |
| ts | timestamp inside the scenario window |
| user_id / device_id | FKs |
| amount | lognormal-ish; VIP median higher |
| currency | mostly EUR/USD |
| country | tx country (may mismatch user for some fraud) |
| channel | web / app / pos |
| psp_id | synthetic `PSP_A` … `PSP_D` |
| segment | denormalized from user |
| auth_result | approved / declined |
| label_fraud | 0/1 |
| label_source | `none` \| `rule_inject` \| `chargeback` |
| chargeback_ts | nullable; late labels for a subset of fraud |

## Generation story (one coherent prior)

- Overall fraud rate aimed at **~0.5–2%** (scenario knobs in `seed.yaml`).
- **VIP fraud prior lower than mass** — VIP still gets some ring / geo attacks, just rarer.
- Amounts ~ lognormal; VIP shift upward.
- Fraud rings: short bursts of txs + reused devices (`shared_flag`).
- A slice of fraud has **country ≠ user.country**.
- Window: typically 60–90 days ending “today” relative to generation time (fixed from seed for reproducibility of *relative* offsets; absolute dates are generated from a fixed anchor in code so runs stay bit-stable).
- Late labels: some fraud gets `chargeback_ts` days after `ts`; until then a naive as-of join would miss them — intentional.

## Outputs

`make data` writes:

- `data/users.csv`
- `data/devices.csv`
- `data/transactions.csv`

`make features` writes parquet under `data/features_*.parquet` from the SQL views.

`make train` fits `HistGradientBoostingClassifier` on `features_wide.parquet`
with a time-quantile split (train past / eval future), writes
`artifacts/model.joblib` and `reports/metrics.{json,md}`.

Large files are gitignored. Regenerate on demand.
