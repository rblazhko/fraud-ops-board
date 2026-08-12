"""Basic invariants for the synthetic generator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from synth.generate import generate, load_seed, make_rng

ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "seed.yaml"


def test_fraud_rate_in_band(tmp_path: Path):
    generate(SEED, tmp_path)
    tx = pd.read_csv(tmp_path / "transactions.csv")
    cfg = load_seed(SEED)["scenario"]
    rate = tx["label_fraud"].mean()
    assert cfg["fraud_rate_min"] <= rate <= cfg["fraud_rate_max"], rate


def test_vip_prior_lower_than_mass(tmp_path: Path):
    generate(SEED, tmp_path)
    users = pd.read_csv(tmp_path / "users.csv")
    vip = users.loc[users["segment"] == "vip", "risk_prior"].mean()
    mass = users.loc[users["segment"] == "mass", "risk_prior"].mean()
    assert vip < mass


def test_vip_observed_fraud_not_higher_than_mass(tmp_path: Path):
    # soft check: with our priors, mass should usually show higher fraud rate
    generate(SEED, tmp_path)
    tx = pd.read_csv(tmp_path / "transactions.csv")
    rates = tx.groupby("segment")["label_fraud"].mean()
    assert rates.get("vip", 0) <= rates.get("mass", 1) + 0.005


def test_seed_reproducibility(tmp_path: Path):
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    generate(SEED, out_a)
    generate(SEED, out_b)
    for name in ("users.csv", "devices.csv", "transactions.csv"):
        a = (out_a / name).read_text()
        b = (out_b / name).read_text()
        assert a == b, name


def test_pcg64_wiring():
    cfg = load_seed(SEED)
    rng = make_rng(cfg["master_seed"])
    assert isinstance(rng, np.random.Generator)
    x = rng.random(8)
    rng2 = make_rng(cfg["master_seed"])
    assert np.allclose(x, rng2.random(8))


def test_late_chargebacks_exist(tmp_path: Path):
    generate(SEED, tmp_path)
    tx = pd.read_csv(tmp_path / "transactions.csv")
    fraud = tx[tx["label_fraud"] == 1]
    assert fraud["chargeback_ts"].notna().any()
    assert (fraud["label_source"] == "chargeback").any()
