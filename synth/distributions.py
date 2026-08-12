"""Amount and velocity helpers for synthetic txs."""

from __future__ import annotations

import numpy as np


COUNTRIES = ("DE", "FR", "NL", "PL", "ES", "IT", "GB", "US")
CURRENCIES = ("EUR", "EUR", "EUR", "USD", "GBP")  # EUR-heavy
CHANNELS = ("web", "app", "pos")
PSPS = ("PSP_A", "PSP_B", "PSP_C", "PSP_D")


def sample_amount(rng: np.random.Generator, segment: str, n: int = 1) -> np.ndarray:
    """Lognormal-ish amounts; VIP median higher."""
    if segment == "vip":
        mu, sigma = 4.2, 0.55  # ~median ~67
    else:
        mu, sigma = 3.1, 0.7  # ~median ~22
    out = rng.lognormal(mean=mu, sigma=sigma, size=n)
    return np.round(out, 2)


def sample_country(rng: np.random.Generator, n: int = 1) -> np.ndarray:
    return rng.choice(COUNTRIES, size=n)


def burst_offsets_hours(rng: np.random.Generator, n: int, span_hours: float = 6.0) -> np.ndarray:
    """Cluster n events inside a short window (fraud ring velocity)."""
    if n <= 0:
        return np.array([], dtype=float)
    # half-normal-ish clustering toward the burst start
    raw = np.abs(rng.normal(0.0, span_hours / 3.0, size=n))
    return np.clip(raw, 0.0, span_hours)


def poisson_tx_count(rng: np.random.Generator, lam: float) -> int:
    return int(rng.poisson(lam))
