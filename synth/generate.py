"""CLI: build synthetic users / devices / transactions."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from synth.distributions import (
    CHANNELS,
    CURRENCIES,
    PSPS,
    burst_offsets_hours,
    poisson_tx_count,
    sample_amount,
    sample_country,
)
from synth.labels import assign_fraud_flags

# Fixed anchor so seed reproducibility does not drift with wall clock.
ANCHOR_END = datetime(2025, 6, 30, 23, 59, 59, tzinfo=timezone.utc)


def load_seed(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def make_rng(master_seed: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(int(master_seed)))


def build_users(rng: np.random.Generator, cfg: dict) -> pd.DataFrame:
    n = int(cfg["n_users"])
    vip_share = float(cfg["vip_share"])
    segments = np.where(rng.random(n) < vip_share, "vip", "mass")
    prior_vip = float(cfg["fraud_prior_vip"])
    prior_mass = float(cfg["fraud_prior_mass"])
    risk = np.where(segments == "vip", prior_vip, prior_mass)
    # small jitter so priors are not a single constant
    risk = risk * rng.uniform(0.85, 1.15, size=n)

    rows = {
        "user_id": [f"U{i:06d}" for i in range(1, n + 1)],
        "segment": segments,
        "tenure_days": rng.integers(7, 2400, size=n),
        "country": sample_country(rng, n),
        "risk_prior": np.round(risk, 5),
    }
    return pd.DataFrame(rows)


def build_devices(rng: np.random.Generator, cfg: dict, window_start: datetime) -> pd.DataFrame:
    n = int(cfg["n_devices"])
    n_rings = int(cfg["n_rings"])
    ring_size = int(cfg["ring_size_devices"])

    shared = np.zeros(n, dtype=int)
    ring_slots = min(n, n_rings * ring_size)
    shared[:ring_slots] = 1
    # shuffle so ring devices are not a contiguous block in ids
    order = rng.permutation(n)
    shared = shared[order]

    offsets = rng.integers(0, int(cfg["window_days"]) * 24 * 3600, size=n)
    first_seen = [window_start + timedelta(seconds=int(s)) for s in offsets]

    return pd.DataFrame(
        {
            "device_id": [f"D{i:06d}" for i in range(1, n + 1)],
            "first_seen": first_seen,
            "shared_flag": shared,
        }
    )


def _ring_device_ids(devices: pd.DataFrame, n_rings: int, ring_size: int) -> list[list[str]]:
    shared = devices.loc[devices["shared_flag"] == 1, "device_id"].tolist()
    rings = []
    for r in range(n_rings):
        chunk = shared[r * ring_size : (r + 1) * ring_size]
        if chunk:
            rings.append(chunk)
    return rings


def build_transactions(
    rng: np.random.Generator,
    cfg: dict,
    users: pd.DataFrame,
    devices: pd.DataFrame,
    window_start: datetime,
    window_end: datetime,
) -> pd.DataFrame:
    n_tx = int(cfg["n_transactions"])
    window_seconds = int((window_end - window_start).total_seconds())

    user_ids = users["user_id"].to_numpy()
    seg_map = users.set_index("user_id")["segment"].to_dict()
    country_map = users.set_index("user_id")["country"].to_dict()
    device_ids = devices["device_id"].to_numpy()
    shared_devices = devices.loc[devices["shared_flag"] == 1, "device_id"].to_numpy()
    if len(shared_devices) == 0:
        shared_devices = device_ids[:1]

    rings = _ring_device_ids(devices, int(cfg["n_rings"]), int(cfg["ring_size_devices"]))

    records: list[dict] = []
    n_ring_tx = int(n_tx * float(cfg.get("ring_tx_share", 0.08)))
    n_normal = n_tx - n_ring_tx

    for _ in range(n_normal):
        uid = str(rng.choice(user_ids))
        seg = seg_map[uid]
        # VIP slightly prefers dedicated devices
        if seg == "vip" and rng.random() < 0.7:
            did = str(rng.choice(device_ids))
        else:
            pool = device_ids if rng.random() > 0.08 else shared_devices
            did = str(rng.choice(pool))
        ts = window_start + timedelta(seconds=int(rng.integers(0, window_seconds)))
        amount = float(sample_amount(rng, seg, 1)[0])
        records.append(
            {
                "ts": ts,
                "user_id": uid,
                "device_id": did,
                "amount": amount,
                "currency": str(rng.choice(CURRENCIES)),
                "country": country_map[uid],
                "channel": str(rng.choice(CHANNELS)),
                "psp_id": str(rng.choice(PSPS)),
                "segment": seg,
                "auth_result": "approved" if rng.random() > 0.04 else "declined",
                "ring_id": -1,
            }
        )

    # Ring bursts: few shared devices, short windows, mixed users
    remaining = n_ring_tx
    ring_id = 0
    while remaining > 0 and rings:
        ring = rings[ring_id % len(rings)]
        burst_n = min(remaining, max(3, poisson_tx_count(rng, 8)))
        burst_anchor = window_start + timedelta(
            seconds=int(rng.integers(0, max(1, window_seconds - 6 * 3600)))
        )
        offsets = burst_offsets_hours(rng, burst_n, span_hours=5.0)
        for j in range(burst_n):
            uid = str(rng.choice(user_ids))
            seg = seg_map[uid]
            did = str(rng.choice(ring))
            ts = burst_anchor + timedelta(hours=float(offsets[j]))
            if ts > window_end:
                ts = window_end - timedelta(minutes=int(rng.integers(1, 40)))
            amount = float(sample_amount(rng, seg, 1)[0])
            # rings: a bit larger amounts on average
            amount = round(amount * float(rng.uniform(1.1, 2.2)), 2)
            records.append(
                {
                    "ts": ts,
                    "user_id": uid,
                    "device_id": did,
                    "amount": amount,
                    "currency": str(rng.choice(CURRENCIES)),
                    "country": country_map[uid],
                    "channel": str(rng.choice(("web", "app"))),  # rings rarely POS
                    "psp_id": str(rng.choice(PSPS)),
                    "segment": seg,
                    "auth_result": "approved" if rng.random() > 0.02 else "declined",
                    "ring_id": ring_id,
                }
            )
        remaining -= burst_n
        ring_id += 1

    tx = pd.DataFrame(records)
    tx = tx.sort_values("ts").reset_index(drop=True)
    tx.insert(0, "tx_id", [f"T{i:08d}" for i in range(1, len(tx) + 1)])

    tx = assign_fraud_flags(
        rng,
        tx,
        users,
        geo_mismatch_fraud_share=float(cfg["geo_mismatch_fraud_share"]),
        late_chargeback_share=float(cfg["late_chargeback_share"]),
        chargeback_lag_days_min=int(cfg["chargeback_lag_days_min"]),
        chargeback_lag_days_max=int(cfg["chargeback_lag_days_max"]),
    )
    return tx


def generate(seed_path: Path, out_dir: Path) -> None:
    cfg_all = load_seed(seed_path)
    master_seed = int(cfg_all["master_seed"])
    cfg = cfg_all["scenario"]
    rng = make_rng(master_seed)

    window_days = int(cfg["window_days"])
    window_end = ANCHOR_END
    window_start = window_end - timedelta(days=window_days)

    users = build_users(rng, cfg)
    devices = build_devices(rng, cfg, window_start)
    txs = build_transactions(rng, cfg, users, devices, window_start, window_end)

    out_dir.mkdir(parents=True, exist_ok=True)
    users.to_csv(out_dir / "users.csv", index=False)
    devices.to_csv(out_dir / "devices.csv", index=False)
    txs.to_csv(out_dir / "transactions.csv", index=False)

    rate = txs["label_fraud"].mean()
    print(f"wrote {len(users)} users, {len(devices)} devices, {len(txs)} txs -> {out_dir}")
    print(f"fraud rate: {rate:.4%} (target band {cfg['fraud_rate_min']}-{cfg['fraud_rate_max']})")


def main() -> None:
    p = argparse.ArgumentParser(description="Generate synthetic fraud-ops tables")
    p.add_argument("--seed", type=Path, default=Path("seed.yaml"))
    p.add_argument("--out", type=Path, default=Path("data"))
    args = p.parse_args()
    generate(args.seed, args.out)


if __name__ == "__main__":
    main()
