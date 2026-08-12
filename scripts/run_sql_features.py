#!/usr/bin/env python3
"""Load synth CSVs into DuckDB, run feature SQL, write parquet."""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


FEATURE_VIEWS = (
    ("feat_velocity", "features_velocity.parquet"),
    ("feat_device", "features_device.parquet"),
    ("feat_amount", "features_amount.parquet"),
)


def run(data_dir: Path, sql_dir: Path, out_dir: Path) -> None:
    users = data_dir / "users.csv"
    devices = data_dir / "devices.csv"
    txs = data_dir / "transactions.csv"
    for p in (users, devices, txs):
        if not p.exists():
            raise SystemExit(f"missing {p}; run `make data` first")

    out_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(database=":memory:")

    con.execute(
        f"""
        CREATE TABLE users AS
        SELECT * FROM read_csv_auto('{users.as_posix()}', header=true);
        """
    )
    con.execute(
        f"""
        CREATE TABLE devices AS
        SELECT
            device_id,
            CAST(first_seen AS TIMESTAMP) AS first_seen,
            shared_flag
        FROM read_csv_auto('{devices.as_posix()}', header=true);
        """
    )
    con.execute(
        f"""
        CREATE TABLE transactions AS
        SELECT
            tx_id,
            CAST(ts AS TIMESTAMP) AS ts,
            user_id,
            device_id,
            amount,
            currency,
            country,
            channel,
            psp_id,
            segment,
            auth_result,
            label_fraud,
            label_source,
            TRY_CAST(chargeback_ts AS TIMESTAMP) AS chargeback_ts
        FROM read_csv_auto('{txs.as_posix()}', header=true);
        """
    )

    # schema file is documentation; views live in 01–03
    for name in sorted(sql_dir.glob("0[1-9]_*.sql")):
        sql = name.read_text()
        con.execute(sql)
        print(f"applied {name.name}")

    for view, fname in FEATURE_VIEWS:
        out = out_dir / fname
        con.execute(f"COPY (SELECT * FROM {view}) TO '{out.as_posix()}' (FORMAT PARQUET)")
        n = con.execute(f"SELECT COUNT(*) FROM {view}").fetchone()[0]
        print(f"wrote {out} ({n} rows)")

    # joined wide table for later modeling
    wide = out_dir / "features_wide.parquet"
    con.execute(
        f"""
        COPY (
            SELECT
                t.tx_id,
                t.ts,
                t.user_id,
                t.device_id,
                t.amount,
                t.segment,
                t.label_fraud,
                t.label_source,
                v.tx_cnt_1h,
                v.tx_cnt_24h,
                v.amt_sum_24h,
                v.burst_flag_1h,
                d.shared_flag,
                d.device_degree_to_date,
                d.device_tx_cnt_24h,
                d.device_users_24h,
                a.amount_z_user,
                a.cold_start_flag,
                a.geo_mismatch
            FROM transactions t
            LEFT JOIN feat_velocity v USING (tx_id)
            LEFT JOIN feat_device d USING (tx_id)
            LEFT JOIN feat_amount a USING (tx_id)
        ) TO '{wide.as_posix()}' (FORMAT PARQUET)
        """
    )
    print(f"wrote {wide}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--sql", type=Path, default=Path("sql"))
    p.add_argument("--out", type=Path, default=Path("data"))
    args = p.parse_args()
    run(args.data, args.sql, args.out)


if __name__ == "__main__":
    main()
