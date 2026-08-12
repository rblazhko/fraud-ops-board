-- Schema docs for DuckDB feature runs (CSVs loaded by scripts/run_sql_features.py).

CREATE TABLE IF NOT EXISTS users (
    user_id     VARCHAR PRIMARY KEY,
    segment     VARCHAR,          -- vip | mass
    tenure_days INTEGER,
    country     VARCHAR,
    risk_prior  DOUBLE
);

CREATE TABLE IF NOT EXISTS devices (
    device_id   VARCHAR PRIMARY KEY,
    first_seen  TIMESTAMP,
    shared_flag INTEGER
);

CREATE TABLE IF NOT EXISTS transactions (
    tx_id         VARCHAR PRIMARY KEY,
    ts            TIMESTAMP,
    user_id       VARCHAR,
    device_id     VARCHAR,
    amount        DOUBLE,
    currency      VARCHAR,
    country       VARCHAR,
    channel       VARCHAR,
    psp_id        VARCHAR,
    segment       VARCHAR,
    auth_result   VARCHAR,
    label_fraud   INTEGER,
    label_source  VARCHAR,
    chargeback_ts TIMESTAMP
);
