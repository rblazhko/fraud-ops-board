-- Rolling 1h / 24h velocity per user (prior txs only; self excluded via ts <).

CREATE OR REPLACE VIEW feat_velocity AS
WITH base AS (
    SELECT
        t.tx_id,
        t.ts,
        t.user_id,
        t.device_id,
        t.amount,
        t.label_fraud
    FROM transactions t
),
counts AS (
    SELECT
        b.tx_id,
        b.ts,
        b.user_id,
        (
            SELECT COUNT(*)
            FROM transactions x
            WHERE x.user_id = b.user_id
              AND x.ts < b.ts
              AND x.ts >= b.ts - INTERVAL 1 HOUR
        ) AS tx_cnt_1h,
        (
            SELECT COUNT(*)
            FROM transactions x
            WHERE x.user_id = b.user_id
              AND x.ts < b.ts
              AND x.ts >= b.ts - INTERVAL 24 HOUR
        ) AS tx_cnt_24h,
        (
            SELECT COALESCE(SUM(x.amount), 0)
            FROM transactions x
            WHERE x.user_id = b.user_id
              AND x.ts < b.ts
              AND x.ts >= b.ts - INTERVAL 24 HOUR
        ) AS amt_sum_24h
    FROM base b
)
SELECT
    c.tx_id,
    c.ts,
    c.user_id,
    c.tx_cnt_1h,
    c.tx_cnt_24h,
    c.amt_sum_24h,
    CASE
        WHEN c.tx_cnt_24h > 0 AND c.tx_cnt_1h >= 3 AND (c.tx_cnt_1h * 1.0 / c.tx_cnt_24h) >= 0.5
            THEN 1
        ELSE 0
    END AS burst_flag_1h
FROM counts c;
