-- Amount z vs prior user history; cold-start (<3 txs) → NULL z + flag.

CREATE OR REPLACE VIEW feat_amount AS
WITH hist AS (
    SELECT
        t.tx_id,
        t.ts,
        t.user_id,
        t.amount,
        t.segment,
        (
            SELECT AVG(x.amount)
            FROM transactions x
            WHERE x.user_id = t.user_id
              AND x.ts < t.ts
        ) AS amt_mean_hist,
        (
            SELECT STDDEV_SAMP(x.amount)
            FROM transactions x
            WHERE x.user_id = t.user_id
              AND x.ts < t.ts
        ) AS amt_std_hist,
        (
            SELECT COUNT(*)
            FROM transactions x
            WHERE x.user_id = t.user_id
              AND x.ts < t.ts
        ) AS n_hist
    FROM transactions t
)
SELECT
    h.tx_id,
    h.ts,
    h.user_id,
    h.segment,
    h.amount,
    h.n_hist,
    h.amt_mean_hist,
    h.amt_std_hist,
    CASE
        WHEN h.n_hist < 3 OR h.amt_std_hist IS NULL OR h.amt_std_hist = 0
            THEN NULL
        ELSE (h.amount - h.amt_mean_hist) / h.amt_std_hist
    END AS amount_z_user,
    CASE WHEN h.n_hist < 3 THEN 1 ELSE 0 END AS cold_start_flag,
    CASE
        WHEN t.country <> u.country THEN 1 ELSE 0
    END AS geo_mismatch
FROM hist h
JOIN transactions t USING (tx_id)
JOIN users u ON u.user_id = h.user_id;
