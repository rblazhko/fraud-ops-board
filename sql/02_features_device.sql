-- Device reuse; degree counts distinct users on the device up to this tx (inclusive).

CREATE OR REPLACE VIEW feat_device AS
WITH tx AS (
    SELECT
        t.tx_id,
        t.ts,
        t.user_id,
        t.device_id,
        d.shared_flag
    FROM transactions t
    LEFT JOIN devices d USING (device_id)
)
SELECT
    a.tx_id,
    a.ts,
    a.user_id,
    a.device_id,
    a.shared_flag,
    (
        SELECT COUNT(DISTINCT b.user_id)
        FROM transactions b
        WHERE b.device_id = a.device_id
          AND b.ts <= a.ts
    ) AS device_degree_to_date,
    (
        SELECT COUNT(*)
        FROM transactions b
        WHERE b.device_id = a.device_id
          AND b.ts < a.ts
          AND b.ts >= a.ts - INTERVAL 24 HOUR
    ) AS device_tx_cnt_24h,
    (
        SELECT COUNT(DISTINCT b.user_id)
        FROM transactions b
        WHERE b.device_id = a.device_id
          AND b.ts < a.ts
          AND b.ts >= a.ts - INTERVAL 24 HOUR
    ) AS device_users_24h
FROM tx a;
