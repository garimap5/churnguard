-- Feature layer. Runs on SQLite/Postgres-style SQL (window functions + CTEs).
-- Only *pre-outcome* columns are selected: no churned / treated / true_uplift here (leakage guard,
-- enforced by tests/test_features.py).

WITH ranked AS (            -- trailing 3-month moving averages via window functions
    SELECT sub_id, month, data_gb, voice_min, recharge_amt, drop_rate,
           AVG(data_gb)      OVER w3 AS data_ma3,
           AVG(voice_min)    OVER w3 AS voice_ma3,
           AVG(recharge_amt) OVER w3 AS rch_ma3,
           AVG(drop_rate)    OVER w3 AS drop_ma3
    FROM usage_monthly
    WINDOW w3 AS (PARTITION BY sub_id ORDER BY month ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)
),
latest AS (SELECT * FROM ranked WHERE month = (SELECT MAX(month) FROM usage_monthly)),
early  AS (                 -- first half of the observation window, the baseline for trends
    SELECT sub_id, AVG(data_gb) AS data_early, AVG(voice_min) AS voice_early, AVG(recharge_amt) AS rch_early
    FROM usage_monthly WHERE month <= 3 GROUP BY sub_id
),
peak   AS (SELECT sub_id, MAX(drop_rate) AS drop_rate_max FROM usage_monthly GROUP BY sub_id)

SELECT s.sub_id,
       s.plan_type, s.tenure_months, s.arpu, s.device_tier, s.is_5g, s.circle,
       s.offer_redemptions_6m, s.marketing_consent, s.dnd,
       l.data_ma3                                   AS data_gb_3m,
       l.data_ma3  / NULLIF(e.data_early, 0)        AS data_trend,      -- <1 => usage falling
       l.voice_ma3                                  AS voice_min_3m,
       l.voice_ma3 / NULLIF(e.voice_early, 0)       AS voice_trend,
       l.rch_ma3                                    AS recharge_3m,
       l.rch_ma3   / NULLIF(e.rch_early, 0)         AS recharge_trend,
       l.drop_ma3                                   AS drop_rate_3m,
       p.drop_rate_max,
       COALESCE(t.tickets_90d, 0)                   AS tickets_90d,
       COALESCE(t.unresolved_tickets, 0)            AS unresolved_tickets
FROM subscribers s
JOIN latest l ON l.sub_id = s.sub_id
JOIN early  e ON e.sub_id = s.sub_id
JOIN peak   p ON p.sub_id = s.sub_id
LEFT JOIN tickets t ON t.sub_id = s.sub_id;
