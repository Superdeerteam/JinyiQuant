-- Export from the authorized official online table; do not derive the ID from stock-code digits.
-- Restrict dates to the approved development period and validate one-to-one keys locally.
SELECT DISTINCT instrument_id, instrument
FROM cpt_jyc_2026_stock_bar1m
WHERE date >= '2020-01-01' AND date < '2025-01-01';
