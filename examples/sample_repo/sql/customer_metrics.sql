-- Customer lifetime metrics, rebuilt nightly.
CREATE OR REPLACE TABLE shop.gold.customer_metrics AS
SELECT
    o.customer_id,
    count(*)            AS orders,
    sum(o.amount)       AS lifetime_value,
    max(o.created_at)   AS last_order_at
FROM shop.silver.orders o
JOIN shop.silver.customers c
  ON c.customer_id = o.customer_id
WHERE o.status <> 'CANCELLED'
GROUP BY o.customer_id;

-- Upsert yesterday's revenue into the reporting table.
MERGE INTO shop.gold.revenue_report t
USING shop.gold.daily_revenue s
  ON t.order_date = s.order_date AND t.store_id = s.store_id
WHEN MATCHED THEN UPDATE SET revenue = s.revenue
WHEN NOT MATCHED THEN INSERT *;
