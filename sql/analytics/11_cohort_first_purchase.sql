/*
  First purchase cohorts — rewritten without correlated subquery.
  Pre-aggregates customer revenue first, then joins. Much faster.
*/
WITH customer_revenue AS (
    SELECT
        customer_id,
        SUM(revenue) AS total_customer_revenue
    FROM fact_orders_clean
    WHERE order_status = 'delivered'
    GROUP BY customer_id
),
first_orders AS (
    SELECT
        f.customer_id,
        MIN(f.order_purchase_ts) AS first_order_ts
    FROM fact_orders_clean f
    WHERE f.order_status = 'delivered'
    GROUP BY f.customer_id
)
SELECT
    CAST(strftime('%Y', fo.first_order_ts) AS TEXT) || '-' ||
    CASE WHEN CAST(strftime('%m', fo.first_order_ts) AS INTEGER) < 10
         THEN '0' || CAST(strftime('%m', fo.first_order_ts) AS TEXT)
         ELSE      CAST(strftime('%m', fo.first_order_ts) AS TEXT)
    END                                     AS cohort_month,
    COUNT(DISTINCT fo.customer_id)          AS cohort_size,
    ROUND(AVG(cr.total_customer_revenue), 2) AS avg_customer_revenue
FROM first_orders fo
JOIN customer_revenue cr ON fo.customer_id = cr.customer_id
GROUP BY cohort_month
ORDER BY cohort_month;