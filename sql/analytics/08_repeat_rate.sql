/*
  Customer repeat purchase analysis.
  Repeat rate = % of customers who placed more than one order.
  This is one of the most important e-commerce health metrics.
  A repeat rate below 20% signals poor retention.
*/
WITH customer_order_counts AS (
    SELECT
        customer_id,
        COUNT(DISTINCT order_id)    AS order_count,
        MIN(order_purchase_ts)      AS first_order_date,
        MAX(order_purchase_ts)      AS last_order_date,
        ROUND(SUM(revenue), 2)      AS lifetime_revenue
    FROM fact_orders_clean
    WHERE order_status = 'delivered'
    GROUP BY customer_id
),
classified AS (
    SELECT
        *,
        CASE
            WHEN order_count = 1  THEN 'one-time'
            WHEN order_count = 2  THEN 'two-time'
            WHEN order_count <= 5 THEN 'regular'
            ELSE                       'vip'
        END AS buyer_type
    FROM customer_order_counts
)
SELECT
    buyer_type,
    COUNT(*)                                        AS customer_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pct_of_customers,
    ROUND(AVG(lifetime_revenue), 2)                 AS avg_lifetime_revenue,
    ROUND(SUM(lifetime_revenue), 2)                 AS total_revenue_contributed,
    ROUND(
        SUM(lifetime_revenue) * 100.0 /
        SUM(SUM(lifetime_revenue)) OVER (),
    2)                                              AS pct_of_total_revenue
FROM classified
GROUP BY buyer_type
ORDER BY avg_lifetime_revenue DESC;