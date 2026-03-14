/*
  Revenue breakdown by Brazilian state.
  JOIN to dim_customers to get state information.
  Useful for regional targeting decisions.
*/
SELECT
    c.state,
    COUNT(DISTINCT f.customer_id)               AS unique_customers,
    COUNT(DISTINCT f.order_id)                  AS total_orders,
    ROUND(SUM(f.revenue), 2)                    AS total_revenue,
    ROUND(AVG(f.revenue), 2)                    AS avg_order_value,
    ROUND(SUM(f.gross_profit), 2)               AS total_gross_profit,
    ROUND(AVG(f.freight_value), 2)              AS avg_freight,
    ROUND(AVG(f.review_score), 2)               AS avg_review_score,
    -- Each state's share of total revenue
    ROUND(
        SUM(f.revenue) * 100.0 /
        SUM(SUM(f.revenue)) OVER (),
    2)                                          AS revenue_share_pct,
    RANK() OVER (ORDER BY SUM(f.revenue) DESC)  AS revenue_rank
FROM fact_orders_clean f
JOIN dim_customers_clean c ON f.customer_id = c.customer_id
WHERE f.order_status = 'delivered'
  AND c.state != 'unknown'
GROUP BY c.state
ORDER BY total_revenue DESC;