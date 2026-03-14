/*
  Assigns each order a revenue percentile rank.
  Useful for identifying what "a typical order" looks like
  vs what the top 5% of orders look like.
  PERCENT_RANK returns a value from 0.0 to 1.0.
*/
SELECT
    order_id,
    customer_id,
    revenue,
    order_value_tier,
    order_status,
    -- Where does this order sit in the overall revenue distribution?
    ROUND(PERCENT_RANK() OVER (ORDER BY revenue) * 100, 2)  AS revenue_percentile,
    -- Rank within the same order value tier
    RANK() OVER (
        PARTITION BY order_value_tier
        ORDER BY revenue DESC
    )                                                        AS rank_within_tier,
    -- Running count of orders at or below this revenue value
    COUNT(*) OVER (
        ORDER BY revenue
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    )                                                        AS cumulative_order_count
FROM fact_orders_clean
WHERE order_status = 'delivered'
ORDER BY revenue DESC
LIMIT 100;