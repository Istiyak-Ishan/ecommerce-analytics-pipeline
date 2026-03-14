/*
  Order status breakdown — identifies how much revenue is
  at risk from cancelled or unavailable orders.
*/
SELECT
    order_status,
    COUNT(*)                                        AS order_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pct_of_orders,
    ROUND(SUM(revenue), 2)                          AS revenue_at_status,
    ROUND(AVG(revenue), 2)                          AS avg_order_value,
    ROUND(SUM(discount_amount), 2)                  AS total_discounts,
    ROUND(AVG(review_score), 2)                     AS avg_review_score
FROM fact_orders_clean
GROUP BY order_status
ORDER BY order_count DESC;