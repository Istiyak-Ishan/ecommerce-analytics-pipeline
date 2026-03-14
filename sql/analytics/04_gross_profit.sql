/*
  Gross profit analysis by product category.
  Identifies which categories drive profit vs which just drive volume.
  A category can have high revenue but low margin — this query catches that.
*/
SELECT
    p.category_english                              AS category,
    COUNT(DISTINCT f.order_id)                      AS total_orders,
    ROUND(SUM(f.revenue), 2)                        AS total_revenue,
    ROUND(SUM(f.gross_profit), 2)                   AS total_gross_profit,
    ROUND(AVG(f.revenue), 2)                        AS avg_order_value,
    ROUND(SUM(f.gross_profit) / NULLIF(SUM(f.revenue), 0) * 100, 2)
                                                    AS gross_margin_pct,
    ROUND(SUM(f.freight_value), 2)                  AS total_freight_cost,
    ROUND(AVG(f.freight_value), 2)                  AS avg_freight_per_order,
    -- Rank categories by gross profit
    RANK() OVER (ORDER BY SUM(f.gross_profit) DESC) AS profit_rank
FROM fact_orders_clean f
JOIN dim_products_clean p ON f.product_id = p.product_id
WHERE f.order_status = 'delivered'
  AND p.category_english != 'unknown'
GROUP BY p.category_english
ORDER BY total_gross_profit DESC
LIMIT 30;