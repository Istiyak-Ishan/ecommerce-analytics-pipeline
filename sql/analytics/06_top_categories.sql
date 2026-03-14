/*
  Full category performance with quartile classification.
  NTILE(4) splits categories into 4 equal performance bands.
  This powers the "Product Performance" dashboard section.
*/
WITH category_metrics AS (
    SELECT
        p.category_english                          AS category,
        COUNT(DISTINCT f.order_id)                  AS total_orders,
        COUNT(DISTINCT f.customer_id)               AS unique_buyers,
        ROUND(SUM(f.revenue), 2)                    AS total_revenue,
        ROUND(AVG(f.revenue), 2)                    AS avg_order_value,
        ROUND(SUM(f.gross_profit), 2)               AS total_gross_profit,
        ROUND(AVG(f.review_score), 2)               AS avg_review,
        ROUND(AVG(f.discount_pct), 2)               AS avg_discount_pct,
        SUM(f.is_late_delivery)                     AS late_deliveries,
        ROUND(
            SUM(f.is_late_delivery) * 100.0 /
            NULLIF(COUNT(*), 0),
        2)                                          AS late_delivery_pct
    FROM fact_orders_clean f
    JOIN dim_products_clean p ON f.product_id = p.product_id
    WHERE f.order_status = 'delivered'
      AND p.category_english != 'unknown'
    GROUP BY p.category_english
)
SELECT
    category,
    total_orders,
    unique_buyers,
    total_revenue,
    avg_order_value,
    total_gross_profit,
    avg_review,
    avg_discount_pct,
    late_delivery_pct,
    -- Classify each category into performance quartile
    NTILE(4) OVER (ORDER BY total_revenue DESC)     AS revenue_quartile,
    -- 1 = top 25%, 4 = bottom 25%
    CASE NTILE(4) OVER (ORDER BY total_revenue DESC)
        WHEN 1 THEN 'top performer'
        WHEN 2 THEN 'solid'
        WHEN 3 THEN 'average'
        WHEN 4 THEN 'underperformer'
    END                                             AS performance_tier
FROM category_metrics
ORDER BY total_revenue DESC;