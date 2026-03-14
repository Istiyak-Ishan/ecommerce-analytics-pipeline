/*
  Top 20 individual products by revenue.
  Uses RANK() so tied products share the same rank.
  Also shows return signal via avg review score.
*/
WITH product_stats AS (
    SELECT
        f.product_id,
        p.category_english,
        COUNT(DISTINCT f.order_id)          AS total_orders,
        ROUND(SUM(f.revenue), 2)            AS total_revenue,
        ROUND(AVG(f.revenue), 2)            AS avg_selling_price,
        ROUND(SUM(f.gross_profit), 2)       AS total_gross_profit,
        ROUND(AVG(f.review_score), 2)       AS avg_review_score,
        SUM(f.has_discount)                 AS discounted_orders
    FROM fact_orders_clean f
    JOIN dim_products_clean p ON f.product_id = p.product_id
    WHERE f.order_status = 'delivered'
    GROUP BY f.product_id, p.category_english
)
SELECT
    product_id,
    category_english,
    total_orders,
    total_revenue,
    avg_selling_price,
    total_gross_profit,
    avg_review_score,
    discounted_orders,
    RANK()       OVER (ORDER BY total_revenue    DESC) AS revenue_rank,
    RANK()       OVER (ORDER BY total_orders     DESC) AS volume_rank,
    RANK()       OVER (ORDER BY total_gross_profit DESC) AS profit_rank
FROM product_stats
ORDER BY revenue_rank
LIMIT 20;