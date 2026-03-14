/*
  Ranks all customers by lifetime value.
  Identifies your most valuable customers — the ones who
  should receive VIP treatment and loyalty offers.
  Demonstrates DENSE_RANK() which handles ties correctly.
*/
WITH customer_ltv AS (
    SELECT
        f.customer_id,
        c.state,
        c.rfm_segment,
        c.customer_segment,
        c.churn_flag,
        COUNT(DISTINCT f.order_id)          AS total_orders,
        ROUND(SUM(f.revenue), 2)            AS lifetime_revenue,
        ROUND(AVG(f.revenue), 2)            AS avg_order_value,
        ROUND(SUM(f.gross_profit), 2)       AS lifetime_gross_profit,
        ROUND(AVG(f.review_score), 2)       AS avg_review_score,
        ROUND(AVG(f.discount_pct), 2)       AS avg_discount_pct,
        MIN(f.order_purchase_ts)            AS first_order,
        MAX(f.order_purchase_ts)            AS last_order
    FROM fact_orders_clean f
    JOIN dim_customers_clean c ON f.customer_id = c.customer_id
    WHERE f.order_status = 'delivered'
    GROUP BY f.customer_id, c.state, c.rfm_segment,
             c.customer_segment, c.churn_flag
)
SELECT
    customer_id,
    state,
    rfm_segment,
    customer_segment,
    churn_flag,
    total_orders,
    lifetime_revenue,
    avg_order_value,
    lifetime_gross_profit,
    avg_review_score,
    avg_discount_pct,
    first_order,
    last_order,
    -- DENSE_RANK: no gaps in ranking when there are ties
    DENSE_RANK() OVER (ORDER BY lifetime_revenue DESC)  AS ltv_rank,
    -- What percentile is this customer in?
    ROUND(PERCENT_RANK() OVER (
        ORDER BY lifetime_revenue
    ) * 100, 2)                                         AS ltv_percentile,
    -- Revenue share of this customer vs all customers
    ROUND(lifetime_revenue * 100.0 /
        SUM(lifetime_revenue) OVER (), 4)               AS revenue_share_pct
FROM customer_ltv
ORDER BY ltv_rank
LIMIT 50;