/*
  Marketing channel performance including ROI calculation.
  LTV:CAC ratio > 3 is healthy. Below 1 means you lose money acquiring customers.
*/
WITH channel_stats AS (
    SELECT
        ch.channel_id,
        ch.channel_name,
        ch.channel_type,
        ch.cac,
        COUNT(DISTINCT f.customer_id)               AS customers_acquired,
        COUNT(DISTINCT f.order_id)                  AS total_orders,
        ROUND(SUM(f.revenue), 2)                    AS total_revenue,
        ROUND(AVG(f.revenue), 2)                    AS avg_order_value,
        ROUND(SUM(f.gross_profit), 2)               AS total_gross_profit,
        ROUND(AVG(f.discount_pct), 2)               AS avg_discount_pct
    FROM fact_orders_clean f
    JOIN dim_channels ch ON f.channel_id = ch.channel_id
    WHERE f.order_status = 'delivered'
    GROUP BY ch.channel_id, ch.channel_name, ch.channel_type, ch.cac
)
SELECT
    channel_name,
    channel_type,
    customers_acquired,
    total_orders,
    total_revenue,
    avg_order_value,
    total_gross_profit,
    avg_discount_pct,
    cac                                             AS cost_per_acquisition,
    -- Total acquisition spend = CAC × customers acquired
    ROUND(cac * customers_acquired, 2)              AS total_acquisition_cost,
    -- Customer LTV (simplified) = avg revenue × avg orders per customer
    ROUND(total_revenue / NULLIF(customers_acquired, 0), 2) AS avg_customer_ltv,
    -- LTV:CAC ratio — the single most important marketing metric
    ROUND(
        (total_revenue / NULLIF(customers_acquired, 0))
        / NULLIF(cac, 0),
    2)                                              AS ltv_cac_ratio,
    -- Revenue per £1 spent on acquisition
    ROUND(
        total_revenue / NULLIF(cac * customers_acquired, 0),
    2)                                              AS revenue_per_cac_dollar
FROM channel_stats
ORDER BY ltv_cac_ratio DESC;