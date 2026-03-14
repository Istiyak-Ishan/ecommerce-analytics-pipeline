/*
  Top-level revenue KPIs.
  This is the first number an exec looks at every morning.
*/
SELECT
    COUNT(DISTINCT order_id)                        AS total_orders,
    COUNT(DISTINCT customer_id)                     AS total_customers,
    ROUND(SUM(revenue), 2)                          AS total_revenue,
    ROUND(SUM(gross_profit), 2)                     AS total_gross_profit,
    ROUND(AVG(revenue), 2)                          AS avg_order_value,
    ROUND(SUM(gross_profit) / SUM(revenue) * 100, 2) AS gross_margin_pct,
    ROUND(SUM(discount_amount), 2)                  AS total_discounts_given,
    ROUND(SUM(discount_amount) / SUM(unit_price) * 100, 2) AS discount_rate_pct
FROM fact_orders_clean
WHERE order_status = 'delivered';