/*
  Average Order Value broken down by customer segment and channel.
  High AOV segments are the ones to invest in for CLV.
*/
SELECT
    c.customer_segment,
    ch.channel_name,
    COUNT(DISTINCT f.order_id)           AS total_orders,
    ROUND(AVG(f.revenue), 2)             AS avg_order_value,
    ROUND(MIN(f.revenue), 2)             AS min_order_value,
    ROUND(MAX(f.revenue), 2)             AS max_order_value,
    ROUND(AVG(f.discount_amount), 2)     AS avg_discount,
    ROUND(AVG(f.discount_pct), 2)        AS avg_discount_pct
FROM fact_orders_clean f
JOIN dim_customers_clean c  ON f.customer_id = c.customer_id
JOIN dim_channels          ch ON f.channel_id  = ch.channel_id
WHERE f.order_status = 'delivered'
GROUP BY c.customer_segment, ch.channel_name
ORDER BY avg_order_value DESC;