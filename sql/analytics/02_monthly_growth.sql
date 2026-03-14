/*
  Monthly revenue trend with month-over-month growth rate.
  Uses LAG() window function to compare each month to the previous one.
  This is what goes into the "Revenue Trends" dashboard section.
*/
WITH monthly_revenue AS (
    SELECT
        t.year,
        t.month,
        t.month_name,
        CAST(t.year AS TEXT) || '-' ||
            CASE WHEN t.month < 10
                 THEN '0' || CAST(t.month AS TEXT)
                 ELSE CAST(t.month AS TEXT)
            END                              AS year_month,
        COUNT(DISTINCT f.order_id)           AS total_orders,
        COUNT(DISTINCT f.customer_id)        AS unique_customers,
        ROUND(SUM(f.revenue), 2)             AS monthly_revenue,
        ROUND(AVG(f.revenue), 2)             AS avg_order_value,
        ROUND(SUM(f.gross_profit), 2)        AS monthly_gross_profit
    FROM fact_orders_clean f
    JOIN dim_time t ON f.date_id = t.date_id
    WHERE f.order_status = 'delivered'
    GROUP BY t.year, t.month, t.month_name
),
with_growth AS (
    SELECT
        *,
        LAG(monthly_revenue) OVER (
            ORDER BY year, month
        )                                    AS prev_month_revenue,

        ROUND(
            (monthly_revenue -
             LAG(monthly_revenue) OVER (ORDER BY year, month))
            / NULLIF(LAG(monthly_revenue) OVER (ORDER BY year, month), 0)
            * 100,
        2)                                   AS mom_growth_pct   -- Month-over-Month

    FROM monthly_revenue
)
SELECT
    year_month,
    month_name,
    year,
    total_orders,
    unique_customers,
    monthly_revenue,
    avg_order_value,
    monthly_gross_profit,
    prev_month_revenue,
    mom_growth_pct,
    -- Running total revenue (cumulative)
    ROUND(SUM(monthly_revenue) OVER (
        ORDER BY year, month
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ), 2)                                    AS cumulative_revenue
FROM with_growth
ORDER BY year, month;