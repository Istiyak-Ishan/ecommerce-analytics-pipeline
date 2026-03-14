/*
  3-month rolling average revenue.
  Smooths out seasonal spikes so you can see the real trend.
  The rolling average is the line chart executives care about most.
*/
WITH monthly AS (
    SELECT
        t.year,
        t.month,
        CAST(t.year AS TEXT) || '-' ||
            CASE WHEN t.month < 10
                 THEN '0' || CAST(t.month AS TEXT)
                 ELSE CAST(t.month AS TEXT)
            END                                     AS year_month,
        ROUND(SUM(f.revenue), 2)                    AS monthly_revenue
    FROM fact_orders_clean f
    JOIN dim_time t ON f.date_id = t.date_id
    WHERE f.order_status = 'delivered'
    GROUP BY t.year, t.month
)
SELECT
    year_month,
    monthly_revenue,
    -- 3-month rolling average (current + 2 prior months)
    ROUND(AVG(monthly_revenue) OVER (
        ORDER BY year, month
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ), 2)                                           AS rolling_3m_avg,
    -- 3-month rolling total
    ROUND(SUM(monthly_revenue) OVER (
        ORDER BY year, month
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ), 2)                                           AS rolling_3m_total,
    -- Month rank overall
    RANK() OVER (ORDER BY monthly_revenue DESC)     AS revenue_rank
FROM monthly
ORDER BY year, month;