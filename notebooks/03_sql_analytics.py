"""
Phase 4 — SQL Analytics Runner
================================
Runs all analytics queries against the cleaned database
and saves results as CSVs for dashboard and reporting.

Run from project root:
    python notebooks/03_sql_analytics.py
"""

import sqlite3
import pandas as pd
from pathlib import Path

DB_PATH  = Path("data/ecommerce.db")
SQL_DIR  = Path("sql/analytics")
OUT_DIR  = Path("data/processed/analytics")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def run_query(sql: str, conn) -> pd.DataFrame:
    return pd.read_sql(sql, conn)

def run_file(filename: str, conn) -> pd.DataFrame:
    path = SQL_DIR / filename
    sql  = path.read_text()
    df   = pd.read_sql(sql, conn)
    out  = OUT_DIR / filename.replace(".sql", ".csv")
    df.to_csv(out, index=False)
    print(f"  ✓ {filename:<45} {len(df):>6} rows → {out.name}")
    return df

conn = sqlite3.connect(DB_PATH)

print("\n" + "="*60)
print("  PHASE 4 — SQL ANALYTICS")
print("="*60 + "\n")

files = [
    "01_total_revenue.sql",
    "02_monthly_growth.sql",
    "03_avg_order_value.sql",
    "04_gross_profit.sql",
    "05_top_products.sql",
    "06_top_categories.sql",
    "07_top_regions.sql",
    "08_repeat_rate.sql",
    "09_channel_performance.sql",
    "10_order_status_breakdown.sql",
    "11_cohort_first_purchase.sql",
    "12_rolling_revenue.sql",
    "13_revenue_ranking.sql",
    "14_customer_ranking.sql",
]

results = {}
for f in files:
    results[f] = run_file(f, conn)

conn.close()
print("\n  All queries complete. Results saved to data/processed/analytics/\n")