"""
Phase 2 — Data Loading Pipeline
================================
Reads raw Olist CSVs, builds the star schema,
and writes everything into a local SQLite database.

Run from the project root with venv active:
    python notebooks/01_load_data.py
"""

import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

DATA_RAW  = Path("data/raw")
DATA_PROC = Path("data/processed")
DB_PATH   = Path("data/ecommerce.db")

DATA_PROC.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — LOAD RAW CSVs
# ─────────────────────────────────────────────────────────────────────────────

def load_raw_data() -> dict:
    """Load all 9 Olist CSV files into a dictionary of DataFrames."""

    files = {
        "orders":       "olist_orders_dataset.csv",
        "order_items":  "olist_order_items_dataset.csv",
        "customers":    "olist_customers_dataset.csv",
        "products":     "olist_products_dataset.csv",
        "sellers":      "olist_sellers_dataset.csv",
        "payments":     "olist_order_payments_dataset.csv",
        "reviews":      "olist_order_reviews_dataset.csv",
        "geo":          "olist_geolocation_dataset.csv",
        "category":     "product_category_name_translation.csv",
    }

    raw = {}
    print("=" * 55)
    print("STEP 1 — Loading raw CSV files")
    print("=" * 55)

    for key, filename in files.items():
        path = DATA_RAW / filename
        if not path.exists():
            raise FileNotFoundError(
                f"\n[ERROR] File not found: {path}"
                f"\nMake sure you ran: kaggle datasets download "
                f"-d olistbr/brazilian-ecommerce -p data/raw --unzip"
            )
        raw[key] = pd.read_csv(path)
        print(f"  ✓ {key:<15} {raw[key].shape[0]:>7,} rows  "
              f"{raw[key].shape[1]:>3} cols")

    return raw


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — BUILD dim_time
# ─────────────────────────────────────────────────────────────────────────────

def build_dim_time(start: str = "2016-01-01",
                   end:   str = "2019-12-31") -> pd.DataFrame:
    """
    Generate a complete date dimension table.
    date_id format: YYYYMMDD integer (e.g. 20180315)
    This lets us do fast integer joins instead of string date comparisons.
    """

    dates = pd.date_range(start=start, end=end, freq="D")
    df = pd.DataFrame({"full_date": dates})

    df["date_id"]     = df["full_date"].dt.strftime("%Y%m%d").astype(int)
    df["year"]        = df["full_date"].dt.year
    df["quarter"]     = df["full_date"].dt.quarter
    df["month"]       = df["full_date"].dt.month
    df["month_name"]  = df["full_date"].dt.strftime("%B")
    df["week"]        = df["full_date"].dt.isocalendar().week.astype(int)
    df["day_of_week"] = df["full_date"].dt.strftime("%A")
    df["is_weekend"]  = (df["full_date"].dt.dayofweek >= 5).astype(int)
    df["is_holiday"]  = 0   # placeholder — can enrich with Brazilian holidays

    df["full_date"] = df["full_date"].dt.strftime("%Y-%m-%d")

    return df[[
        "date_id", "full_date", "year", "quarter",
        "month", "month_name", "week", "day_of_week",
        "is_weekend", "is_holiday"
    ]]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — BUILD dim_customers
# ─────────────────────────────────────────────────────────────────────────────

def build_dim_customers(raw: dict) -> pd.DataFrame:
    """
    Build the customer dimension table.
    We join the first_purchase_date from the orders table
    so the dimension carries this useful attribute.
    Customer segment starts as 'new' — Phase 3 will
    reclassify using RFM logic.
    """

    df = raw["customers"].copy()

    df = df.rename(columns={
        "customer_city":              "city",
        "customer_state":             "state",
        "customer_zip_code_prefix":   "zip_code",
    })

    # Compute each customer's first purchase date
    orders = raw["orders"].copy()
    orders["order_purchase_timestamp"] = pd.to_datetime(
        orders["order_purchase_timestamp"], errors="coerce"
    )
    first_purchase = (
        orders.groupby("customer_id")["order_purchase_timestamp"]
        .min()
        .dt.strftime("%Y-%m-%d")
        .reset_index()
        .rename(columns={"order_purchase_timestamp": "first_purchase_date"})
    )

    df = df.merge(first_purchase, on="customer_id", how="left")
    df["customer_segment"] = "new"

    return df[[
        "customer_id", "customer_unique_id",
        "city", "state", "zip_code",
        "first_purchase_date", "customer_segment"
    ]]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — BUILD dim_products
# ─────────────────────────────────────────────────────────────────────────────
def build_dim_products(raw: dict) -> pd.DataFrame:
    """
    Build the product dimension table.
    Joins the English category translation.
    Price is the median price observed in order_items.
    """

    df = raw["products"].copy()

    # Print actual column names so we can always debug easily
    print(f"\n  [DEBUG] products columns: {list(df.columns)}")

    # Rename ALL Olist product columns to our schema names
    df = df.rename(columns={
        "product_name_lenght":        "product_name_length",   # Olist typo
        "product_description_lenght": "product_desc_length",   # Olist typo
        "product_photos_qty":         "photos_qty",            # ← this was the bug
        "product_weight_g":           "weight_g",
        "product_length_cm":          "length_cm",
        "product_height_cm":          "height_cm",
        "product_width_cm":           "width_cm",
        "product_category_name":      "category_name",
    })

    # Join English category names
    cat = raw["category"].rename(columns={
        "product_category_name":         "category_name",
        "product_category_name_english": "category_english",
    })
    df = df.merge(cat, on="category_name", how="left")

    # Derive median price per product from order_items
    price_map = (
        raw["order_items"]
        .groupby("product_id")["price"]
        .median()
        .reset_index()
    )
    df = df.merge(price_map, on="product_id", how="left")

    df["is_active"] = 1

    # Only keep columns that actually exist after renaming
    keep = [
        "product_id", "category_name", "category_english",
        "product_name_length", "product_desc_length", "photos_qty",
        "weight_g", "length_cm", "height_cm", "width_cm",
        "price", "is_active"
    ]

    # Drop any that are still missing (defensive)
    keep = [c for c in keep if c in df.columns]

    return df[keep]

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — BUILD fact_orders
# ─────────────────────────────────────────────────────────────────────────────

def build_fact_orders(raw: dict) -> pd.DataFrame:
    """
    Build the central fact table.

    Joins: orders + order_items + payments + reviews
    Simulates: discount_amount, channel_id, gross_profit
    (the Olist dataset does not include marketing channels
     or discounts, so we simulate realistic distributions)
    """

    np.random.seed(42)

    orders  = raw["orders"].copy()
    items   = raw["order_items"].copy()
    pays    = raw["payments"].copy()
    reviews = raw["reviews"].copy()

    # ── Parse all timestamp columns ──────────────────────────────────────────
    ts_cols = {
        "order_purchase_timestamp":      "order_purchase_ts",
        "order_approved_at":             "order_approved_ts",
        "order_delivered_customer_date": "order_delivered_ts",
        "order_estimated_delivery_date": "estimated_delivery",
    }
    for original, new_name in ts_cols.items():
        orders[new_name] = pd.to_datetime(
            orders[original], errors="coerce"
        ).dt.strftime("%Y-%m-%d %H:%M:%S")
    orders = orders.drop(columns=list(ts_cols.keys()))

    # ── Derive date_id from purchase timestamp ────────────────────────────────
    orders["date_id"] = pd.to_datetime(
        orders["order_purchase_ts"], errors="coerce"
    ).dt.strftime("%Y%m%d").astype("Int64")

    # ── Aggregate payments per order ──────────────────────────────────────────
    # Some orders use multiple payment methods (e.g. card + voucher)
    pay_agg = (
        pays.groupby("order_id")
        .agg(
            unit_price    = ("payment_value",        "sum"),
            payment_type  = ("payment_type",         "first"),
            installments  = ("payment_installments", "max"),
        )
        .reset_index()
    )

    # ── Aggregate order items per order ───────────────────────────────────────
    # Some orders contain multiple products — we keep the primary product
    items_agg = (
        items.groupby("order_id")
        .agg(
            product_id    = ("product_id",      "first"),
            seller_id     = ("seller_id",       "first"),
            quantity      = ("order_item_id",   "count"),
            freight_value = ("freight_value",   "sum"),
        )
        .reset_index()
    )

    # ── Aggregate review scores ───────────────────────────────────────────────
    review_agg = (
        reviews.groupby("order_id")["review_score"]
        .mean()
        .round()
        .astype("Int64")
        .reset_index()
    )

    # ── Merge all sources ─────────────────────────────────────────────────────
    df = orders.merge(items_agg,   on="order_id", how="left")
    df = df.merge(pay_agg,         on="order_id", how="left")
    df = df.merge(review_agg,      on="order_id", how="left")

    # ── Simulate discount_amount ──────────────────────────────────────────────
    # 15% of orders received a discount between 5% and 30%
    has_discount  = np.random.random(len(df)) < 0.15
    discount_pct  = np.random.uniform(0.05, 0.30, len(df))
    df["discount_amount"] = np.where(
        has_discount,
        (df["unit_price"] * discount_pct).round(2),
        0.0
    )

    # ── Simulate channel assignment ───────────────────────────────────────────
    channels = ["CH001", "CH002", "CH003", "CH004", "CH005", "CH006"]
    weights  = [0.25,    0.30,    0.15,    0.15,    0.05,    0.10  ]
    df["channel_id"] = np.random.choice(channels, size=len(df), p=weights)

    # ── Compute revenue and gross profit ──────────────────────────────────────
    df["revenue"]      = (df["unit_price"] - df["discount_amount"]).round(2)
    df["gross_profit"] = ((df["unit_price"] * 0.40) - df["freight_value"]).round(2)

    # ── Final column selection and ordering ───────────────────────────────────
    return df[[
        "order_id", "customer_id", "product_id", "seller_id",
        "date_id",  "channel_id",
        "quantity", "unit_price", "freight_value", "discount_amount",
        "revenue",  "gross_profit",
        "order_status", "payment_type", "installments", "review_score",
        "order_purchase_ts", "order_approved_ts",
        "order_delivered_ts", "estimated_delivery",
    ]]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — WRITE TO SQLITE
# ─────────────────────────────────────────────────────────────────────────────

def write_to_sqlite(tables: dict, db_path: Path) -> None:
    """Write all DataFrames to the SQLite warehouse."""

    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove old DB so we always start clean
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)

    # Load and execute schema SQL files (creates tables + indexes)
    schema_dir = Path("sql/schema")
    schema_order = [
        "dim_time.sql",
        "dim_customers.sql",
        "dim_products.sql",
        "dim_channels.sql",
        "fact_orders.sql",
    ]

    print("\n" + "=" * 55)
    print("STEP 6a — Creating schema from SQL files")
    print("=" * 55)

    for filename in schema_order:
        path = schema_dir / filename
        if path.exists():
            conn.executescript(path.read_text())
            print(f"  ✓ Applied {filename}")
        else:
            print(f"  ⚠ Skipped (not found): {filename}")

    # Write DataFrames into the tables
    print("\n" + "=" * 55)
    print("STEP 6b — Writing data to tables")
    print("=" * 55)

    for name, df in tables.items():
        df.to_sql(name, conn, if_exists="replace", index=False)
        print(f"  ✓ {name:<20} {len(df):>8,} rows written")

    conn.commit()
    conn.close()
    print(f"\n  Database saved → {db_path}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — VALIDATE
# ─────────────────────────────────────────────────────────────────────────────

def validate_database(db_path: Path) -> None:
    """Run basic sanity checks on the loaded data."""

    conn = sqlite3.connect(db_path)

    print("\n" + "=" * 55)
    print("STEP 7 — Validation")
    print("=" * 55)

    checks = {
        "Total orders": """
            SELECT COUNT(*) FROM fact_orders
        """,
        "Delivered orders": """
            SELECT COUNT(*) FROM fact_orders
            WHERE order_status = 'delivered'
        """,
        "Total revenue (delivered)": """
            SELECT ROUND(SUM(revenue), 2)
            FROM fact_orders
            WHERE order_status = 'delivered'
        """,
        "Avg order value": """
            SELECT ROUND(AVG(revenue), 2)
            FROM fact_orders
            WHERE order_status = 'delivered'
        """,
        "Unique customers": """
            SELECT COUNT(DISTINCT customer_id)
            FROM dim_customers
        """,
        "Unique products": """
            SELECT COUNT(DISTINCT product_id)
            FROM dim_products
        """,
        "Orphaned orders (bad joins)": """
            SELECT COUNT(*)
            FROM fact_orders f
            LEFT JOIN dim_customers c ON f.customer_id = c.customer_id
            WHERE c.customer_id IS NULL
        """,
        "Orders with missing date_id": """
            SELECT COUNT(*)
            FROM fact_orders
            WHERE date_id IS NULL
        """,
        "Negative revenue rows": """
            SELECT COUNT(*)
            FROM fact_orders
            WHERE revenue < 0
        """,
    }

    for label, query in checks.items():
        result = pd.read_sql(query, conn).iloc[0, 0]
        print(f"  {label:<35} {result}")

    # Date range
    date_range = pd.read_sql("""
        SELECT
            MIN(t.full_date) AS earliest,
            MAX(t.full_date) AS latest
        FROM fact_orders f
        JOIN dim_time t ON f.date_id = t.date_id
    """, conn)
    print(f"\n  Date range: {date_range.iloc[0,0]}  →  {date_range.iloc[0,1]}")

    # Revenue by channel
    print("\n  Revenue by channel:")
    channel_rev = pd.read_sql("""
        SELECT
            c.channel_name,
            COUNT(*)                          AS orders,
            ROUND(SUM(f.revenue), 2)          AS total_revenue,
            ROUND(AVG(f.revenue), 2)          AS avg_order_value
        FROM fact_orders f
        JOIN dim_channels c ON f.channel_id = c.channel_id
        WHERE f.order_status = 'delivered'
        GROUP BY c.channel_name
        ORDER BY total_revenue DESC
    """, conn)
    print(channel_rev.to_string(index=False))

    conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 55)
    print("  E-COMMERCE ANALYTICS — PHASE 2 DATA PIPELINE")
    print("=" * 55 + "\n")

    # 1. Load raw data
    raw = load_raw_data()

    # 2. Build dimension tables
    print("\n" + "=" * 55)
    print("STEP 2–5 — Building star schema tables")
    print("=" * 55)

    dim_time      = build_dim_time()
    print(f"  ✓ dim_time        {len(dim_time):>8,} rows")

    dim_customers = build_dim_customers(raw)
    print(f"  ✓ dim_customers   {len(dim_customers):>8,} rows")

    dim_products  = build_dim_products(raw)
    print(f"  ✓ dim_products    {len(dim_products):>8,} rows")

    fact_orders   = build_fact_orders(raw)
    print(f"  ✓ fact_orders     {len(fact_orders):>8,} rows")

    # 3. Write to SQLite
    tables = {
        "dim_time":      dim_time,
        "dim_customers": dim_customers,
        "dim_products":  dim_products,
        "fact_orders":   fact_orders,
    }
    write_to_sqlite(tables, DB_PATH)

    # 4. Validate
    validate_database(DB_PATH)

    # 5. Save processed CSVs for notebooks
    print("\n" + "=" * 55)
    print("STEP 8 — Saving processed CSVs")
    print("=" * 55)
    for name, df in tables.items():
        out = DATA_PROC / f"{name}.csv"
        df.to_csv(out, index=False)
        print(f"  ✓ Saved {out}")

    print("\n  Phase 2 complete. Ready for Phase 3.\n")


if __name__ == "__main__":
    main()