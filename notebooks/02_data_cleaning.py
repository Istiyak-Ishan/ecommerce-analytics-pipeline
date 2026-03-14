"""
Phase 3 — Data Cleaning Pipeline
==================================
Reads the raw star schema from ecommerce.db,
applies a full professional cleaning pipeline,
engineers features, and saves a clean analytical
dataset ready for EDA, cohort analysis, and ML.

Run from project root with venv active:
    python notebooks/02_data_cleaning.py
"""

import sqlite3
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

DB_PATH      = Path("data/ecommerce.db")
PROCESSED    = Path("data/processed")
REPORTS      = Path("reports")

PROCESSED.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

# Reference date for recency calculations — last date in dataset
REFERENCE_DATE = pd.Timestamp("2018-10-17")

# Audit log — every cleaning action gets recorded here
audit_log = []

def log(step: str, table: str, action: str, rows_affected: int, detail: str = ""):
    """Append one record to the audit log."""
    audit_log.append({
        "timestamp":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "step":          step,
        "table":         table,
        "action":        action,
        "rows_affected": rows_affected,
        "detail":        detail,
    })
    symbol = "✓" if rows_affected == 0 else "!"
    print(f"  [{symbol}] {table:<20} {action:<35} {rows_affected:>6} rows  {detail}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — LOAD AND PROFILE
# ─────────────────────────────────────────────────────────────────────────────

def load_tables(db_path: Path) -> dict:
    """Load all tables from SQLite into DataFrames."""
    conn = sqlite3.connect(db_path)
    tables = {}
    for name in ["fact_orders", "dim_customers", "dim_products",
                 "dim_time", "dim_channels"]:
        tables[name] = pd.read_sql(f"SELECT * FROM {name}", conn)
    conn.close()
    return tables


def profile_table(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Generate a full data quality profile for one table.
    Returns a summary DataFrame showing nulls, dtypes,
    unique counts, and basic stats per column.
    """
    profile = pd.DataFrame({
        "column":       df.columns,
        "dtype":        df.dtypes.values,
        "null_count":   df.isnull().sum().values,
        "null_pct":     (df.isnull().mean() * 100).round(2).values,
        "unique_count": df.nunique().values,
        "sample_value": [df[c].dropna().iloc[0]
                         if df[c].notna().any() else None
                         for c in df.columns],
    })

    print(f"\n  {'─'*60}")
    print(f"  Profile: {name}  ({len(df):,} rows × {len(df.columns)} cols)")
    print(f"  {'─'*60}")
    print(profile.to_string(index=False))

    # Flag columns with high null rate
    high_null = profile[profile["null_pct"] > 20]
    if len(high_null):
        print(f"\n  WARNING — High null columns (>20%):")
        for _, row in high_null.iterrows():
            print(f"    {row['column']}: {row['null_pct']}% null")

    return profile


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — REMOVE DUPLICATES
# ─────────────────────────────────────────────────────────────────────────────

def remove_duplicates(tables: dict) -> dict:
    """
    Remove duplicate rows from all tables.
    For fact_orders: deduplicate on order_id (primary key).
    For dimension tables: deduplicate on their primary keys.
    """
    print("\n" + "=" * 60)
    print("STEP 2 — Removing duplicates")
    print("=" * 60)

    pk_map = {
        "fact_orders":   "order_id",
        "dim_customers": "customer_id",
        "dim_products":  "product_id",
        "dim_time":      "date_id",
        "dim_channels":  "channel_id",
    }

    for name, pk in pk_map.items():
        df = tables[name]
        before = len(df)

        # Full row duplicates
        df = df.drop_duplicates()
        full_dups = before - len(df)

        # Primary key duplicates (keep first occurrence)
        pk_dups = df.duplicated(subset=[pk], keep="first").sum()
        df = df.drop_duplicates(subset=[pk], keep="first")

        tables[name] = df
        log("step2", name, "drop full-row duplicates", full_dups)
        log("step2", name, "drop primary-key duplicates", pk_dups,
            f"pk={pk}")

    return tables


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — HANDLE MISSING VALUES
# ─────────────────────────────────────────────────────────────────────────────

def handle_missing_values(tables: dict) -> dict:
    """
    Handle nulls with the correct strategy per column:

    - Numeric measures  → fill with 0 or median (context-dependent)
    - Categorical text  → fill with 'unknown'
    - Timestamps        → leave null but add a boolean flag column
    - Critical ID cols  → drop the row if null (can't analyse without it)
    """
    print("\n" + "=" * 60)
    print("STEP 3 — Handling missing values")
    print("=" * 60)

    # ── fact_orders ──────────────────────────────────────────────────────────
    fo = tables["fact_orders"].copy()

    # Critical: drop orders with no customer_id (can't track them)
    null_customers = fo["customer_id"].isnull().sum()
    fo = fo.dropna(subset=["customer_id"])
    log("step3", "fact_orders", "drop rows — null customer_id",
        null_customers)

    # Critical: drop orders with no date_id (can't do time analysis)
    null_dates = fo["date_id"].isnull().sum()
    fo = fo.dropna(subset=["date_id"])
    log("step3", "fact_orders", "drop rows — null date_id", null_dates)

    # Numeric measures: fill with 0 where null means "none"
    numeric_fill_zero = ["discount_amount", "freight_value"]
    for col in numeric_fill_zero:
        n = fo[col].isnull().sum()
        fo[col] = fo[col].fillna(0)
        log("step3", "fact_orders", f"fill 0 — {col}", n)

    # unit_price: fill nulls with median (can't have a £0 order)
    n = fo["unit_price"].isnull().sum()
    median_price = fo["unit_price"].median()
    fo["unit_price"] = fo["unit_price"].fillna(median_price)
    log("step3", "fact_orders", "fill median — unit_price", n,
        f"median={median_price:.2f}")

    # Review score: fill with -1 to flag "no review given"
    n = fo["review_score"].isnull().sum()
    fo["review_score"] = fo["review_score"].fillna(-1).astype(int)
    log("step3", "fact_orders", "fill -1 — review_score (no review)", n)

    # payment_type: fill unknown
    n = fo["payment_type"].isnull().sum()
    fo["payment_type"] = fo["payment_type"].fillna("unknown")
    log("step3", "fact_orders", "fill 'unknown' — payment_type", n)

    # Timestamps: add a boolean flag for missing delivery date
    # (critical for delivery delay feature engineering later)
    fo["has_delivery_date"] = fo["order_delivered_ts"].notna().astype(int)
    n = fo["order_delivered_ts"].isnull().sum()
    log("step3", "fact_orders",
        "add flag — has_delivery_date", n,
        "1=delivered, 0=missing")

    # Recalculate revenue and gross_profit in case unit_price was filled
    fo["revenue"]      = (fo["unit_price"] - fo["discount_amount"]).round(2)
    fo["gross_profit"] = ((fo["unit_price"] * 0.40) - fo["freight_value"]).round(2)

    tables["fact_orders"] = fo

    # ── dim_products ─────────────────────────────────────────────────────────
    dp = tables["dim_products"].copy()

    # Category: fill unknown so groupbys don't silently drop them
    n = dp["category_english"].isnull().sum()
    dp["category_english"] = dp["category_english"].fillna("unknown")
    dp["category_name"]    = dp["category_name"].fillna("unknown")
    log("step3", "dim_products", "fill 'unknown' — category", n)

    # Physical dims: fill with median (needed for shipping cost models)
    for col in ["weight_g", "length_cm", "height_cm", "width_cm"]:
        n = dp[col].isnull().sum()
        med = dp[col].median()
        dp[col] = dp[col].fillna(med)
        log("step3", "dim_products", f"fill median — {col}", n,
            f"median={med:.1f}")

    # photos_qty: fill 0 (no photos is a valid state)
    n = dp["photos_qty"].isnull().sum()
    dp["photos_qty"] = dp["photos_qty"].fillna(0).astype(int)
    log("step3", "dim_products", "fill 0 — photos_qty", n)

    # price: fill with median
    n = dp["price"].isnull().sum()
    med = dp["price"].median()
    dp["price"] = dp["price"].fillna(med)
    log("step3", "dim_products", "fill median — price", n,
        f"median={med:.2f}")

    tables["dim_products"] = dp

    # ── dim_customers ─────────────────────────────────────────────────────────
    dc = tables["dim_customers"].copy()

    n = dc["city"].isnull().sum()
    dc["city"]  = dc["city"].fillna("unknown")
    dc["state"] = dc["state"].fillna("unknown")
    log("step3", "dim_customers", "fill 'unknown' — city/state", n)

    tables["dim_customers"] = dc

    return tables


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — FIX DATA TYPES
# ─────────────────────────────────────────────────────────────────────────────

def fix_data_types(tables: dict) -> dict:
    """
    Enforce correct dtypes on every column:
    - Timestamps as pd.Timestamp
    - IDs as string (not int — they're identifiers, not numbers)
    - Prices / revenue as float64
    - Integers as int64
    - Categories as pd.Categorical (saves memory, enables fast groupby)
    """
    print("\n" + "=" * 60)
    print("STEP 4 — Fixing data types")
    print("=" * 60)

    # ── fact_orders ──────────────────────────────────────────────────────────
    fo = tables["fact_orders"].copy()

    # Parse all timestamp strings into proper datetime objects
    ts_cols = ["order_purchase_ts", "order_approved_ts",
               "order_delivered_ts", "estimated_delivery"]
    for col in ts_cols:
        if col in fo.columns:
            before_dtype = fo[col].dtype
            fo[col] = pd.to_datetime(fo[col], errors="coerce")
            log("step4", "fact_orders", f"str → datetime — {col}", 0,
                f"{before_dtype} → datetime64")

    # Enforce numeric types
    float_cols = ["unit_price", "freight_value", "discount_amount",
                  "revenue", "gross_profit"]
    for col in float_cols:
        fo[col] = pd.to_numeric(fo[col], errors="coerce").astype("float64")

    int_cols = ["quantity", "installments", "review_score",
                "date_id", "has_delivery_date"]
    for col in int_cols:
        if col in fo.columns:
            fo[col] = pd.to_numeric(fo[col], errors="coerce").astype("Int64")

    # Convert string ID columns to str (safety against int coercion)
    for col in ["order_id", "customer_id", "product_id",
                "seller_id", "channel_id"]:
        fo[col] = fo[col].astype(str)

    # Low-cardinality text columns → Categorical (saves ~60% memory)
    cat_cols = ["order_status", "payment_type"]
    for col in cat_cols:
        fo[col] = fo[col].astype("category")
        log("step4", "fact_orders", f"str → category — {col}", 0)

    tables["fact_orders"] = fo

    # ── dim_products ─────────────────────────────────────────────────────────
    dp = tables["dim_products"].copy()
    dp["price"]    = pd.to_numeric(dp["price"],    errors="coerce").astype("float64")
    dp["weight_g"] = pd.to_numeric(dp["weight_g"], errors="coerce").astype("float64")
    for col in ["category_english", "category_name"]:
        dp[col] = dp[col].astype("category")
    log("step4", "dim_products", "enforced float64 + category dtypes", 0)
    tables["dim_products"] = dp

    # ── dim_customers ─────────────────────────────────────────────────────────
    dc = tables["dim_customers"].copy()
    dc["first_purchase_date"] = pd.to_datetime(
        dc["first_purchase_date"], errors="coerce"
    )
    dc["state"] = dc["state"].astype("category")
    log("step4", "dim_customers", "str → datetime — first_purchase_date", 0)
    tables["dim_customers"] = dc

    # ── dim_time ──────────────────────────────────────────────────────────────
    dt = tables["dim_time"].copy()
    dt["full_date"]  = pd.to_datetime(dt["full_date"], errors="coerce")
    dt["is_weekend"] = dt["is_weekend"].astype(bool)
    dt["is_holiday"] = dt["is_holiday"].astype(bool)
    log("step4", "dim_time", "str → datetime — full_date", 0)
    tables["dim_time"] = dt

    return tables


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — DETECT AND HANDLE OUTLIERS
# ─────────────────────────────────────────────────────────────────────────────

def handle_outliers(tables: dict) -> dict:
    """
    Use the IQR (Interquartile Range) method to detect outliers
    in price and revenue columns.

    Strategy:
    - Log the outlier count (for the audit report)
    - CAP outliers at the fence value (Winsorization)
      rather than dropping rows — we don't want to lose data,
      just reduce the distorting effect of extreme values.

    IQR method:
        Q1 = 25th percentile
        Q3 = 75th percentile
        IQR = Q3 - Q1
        Lower fence = Q1 - 1.5 * IQR
        Upper fence = Q3 + 1.5 * IQR
    """
    print("\n" + "=" * 60)
    print("STEP 5 — Detecting and capping outliers")
    print("=" * 60)

    fo = tables["fact_orders"].copy()

    def iqr_bounds(series: pd.Series):
        Q1  = series.quantile(0.25)
        Q3  = series.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        return lower, upper

    outlier_cols = ["unit_price", "revenue", "freight_value", "gross_profit"]

    for col in outlier_cols:
        series = fo[col].dropna()
        lower, upper = iqr_bounds(series)

        n_low  = (fo[col] < lower).sum()
        n_high = (fo[col] > upper).sum()
        total  = n_low + n_high

        # Add flag BEFORE capping so we know which rows were affected
        fo[f"{col}_outlier_flag"] = (
            (fo[col] < lower) | (fo[col] > upper)
        ).astype(int)

        # Winsorize: cap at fences
        fo[col] = fo[col].clip(lower=lower, upper=upper)

        log("step5", "fact_orders",
            f"cap outliers — {col}",
            total,
            f"lower={lower:.2f}, upper={upper:.2f}, "
            f"low_count={n_low}, high_count={n_high}")

    # Special case: any revenue still negative after capping → set to 0
    neg_rev = (fo["revenue"] < 0).sum()
    fo["revenue"] = fo["revenue"].clip(lower=0)
    log("step5", "fact_orders", "clip negative revenue → 0", neg_rev)

    # Check for physically impossible values
    impossible_price = (fo["unit_price"] <= 0).sum()
    fo = fo[fo["unit_price"] > 0]
    log("step5", "fact_orders",
        "drop rows — unit_price <= 0", impossible_price)

    tables["fact_orders"] = fo
    return tables


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(tables: dict) -> dict:
    """
    Create analytical features that don't exist in the raw data
    but are essential for EDA, cohort analysis, CLV, and ML.

    Features created:
    ─────────────────────────────────────────────────
    On fact_orders:
      delivery_delay_days    — actual vs estimated delivery
      is_late_delivery       — boolean flag
      order_value_tier       — Low / Medium / High / Premium
      has_discount           — boolean flag
      discount_pct           — what % was discounted
      order_hour             — hour of purchase (0–23)
      order_day_of_week      — Monday–Sunday
      order_month            — 1–12
      order_quarter          — Q1–Q4
      days_since_purchase    — recency from reference date

    On dim_customers:
      total_orders           — how many orders placed
      total_revenue          — lifetime revenue
      avg_order_value        — mean order value
      first_order_date       — earliest purchase
      last_order_date        — most recent purchase
      recency_days           — days since last order
      frequency              — total order count (RFM F)
      monetary               — total revenue (RFM M)
      recency                — days since last order (RFM R)
      rfm_segment            — Champions / Loyal / At Risk / Lost
      is_repeat_buyer        — ordered more than once
      churn_flag             — no order in last 180 days
      customer_segment       — updated segment label
    """
    print("\n" + "=" * 60)
    print("STEP 6 — Feature engineering")
    print("=" * 60)

    fo = tables["fact_orders"].copy()
    dc = tables["dim_customers"].copy()

    # ── 6a. Delivery features ─────────────────────────────────────────────────
    fo["estimated_delivery"] = pd.to_datetime(fo["estimated_delivery"], errors="coerce")
    fo["order_delivered_ts"] = pd.to_datetime(fo["order_delivered_ts"], errors="coerce")

    fo["delivery_delay_days"] = (
        (fo["order_delivered_ts"] - fo["estimated_delivery"])
        .dt.days
    )

    # Only meaningful for delivered orders
    fo["is_late_delivery"] = (
        (fo["delivery_delay_days"] > 0) &
        (fo["order_status"] == "delivered")
    ).astype(int)

    n_late = fo["is_late_delivery"].sum()
    log("step6", "fact_orders", "engineer — delivery_delay_days + is_late", n_late,
        f"{n_late:,} late deliveries")

    # ── 6b. Order value tier ──────────────────────────────────────────────────
    # Bin order value into business-meaningful tiers
    fo["order_value_tier"] = pd.cut(
        fo["revenue"],
        bins=[0, 50, 150, 500, np.inf],
        labels=["low", "medium", "high", "premium"],
        right=True
    ).astype(str)
    log("step6", "fact_orders", "engineer — order_value_tier", 0,
        "bins: 0|50|150|500|inf")

    # ── 6c. Discount features ─────────────────────────────────────────────────
    fo["has_discount"] = (fo["discount_amount"] > 0).astype(int)
    fo["discount_pct"] = np.where(
        fo["unit_price"] > 0,
        (fo["discount_amount"] / fo["unit_price"] * 100).round(2),
        0
    )
    n_discounted = fo["has_discount"].sum()
    log("step6", "fact_orders", "engineer — has_discount + discount_pct",
        n_discounted, f"{n_discounted:,} discounted orders")

    # ── 6d. Temporal features ─────────────────────────────────────────────────
    fo["order_purchase_ts"] = pd.to_datetime(fo["order_purchase_ts"], errors="coerce")

    fo["order_hour"]        = fo["order_purchase_ts"].dt.hour
    fo["order_day_of_week"] = fo["order_purchase_ts"].dt.day_name()
    fo["order_month"]       = fo["order_purchase_ts"].dt.month
    fo["order_quarter"]     = fo["order_purchase_ts"].dt.quarter
    fo["order_year"]        = fo["order_purchase_ts"].dt.year
    fo["days_since_purchase"] = (
        REFERENCE_DATE - fo["order_purchase_ts"]
    ).dt.days
    log("step6", "fact_orders",
        "engineer — temporal features (hour, dow, month, quarter)", 0)

    # ── 6e. RFM + Customer-level features ─────────────────────────────────────
    # Aggregate delivered orders only for RFM (cancelled orders
    # shouldn't count as purchases)
    delivered = fo[fo["order_status"] == "delivered"].copy()

    rfm = (
        delivered.groupby("customer_id")
        .agg(
            total_orders      = ("order_id",           "count"),
            total_revenue     = ("revenue",            "sum"),
            avg_order_value   = ("revenue",            "mean"),
            first_order_date  = ("order_purchase_ts",  "min"),
            last_order_date   = ("order_purchase_ts",  "max"),
        )
        .reset_index()
    )

    rfm["total_revenue"]   = rfm["total_revenue"].round(2)
    rfm["avg_order_value"] = rfm["avg_order_value"].round(2)

    # Recency = days from last order to reference date
    rfm["recency_days"] = (
        REFERENCE_DATE - rfm["last_order_date"]
    ).dt.days

    # RFM aliases (standard naming for ML features)
    rfm["recency"]   = rfm["recency_days"]
    rfm["frequency"] = rfm["total_orders"]
    rfm["monetary"]  = rfm["total_revenue"]

    # ── 6f. RFM segmentation ──────────────────────────────────────────────────
    # Score each dimension 1–4 using quartile ranks
    # For recency: LOWER is better (bought recently), so we invert
    rfm["r_score"] = pd.qcut(rfm["recency"],   q=4,
                              labels=[4, 3, 2, 1]).astype(int)
    rfm["f_score"] = pd.qcut(rfm["frequency"].rank(method="first"),
                              q=4, labels=[1, 2, 3, 4]).astype(int)
    rfm["m_score"] = pd.qcut(rfm["monetary"],  q=4,
                              labels=[1, 2, 3, 4]).astype(int)

    rfm["rfm_score"] = (
        rfm["r_score"].astype(str) +
        rfm["f_score"].astype(str) +
        rfm["m_score"].astype(str)
    )

    def rfm_label(row):
        r, f, m = row["r_score"], row["f_score"], row["m_score"]
        if r >= 4 and f >= 4:
            return "champions"
        elif r >= 3 and f >= 3:
            return "loyal"
        elif r >= 3 and f <= 2:
            return "potential_loyal"
        elif r <= 2 and f >= 3:
            return "at_risk"
        elif r == 1 and f == 1:
            return "lost"
        else:
            return "need_attention"

    rfm["rfm_segment"] = rfm.apply(rfm_label, axis=1)

    # ── 6g. Behavioural flags ─────────────────────────────────────────────────
    rfm["is_repeat_buyer"] = (rfm["total_orders"] > 1).astype(int)

    # Churn flag: no purchase in 180 days
    rfm["churn_flag"] = (rfm["recency_days"] > 180).astype(int)

    # Final customer segment (used in dim_customers)
    def customer_segment(row):
        if row["churn_flag"] == 1:
            return "churned"
        elif row["rfm_segment"] == "champions":
            return "vip"
        elif row["total_orders"] > 1:
            return "returning"
        else:
            return "new"

    rfm["customer_segment"] = rfm.apply(customer_segment, axis=1)

    # Log segment distribution
    seg_dist = rfm["rfm_segment"].value_counts().to_dict()
    log("step6", "dim_customers",
        "engineer — RFM scores + segments", 0,
        str(seg_dist))

# ── 6h. Merge RFM back into dim_customers ─────────────────────────────────
    rfm_merge_cols = [
        "customer_id", "total_orders", "total_revenue", "avg_order_value",
        "first_order_date", "last_order_date", "recency_days",
        "recency", "frequency", "monetary",
        "r_score", "f_score", "m_score", "rfm_score", "rfm_segment",
        "is_repeat_buyer", "churn_flag", "customer_segment"
    ]
    dc = dc.drop(columns=["customer_segment"], errors="ignore")  # ← ADD THIS
    dc = dc.merge(rfm[rfm_merge_cols], on="customer_id", how="left")

    # Customers with no delivered orders get default values
    dc["total_orders"]    = dc["total_orders"].fillna(0).astype(int)
    dc["total_revenue"]   = dc["total_revenue"].fillna(0)
    dc["rfm_segment"]     = dc["rfm_segment"].fillna("unknown")
    dc["customer_segment"] = dc["customer_segment"].fillna("new")
    dc["churn_flag"]      = dc["churn_flag"].fillna(1).astype(int)
    dc["is_repeat_buyer"] = dc["is_repeat_buyer"].fillna(0).astype(int)

    tables["fact_orders"]   = fo
    tables["dim_customers"] = dc

    return tables
# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — FINAL VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def validate_clean_data(tables: dict) -> None:
    """
    Run a suite of data quality assertions on the cleaned tables.
    These are the checks a senior analyst would run before
    handing data off to the business.
    """
    print("\n" + "=" * 60)
    print("STEP 7 — Final validation")
    print("=" * 60)

    fo = tables["fact_orders"]
    dc = tables["dim_customers"]
    dp = tables["dim_products"]

    checks = {
        "No null customer_id in fact_orders":
            fo["customer_id"].isnull().sum() == 0,

        "No null date_id in fact_orders":
            fo["date_id"].isnull().sum() == 0,

        "No negative revenue":
            (fo["revenue"] < 0).sum() == 0,

        "No zero-price orders":
            (fo["unit_price"] <= 0).sum() == 0,

        "All review scores valid (-1 or 1–5)":
            fo["review_score"].isin([-1, 1, 2, 3, 4, 5]).all(),

        "No null category in dim_products":
            dp["category_english"].isnull().sum() == 0,

        "RFM segments populated":
            dc["rfm_segment"].isnull().sum() == 0,

        "churn_flag is 0 or 1 only":
            dc["churn_flag"].isin([0, 1]).all(),

        "order_value_tier has 4 levels":
            fo["order_value_tier"].nunique() == 4,
    }

    all_passed = True
    for description, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        symbol = "✓" if passed else "✗"
        print(f"  [{symbol}] {status}  {description}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n  All validation checks passed.")
    else:
        print("\n  Some checks failed — review the log above.")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — SAVE OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────

def save_outputs(tables: dict) -> None:
    """
    Save three output formats:
    1. Cleaned CSVs for notebooks and Tableau/Power BI
    2. Updated SQLite tables (cleaned versions)
    3. Audit log as CSV for documentation
    """
    print("\n" + "=" * 60)
    print("STEP 8 — Saving outputs")
    print("=" * 60)

    # ── Save cleaned CSVs ─────────────────────────────────────────────────────
    for name, df in tables.items():
        out = PROCESSED / f"{name}_clean.csv"
        df.to_csv(out, index=False)
        print(f"  ✓ Saved {out}  ({len(df):,} rows)")

    # ── Save one master analytical file ──────────────────────────────────────
    # Join fact_orders with customer RFM and product category
    # This is the single file used for 90% of downstream analysis
    fo = tables["fact_orders"].copy()
    dc = tables["dim_customers"][[
        "customer_id", "state", "rfm_segment",
        "customer_segment", "churn_flag", "is_repeat_buyer",
        "recency", "frequency", "monetary"
    ]].copy()
    dp = tables["dim_products"][[
        "product_id", "category_english", "price"
    ]].copy()
    dt = tables["dim_time"][[
        "date_id", "year", "quarter", "month_name",
        "day_of_week", "is_weekend"
    ]].copy()
    dc_ch = tables["dim_channels"][[
        "channel_id", "channel_name", "cac"
    ]].copy()

    master = (
        fo
        .merge(dc,    on="customer_id", how="left")
        .merge(dp,    on="product_id",  how="left")
        .merge(dt,    on="date_id",     how="left")
        .merge(dc_ch, on="channel_id",  how="left")
    )

    master_path = PROCESSED / "master_analytical.csv"
    master.to_csv(master_path, index=False)
    print(f"\n  ✓ Master analytical file: {master_path}")
    print(f"    {len(master):,} rows × {len(master.columns)} columns")

    # ── Save audit log ────────────────────────────────────────────────────────
    audit_df = pd.DataFrame(audit_log)
    audit_path = REPORTS / "cleaning_audit_log.csv"
    audit_df.to_csv(audit_path, index=False)
    print(f"\n  ✓ Audit log: {audit_path}")
    print(f"    {len(audit_df)} cleaning actions recorded")

    # ── Update SQLite with cleaned tables ─────────────────────────────────────
    conn = sqlite3.connect(DB_PATH)
    for name, df in tables.items():
        # Save cleaned versions under new names
        clean_name = f"{name}_clean"
        df_save = df.copy()
        # Convert all datetime cols to string for SQLite compatibility
        for col in df_save.select_dtypes(include=["datetime64[ns]"]).columns:
            df_save[col] = df_save[col].astype(str)
        df_save.to_sql(clean_name, conn, if_exists="replace", index=False)
    conn.close()
    print(f"\n  ✓ Cleaned tables written to {DB_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 — PRINT SUMMARY REPORT
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(tables: dict) -> None:
    """Print a business-readable summary of the clean dataset."""

    fo = tables["fact_orders"]
    dc = tables["dim_customers"]

    delivered = fo[fo["order_status"] == "delivered"]

    print("\n" + "=" * 60)
    print("  CLEAN DATASET SUMMARY")
    print("=" * 60)

    print(f"\n  Orders")
    print(f"    Total orders:         {len(fo):>10,}")
    print(f"    Delivered orders:     {len(delivered):>10,}")
    print(f"    Total revenue:        {delivered['revenue'].sum():>10,.2f}")
    print(f"    Avg order value:      {delivered['revenue'].mean():>10,.2f}")
    print(f"    Orders with discount: {fo['has_discount'].sum():>10,}"
          f"  ({fo['has_discount'].mean()*100:.1f}%)")
    print(f"    Late deliveries:      {fo['is_late_delivery'].sum():>10,}")

    print(f"\n  Customers")
    print(f"    Total customers:      {len(dc):>10,}")
    print(f"    Repeat buyers:        {dc['is_repeat_buyer'].sum():>10,}"
          f"  ({dc['is_repeat_buyer'].mean()*100:.1f}%)")
    print(f"    Churned (>180 days):  {dc['churn_flag'].sum():>10,}"
          f"  ({dc['churn_flag'].mean()*100:.1f}%)")

    print(f"\n  RFM Segments")
    seg = dc["rfm_segment"].value_counts()
    for segment, count in seg.items():
        pct = count / len(dc) * 100
        bar = "█" * int(pct / 2)
        print(f"    {segment:<20} {count:>7,}  {pct:5.1f}%  {bar}")

    print(f"\n  Order value tiers")
    tiers = fo["order_value_tier"].value_counts()
    for tier, count in tiers.items():
        pct = count / len(fo) * 100
        print(f"    {tier:<12} {count:>8,}  {pct:5.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  E-COMMERCE ANALYTICS — PHASE 3 CLEANING PIPELINE")
    print("=" * 60)

    # 1. Load
    tables = load_tables(DB_PATH)

    # 2. Profile (shows you what we're dealing with before touching anything)
    print("\n" + "=" * 60)
    print("STEP 1 — Data profiling")
    print("=" * 60)
    for name in ["fact_orders", "dim_products", "dim_customers"]:
        profile_table(tables[name], name)

    # 3. Clean
    tables = remove_duplicates(tables)
    tables = handle_missing_values(tables)
    tables = fix_data_types(tables)
    tables = handle_outliers(tables)
    tables = engineer_features(tables)

    # 4. Validate
    validate_clean_data(tables)

    # 5. Save
    save_outputs(tables)

    # 6. Summary
    print_summary(tables)

    print("\n  Phase 3 complete. Ready for Phase 4 — SQL Analytics.\n")


if __name__ == "__main__":
    main()