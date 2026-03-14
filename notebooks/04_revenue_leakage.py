"""
Phase 5 — Revenue Leakage Analysis
=====================================
Identifies and quantifies every source of revenue leakage
in the e-commerce pipeline:

  L1 — Excessive discount usage
  L2 — Cancelled order revenue loss
  L3 — Pricing anomalies (same product sold at very different prices)
  L4 — Late delivery impact (review damage + estimated refund risk)
  L5 — Freight billing mismatches
  L6 — High-value customers receiving unnecessary discounts

Outputs:
  reports/leakage_summary.csv
  reports/leakage_detail_*.csv  (one per source)
  reports/leakage_report.txt    (executive summary)

Run from project root:
    python notebooks/04_revenue_leakage.py
"""

import sqlite3
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

DB_PATH  = Path("data/ecommerce.db")
REPORTS  = Path("reports")
REPORTS.mkdir(parents=True, exist_ok=True)

# Thresholds — tweak these to match business rules
EXCESSIVE_DISCOUNT_THRESHOLD = 0.20   # >20% discount = excessive
HIGH_VALUE_ORDER_THRESHOLD   = 200    # orders above this are "high value"
PRICING_VARIANCE_THRESHOLD   = 0.50   # >50% price variance = anomaly
LATE_REFUND_RISK_RATE         = 0.08  # 8% of late orders assumed to request refund
REFUND_PARTIAL_RATE           = 0.40  # assume 40% partial refund of order value

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_data() -> tuple:
    """Load the cleaned tables from SQLite."""
    conn = sqlite3.connect(DB_PATH)

    orders = pd.read_sql("""
        SELECT
            f.*,
            c.state,
            c.rfm_segment,
            c.customer_segment,
            c.churn_flag,
            c.total_orders      AS customer_total_orders,
            c.monetary          AS customer_monetary,
            p.category_english,
            ch.channel_name,
            ch.cac
        FROM fact_orders_clean f
        LEFT JOIN dim_customers_clean c  ON f.customer_id  = c.customer_id
        LEFT JOIN dim_products_clean  p  ON f.product_id   = p.product_id
        LEFT JOIN dim_channels        ch ON f.channel_id   = ch.channel_id
    """, conn)

    conn.close()

    # Parse timestamps
    for col in ["order_purchase_ts", "order_delivered_ts", "estimated_delivery"]:
        orders[col] = pd.to_datetime(orders[col], errors="coerce")

    print(f"  Loaded {len(orders):,} orders")
    print(f"  Columns: {list(orders.columns)}\n")
    return orders


# ─────────────────────────────────────────────────────────────────────────────
# L1 — EXCESSIVE DISCOUNT USAGE
# ─────────────────────────────────────────────────────────────────────────────

def analyse_discount_leakage(orders: pd.DataFrame) -> dict:
    """
    Identifies orders where discounts were given unnecessarily or excessively.

    Leakage categories:
      A. Discount > 20% of unit price on delivered orders
      B. High-value orders (>R$200) that received any discount
         — these customers likely would have bought anyway
      C. VIP / champions customers given discounts
         — they are your most loyal, don't need incentivising
    """
    print("=" * 60)
    print("L1 — Excessive Discount Analysis")
    print("=" * 60)

    delivered = orders[orders["order_status"] == "delivered"].copy()

    # ── A. Excessive discount rate ────────────────────────────────────────────
    delivered["discount_pct"] = pd.to_numeric(
        delivered["discount_pct"], errors="coerce"
    ).fillna(0)

    excessive = delivered[
        delivered["discount_pct"] > (EXCESSIVE_DISCOUNT_THRESHOLD * 100)
    ].copy()

    leakage_a = excessive["discount_amount"].sum()
    print(f"\n  A. Orders with discount > {EXCESSIVE_DISCOUNT_THRESHOLD*100:.0f}%")
    print(f"     Orders affected:   {len(excessive):>8,}")
    print(f"     Total discount:    R$ {leakage_a:>12,.2f}")
    print(f"     Avg discount/order:R$ {excessive['discount_amount'].mean():>8,.2f}")

    # ── B. High-value orders that got a discount ──────────────────────────────
    high_value_discounted = delivered[
        (delivered["unit_price"] > HIGH_VALUE_ORDER_THRESHOLD) &
        (delivered["discount_amount"] > 0)
    ].copy()

    leakage_b = high_value_discounted["discount_amount"].sum()
    print(f"\n  B. High-value orders (>R${HIGH_VALUE_ORDER_THRESHOLD}) given discount")
    print(f"     Orders affected:   {len(high_value_discounted):>8,}")
    print(f"     Total discount:    R$ {leakage_b:>12,.2f}")

    # ── C. VIP / Champions given discounts ────────────────────────────────────
    loyal_discounted = delivered[
        (delivered["rfm_segment"].isin(["champions", "loyal"])) &
        (delivered["discount_amount"] > 0)
    ].copy()

    leakage_c = loyal_discounted["discount_amount"].sum()
    print(f"\n  C. Champions/Loyal customers given discounts")
    print(f"     Customers affected:{len(loyal_discounted['customer_id'].unique()):>8,}")
    print(f"     Total discount:    R$ {leakage_c:>12,.2f}")

    # ── Discount leakage by channel ───────────────────────────────────────────
    channel_discount = (
        delivered[delivered["discount_amount"] > 0]
        .groupby("channel_name")
        .agg(
            discounted_orders = ("order_id",        "count"),
            total_discount    = ("discount_amount", "sum"),
            avg_discount_pct  = ("discount_pct",    "mean"),
        )
        .round(2)
        .sort_values("total_discount", ascending=False)
    )
    print(f"\n  Discount leakage by channel:")
    print(channel_discount.to_string())

    # ── Save detail file ──────────────────────────────────────────────────────
    detail = pd.concat([excessive, high_value_discounted]).drop_duplicates("order_id")
    detail.to_csv(REPORTS / "leakage_L1_discounts.csv", index=False)

    total_l1 = leakage_a + leakage_b + leakage_c
    print(f"\n  TOTAL L1 leakage estimate: R$ {total_l1:,.2f}")

    return {
        "source":           "L1 — Excessive discounts",
        "leakage_amount":   round(total_l1, 2),
        "orders_affected":  len(excessive) + len(high_value_discounted),
        "detail_a":         round(leakage_a, 2),
        "detail_b":         round(leakage_b, 2),
        "detail_c":         round(leakage_c, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# L2 — CANCELLED ORDER REVENUE LOSS
# ─────────────────────────────────────────────────────────────────────────────

def analyse_cancellation_leakage(orders: pd.DataFrame) -> dict:
    """
    Quantifies revenue lost from cancelled and unavailable orders.

    We calculate:
      - Direct revenue loss (what the order was worth)
      - Which categories have the highest cancellation rates
      - Which channels have the most cancellations
      - Whether cancellations cluster by customer segment
    """
    print("\n" + "=" * 60)
    print("L2 — Cancellation Revenue Loss")
    print("=" * 60)

    cancelled = orders[
        orders["order_status"].isin(["canceled", "unavailable"])
    ].copy()

    leakage = cancelled["unit_price"].sum()

    print(f"\n  Cancelled/unavailable orders: {len(cancelled):,}")
    print(f"  Revenue lost:                 R$ {leakage:,.2f}")
    print(f"  Avg value of cancelled order: R$ {cancelled['unit_price'].mean():,.2f}")

    # Cancellation rate by category
    total_by_cat = (
        orders.groupby("category_english")["order_id"].count()
        .rename("total_orders")
    )
    cancelled_by_cat = (
        cancelled.groupby("category_english")
        .agg(
            cancelled_orders  = ("order_id",    "count"),
            revenue_lost      = ("unit_price",  "sum"),
        )
    )
    cat_analysis = cancelled_by_cat.join(total_by_cat)
    cat_analysis["cancellation_rate_pct"] = (
        cat_analysis["cancelled_orders"] /
        cat_analysis["total_orders"] * 100
    ).round(2)
    cat_analysis = cat_analysis.sort_values("revenue_lost", ascending=False)

    print(f"\n  Top 10 categories by cancelled revenue:")
    print(cat_analysis.head(10).to_string())

    # Cancellation by channel
    channel_cancel = (
        cancelled.groupby("channel_name")
        .agg(
            cancelled_orders = ("order_id",   "count"),
            revenue_lost     = ("unit_price", "sum"),
        )
        .sort_values("revenue_lost", ascending=False)
    )
    print(f"\n  Cancellations by channel:")
    print(channel_cancel.to_string())

    cancelled.to_csv(REPORTS / "leakage_L2_cancellations.csv", index=False)

    print(f"\n  TOTAL L2 leakage estimate: R$ {leakage:,.2f}")

    return {
        "source":          "L2 — Cancellation losses",
        "leakage_amount":  round(leakage, 2),
        "orders_affected": len(cancelled),
    }


# ─────────────────────────────────────────────────────────────────────────────
# L3 — PRICING ANOMALIES
# ─────────────────────────────────────────────────────────────────────────────

def analyse_pricing_anomalies(orders: pd.DataFrame) -> dict:
    """
    Detects products being sold at significantly different prices
    across different orders — a sign of pricing inconsistency,
    misconfigured discount rules, or seller fraud.

    Method:
      For each product, calculate:
        - median selling price
        - coefficient of variation (std / mean)
      Flag products where some orders sold below 50% of median price.
    """
    print("\n" + "=" * 60)
    print("L3 — Pricing Anomaly Analysis")
    print("=" * 60)

    delivered = orders[orders["order_status"] == "delivered"].copy()

    # Products sold in at least 5 orders (need volume to detect variance)
    product_stats = (
        delivered.groupby("product_id")
        .agg(
            order_count   = ("order_id",    "count"),
            mean_price    = ("unit_price",  "mean"),
            median_price  = ("unit_price",  "median"),
            std_price     = ("unit_price",  "std"),
            min_price     = ("unit_price",  "min"),
            max_price     = ("unit_price",  "max"),
            category      = ("category_english", "first"),
        )
        .query("order_count >= 5")
        .reset_index()
    )

    product_stats["cv"] = (
        product_stats["std_price"] /
        product_stats["mean_price"].replace(0, np.nan)
    ).fillna(0).round(4)

    product_stats["price_range_pct"] = (
        (product_stats["max_price"] - product_stats["min_price"]) /
        product_stats["median_price"].replace(0, np.nan) * 100
    ).fillna(0).round(2)

    # Anomalous products: price range > 50% of median
    anomalous = product_stats[
        product_stats["price_range_pct"] > (PRICING_VARIANCE_THRESHOLD * 100)
    ].copy()

    print(f"\n  Products with >50% price variance: {len(anomalous):,}")
    print(f"  (out of {len(product_stats):,} products with 5+ orders)")

    # Estimate leakage: for each anomalous product, sum up revenue
    # lost on orders priced below the median
    leakage_total = 0
    anomaly_details = []

    for _, prod in anomalous.iterrows():
        prod_orders = delivered[
            delivered["product_id"] == prod["product_id"]
        ].copy()
        median_p = prod["median_price"]

        # Orders where customer paid less than median
        underpriced = prod_orders[prod_orders["unit_price"] < median_p * 0.80]
        potential_lost = (
            (median_p - underpriced["unit_price"]) *
            underpriced["quantity"]
        ).sum()
        leakage_total += potential_lost

        anomaly_details.append({
            "product_id":        prod["product_id"],
            "category":          prod["category"],
            "order_count":       prod["order_count"],
            "median_price":      round(median_p, 2),
            "min_price":         round(prod["min_price"], 2),
            "max_price":         round(prod["max_price"], 2),
            "price_range_pct":   prod["price_range_pct"],
            "cv":                prod["cv"],
            "estimated_leakage": round(potential_lost, 2),
        })

    anomaly_df = pd.DataFrame(anomaly_details).sort_values(
        "estimated_leakage", ascending=False
    )

    print(f"\n  Top 10 products by pricing leakage:")
    print(anomaly_df.head(10).to_string(index=False))

    anomaly_df.to_csv(REPORTS / "leakage_L3_pricing.csv", index=False)

    print(f"\n  TOTAL L3 leakage estimate: R$ {leakage_total:,.2f}")

    return {
        "source":            "L3 — Pricing anomalies",
        "leakage_amount":    round(leakage_total, 2),
        "products_affected": len(anomalous),
    }


# ─────────────────────────────────────────────────────────────────────────────
# L4 — LATE DELIVERY IMPACT
# ─────────────────────────────────────────────────────────────────────────────

def analyse_late_delivery_leakage(orders: pd.DataFrame) -> dict:
    """
    Estimates revenue at risk from late deliveries.

    Two components:
      A. Refund risk: a % of late orders will request partial refunds
      B. Review damage: late orders get lower review scores,
         which reduces repeat purchase probability

    We also analyse whether certain categories or regions
    have disproportionately high late delivery rates.
    """
    print("\n" + "=" * 60)
    print("L4 — Late Delivery Impact")
    print("=" * 60)

    delivered = orders[orders["order_status"] == "delivered"].copy()

    # Ensure numeric
    delivered["is_late_delivery"] = pd.to_numeric(
        delivered["is_late_delivery"], errors="coerce"
    ).fillna(0)
    delivered["delivery_delay_days"] = pd.to_numeric(
        delivered["delivery_delay_days"], errors="coerce"
    )

    late = delivered[delivered["is_late_delivery"] == 1].copy()
    on_time = delivered[delivered["is_late_delivery"] == 0].copy()

    late_rate = len(late) / len(delivered) * 100

    print(f"\n  Total delivered orders:  {len(delivered):>8,}")
    print(f"  Late deliveries:         {len(late):>8,}  ({late_rate:.1f}%)")
    print(f"  Avg delay (late orders): {late['delivery_delay_days'].mean():>8.1f} days")
    print(f"  Max delay:               {late['delivery_delay_days'].max():>8.0f} days")

    # ── A. Refund risk estimate ───────────────────────────────────────────────
    refund_risk_orders = len(late) * LATE_REFUND_RISK_RATE
    avg_order_value    = late["revenue"].mean()
    refund_leakage     = (
        refund_risk_orders *
        avg_order_value *
        REFUND_PARTIAL_RATE
    )

    print(f"\n  A. Estimated refund risk from late deliveries")
    print(f"     Orders likely to claim: {refund_risk_orders:>8,.0f}")
    print(f"     Avg order value:        R$ {avg_order_value:>8,.2f}")
    print(f"     Estimated leakage:      R$ {refund_leakage:>8,.2f}")

    # ── B. Review score impact ────────────────────────────────────────────────
    late["review_score"]    = pd.to_numeric(late["review_score"],    errors="coerce")
    on_time["review_score"] = pd.to_numeric(on_time["review_score"], errors="coerce")

    late_review    = late[late["review_score"] > 0]["review_score"].mean()
    on_time_review = on_time[on_time["review_score"] > 0]["review_score"].mean()

    print(f"\n  B. Review score comparison")
    print(f"     Avg review — on-time:  {on_time_review:.2f} / 5.0")
    print(f"     Avg review — late:     {late_review:.2f} / 5.0")
    print(f"     Review drop:           {on_time_review - late_review:.2f} points")

    # ── Late delivery by state ────────────────────────────────────────────────
    late_by_state = (
        delivered.groupby("state")
        .agg(
            total_orders    = ("order_id",          "count"),
            late_orders     = ("is_late_delivery",  "sum"),
            avg_delay_days  = ("delivery_delay_days","mean"),
            revenue_at_risk = ("revenue",           "sum"),
        )
        .reset_index()
    )
    late_by_state["late_rate_pct"] = (
        late_by_state["late_orders"] /
        late_by_state["total_orders"] * 100
    ).round(2)
    late_by_state = late_by_state.sort_values("late_rate_pct", ascending=False)

    print(f"\n  Top 10 states by late delivery rate:")
    print(late_by_state.head(10)[
        ["state", "total_orders", "late_orders",
         "late_rate_pct", "avg_delay_days"]
    ].to_string(index=False))

    late.to_csv(REPORTS / "leakage_L4_late_delivery.csv", index=False)

    print(f"\n  TOTAL L4 leakage estimate: R$ {refund_leakage:,.2f}")

    return {
        "source":           "L4 — Late delivery refund risk",
        "leakage_amount":   round(refund_leakage, 2),
        "orders_affected":  len(late),
        "late_rate_pct":    round(late_rate, 2),
        "avg_delay_days":   round(late["delivery_delay_days"].mean(), 1),
        "review_drop":      round(on_time_review - late_review, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# L5 — FREIGHT BILLING MISMATCHES
# ─────────────────────────────────────────────────────────────────────────────

def analyse_freight_leakage(orders: pd.DataFrame) -> dict:
    """
    Detects freight billing anomalies.

    Cases we look for:
      A. Orders with zero freight where freight is expected
         (products are heavy/large — freight should not be zero)
      B. Freight cost is disproportionately high vs order value
         (freight > 30% of order value = potential misbilling)
      C. Products in same category with very different freight costs
    """
    print("\n" + "=" * 60)
    print("L5 — Freight Billing Mismatch Analysis")
    print("=" * 60)

    delivered = orders[orders["order_status"] == "delivered"].copy()
    delivered["weight_g"] = pd.to_numeric(
        delivered.get("weight_g", pd.Series([np.nan]*len(delivered))),
        errors="coerce"
    ).fillna(0)

    # ── A. Zero freight on heavy items ────────────────────────────────────────
    zero_freight_heavy = delivered[
        (delivered["freight_value"] == 0) &
        (delivered["weight_g"] > 5000)       # heavier than 5kg
    ].copy()

    # Estimate expected freight using median freight for similar-weight orders
    median_freight = delivered[
        (delivered["freight_value"] > 0) &
        (delivered["weight_g"] > 5000)
    ]["freight_value"].median()

    leakage_a = len(zero_freight_heavy) * (median_freight if not np.isnan(median_freight) else 0)

    print(f"\n  A. Heavy items (>5kg) with zero freight")
    print(f"     Orders affected:        {len(zero_freight_heavy):>8,}")
    print(f"     Median expected freight:R$ {median_freight if not np.isnan(median_freight) else 0:>8,.2f}")
    print(f"     Estimated leakage:      R$ {leakage_a:>8,.2f}")

    # ── B. Freight > 30% of order value ──────────────────────────────────────
    delivered["freight_pct"] = np.where(
        delivered["unit_price"] > 0,
        delivered["freight_value"] / delivered["unit_price"] * 100,
        0
    )
    high_freight = delivered[delivered["freight_pct"] > 30].copy()
    excess_freight = (
        high_freight["freight_value"] -
        (high_freight["unit_price"] * 0.30)
    ).sum()

    print(f"\n  B. Orders where freight > 30% of order value")
    print(f"     Orders affected:        {len(high_freight):>8,}")
    print(f"     Excess freight charged: R$ {excess_freight:>8,.2f}")

    # ── Freight by category ───────────────────────────────────────────────────
    freight_cat = (
        delivered.groupby("category_english")
        .agg(
            avg_freight       = ("freight_value", "mean"),
            avg_order_value   = ("unit_price",    "mean"),
            total_freight     = ("freight_value", "sum"),
        )
        .round(2)
        .sort_values("avg_freight", ascending=False)
        .head(15)
    )
    print(f"\n  Top 15 categories by avg freight cost:")
    print(freight_cat.to_string())

    high_freight.to_csv(REPORTS / "leakage_L5_freight.csv", index=False)

    total_l5 = leakage_a + max(excess_freight, 0)
    print(f"\n  TOTAL L5 leakage estimate: R$ {total_l5:,.2f}")

    return {
        "source":          "L5 — Freight billing mismatches",
        "leakage_amount":  round(total_l5, 2),
        "orders_affected": len(zero_freight_heavy) + len(high_freight),
    }


# ─────────────────────────────────────────────────────────────────────────────
# L6 — HIGH-VALUE CUSTOMER DISCOUNT ABUSE
# ─────────────────────────────────────────────────────────────────────────────

def analyse_clv_discount_leakage(orders: pd.DataFrame) -> dict:
    """
    Identifies cases where your most valuable customers
    are receiving discounts they don't need.

    The insight: a Champions customer with R$2,000 LTV
    doesn't need a 25% discount to complete a R$50 purchase.
    This is pure margin erosion.
    """
    print("\n" + "=" * 60)
    print("L6 — High-Value Customer Discount Abuse")
    print("=" * 60)

    delivered = orders[orders["order_status"] == "delivered"].copy()
    delivered["customer_monetary"] = pd.to_numeric(
        delivered["customer_monetary"], errors="coerce"
    ).fillna(0)

    # High-LTV customers = top 20% by lifetime monetary value
    ltv_threshold = delivered["customer_monetary"].quantile(0.80)

    high_ltv_discounted = delivered[
        (delivered["customer_monetary"] >= ltv_threshold) &
        (delivered["discount_amount"] > 0)
    ].copy()

    leakage = high_ltv_discounted["discount_amount"].sum()

    print(f"\n  LTV threshold (top 20%):  R$ {ltv_threshold:,.2f}")
    print(f"  High-LTV customers:        {delivered[delivered['customer_monetary'] >= ltv_threshold]['customer_id'].nunique():,}")
    print(f"  High-LTV orders discounted:{len(high_ltv_discounted):,}")
    print(f"  Total unnecessary discount:R$ {leakage:,.2f}")

    # Breakdown by rfm_segment
    seg_breakdown = (
        high_ltv_discounted.groupby("rfm_segment")
        .agg(
            customers        = ("customer_id",      "nunique"),
            discounted_orders= ("order_id",         "count"),
            total_discount   = ("discount_amount",  "sum"),
            avg_discount_pct = ("discount_pct",     "mean"),
        )
        .round(2)
        .sort_values("total_discount", ascending=False)
    )
    print(f"\n  Breakdown by RFM segment:")
    print(seg_breakdown.to_string())

    high_ltv_discounted.to_csv(REPORTS / "leakage_L6_clv_discounts.csv", index=False)

    print(f"\n  TOTAL L6 leakage estimate: R$ {leakage:,.2f}")

    return {
        "source":          "L6 — High-LTV customer discounts",
        "leakage_amount":  round(leakage, 2),
        "orders_affected": len(high_ltv_discounted),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY AND VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def build_leakage_summary(results: list, orders: pd.DataFrame) -> pd.DataFrame:
    """
    Combines all leakage sources into a single summary DataFrame.
    Calculates each source as % of total revenue.
    """
    delivered = orders[orders["order_status"] == "delivered"]
    total_revenue = delivered["revenue"].sum()

    summary = pd.DataFrame(results)
    summary["pct_of_revenue"] = (
        summary["leakage_amount"] / total_revenue * 100
    ).round(3)
    summary["total_revenue"] = round(total_revenue, 2)

    total_leakage = summary["leakage_amount"].sum()
    total_pct     = total_leakage / total_revenue * 100

    print("\n" + "=" * 60)
    print("  LEAKAGE SUMMARY")
    print("=" * 60)
    print(f"\n  Total revenue (delivered): R$ {total_revenue:>14,.2f}")
    print(f"  Total identified leakage:  R$ {total_leakage:>14,.2f}")
    print(f"  Leakage as % of revenue:       {total_pct:>10.2f}%\n")

    for _, row in summary.iterrows():
        bar = "█" * int(row["pct_of_revenue"] * 4)
        print(f"  {row['source']:<38}"
              f"  R$ {row['leakage_amount']:>10,.0f}"
              f"  ({row['pct_of_revenue']:.2f}%)  {bar}")

    return summary, total_revenue, total_leakage


def plot_leakage(summary: pd.DataFrame, total_revenue: float,
                 total_leakage: float) -> None:
    """
    Creates two publication-quality charts:
      1. Waterfall chart — revenue breakdown
      2. Bar chart — leakage by source
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor("#FAFAFA")

    colors = {
        "L1 — Excessive discounts":         "#E24B4A",
        "L2 — Cancellation losses":         "#D85A30",
        "L3 — Pricing anomalies":           "#BA7517",
        "L4 — Late delivery refund risk":   "#7F77DD",
        "L5 — Freight billing mismatches":  "#1D9E75",
        "L6 — High-LTV customer discounts": "#D4537E",
    }

    # ── Chart 1: Horizontal bar — leakage by source ───────────────────────────
    ax1 = axes[0]
    ax1.set_facecolor("#FAFAFA")

    sorted_df = summary.sort_values("leakage_amount", ascending=True)
    bar_colors = [colors.get(s, "#888780") for s in sorted_df["source"]]

    bars = ax1.barh(
        range(len(sorted_df)),
        sorted_df["leakage_amount"],
        color=bar_colors,
        height=0.6,
        edgecolor="none"
    )

    ax1.set_yticks(range(len(sorted_df)))
    ax1.set_yticklabels(
        [s.replace(" — ", "\n") for s in sorted_df["source"]],
        fontsize=9
    )
    ax1.set_xlabel("Revenue Leakage (R$)", fontsize=10)
    ax1.set_title("Revenue Leakage by Source", fontsize=13,
                  fontweight="bold", pad=15)
    ax1.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"R${x:,.0f}")
    )
    ax1.spines[["top", "right", "left"]].set_visible(False)
    ax1.tick_params(axis="y", length=0)

    # Add value labels
    for bar, (_, row) in zip(bars, sorted_df.iterrows()):
        ax1.text(
            bar.get_width() + total_leakage * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"R${row['leakage_amount']:,.0f}\n({row['pct_of_revenue']:.2f}%)",
            va="center", fontsize=8, color="#444441"
        )

    # ── Chart 2: Pie — proportion of total revenue ───────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("#FAFAFA")

    recovered_revenue = total_revenue - total_leakage
    pie_values = list(summary["leakage_amount"]) + [recovered_revenue]
    pie_labels = list(summary["source"].str.split(" — ").str[0]) + ["Retained revenue"]
    pie_colors = list(colors.values()) + ["#1D9E75"]

    wedges, texts, autotexts = ax2.pie(
        pie_values,
        labels=None,
        colors=pie_colors,
        autopct=lambda p: f"{p:.1f}%" if p > 1 else "",
        startangle=140,
        pctdistance=0.75,
        wedgeprops={"edgecolor": "white", "linewidth": 2}
    )
    for t in autotexts:
        t.set_fontsize(8)
        t.set_color("white")
        t.set_fontweight("bold")

    ax2.set_title(
        f"Revenue Composition\nTotal: R${total_revenue:,.0f}",
        fontsize=13, fontweight="bold", pad=15
    )
    ax2.legend(
        wedges, pie_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2, fontsize=8,
        frameon=False
    )

    plt.tight_layout()
    out = REPORTS / "leakage_analysis.png"
    plt.savefig(out, dpi=150, bbox_inches="tight",
                facecolor="#FAFAFA")
    plt.close()
    print(f"\n  Chart saved → {out}")


def write_executive_report(summary: pd.DataFrame,
                            total_revenue: float,
                            total_leakage: float) -> None:
    """
    Writes a plain-text executive summary report.
    This is what you include in your portfolio presentation.
    """
    report = f"""
REVENUE LEAKAGE DIAGNOSTIC REPORT
E-Commerce Analytics — Phase 5
{'='*60}

EXECUTIVE SUMMARY
─────────────────────────────────────────────────────────
Total delivered revenue:     R$ {total_revenue:>14,.2f}
Total identified leakage:    R$ {total_leakage:>14,.2f}
Leakage as % of revenue:         {total_leakage/total_revenue*100:>10.2f}%
Potential annual recovery:   R$ {total_leakage:>14,.2f}

LEAKAGE BY SOURCE
─────────────────────────────────────────────────────────
"""
    for _, row in summary.sort_values("leakage_amount", ascending=False).iterrows():
        report += (
            f"  {row['source']:<40}"
            f"  R$ {row['leakage_amount']:>10,.2f}"
            f"  ({row['pct_of_revenue']:.3f}% of revenue)\n"
        )

    report += f"""
RECOMMENDATIONS
─────────────────────────────────────────────────────────
1. DISCOUNTS  — Implement discount eligibility rules. Block discounts
               on orders >R${HIGH_VALUE_ORDER_THRESHOLD} and for Champions/Loyal segments.
               Estimated saving: R$ {summary[summary['source'].str.startswith('L1')]['leakage_amount'].sum():,.0f}

2. CANCELLATIONS — Investigate top cancellation categories.
               Consider stock availability improvements and
               seller performance SLAs.
               Estimated saving: R$ {summary[summary['source'].str.startswith('L2')]['leakage_amount'].sum():,.0f}

3. PRICING     — Standardise product pricing rules for the {summary[summary['source'].str.startswith('L3')]['products_affected'].sum() if 'products_affected' in summary.columns else 'N'} anomalous products.
               Implement min/max price guardrails per category.

4. DELIVERY    — Late delivery rate is significant. Partner with
               logistics providers in high-delay states.
               Proactive customer communication can reduce refund requests.

5. FREIGHT     — Audit zero-freight orders on heavy products.
               Implement weight-based freight floor pricing.

NOTE: Leakage estimates are conservative. Actual recoverable
revenue may be higher with targeted interventions.
"""
    out = REPORTS / "leakage_report.txt"
    out.write_text(report, encoding="utf-8")
    print(f"  Report saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  E-COMMERCE ANALYTICS — PHASE 5 REVENUE LEAKAGE")
    print("=" * 60 + "\n")

    orders = load_data()

    results = []
    results.append(analyse_discount_leakage(orders))
    results.append(analyse_cancellation_leakage(orders))
    results.append(analyse_pricing_anomalies(orders))
    results.append(analyse_late_delivery_leakage(orders))
    results.append(analyse_freight_leakage(orders))
    results.append(analyse_clv_discount_leakage(orders))

    summary, total_revenue, total_leakage = build_leakage_summary(
        results, orders
    )

    # Save summary CSV
    summary.to_csv(REPORTS / "leakage_summary.csv", index=False)
    print(f"\n  Summary saved → reports/leakage_summary.csv")

    plot_leakage(summary, total_revenue, total_leakage)
    write_executive_report(summary, total_revenue, total_leakage)

    print("\n  Phase 5 complete. Ready for Phase 6 — Cohort Analysis.\n")


if __name__ == "__main__":
    main()