"""
Phase 7 — Customer Lifetime Value Model
==========================================
Calculates CLV at three levels:
  1. Overall business CLV
  2. CLV by RFM segment
  3. CLV by acquisition channel (with LTV:CAC ratio)

Also computes:
  - Payback period per channel
  - CLV distribution across customer base
  - Top 20% vs bottom 80% revenue contribution (Pareto)
  - CLV-based customer tiering

Outputs:
  data/processed/clv_customers.csv     — CLV for every customer
  data/processed/clv_by_segment.csv    — CLV aggregated by segment
  data/processed/clv_by_channel.csv    — CLV + LTV:CAC by channel
  reports/clv_report.txt               — executive summary
  reports/clv_*.png                    — 4 charts

Run from project root:
    python notebooks/06_clv_model.py
"""

import warnings
import sqlite3
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

DB_PATH   = Path("data/ecommerce.db")
PROCESSED = Path("data/processed")
REPORTS   = Path("reports")
REPORTS.mkdir(parents=True, exist_ok=True)

GROSS_MARGIN_RATE  = 0.40   # 40% gross margin assumption
DATASET_SPAN_YEARS = 2.0    # Olist data spans ~2 years


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """
    Load delivered orders joined with customer, channel,
    and product info. We use customer_unique_id so repeat
    buyers across multiple customer_ids are merged correctly.
    """
    conn = sqlite3.connect(DB_PATH)

    df = pd.read_sql("""
        SELECT
            f.order_id,
            c.customer_unique_id,
            f.customer_id,
            f.order_purchase_ts,
            f.revenue,
            f.gross_profit,
            f.discount_amount,
            f.discount_pct,
            f.order_status,
            f.channel_id,
            ch.channel_name,
            ch.cac,
            c.rfm_segment,
            c.customer_segment,
            c.state,
            p.category_english
        FROM fact_orders_clean    f
        JOIN dim_customers_clean  c  ON f.customer_id  = c.customer_id
        JOIN dim_channels         ch ON f.channel_id   = ch.channel_id
        LEFT JOIN dim_products_clean p ON f.product_id = p.product_id
        WHERE f.order_status = 'delivered'
    """, conn)

    conn.close()

    df["order_purchase_ts"] = pd.to_datetime(
        df["order_purchase_ts"], errors="coerce"
    )
    df = df.dropna(subset=["order_purchase_ts"])

    print(f"  Loaded {len(df):,} delivered orders")
    print(f"  Unique customers: {df['customer_unique_id'].nunique():,}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — COMPUTE CUSTOMER-LEVEL CLV COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────

def compute_customer_clv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes CLV components for every unique customer.

    Formula:
        CLV = AOV × Purchase_Frequency × Customer_Lifespan × Gross_Margin

    Where:
        AOV                = average revenue per order
        Purchase_Frequency = total orders / dataset span in years
        Customer_Lifespan  = estimated years the customer remains active
                             (derived from recency and frequency signals)
        Gross_Margin       = 40% assumption

    Customer lifespan estimation:
        We use a simple but realistic heuristic:
        - 1-order customers: 1 year lifespan (they may never return)
        - 2-order customers: 1.5 years
        - 3-5 orders:        2.5 years
        - 6+ orders:         4.0 years
        This avoids needing a full BG/NBD probabilistic model
        while still being defensible in a business context.
    """

    # Aggregate per unique customer
    customer_stats = (
        df.groupby("customer_unique_id")
        .agg(
            total_orders       = ("order_id",           "count"),
            total_revenue      = ("revenue",            "sum"),
            total_gross_profit = ("gross_profit",       "sum"),
            avg_order_value    = ("revenue",            "mean"),
            avg_discount_pct   = ("discount_pct",       "mean"),
            total_discounts    = ("discount_amount",    "sum"),
            first_order        = ("order_purchase_ts",  "min"),
            last_order         = ("order_purchase_ts",  "max"),
            rfm_segment        = ("rfm_segment",        "first"),
            customer_segment   = ("customer_segment",   "first"),
            state              = ("state",              "first"),
            channel_id         = ("channel_id",         "first"),
            channel_name       = ("channel_name",       "first"),
            cac                = ("cac",                "first"),
        )
        .reset_index()
    )

    # ── AOV ───────────────────────────────────────────────────────────────────
    customer_stats["aov"] = (
        customer_stats["total_revenue"] /
        customer_stats["total_orders"]
    ).round(2)

    # ── Purchase Frequency (orders per year) ──────────────────────────────────
    customer_stats["purchase_frequency"] = (
        customer_stats["total_orders"] / DATASET_SPAN_YEARS
    ).round(4)

    # ── Customer Lifespan (years) ─────────────────────────────────────────────
    def estimate_lifespan(n_orders: int) -> float:
        if n_orders == 1:   return 1.0
        elif n_orders == 2: return 1.5
        elif n_orders <= 5: return 2.5
        else:               return 4.0

    customer_stats["customer_lifespan"] = (
        customer_stats["total_orders"]
        .apply(estimate_lifespan)
    )

    # ── CLV ───────────────────────────────────────────────────────────────────
    customer_stats["clv"] = (
        customer_stats["aov"] *
        customer_stats["purchase_frequency"] *
        customer_stats["customer_lifespan"] *
        GROSS_MARGIN_RATE
    ).round(2)

    # ── Active months (actual span in data) ───────────────────────────────────
    customer_stats["active_days"] = (
        customer_stats["last_order"] -
        customer_stats["first_order"]
    ).dt.days.fillna(0).astype(int)

    # ── LTV:CAC per customer ──────────────────────────────────────────────────
    customer_stats["ltv_cac_ratio"] = np.where(
        customer_stats["cac"] > 0,
        (customer_stats["clv"] / customer_stats["cac"]).round(2),
        np.nan
    )

    # ── CLV tier ─────────────────────────────────────────────────────────────
    clv_percentiles = customer_stats["clv"].quantile([0.25, 0.5, 0.75, 0.90])

    def clv_tier(clv: float) -> str:
        if clv >= clv_percentiles[0.90]: return "platinum"
        elif clv >= clv_percentiles[0.75]: return "gold"
        elif clv >= clv_percentiles[0.50]: return "silver"
        elif clv >= clv_percentiles[0.25]: return "bronze"
        else: return "low"

    customer_stats["clv_tier"] = customer_stats["clv"].apply(clv_tier)

    print(f"\n  CLV computed for {len(customer_stats):,} customers")
    print(f"  Avg CLV:      R$ {customer_stats['clv'].mean():,.2f}")
    print(f"  Median CLV:   R$ {customer_stats['clv'].median():,.2f}")
    print(f"  Max CLV:      R$ {customer_stats['clv'].max():,.2f}")

    return customer_stats


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — CLV BY RFM SEGMENT
# ─────────────────────────────────────────────────────────────────────────────

def clv_by_segment(customer_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates CLV metrics by RFM segment.
    This tells you which segments are most valuable
    and where to focus retention investment.
    """

    seg = (
        customer_stats.groupby("rfm_segment")
        .agg(
            customer_count    = ("customer_unique_id", "count"),
            avg_clv           = ("clv",                "mean"),
            median_clv        = ("clv",                "median"),
            total_clv         = ("clv",                "sum"),
            avg_aov           = ("aov",                "mean"),
            avg_frequency     = ("purchase_frequency", "mean"),
            avg_lifespan      = ("customer_lifespan",  "mean"),
            avg_orders        = ("total_orders",       "mean"),
            avg_revenue       = ("total_revenue",      "mean"),
            avg_discount_pct  = ("avg_discount_pct",   "mean"),
        )
        .reset_index()
        .round(2)
        .sort_values("avg_clv", ascending=False)
    )

    seg["clv_share_pct"] = (
        seg["total_clv"] / seg["total_clv"].sum() * 100
    ).round(2)

    print("\n" + "=" * 60)
    print("  CLV BY RFM SEGMENT")
    print("=" * 60)
    print(seg[[
        "rfm_segment", "customer_count", "avg_clv",
        "avg_aov", "avg_orders", "clv_share_pct"
    ]].to_string(index=False))

    return seg


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — CLV BY ACQUISITION CHANNEL (LTV:CAC)
# ─────────────────────────────────────────────────────────────────────────────

def clv_by_channel(customer_stats: pd.DataFrame) -> pd.DataFrame:
    """
    The most important output of CLV analysis for a business:
    LTV:CAC ratio by acquisition channel.

    Interpretation:
      LTV:CAC > 3   = healthy, invest more in this channel
      LTV:CAC 1–3   = marginal, optimise or monitor
      LTV:CAC < 1   = losing money on every customer acquired

    Payback period = CAC / (CLV / customer_lifespan_months)
    = how many months until you recover the acquisition cost
    """

    ch = (
        customer_stats.groupby(["channel_id", "channel_name"])
        .agg(
            customer_count   = ("customer_unique_id", "count"),
            avg_clv          = ("clv",                "mean"),
            total_clv        = ("clv",                "sum"),
            avg_aov          = ("aov",                "mean"),
            avg_orders       = ("total_orders",       "mean"),
            avg_frequency    = ("purchase_frequency", "mean"),
            avg_lifespan     = ("customer_lifespan",  "mean"),
            avg_revenue      = ("total_revenue",      "mean"),
            cac              = ("cac",                "first"),
        )
        .reset_index()
        .round(2)
    )

    # LTV:CAC ratio
    ch["ltv_cac_ratio"] = np.where(
        ch["cac"] > 0,
        (ch["avg_clv"] / ch["cac"]).round(2),
        np.nan
    )

    # Total acquisition cost
    ch["total_acquisition_cost"] = (
        ch["cac"] * ch["customer_count"]
    ).round(2)

    # Payback period in months
    # Monthly CLV = CLV / (lifespan * 12)
    ch["monthly_clv"] = (
        ch["avg_clv"] /
        (ch["avg_lifespan"] * 12)
    ).round(4)

    ch["payback_months"] = np.where(
        ch["monthly_clv"] > 0,
        (ch["cac"] / ch["monthly_clv"]).round(1),
        np.nan
    )

    # ROI = (total_clv - total_acquisition_cost) / total_acquisition_cost
    ch["channel_roi_pct"] = np.where(
        ch["total_acquisition_cost"] > 0,
        ((ch["total_clv"] - ch["total_acquisition_cost"]) /
         ch["total_acquisition_cost"] * 100).round(1),
        np.nan
    )

    ch["health"] = ch["ltv_cac_ratio"].apply(
        lambda x: "invest" if x >= 3
        else ("monitor" if x >= 1 else "review")
        if not pd.isna(x) else "n/a"
    )

    ch = ch.sort_values("ltv_cac_ratio", ascending=False)

    print("\n" + "=" * 60)
    print("  CLV BY CHANNEL — LTV:CAC ANALYSIS")
    print("=" * 60)
    print(ch[[
        "channel_name", "customer_count", "avg_clv",
        "cac", "ltv_cac_ratio", "payback_months",
        "channel_roi_pct", "health"
    ]].to_string(index=False))

    return ch


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — PARETO ANALYSIS (80/20 rule)
# ─────────────────────────────────────────────────────────────────────────────

def pareto_analysis(customer_stats: pd.DataFrame) -> dict:
    """
    Tests the Pareto principle: do 20% of customers generate 80% of revenue?
    In most e-commerce businesses this is closer to 5% generating 50%.
    Knowing the actual split helps focus retention investment.
    """

    sorted_customers = customer_stats.sort_values(
        "total_revenue", ascending=False
    ).copy()

    sorted_customers["cumulative_revenue"] = (
        sorted_customers["total_revenue"].cumsum()
    )
    total_rev = sorted_customers["total_revenue"].sum()
    sorted_customers["cumulative_pct"] = (
        sorted_customers["cumulative_revenue"] / total_rev * 100
    )
    sorted_customers["customer_rank_pct"] = (
        np.arange(1, len(sorted_customers) + 1) /
        len(sorted_customers) * 100
    )

    # Find what % of customers generate 80% of revenue
    top_80_pct = sorted_customers[
        sorted_customers["cumulative_pct"] <= 80
    ]

    pct_customers_for_80_rev = (
        len(top_80_pct) / len(sorted_customers) * 100
    )

    # Find what % of revenue top 20% of customers generate
    top_20_customers = sorted_customers.head(
        int(len(sorted_customers) * 0.20)
    )
    top_20_revenue_pct = (
        top_20_customers["total_revenue"].sum() / total_rev * 100
    )

    # Top 5% revenue contribution
    top_5_customers = sorted_customers.head(
        int(len(sorted_customers) * 0.05)
    )
    top_5_revenue_pct = (
        top_5_customers["total_revenue"].sum() / total_rev * 100
    )

    print("\n" + "=" * 60)
    print("  PARETO ANALYSIS — REVENUE CONCENTRATION")
    print("=" * 60)
    print(f"  Top  5% of customers generate: {top_5_revenue_pct:.1f}% of revenue")
    print(f"  Top 20% of customers generate: {top_20_revenue_pct:.1f}% of revenue")
    print(f"  {pct_customers_for_80_rev:.1f}% of customers generate 80% of revenue")

    return {
        "sorted_customers":           sorted_customers,
        "pct_customers_for_80_rev":   round(pct_customers_for_80_rev, 1),
        "top_20_revenue_pct":         round(top_20_revenue_pct, 1),
        "top_5_revenue_pct":          round(top_5_revenue_pct, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — CHARTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_clv_by_segment(seg_df: pd.DataFrame) -> None:
    """Bar chart of avg CLV by RFM segment."""

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor("#FAFAFA")

    segment_colors = {
        "champions":      "#534AB7",
        "loyal":          "#7F77DD",
        "potential_loyal":"#AFA9EC",
        "need_attention": "#EF9F27",
        "at_risk":        "#D85A30",
        "lost":           "#E24B4A",
        "unknown":        "#888780",
    }

    # ── Chart 1: Avg CLV by segment ───────────────────────────────────────────
    ax1 = axes[0]
    ax1.set_facecolor("#FAFAFA")
    sorted_seg = seg_df.sort_values("avg_clv", ascending=True)
    colors = [segment_colors.get(s, "#888780") for s in sorted_seg["rfm_segment"]]

    bars = ax1.barh(
        sorted_seg["rfm_segment"],
        sorted_seg["avg_clv"],
        color=colors,
        height=0.6,
        edgecolor="none"
    )
    for bar, val in zip(bars, sorted_seg["avg_clv"]):
        ax1.text(
            bar.get_width() + sorted_seg["avg_clv"].max() * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"R${val:,.0f}",
            va="center", fontsize=9, color="#444441"
        )
    ax1.set_xlabel("Average CLV (R$)", fontsize=10)
    ax1.set_title("Avg CLV by RFM Segment", fontsize=12,
                  fontweight="bold", pad=12, color="#2C2C2A")
    ax1.spines[["top", "right", "left"]].set_visible(False)
    ax1.tick_params(axis="y", length=0)
    ax1.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"R${x:,.0f}")
    )

    # ── Chart 2: CLV share by segment (stacked 100% bar) ─────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("#FAFAFA")
    seg_sorted = seg_df.sort_values("clv_share_pct", ascending=False)

    wedge_colors = [segment_colors.get(s, "#888780")
                    for s in seg_sorted["rfm_segment"]]
    wedges, texts, autotexts = ax2.pie(
        seg_sorted["clv_share_pct"],
        labels=seg_sorted["rfm_segment"],
        colors=wedge_colors,
        autopct=lambda p: f"{p:.1f}%" if p > 3 else "",
        startangle=140,
        pctdistance=0.78,
        wedgeprops={"edgecolor": "white", "linewidth": 2}
    )
    for t in autotexts:
        t.set_fontsize(8)
        t.set_color("white")
        t.set_fontweight("bold")
    for t in texts:
        t.set_fontsize(9)
    ax2.set_title("CLV Share by Segment", fontsize=12,
                  fontweight="bold", pad=12, color="#2C2C2A")

    plt.tight_layout()
    out = REPORTS / "clv_by_segment.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#FAFAFA")
    plt.close()
    print(f"  Chart saved → {out}")


def plot_ltv_cac(ch_df: pd.DataFrame) -> None:
    """
    LTV:CAC ratio bar chart with health zone shading.
    The 3x line is the industry benchmark.
    """

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")

    # Health zone shading
    xmax = len(ch_df) + 0.5
    ax.axhspan(0, 1,   color="#FCEBEB", alpha=0.6, zorder=0)
    ax.axhspan(1, 3,   color="#FAEEDA", alpha=0.4, zorder=0)
    ax.axhspan(3, 100, color="#EAF3DE", alpha=0.4, zorder=0)

    ax.axhline(y=3, color="#3B6D11", linewidth=1.5,
               linestyle="--", label="Healthy threshold (3x)", zorder=2)
    ax.axhline(y=1, color="#A32D2D", linewidth=1.5,
               linestyle="--", label="Break-even (1x)", zorder=2)

    ch_valid = ch_df.dropna(subset=["ltv_cac_ratio"]).copy()

    bar_colors = ch_valid["ltv_cac_ratio"].apply(
        lambda x: "#639922" if x >= 3
        else ("#EF9F27" if x >= 1 else "#E24B4A")
    )
    bars = ax.bar(
        ch_valid["channel_name"],
        ch_valid["ltv_cac_ratio"],
        color=bar_colors,
        width=0.6,
        edgecolor="none",
        zorder=3
    )

    for bar, (_, row) in zip(bars, ch_valid.iterrows()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{row['ltv_cac_ratio']:.1f}x\n(R${row['avg_clv']:,.0f} CLV)",
            ha="center", va="bottom",
            fontsize=9, color="#444441"
        )

    ax.set_xlabel("Acquisition Channel", fontsize=11)
    ax.set_ylabel("LTV:CAC Ratio", fontsize=11)
    ax.set_title(
        "LTV:CAC Ratio by Acquisition Channel\n"
        "Green = healthy (>3x)  |  Amber = monitor (1-3x)  |  Red = review (<1x)",
        fontsize=12, fontweight="bold", pad=15, color="#2C2C2A"
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9, frameon=False)
    ax.set_ylim(0, max(ch_valid["ltv_cac_ratio"].max() * 1.3, 5))

    # Zone labels
    ax.text(xmax - 0.1, 0.5, "Losing money",
            ha="right", va="center", fontsize=8, color="#A32D2D", alpha=0.8)
    ax.text(xmax - 0.1, 2.0, "Marginal",
            ha="right", va="center", fontsize=8, color="#854F0B", alpha=0.8)
    ax.text(xmax - 0.1, 4.5, "Healthy",
            ha="right", va="center", fontsize=8, color="#3B6D11", alpha=0.8)

    plt.tight_layout()
    out = REPORTS / "clv_ltv_cac.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#FAFAFA")
    plt.close()
    print(f"  Chart saved → {out}")


def plot_clv_distribution(customer_stats: pd.DataFrame) -> None:
    """
    Histogram of CLV distribution across all customers.
    Shows that CLV is heavily right-skewed — most customers
    have low CLV but a small tail drives most value.
    """

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor("#FAFAFA")

    # ── Chart 1: CLV histogram ────────────────────────────────────────────────
    ax1 = axes[0]
    ax1.set_facecolor("#FAFAFA")

    clv_cap = customer_stats["clv"].quantile(0.95)
    clv_data = customer_stats[customer_stats["clv"] <= clv_cap]["clv"]

    ax1.hist(
        clv_data,
        bins=50,
        color="#7F77DD",
        edgecolor="none",
        alpha=0.85
    )
    ax1.axvline(
        customer_stats["clv"].mean(),
        color="#D85A30", linewidth=2,
        linestyle="--",
        label=f"Mean CLV: R${customer_stats['clv'].mean():,.0f}"
    )
    ax1.axvline(
        customer_stats["clv"].median(),
        color="#1D9E75", linewidth=2,
        linestyle="--",
        label=f"Median CLV: R${customer_stats['clv'].median():,.0f}"
    )
    ax1.set_xlabel("Customer Lifetime Value (R$)", fontsize=10)
    ax1.set_ylabel("Number of customers", fontsize=10)
    ax1.set_title("CLV Distribution\n(capped at 95th percentile)",
                  fontsize=12, fontweight="bold", pad=12, color="#2C2C2A")
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.legend(fontsize=9, frameon=False)
    ax1.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"R${x:,.0f}")
    )

    # ── Chart 2: CLV by tier ──────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("#FAFAFA")

    tier_order = ["platinum", "gold", "silver", "bronze", "low"]
    tier_colors = {
        "platinum": "#534AB7",
        "gold":     "#EF9F27",
        "silver":   "#888780",
        "bronze":   "#D85A30",
        "low":      "#D3D1C7",
    }
    tier_stats = (
        customer_stats.groupby("clv_tier")
        .agg(
            customer_count = ("customer_unique_id", "count"),
            avg_clv        = ("clv",                "mean"),
            total_revenue  = ("total_revenue",      "sum"),
        )
        .reset_index()
    )
    tier_stats = tier_stats.set_index("clv_tier").reindex(
        [t for t in tier_order if t in tier_stats.index]
    ).reset_index()

    bars = ax2.bar(
        tier_stats["clv_tier"],
        tier_stats["avg_clv"],
        color=[tier_colors.get(t, "#888780") for t in tier_stats["clv_tier"]],
        width=0.6,
        edgecolor="none"
    )
    for bar, (_, row) in zip(bars, tier_stats.iterrows()):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            f"R${row['avg_clv']:,.0f}\n({row['customer_count']:,} customers)",
            ha="center", va="bottom",
            fontsize=9, color="#444441"
        )
    ax2.set_xlabel("CLV Tier", fontsize=10)
    ax2.set_ylabel("Average CLV (R$)", fontsize=10)
    ax2.set_title("Avg CLV by Customer Tier",
                  fontsize=12, fontweight="bold", pad=12, color="#2C2C2A")
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"R${x:,.0f}")
    )

    plt.tight_layout()
    out = REPORTS / "clv_distribution.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#FAFAFA")
    plt.close()
    print(f"  Chart saved → {out}")


def plot_pareto_curve(pareto: dict) -> None:
    """
    Lorenz curve showing revenue concentration.
    The further the curve bows from the diagonal,
    the more concentrated revenue is in a few customers.
    """

    sorted_df = pareto["sorted_customers"]

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")

    # Lorenz curve
    ax.plot(
        sorted_df["customer_rank_pct"],
        sorted_df["cumulative_pct"],
        color="#534AB7",
        linewidth=2.5,
        label="Revenue concentration curve"
    )

    # Perfect equality line
    ax.plot(
        [0, 100], [0, 100],
        color="#B4B2A9",
        linewidth=1.5,
        linestyle="--",
        label="Perfect equality"
    )

    # Annotate key points
    for pct in [5, 20, 50]:
        mask = sorted_df["customer_rank_pct"] <= pct
        if mask.any():
            rev_pct = sorted_df[mask]["cumulative_pct"].max()
            ax.annotate(
                f"Top {pct}% → {rev_pct:.0f}% revenue",
                xy=(pct, rev_pct),
                xytext=(pct + 5, rev_pct - 8),
                fontsize=9,
                color="#3C3489",
                arrowprops=dict(
                    arrowstyle="->",
                    color="#3C3489",
                    lw=1.2
                )
            )

    # Shade the gap between curves
    ax.fill_between(
        sorted_df["customer_rank_pct"],
        sorted_df["customer_rank_pct"],
        sorted_df["cumulative_pct"],
        alpha=0.08,
        color="#7F77DD"
    )

    ax.set_xlabel("% of customers (ranked by revenue)", fontsize=11)
    ax.set_ylabel("Cumulative % of total revenue", fontsize=11)
    ax.set_title(
        "Revenue Concentration — Pareto Curve\n"
        f"Top 20% of customers → {pareto['top_20_revenue_pct']:.0f}% of revenue",
        fontsize=12, fontweight="bold", pad=15, color="#2C2C2A"
    )
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=10, frameon=False)
    ax.grid(linestyle="--", alpha=0.3)

    plt.tight_layout()
    out = REPORTS / "clv_pareto_curve.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#FAFAFA")
    plt.close()
    print(f"  Chart saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — EXECUTIVE REPORT
# ─────────────────────────────────────────────────────────────────────────────

def write_clv_report(customer_stats: pd.DataFrame,
                     seg_df: pd.DataFrame,
                     ch_df: pd.DataFrame,
                     pareto: dict) -> None:

    total_customers  = len(customer_stats)
    avg_clv          = customer_stats["clv"].mean()
    median_clv       = customer_stats["clv"].median()
    total_clv        = customer_stats["clv"].sum()
    avg_aov          = customer_stats["aov"].mean()
    avg_freq         = customer_stats["purchase_frequency"].mean()
    avg_lifespan     = customer_stats["customer_lifespan"].mean()

    best_channel     = ch_df.dropna(subset=["ltv_cac_ratio"]).iloc[0]
    worst_channel    = ch_df.dropna(subset=["ltv_cac_ratio"]).iloc[-1]
    best_segment     = seg_df.iloc[0]

    report = f"""
CUSTOMER LIFETIME VALUE REPORT
E-Commerce Analytics — Phase 7
{'='*60}

OVERALL CLV SUMMARY
─────────────────────────────────────────────────────────
Total customers analysed:    {total_customers:>10,}
Average CLV:                 R$ {avg_clv:>10,.2f}
Median CLV:                  R$ {median_clv:>10,.2f}
Total projected CLV:         R$ {total_clv:>10,.2f}

CLV COMPONENTS (AVERAGES)
─────────────────────────────────────────────────────────
Avg Order Value (AOV):       R$ {avg_aov:>10,.2f}
Purchase Frequency:              {avg_freq:>10.2f} orders/year
Customer Lifespan:               {avg_lifespan:>10.2f} years
Gross Margin Rate:               {GROSS_MARGIN_RATE*100:>9.0f}%

CLV BY RFM SEGMENT
─────────────────────────────────────────────────────────
Best segment: {best_segment['rfm_segment']}
  Avg CLV:    R$ {best_segment['avg_clv']:,.2f}
  Avg AOV:    R$ {best_segment['avg_aov']:,.2f}
  Avg orders: {best_segment['avg_orders']:.1f}

LTV:CAC BY CHANNEL
─────────────────────────────────────────────────────────
Best channel:  {best_channel['channel_name']:<20} LTV:CAC = {best_channel['ltv_cac_ratio']:.1f}x
Worst channel: {worst_channel['channel_name']:<20} LTV:CAC = {worst_channel['ltv_cac_ratio']:.1f}x

REVENUE CONCENTRATION (PARETO)
─────────────────────────────────────────────────────────
Top  5% of customers → {pareto['top_5_revenue_pct']:.1f}% of revenue
Top 20% of customers → {pareto['top_20_revenue_pct']:.1f}% of revenue
{pareto['pct_customers_for_80_rev']:.1f}% of customers generate 80% of revenue

RECOMMENDATIONS
─────────────────────────────────────────────────────────
1. INVEST in {best_channel['channel_name']} — highest LTV:CAC at {best_channel['ltv_cac_ratio']:.1f}x.
   Scale budget here before other channels.

2. REVIEW {worst_channel['channel_name']} — LTV:CAC of {worst_channel['ltv_cac_ratio']:.1f}x
   means the channel may not be profitable. Audit or pause.

3. PROTECT the top {pareto['pct_customers_for_80_rev']:.0f}% of customers who drive 80%
   of revenue. Assign dedicated account management and
   ensure they never receive a bad delivery experience.

4. UPSELL to bronze/silver tier customers — a single
   additional purchase moves them significantly up
   the CLV curve given the low baseline frequency.
"""

    out = REPORTS / "clv_report.txt"
    out.write_text(report, encoding="utf-8")
    print(f"  Report saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  E-COMMERCE ANALYTICS — PHASE 7 CLV MODEL")
    print("=" * 60 + "\n")

    # 1. Load
    df = load_data()

    # 2. Customer-level CLV
    print("\nComputing customer CLV...")
    customer_stats = compute_customer_clv(df)

    # 3. CLV by segment
    seg_df = clv_by_segment(customer_stats)

    # 4. CLV by channel
    ch_df = clv_by_channel(customer_stats)

    # 5. Pareto
    pareto = pareto_analysis(customer_stats)

    # 6. Save CSVs
    print("\nSaving outputs...")
    customer_stats.to_csv(
        PROCESSED / "clv_customers.csv", index=False
    )
    seg_df.to_csv(PROCESSED / "clv_by_segment.csv", index=False)
    ch_df.to_csv(PROCESSED / "clv_by_channel.csv", index=False)
    print(f"  Saved clv_customers.csv  ({len(customer_stats):,} rows)")
    print(f"  Saved clv_by_segment.csv ({len(seg_df)} segments)")
    print(f"  Saved clv_by_channel.csv ({len(ch_df)} channels)")

    # 7. Charts
    print("\nGenerating charts...")
    plot_clv_by_segment(seg_df)
    plot_ltv_cac(ch_df)
    plot_clv_distribution(customer_stats)
    plot_pareto_curve(pareto)

    # 8. Report
    write_clv_report(customer_stats, seg_df, ch_df, pareto)

    print("\n  Phase 7 complete. Ready for Phase 8 — Churn Prediction ML.\n")


if __name__ == "__main__":
    main()