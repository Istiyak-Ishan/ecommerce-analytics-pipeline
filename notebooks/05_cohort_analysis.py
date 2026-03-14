"""
Phase 6 — Cohort Retention Analysis
======================================
Builds a full customer retention cohort matrix showing
what percentage of each monthly cohort returns to purchase
in subsequent months.

Outputs:
  data/processed/cohort_matrix.csv        — raw retention counts
  data/processed/cohort_retention_pct.csv — percentage matrix
  reports/cohort_heatmap.png              — publication heatmap
  reports/cohort_revenue_heatmap.png      — revenue per cohort
  reports/cohort_report.txt               — business summary

Run from project root:
    python notebooks/05_cohort_analysis.py
"""

import warnings
import sqlite3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
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
PROCESSED.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_orders() -> pd.DataFrame:
    """
    Load delivered orders with customer_unique_id.
    We use customer_unique_id (not customer_id) because
    Olist assigns a new customer_id for each order —
    unique_id is the true person identifier for repeat analysis.
    """
    conn = sqlite3.connect(DB_PATH)

    orders = pd.read_sql("""
        SELECT
            f.order_id,
            c.customer_unique_id,
            f.order_purchase_ts,
            f.revenue,
            f.order_status
        FROM fact_orders_clean f
        JOIN dim_customers_clean c ON f.customer_id = c.customer_id
        WHERE f.order_status = 'delivered'
    """, conn)

    conn.close()

    orders["order_purchase_ts"] = pd.to_datetime(
        orders["order_purchase_ts"], errors="coerce"
    )
    orders = orders.dropna(subset=["order_purchase_ts"])

    print(f"  Loaded {len(orders):,} delivered orders")
    print(f"  Unique customers: {orders['customer_unique_id'].nunique():,}")
    print(f"  Date range: {orders['order_purchase_ts'].min().date()} "
          f"→ {orders['order_purchase_ts'].max().date()}")

    return orders


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — ASSIGN COHORT MONTH
# ─────────────────────────────────────────────────────────────────────────────

def assign_cohorts(orders: pd.DataFrame) -> pd.DataFrame:
    """
    For each customer, find their FIRST purchase month — that is their cohort.
    Then for every subsequent order, calculate how many months later it was
    relative to their cohort month (cohort_period).

    cohort_period = 0 means the customer bought in their cohort month
    cohort_period = 1 means they came back 1 month later
    cohort_period = 6 means they came back 6 months later
    """

    # Create order_month as a Period for clean month arithmetic
    orders["order_month"] = orders["order_purchase_ts"].dt.to_period("M")

    # Find each customer's first purchase month = their cohort
    first_purchase = (
        orders.groupby("customer_unique_id")["order_month"]
        .min()
        .reset_index()
        .rename(columns={"order_month": "cohort_month"})
    )

    # Join cohort month back to every order
    orders = orders.merge(first_purchase, on="customer_unique_id", how="left")

    # Calculate months elapsed since cohort (cohort_period)
    orders["cohort_period"] = (
        orders["order_month"] - orders["cohort_month"]
    ).apply(lambda x: x.n)   # .n gives the integer month difference

    print(f"\n  Cohort period range: 0 → {orders['cohort_period'].max()} months")
    print(f"  Total cohorts: {orders['cohort_month'].nunique()}")

    return orders


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — BUILD RETENTION MATRIX
# ─────────────────────────────────────────────────────────────────────────────

def build_retention_matrix(orders: pd.DataFrame) -> tuple:
    """
    Builds two matrices:
      1. cohort_counts  — raw number of unique customers per cohort per period
      2. retention_pct  — each count divided by cohort size (period 0)

    The result is a DataFrame where:
      - rows = cohort months (Jan 2017, Feb 2017, ...)
      - columns = cohort periods (0, 1, 2, 3, ...)
      - values = % of original cohort still purchasing
    """

    # Count unique customers per cohort × period
    cohort_counts = (
        orders.groupby(["cohort_month", "cohort_period"])
        ["customer_unique_id"]
        .nunique()
        .reset_index()
        .rename(columns={"customer_unique_id": "customers"})
    )

    # Pivot to matrix shape
    cohort_matrix = cohort_counts.pivot_table(
        index="cohort_month",
        columns="cohort_period",
        values="customers"
    )

    # Cohort size = number of customers in period 0 (first purchase month)
    cohort_sizes = cohort_matrix[0]

    # Retention % = each cell divided by that row's cohort size
    retention_pct = cohort_matrix.divide(cohort_sizes, axis=0) * 100

    # Keep only first 13 periods (0 to 12 months) for readability
    max_period = min(13, cohort_matrix.shape[1])
    cohort_matrix  = cohort_matrix.iloc[:, :max_period]
    retention_pct  = retention_pct.iloc[:, :max_period]

    # Convert period index to string for display
    cohort_matrix.index  = cohort_matrix.index.astype(str)
    retention_pct.index  = retention_pct.index.astype(str)

    print(f"\n  Cohort matrix shape: {cohort_matrix.shape}")
    print(f"  ({cohort_matrix.shape[0]} cohorts × "
          f"{cohort_matrix.shape[1]} periods)")

    return cohort_matrix, retention_pct, cohort_sizes


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — BUILD REVENUE COHORT MATRIX
# ─────────────────────────────────────────────────────────────────────────────

def build_revenue_matrix(orders: pd.DataFrame) -> pd.DataFrame:
    """
    Builds a revenue version of the cohort matrix.
    Instead of % of customers retained, shows average revenue
    generated per customer per cohort period.

    This answers: not just "did they come back?" but
    "how much did they spend when they came back?"
    """

    revenue_cohort = (
        orders.groupby(["cohort_month", "cohort_period"])
        .agg(
            total_revenue = ("revenue",              "sum"),
            unique_customers = ("customer_unique_id","nunique"),
        )
        .reset_index()
    )

    revenue_cohort["avg_revenue_per_customer"] = (
        revenue_cohort["total_revenue"] /
        revenue_cohort["unique_customers"]
    ).round(2)

    revenue_matrix = revenue_cohort.pivot_table(
        index="cohort_month",
        columns="cohort_period",
        values="avg_revenue_per_customer"
    )

    max_period = min(13, revenue_matrix.shape[1])
    revenue_matrix = revenue_matrix.iloc[:, :max_period]
    revenue_matrix.index = revenue_matrix.index.astype(str)

    return revenue_matrix


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — PLOT RETENTION HEATMAP
# ─────────────────────────────────────────────────────────────────────────────

def plot_retention_heatmap(retention_pct: pd.DataFrame,
                            cohort_matrix: pd.DataFrame) -> None:
    """
    Produces the signature cohort heatmap.

    Design choices:
    - Purple→white color scale (dark = high retention, light = low)
    - Period 0 always shows 100% (masked differently)
    - Each cell shows the % value
    - Cohort sizes shown in the row labels
    - Only cohorts with enough data shown (at least 3 months)
    """

    # Filter to cohorts with at least 3 periods of data
    valid_cohorts = retention_pct[retention_pct[1].notna()].copy()

    fig, ax = plt.subplots(figsize=(16, 10))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")

    # Create a custom purple colormap
    purple_cmap = LinearSegmentedColormap.from_list(
        "purple_retention",
        ["#EEEDFE", "#AFA9EC", "#7F77DD", "#534AB7", "#3C3489", "#26215C"]
    )

    # Mask period 0 (always 100%, treated separately)
    plot_data = valid_cohorts.copy()
    plot_data_display = plot_data.copy()
    plot_data_display[0] = np.nan  # hide period 0 from color scale

    sns.heatmap(
        plot_data_display,
        annot=False,       # we'll add custom annotations
        fmt=".1f",
        cmap=purple_cmap,
        linewidths=0.5,
        linecolor="#F1EFE8",
        ax=ax,
        vmin=0,
        vmax=15,           # cap at 15% so colour differences are visible
        cbar_kws={
            "label": "Retention %",
            "shrink": 0.6,
            "pad": 0.02,
        }
    )

    # Add custom cell annotations
    for i in range(len(plot_data.index)):
        for j in range(len(plot_data.columns)):
            val = plot_data.iloc[i, j]
            period = plot_data.columns[j]

            if pd.isna(val):
                # Future period — show dash
                ax.text(j + 0.5, i + 0.5, "—",
                        ha="center", va="center",
                        fontsize=8, color="#B4B2A9")
            elif period == 0:
                # Period 0 — always 100%, highlight differently
                ax.text(j + 0.5, i + 0.5, "100%",
                        ha="center", va="center",
                        fontsize=8, fontweight="bold",
                        color="#26215C",
                        bbox=dict(boxstyle="round,pad=0.2",
                                  facecolor="#CECBF6",
                                  edgecolor="none"))
            else:
                # Retention periods
                color = "white" if val > 6 else "#26215C"
                ax.text(j + 0.5, i + 0.5, f"{val:.1f}%",
                        ha="center", va="center",
                        fontsize=8, color=color)

    # Column labels
    col_labels = [f"M+{c}" if c > 0 else "M0\n(cohort)" 
                  for c in plot_data.columns]
    ax.set_xticklabels(col_labels, fontsize=9, rotation=0)

    # Row labels with cohort sizes
    cohort_size_col = cohort_matrix[0]
    row_labels = []
    for idx in valid_cohorts.index:
        size = int(cohort_size_col.get(idx, 0))
        row_labels.append(f"{idx}  (n={size:,})")
    ax.set_yticklabels(row_labels, fontsize=9, rotation=0)

    ax.set_title(
        "Customer Retention Cohort Matrix\n"
        "% of each cohort returning to purchase in subsequent months",
        fontsize=14, fontweight="bold", pad=20,
        color="#2C2C2A"
    )
    ax.set_xlabel("Months since first purchase", fontsize=11,
                  labelpad=10)
    ax.set_ylabel("Cohort (first purchase month)", fontsize=11,
                  labelpad=10)

    plt.tight_layout()
    out = REPORTS / "cohort_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight",
                facecolor="#FAFAFA")
    plt.close()
    print(f"  Chart saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — PLOT REVENUE HEATMAP
# ─────────────────────────────────────────────────────────────────────────────

def plot_revenue_heatmap(revenue_matrix: pd.DataFrame) -> None:
    """
    Heatmap showing average revenue per returning customer
    per cohort period. Teal colormap.
    """

    valid = revenue_matrix[revenue_matrix[1].notna()].copy()
    display = valid.copy()
    display[0] = np.nan  # suppress period 0 from colour scale

    fig, ax = plt.subplots(figsize=(16, 10))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")

    teal_cmap = LinearSegmentedColormap.from_list(
        "teal_revenue",
        ["#E1F5EE", "#5DCAA5", "#1D9E75", "#0F6E56", "#085041"]
    )

    sns.heatmap(
        display,
        annot=False,
        cmap=teal_cmap,
        linewidths=0.5,
        linecolor="#F1EFE8",
        ax=ax,
        cbar_kws={
            "label": "Avg revenue per customer (R$)",
            "shrink": 0.6,
        }
    )

    for i in range(len(valid.index)):
        for j in range(len(valid.columns)):
            val = valid.iloc[i, j]
            period = valid.columns[j]
            if pd.isna(val):
                ax.text(j + 0.5, i + 0.5, "—",
                        ha="center", va="center",
                        fontsize=8, color="#B4B2A9")
            else:
                color = "white" if val > 120 else "#085041"
                prefix = "R$" if period == 0 else ""
                ax.text(j + 0.5, i + 0.5,
                        f"{prefix}{val:.0f}",
                        ha="center", va="center",
                        fontsize=8, color=color)

    col_labels = [f"M+{c}" if c > 0 else "M0" for c in valid.columns]
    ax.set_xticklabels(col_labels, fontsize=9, rotation=0)
    ax.set_yticklabels(valid.index, fontsize=9, rotation=0)

    ax.set_title(
        "Average Revenue per Customer by Cohort\n"
        "R$ generated per returning customer in each period",
        fontsize=14, fontweight="bold", pad=20,
        color="#2C2C2A"
    )
    ax.set_xlabel("Months since first purchase", fontsize=11, labelpad=10)
    ax.set_ylabel("Cohort (first purchase month)", fontsize=11, labelpad=10)

    plt.tight_layout()
    out = REPORTS / "cohort_revenue_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#FAFAFA")
    plt.close()
    print(f"  Chart saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — PLOT RETENTION CURVES
# ─────────────────────────────────────────────────────────────────────────────

def plot_retention_curves(retention_pct: pd.DataFrame) -> None:
    """
    Line chart showing retention curves for each cohort overlaid.
    Helps identify whether newer cohorts retain better than older ones
    (improvement over time = product is getting better).
    Also plots the average retention curve in bold.
    """

    valid = retention_pct[retention_pct[1].notna()].copy()

    # Only keep periods 0–12
    periods = [c for c in valid.columns if c <= 12]
    valid = valid[periods]

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")

    # Plot individual cohort lines (thin, muted)
    colors_range = plt.cm.Blues(np.linspace(0.3, 0.85, len(valid)))
    for i, (cohort, row) in enumerate(valid.iterrows()):
        data = row.dropna()
        if len(data) > 2:
            ax.plot(
                data.index,
                data.values,
                color=colors_range[i],
                linewidth=1.2,
                alpha=0.6
            )

    # Average retention curve (bold)
    avg_curve = valid.mean(skipna=True)
    ax.plot(
        avg_curve.index,
        avg_curve.values,
        color="#534AB7",
        linewidth=3,
        zorder=10,
        label="Average across all cohorts"
    )

    # Annotate average values at key periods
    for period in [1, 3, 6, 12]:
        if period in avg_curve.index and not pd.isna(avg_curve[period]):
            ax.annotate(
                f"M+{period}: {avg_curve[period]:.1f}%",
                xy=(period, avg_curve[period]),
                xytext=(period + 0.3, avg_curve[period] + 0.5),
                fontsize=9,
                color="#3C3489",
                fontweight="bold"
            )

    ax.set_title(
        "Customer Retention Curves by Cohort",
        fontsize=14, fontweight="bold", pad=15, color="#2C2C2A"
    )
    ax.set_xlabel("Months since first purchase", fontsize=11)
    ax.set_ylabel("% of cohort retained", fontsize=11)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, max(valid[1].max() * 1.2, 15))
    ax.legend(fontsize=10, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    out = REPORTS / "cohort_retention_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#FAFAFA")
    plt.close()
    print(f"  Chart saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — COHORT SIZE BAR CHART
# ─────────────────────────────────────────────────────────────────────────────

def plot_cohort_sizes(orders: pd.DataFrame) -> None:
    """
    Bar chart of new customers acquired per month.
    This shows whether the business is growing its customer base.
    """

    cohort_sizes = (
        orders.groupby(
            orders["order_purchase_ts"].dt.to_period("M")
        )["customer_unique_id"]
        .nunique()
        .reset_index()
    )
    cohort_sizes.columns = ["month", "new_customers"]
    cohort_sizes["month"] = cohort_sizes["month"].astype(str)

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")

    bars = ax.bar(
        cohort_sizes["month"],
        cohort_sizes["new_customers"],
        color="#7F77DD",
        edgecolor="none",
        width=0.7
    )

    # 3-month rolling average line
    cohort_sizes["rolling_avg"] = (
        cohort_sizes["new_customers"]
        .rolling(3, min_periods=1)
        .mean()
    )
    ax.plot(
        cohort_sizes["month"],
        cohort_sizes["rolling_avg"],
        color="#D85A30",
        linewidth=2.5,
        label="3-month rolling avg",
        zorder=5
    )

    ax.set_title(
        "New Customers Acquired per Month",
        fontsize=14, fontweight="bold", pad=15, color="#2C2C2A"
    )
    ax.set_xlabel("Month", fontsize=11)
    ax.set_ylabel("New customers", fontsize=11)
    ax.tick_params(axis="x", rotation=45)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:,.0f}")
    )
    ax.legend(fontsize=10, frameon=False)

    plt.tight_layout()
    out = REPORTS / "cohort_new_customers.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#FAFAFA")
    plt.close()
    print(f"  Chart saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 — COMPUTE KEY METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_cohort_metrics(retention_pct: pd.DataFrame,
                            cohort_matrix: pd.DataFrame,
                            orders: pd.DataFrame) -> dict:
    """
    Derives the key business metrics from the cohort data.
    These numbers go into the executive report and portfolio.
    """

    # Average retention at each period across all cohorts
    avg_ret = retention_pct.mean(skipna=True)

    # Overall repeat purchase rate
    customer_orders = (
        orders.groupby("customer_unique_id")["order_id"].count()
    )
    repeat_rate = (customer_orders > 1).mean() * 100

    # Average number of orders per repeat customer
    avg_orders_repeat = customer_orders[customer_orders > 1].mean()

    # Best and worst retention cohorts (at M+1)
    m1 = retention_pct[1].dropna()
    best_cohort  = m1.idxmax()
    worst_cohort = m1.idxmin()

    # Average cohort size
    avg_cohort_size = cohort_matrix[0].mean()

    metrics = {
        "avg_m1_retention":     round(avg_ret.get(1, 0), 2),
        "avg_m3_retention":     round(avg_ret.get(3, 0), 2),
        "avg_m6_retention":     round(avg_ret.get(6, 0), 2),
        "avg_m12_retention":    round(avg_ret.get(12, 0), 2),
        "overall_repeat_rate":  round(repeat_rate, 2),
        "avg_orders_repeat_customer": round(avg_orders_repeat, 2),
        "best_m1_cohort":       best_cohort,
        "best_m1_retention":    round(m1.max(), 2),
        "worst_m1_cohort":      worst_cohort,
        "worst_m1_retention":   round(m1.min(), 2),
        "avg_cohort_size":      round(avg_cohort_size, 0),
        "total_cohorts":        len(retention_pct),
    }

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# STEP 10 — EXECUTIVE REPORT
# ─────────────────────────────────────────────────────────────────────────────

def write_cohort_report(metrics: dict) -> None:
    """Write a plain-text cohort analysis business report."""

    report = f"""
COHORT RETENTION ANALYSIS REPORT
E-Commerce Analytics — Phase 6
{'='*60}

RETENTION SUMMARY
─────────────────────────────────────────────────────────
Overall repeat purchase rate:     {metrics['overall_repeat_rate']:.2f}%
Avg orders per repeat customer:   {metrics['avg_orders_repeat_customer']:.2f}
Average cohort size (monthly):    {metrics['avg_cohort_size']:,.0f} customers
Total cohorts analysed:           {metrics['total_cohorts']}

AVERAGE RETENTION BY PERIOD
─────────────────────────────────────────────────────────
Month +1  (1 month after first buy):  {metrics['avg_m1_retention']:.2f}%
Month +3  (3 months after first buy): {metrics['avg_m3_retention']:.2f}%
Month +6  (6 months after first buy): {metrics['avg_m6_retention']:.2f}%
Month +12 (1 year after first buy):   {metrics['avg_m12_retention']:.2f}%

BEST AND WORST COHORTS (M+1 retention)
─────────────────────────────────────────────────────────
Best cohort:   {metrics['best_m1_cohort']}  ({metrics['best_m1_retention']:.2f}% returned in M+1)
Worst cohort:  {metrics['worst_m1_cohort']}  ({metrics['worst_m1_retention']:.2f}% returned in M+1)

BUSINESS INTERPRETATION
─────────────────────────────────────────────────────────
An M+1 retention rate of {metrics['avg_m1_retention']:.2f}% means that for every
100 new customers acquired, only {metrics['avg_m1_retention']:.0f} return the following month.

Industry benchmark for e-commerce:
  - Good:      5-10% monthly retention
  - Average:   2-5%  monthly retention
  - Concerning: <2%  monthly retention

RECOMMENDATIONS
─────────────────────────────────────────────────────────
1. ONBOARDING  — Customers who don't return in M+1 are likely lost.
                 Implement a 30-day post-purchase email sequence.

2. REACTIVATION — Target M+2 and M+3 dropoffs with personalised
                  offers. These customers still remember you.

3. LOYALTY     — The {metrics['avg_orders_repeat_customer']:.1f}x avg orders from repeat customers
                 shows high value. Formalise a loyalty programme.

4. COHORT MONITORING — Track each new monthly cohort's M+1
                       retention as a leading KPI. A declining
                       trend signals product or service issues early.
"""

    out = REPORTS / "cohort_report.txt"
    out.write_text(report, encoding="utf-8")
    print(f"  Report saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  E-COMMERCE ANALYTICS — PHASE 6 COHORT ANALYSIS")
    print("=" * 60 + "\n")

    # 1. Load
    orders = load_orders()

    # 2. Assign cohorts
    print("\nAssigning cohorts...")
    orders = assign_cohorts(orders)

    # 3. Build matrices
    print("\nBuilding retention matrices...")
    cohort_matrix, retention_pct, cohort_sizes = build_retention_matrix(orders)
    revenue_matrix = build_revenue_matrix(orders)

    # 4. Save CSVs
    cohort_matrix.to_csv(PROCESSED / "cohort_matrix.csv")
    retention_pct.to_csv(PROCESSED / "cohort_retention_pct.csv")
    revenue_matrix.to_csv(PROCESSED / "cohort_revenue_matrix.csv")
    print(f"\n  Saved cohort_matrix.csv")
    print(f"  Saved cohort_retention_pct.csv")
    print(f"  Saved cohort_revenue_matrix.csv")

    # 5. Print retention matrix to console
    print("\n" + "=" * 60)
    print("  RETENTION MATRIX (% of cohort returning)")
    print("=" * 60)
    pd.set_option("display.max_columns", 15)
    pd.set_option("display.width", 120)
    pd.set_option("display.float_format", "{:.1f}".format)
    print(retention_pct.fillna(0).to_string())

    # 6. Compute metrics
    metrics = compute_cohort_metrics(retention_pct, cohort_matrix, orders)

    print("\n" + "=" * 60)
    print("  KEY COHORT METRICS")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"  {k:<40} {v}")

    # 7. Charts
    print("\nGenerating charts...")
    plot_retention_heatmap(retention_pct, cohort_matrix)
    plot_revenue_heatmap(revenue_matrix)
    plot_retention_curves(retention_pct)
    plot_cohort_sizes(orders)

    # 8. Report
    write_cohort_report(metrics)

    print("\n  Phase 6 complete. Ready for Phase 7 — CLV Model.\n")


if __name__ == "__main__":
    main()