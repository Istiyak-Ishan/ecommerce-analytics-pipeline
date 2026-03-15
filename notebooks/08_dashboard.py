"""
Phase 9 — Professional Dashboard (Pure Python)
================================================
Generates 18 publication-quality charts as PNGs
then compiles them into one PDF report.

Run: python notebooks/08_dashboard.py

Outputs:
    reports/charts/          ← 18 individual PNG charts
    reports/dashboard.pdf    ← full compiled PDF report
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from pathlib import Path

warnings.filterwarnings("ignore")
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.linestyle":   "--",
    "figure.dpi":       150,
    "savefig.dpi":      150,
    "savefig.bbox":     "tight",
    "savefig.facecolor":"white",
})

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

PROCESSED = Path("data/processed")
ANALYTICS  = Path("data/processed/analytics")
REPORTS    = Path("reports")
CHARTS     = REPORTS / "charts"
CHARTS.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# COLORS
# ─────────────────────────────────────────────────────────────────────────────

PURPLE  = "#534AB7"
PURPLE2 = "#7F77DD"
PURPLE3 = "#AFA9EC"
PURPLE4 = "#EEEDFE"
TEAL    = "#1D9E75"
TEAL2   = "#9FE1CB"
CORAL   = "#D85A30"
AMBER   = "#EF9F27"
RED     = "#E24B4A"
GRAY    = "#888780"
GRAY2   = "#D3D1C7"
BG      = "#F8F7FF"
TEXT    = "#2C2C2A"
SUBTEXT = "#5F5E5A"

PURPLE_CMAP = LinearSegmentedColormap.from_list(
    "purple_ret",
    ["#FFFFFF", "#EEEDFE", "#AFA9EC", "#7F77DD", "#534AB7", "#26215C"]
)

SEG_COLORS = {
    "champions":       PURPLE,
    "loyal":           PURPLE2,
    "potential_loyal": PURPLE3,
    "need_attention":  AMBER,
    "at_risk":         CORAL,
    "lost":            RED,
    "unknown":         GRAY,
}


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_data() -> dict:
    print("Loading data...")
    d = {}

    d["orders"]     = pd.read_csv(PROCESSED / "master_analytical.csv",
                                   low_memory=False)
    d["clv"]        = pd.read_csv(PROCESSED / "clv_customers.csv")
    d["clv_seg"]    = pd.read_csv(PROCESSED / "clv_by_segment.csv")
    d["clv_ch"]     = pd.read_csv(PROCESSED / "clv_by_channel.csv")
    d["churn"]      = pd.read_csv(PROCESSED / "churn_predictions.csv")
    d["leakage"]    = pd.read_csv(REPORTS   / "leakage_summary.csv")
    d["monthly"]    = pd.read_csv(ANALYTICS / "02_monthly_growth.csv")
    d["categories"] = pd.read_csv(ANALYTICS / "06_top_categories.csv")
    d["regions"]    = pd.read_csv(ANALYTICS / "07_top_regions.csv")
    d["cohort"]     = pd.read_csv(PROCESSED / "cohort_retention_pct.csv")

    d["orders"]["order_purchase_ts"] = pd.to_datetime(
        d["orders"]["order_purchase_ts"], errors="coerce"
    )
    d["del"] = d["orders"][
        d["orders"]["order_status"] == "delivered"
    ].copy()

    # Compute rolling avg if missing
    if "rolling_3m_avg" not in d["monthly"].columns:
        d["monthly"]["rolling_3m_avg"] = (
            d["monthly"]["monthly_revenue"]
            .rolling(3, min_periods=1).mean()
        )

    # Compute gross_margin_pct if missing
    if "gross_margin_pct" not in d["categories"].columns:
        d["categories"]["gross_margin_pct"] = (
            d["categories"]["total_gross_profit"] /
            d["categories"]["total_revenue"] * 100
        ).round(2)

    # Set cohort index
    if "cohort_month" in d["cohort"].columns:
        d["cohort"] = d["cohort"].set_index("cohort_month")

    print(f"  Delivered orders: {len(d['del']):,}")
    return d


# ─────────────────────────────────────────────────────────────────────────────
# COMPUTE KPIs
# ─────────────────────────────────────────────────────────────────────────────

def compute_kpis(d: dict) -> dict:
    dl    = d["del"]
    churn = d["churn"]
    clv   = d["clv"]
    leak  = d["leakage"]

    total_revenue   = dl["revenue"].sum()
    total_orders    = dl["order_id"].nunique()
    total_customers = dl["customer_id"].nunique()
    aov             = dl["revenue"].mean()
    gross_profit    = dl["gross_profit"].sum()
    gross_margin    = gross_profit / total_revenue * 100
    total_disc      = dl["discount_amount"].sum()
    disc_rate       = total_disc / total_revenue * 100
    total_freight   = dl["freight_value"].sum()
    late_count      = int(pd.to_numeric(
        dl["is_late_delivery"], errors="coerce"
    ).fillna(0).sum())
    late_rate       = late_count / total_orders * 100
    churn_rate      = churn["churn_flag"].mean() * 100
    high_risk_n     = len(churn[churn["churn_risk_tier"] == "high"])
    rev_at_risk     = churn[
        churn["churn_risk_tier"] == "high"
    ]["total_revenue"].sum()
    total_leak      = leak["leakage_amount"].sum()
    leak_pct        = total_leak / total_revenue * 100
    repeat_rate     = (clv["total_orders"] > 1).mean() * 100
    avg_clv         = clv["clv"].mean()

    return dict(
        total_revenue=total_revenue,
        total_orders=total_orders,
        total_customers=total_customers,
        aov=aov,
        gross_profit=gross_profit,
        gross_margin=gross_margin,
        total_disc=total_disc,
        disc_rate=disc_rate,
        total_freight=total_freight,
        late_count=late_count,
        late_rate=late_rate,
        churn_rate=churn_rate,
        high_risk_n=high_risk_n,
        rev_at_risk=rev_at_risk,
        total_leak=total_leak,
        leak_pct=leak_pct,
        repeat_rate=repeat_rate,
        avg_clv=avg_clv,
    )


# ─────────────────────────────────────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────────────────────────────────────

def save(fig: plt.Figure, name: str) -> Path:
    out = CHARTS / f"{name}.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {name}.png")
    return out


def styled_ax(ax, title: str, xlabel: str = "",
              ylabel: str = "") -> None:
    ax.set_title(title, fontsize=13, fontweight="bold",
                 color=TEXT, pad=14)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10, color=SUBTEXT)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10, color=SUBTEXT)
    ax.tick_params(colors=SUBTEXT)
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_color(GRAY2)


# ─────────────────────────────────────────────────────────────────────────────
# CHART 01 — KPI SUMMARY PAGE
# ─────────────────────────────────────────────────────────────────────────────

def chart_kpi_summary(k: dict) -> Path:
    fig = plt.figure(figsize=(18, 7), facecolor=BG)
    fig.suptitle(
        "E-Commerce Revenue Intelligence Dashboard — Executive Summary",
        fontsize=18, fontweight="bold", color=TEXT, y=1.02
    )

    kpis = [
        ("Total Revenue",      f"R${k['total_revenue']:,.0f}",  PURPLE),
        ("Total Orders",       f"{k['total_orders']:,}",        PURPLE),
        ("Avg Order Value",    f"R${k['aov']:,.2f}",            PURPLE),
        ("Gross Margin",       f"{k['gross_margin']:.1f}%",     TEAL),
        ("Repeat Rate",        f"{k['repeat_rate']:.1f}%",      TEAL),
        ("Revenue Leakage",    f"R${k['total_leak']:,.0f}",     AMBER),
        ("Discount Rate",      f"{k['disc_rate']:.1f}%",        CORAL),
        ("Late Deliveries",    f"{k['late_rate']:.1f}%",        CORAL),
        ("Churn Rate",         f"{k['churn_rate']:.1f}%",       RED),
        ("High-Risk Customers",f"{k['high_risk_n']:,}",         RED),
        ("Revenue at Risk",    f"R${k['rev_at_risk']:,.0f}",    RED),
        ("Avg CLV",            f"R${k['avg_clv']:,.0f}",        TEAL),
    ]

    cols = 6
    rows = 2
    axes = []
    for i, (label, value, color) in enumerate(kpis):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.set_facecolor("white")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])

        # Card background
        ax.add_patch(FancyBboxPatch(
            (0.05, 0.05), 0.90, 0.90,
            boxstyle="round,pad=0.02",
            linewidth=1.5,
            edgecolor=color + "44",
            facecolor="white",
            transform=ax.transAxes,
        ))

        ax.text(0.5, 0.72, value,
                ha="center", va="center",
                fontsize=18, fontweight="bold",
                color=color, transform=ax.transAxes)
        ax.text(0.5, 0.30, label,
                ha="center", va="center",
                fontsize=9, color=SUBTEXT,
                transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        axes.append(ax)

    plt.tight_layout(pad=1.5)
    return save(fig, "01_kpi_summary")


# ─────────────────────────────────────────────────────────────────────────────
# CHART 02 — MONTHLY REVENUE
# ─────────────────────────────────────────────────────────────────────────────

def chart_monthly_revenue(monthly: pd.DataFrame) -> Path:
    fig, ax1 = plt.subplots(figsize=(16, 6), facecolor="white")
    ax2 = ax1.twinx()

    x     = range(len(monthly))
    bars  = ax1.bar(x, monthly["monthly_revenue"],
                    color=PURPLE4, edgecolor=PURPLE,
                    linewidth=0.5, label="Monthly Revenue", zorder=2)

    ax1.plot(x, monthly["rolling_3m_avg"],
             color=CORAL, linewidth=2.5,
             label="3M Rolling Avg", zorder=3)

    ax2.plot(x, monthly["cumulative_revenue"],
             color=TEAL, linewidth=2,
             linestyle="--", label="Cumulative", zorder=3)

    ax1.set_xticks(x)
    ax1.set_xticklabels(monthly["year_month"],
                         rotation=45, ha="right", fontsize=9)
    ax1.set_ylabel("Monthly Revenue (R$)", color=TEXT)
    ax2.set_ylabel("Cumulative Revenue (R$)", color=TEAL)
    ax1.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"R${v:,.0f}")
    )
    ax2.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"R${v:,.0f}")
    )
    ax2.tick_params(colors=TEAL)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc="upper left", fontsize=9)

    styled_ax(ax1, "Monthly Revenue Trend")
    ax1.set_facecolor("white")
    ax2.set_facecolor("white")
    fig.tight_layout()
    return save(fig, "02_monthly_revenue")


# ─────────────────────────────────────────────────────────────────────────────
# CHART 03 — MoM GROWTH
# ─────────────────────────────────────────────────────────────────────────────

def chart_mom_growth(monthly: pd.DataFrame) -> Path:
    df     = monthly.dropna(subset=["mom_growth_pct"]).copy()
    colors = [TEAL if v >= 0 else RED
              for v in df["mom_growth_pct"]]

    fig, ax = plt.subplots(figsize=(16, 5), facecolor="white")
    x = range(len(df))
    ax.bar(x, df["mom_growth_pct"], color=colors,
           edgecolor="white", linewidth=0.5)
    ax.axhline(0, color=TEXT, linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(df["year_month"],
                        rotation=45, ha="right", fontsize=9)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v:+.1f}%")
    )

    # Annotate bars
    for xi, val in zip(x, df["mom_growth_pct"]):
        if abs(val) > 5:
            ax.text(xi, val + (1 if val >= 0 else -1.5),
                    f"{val:+.1f}%",
                    ha="center", fontsize=7.5,
                    color=TEAL if val >= 0 else RED)

    styled_ax(ax, "Month-over-Month Revenue Growth %",
              ylabel="Growth %")
    ax.add_patch(mpatches.FancyArrowPatch(
        (0, 0), (0, 0), color=TEAL, label="Growth"
    ))
    green_patch = mpatches.Patch(color=TEAL, label="Growth")
    red_patch   = mpatches.Patch(color=RED,  label="Decline")
    ax.legend(handles=[green_patch, red_patch],
              fontsize=9, loc="upper left")
    fig.tight_layout()
    return save(fig, "03_mom_growth")


# ─────────────────────────────────────────────────────────────────────────────
# CHART 04 — REVENUE WATERFALL
# ─────────────────────────────────────────────────────────────────────────────

def chart_waterfall(k: dict) -> Path:
    gross  = k["total_revenue"] + k["total_disc"] + k["total_freight"]
    labels = ["Gross\nRevenue", "Discounts", "Freight",
              "Net\nRevenue", "Gross\nProfit"]
    values = [gross, -k["total_disc"], -k["total_freight"],
              k["total_revenue"], k["gross_profit"]]
    colors = [PURPLE, RED, RED, TEAL, TEAL]
    types  = ["abs", "rel", "rel", "total", "total"]

    fig, ax = plt.subplots(figsize=(12, 6), facecolor="white")
    running = 0
    bottoms = []
    for i, (val, typ) in enumerate(zip(values, types)):
        if typ == "abs":
            bottom = 0
            running = val
        elif typ == "rel":
            bottom = running if val < 0 else running
            running += val
        else:
            bottom = 0
        bottoms.append(bottom)

    running = 0
    for i, (label, val, color, typ) in enumerate(
            zip(labels, values, colors, types)):
        if typ == "abs":
            ax.bar(i, val, color=color, alpha=0.85,
                   edgecolor="white", linewidth=1.5)
            running = val
        elif typ == "rel":
            if val < 0:
                ax.bar(i, abs(val), bottom=running + val,
                       color=color, alpha=0.85,
                       edgecolor="white", linewidth=1.5)
                # Connector
                ax.plot([i-0.4, i+0.4], [running, running],
                        color=GRAY2, linewidth=1, linestyle="--")
                running += val
            else:
                ax.bar(i, val, bottom=running,
                       color=color, alpha=0.85,
                       edgecolor="white", linewidth=1.5)
                running += val
        else:
            ax.bar(i, val, color=color, alpha=0.85,
                   edgecolor="white", linewidth=1.5)

        ax.text(i, (running if typ != "rel" else running) + gross * 0.01,
                f"R${abs(val):,.0f}",
                ha="center", fontsize=9,
                fontweight="bold", color=TEXT)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"R${v:,.0f}")
    )
    styled_ax(ax, "Revenue Waterfall — Gross to Net to Profit",
              ylabel="Amount (R$)")
    fig.tight_layout()
    return save(fig, "04_revenue_waterfall")


# ─────────────────────────────────────────────────────────────────────────────
# CHART 05 — REVENUE BY CHANNEL
# ─────────────────────────────────────────────────────────────────────────────

def chart_channel(dl: pd.DataFrame) -> Path:
    ch = (
        dl.groupby("channel_name")
        .agg(revenue=("revenue", "sum"),
             orders=("order_id", "count"))
        .reset_index()
        .sort_values("revenue", ascending=False)
    )
    palette = [PURPLE, TEAL, CORAL, AMBER, RED, GRAY]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                              facecolor="white")

    # Pie
    wedges, texts, autotexts = axes[0].pie(
        ch["revenue"],
        labels=ch["channel_name"],
        colors=palette[:len(ch)],
        autopct="%1.1f%%",
        startangle=140,
        pctdistance=0.75,
        wedgeprops=dict(edgecolor="white", linewidth=2),
    )
    for at in autotexts:
        at.set_fontsize(9)
        at.set_color("white")
        at.set_fontweight("bold")
    circle = plt.Circle((0, 0), 0.50, color="white")
    axes[0].add_artist(circle)
    styled_ax(axes[0], "Revenue Share by Channel")
    axes[0].set_facecolor("white")

    # Bar
    axes[1].barh(ch["channel_name"], ch["orders"],
                  color=palette[:len(ch)],
                  edgecolor="white", linewidth=0.5)
    for i, v in enumerate(ch["orders"]):
        axes[1].text(v + ch["orders"].max() * 0.01, i,
                     f"{v:,}", va="center", fontsize=9)
    axes[1].xaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v:,.0f}")
    )
    styled_ax(axes[1], "Orders by Channel",
              xlabel="Order Count")
    axes[1].invert_yaxis()

    fig.tight_layout()
    return save(fig, "05_channel_performance")


# ─────────────────────────────────────────────────────────────────────────────
# CHART 06 — LEAKAGE BAR
# ─────────────────────────────────────────────────────────────────────────────

def chart_leakage_bar(leakage: pd.DataFrame,
                       total_revenue: float) -> Path:
    leak   = leakage.sort_values(
        "leakage_amount", ascending=True
    ).copy()
    total  = leak["leakage_amount"].sum()
    colors = [RED, CORAL, AMBER, PURPLE2, TEAL, GRAY2][:len(leak)]

    # Shorten source labels
    leak["label"] = leak["source"].apply(
        lambda s: s.split(" — ")[-1] if " — " in s else s
    )

    fig, ax = plt.subplots(figsize=(14, 6), facecolor="white")
    bars = ax.barh(leak["label"], leak["leakage_amount"],
                   color=colors, edgecolor="white", linewidth=0.5)

    for bar, (_, row) in zip(bars, leak.iterrows()):
        ax.text(
            bar.get_width() + total * 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"R${row['leakage_amount']:,.0f}  ({row['pct_of_revenue']:.2f}%)",
            va="center", fontsize=10, color=TEXT
        )

    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"R${v:,.0f}")
    )
    ax.set_xlim(0, leak["leakage_amount"].max() * 1.45)

    # Total annotation
    ax.text(0.98, 0.97,
            f"Total Leakage\nR${total:,.0f}\n"
            f"({total/total_revenue*100:.1f}% of revenue)",
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=11, fontweight="bold",
            color=RED,
            bbox=dict(boxstyle="round,pad=0.5",
                      facecolor="#FFF0F0",
                      edgecolor=RED,
                      linewidth=1.5))

    styled_ax(ax, "Revenue Leakage by Source",
              xlabel="Leakage Amount (R$)")
    fig.tight_layout()
    return save(fig, "06_leakage_bar")


# ─────────────────────────────────────────────────────────────────────────────
# CHART 07 — TOP CATEGORIES
# ─────────────────────────────────────────────────────────────────────────────

def chart_top_categories(categories: pd.DataFrame) -> Path:
    top = categories.head(20).sort_values(
        "total_revenue", ascending=True
    )
    tier_colors = {
        "top performer":  PURPLE,
        "solid":          TEAL,
        "average":        AMBER,
        "underperformer": GRAY,
    }
    colors = [tier_colors.get(str(t), GRAY)
              for t in top["performance_tier"]]

    fig, ax = plt.subplots(figsize=(14, 9), facecolor="white")
    bars = ax.barh(top["category"], top["total_revenue"],
                   color=colors, edgecolor="white", linewidth=0.3)

    for bar, val in zip(bars, top["total_revenue"]):
        ax.text(bar.get_width() + top["total_revenue"].max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"R${val:,.0f}",
                va="center", fontsize=9, color=TEXT)

    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"R${v:,.0f}")
    )
    ax.set_xlim(0, top["total_revenue"].max() * 1.22)

    # Legend
    handles = [
        mpatches.Patch(color=PURPLE, label="Top performer"),
        mpatches.Patch(color=TEAL,   label="Solid"),
        mpatches.Patch(color=AMBER,  label="Average"),
        mpatches.Patch(color=GRAY,   label="Underperformer"),
    ]
    ax.legend(handles=handles, fontsize=9,
              loc="lower right")

    styled_ax(ax, "Top 20 Categories by Revenue",
              xlabel="Total Revenue (R$)")
    fig.tight_layout()
    return save(fig, "07_top_categories")


# ─────────────────────────────────────────────────────────────────────────────
# CHART 08 — QUADRANT SCATTER
# ─────────────────────────────────────────────────────────────────────────────

def chart_quadrant(categories: pd.DataFrame) -> Path:
    df = categories.dropna(
        subset=["total_orders", "gross_margin_pct"]
    ).copy()
    med_o = df["total_orders"].median()
    med_m = df["gross_margin_pct"].median()

    def quad(row):
        hi_o = row["total_orders"]     >= med_o
        hi_m = row["gross_margin_pct"] >= med_m
        if hi_o and hi_m:     return "Stars"
        if hi_o and not hi_m: return "Volume drivers"
        if not hi_o and hi_m: return "Niche high-margin"
        return "Underperformers"

    df["quad"]  = df.apply(quad, axis=1)
    q_colors    = {
        "Stars":             PURPLE,
        "Volume drivers":    TEAL,
        "Niche high-margin": AMBER,
        "Underperformers":   RED,
    }

    fig, ax = plt.subplots(figsize=(14, 9), facecolor="white")
    sizes = (
        (df["total_revenue"] - df["total_revenue"].min()) /
        (df["total_revenue"].max() - df["total_revenue"].min())
    ) * 400 + 30

    for q, color in q_colors.items():
        mask = df["quad"] == q
        ax.scatter(
            df[mask]["total_orders"],
            df[mask]["gross_margin_pct"],
            s=sizes[mask],
            color=color,
            alpha=0.70,
            edgecolors="white",
            linewidth=0.8,
            label=q,
            zorder=3,
        )
        for _, row in df[mask].iterrows():
            ax.annotate(
                row["category"][:14],
                (row["total_orders"], row["gross_margin_pct"]),
                fontsize=6.5,
                color=TEXT,
                xytext=(4, 4),
                textcoords="offset points",
            )

    ax.axvline(med_o, color=GRAY2, linewidth=1.2,
               linestyle="--", zorder=1)
    ax.axhline(med_m, color=GRAY2, linewidth=1.2,
               linestyle="--", zorder=1)

    x_max = df["total_orders"].max()
    y_max = df["gross_margin_pct"].max()
    for label, x, y, color in [
        ("STARS",        x_max*0.82, y_max*0.90, PURPLE),
        ("VOLUME",       x_max*0.82, y_max*0.08, TEAL),
        ("NICHE",        x_max*0.08, y_max*0.90, AMBER),
        ("UNDERPERFORM", x_max*0.08, y_max*0.08, RED),
    ]:
        ax.text(x, y, label, fontsize=13,
                fontweight="bold", color=color,
                alpha=0.25, ha="center")

    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v:.1f}%")
    )
    ax.legend(fontsize=9, loc="upper right")
    styled_ax(ax, "Volume vs Margin Quadrant Analysis",
              xlabel="Total Orders",
              ylabel="Gross Margin %")
    fig.tight_layout()
    return save(fig, "08_quadrant_scatter")


# ─────────────────────────────────────────────────────────────────────────────
# CHART 09 — REGIONAL PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────

def chart_regions(regions: pd.DataFrame) -> Path:
    top = regions.sort_values(
        "total_revenue", ascending=False
    ).head(15).sort_values("total_revenue", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7),
                              facecolor="white")

    # Revenue
    colors1 = plt.cm.get_cmap("Purples")(
        np.linspace(0.35, 0.85, len(top))
    )
    axes[0].barh(top["state"], top["total_revenue"],
                  color=colors1, edgecolor="white", linewidth=0.5)
    for i, v in enumerate(top["total_revenue"]):
        axes[0].text(v + top["total_revenue"].max()*0.01,
                     i, f"R${v:,.0f}",
                     va="center", fontsize=9)
    axes[0].xaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"R${v:,.0f}")
    )
    styled_ax(axes[0], "Revenue by State (Top 15)",
              xlabel="Total Revenue (R$)")

    # AOV
    colors2 = plt.cm.get_cmap("Greens")(
        np.linspace(0.35, 0.85, len(top))
    )
    axes[1].barh(top["state"], top["avg_order_value"],
                  color=colors2, edgecolor="white", linewidth=0.5)
    for i, v in enumerate(top["avg_order_value"]):
        axes[1].text(v + top["avg_order_value"].max()*0.01,
                     i, f"R${v:,.0f}",
                     va="center", fontsize=9)
    axes[1].xaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"R${v:,.0f}")
    )
    styled_ax(axes[1], "Avg Order Value by State",
              xlabel="Avg Order Value (R$)")

    fig.tight_layout()
    return save(fig, "09_regional_performance")


# ─────────────────────────────────────────────────────────────────────────────
# CHART 10 — RFM SEGMENTS
# ─────────────────────────────────────────────────────────────────────────────

def chart_rfm(clv_seg: pd.DataFrame) -> Path:
    seg    = clv_seg.copy()
    colors = [SEG_COLORS.get(s, GRAY) for s in seg["rfm_segment"]]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6),
                              facecolor="white")

    # Pie
    wedges, texts, autotexts = axes[0].pie(
        seg["customer_count"],
        labels=seg["rfm_segment"],
        colors=colors,
        autopct="%1.1f%%",
        startangle=140,
        pctdistance=0.78,
        wedgeprops=dict(edgecolor="white", linewidth=2),
    )
    for at in autotexts:
        at.set_fontsize(8)
        at.set_fontweight("bold")
    circle = plt.Circle((0, 0), 0.50, color="white")
    axes[0].add_artist(circle)
    styled_ax(axes[0], "Customer Count by Segment")
    axes[0].set_facecolor("white")

    # Avg CLV
    s2 = seg.sort_values("avg_clv", ascending=True)
    clr2 = [SEG_COLORS.get(s, GRAY) for s in s2["rfm_segment"]]
    axes[1].barh(s2["rfm_segment"], s2["avg_clv"],
                  color=clr2, edgecolor="white")
    for i, v in enumerate(s2["avg_clv"]):
        axes[1].text(v + s2["avg_clv"].max()*0.01, i,
                     f"R${v:,.0f}", va="center", fontsize=9)
    axes[1].xaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"R${v:,.0f}")
    )
    styled_ax(axes[1], "Avg CLV by Segment",
              xlabel="Avg CLV (R$)")

    # Avg Orders
    s3 = seg.sort_values("avg_orders", ascending=True)
    clr3 = [SEG_COLORS.get(s, GRAY) for s in s3["rfm_segment"]]
    axes[2].barh(s3["rfm_segment"], s3["avg_orders"],
                  color=clr3, edgecolor="white")
    for i, v in enumerate(s3["avg_orders"]):
        axes[2].text(v + s3["avg_orders"].max()*0.01, i,
                     f"{v:.1f}", va="center", fontsize=9)
    styled_ax(axes[2], "Avg Orders by Segment",
              xlabel="Avg Orders")

    fig.suptitle("RFM Segment Overview",
                 fontsize=14, fontweight="bold",
                 color=TEXT, y=1.01)
    fig.tight_layout()
    return save(fig, "10_rfm_segments")


# ─────────────────────────────────────────────────────────────────────────────
# CHART 11 — LTV:CAC
# ─────────────────────────────────────────────────────────────────────────────

def chart_ltv_cac(clv_ch: pd.DataFrame) -> Path:
    ch = clv_ch.dropna(subset=["ltv_cac_ratio"]).sort_values(
        "ltv_cac_ratio", ascending=True
    ).copy()

    r_colors = [
        TEAL if v >= 3 else (AMBER if v >= 1 else RED)
        for v in ch["ltv_cac_ratio"]
    ]
    p_colors = [
        TEAL if v <= 12 else (AMBER if v <= 24 else RED)
        for v in ch["payback_months"].fillna(999)
    ]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6),
                              facecolor="white")

    # Ratio
    bars = axes[0].barh(ch["channel_name"], ch["ltv_cac_ratio"],
                         color=r_colors, edgecolor="white")
    for bar, val in zip(bars, ch["ltv_cac_ratio"]):
        axes[0].text(bar.get_width() + 0.05,
                     bar.get_y() + bar.get_height()/2,
                     f"{val:.1f}x", va="center", fontsize=10,
                     fontweight="bold")
    axes[0].axvline(3, color=TEAL, linewidth=1.5,
                    linestyle="--", label="Healthy (3x)")
    axes[0].axvline(1, color=RED, linewidth=1.5,
                    linestyle="--", label="Break-even (1x)")
    axes[0].legend(fontsize=9)
    # Shade zones
    axes[0].axvspan(0, 1, alpha=0.07, color=RED)
    axes[0].axvspan(1, 3, alpha=0.05, color=AMBER)
    axes[0].axvspan(3, ch["ltv_cac_ratio"].max()*1.3,
                    alpha=0.05, color=TEAL)
    styled_ax(axes[0], "LTV:CAC Ratio by Channel",
              xlabel="LTV:CAC Ratio")

    # Payback
    bars2 = axes[1].barh(ch["channel_name"], ch["payback_months"],
                          color=p_colors, edgecolor="white")
    for bar, val in zip(bars2, ch["payback_months"]):
        if not pd.isna(val):
            axes[1].text(bar.get_width() + 0.5,
                         bar.get_y() + bar.get_height()/2,
                         f"{val:.0f}m", va="center", fontsize=10)
    styled_ax(axes[1], "Payback Period (months)",
              xlabel="Months")

    fig.tight_layout()
    return save(fig, "11_ltv_cac")


# ─────────────────────────────────────────────────────────────────────────────
# CHART 12 — CLV DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

def chart_clv_dist(clv: pd.DataFrame) -> Path:
    cap  = clv["clv"].quantile(0.95)
    data = clv[clv["clv"] <= cap]["clv"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6),
                              facecolor="white")

    # Histogram
    axes[0].hist(data, bins=60, color=PURPLE2,
                  edgecolor="white", linewidth=0.3, alpha=0.85)
    axes[0].axvline(clv["clv"].mean(), color=CORAL,
                    linewidth=2, linestyle="--",
                    label=f"Mean R${clv['clv'].mean():,.0f}")
    axes[0].axvline(clv["clv"].median(), color=TEAL,
                    linewidth=2, linestyle="--",
                    label=f"Median R${clv['clv'].median():,.0f}")
    axes[0].xaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"R${v:,.0f}")
    )
    axes[0].legend(fontsize=9)
    styled_ax(axes[0], "CLV Distribution (95th pct cap)",
              xlabel="CLV (R$)", ylabel="Customers")

    # By tier
    if "clv_tier" in clv.columns:
        tier_order  = ["platinum", "gold", "silver", "bronze", "low"]
        tier_colors = {
            "platinum": PURPLE, "gold": AMBER,
            "silver":   GRAY,   "bronze": CORAL,
            "low":      GRAY2,
        }
        ts = (
            clv.groupby("clv_tier")
            .agg(avg_clv=("clv", "mean"),
                 count=("clv", "count"))
            .reset_index()
        )
        ts = ts.set_index("clv_tier").reindex(
            [t for t in tier_order if t in ts.index]
        ).reset_index()

        bar_colors = [tier_colors.get(t, GRAY) for t in ts["clv_tier"]]
        bars = axes[1].bar(ts["clv_tier"], ts["avg_clv"],
                            color=bar_colors, edgecolor="white")
        for bar, (_, row) in zip(bars, ts.iterrows()):
            axes[1].text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + ts["avg_clv"].max()*0.02,
                f"R${row['avg_clv']:,.0f}\n({row['count']:,})",
                ha="center", fontsize=9, color=TEXT
            )
        axes[1].yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"R${v:,.0f}")
        )
        styled_ax(axes[1], "Avg CLV by Customer Tier",
                  xlabel="Tier", ylabel="Avg CLV (R$)")

    fig.tight_layout()
    return save(fig, "12_clv_distribution")


# ─────────────────────────────────────────────────────────────────────────────
# CHART 13 — PARETO CURVE
# ─────────────────────────────────────────────────────────────────────────────

def chart_pareto(clv: pd.DataFrame) -> Path:
    sc = clv.sort_values("total_revenue", ascending=False).copy()
    sc["cum_rev"]     = sc["total_revenue"].cumsum()
    total             = sc["total_revenue"].sum()
    sc["cum_rev_pct"] = sc["cum_rev"] / total * 100
    sc["cust_pct"]    = np.arange(1, len(sc)+1) / len(sc) * 100

    fig, ax = plt.subplots(figsize=(12, 8), facecolor="white")

    ax.fill_between(sc["cust_pct"], sc["cum_rev_pct"],
                    alpha=0.12, color=PURPLE)
    ax.plot(sc["cust_pct"], sc["cum_rev_pct"],
            color=PURPLE, linewidth=3,
            label="Revenue concentration")
    ax.plot([0, 100], [0, 100],
            color=GRAY2, linewidth=1.5,
            linestyle="--", label="Perfect equality")

    for pct in [5, 20, 50]:
        mask = sc["cust_pct"] <= pct
        if mask.any():
            rev = sc[mask]["cum_rev_pct"].max()
            ax.annotate(
                f"Top {pct}%\n→ {rev:.0f}%",
                xy=(pct, rev),
                xytext=(pct + 6, rev - 10),
                fontsize=10, fontweight="bold",
                color=PURPLE,
                arrowprops=dict(
                    arrowstyle="->",
                    color=PURPLE,
                    lw=1.5,
                ),
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor=PURPLE4,
                          edgecolor=PURPLE,
                          linewidth=1),
            )

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v:.0f}%")
    )
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v:.0f}%")
    )
    ax.legend(fontsize=10)
    styled_ax(ax, "Revenue Concentration — Pareto / Lorenz Curve",
              xlabel="% of Customers (ranked by revenue)",
              ylabel="Cumulative % of Revenue")
    fig.tight_layout()
    return save(fig, "13_pareto_curve")


# ─────────────────────────────────────────────────────────────────────────────
# CHART 14 — COHORT HEATMAP
# ─────────────────────────────────────────────────────────────────────────────

def chart_cohort_heatmap(cohort: pd.DataFrame) -> Path:
    c = cohort.copy()
    c.index = c.index.astype(str)

    int_cols = sorted(
        [col for col in c.columns
         if str(col).lstrip("-").isdigit()
         and 0 <= int(col) <= 12],
        key=lambda x: int(x)
    )
    c = c[int_cols].replace(0, np.nan)

    xlabels = [f"M+{int(col)}" if int(col) > 0 else "M0"
               for col in int_cols]

    height = max(8, len(c) * 0.38)
    fig, ax = plt.subplots(figsize=(16, height),
                            facecolor="white")

    mask = c.isnull()
    sns.heatmap(
        c,
        mask=mask,
        cmap=PURPLE_CMAP,
        vmin=0, vmax=12,
        annot=True,
        fmt=".1f",
        annot_kws={"size": 8},
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        cbar_kws={"label": "Retention %",
                   "shrink": 0.6},
        xticklabels=xlabels,
    )

    # Mark period 0 differently
    for j, col in enumerate(int_cols):
        if int(col) == 0:
            for i in range(len(c)):
                ax.add_patch(plt.Rectangle(
                    (j, i), 1, 1,
                    fill=True,
                    facecolor=PURPLE4,
                    edgecolor="white",
                    linewidth=0.5,
                    zorder=1,
                ))

    ax.set_title("Customer Retention Cohort Matrix\n"
                 "% of each monthly cohort returning in subsequent months",
                 fontsize=13, fontweight="bold",
                 color=TEXT, pad=14)
    ax.set_xlabel("Months since first purchase",
                  fontsize=10, color=SUBTEXT)
    ax.set_ylabel("Cohort (first purchase month)",
                  fontsize=10, color=SUBTEXT)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

    fig.tight_layout()
    return save(fig, "14_cohort_heatmap")


# ─────────────────────────────────────────────────────────────────────────────
# CHART 15 — RETENTION CURVES
# ─────────────────────────────────────────────────────────────────────────────

def chart_retention_curves(cohort: pd.DataFrame) -> Path:
    c = cohort.copy()
    c.index = c.index.astype(str)

    int_cols = sorted(
        [col for col in c.columns
         if str(col).lstrip("-").isdigit()
         and 1 <= int(col) <= 12],
        key=lambda x: int(x)
    )
    periods = [int(col) for col in int_cols]

    fig, ax = plt.subplots(figsize=(14, 7), facecolor="white")

    # Individual curves
    for idx in c.index:
        vals = [c.loc[idx, col] for col in int_cols]
        pts  = [(p, v) for p, v in zip(periods, vals)
                if v is not None
                and not pd.isna(v) and v > 0]
        if len(pts) >= 2:
            px_, py_ = zip(*pts)
            ax.plot(list(px_), list(py_),
                    color=PURPLE3, linewidth=1,
                    alpha=0.4)

    # Average
    avg_vals = []
    for col in int_cols:
        v = c[col].replace(0, np.nan).dropna()
        avg_vals.append(v.mean() if len(v) > 0 else None)

    ax.plot(periods, avg_vals,
            color=PURPLE, linewidth=3.5,
            marker="o", markersize=8,
            markerfacecolor=PURPLE,
            markeredgecolor="white",
            markeredgewidth=2,
            label="Average", zorder=5)

    # Annotate
    for p in [1, 3, 6]:
        if p in periods:
            v = avg_vals[periods.index(p)]
            if v is not None:
                ax.annotate(
                    f"M+{p}: {v:.1f}%",
                    (p, v),
                    xytext=(p + 0.3, v + 0.4),
                    fontsize=10, fontweight="bold",
                    color=PURPLE,
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor=PURPLE4,
                              edgecolor=PURPLE,
                              linewidth=1),
                )

    ax.set_xticks(periods)
    ax.set_xlabel("Months since first purchase",
                  fontsize=10, color=SUBTEXT)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v:.1f}%")
    )

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=PURPLE3, linewidth=1,
               alpha=0.6, label="Individual cohorts"),
        Line2D([0], [0], color=PURPLE, linewidth=3,
               label="Average"),
    ]
    ax.legend(handles=legend_elements, fontsize=10)
    styled_ax(ax, "Retention Curves by Cohort",
              xlabel="Months since first purchase",
              ylabel="% of Cohort Retained")
    fig.tight_layout()
    return save(fig, "15_retention_curves")


# ─────────────────────────────────────────────────────────────────────────────
# CHART 16 — CHURN PROBABILITY DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

def chart_churn_dist(churn: pd.DataFrame) -> Path:
    active  = churn[churn["churn_flag"] == 0]["churn_probability"]
    churned = churn[churn["churn_flag"] == 1]["churn_probability"]

    fig, ax = plt.subplots(figsize=(13, 6), facecolor="white")

    ax.hist(active, bins=50, color=TEAL, alpha=0.70,
            edgecolor="white", linewidth=0.3,
            label=f"Active (n={len(active):,})")
    ax.hist(churned, bins=50, color=RED, alpha=0.70,
            edgecolor="white", linewidth=0.3,
            label=f"Churned (n={len(churned):,})")
    ax.axvline(0.5, color=TEXT, linewidth=2,
               linestyle="--",
               label="Decision threshold (0.5)")

    ax.set_xlim(0, 1)
    ax.legend(fontsize=10)
    styled_ax(ax, "Churn Probability Distribution\n"
              "Clear separation between Active and Churned = good model",
              xlabel="Predicted Churn Probability",
              ylabel="Number of Customers")
    fig.tight_layout()
    return save(fig, "16_churn_distribution")


# ─────────────────────────────────────────────────────────────────────────────
# CHART 17 — CHURN RISK TIERS
# ─────────────────────────────────────────────────────────────────────────────

def chart_churn_tiers(churn: pd.DataFrame) -> Path:
    ts = (
        churn.groupby("churn_risk_tier", observed=True)
        .agg(count=("customer_unique_id", "count"),
             actual_churn=("churn_flag", "mean"))
        .reset_index()
    )
    ts = ts.set_index("churn_risk_tier").reindex(
        [t for t in ["low", "medium", "high"]
         if t in ts.index]
    ).reset_index()

    tier_colors = {
        "low": TEAL, "medium": AMBER, "high": RED
    }
    colors = [tier_colors.get(t, GRAY)
              for t in ts["churn_risk_tier"]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                              facecolor="white")

    # Count
    bars1 = axes[0].bar(ts["churn_risk_tier"], ts["count"],
                         color=colors, edgecolor="white",
                         linewidth=0.5)
    for bar, val in zip(bars1, ts["count"]):
        axes[0].text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + ts["count"].max()*0.02,
            f"{val:,}", ha="center",
            fontsize=11, fontweight="bold"
        )
    styled_ax(axes[0], "Customer Count by Churn Risk Tier",
              xlabel="Risk Tier",
              ylabel="Customers")

    # Churn rate
    bars2 = axes[1].bar(ts["churn_risk_tier"],
                         ts["actual_churn"] * 100,
                         color=colors, edgecolor="white",
                         linewidth=0.5)
    for bar, val in zip(bars2, ts["actual_churn"]):
        axes[1].text(
            bar.get_x() + bar.get_width()/2,
            val*100 + ts["actual_churn"].max()*2,
            f"{val*100:.1f}%", ha="center",
            fontsize=11, fontweight="bold"
        )
    axes[1].yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v:.0f}%")
    )
    styled_ax(axes[1], "Actual Churn Rate by Risk Tier",
              xlabel="Risk Tier",
              ylabel="Churn Rate %")

    fig.tight_layout()
    return save(fig, "17_churn_risk_tiers")


# ─────────────────────────────────────────────────────────────────────────────
# CHART 18 — CHURN BY RFM SEGMENT
# ─────────────────────────────────────────────────────────────────────────────

def chart_churn_rfm(churn: pd.DataFrame) -> Path:
    high = churn[churn["churn_risk_tier"] == "high"]
    seg  = (
        high.groupby("rfm_segment")
        .agg(count=("customer_unique_id", "count"),
             rev_at_risk=("total_revenue", "sum"))
        .reset_index()
        .sort_values("count", ascending=True)
    )
    seg_colors = [SEG_COLORS.get(s, GRAY)
                  for s in seg["rfm_segment"]]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6),
                              facecolor="white")

    # Count
    bars1 = axes[0].barh(seg["rfm_segment"], seg["count"],
                          color=seg_colors, edgecolor="white")
    for bar, val in zip(bars1, seg["count"]):
        axes[0].text(bar.get_width() + seg["count"].max()*0.01,
                     bar.get_y() + bar.get_height()/2,
                     f"{val:,}", va="center", fontsize=10)
    styled_ax(axes[0], "High-Risk Count by RFM Segment",
              xlabel="Customers")

    # Revenue at risk
    bars2 = axes[1].barh(seg["rfm_segment"], seg["rev_at_risk"],
                          color=seg_colors, edgecolor="white")
    for bar, val in zip(bars2, seg["rev_at_risk"]):
        axes[1].text(
            bar.get_width() + seg["rev_at_risk"].max()*0.01,
            bar.get_y() + bar.get_height()/2,
            f"R${val:,.0f}", va="center", fontsize=10
        )
    axes[1].xaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"R${v:,.0f}")
    )
    styled_ax(axes[1], "Revenue at Risk by RFM Segment",
              xlabel="Revenue (R$)")

    fig.suptitle("High-Risk Customers by RFM Segment",
                 fontsize=14, fontweight="bold",
                 color=TEXT, y=1.01)
    fig.tight_layout()
    return save(fig, "18_churn_by_rfm")


# ─────────────────────────────────────────────────────────────────────────────
# COMPILE PDF
# ─────────────────────────────────────────────────────────────────────────────

def compile_pdf(chart_paths: list, k: dict) -> Path:
    from fpdf import FPDF

    def s(text: str) -> str:
        """Strip any character that latin-1 cannot encode."""
        return text.encode("latin-1", errors="ignore").decode("latin-1")

    print("\nCompiling PDF...")

    class PDF(FPDF):
        def header(self):
            self.set_fill_color(83, 74, 183)
            self.rect(0, 0, 210, 18, "F")
            self.set_font("Helvetica", "B", 11)
            self.set_text_color(255, 255, 255)
            self.set_y(4)
            self.cell(0, 10,
                s("E-Commerce Revenue Intelligence Dashboard"),
                align="C")
            self.set_text_color(0, 0, 0)

        def footer(self):
            self.set_y(-12)
            self.set_font("Helvetica", "", 8)
            self.set_text_color(136, 135, 128)
            self.cell(0, 10,
                s("Page " + str(self.page_no()) +
                  " - Olist Brazilian E-Commerce - "
                  "Python, pandas, scikit-learn, matplotlib"),
                align="C")
            self.set_text_color(0, 0, 0)

    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(10, 20, 10)

    # Cover page
    pdf.add_page()
    pdf.set_fill_color(83, 74, 183)
    pdf.rect(0, 0, 210, 297, "F")

    pdf.set_font("Helvetica", "B", 28)
    pdf.set_text_color(255, 255, 255)
    pdf.set_y(60)
    pdf.cell(0, 14, s("E-Commerce Revenue"), align="C", ln=True)
    pdf.cell(0, 14, s("Intelligence Dashboard"), align="C", ln=True)

    pdf.set_font("Helvetica", "", 14)
    pdf.set_text_color(175, 169, 236)
    pdf.ln(8)
    pdf.cell(0, 10, s("Olist Brazilian E-Commerce | 2016-2018"),
             align="C", ln=True)
    pdf.cell(0, 10, s("End-to-End Analytics Pipeline"),
             align="C", ln=True)

    pdf.ln(16)
    kpi_items = [
        ("Total Revenue",    s(f"R${k['total_revenue']:,.0f}")),
        ("Total Orders",     s(f"{k['total_orders']:,}")),
        ("Avg Order Value",  s(f"R${k['aov']:,.2f}")),
        ("Gross Margin",     s(f"{k['gross_margin']:.1f}%")),
        ("Revenue Leakage",  s(f"R${k['total_leak']:,.0f}")),
        ("Churn Rate",       s(f"{k['churn_rate']:.1f}%")),
    ]
    col_w   = 60
    x_start = (210 - col_w * 3) / 2

    for i, (label, value) in enumerate(kpi_items):
        col = i % 3
        row = i // 3
        x   = x_start + col * col_w
        y   = 170 + row * 32

        pdf.set_fill_color(255, 255, 255)
        pdf.set_draw_color(175, 169, 236)
        pdf.set_line_width(0.3)
        pdf.rect(x, y, col_w - 4, 28, "FD")

        pdf.set_xy(x, y + 4)
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_text_color(83, 74, 183)
        pdf.cell(col_w - 4, 8, s(value), align="C", ln=True)

        pdf.set_xy(x, y + 14)
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(89, 90, 90)
        pdf.cell(col_w - 4, 6, s(label), align="C")

    pdf.set_xy(0, 265)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(175, 169, 236)
    pdf.cell(0, 8,
        s("Built with Python, pandas, scikit-learn, "
          "matplotlib, seaborn"),
        align="C")

    # Chart pages
    sections = [
        ("Executive Summary",        [0]),
        ("Revenue Trends",            [1, 2, 3]),
        ("Channel Performance",       [4]),
        ("Revenue Leakage",           [5]),
        ("Product Performance",       [6, 7]),
        ("Regional Performance",      [8]),
        ("Customer Segmentation",     [9]),
        ("LTV CAC Analysis",          [10]),
        ("CLV Distribution",          [11]),
        ("Pareto Curve",              [12]),
        ("Cohort Retention Matrix",   [13]),
        ("Retention Curves",          [14]),
        ("Churn Distribution",        [15]),
        ("Churn Risk Tiers",          [16]),
        ("Churn by Segment",          [17]),
    ]

    for section_name, indices in sections:
        for i, idx in enumerate(indices):
            if idx < len(chart_paths):
                path = chart_paths[idx]
                if path.exists():
                    pdf.add_page()
                    pdf.set_fill_color(248, 247, 255)
                    pdf.rect(0, 18, 210, 10, "F")
                    pdf.set_xy(10, 19)
                    pdf.set_font("Helvetica", "B", 11)
                    pdf.set_text_color(83, 74, 183)
                    pdf.cell(0, 8, s(section_name))
                    pdf.set_text_color(0, 0, 0)
                    pdf.image(str(path), x=10, y=30, w=190)

    out = REPORTS / "dashboard.pdf"
    pdf.output(str(out))
    size_mb = out.stat().st_size / 1024 / 1024
    print(f"  PDF saved -> {out}  ({size_mb:.1f} MB)")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  PHASE 9 — DASHBOARD GENERATION")
    print("="*60)

    d = load_data()
    k = compute_kpis(d)

    print("\nGenerating charts...")
    paths = [
        chart_kpi_summary(k),
        chart_monthly_revenue(d["monthly"]),
        chart_mom_growth(d["monthly"]),
        chart_waterfall(k),
        chart_channel(d["del"]),
        chart_leakage_bar(d["leakage"], k["total_revenue"]),
        chart_top_categories(d["categories"]),
        chart_quadrant(d["categories"]),
        chart_regions(d["regions"]),
        chart_rfm(d["clv_seg"]),
        chart_ltv_cac(d["clv_ch"]),
        chart_clv_dist(d["clv"]),
        chart_pareto(d["clv"]),
        chart_cohort_heatmap(d["cohort"]),
        chart_retention_curves(d["cohort"]),
        chart_churn_dist(d["churn"]),
        chart_churn_tiers(d["churn"]),
        chart_churn_rfm(d["churn"]),
    ]

    pdf_path = compile_pdf(paths, k)

    print(f"\n{'='*60}")
    print(f"  Done.")
    print(f"  Charts: {len(paths)} PNGs in reports/charts/")
    print(f"  PDF:    {pdf_path}")
    print(f"\n  Open: reports\\dashboard.pdf")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()