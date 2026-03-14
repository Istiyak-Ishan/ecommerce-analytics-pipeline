"""
Phase 9 — Professional Analytics Dashboard
============================================
Single HTML file. Open in Chrome or Firefox.

Run: python notebooks/08_dashboard.py
Out: reports/dashboard.html
"""

import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
from pathlib import Path

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

PROCESSED = Path("data/processed")
ANALYTICS  = Path("data/processed/analytics")
REPORTS    = Path("reports")
REPORTS.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# COLORS
# ─────────────────────────────────────────────────────────────────────────────

C = {
    "purple":      "#534AB7",
    "purple_mid":  "#7F77DD",
    "purple_lite": "#AFA9EC",
    "purple_pale": "#EEEDFE",
    "teal":        "#1D9E75",
    "teal_lite":   "#9FE1CB",
    "coral":       "#D85A30",
    "amber":       "#EF9F27",
    "red":         "#E24B4A",
    "red_lite":    "#F7C1C1",
    "gray":        "#888780",
    "gray_lite":   "#D3D1C7",
    "bg":          "#F8F7FF",
    "card":        "#FFFFFF",
    "text":        "#2C2C2A",
    "subtext":     "#5F5E5A",
    "border":      "#E8E6F0",
}

PURPLE_SCALE = [
    [0.00, "#FFFFFF"],
    [0.05, "#EEEDFE"],
    [0.20, "#AFA9EC"],
    [0.50, "#7F77DD"],
    [0.75, "#534AB7"],
    [1.00, "#26215C"],
]

SEG_COLORS = {
    "champions":       C["purple"],
    "loyal":           C["purple_mid"],
    "potential_loyal": C["purple_lite"],
    "need_attention":  C["amber"],
    "at_risk":         C["coral"],
    "lost":            C["red"],
    "unknown":         C["gray"],
}

def L(extra: dict = None) -> dict:
    """Base layout — no margin or legend so charts can set their own."""
    base = dict(
        plot_bgcolor=C["card"],
        paper_bgcolor=C["card"],
        font=dict(family="'Segoe UI', Arial, sans-serif",
                  color=C["text"], size=12),
        hovermode="closest",
    )
    if extra:
        base.update(extra)
    return base


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

    # Parse timestamps
    d["orders"]["order_purchase_ts"] = pd.to_datetime(
        d["orders"]["order_purchase_ts"], errors="coerce"
    )

    # Delivered orders
    d["del"] = d["orders"][
        d["orders"]["order_status"] == "delivered"
    ].copy()

    # Compute rolling avg if missing
    if "rolling_3m_avg" not in d["monthly"].columns:
        d["monthly"]["rolling_3m_avg"] = (
            d["monthly"]["monthly_revenue"]
            .rolling(3, min_periods=1).mean()
        )

    # Compute gross_margin_pct for categories if missing
    if "gross_margin_pct" not in d["categories"].columns:
        d["categories"]["gross_margin_pct"] = (
            d["categories"]["total_gross_profit"] /
            d["categories"]["total_revenue"] * 100
        ).round(2)

    # cohort: set cohort_month as index
    if "cohort_month" in d["cohort"].columns:
        d["cohort"] = d["cohort"].set_index("cohort_month")

    print(f"  Orders:           {len(d['orders']):,}")
    print(f"  Delivered:        {len(d['del']):,}")
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
    # master_analytical uses customer_id
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

    churn_rate   = churn["churn_flag"].mean() * 100
    high_risk_n  = len(churn[churn["churn_risk_tier"] == "high"])
    rev_at_risk  = churn[
        churn["churn_risk_tier"] == "high"
    ]["total_revenue"].sum()

    total_leak   = leak["leakage_amount"].sum()
    leak_pct     = total_leak / total_revenue * 100
    repeat_rate  = (clv["total_orders"] > 1).mean() * 100
    avg_clv      = clv["clv"].mean()

    k = dict(
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

    print("\nKPIs:")
    for key, val in k.items():
        if isinstance(val, float):
            print(f"  {key:<22} {val:>14,.2f}")
        else:
            print(f"  {key:<22} {val:>14,}")
    return k


# ─────────────────────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────────────────────

def c_monthly(monthly: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Bar(
        x=monthly["year_month"],
        y=monthly["monthly_revenue"],
        name="Monthly Revenue",
        marker=dict(
            color=monthly["monthly_revenue"],
            colorscale=[[0, C["purple_pale"]], [1, C["purple"]]],
            showscale=False,
        ),
        hovertemplate="<b>%{x}</b><br>R$%{y:,.0f}<extra></extra>",
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=monthly["year_month"],
        y=monthly["rolling_3m_avg"],
        name="3M Avg",
        mode="lines",
        line=dict(color=C["coral"], width=3),
        hovertemplate="<b>%{x}</b><br>3M Avg: R$%{y:,.0f}<extra></extra>",
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=monthly["year_month"],
        y=monthly["cumulative_revenue"],
        name="Cumulative",
        mode="lines",
        line=dict(color=C["teal"], width=2, dash="dot"),
        hovertemplate="<b>%{x}</b><br>Cumulative: R$%{y:,.0f}<extra></extra>",
    ), secondary_y=True)

    fig.update_layout(L({
        "title": dict(text="<b>Monthly Revenue Trend</b>",
                      font=dict(size=15)),
        "xaxis": dict(showgrid=False, tickangle=45,
                      tickfont=dict(size=10)),
        "yaxis": dict(title="Revenue (R$)", gridcolor=C["border"],
                      tickprefix="R$", tickformat=",.0f"),
        "yaxis2": dict(title="Cumulative (R$)", showgrid=False,
                       tickprefix="R$", tickformat=",.0f"),
        "legend": dict(orientation="h", y=1.12, x=0),
        "margin": dict(t=70, b=60, l=70, r=60),
        "height": 420,
    }))
    return fig


def c_mom(monthly: pd.DataFrame) -> go.Figure:
    df     = monthly.dropna(subset=["mom_growth_pct"]).copy()
    colors = [C["teal"] if v >= 0 else C["red"]
              for v in df["mom_growth_pct"]]

    fig = go.Figure(go.Bar(
        x=df["year_month"],
        y=df["mom_growth_pct"],
        marker_color=colors,
        text=[f"{v:+.1f}%" for v in df["mom_growth_pct"]],
        textposition="outside",
        textfont=dict(size=9),
        hovertemplate="<b>%{x}</b><br>%{y:+.2f}%<extra></extra>",
    ))
    fig.add_hline(y=0, line_color=C["text"], line_width=1.5)

    fig.update_layout(L({
        "title": dict(
            text="<b>Month-over-Month Growth</b>  "
                 "<sup>Green = growth | Red = decline</sup>",
            font=dict(size=15)),
        "xaxis": dict(showgrid=False, tickangle=45,
                      tickfont=dict(size=10)),
        "yaxis": dict(gridcolor=C["border"], ticksuffix="%"),
        "showlegend": False,
        "margin": dict(t=70, b=60, l=60, r=40),
        "height": 360,
    }))
    return fig


def c_waterfall(k: dict) -> go.Figure:
    gross = k["total_revenue"] + k["total_disc"] + k["total_freight"]

    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["absolute", "relative", "relative", "total", "total"],
        x=["Gross Revenue", "– Discounts", "– Freight",
           "Net Revenue", "Gross Profit"],
        y=[gross, -k["total_disc"], -k["total_freight"],
           k["total_revenue"], k["gross_profit"]],
        text=[f"R${abs(v):,.0f}" for v in [
            gross, -k["total_disc"], -k["total_freight"],
            k["total_revenue"], k["gross_profit"]
        ]],
        textposition="outside",
        connector=dict(line=dict(color=C["border"], width=1.5)),
        increasing=dict(marker=dict(color=C["teal"])),
        decreasing=dict(marker=dict(color=C["red"])),
        totals=dict(marker=dict(color=C["purple"])),
        hovertemplate="<b>%{x}</b><br>R$%{y:,.0f}<extra></extra>",
    ))
    fig.update_layout(L({
        "title": dict(
            text="<b>Revenue Waterfall</b>  "
                 "<sup>Gross → Discounts → Freight → Net → Profit</sup>",
            font=dict(size=15)),
        "yaxis": dict(gridcolor=C["border"],
                      tickprefix="R$", tickformat=",.0f"),
        "xaxis": dict(showgrid=False),
        "showlegend": False,
        "margin": dict(t=70, b=50, l=80, r=50),
        "height": 400,
    }))
    return fig


def c_channel(dl: pd.DataFrame) -> go.Figure:
    # master_analytical has channel_name directly
    ch = (
        dl.groupby("channel_name")
        .agg(revenue=("revenue", "sum"),
             orders=("order_id", "count"))
        .reset_index()
        .sort_values("revenue", ascending=False)
    )
    palette = [C["purple"], C["teal"], C["coral"],
               C["amber"], C["red"], C["gray"]]

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "domain"}, {"type": "bar"}]],
        subplot_titles=["Revenue Share %", "Orders by Channel"],
    )
    fig.add_trace(go.Pie(
        labels=ch["channel_name"],
        values=ch["revenue"],
        hole=0.50,
        marker=dict(colors=palette[:len(ch)]),
        textinfo="label+percent",
        textfont=dict(size=11),
        hovertemplate="<b>%{label}</b><br>"
                      "R$%{value:,.0f}<br>%{percent}<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        y=ch["channel_name"],
        x=ch["orders"],
        orientation="h",
        marker_color=palette[:len(ch)],
        text=ch["orders"],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>%{x:,}<extra></extra>",
    ), row=1, col=2)

    fig.update_layout(L({
        "title": dict(text="<b>Revenue and Orders by Channel</b>",
                      font=dict(size=15)),
        "showlegend": False,
        "margin": dict(t=70, b=50, l=60, r=40),
        "height": 380,
    }))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig


def c_leakage_bar(leakage: pd.DataFrame,
                   total_revenue: float) -> go.Figure:
    leak   = leakage.sort_values("leakage_amount",
                                  ascending=True).copy()
    total  = leak["leakage_amount"].sum()
    colors = [C["red"], C["coral"], C["amber"],
              C["purple_mid"], C["teal"],
              C["gray_lite"]][:len(leak)]

    fig = go.Figure(go.Bar(
        x=leak["leakage_amount"],
        y=leak["source"],
        orientation="h",
        marker=dict(color=colors,
                    line=dict(color="white", width=1)),
        text=[f"  R${v:,.0f}  ({p:.2f}%)"
              for v, p in zip(leak["leakage_amount"],
                              leak["pct_of_revenue"])],
        textposition="outside",
        textfont=dict(size=11),
        hovertemplate="<b>%{y}</b><br>"
                      "R$%{x:,.0f}<extra></extra>",
    ))
    fig.add_annotation(
        x=leak["leakage_amount"].max() * 0.98,
        y=len(leak) - 0.5,
        text=f"<b>Total: R${total:,.0f}<br>"
             f"({total/total_revenue*100:.1f}% of revenue)</b>",
        showarrow=False,
        bgcolor=C["red_lite"],
        bordercolor=C["red"],
        borderwidth=1, borderpad=8,
        font=dict(size=12, color=C["red"]),
    )
    fig.update_layout(L({
        "title": dict(
            text="<b>Revenue Leakage by Source</b>  "
                 "<sup>Recoverable revenue identified</sup>",
            font=dict(size=15)),
        "xaxis": dict(tickprefix="R$", tickformat=",.0f",
                      showgrid=False),
        "yaxis": dict(showgrid=False, tickfont=dict(size=11)),
        "showlegend": False,
        "margin": dict(t=70, b=50, l=240, r=230),
        "height": 400,
    }))
    return fig


def c_leakage_sun(leakage: pd.DataFrame,
                   total_revenue: float) -> go.Figure:
    total_leak = leakage["leakage_amount"].sum()
    retained   = total_revenue - total_leak
    sources    = leakage["source"].str.split(" — ").str[-1].tolist()
    amounts    = leakage["leakage_amount"].tolist()

    labels  = ["Total Revenue"] + sources + ["Retained"]
    parents = [""] + ["Total Revenue"] * len(leakage) + ["Total Revenue"]
    values  = [total_revenue] + amounts + [retained]
    colors  = ([C["purple"]] +
               [C["red"], C["coral"], C["amber"],
                C["purple_mid"], C["teal"],
                C["gray"]][:len(leakage)] +
               [C["teal"]])

    fig = go.Figure(go.Sunburst(
        labels=labels, parents=parents,
        values=values, branchvalues="total",
        marker=dict(colors=colors),
        hovertemplate="<b>%{label}</b><br>"
                      "R$%{value:,.0f}<br>"
                      "%{percentParent:.1%}<extra></extra>",
        textfont=dict(size=11),
    ))
    fig.update_layout(L({
        "title": dict(
            text="<b>Revenue Composition</b>  "
                 "<sup>Leakage sources vs retained</sup>",
            font=dict(size=15)),
        "margin": dict(t=70, b=20, l=20, r=20),
        "height": 400,
    }))
    return fig


def c_top_cat(categories: pd.DataFrame) -> go.Figure:
    # column name is 'category' (not category_english)
    top = categories.head(20).sort_values(
        "total_revenue", ascending=True
    )
    tier_colors = {
        "top performer":  C["purple"],
        "solid":          C["teal"],
        "average":        C["amber"],
        "underperformer": C["gray"],
    }
    colors = [tier_colors.get(str(t), C["gray"])
              for t in top["performance_tier"]]

    fig = go.Figure(go.Bar(
        x=top["total_revenue"],
        y=top["category"],
        orientation="h",
        marker=dict(color=colors,
                    line=dict(color="white", width=0.5)),
        text=[f"R${v:,.0f}" for v in top["total_revenue"]],
        textposition="outside",
        textfont=dict(size=10),
        hovertemplate="<b>%{y}</b><br>"
                      "R$%{x:,.0f}<extra></extra>",
    ))
    fig.update_layout(L({
        "title": dict(
            text="<b>Top 20 Categories by Revenue</b>  "
                 "<sup>Color = performance tier</sup>",
            font=dict(size=15)),
        "xaxis": dict(tickprefix="R$", tickformat=",.0f",
                      showgrid=False),
        "yaxis": dict(showgrid=False, tickfont=dict(size=10)),
        "showlegend": False,
        "margin": dict(t=70, b=50, l=210, r=160),
        "height": 580,
    }))
    return fig


def c_quadrant(categories: pd.DataFrame) -> go.Figure:
    # uses: total_orders, gross_margin_pct, total_revenue,
    #       category, performance_tier
    df = categories.dropna(
        subset=["total_orders", "gross_margin_pct"]
    ).copy()
    med_o = df["total_orders"].median()
    med_m = df["gross_margin_pct"].median()

    def quad(row):
        hi_o = row["total_orders"]    >= med_o
        hi_m = row["gross_margin_pct"] >= med_m
        if hi_o and hi_m:     return "Stars"
        if hi_o and not hi_m: return "Volume drivers"
        if not hi_o and hi_m: return "Niche high-margin"
        return "Underperformers"

    df["quad"] = df.apply(quad, axis=1)
    q_colors = {
        "Stars":             C["purple"],
        "Volume drivers":    C["teal"],
        "Niche high-margin": C["amber"],
        "Underperformers":   C["red"],
    }

    fig = go.Figure()
    for q, color in q_colors.items():
        sub = df[df["quad"] == q]
        fig.add_trace(go.Scatter(
            x=sub["total_orders"],
            y=sub["gross_margin_pct"],
            mode="markers+text",
            name=q,
            marker=dict(
                color=color,
                size=sub["total_revenue"] /
                     sub["total_revenue"].max() * 38 + 7,
                opacity=0.72,
                line=dict(color="white", width=1),
            ),
            text=sub["category"].str[:14],
            textposition="top center",
            textfont=dict(size=8),
            hovertemplate="<b>%{text}</b><br>"
                          "Orders: %{x:,}<br>"
                          "Margin: %{y:.1f}%<extra></extra>",
        ))

    fig.add_vline(x=med_o, line_dash="dash",
                  line_color=C["gray"], line_width=1)
    fig.add_hline(y=med_m, line_dash="dash",
                  line_color=C["gray"], line_width=1)

    x_max = df["total_orders"].max()
    y_max = df["gross_margin_pct"].max()
    for label, x, y, col in [
        ("STARS",        x_max*0.82, y_max*0.92, C["purple"]),
        ("VOLUME",       x_max*0.82, y_max*0.08, C["teal"]),
        ("NICHE",        x_max*0.08, y_max*0.92, C["amber"]),
        ("UNDERPERFORM", x_max*0.08, y_max*0.08, C["red"]),
    ]:
        fig.add_annotation(
            x=x, y=y, text=f"<b>{label}</b>",
            showarrow=False,
            font=dict(size=11, color=col),
            opacity=0.35,
        )

    fig.update_layout(L({
        "title": dict(
            text="<b>Volume vs Margin Quadrant</b>  "
                 "<sup>Bubble = revenue | Top-right = Stars</sup>",
            font=dict(size=15)),
        "xaxis": dict(title="Total Orders",
                      showgrid=True, gridcolor=C["border"]),
        "yaxis": dict(title="Gross Margin %",
                      showgrid=True, gridcolor=C["border"],
                      ticksuffix="%"),
        "legend": dict(orientation="h", y=-0.14),
        "margin": dict(t=70, b=80, l=60, r=40),
        "height": 520,
    }))
    return fig


def c_regions(regions: pd.DataFrame) -> go.Figure:
    # columns: state, total_revenue, avg_order_value
    top = regions.sort_values(
        "total_revenue", ascending=False
    ).head(15).sort_values("total_revenue", ascending=True)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Revenue by State (Top 15)",
                         "Avg Order Value by State"],
        horizontal_spacing=0.12,
    )
    fig.add_trace(go.Bar(
        y=top["state"], x=top["total_revenue"],
        orientation="h",
        marker=dict(
            color=top["total_revenue"],
            colorscale=[[0, C["purple_pale"]], [1, C["purple"]]],
            showscale=False,
        ),
        text=[f"R${v:,.0f}" for v in top["total_revenue"]],
        textposition="outside",
        textfont=dict(size=10),
        hovertemplate="<b>%{y}</b><br>R$%{x:,.0f}<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        y=top["state"], x=top["avg_order_value"],
        orientation="h",
        marker=dict(
            color=top["avg_order_value"],
            colorscale=[[0, C["teal_lite"]], [1, C["teal"]]],
            showscale=False,
        ),
        text=[f"R${v:,.0f}" for v in top["avg_order_value"]],
        textposition="outside",
        textfont=dict(size=10),
        hovertemplate="<b>%{y}</b><br>R$%{x:,.0f}<extra></extra>",
    ), row=1, col=2)

    fig.update_layout(L({
        "title": dict(text="<b>Regional Performance</b>",
                      font=dict(size=15)),
        "showlegend": False,
        "margin": dict(t=70, b=50, l=60, r=100),
        "height": 460,
    }))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig


def c_rfm(clv_seg: pd.DataFrame) -> go.Figure:
    # columns: rfm_segment, customer_count, avg_clv, avg_orders
    seg    = clv_seg.copy()
    colors = [SEG_COLORS.get(s, C["gray"]) for s in seg["rfm_segment"]]

    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{"type": "domain"}, {"type": "bar"}, {"type": "bar"}]],
        subplot_titles=["Customer Count", "Avg CLV (R$)", "Avg Orders"],
    )
    fig.add_trace(go.Pie(
        labels=seg["rfm_segment"],
        values=seg["customer_count"],
        hole=0.50,
        marker=dict(colors=colors),
        textinfo="percent",
        textfont=dict(size=10),
        hovertemplate="<b>%{label}</b><br>"
                      "%{value:,}<br>%{percent}<extra></extra>",
    ), row=1, col=1)

    s2 = seg.sort_values("avg_clv", ascending=True)
    fig.add_trace(go.Bar(
        y=s2["rfm_segment"], x=s2["avg_clv"],
        orientation="h",
        marker_color=[SEG_COLORS.get(s, C["gray"])
                      for s in s2["rfm_segment"]],
        text=[f"R${v:,.0f}" for v in s2["avg_clv"]],
        textposition="outside",
        textfont=dict(size=10),
        hovertemplate="<b>%{y}</b><br>R$%{x:,.0f}<extra></extra>",
    ), row=1, col=2)

    s3 = seg.sort_values("avg_orders", ascending=True)
    fig.add_trace(go.Bar(
        y=s3["rfm_segment"], x=s3["avg_orders"],
        orientation="h",
        marker_color=[SEG_COLORS.get(s, C["gray"])
                      for s in s3["rfm_segment"]],
        text=[f"{v:.1f}" for v in s3["avg_orders"]],
        textposition="outside",
        textfont=dict(size=10),
        hovertemplate="<b>%{y}</b><br>%{x:.1f}<extra></extra>",
    ), row=1, col=3)

    fig.update_layout(L({
        "title": dict(text="<b>RFM Segment Overview</b>",
                      font=dict(size=15)),
        "showlegend": False,
        "margin": dict(t=70, b=50, l=140, r=80),
        "height": 420,
    }))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig


def c_ltv_cac(clv_ch: pd.DataFrame) -> go.Figure:
    # columns: channel_name, ltv_cac_ratio, payback_months
    ch = clv_ch.dropna(subset=["ltv_cac_ratio"]).sort_values(
        "ltv_cac_ratio", ascending=True
    ).copy()

    r_colors = [
        C["teal"] if v >= 3
        else (C["amber"] if v >= 1 else C["red"])
        for v in ch["ltv_cac_ratio"]
    ]
    p_colors = [
        C["teal"] if v <= 12
        else (C["amber"] if v <= 24 else C["red"])
        for v in ch["payback_months"].fillna(999)
    ]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["LTV:CAC Ratio", "Payback Period (months)"],
        horizontal_spacing=0.14,
    )
    fig.add_trace(go.Bar(
        y=ch["channel_name"], x=ch["ltv_cac_ratio"],
        orientation="h", marker_color=r_colors,
        text=[f"{v:.1f}x" for v in ch["ltv_cac_ratio"]],
        textposition="outside", textfont=dict(size=11),
        hovertemplate="<b>%{y}</b><br>%{x:.2f}x<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        y=ch["channel_name"], x=ch["payback_months"],
        orientation="h", marker_color=p_colors,
        text=[f"{v:.0f}m" if not pd.isna(v) else "N/A"
              for v in ch["payback_months"]],
        textposition="outside", textfont=dict(size=11),
        hovertemplate="<b>%{y}</b><br>%{x:.0f}m<extra></extra>",
    ), row=1, col=2)

    for val, label, col in [
        (3, "Healthy (3x)", C["teal"]),
        (1, "Break-even",   C["red"]),
    ]:
        fig.add_vline(
            x=val, line_dash="dash",
            line_color=col, line_width=1.5,
            annotation_text=label,
            annotation_font=dict(size=10, color=col),
            row=1, col=1,
        )

    fig.update_layout(L({
        "title": dict(
            text="<b>LTV:CAC and Payback Period</b>  "
                 "<sup>Green ≥ 3x | Amber 1-3x | Red < 1x</sup>",
            font=dict(size=15)),
        "showlegend": False,
        "margin": dict(t=70, b=50, l=140, r=100),
        "height": 400,
    }))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig


def c_clv_dist(clv: pd.DataFrame) -> go.Figure:
    # columns: clv, clv_tier
    cap  = clv["clv"].quantile(0.95)
    data = clv[clv["clv"] <= cap]["clv"]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["CLV Distribution", "Avg CLV by Tier"],
    )
    fig.add_trace(go.Histogram(
        x=data, nbinsx=60,
        marker=dict(color=C["purple_mid"],
                    line=dict(color="white", width=0.3)),
        opacity=0.85,
        hovertemplate="CLV: R$%{x:,.0f}<br>"
                      "Count: %{y:,}<extra></extra>",
    ), row=1, col=1)

    for val, label, col in [
        (clv["clv"].mean(),   f"Mean R${clv['clv'].mean():,.0f}",
         C["coral"]),
        (clv["clv"].median(), f"Med R${clv['clv'].median():,.0f}",
         C["teal"]),
    ]:
        fig.add_vline(x=val, line_dash="dash",
                      line_color=col, line_width=2,
                      annotation_text=label,
                      annotation_font=dict(size=10, color=col),
                      row=1, col=1)

    tier_order  = ["platinum", "gold", "silver", "bronze", "low"]
    tier_colors = {
        "platinum": C["purple"], "gold": C["amber"],
        "silver":   C["gray"],   "bronze": C["coral"],
        "low":      C["gray_lite"],
    }
    if "clv_tier" in clv.columns:
        ts = (
            clv.groupby("clv_tier")
            .agg(avg_clv=("clv", "mean"),
                 count=("clv", "count"))
            .reset_index()
        )
        ts = ts.set_index("clv_tier").reindex(
            [t for t in tier_order if t in ts.index]
        ).reset_index()

        fig.add_trace(go.Bar(
            x=ts["clv_tier"], y=ts["avg_clv"],
            marker_color=[tier_colors.get(t, C["gray"])
                          for t in ts["clv_tier"]],
            text=[f"R${v:,.0f}\nn={c:,}"
                  for v, c in zip(ts["avg_clv"], ts["count"])],
            textposition="outside", textfont=dict(size=10),
            hovertemplate="<b>%{x}</b><br>"
                          "R$%{y:,.0f}<extra></extra>",
        ), row=1, col=2)

    fig.update_layout(L({
        "title": dict(text="<b>CLV Distribution</b>",
                      font=dict(size=15)),
        "showlegend": False,
        "margin": dict(t=70, b=50, l=60, r=60),
        "height": 400,
    }))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig


def c_pareto(clv: pd.DataFrame) -> go.Figure:
    # columns: total_revenue, clv
    sc = clv.sort_values("total_revenue", ascending=False).copy()
    sc["cum_rev"]     = sc["total_revenue"].cumsum()
    total             = sc["total_revenue"].sum()
    sc["cum_rev_pct"] = sc["cum_rev"] / total * 100
    sc["cust_pct"]    = np.arange(1, len(sc)+1) / len(sc) * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sc["cust_pct"], y=sc["cum_rev_pct"],
        mode="lines",
        name="Revenue concentration",
        line=dict(color=C["purple"], width=3),
        fill="tonexty", fillcolor=C["purple_pale"],
        hovertemplate="Top %{x:.1f}% → %{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[0, 100], y=[0, 100],
        mode="lines", name="Perfect equality",
        line=dict(color=C["gray_lite"], width=1.5, dash="dash"),
    ))
    for pct in [5, 20, 50]:
        mask = sc["cust_pct"] <= pct
        if mask.any():
            rev = sc[mask]["cum_rev_pct"].max()
            fig.add_annotation(
                x=pct, y=rev,
                text=f"<b>Top {pct}%<br>→ {rev:.0f}%</b>",
                showarrow=True, arrowhead=2,
                arrowcolor=C["purple"],
                ax=35, ay=-30,
                bgcolor=C["purple_pale"],
                bordercolor=C["purple"],
                borderwidth=1, borderpad=6,
                font=dict(size=10, color=C["purple"]),
            )
    fig.update_layout(L({
        "title": dict(text="<b>Pareto / Lorenz Curve</b>",
                      font=dict(size=15)),
        "xaxis": dict(title="% of customers",
                      showgrid=True, gridcolor=C["border"],
                      ticksuffix="%", range=[0, 100]),
        "yaxis": dict(title="Cumulative % of revenue",
                      showgrid=True, gridcolor=C["border"],
                      ticksuffix="%", range=[0, 100]),
        "legend": dict(orientation="h", y=-0.14),
        "margin": dict(t=70, b=80, l=70, r=40),
        "height": 440,
    }))
    return fig


def c_cohort(cohort: pd.DataFrame) -> go.Figure:
    c = cohort.copy()
    c.index = c.index.astype(str)

    int_cols = sorted(
        [col for col in c.columns
         if str(col).lstrip("-").isdigit()
         and 0 <= int(col) <= 12],
        key=lambda x: int(x)
    )
    c = c[int_cols].replace(0, np.nan)

    xlabels  = [f"M+{int(col)}" if int(col) > 0 else "M0"
                for col in int_cols]
    text_vals = [
        [f"{v:.1f}%" if not pd.isna(v) else "—" for v in row]
        for row in c.values
    ]

    fig = go.Figure(go.Heatmap(
        z=c.values,
        x=xlabels,
        y=c.index.tolist(),
        colorscale=PURPLE_SCALE,
        text=text_vals,
        texttemplate="%{text}",
        textfont=dict(size=9),
        zmin=0, zmax=12,
        showscale=True,
        colorbar=dict(
            title=dict(text="Retention %", side="right"),
            ticksuffix="%", len=0.8,
        ),
        hovertemplate="<b>Cohort: %{y}</b><br>"
                      "<b>Period: %{x}</b><br>"
                      "%{text}<extra></extra>",
    ))
    fig.update_layout(L({
        "title": dict(
            text="<b>Cohort Retention Matrix</b>  "
                 "<sup>% returning each month | "
                 "Darker = higher retention</sup>",
            font=dict(size=15)),
        "xaxis": dict(side="top", showgrid=False,
                      tickfont=dict(size=10)),
        "yaxis": dict(showgrid=False, autorange="reversed",
                      tickfont=dict(size=10)),
        "margin": dict(t=110, b=20, l=110, r=80),
        "height": max(380, len(c) * 26 + 130),
    }))
    return fig


def c_retention(cohort: pd.DataFrame) -> go.Figure:
    c = cohort.copy()
    c.index = c.index.astype(str)

    int_cols = sorted(
        [col for col in c.columns
         if str(col).lstrip("-").isdigit()
         and 1 <= int(col) <= 12],
        key=lambda x: int(x)
    )
    periods = [int(col) for col in int_cols]
    fig     = go.Figure()

    for idx in c.index:
        vals = [c.loc[idx, col] for col in int_cols]
        pts  = [(p, v) for p, v in zip(periods, vals)
                if v is not None and not pd.isna(v) and v > 0]
        if len(pts) >= 2:
            px_, py_ = zip(*pts)
            fig.add_trace(go.Scatter(
                x=list(px_), y=list(py_),
                mode="lines",
                line=dict(color=C["purple_lite"], width=1),
                showlegend=False, opacity=0.45,
                hoverinfo="skip",
            ))

    avg_vals = []
    for col in int_cols:
        v = c[col].replace(0, np.nan).dropna()
        avg_vals.append(v.mean() if len(v) > 0 else None)

    fig.add_trace(go.Scatter(
        x=periods, y=avg_vals,
        mode="lines+markers", name="Average",
        line=dict(color=C["purple"], width=3.5),
        marker=dict(size=8, color=C["purple"],
                    line=dict(color="white", width=2)),
        hovertemplate="M+%{x}: %{y:.2f}%<extra></extra>",
    ))

    for p in [1, 3, 6]:
        if p in periods:
            v = avg_vals[periods.index(p)]
            if v is not None:
                fig.add_annotation(
                    x=p, y=v,
                    text=f"<b>M+{p}: {v:.1f}%</b>",
                    showarrow=True, arrowhead=2,
                    arrowcolor=C["purple"],
                    ax=25, ay=-25,
                    bgcolor=C["purple_pale"],
                    bordercolor=C["purple"],
                    borderwidth=1, borderpad=5,
                    font=dict(size=10, color=C["purple"]),
                )

    fig.update_layout(L({
        "title": dict(
            text="<b>Retention Curves by Cohort</b>  "
                 "<sup>Thin = individual | Bold = average</sup>",
            font=dict(size=15)),
        "xaxis": dict(title="Months since first purchase",
                      showgrid=False, tickmode="linear", dtick=1),
        "yaxis": dict(title="% retained",
                      showgrid=True, gridcolor=C["border"],
                      ticksuffix="%"),
        "legend": dict(orientation="h", y=-0.14),
        "margin": dict(t=70, b=80, l=60, r=40),
        "height": 420,
    }))
    return fig


def c_churn_dist(churn: pd.DataFrame) -> go.Figure:
    # columns: churn_flag, churn_probability
    active  = churn[churn["churn_flag"] == 0]["churn_probability"]
    churned = churn[churn["churn_flag"] == 1]["churn_probability"]

    fig = go.Figure()
    for data, name, col in [
        (active,  f"Active  (n={len(active):,})",  C["teal"]),
        (churned, f"Churned (n={len(churned):,})", C["red"]),
    ]:
        fig.add_trace(go.Histogram(
            x=data, name=name, nbinsx=50,
            marker=dict(color=col,
                        line=dict(color="white", width=0.3)),
            opacity=0.72,
            hovertemplate="Prob: %{x:.2f}<br>"
                          "Count: %{y:,}<extra></extra>",
        ))

    fig.add_vline(x=0.5, line_dash="dash",
                  line_color=C["text"], line_width=2,
                  annotation_text="Threshold (0.5)",
                  annotation_font=dict(size=10))

    fig.update_layout(L({
        "title": dict(
            text="<b>Churn Probability Distribution</b>  "
                 "<sup>Clear separation = good model</sup>",
            font=dict(size=15)),
        "barmode": "overlay",
        "xaxis": dict(title="Predicted Churn Probability",
                      showgrid=False, range=[0, 1]),
        "yaxis": dict(title="Customers",
                      showgrid=True, gridcolor=C["border"]),
        "legend": dict(orientation="h", y=1.12),
        "margin": dict(t=80, b=60, l=60, r=40),
        "height": 400,
    }))
    return fig


def c_churn_tiers(churn: pd.DataFrame) -> go.Figure:
    # columns: churn_risk_tier, churn_flag
    ts = (
        churn.groupby("churn_risk_tier", observed=True)
        .agg(count=("customer_unique_id", "count"),
             actual_churn=("churn_flag", "mean"))
        .reset_index()
    )
    ts = ts.set_index("churn_risk_tier").reindex(
        [t for t in ["low", "medium", "high"] if t in ts.index]
    ).reset_index()

    tier_colors = {
        "low": C["teal"], "medium": C["amber"], "high": C["red"]
    }

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Count by Risk Tier",
                         "Actual Churn Rate %"],
    )
    fig.add_trace(go.Bar(
        x=ts["churn_risk_tier"],
        y=ts["count"],
        marker_color=[tier_colors.get(t, C["gray"])
                      for t in ts["churn_risk_tier"]],
        text=[f"{v:,}" for v in ts["count"]],
        textposition="outside", textfont=dict(size=12),
        hovertemplate="<b>%{x}</b><br>%{y:,}<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=ts["churn_risk_tier"],
        y=ts["actual_churn"] * 100,
        marker_color=[tier_colors.get(t, C["gray"])
                      for t in ts["churn_risk_tier"]],
        text=[f"{v*100:.1f}%" for v in ts["actual_churn"]],
        textposition="outside", textfont=dict(size=12),
        hovertemplate="<b>%{x}</b><br>%{y:.1f}%<extra></extra>",
    ), row=1, col=2)

    fig.update_layout(L({
        "title": dict(
            text="<b>Churn Risk Tier Analysis</b>  "
                 "<sup>Validates model calibration</sup>",
            font=dict(size=15)),
        "showlegend": False,
        "margin": dict(t=70, b=50, l=60, r=40),
        "height": 380,
    }))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False, row=1, col=2, ticksuffix="%")
    return fig


def c_churn_rfm(churn: pd.DataFrame) -> go.Figure:
    # columns: churn_risk_tier, rfm_segment,
    #          customer_unique_id, total_revenue
    high = churn[churn["churn_risk_tier"] == "high"]
    seg  = (
        high.groupby("rfm_segment")
        .agg(count=("customer_unique_id", "count"),
             rev_at_risk=("total_revenue", "sum"))
        .reset_index()
        .sort_values("count", ascending=True)
    )
    seg_colors = [SEG_COLORS.get(s, C["gray"])
                  for s in seg["rfm_segment"]]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["High-Risk Count by Segment",
                         "Revenue at Risk (R$)"],
        horizontal_spacing=0.12,
    )
    fig.add_trace(go.Bar(
        y=seg["rfm_segment"], x=seg["count"],
        orientation="h", marker_color=seg_colors,
        text=seg["count"], textposition="outside",
        textfont=dict(size=11),
        hovertemplate="<b>%{y}</b><br>%{x:,}<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        y=seg["rfm_segment"], x=seg["rev_at_risk"],
        orientation="h", marker_color=seg_colors,
        text=[f"R${v:,.0f}" for v in seg["rev_at_risk"]],
        textposition="outside", textfont=dict(size=11),
        hovertemplate="<b>%{y}</b><br>"
                      "R$%{x:,.0f}<extra></extra>",
    ), row=1, col=2)

    fig.update_layout(L({
        "title": dict(
            text="<b>High-Risk Customers by RFM Segment</b>",
            font=dict(size=15)),
        "showlegend": False,
        "margin": dict(t=70, b=50, l=150, r=120),
        "height": 380,
    }))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# HTML HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def to_html(fig: go.Figure, div_id: str) -> str:
    return pio.to_html(
        fig, full_html=False, include_plotlyjs=False,
        div_id=div_id,
        config={
            "displayModeBar": True,
            "modeBarButtonsToRemove": ["select2d", "lasso2d"],
            "displaylogo": False,
            "toImageButtonOptions": {
                "format": "png", "filename": div_id, "scale": 2,
            },
        },
    )


def kpi(label, value, sub="", color="#534AB7"):
    return f"""
<div class="kpi-card">
  <div class="kpi-label">{label}</div>
  <div class="kpi-value" style="color:{color}">{value}</div>
  <div class="kpi-sub">{sub}</div>
</div>"""


def sec(title, sub=""):
    s = f'<div class="section-sub">{sub}</div>' if sub else ""
    return f"""
<div class="section-header">
  <div class="section-title">{title}</div>{s}
</div>"""


def ins(icon, title, body, color="#534AB7"):
    return f"""
<div class="insight-card" style="border-left:4px solid {color}">
  <div class="insight-title" style="color:{color}">{icon} {title}</div>
  <div class="insight-body">{body}</div>
</div>"""


def wrap(h):
    return f'<div class="chart-card">{h}</div>'


def g2(a, b):
    return (f'<div class="grid-2">'
            f'<div class="chart-card">{a}</div>'
            f'<div class="chart-card">{b}</div>'
            f'</div>')


# ─────────────────────────────────────────────────────────────────────────────
# BUILD HTML
# ─────────────────────────────────────────────────────────────────────────────

def build_html(d: dict, k: dict) -> str:
    print("\nBuilding charts...")

    ch = {
        "monthly":   to_html(c_monthly(d["monthly"]),            "monthly"),
        "mom":       to_html(c_mom(d["monthly"]),                 "mom"),
        "waterfall": to_html(c_waterfall(k),                      "waterfall"),
        "channel":   to_html(c_channel(d["del"]),                 "channel"),
        "leak_bar":  to_html(c_leakage_bar(
                         d["leakage"], k["total_revenue"]),        "leak_bar"),
        "leak_sun":  to_html(c_leakage_sun(
                         d["leakage"], k["total_revenue"]),        "leak_sun"),
        "top_cat":   to_html(c_top_cat(d["categories"]),          "top_cat"),
        "quadrant":  to_html(c_quadrant(d["categories"]),         "quadrant"),
        "regions":   to_html(c_regions(d["regions"]),             "regions"),
        "rfm":       to_html(c_rfm(d["clv_seg"]),                 "rfm"),
        "ltv_cac":   to_html(c_ltv_cac(d["clv_ch"]),              "ltv_cac"),
        "clv_dist":  to_html(c_clv_dist(d["clv"]),                "clv_dist"),
        "pareto":    to_html(c_pareto(d["clv"]),                   "pareto"),
        "cohort":    to_html(c_cohort(d["cohort"]),               "cohort"),
        "retention": to_html(c_retention(d["cohort"]),            "retention"),
        "ch_dist":   to_html(c_churn_dist(d["churn"]),            "ch_dist"),
        "ch_tiers":  to_html(c_churn_tiers(d["churn"]),           "ch_tiers"),
        "ch_rfm":    to_html(c_churn_rfm(d["churn"]),             "ch_rfm"),
    }
    print(f"  {len(ch)} charts built")

    # High-risk table
    hr = (
        d["churn"][d["churn"]["churn_risk_tier"] == "high"]
        .sort_values("churn_probability", ascending=False)
        .head(30)
    )
    rows = ""
    for _, row in hr.iterrows():
        prob = row["churn_probability"]
        pc   = C["red"] if prob >= 0.8 else C["coral"]
        seg  = str(row.get("rfm_segment", "")).replace("_", "-")
        rows += f"""
<tr>
  <td><code>{str(row['customer_unique_id'])[:22]}...</code></td>
  <td><span class="badge badge-{seg}">{row.get('rfm_segment','')}</span></td>
  <td>R${row['total_revenue']:,.2f}</td>
  <td>R${row['clv']:,.2f}</td>
  <td style="color:{pc};font-weight:700">{prob:.4f}</td>
  <td>{row.get('channel_name','')}</td>
  <td>{row.get('state','')}</td>
</tr>"""

    css = f"""
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Segoe UI',Arial,sans-serif;
      background:{C["bg"]};color:{C["text"]};
      font-size:14px;line-height:1.6}}
.header{{background:linear-gradient(135deg,#2D2580,
  {C["purple"]},{C["purple_mid"]});
  padding:32px 40px;color:white;
  position:relative;overflow:hidden}}
.header::before{{content:'';position:absolute;
  top:-70px;right:-70px;width:280px;height:280px;
  background:rgba(255,255,255,0.06);border-radius:50%}}
.header h1{{font-size:26px;font-weight:700;margin-bottom:6px}}
.header .sub{{font-size:13px;opacity:0.75;margin-bottom:16px}}
.badges{{display:flex;gap:10px;flex-wrap:wrap}}
.badge-pill{{background:rgba(255,255,255,0.18);
  border:1px solid rgba(255,255,255,0.30);
  border-radius:20px;padding:4px 14px;
  font-size:12px;font-weight:500}}
.nav{{background:white;
  border-bottom:1px solid {C["border"]};
  padding:0 40px;position:sticky;top:0;z-index:100;
  display:flex;gap:0;
  box-shadow:0 2px 8px rgba(0,0,0,0.06)}}
.nav a{{display:inline-block;padding:14px 18px;
  text-decoration:none;color:{C["subtext"]};
  font-size:13px;font-weight:500;
  border-bottom:3px solid transparent;
  transition:all 0.2s}}
.nav a:hover{{color:{C["purple"]};
  border-bottom-color:{C["purple"]};
  background:{C["purple_pale"]}}}
.container{{max-width:1380px;margin:0 auto;
  padding:32px 24px 60px 24px}}
.section-header{{margin:40px 0 18px 0;
  padding-bottom:12px;
  border-bottom:2px solid {C["border"]}}}
.section-title{{font-size:20px;font-weight:700;
  display:flex;align-items:center;gap:10px}}
.section-title::before{{content:'';display:inline-block;
  width:5px;height:22px;
  background:{C["purple"]};border-radius:3px}}
.section-sub{{font-size:13px;color:{C["subtext"]};
  margin-top:4px;padding-left:15px}}
.kpi-row{{display:grid;
  grid-template-columns:repeat(auto-fit,minmax(155px,1fr));
  gap:14px;margin-bottom:20px}}
.kpi-card{{background:white;border-radius:12px;
  padding:18px 20px;
  box-shadow:0 1px 4px rgba(83,74,183,0.08),
             0 0 0 1px {C["border"]};
  transition:box-shadow 0.2s}}
.kpi-card:hover{{box-shadow:0 4px 16px rgba(83,74,183,0.14),
  0 0 0 1px {C["purple_lite"]}}}
.kpi-label{{font-size:11px;font-weight:600;
  color:{C["subtext"]};text-transform:uppercase;
  letter-spacing:0.5px;margin-bottom:6px}}
.kpi-value{{font-size:25px;font-weight:700;
  line-height:1.1;margin-bottom:4px}}
.kpi-sub{{font-size:11px;color:{C["gray"]}}}
.chart-card{{background:white;border-radius:12px;
  padding:8px;
  box-shadow:0 1px 4px rgba(83,74,183,0.08),
             0 0 0 1px {C["border"]};
  margin-bottom:20px}}
.grid-2{{display:grid;grid-template-columns:1fr 1fr;
  gap:20px;margin-bottom:20px}}
.insight-row{{display:grid;
  grid-template-columns:repeat(auto-fit,minmax(250px,1fr));
  gap:14px;margin-bottom:24px}}
.insight-card{{background:white;border-radius:10px;
  padding:16px 18px;
  box-shadow:0 1px 4px rgba(0,0,0,0.06),
             0 0 0 1px {C["border"]}}}
.insight-title{{font-size:13px;font-weight:700;margin-bottom:6px}}
.insight-body{{font-size:12px;color:{C["subtext"]};line-height:1.5}}
.table-wrap{{background:white;border-radius:12px;
  padding:20px;
  box-shadow:0 1px 4px rgba(83,74,183,0.08),
             0 0 0 1px {C["border"]};
  margin-bottom:20px;overflow-x:auto}}
.table-wrap h3{{font-size:15px;font-weight:700;
  margin-bottom:14px;padding-left:10px;
  border-left:4px solid {C["red"]}}}
table{{width:100%;border-collapse:collapse;font-size:12px}}
th{{background:{C["red"]};color:white;padding:10px 12px;
  text-align:left;font-weight:600;font-size:11px;
  text-transform:uppercase;letter-spacing:0.3px}}
td{{padding:9px 12px;border-bottom:1px solid {C["border"]};
  vertical-align:middle}}
tr:nth-child(even) td{{background:#FFF8F8}}
tr:hover td{{background:#FFF0F0}}
code{{font-size:10px;background:{C["bg"]};
  padding:2px 5px;border-radius:4px;color:{C["purple"]}}}
.badge{{display:inline-block;padding:2px 10px;
  border-radius:12px;font-size:10px;font-weight:600;
  text-transform:uppercase;letter-spacing:0.3px}}
.badge-champions{{background:{C["purple_pale"]};color:{C["purple"]}}}
.badge-loyal{{background:{C["purple_pale"]};color:{C["purple_mid"]}}}
.badge-potential-loyal{{background:{C["purple_pale"]};color:{C["purple_lite"]}}}
.badge-need-attention{{background:#FAEEDA;color:{C["amber"]}}}
.badge-at-risk{{background:#FAECE7;color:{C["coral"]}}}
.badge-lost{{background:{C["red_lite"]};color:{C["red"]}}}
.badge-unknown{{background:#F1EFE8;color:{C["gray"]}}}
.footer{{background:{C["text"]};
  color:rgba(255,255,255,0.55);
  text-align:center;padding:24px;
  font-size:12px;margin-top:60px}}
@media(max-width:900px){{
  .grid-2{{grid-template-columns:1fr}}
  .nav{{overflow-x:auto}}
  .header,.container{{padding-left:16px;padding-right:16px}}
}}"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>E-Commerce Analytics Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>{css}</style>
</head>
<body>

<div class="header">
  <h1>E-Commerce Revenue Intelligence Dashboard</h1>
  <div class="sub">Olist Brazilian E-Commerce &nbsp;|&nbsp;
    2016–2018 &nbsp;|&nbsp; End-to-End Analytics Pipeline</div>
  <div class="badges">
    <span class="badge-pill">99,441 Orders</span>
    <span class="badge-pill">R${k["total_revenue"]:,.0f} Revenue</span>
    <span class="badge-pill">{k["total_customers"]:,} Customers</span>
    <span class="badge-pill">9 Phases of Analysis</span>
    <span class="badge-pill">Churn ROC-AUC 0.80</span>
    <span class="badge-pill">R${k["total_leak"]:,.0f} Leakage Found</span>
  </div>
</div>

<nav class="nav">
  <a href="#exec">Executive Summary</a>
  <a href="#trends">Revenue Trends</a>
  <a href="#leakage">Leakage</a>
  <a href="#products">Products</a>
  <a href="#customers">Customers & CLV</a>
  <a href="#cohort">Cohort Retention</a>
  <a href="#churn">Churn Risk</a>
</nav>

<div class="container">

<div id="exec">
{sec("Executive Summary",
     "Top-level KPIs and business health at a glance")}
<div class="kpi-row">
  {kpi("Total Revenue",    f"R${k['total_revenue']:,.0f}",
       "Delivered orders", C["purple"])}
  {kpi("Total Orders",     f"{k['total_orders']:,}",
       "Delivered",        C["purple"])}
  {kpi("Avg Order Value",  f"R${k['aov']:,.2f}",
       "Per order",        C["purple"])}
  {kpi("Gross Margin",     f"{k['gross_margin']:.1f}%",
       "After freight",    C["teal"])}
  {kpi("Total Customers",  f"{k['total_customers']:,}",
       "Unique buyers",    C["teal"])}
  {kpi("Repeat Rate",      f"{k['repeat_rate']:.1f}%",
       "2+ orders",        C["teal"])}
</div>
<div class="kpi-row">
  {kpi("Revenue Leakage",     f"R${k['total_leak']:,.0f}",
       f"{k['leak_pct']:.1f}% of revenue",   C["amber"])}
  {kpi("Discount Rate",       f"{k['disc_rate']:.1f}%",
       f"R${k['total_disc']:,.0f} given",     C["coral"])}
  {kpi("Late Deliveries",     f"{k['late_rate']:.1f}%",
       f"{k['late_count']:,} orders",         C["coral"])}
  {kpi("Churn Rate",          f"{k['churn_rate']:.1f}%",
       "180+ days inactive",                  C["red"])}
  {kpi("High-Risk Customers", f"{k['high_risk_n']:,}",
       "Likely to churn",                     C["red"])}
  {kpi("Revenue at Risk",     f"R${k['rev_at_risk']:,.0f}",
       "High-risk segment",                   C["red"])}
</div>
<div class="insight-row">
  {ins("💡","Revenue Leakage",
    f"R${k['total_leak']:,.0f} ({k['leak_pct']:.1f}%) across 6 sources. "
    f"Excessive discounts = 54% of all leakage.",C["amber"])}
  {ins("💡","Churn Intelligence",
    f"{k['high_risk_n']:,} high-risk customers. "
    f"R${k['rev_at_risk']:,.0f} at stake. "
    f"Top predictor: delivery experience.",C["red"])}
  {ins("💡","Customer Value",
    f"Avg CLV R${k['avg_clv']:,.0f}. "
    f"Organic Search = best LTV:CAC. "
    f"Top 20% customers → ~70% of revenue.",C["purple"])}
  {ins("💡","Retention Signal",
    f"M+1 retention ~4-5%. "
    f"Review score drops 2.02 pts on late delivery — "
    f"strongest churn signal.",C["teal"])}
</div>
{g2(ch["monthly"], ch["channel"])}
</div>

<div id="trends">
{sec("Revenue Trends",
     "Monthly growth, waterfall, and channel split")}
{wrap(ch["mom"])}
{g2(ch["waterfall"], ch["channel"])}
</div>

<div id="leakage">
{sec("Revenue Leakage Analysis",
     f"R${k['total_leak']:,.0f} ({k['leak_pct']:.2f}%) recoverable")}
{g2(ch["leak_bar"], ch["leak_sun"])}
</div>

<div id="products">
{sec("Product Performance",
     "Category revenue, margin quadrant, regional breakdown")}
{g2(ch["top_cat"], ch["quadrant"])}
{wrap(ch["regions"])}
</div>

<div id="customers">
{sec("Customer Segmentation & Lifetime Value",
     "RFM, CLV distribution, LTV:CAC, Pareto curve")}
{wrap(ch["rfm"])}
{g2(ch["ltv_cac"], ch["clv_dist"])}
{wrap(ch["pareto"])}
</div>

<div id="cohort">
{sec("Cohort Retention Analysis",
     "% of each monthly cohort returning in subsequent months")}
{wrap(ch["cohort"])}
{wrap(ch["retention"])}
</div>

<div id="churn">
{sec("Churn Risk Intelligence",
     f"Gradient Boosting | ROC-AUC 0.8019 | "
     f"{k['high_risk_n']:,} high-risk customers")}
{g2(ch["ch_dist"], ch["ch_tiers"])}
{wrap(ch["ch_rfm"])}
<div class="table-wrap">
  <h3>High-Risk Customers — Immediate Action Required
      (Top 30 by Churn Probability)</h3>
  <table>
    <thead><tr>
      <th>Customer ID</th><th>Segment</th>
      <th>Lifetime Revenue</th><th>CLV</th>
      <th>Churn Probability</th><th>Channel</th><th>State</th>
    </tr></thead>
    <tbody>{rows}</tbody>
  </table>
</div>
</div>

</div>
<div class="footer">
  E-Commerce Revenue Intelligence Dashboard &nbsp;|&nbsp;
  Python · Plotly · pandas · scikit-learn &nbsp;|&nbsp;
  Olist Brazilian E-Commerce | 99,441 Orders | 2016–2018
</div>
</body></html>"""

    return html


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  PHASE 9 — DASHBOARD GENERATION")
    print("="*60)

    d    = load_data()
    k    = compute_kpis(d)
    html = build_html(d, k)

    out     = REPORTS / "dashboard.html"
    out.write_text(html, encoding="utf-8")
    size_mb = out.stat().st_size / 1024 / 1024

    print(f"\n{'='*60}")
    print(f"  Saved → {out}")
    print(f"  Size:   {size_mb:.1f} MB")
    print(f"  Charts: 18 interactive")
    print(f"\n  Open in Chrome or Firefox:")
    print(f"  {out.resolve()}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()