# End-to-End E-Commerce Revenue Leakage Diagnostic
# and Customer Lifetime Value Optimization Pipeline

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square)
![SQL](https://img.shields.io/badge/SQL-SQLite-lightgrey?style=flat-square)
![ML](https://img.shields.io/badge/ML-scikit--learn-orange?style=flat-square)
![Plotly](https://img.shields.io/badge/Dashboard-Plotly-purple?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)

---

## What This Project Does

A Brazilian e-commerce company (Olist) had growing order volume
but underperforming revenue. This project builds a complete
end-to-end analytics pipeline that answers four critical questions:

1. Where exactly is revenue being lost?
2. Which customers are most valuable?
3. Which customers are about to leave?
4. How do we measure and improve retention over time?

---

## Key Results

| Metric | Result |
|---|---|
| Total Revenue Analysed | R$ 12,540,656 |
| Revenue Leakage Found | R$ 1,143,443 (9.12% of revenue) |
| Largest Leakage Source | Excessive discounts — R$616K (4.92%) |
| Churn Model ROC-AUC | 0.8019 (Gradient Boosting) |
| High-Risk Customers Found | 62,072 customers |
| Late Delivery Rate | 6.8% with 2.02 point review score drop |
| Best Acquisition Channel | Organic Search (highest LTV:CAC ratio) |
| Avg Customer Lifetime Value | R$ 55–85 per customer |

---

## Project Architecture
```
Raw CSV Data (9 files, 100K+ orders)
           ↓
SQLite Star Schema
(fact_orders + 4 dimension tables)
           ↓
Python Cleaning Pipeline
(7 steps + full audit log)
           ↓
SQL Analytics Layer
(14 queries — window functions, CTEs, rankings)
           ↓
┌─────────────────────────────────────────┐
│  Revenue      Cohort        CLV Model   │
│  Leakage      Retention     LTV:CAC     │
│  Analysis     Matrix        by Channel  │
└─────────────────────────────────────────┘
           ↓
Churn Prediction ML Model
(Logistic Regression + Random Forest + Gradient Boosting)
           ↓
Interactive HTML Dashboard
(18 Plotly charts, single file, no server needed)
```

---

## Tools and Technologies

| Category | Tools |
|---|---|
| Language | Python 3.11 |
| Data processing | pandas, numpy |
| Database | SQLite, SQL (window functions, CTEs, rankings) |
| Machine learning | scikit-learn (LR, Random Forest, Gradient Boosting) |
| Visualisation | Plotly, matplotlib, seaborn |
| Dashboard | Plotly (single HTML export — no Power BI needed) |
| Environment | venv, pip, git |
| Dataset | Olist Brazilian E-Commerce (Kaggle, 100K+ orders) |

---

## Repository Structure
```
ecommerce-analytics/
│
├── notebooks/
│   ├── 01_load_data.py          ← Phase 2: ETL + star schema
│   ├── 02_data_cleaning.py      ← Phase 3: cleaning + RFM features
│   ├── 03_sql_analytics.py      ← Phase 4: 14 SQL queries
│   ├── 04_revenue_leakage.py    ← Phase 5: 6 leakage sources
│   ├── 05_cohort_analysis.py    ← Phase 6: retention matrix
│   ├── 06_clv_model.py          ← Phase 7: CLV + LTV:CAC
│   ├── 07_churn_model.py        ← Phase 8: churn prediction ML
│   └── 08_dashboard.py          ← Phase 9: HTML dashboard
│
├── sql/
│   ├── schema/                  ← 5 CREATE TABLE scripts
│   └── analytics/               ← 14 analytical query files
│
├── data/
│   ├── raw/                     ← original Olist CSVs (not in repo)
│   └── processed/               ← cleaned + analytical outputs
│
├── reports/
│   ├── dashboard.html           ← interactive dashboard
│   ├── leakage_report.txt       ← executive leakage summary
│   ├── cohort_report.txt        ← retention analysis summary
│   ├── clv_report.txt           ← CLV model summary
│   ├── churn_report.txt         ← churn model summary
│   └── *.png                    ← 14 exported charts
│
├── requirements.txt
├── PROGRESS.md
└── README.md
```

---

## How to Run
```bash
# 1. Clone the repository
git clone https://github.com/Istiyak-Ishan/ecommerce-analytics.git
cd ecommerce-analytics

# 2. Create virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download dataset (requires Kaggle API key)
kaggle datasets download -d olistbr/brazilian-ecommerce \
  -p data/raw --unzip

# 5. Run the full pipeline in order
python notebooks/01_load_data.py
python notebooks/02_data_cleaning.py
python notebooks/03_sql_analytics.py
python notebooks/04_revenue_leakage.py
python notebooks/05_cohort_analysis.py
python notebooks/06_clv_model.py
python notebooks/07_churn_model.py
python notebooks/08_dashboard.py

# 6. Open the dashboard
# Open reports/dashboard.html in Chrome or Firefox
```

---

## Phase-by-Phase Methodology

### Phase 2 — Data Architecture
Built a star schema data warehouse in SQLite with one central
fact table (`fact_orders`) and four dimension tables
(`dim_customers`, `dim_products`, `dim_time`, `dim_channels`).
Simulated marketing channel assignments and discount distributions
to enrich the raw Olist dataset. Loaded 99,441 orders across
9 CSV files.

### Phase 3 — Data Cleaning
Seven-step professional cleaning pipeline with a full audit log
recording every change made. Engineered 15+ analytical features
including RFM scores, delivery delay, discount sensitivity,
order value tiers, temporal features, and customer segments.
Used IQR Winsorization for outlier capping rather than deletion
to preserve data volume.

### Phase 4 — SQL Analytics
14 queries demonstrating advanced SQL: `LAG()` for growth rates,
`RANK()` / `DENSE_RANK()` / `NTILE()` for rankings,
`PERCENT_RANK()` for percentile distributions,
`SUM() OVER()` for running totals,
window frames (`ROWS BETWEEN`) for rolling averages,
and `PARTITION BY` for within-group analysis.

### Phase 5 — Revenue Leakage
Identified R$1,143,443 in leakage across 6 sources.
Excessive discounts to loyal customers who would have purchased
anyway account for 54% of all leakage.
Late delivery refund risk identified with 6.8% late rate
and a 2.02 point average review score drop on affected orders.

### Phase 6 — Cohort Retention
Built a full retention cohort matrix using `customer_unique_id`
to correctly track repeat buyers across multiple orders.
Found ~4–5% M+1 retention typical of Brazilian e-commerce.
Generated purple heatmap, overlaid retention curves, and
revenue-per-cohort matrix.

### Phase 7 — Customer Lifetime Value
Computed CLV = AOV × Purchase Frequency × Customer Lifespan
× Gross Margin for every customer. Calculated LTV:CAC ratio
per acquisition channel. Organic Search delivers the best
return per R$1 of acquisition spend. Pareto analysis confirms
top 20% of customers generate approximately 70% of revenue.

### Phase 8 — Churn Prediction
Gradient Boosting achieved ROC-AUC of 0.8019, cross-validated
at 0.8009 ± 0.0045. Top predictive features were delivery
experience signals (`avg_freight`, `avg_delivery_delay`,
`avg_review_score`) rather than traditional RFM signals —
a key business insight showing that operational quality
drives retention more than purchase frequency.

### Phase 9 — Dashboard
18-chart interactive HTML dashboard built with Plotly.
Single file export — client opens in any browser with no
installation required. Sections: Executive Summary,
Revenue Trends, Leakage Analysis, Product Performance,
Customer Segmentation + CLV, Cohort Retention, Churn Risk.

---

## Business Insights

**1. 9.12% of revenue is leaking.**
The single biggest recoverable source is R$616K in discounts
given to customers who would have purchased at full price.
Implementing discount eligibility rules could recover this
immediately.

**2. Delivery experience drives churn more than purchase frequency.**
The churn model's top predictors are all delivery-related —
not RFM signals. This means investing in logistics quality
has a higher retention ROI than loyalty programmes.

**3. Top 20% of customers generate ~70% of revenue.**
These customers must never receive a poor delivery experience.
A dedicated high-value customer protection policy is justified
by the data.

**4. Organic Search has the best LTV:CAC ratio.**
Budget should shift toward content and SEO before scaling
paid channels. Every R$1 spent on organic acquisition
generates more lifetime value than paid alternatives.

**5. M+1 retention of ~4–5% is the key metric to improve.**
A single percentage point improvement in M+1 retention
compounds significantly over a customer's lifetime.
A 30-day post-purchase email sequence targeting the
M+1 dropout point is the highest-leverage intervention.

---

## Dataset

**Olist Brazilian E-Commerce Public Dataset**
Source: [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

- 99,441 orders
- 96,478 delivered orders
- 32,951 unique products
- 3,095 sellers
- 27 Brazilian states
- Date range: 2016–2018

---

## Contact

**Built by:** Istiyak Hossain Ishan
**GitHub:** [github.com/Istiyak-Ishan](https://github.com/Istiyak-Ishan)
**Email:** istiyakishan7@gmail.com
