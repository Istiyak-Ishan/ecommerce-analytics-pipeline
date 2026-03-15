# End-to-End E-Commerce Revenue Leakage Diagnostic
# and Customer Lifetime Value Optimization Pipeline

![Python](https://img.shields.io/badge/Python-3.11-blue)
![SQL](https://img.shields.io/badge/SQL-SQLite-lightgrey)
![ML](https://img.shields.io/badge/ML-scikit--learn-orange)
![Viz](https://img.shields.io/badge/Viz-matplotlib%20%7C%20seaborn-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## Overview

A complete end-to-end analytics pipeline built on 99,441 real
e-commerce orders from the Olist Brazilian E-Commerce dataset.
The project covers data engineering, SQL analytics, revenue
diagnostics, customer segmentation, cohort retention analysis,
customer lifetime value modelling, and churn prediction —
delivering a 18-chart PDF dashboard and actionable business
recommendations.

---

## Business Problem

The business was experiencing revenue underperformance despite
growing order volume. The core questions driving this project:

- Where exactly is revenue leaking and how much is recoverable?
- Which customers are most valuable and how long will they stay?
- Which customers are about to churn and what is the revenue at risk?
- How do different acquisition channels compare on lifetime value?
- What does retention look like across monthly customer cohorts?

---

## Key Results

| Metric | Value |
|---|---|
| Total revenue analysed | R$ 12,540,656 |
| Total revenue leakage identified | R$ 1,143,443 (9.1%) |
| Largest leakage source | Excessive discounts (R$ 616K) |
| Late delivery rate | 6.8% with 2.02-point review drop |
| Churn model ROC-AUC | 0.8019 (Gradient Boosting) |
| High-risk customers identified | 62,072 |
| Revenue at risk from churn | R$ 8.2M |
| Best acquisition channel | Organic Search (highest LTV:CAC) |
| Top churn predictor | Delivery experience, not purchase frequency |

---

## Project Architecture
```
Raw CSV Data (9 files, Olist Brazilian E-Commerce)
        |
        v
SQLite Star Schema
  fact_orders + dim_customers + dim_products
  + dim_time + dim_channels
        |
        v
Python Cleaning Pipeline
  7-step cleaning + RFM feature engineering
  + outlier detection + audit log
        |
        v
SQL Analytics Layer
  14 queries: revenue KPIs, growth rates,
  cohort foundations, channel ROI,
  product rankings, regional breakdown
        |
        v
    +---+---+---+---+
    |   |   |   |   |
    v   v   v   v   v
Revenue  Cohort  CLV    Churn
Leakage  Retention Model  Prediction
Analysis  Matrix   LTV:CAC  ML Model
    |   |   |   |   |
    +---+---+---+---+
        |
        v
18-Chart PDF Dashboard
  matplotlib + seaborn
  KPI summary, trends, leakage,
  products, CLV, cohort heatmap, churn risk
```

---

## Dataset

**Olist Brazilian E-Commerce Public Dataset**
Source: Kaggle — `olistbr/brazilian-ecommerce`

9 CSV files covering 100,000+ real orders placed between
2016 and 2018 across multiple product categories and
27 Brazilian states.

| Table | Rows |
|---|---|
| orders | 99,441 |
| order_items | 112,650 |
| customers | 99,441 |
| products | 32,951 |
| payments | 103,886 |
| reviews | 99,224 |
| sellers | 3,095 |

---

## Repository Structure
```
ecommerce-analytics/
|
|-- notebooks/
|   |-- 01_load_data.py          Phase 2: ETL + star schema
|   |-- 02_data_cleaning.py      Phase 3: cleaning + features
|   |-- 03_sql_analytics.py      Phase 4: 14 SQL queries
|   |-- 04_revenue_leakage.py    Phase 5: leakage analysis
|   |-- 05_cohort_analysis.py    Phase 6: cohort matrix
|   |-- 06_clv_model.py          Phase 7: CLV + LTV:CAC
|   |-- 07_churn_model.py        Phase 8: churn prediction
|   |-- 08_dashboard.py          Phase 9: PDF dashboard
|
|-- sql/
|   |-- schema/                  5 CREATE TABLE scripts
|   |-- analytics/               14 analytical SQL queries
|
|-- reports/
|   |-- dashboard.pdf            18-chart compiled PDF
|   |-- charts/                  18 individual PNG charts
|   |-- leakage_report.txt       Revenue leakage summary
|   |-- cohort_report.txt        Cohort retention findings
|   |-- clv_report.txt           CLV and LTV:CAC findings
|   |-- churn_report.txt         Churn model findings
|
|-- requirements.txt
|-- .gitignore
|-- README.md
```

---

## Tech Stack

| Area | Tools |
|---|---|
| Language | Python 3.11 |
| Data manipulation | pandas, numpy |
| Database | SQLite |
| SQL | Window functions, CTEs, RANK, LAG, NTILE |
| Machine learning | scikit-learn |
| ML models | Logistic Regression, Random Forest, Gradient Boosting |
| Visualisation | matplotlib, seaborn |
| PDF generation | fpdf2 |
| Environment | venv, pip, git |

---

## How to Run
```bash
# 1. Clone the repository
git clone https://github.com/Istiyak-Ishan/ecommerce-analytics.git
cd ecommerce-analytics

# 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

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
# reports/dashboard.pdf
```

---

## Phase Breakdown

### Phase 1 — Data Architecture

Built a star schema data warehouse in SQLite with one central
fact table and four dimension tables. The fact table holds
99,441 order records with revenue, discount, freight, and
gross profit measures. Dimension tables cover customers,
products, time, and marketing channels. Indexes applied on
all foreign keys for fast analytical queries.

### Phase 2 — Data Cleaning Pipeline

A seven-step professional cleaning pipeline with a full audit
log recording every transformation. Steps covered duplicate
removal, targeted null handling per column (fill, flag, or
drop depending on business context), dtype enforcement,
and outlier capping using the IQR Winsorization method.

Feature engineering produced 15 new analytical columns
including RFM scores for every customer, delivery delay
flags, discount sensitivity signals, order value tiers,
and a churn flag based on 180-day inactivity.

### Phase 3 — SQL Analytics

Fourteen SQL queries covering all key business metrics.
SQL techniques demonstrated include window functions
(LAG, RANK, DENSE_RANK, NTILE, PERCENT_RANK),
common table expressions, rolling averages using
ROWS BETWEEN frames, and PARTITION BY for within-group
rankings. All query results saved as CSVs for downstream
analysis and dashboard use.

### Phase 4 — Revenue Leakage Analysis

Identified R$ 1,143,443 in recoverable revenue leakage
across six sources. Excessive discounts were the largest
source at R$ 616,348 (4.92% of revenue), broken down
into three sub-categories: orders discounted above 20%,
high-value orders that received unnecessary discounts, and
Champions/Loyal customers who were given discounts they
did not need to convert.

Additional sources included cancellation losses (R$ 178K),
high-LTV customer discount abuse (R$ 190K), freight
billing mismatches (R$ 81K), pricing anomalies across
2,043 products (R$ 49K), and late delivery refund risk
(R$ 29K).

### Phase 5 — Cohort Retention Analysis

Built a full customer retention cohort matrix using
customer_unique_id to correctly track repeat buyers across
Olist's order-level customer ID system. Average M+1
retention of 4-5% is consistent with Brazilian e-commerce
benchmarks. The cohort heatmap visualises all 25 monthly
cohorts with purple gradient intensity encoding retention
percentage.

### Phase 6 — Customer Lifetime Value Model

Computed CLV for every customer using the formula:
CLV = AOV x Purchase Frequency x Customer Lifespan
x Gross Margin.

Customer lifespan estimated from purchase frequency tiers
(1.0 to 4.0 years) as a defensible heuristic without
requiring a probabilistic BG/NBD model. LTV:CAC ratio
calculated per acquisition channel, and a Pareto analysis
confirmed the top 20% of customers generate approximately
70% of total revenue.

### Phase 7 — Churn Prediction Model

Trained three classifiers (Logistic Regression, Random
Forest, Gradient Boosting) inside sklearn Pipelines with
stratified 80/20 train-test splits. The Gradient Boosting
model achieved the best performance with ROC-AUC of 0.8019
and F1 score of 0.848, cross-validated at 0.8009
(+/- 0.0045).

The most important finding: the top churn predictors were
delivery experience features (avg_freight,
avg_delivery_delay, avg_review_score) rather than RFM
signals. Customers who experienced poor delivery churn
faster than customers with low purchase frequency.

All customers scored with churn probability and risk tier
(low/medium/high) for CRM targeting.

### Phase 8 — Dashboard

Generated 18 publication-quality charts using matplotlib
and seaborn, compiled into a single professional PDF report
using fpdf2. The dashboard covers seven sections: executive
summary with 12 KPI cards, revenue trends, leakage
analysis, product performance, customer segmentation and
CLV, cohort retention heatmap, and churn risk intelligence
with a high-risk customer action table.

---

## Business Insights

**1. Discount strategy needs restructuring.**
R$ 616K is being lost to unnecessary discounts. Champions
and Loyal customers — the most valuable segment — received
the highest average discount percentage (27.8%). These
customers do not need price incentives to purchase.
Implementing discount eligibility rules by customer segment
and order value could recover a significant portion of this
leakage immediately.

**2. Delivery experience is the primary churn driver.**
The churn model found that delivery-related features
outranked all RFM signals in predictive importance. Late
deliveries cause a 2.02-point drop in review scores and
are the strongest leading indicator of customer loss.
Improving logistics SLAs in high-delay states (AL, MA, SE)
would have a direct impact on retention.

**3. Revenue is highly concentrated.**
The top 20% of customers generate approximately 70% of
revenue. This concentration means a targeted retention
programme for the top customer tier would protect the
majority of revenue at relatively low cost.

**4. Organic Search delivers the best unit economics.**
Across all acquisition channels, Organic Search has the
highest LTV:CAC ratio with the shortest payback period.
Reallocating budget from high-CAC paid channels toward
content and SEO would improve overall channel ROI.

**5. Cohort retention is recoverable.**
M+1 retention of 4-5% leaves significant room for
improvement. A structured 30-day post-purchase email
sequence targeting the M+1 drop-off point could
meaningfully improve lifetime value across all cohorts.

---

## Results Summary
```
Total Revenue Analysed:      R$ 12,540,656
Revenue Leakage Found:       R$  1,143,443  (9.1%)
Churn Model ROC-AUC:         0.8019
High-Risk Customers:         62,072
Charts Generated:            18
SQL Queries Written:         14
Features Engineered:         15+
Pipeline Scripts:            8
```

---

---

## Contact

**Istiyak Hossain Ishan**

- GitHub: [github.com/Istiyak-Ishan](https://github.com/Istiyak-Ishan)
- Email: [istiyakishan7@gmail.com](mailto:istiyakishan7@gmail.com)