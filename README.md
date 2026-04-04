# E-Commerce Analytics Pipeline

End-to-end analytics project on 99,441 real orders from the Olist Brazilian E-Commerce dataset. Covers data engineering, SQL analytics, revenue diagnostics, customer segmentation, cohort retention, CLV modelling, and churn prediction — with an 18-chart PDF dashboard as the final output.

## What this project does

The central question was: *why is revenue underperforming despite growing order volume?*

The pipeline works through the answer in 8 phases:

1. **ETL** — Load 9 CSV files into a SQLite star schema
2. **Cleaning** — 7-step pipeline with audit log + 15 engineered features (RFM, delivery flags, churn flag)
3. **SQL Analytics** — 14 queries using window functions, CTEs, rolling averages, and PARTITION BY
4. **Revenue Leakage** — Identify and quantify where money is being lost
5. **Cohort Retention** — Monthly cohort matrix across 25 cohorts
6. **CLV Model** — Customer lifetime value + LTV:CAC by acquisition channel
7. **Churn Prediction** — ML classifiers to score every customer by churn risk
8. **Dashboard** — 18-chart PDF report compiled with matplotlib + fpdf2

## Key Results

| | |
|---|---|
| Total revenue analysed | R$ 12,540,656 |
| Revenue leakage found | R$ 1,143,443 (9.1%) |
| Biggest leakage source | Unnecessary discounts — R$ 616K |
| Late delivery rate | 6.8% → 2-point avg review drop |
| Churn model ROC-AUC | 0.8019 (Gradient Boosting) |
| High-risk customers flagged | 62,072 |
| Revenue at risk | R$ 8.2M |

## Main Findings

**Discounts are being given to the wrong customers.** Champions and Loyal customers — the highest-value segment — received an average 27.8% discount. They don't need incentives to purchase. Fixing discount eligibility rules alone could recover the majority of the R$ 616K leakage.

**Delivery experience drives churn more than purchase frequency.** The churn model's top predictors were all delivery-related (freight cost, delivery delay, review score) — not RFM signals. Customers who had a bad delivery experience churned faster than customers who simply bought less.

**Revenue is concentrated.** Top 20% of customers generate ~70% of revenue. A targeted retention programme for this tier protects most of the business at low cost.

**Organic Search has the best unit economics.** Highest LTV:CAC ratio and shortest payback period across all channels.

## Project Structure
```
ecommerce-analytics-pipeline/
├── notebooks/
│   ├── 01_load_data.py          # ETL + star schema
│   ├── 02_data_cleaning.py      # Cleaning + feature engineering
│   ├── 03_sql_analytics.py      # 14 SQL queries
│   ├── 04_revenue_leakage.py    # Leakage analysis
│   ├── 05_cohort_analysis.py    # Cohort retention matrix
│   ├── 06_clv_model.py          # CLV + LTV:CAC by channel
│   ├── 07_churn_model.py        # Churn prediction ML
│   └── 08_dashboard.py          # PDF dashboard generation
├── sql/
│   ├── schema/                  # 5 CREATE TABLE scripts
│   └── analytics/               # 14 analytical queries
├── reports/
│   ├── dashboard.pdf            # 18-chart PDF output
│   ├── charts/                  # Individual PNGs
│   └── *.txt                    # Per-phase findings
├── requirements.txt
└── .gitignore
```

## Quickstart
```bash
git clone https://github.com/Istiyak-Ishan/ecommerce-analytics-pipeline.git
cd ecommerce-analytics-pipeline

python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Download dataset (requires Kaggle API key)
kaggle datasets download -d olistbr/brazilian-ecommerce -p data/raw --unzip

# Run pipeline in order
python notebooks/01_load_data.py
python notebooks/02_data_cleaning.py
python notebooks/03_sql_analytics.py
python notebooks/04_revenue_leakage.py
python notebooks/05_cohort_analysis.py
python notebooks/06_clv_model.py
python notebooks/07_churn_model.py
python notebooks/08_dashboard.py

# Output: reports/dashboard.pdf
```

## Stack
`Python` · `SQLite` · `pandas` · `scikit-learn` · `matplotlib` · `seaborn` · `fpdf2`
