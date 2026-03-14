"""
Phase 8 — Churn Prediction ML Model
======================================
Builds a binary churn classifier using RFM + CLV features.

Label:
  churn_flag = 1 if customer has not purchased in 180+ days
  churn_flag = 0 if customer is still active

Models trained:
  1. Logistic Regression   (interpretable baseline)
  2. Random Forest         (feature importance)
  3. Gradient Boosting     (best performance)

Evaluation:
  Accuracy, Precision, Recall, F1, ROC-AUC
  Confusion matrix, ROC curve, Feature importance

Outputs:
  data/processed/churn_predictions.csv  — predictions for all customers
  reports/churn_*.png                   — 5 evaluation charts
  reports/churn_report.txt              — business summary

Run from project root:
    python notebooks/07_churn_model.py
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

from sklearn.model_selection   import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing     import StandardScaler, LabelEncoder
from sklearn.linear_model      import LogisticRegression
from sklearn.ensemble          import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics           import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
    confusion_matrix, classification_report
)
from sklearn.pipeline          import Pipeline
from sklearn.impute             import SimpleImputer

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

PROCESSED  = Path("data/processed")
REPORTS    = Path("reports")
REPORTS.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE    = 0.20


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — LOAD AND PREPARE FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def load_features() -> pd.DataFrame:
    """
    Load the CLV customers file produced in Phase 7.
    It already contains all the RFM + CLV features we need.
    We add a few extra behavioural features from the orders table.
    """

    # Load CLV file (primary feature source)
    clv_path = PROCESSED / "clv_customers.csv"
    if not clv_path.exists():
        raise FileNotFoundError(
            "clv_customers.csv not found. "
            "Please run Phase 7 first: python notebooks/06_clv_model.py"
        )

    df = pd.read_csv(clv_path)
    print(f"  Loaded clv_customers.csv: {len(df):,} customers")

    # Load additional behavioural features from orders
    db = Path("data/ecommerce.db")
    conn = sqlite3.connect(db)

    order_features = pd.read_sql("""
        SELECT
            c.customer_unique_id,
            AVG(f.review_score)              AS avg_review_score,
            AVG(f.discount_pct)              AS avg_discount_pct,
            SUM(f.has_discount)              AS total_discounted_orders,
            AVG(f.freight_value)             AS avg_freight,
            AVG(f.delivery_delay_days)       AS avg_delivery_delay,
            SUM(f.is_late_delivery)          AS total_late_deliveries,
            AVG(f.order_hour)                AS avg_order_hour,
            COUNT(DISTINCT f.order_quarter)  AS quarters_active,
            COUNT(DISTINCT f.order_year)     AS years_active,
            AVG(f.installments)              AS avg_installments
        FROM fact_orders_clean f
        JOIN dim_customers_clean c ON f.customer_id = c.customer_id
        WHERE f.order_status = 'delivered'
        GROUP BY c.customer_unique_id
    """, conn)
    # conn.close() moved to after churn_flags query — see above

# Fetch churn_flag from dim_customers_clean
    churn_flags = pd.read_sql("""
        SELECT customer_unique_id, churn_flag
        FROM dim_customers_clean
        GROUP BY customer_unique_id
    """, conn)

    # Merge additional features
    df = df.merge(order_features, on="customer_unique_id", how="left")
    df = df.merge(churn_flags,    on="customer_unique_id", how="left")

    # If churn_flag still missing, derive it from recency_days
    if "churn_flag" not in df.columns or df["churn_flag"].isnull().all():
        df["churn_flag"] = (df["recency_days"] > 180).astype(int)

    print(f"  After merge: {len(df):,} customers × {len(df.columns)} columns")
    return df


def engineer_features(df: pd.DataFrame) -> tuple:
    """
    Select and prepare the final feature matrix for ML.

    Features used:
    ──────────────────────────────────────────────────────
    RFM:
      recency_days         — days since last purchase (R)
      total_orders         — number of orders placed  (F)
      total_revenue        — lifetime revenue         (M)

    CLV derived:
      aov                  — average order value
      purchase_frequency   — orders per year
      customer_lifespan    — estimated years active
      clv                  — computed CLV
      ltv_cac_ratio        — CLV / acquisition cost

    Behavioural:
      avg_review_score     — mean review given
      avg_discount_pct     — how discount-sensitive are they
      avg_freight          — avg freight paid
      avg_delivery_delay   — avg days late/early
      total_late_deliveries— total bad delivery experiences
      avg_order_hour       — time of day they shop
      quarters_active      — how many quarters they bought in
      avg_installments     — avg payment installments

    Target:
      churn_flag           — 1 if no purchase in 180+ days
    """

    feature_cols = [
        # RFM
        "recency_days",
        "total_orders",
        "total_revenue",
        # CLV
        "aov",
        "purchase_frequency",
        "customer_lifespan",
        "clv",
        # Behavioural
        "avg_review_score",
        "avg_discount_pct",
        "avg_freight",
        "avg_delivery_delay",
        "total_late_deliveries",
        "avg_order_hour",
        "quarters_active",
        "avg_installments",
        "total_discounted_orders",
    ]

    # Only use cols that exist (defensive)
    feature_cols = [c for c in feature_cols if c in df.columns]

    target_col = "churn_flag"

    # Drop rows where target is null
    df = df.dropna(subset=[target_col])

    X = df[feature_cols].copy()
    y = df[target_col].astype(int)

    # Fill remaining nulls with column median
    X = X.fillna(X.median(numeric_only=True))

    # Log-transform heavy right-skewed features to help linear models
    skewed = ["recency_days", "total_revenue", "clv", "total_orders"]
    for col in skewed:
        if col in X.columns:
            X[f"log_{col}"] = np.log1p(X[col])

    print(f"\n  Feature matrix: {X.shape[0]:,} rows × {X.shape[1]} features")
    print(f"  Churn rate: {y.mean()*100:.1f}%  "
          f"({y.sum():,} churned / {(y==0).sum():,} active)")

    return X, y, df, feature_cols


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────

def split_data(X: pd.DataFrame, y: pd.Series) -> tuple:
    """
    Stratified split preserves the churn ratio in both train and test.
    This is critical when classes are imbalanced.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y          # maintain class balance in both sets
    )

    print(f"\n  Train set: {len(X_train):,} rows  "
          f"(churn rate: {y_train.mean()*100:.1f}%)")
    print(f"  Test set:  {len(X_test):,} rows  "
          f"(churn rate: {y_test.mean()*100:.1f}%)")

    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — TRAIN ALL MODELS
# ─────────────────────────────────────────────────────────────────────────────

def train_models(X_train, X_test, y_train, y_test) -> dict:
    """
    Trains three models inside sklearn Pipelines.
    Each pipeline: Imputer → Scaler → Classifier

    Using a Pipeline ensures:
      - No data leakage (scaler fitted on train only)
      - Clean, reproducible workflow
      - Easy to export for production

    Returns a dict of {name: {pipeline, metrics, predictions}}
    """

    print("\n" + "=" * 60)
    print("  TRAINING MODELS")
    print("=" * 60)

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            class_weight="balanced",   # handles class imbalance
            C=0.1
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=20,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=RANDOM_STATE
        ),
    }

    results = {}

    for name, clf in models.items():
        print(f"\n  Training {name}...")

        # Build pipeline: impute → scale → classify
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
            ("clf",     clf),
        ])

        # Fit on training data
        pipeline.fit(X_train, y_train)

        # Predictions
        y_pred       = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

        # Cross-validation score (5-fold stratified)
        cv = StratifiedKFold(n_splits=5, shuffle=True,
                             random_state=RANDOM_STATE)
        cv_roc_auc = cross_val_score(
            pipeline, X_train, y_train,
            cv=cv, scoring="roc_auc", n_jobs=-1
        )

        # Metrics
        metrics = {
            "accuracy":         round(accuracy_score(y_test, y_pred), 4),
            "precision":        round(precision_score(y_test, y_pred,
                                      zero_division=0), 4),
            "recall":           round(recall_score(y_test, y_pred,
                                      zero_division=0), 4),
            "f1":               round(f1_score(y_test, y_pred,
                                      zero_division=0), 4),
            "roc_auc":          round(roc_auc_score(y_test, y_pred_proba), 4),
            "cv_roc_auc_mean":  round(cv_roc_auc.mean(), 4),
            "cv_roc_auc_std":   round(cv_roc_auc.std(), 4),
        }

        results[name] = {
            "pipeline":     pipeline,
            "metrics":      metrics,
            "y_pred":       y_pred,
            "y_pred_proba": y_pred_proba,
        }

        print(f"    Accuracy:     {metrics['accuracy']:.4f}")
        print(f"    Precision:    {metrics['precision']:.4f}")
        print(f"    Recall:       {metrics['recall']:.4f}")
        print(f"    F1 Score:     {metrics['f1']:.4f}")
        print(f"    ROC-AUC:      {metrics['roc_auc']:.4f}")
        print(f"    CV ROC-AUC:   {metrics['cv_roc_auc_mean']:.4f} "
              f"(+/- {metrics['cv_roc_auc_std']:.4f})")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — IDENTIFY BEST MODEL
# ─────────────────────────────────────────────────────────────────────────────

def identify_best_model(results: dict) -> str:
    """
    Ranks models by ROC-AUC (best metric for imbalanced
    binary classification). Returns name of best model.
    """
    print("\n" + "=" * 60)
    print("  MODEL COMPARISON")
    print("=" * 60)

    comparison = []
    for name, res in results.items():
        m = res["metrics"]
        comparison.append({
            "model":        name,
            "accuracy":     m["accuracy"],
            "precision":    m["precision"],
            "recall":       m["recall"],
            "f1":           m["f1"],
            "roc_auc":      m["roc_auc"],
            "cv_roc_auc":   m["cv_roc_auc_mean"],
        })

    comp_df = pd.DataFrame(comparison).sort_values(
        "roc_auc", ascending=False
    )
    print(comp_df.to_string(index=False))

    best = comp_df.iloc[0]["model"]
    print(f"\n  Best model: {best} "
          f"(ROC-AUC = {comp_df.iloc[0]['roc_auc']:.4f})")

    return best, comp_df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────

def get_feature_importance(results: dict,
                            X: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts feature importance from tree-based models.
    For Logistic Regression we use absolute coefficient values.
    Returns a sorted DataFrame for plotting.
    """

    importance_records = {}

    for name, res in results.items():
        clf = res["pipeline"].named_steps["clf"]

        if hasattr(clf, "feature_importances_"):
            # Tree models: Gini importance
            importance_records[name] = clf.feature_importances_
        elif hasattr(clf, "coef_"):
            # Linear models: absolute coefficient
            importance_records[name] = np.abs(clf.coef_[0])

    imp_df = pd.DataFrame(
        importance_records, index=X.columns
    ).reset_index()
    imp_df.columns = ["feature"] + list(importance_records.keys())

    # Average importance across all models
    model_cols = list(importance_records.keys())
    imp_df["avg_importance"] = imp_df[model_cols].mean(axis=1)
    imp_df = imp_df.sort_values("avg_importance", ascending=False)

    print("\n  Top 10 features by average importance:")
    print(imp_df[["feature", "avg_importance"]].head(10).to_string(
        index=False
    ))

    return imp_df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — GENERATE PREDICTIONS FOR ALL CUSTOMERS
# ─────────────────────────────────────────────────────────────────────────────

def generate_predictions(best_model_name: str,
                          results: dict,
                          df: pd.DataFrame,
                          X: pd.DataFrame,
                          y: pd.Series) -> pd.DataFrame:
    """
    Uses the best model to score ALL customers with:
      - churn_probability  (0–1 score)
      - churn_prediction   (0 or 1 at 0.5 threshold)
      - churn_risk_tier    (high / medium / low)

    This output can feed directly into CRM for targeted campaigns.
    """

    best_pipeline = results[best_model_name]["pipeline"]
    proba = best_pipeline.predict_proba(X)[:, 1]
    pred  = best_pipeline.predict(X)

    # Build output file
    base_cols = ["customer_unique_id", "churn_flag",
                 "rfm_segment", "customer_segment",
                 "total_orders", "total_revenue",
                 "clv", "channel_name", "state"]

    for recency_col in ["recency_days", "active_days", "days_since_purchase"]:
        if recency_col in df.columns:
            base_cols.append(recency_col)
            break

    base_cols = [c for c in base_cols if c in df.columns]
    output = df[base_cols].copy()

    output["churn_probability"] = proba.round(4)
    output["churn_prediction"]  = pred
    output["churn_risk_tier"]   = pd.cut(
        output["churn_probability"],
        bins=[0, 0.33, 0.66, 1.0],
        labels=["low", "medium", "high"]
    )

    # Segment summary
    print("\n  Churn risk tier distribution:")
    tier_dist = output["churn_risk_tier"].value_counts()
    for tier, count in tier_dist.items():
        pct = count / len(output) * 100
        print(f"    {tier:<10} {count:>7,}  ({pct:.1f}%)")

    print("\n  High-risk customers by RFM segment:")
    high_risk = output[output["churn_risk_tier"] == "high"]
    seg_dist  = high_risk["rfm_segment"].value_counts()
    print(seg_dist.to_string())

    return output


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — CHARTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrices(results: dict,
                             y_test: pd.Series) -> None:
    """3-panel confusion matrix — one per model."""

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor("#FAFAFA")

    for ax, (name, res) in zip(axes, results.items()):
        cm = confusion_matrix(y_test, res["y_pred"])
        sns.heatmap(
            cm, annot=True, fmt="d",
            cmap="Blues",
            ax=ax,
            linewidths=0.5,
            cbar=False,
            annot_kws={"size": 14, "weight": "bold"}
        )
        ax.set_facecolor("#FAFAFA")
        ax.set_title(
            f"{name}\n"
            f"F1={res['metrics']['f1']:.3f}  "
            f"AUC={res['metrics']['roc_auc']:.3f}",
            fontsize=11, fontweight="bold", pad=10,
            color="#2C2C2A"
        )
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("Actual", fontsize=10)
        ax.set_xticklabels(["Active", "Churned"], fontsize=9)
        ax.set_yticklabels(["Active", "Churned"], fontsize=9,
                            rotation=0)

    plt.suptitle("Confusion Matrices — All Models",
                 fontsize=13, fontweight="bold", y=1.02,
                 color="#2C2C2A")
    plt.tight_layout()
    out = REPORTS / "churn_confusion_matrices.png"
    plt.savefig(out, dpi=150, bbox_inches="tight",
                facecolor="#FAFAFA")
    plt.close()
    print(f"  Chart saved → {out}")


def plot_roc_curves(results: dict, y_test: pd.Series) -> None:
    """Overlaid ROC curves for all three models."""

    fig, ax = plt.subplots(figsize=(9, 8))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")

    line_colors = {
        "Logistic Regression": "#888780",
        "Random Forest":       "#7F77DD",
        "Gradient Boosting":   "#1D9E75",
    }

    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res["y_pred_proba"])
        auc = res["metrics"]["roc_auc"]
        ax.plot(
            fpr, tpr,
            label=f"{name}  (AUC = {auc:.4f})",
            color=line_colors.get(name, "#444441"),
            linewidth=2.5
        )

    # Random classifier baseline
    ax.plot(
        [0, 1], [0, 1],
        linestyle="--",
        color="#D3D1C7",
        linewidth=1.5,
        label="Random classifier (AUC = 0.50)"
    )

    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curves — Churn Prediction Models",
                 fontsize=13, fontweight="bold", pad=15,
                 color="#2C2C2A")
    ax.legend(fontsize=10, frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(linestyle="--", alpha=0.3)

    plt.tight_layout()
    out = REPORTS / "churn_roc_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight",
                facecolor="#FAFAFA")
    plt.close()
    print(f"  Chart saved → {out}")


def plot_feature_importance(imp_df: pd.DataFrame) -> None:
    """Horizontal bar chart — top 15 features by avg importance."""

    top15 = imp_df.head(15).sort_values("avg_importance", ascending=True)

    fig, ax = plt.subplots(figsize=(11, 8))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")

    bar_colors = ["#534AB7" if "log_" in f or f in
                  ["recency_days","total_orders","total_revenue","clv"]
                  else "#1D9E75"
                  for f in top15["feature"]]

    bars = ax.barh(
        top15["feature"],
        top15["avg_importance"],
        color=bar_colors,
        height=0.65,
        edgecolor="none"
    )

    for bar, val in zip(bars, top15["avg_importance"]):
        ax.text(
            bar.get_width() + top15["avg_importance"].max() * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center", fontsize=8.5, color="#444441"
        )

    ax.set_xlabel("Average Feature Importance", fontsize=11)
    ax.set_title("Top 15 Features — Churn Prediction\n"
                 "Purple = RFM/CLV   Green = behavioural",
                 fontsize=12, fontweight="bold", pad=14,
                 color="#2C2C2A")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)

    plt.tight_layout()
    out = REPORTS / "churn_feature_importance.png"
    plt.savefig(out, dpi=150, bbox_inches="tight",
                facecolor="#FAFAFA")
    plt.close()
    print(f"  Chart saved → {out}")


def plot_churn_probability_dist(predictions: pd.DataFrame) -> None:
    """
    Histogram of churn probability scores separated by
    actual churn label. A good model will show clear
    separation between the two distributions.
    """

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor("#FAFAFA")

    # ── Chart 1: Overlapping distributions ───────────────────────────────────
    ax1 = axes[0]
    ax1.set_facecolor("#FAFAFA")

    active  = predictions[predictions["churn_flag"] == 0]["churn_probability"]
    churned = predictions[predictions["churn_flag"] == 1]["churn_probability"]

    ax1.hist(active,  bins=50, alpha=0.65, color="#1D9E75",
             label=f"Active  (n={len(active):,})",  edgecolor="none")
    ax1.hist(churned, bins=50, alpha=0.65, color="#E24B4A",
             label=f"Churned (n={len(churned):,})", edgecolor="none")
    ax1.axvline(0.5, color="#444441", linewidth=1.5,
                linestyle="--", label="Decision threshold (0.5)")
    ax1.set_xlabel("Predicted churn probability", fontsize=10)
    ax1.set_ylabel("Number of customers", fontsize=10)
    ax1.set_title("Churn Probability Distribution\nby Actual Label",
                  fontsize=12, fontweight="bold", pad=12,
                  color="#2C2C2A")
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.legend(fontsize=9, frameon=False)

    # ── Chart 2: Churn rate by risk tier ─────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("#FAFAFA")

    tier_stats = (
        predictions.groupby("churn_risk_tier", observed=True)
        .agg(
            customer_count = ("customer_unique_id", "count"),
            actual_churn   = ("churn_flag",         "mean"),
            avg_clv        = ("clv",                "mean"),
            avg_revenue    = ("total_revenue",      "mean"),
        )
        .reset_index()
    )
    tier_order  = ["low", "medium", "high"]
    tier_colors = {"low": "#1D9E75", "medium": "#EF9F27", "high": "#E24B4A"}

    tier_stats = tier_stats.set_index("churn_risk_tier").reindex(
        [t for t in tier_order if t in tier_stats.index]
    ).reset_index()

    bars = ax2.bar(
        tier_stats["churn_risk_tier"],
        tier_stats["actual_churn"] * 100,
        color=[tier_colors[t] for t in tier_stats["churn_risk_tier"]],
        width=0.55,
        edgecolor="none"
    )
    for bar, (_, row) in zip(bars, tier_stats.iterrows()):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{row['actual_churn']*100:.1f}%\n({row['customer_count']:,} customers)\n"
            f"Avg CLV: R${row['avg_clv']:,.0f}",
            ha="center", va="bottom",
            fontsize=8.5, color="#444441"
        )
    ax2.set_xlabel("Churn Risk Tier", fontsize=10)
    ax2.set_ylabel("Actual Churn Rate (%)", fontsize=10)
    ax2.set_title("Actual Churn Rate by Predicted Risk Tier\n"
                  "(validates model calibration)",
                  fontsize=12, fontweight="bold", pad=12,
                  color="#2C2C2A")
    ax2.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = REPORTS / "churn_probability_distribution.png"
    plt.savefig(out, dpi=150, bbox_inches="tight",
                facecolor="#FAFAFA")
    plt.close()
    print(f"  Chart saved → {out}")


def plot_model_comparison(comp_df: pd.DataFrame) -> None:
    """Grouped bar chart comparing all models on all metrics."""

    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    models  = comp_df["model"].tolist()
    x       = np.arange(len(metrics))
    width   = 0.25

    fig, ax = plt.subplots(figsize=(13, 7))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")

    model_colors = ["#888780", "#7F77DD", "#1D9E75"]

    for i, (_, row) in enumerate(comp_df.iterrows()):
        values = [row[m] for m in metrics]
        offset = (i - 1) * width
        bars = ax.bar(
            x + offset, values,
            width=width,
            label=row["model"],
            color=model_colors[i],
            edgecolor="none",
            alpha=0.9
        )
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center", va="bottom",
                fontsize=8, color="#444441"
            )

    ax.set_xticks(x)
    ax.set_xticklabels(
        ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"],
        fontsize=10
    )
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Model Performance Comparison — All Metrics",
                 fontsize=13, fontweight="bold", pad=15,
                 color="#2C2C2A")
    ax.legend(fontsize=10, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(y=0.5, color="#D3D1C7", linewidth=1,
               linestyle="--", alpha=0.7)

    plt.tight_layout()
    out = REPORTS / "churn_model_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight",
                facecolor="#FAFAFA")
    plt.close()
    print(f"  Chart saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — EXECUTIVE REPORT
# ─────────────────────────────────────────────────────────────────────────────

def write_churn_report(best_name: str,
                       results: dict,
                       comp_df: pd.DataFrame,
                       predictions: pd.DataFrame,
                       imp_df: pd.DataFrame) -> None:

    best_metrics = results[best_name]["metrics"]
    high_risk    = predictions[predictions["churn_risk_tier"] == "high"]
    top_features = imp_df.head(5)["feature"].tolist()

    report = f"""
CHURN PREDICTION MODEL REPORT
E-Commerce Analytics — Phase 8
{'='*60}

BEST MODEL: {best_name}
─────────────────────────────────────────────────────────
Accuracy:         {best_metrics['accuracy']:.4f}
Precision:        {best_metrics['precision']:.4f}
Recall:           {best_metrics['recall']:.4f}
F1 Score:         {best_metrics['f1']:.4f}
ROC-AUC:          {best_metrics['roc_auc']:.4f}
CV ROC-AUC:       {best_metrics['cv_roc_auc_mean']:.4f} (+/- {best_metrics['cv_roc_auc_std']:.4f})

MODEL COMPARISON
─────────────────────────────────────────────────────────
{comp_df[['model','accuracy','precision','recall','f1','roc_auc']].to_string(index=False)}

CHURN RISK SEGMENTATION
─────────────────────────────────────────────────────────
High-risk customers:   {len(high_risk):,}
  Avg CLV:             R$ {high_risk['clv'].mean():,.2f}
  Avg revenue:         R$ {high_risk['total_revenue'].mean():,.2f}
  Revenue at risk:     R$ {high_risk['total_revenue'].sum():,.2f}

TOP PREDICTIVE FEATURES
─────────────────────────────────────────────────────────
{chr(10).join(f'  {i+1}. {f}' for i, f in enumerate(top_features))}

BUSINESS RECOMMENDATIONS
─────────────────────────────────────────────────────────
1. IMMEDIATE ACTION — Contact the {len(high_risk):,} high-risk customers
   with personalised win-back offers. Revenue at risk:
   R$ {high_risk['total_revenue'].sum():,.0f}

2. EARLY WARNING — Monitor recency_days as the leading
   indicator. Customers crossing 90 days inactive should
   trigger an automated re-engagement email.

3. REVIEW SCORE — Low review scores are a strong churn
   predictor. A dedicated post-delivery satisfaction
   intervention can recover at-risk customers early.

4. DISCOUNT SENSITIVITY — Discount-heavy customers churn
   faster when discounts stop. Wean these customers onto
   value-based loyalty rather than price incentives.

5. DEPLOY MODEL — Score all new customers monthly and
   feed predictions into CRM for automated campaigns.
   Re-train quarterly as new data accumulates.
"""

    out = REPORTS / "churn_report.txt"
    out.write_text(report, encoding="utf-8")
    print(f"  Report saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  E-COMMERCE ANALYTICS — PHASE 8 CHURN MODEL")
    print("=" * 60 + "\n")

    # 1. Load features
    df = load_features()

    # 2. Engineer feature matrix
    print("\nEngineering features...")
    X, y, df_full, feature_cols = engineer_features(df)

    # 3. Split
    print("\nSplitting data...")
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 4. Train all models
    results = train_models(X_train, X_test, y_train, y_test)

    # 5. Compare and pick best
    best_name, comp_df = identify_best_model(results)

    # 6. Feature importance
    print("\nExtracting feature importance...")
    imp_df = get_feature_importance(results, X)

    # 7. Score all customers
    print("\nScoring all customers...")
    predictions = generate_predictions(
        best_name, results, df_full, X, y
    )

    # 8. Save outputs
    print("\nSaving outputs...")
    predictions.to_csv(
        PROCESSED / "churn_predictions.csv", index=False
    )
    comp_df.to_csv(REPORTS / "churn_model_comparison.csv", index=False)
    imp_df.to_csv(REPORTS / "churn_feature_importance.csv", index=False)
    print(f"  Saved churn_predictions.csv  ({len(predictions):,} rows)")

    # 9. Charts
    print("\nGenerating charts...")
    plot_confusion_matrices(results, y_test)
    plot_roc_curves(results, y_test)
    plot_feature_importance(imp_df)
    plot_churn_probability_dist(predictions)
    plot_model_comparison(comp_df)

    # 10. Report
    write_churn_report(best_name, results, comp_df,
                       predictions, imp_df)

    print("\n" + "=" * 60)
    print("  PHASE 8 COMPLETE")
    print("=" * 60)
    print(f"\n  Best model:   {best_name}")
    print(f"  ROC-AUC:      {results[best_name]['metrics']['roc_auc']:.4f}")
    print(f"  F1 Score:     {results[best_name]['metrics']['f1']:.4f}")
    print(f"\n  Ready for Phase 9 — Dashboard Design.\n")


if __name__ == "__main__":
    main()