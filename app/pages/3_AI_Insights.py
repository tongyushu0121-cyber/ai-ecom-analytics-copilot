import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="AI Insights", layout="wide")
st.title("AI Insights")

st.caption("Rule-based executive summary (synthetic data only). Replace with LLM later.")

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DEMO_CSV = ROOT_DIR / "data" / "synthetic_orders.csv"

if "orders_df" not in st.session_state:
    st.warning("No data found. Load demo data or go to Home to upload a CSV.")
    if st.button("Load demo data (synthetic)", use_container_width=True):
        df_demo = pd.read_csv(DEMO_CSV)
        if "order_date" in df_demo.columns:
            df_demo["order_date"] = pd.to_datetime(df_demo["order_date"], errors="coerce")
        st.session_state["orders_df"] = df_demo
        st.rerun()
    st.stop()

df = st.session_state["orders_df"]



# -------------------------
# Helpers: column detection
# -------------------------
def pick_col(candidates):
    """Return the first matching column name from candidates (case-insensitive)."""
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    # fuzzy contains
    for cand in candidates:
        for c in df.columns:
            if cand.lower() in c.lower():
                return c
    return None

ORDER_DATE = pick_col(["order_date", "date", "ordered_at", "purchase_date", "created_at"])
ORDER_ID   = pick_col(["order_id", "order", "id"])
SKU        = pick_col(["sku", "product_sku", "item_sku", "asin"])
CHANNEL    = pick_col(["channel", "sales_channel", "platform", "marketplace"])

QTY        = pick_col(["quantity", "qty", "units", "item_qty"])
REVENUE    = pick_col(["revenue", "sales", "sales_amount", "gmv", "item_revenue", "order_revenue", "total"])
PRICE      = pick_col(["price", "unit_price", "selling_price"])
COST       = pick_col(["cost", "cogs", "unit_cost", "product_cost"])
PROFIT     = pick_col(["profit", "gross_profit", "margin_dollars"])
RETURN_FLG = pick_col(["is_return", "returned", "refund", "is_refund", "return_flag", "refund_flag"])

# Normalize order_date if present
if ORDER_DATE is not None:
    df[ORDER_DATE] = pd.to_datetime(df[ORDER_DATE], errors="coerce")

# -------------------------
# Build derived metrics safely
# -------------------------
def safe_sum(series):
    try:
        return float(pd.to_numeric(series, errors="coerce").fillna(0).sum())
    except Exception:
        return np.nan

def safe_mean(series):
    try:
        s = pd.to_numeric(series, errors="coerce")
        return float(s.mean())
    except Exception:
        return np.nan

def format_money(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "N/A"
    return f"${x:,.2f}"

def format_pct(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "N/A"
    return f"{x*100:.1f}%"

# If revenue missing but price+qty exist, compute revenue
if REVENUE is None and PRICE is not None and QTY is not None:
    df["_revenue_calc"] = pd.to_numeric(df[PRICE], errors="coerce") * pd.to_numeric(df[QTY], errors="coerce")
    REVENUE = "_revenue_calc"

# If profit missing but revenue+cost exist, compute profit
if PROFIT is None:
    if REVENUE is not None and COST is not None:
        # If cost is unit_cost and qty exists, estimate total cost = unit_cost * qty
        if QTY is not None:
            df["_cost_total_calc"] = pd.to_numeric(df[COST], errors="coerce") * pd.to_numeric(df[QTY], errors="coerce")
            df["_profit_calc"] = pd.to_numeric(df[REVENUE], errors="coerce") - df["_cost_total_calc"]
        else:
            df["_profit_calc"] = pd.to_numeric(df[REVENUE], errors="coerce") - pd.to_numeric(df[COST], errors="coerce")
        PROFIT = "_profit_calc"

# Basic counts
n_rows = df.shape[0]
n_orders = df[ORDER_ID].nunique() if ORDER_ID else np.nan
n_skus = df[SKU].nunique() if SKU else np.nan
n_channels = df[CHANNEL].nunique() if CHANNEL else np.nan

date_min, date_max = None, None
if ORDER_DATE:
    date_min = df[ORDER_DATE].min()
    date_max = df[ORDER_DATE].max()

total_revenue = safe_sum(df[REVENUE]) if REVENUE else np.nan
total_profit  = safe_sum(df[PROFIT]) if PROFIT else np.nan
total_qty     = safe_sum(df[QTY]) if QTY else np.nan

aov = np.nan
if ORDER_ID and REVENUE:
    rev_by_order = df.groupby(ORDER_ID)[REVENUE].sum(numeric_only=True)
    aov = float(pd.to_numeric(rev_by_order, errors="coerce").mean())

gross_margin = np.nan
if REVENUE and PROFIT and total_revenue and not np.isnan(total_revenue) and total_revenue != 0:
    gross_margin = total_profit / total_revenue

# Return/refund rate if flag exists
return_rate = np.nan
if RETURN_FLG:
    # try interpret as boolean / 0-1 / yes-no
    flg = df[RETURN_FLG]
    if flg.dtype == bool:
        return_rate = float(flg.mean())
    else:
        s = flg.astype(str).str.lower().str.strip()
        mapped = s.isin(["1", "true", "yes", "y", "returned", "refund", "refunded"])
        # If column is numeric 0/1
        num = pd.to_numeric(df[RETURN_FLG], errors="coerce")
        if num.notna().mean() > 0.8:
            return_rate = float(num.fillna(0).clip(0, 1).mean())
        else:
            return_rate = float(mapped.mean())

# -------------------------
# KPIs row
# -------------------------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Rows", f"{n_rows:,}")
c2.metric("Orders", f"{int(n_orders):,}" if not np.isnan(n_orders) else "N/A")
c3.metric("Revenue", format_money(total_revenue))
c4.metric("Profit", format_money(total_profit) if not np.isnan(total_profit) else "N/A")
c5.metric("Gross Margin", format_pct(gross_margin) if not np.isnan(gross_margin) else "N/A")

c6, c7, c8, c9, c10 = st.columns(5)
c6.metric("AOV", format_money(aov) if not np.isnan(aov) else "N/A")
c7.metric("Units", f"{int(total_qty):,}" if not np.isnan(total_qty) else "N/A")
c8.metric("SKUs", f"{int(n_skus):,}" if not np.isnan(n_skus) else "N/A")
c9.metric("Channels", f"{int(n_channels):,}" if not np.isnan(n_channels) else "N/A")
c10.metric("Return Rate", format_pct(return_rate) if not np.isnan(return_rate) else "N/A")

if ORDER_DATE and pd.notna(date_min) and pd.notna(date_max):
    st.write(f"Date range: **{date_min.date()} → {date_max.date()}**")

st.divider()

# -------------------------
# Aggregations for insights
# -------------------------
def top_group(col, metric_col, k=5):
    if col is None or metric_col is None:
        return None
    g = df.groupby(col)[metric_col].sum(numeric_only=True).sort_values(ascending=False).head(k)
    return g

def top_group_count(col, k=5):
    if col is None:
        return None
    g = df.groupby(col).size().sort_values(ascending=False).head(k)
    return g

top_channels_rev = top_group(CHANNEL, REVENUE, 5)
top_skus_rev = top_group(SKU, REVENUE, 5)
top_channels_cnt = top_group_count(CHANNEL, 5)
top_skus_cnt = top_group_count(SKU, 5)

# Trend: revenue by day
rev_by_day = None
if ORDER_DATE and REVENUE:
    tmp = df.dropna(subset=[ORDER_DATE]).copy()
    tmp["_day"] = tmp[ORDER_DATE].dt.date
    rev_by_day = tmp.groupby("_day")[REVENUE].sum(numeric_only=True).sort_index()

# Simple anomaly: max day vs median
anomaly_day, anomaly_ratio = None, np.nan
if rev_by_day is not None and len(rev_by_day) >= 5:
    max_day = rev_by_day.idxmax()
    max_val = float(rev_by_day.max())
    med_val = float(rev_by_day.median())
    if med_val > 0:
        anomaly_day = max_day
        anomaly_ratio = max_val / med_val

# -------------------------
# Executive Summary (rule-based)
# -------------------------
summary_lines = []

# 1) Scale
if ORDER_DATE and pd.notna(date_min) and pd.notna(date_max):
    summary_lines.append(
        f"Dataset covers **{date_min.date()} to {date_max.date()}** with **{n_rows:,} rows** and "
        + (f"**{int(n_orders):,} orders**." if not np.isnan(n_orders) else "orders info unavailable.")
    )
else:
    summary_lines.append(f"Dataset contains **{n_rows:,} rows**." + (f" **{int(n_orders):,} orders**." if not np.isnan(n_orders) else ""))

# 2) Revenue/Profit
if REVENUE:
    line = f"Total revenue is **{format_money(total_revenue)}**"
    if not np.isnan(aov):
        line += f" with an average order value (AOV) of **{format_money(aov)}**"
    line += "."
    summary_lines.append(line)

if PROFIT and not np.isnan(gross_margin):
    summary_lines.append(
        f"Estimated profit is **{format_money(total_profit)}**, implying a gross margin of **{format_pct(gross_margin)}**."
    )

# 3) Concentration: channels
if top_channels_rev is not None and len(top_channels_rev) > 0:
    top1_name = str(top_channels_rev.index[0])
    top1_val = float(top_channels_rev.iloc[0])
    share = (top1_val / total_revenue) if (REVENUE and total_revenue and not np.isnan(total_revenue) and total_revenue != 0) else np.nan
    if not np.isnan(share):
        summary_lines.append(
            f"Revenue concentration: top channel **{top1_name}** contributes **{format_money(top1_val)}** "
            f"({format_pct(share)} of revenue)."
        )
    else:
        summary_lines.append(f"Top channel by revenue is **{top1_name}** ({format_money(top1_val)}).")

# 4) Concentration: SKUs
if top_skus_rev is not None and len(top_skus_rev) > 0:
    sku1 = str(top_skus_rev.index[0])
    sku1_val = float(top_skus_rev.iloc[0])
    summary_lines.append(f"Top SKU by revenue is **{sku1}** with **{format_money(sku1_val)}**.")

# 5) Returns
if not np.isnan(return_rate):
    summary_lines.append(f"Overall return/refund rate is **{format_pct(return_rate)}**.")

# 6) Anomaly
if anomaly_day is not None and not np.isnan(anomaly_ratio) and anomaly_ratio >= 2.0:
    summary_lines.append(
        f"Anomaly candidate: **{anomaly_day}** revenue is ~**{anomaly_ratio:.1f}×** the median day. "
        "This may indicate promotions, bulk orders, or a data integrity issue."
    )

# Always end with "next actions"
summary_lines.append(
    "Next actions: validate high-impact channels/SKUs, investigate outlier dates, and add automated checks for missing values and schema drift."
)

st.subheader("Executive Summary")
for s in summary_lines:
    st.write(f"- {s}")

st.divider()

# -------------------------
# Drilldowns (tables + charts)
# -------------------------
left, right = st.columns(2)

with left:
    st.subheader("Top Channels")
    if top_channels_rev is not None:
        st.dataframe(top_channels_rev.rename("revenue"), width="stretch")
        st.bar_chart(top_channels_rev)
    elif top_channels_cnt is not None:
        st.dataframe(top_channels_cnt.rename("rows"), width="stretch")
        st.bar_chart(top_channels_cnt)
    else:
        st.info("Channel column not found.")

with right:
    st.subheader("Top SKUs")
    if top_skus_rev is not None:
        st.dataframe(top_skus_rev.rename("revenue"), width="stretch")
        st.bar_chart(top_skus_rev)
    elif top_skus_cnt is not None:
        st.dataframe(top_skus_cnt.rename("rows"), width="stretch")
        st.bar_chart(top_skus_cnt)
    else:
        st.info("SKU column not found.")

st.subheader("Trend")
if rev_by_day is not None and len(rev_by_day) > 0:
    st.line_chart(rev_by_day.rename("revenue"))
else:
    st.info("Need order_date + revenue (or price*qty) to show a trend chart.")

