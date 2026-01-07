from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from utils.data_loader import load_orders_csv
from utils.metrics import kpi_summary, time_series, top_breakdown

st.set_page_config(page_title="Dashboard", layout="wide")
st.title("KPI Dashboard")
st.caption("Uses session data loaded from Home. Optional upload can override current data.")

# -------------------------
# Optional uploader (override current df)
# -------------------------
uploaded = st.file_uploader(
    "Optional: upload another orders CSV to override the current dataset",
    type=["csv"]
)

if uploaded:
    try:
        df_new = load_orders_csv(uploaded)
        st.session_state["orders_df"] = df_new
        st.success(f"Dashboard now using uploaded data: {df_new.shape[0]:,} rows × {df_new.shape[1]} cols")
        st.rerun()
    except Exception as e:
        st.error(str(e))
        st.stop()

# -------------------------
# Require data in session_state
# -------------------------
if "orders_df" not in st.session_state:
    st.warning("No data found. Go to Home and click 'Load demo data' or upload a CSV first.")
    st.stop()

df = st.session_state["orders_df"].copy()
df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")

# -------------------------
# Filters
# -------------------------
min_d = df["order_date"].min().date()
max_d = df["order_date"].max().date()

c1, c2, c3 = st.columns([2, 2, 3])
with c1:
    date_range = st.date_input(
        "Date range",
        value=(min_d, max_d),
        min_value=min_d,
        max_value=max_d
    )
with c2:
    freq = st.selectbox("Time bucket", ["Daily", "Weekly"])
with c3:
    channels = sorted(df["channel"].dropna().unique())
    channel_filter = st.multiselect("Channel filter", channels, default=channels)

start_d, end_d = date_range
df_f = df[(df["order_date"].dt.date >= start_d) & (df["order_date"].dt.date <= end_d)].copy()
df_f = df_f[df_f["channel"].isin(channel_filter)].copy()

st.divider()

# -------------------------
# KPI summary
# -------------------------
kpi = kpi_summary(df_f)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("GMV", f"${kpi['gmv']:,.2f}")
k2.metric("Orders", f"{kpi['orders']:,}")
k3.metric("Units", f"{kpi['units']:,}")

if kpi["gross_profit"] is None:
    k4.metric("Gross Profit", "N/A")
    k5.metric("Gross Margin", "N/A")
else:
    k4.metric("Gross Profit", f"${kpi['gross_profit']:,.2f}")
    k5.metric("Gross Margin", f"{kpi['gross_margin']*100:.1f}%")

if kpi.get("return_rate") is not None:
    st.caption(f"Return rate: {kpi['return_rate']*100:.1f}%")

st.divider()

# -------------------------
# Time series
# -------------------------
freq_code = "D" if freq == "Daily" else "W"
ts = time_series(df_f, freq=freq_code)

st.subheader("Trends")

colA, colB = st.columns(2)

with colA:
    fig = plt.figure()
    plt.plot(ts["order_date"], ts["sales"])
    plt.xticks(rotation=30)
    plt.xlabel("Date")
    plt.ylabel("GMV (Sales)")
    st.pyplot(fig, clear_figure=True)

with colB:
    fig = plt.figure()
    plt.plot(ts["order_date"], ts["orders"])
    plt.xticks(rotation=30)
    plt.xlabel("Date")
    plt.ylabel("Orders")
    st.pyplot(fig, clear_figure=True)

if "gross_profit" in ts.columns:
    st.subheader("Profit Trend")
    fig = plt.figure()
    plt.plot(ts["order_date"], ts["gross_profit"])
    plt.xticks(rotation=30)
    plt.xlabel("Date")
    plt.ylabel("Gross Profit")
    st.pyplot(fig, clear_figure=True)

st.divider()

# -------------------------
# Top breakdowns
# -------------------------
st.subheader("Top Contributors")

c1, c2 = st.columns(2)

with c1:
    st.markdown("**Top SKUs by GMV**")
    top_sku_sales = top_breakdown(df_f, by="sku", metric="sales", n=10)
    st.dataframe(top_sku_sales, use_container_width=True)

    st.markdown("**Top SKUs by Gross Profit**")
    top_sku_profit = top_breakdown(df_f, by="sku", metric="gross_profit", n=10)
    st.dataframe(top_sku_profit, use_container_width=True)

with c2:
    st.markdown("**Top Channels by GMV**")
    top_ch_sales = top_breakdown(df_f, by="channel", metric="sales", n=10)
    st.dataframe(top_ch_sales, use_container_width=True)

    st.markdown("**Top Channels by Orders**")
    top_ch_orders = top_breakdown(df_f, by="channel", metric="orders", n=10)
    st.dataframe(top_ch_orders, use_container_width=True)

