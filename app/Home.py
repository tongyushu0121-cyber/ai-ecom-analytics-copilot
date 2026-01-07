

from pathlib import Path

import os
import pandas as pd
import streamlit as st

from utils.data_loader import load_orders_csv

st.write("RUNNING FILE:", __file__)
st.write("CWD:", os.getcwd())

st.set_page_config(page_title="AI Ecom Analytics Copilot", layout="wide")
ROOT_DIR = Path(__file__).resolve().parents[1]  # repo 根目录
DEMO_CSV = ROOT_DIR / "data" / "synthetic_orders.csv"

st.title("AI E-commerce Analytics Copilot")
st.caption("Milestone 1: Upload + Data Profile (Synthetic data only)")
st.markdown("""
### What this demo shows
- Upload or load **synthetic orders data**
- Auto profile: date range, order/SKU/channel counts
- Missing value diagnostics + data preview

Next milestone: AI narrative insights (exec-ready summary + anomaly explanations)
""")
st.subheader("Quick start")
colA, colB = st.columns([1, 2])
with colA:
    if st.button("Load demo data (synthetic)", use_container_width=True):
        df = pd.read_csv(DEMO_CSV)
        df["order_date"] = pd.to_datetime(df["order_date"])
        st.session_state["orders_df"] = df
        st.success("Loaded demo data.")
        st.rerun()

with colB:
    st.info("No setup needed. Click the button to load synthetic data, or upload your own CSV (synthetic only).")
if "orders_df" in st.session_state:
    st.success("Data is ready. You can now open Dashboard / Diagnostics / AI Insights.")
else:
    st.warning("No data loaded yet.")

# =========================
# Sidebar (求职用)
# =========================
st.sidebar.header("About / Links")
st.sidebar.markdown("""
**Yushu Tong**  
Ops Analytics / E-commerce Analytics (LLM + BI)
- Email:tongyushu0121@gmail.com
- LinkedIn: www.linkedin.com/in/yushu-tong
- GitHub: https://github.com/tongyushu0121-cyber/data-analytics-portfolio
""")



# =========================
# Demo button
# =========================
demo_clicked = st.button("Load demo data (synthetic)")

if demo_clicked:
    df = pd.read_csv(DEMO_CSV)
    df["order_date"] = pd.to_datetime(df["order_date"])
    st.session_state["orders_df"] = df
    st.success("Loaded demo data.")

# =========================
# File uploader
# =========================
uploaded = st.file_uploader("Upload orders CSV", type=["csv"])

if uploaded:
    try:
        df = load_orders_csv(uploaded)
        st.session_state["orders_df"] = df
        st.success(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    except Exception as e:
        st.error(str(e))

# =========================
# Render if df exists
# =========================
if "orders_df" in st.session_state:
    df = st.session_state["orders_df"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Date Range",
        f"{df['order_date'].min().date()} → {df['order_date'].max().date()}"
    )
    c2.metric("Unique Orders", f"{df['order_id'].nunique():,}")
    c3.metric("Unique SKUs", f"{df['sku'].nunique():,}")
    c4.metric("Channels", f"{df['channel'].nunique():,}")

    st.subheader("Missing Values (Top 15)")
    st.dataframe(
        df.isna().sum().sort_values(ascending=False).head(15),
        width="stretch"
    )

    st.subheader("Preview")
    st.dataframe(df.head(50), width="stretch")

else:
    st.info("Upload a CSV or click 'Load demo data' to begin.")
st.markdown("> This demo uses synthetic data only. No real customer data is processed.")

