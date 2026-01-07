import streamlit as st
import matplotlib.pyplot as plt

from datetime import timedelta
from utils.diagnostics import slice_by_date, compute_kpis, kpi_delta, drivers, price_volume_mix

st.set_page_config(page_title="Diagnostics", layout="wide")
st.title("Diagnostics: What changed and why?")

if "orders_df" not in st.session_state:
    st.warning("No dataset found. Please go to Home page and upload a CSV first.")
    st.stop()

df = st.session_state["orders_df"]

min_d = df["order_date"].min().date()
max_d = df["order_date"].max().date()

st.subheader("Compare two time windows")




# choose a default window length (up to 7 days)
total_days = (max_d - min_d).days + 1
win = min(7, max(1, total_days // 2))

curr_end_default = max_d
curr_start_default = max_d - timedelta(days=win - 1)

prev_end_default = curr_start_default - timedelta(days=1)
prev_start_default = prev_end_default - timedelta(days=win - 1)

# clamp to min_d
if prev_start_default < min_d:
    prev_start_default = min_d

curr_range = st.date_input(
    "Current window",
    value=(curr_start_default, curr_end_default),
    min_value=min_d,
    max_value=max_d,
    key="curr",
)

prev_range = st.date_input(
    "Previous window",
    value=(prev_start_default, prev_end_default),
    min_value=min_d,
    max_value=max_d,
    key="prev",
)
curr_start, curr_end = curr_range
prev_start, prev_end = prev_range


df_curr = slice_by_date(df, curr_start, curr_end)
df_prev = slice_by_date(df, prev_start, prev_end)

curr = compute_kpis(df_curr)
prev = compute_kpis(df_prev)

st.subheader("KPI change summary")
st.dataframe(kpi_delta(curr, prev), width="stretch")

st.divider()

st.subheader("GMV decomposition (Price / Volume / Mix)")
decomp = price_volume_mix(df_curr, df_prev, by="sku")
st.dataframe(decomp, width="stretch")

fig = plt.figure()
plt.bar(decomp["component"], decomp["value"])
plt.xticks(rotation=30, ha="right")
plt.ylabel("Value")
st.pyplot(fig, clear_figure=True)

st.divider()

st.subheader("Top drivers (who moved the metric)")

a, b = st.columns(2)
with a:
    st.markdown("**Top SKU drivers (GMV delta)**")
    st.dataframe(drivers(df_curr, df_prev, by="sku", metric="sales", top_n=10), width="stretch")

    st.markdown("**Top SKU drivers (Units delta)**")
    st.dataframe(drivers(df_curr, df_prev, by="sku", metric="units", top_n=10), width="stretch")

with b:
    st.markdown("**Top Channel drivers (GMV delta)**")
    st.dataframe(drivers(df_curr, df_prev, by="channel", metric="sales", top_n=10), width="stretch")

    st.markdown("**Top Channel drivers (Orders delta)**")
    st.dataframe(drivers(df_curr, df_prev, by="channel", metric="orders", top_n=10), width="stretch")
# =========================
# AI Copilot: Narrative Summary
# =========================
from app.utils.ai_narrative import (
    NarrativeInputs,
    generate_rule_based_summary,
    generate_ai_summary_with_openai,
)

st.divider()
st.subheader("AI Copilot: Narrative Summary")

# Prepare driver tables for narrative (recompute here to avoid refactoring)
top_sku_sales = drivers(df_curr, df_prev, by="sku", metric="sales", top_n=10)
top_channel_sales = drivers(df_curr, df_prev, by="channel", metric="sales", top_n=10)

inp = NarrativeInputs(
    kpi_delta=kpi_delta(curr, prev),
    decomp=decomp,
    top_sku_sales=top_sku_sales,
    top_channel_sales=top_channel_sales,
)

rule_text = generate_rule_based_summary(inp)

if st.button("Generate Summary"):
    final_text = generate_ai_summary_with_openai(rule_text, context={})
    st.session_state["ai_summary"] = final_text

if "ai_summary" in st.session_state:
    st.markdown(st.session_state["ai_summary"])
    st.code(st.session_state["ai_summary"], language="markdown")
