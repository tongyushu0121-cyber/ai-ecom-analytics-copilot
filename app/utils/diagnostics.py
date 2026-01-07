import pandas as pd

def _add_sales_profit(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["sales"] = out["quantity"] * out["unit_price"]
    if "unit_cost" in out.columns and out["unit_cost"].notna().any():
        out["gross_profit"] = out["quantity"] * (out["unit_price"] - out["unit_cost"])
    else:
        out["gross_profit"] = pd.NA
    return out

def slice_by_date(df: pd.DataFrame, start_date, end_date) -> pd.DataFrame:
    d = df.copy()
    return d[(d["order_date"].dt.date >= start_date) & (d["order_date"].dt.date <= end_date)]

def compute_kpis(df: pd.DataFrame) -> dict:
    d = _add_sales_profit(df)
    gmv = float(d["sales"].sum())
    orders = int(d["order_id"].nunique())
    units = int(d["quantity"].sum())
    aov = (gmv / orders) if orders else 0.0
    asp = (gmv / units) if units else 0.0

    if d["gross_profit"].notna().any():
        gp = float(d["gross_profit"].sum())
        gm = (gp / gmv) if gmv else 0.0
    else:
        gp, gm = None, None

    return {"gmv": gmv, "orders": orders, "units": units, "aov": aov, "asp": asp, "gross_profit": gp, "gross_margin": gm}

def kpi_delta(curr: dict, prev: dict) -> pd.DataFrame:
    rows = []
    for k in ["gmv", "orders", "units", "aov", "asp"]:
        rows.append([k.upper(), prev.get(k, 0), curr.get(k, 0), curr.get(k, 0) - prev.get(k, 0)])
    if curr.get("gross_profit") is not None and prev.get("gross_profit") is not None:
        rows.append(["GROSS_PROFIT", prev["gross_profit"], curr["gross_profit"], curr["gross_profit"] - prev["gross_profit"]])
        rows.append(["GROSS_MARGIN", prev["gross_margin"], curr["gross_margin"], curr["gross_margin"] - prev["gross_margin"]])
    return pd.DataFrame(rows, columns=["metric", "prev", "curr", "delta"])

def drivers(df_curr: pd.DataFrame, df_prev: pd.DataFrame, by: str, metric: str, top_n: int = 10) -> pd.DataFrame:
    c = _add_sales_profit(df_curr)
    p = _add_sales_profit(df_prev)

    if metric == "sales":
        c_agg = c.groupby(by)["sales"].sum().reset_index(name="curr")
        p_agg = p.groupby(by)["sales"].sum().reset_index(name="prev")
    elif metric == "units":
        c_agg = c.groupby(by)["quantity"].sum().reset_index(name="curr")
        p_agg = p.groupby(by)["quantity"].sum().reset_index(name="prev")
    elif metric == "orders":
        c_agg = c.groupby(by)["order_id"].nunique().reset_index(name="curr")
        p_agg = p.groupby(by)["order_id"].nunique().reset_index(name="prev")
    elif metric == "gross_profit":
        if c["gross_profit"].isna().all() or p["gross_profit"].isna().all():
            return pd.DataFrame({by: [], "prev": [], "curr": [], "delta": []})
        c_agg = c.groupby(by)["gross_profit"].sum().reset_index(name="curr")
        p_agg = p.groupby(by)["gross_profit"].sum().reset_index(name="prev")
    else:
        raise ValueError("Unsupported metric")

    m = c_agg.merge(p_agg, on=by, how="outer").fillna(0)
    m["delta"] = m["curr"] - m["prev"]
    return m.sort_values("delta", ascending=False).head(top_n)

def price_volume_mix(df_curr: pd.DataFrame, df_prev: pd.DataFrame, by: str = "sku") -> pd.DataFrame:
    """
    Decompose GMV delta into:
    - Volume effect: (units_c - units_p) * price_p
    - Price effect: units_c * (price_c - price_p)
    - Mix effect: residual
    """
    c = df_curr.groupby(by).agg(units=("quantity", "sum"), price=("unit_price", "mean")).reset_index()
    p = df_prev.groupby(by).agg(units=("quantity", "sum"), price=("unit_price", "mean")).reset_index()
    m = c.merge(p, on=by, how="outer", suffixes=("_c", "_p")).fillna(0)

    gmv_c = float((m["units_c"] * m["price_c"]).sum())
    gmv_p = float((m["units_p"] * m["price_p"]).sum())

    volume = float(((m["units_c"] - m["units_p"]) * m["price_p"]).sum())
    price = float((m["units_c"] * (m["price_c"] - m["price_p"])).sum())
    mix_effect = float((gmv_c - gmv_p) - volume - price)
    return pd.DataFrame(
        [
            ["GMV_prev", gmv_p],
            ["GMV_curr", gmv_c],
            ["Delta", gmv_c - gmv_p],
            ["Volume_effect", volume],
            ["Price_effect", price],
            ["Mix_effect", mix_effect],
        ],
        columns=["component", "value"],
    )

