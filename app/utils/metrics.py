import pandas as pd

def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["sales"] = out["quantity"] * out["unit_price"]

    if "unit_cost" in out.columns and out["unit_cost"].notna().any():
        out["cogs"] = out["quantity"] * out["unit_cost"]
        out["gross_profit"] = out["sales"] - out["cogs"]
    else:
        out["cogs"] = pd.NA
        out["gross_profit"] = pd.NA

    return out


def kpi_summary(df: pd.DataFrame) -> dict:
    df2 = add_derived_columns(df)

    gmv = float(df2["sales"].sum())
    orders = int(df2["order_id"].nunique())
    units = int(df2["quantity"].sum())

    has_profit = "gross_profit" in df2.columns and df2["gross_profit"].notna().any()
    gross_profit = float(df2["gross_profit"].sum()) if has_profit else None
    gross_margin = (gross_profit / gmv) if (has_profit and gmv != 0) else None

    if "is_returned" in df2.columns:
        return_rate = float(df2["is_returned"].mean())
    else:
        return_rate = None

    return {
        "gmv": gmv,
        "orders": orders,
        "units": units,
        "gross_profit": gross_profit,
        "gross_margin": gross_margin,
        "return_rate": return_rate,
    }


def time_series(df: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
    """
    freq: 'D' (daily) or 'W' (weekly)
    """
    df2 = add_derived_columns(df).copy()
    df2["order_date"] = pd.to_datetime(df2["order_date"])
    df2 = df2.set_index("order_date")

    agg = {
        "sales": "sum",
        "order_id": pd.Series.nunique,
        "quantity": "sum",
    }
    if "gross_profit" in df2.columns and df2["gross_profit"].notna().any():
        agg["gross_profit"] = "sum"

    ts = df2.resample(freq).agg(agg).rename(columns={"order_id": "orders", "quantity": "units"})
    ts = ts.reset_index()

    # Return rate by time bucket (if exists)
    if "is_returned" in df2.columns:
        rr = df2["is_returned"].resample(freq).mean().reset_index(name="return_rate")
        ts = ts.merge(rr, on="order_date", how="left")

    return ts


def top_breakdown(df: pd.DataFrame, by: str, metric: str, n: int = 10) -> pd.DataFrame:
    """
    by: 'sku' or 'channel'
    metric: 'sales' or 'gross_profit' or 'units' or 'orders'
    """
    df2 = add_derived_columns(df)

    if metric == "orders":
        out = df2.groupby(by)["order_id"].nunique().reset_index(name="orders")
        return out.sort_values("orders", ascending=False).head(n)

    if metric == "units":
        out = df2.groupby(by)["quantity"].sum().reset_index(name="units")
        return out.sort_values("units", ascending=False).head(n)

    if metric == "sales":
        out = df2.groupby(by)["sales"].sum().reset_index(name="sales")
        return out.sort_values("sales", ascending=False).head(n)

    if metric == "gross_profit":
        if "gross_profit" not in df2.columns or df2["gross_profit"].isna().all():
            return pd.DataFrame({by: [], "gross_profit": []})
        out = df2.groupby(by)["gross_profit"].sum().reset_index(name="gross_profit")
        return out.sort_values("gross_profit", ascending=False).head(n)

    raise ValueError("Unsupported metric")
