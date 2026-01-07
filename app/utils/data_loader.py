import pandas as pd

REQUIRED_COLUMNS = [
    "order_id", "order_date", "channel", "sku", "quantity", "unit_price"
]

OPTIONAL_NUMERIC_COLUMNS = ["unit_cost", "pick_time_sec"]
OPTIONAL_BOOL_COLUMNS = ["is_returned"]

def load_orders_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df.columns = [c.strip() for c in df.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")

    for c in OPTIONAL_NUMERIC_COLUMNS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in OPTIONAL_BOOL_COLUMNS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    invalid = df["order_date"].isna() | df["quantity"].isna() | df["unit_price"].isna()
    df = df.loc[~invalid].copy()

    df = df[df["quantity"] > 0]
    df = df[df["unit_price"] >= 0]

    return df
