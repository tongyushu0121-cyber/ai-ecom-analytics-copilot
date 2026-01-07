from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass
import os
import pandas as pd

@dataclass
class NarrativeInputs:
    kpi_delta: pd.DataFrame
    decomp: pd.DataFrame
    top_sku_sales: pd.DataFrame
    top_channel_sales: pd.DataFrame

def _fmt_money(x: float) -> str:
    return f"${x:,.2f}"

def _fmt_pct(x: float) -> str:
    return f"{x*100:.1f}%"

def generate_rule_based_summary(inp: NarrativeInputs) -> str:
    # Pull key numbers
    d = inp.kpi_delta.set_index("metric")
    gmv_delta = float(d.loc["GMV", "delta"]) if "GMV" in d.index else 0.0
    orders_delta = float(d.loc["ORDERS", "delta"]) if "ORDERS" in d.index else 0.0

    gp_delta = None
    if "GROSS_PROFIT" in d.index:
        gp_delta = float(d.loc["GROSS_PROFIT", "delta"])

    # Decomposition
    de = inp.decomp.set_index("component")["value"].to_dict()
    vol = float(de.get("Volume_effect", 0.0))
    price = float(de.get("Price_effect", 0.0))
    mix = float(de.get("Mix_effect", 0.0))

    # Top drivers
    sku_driver = None
    if len(inp.top_sku_sales) > 0 and "delta" in inp.top_sku_sales.columns:
        r = inp.top_sku_sales.iloc[0]
        sku_driver = (str(r.iloc[0]), float(r["delta"]))  # first col is sku

    ch_driver = None
    if len(inp.top_channel_sales) > 0 and "delta" in inp.top_channel_sales.columns:
        r = inp.top_channel_sales.iloc[0]
        ch_driver = (str(r.iloc[0]), float(r["delta"]))  # first col is channel

    lines: List[str] = []
    lines.append("## Executive Summary")
    lines.append(f"- GMV changed by {_fmt_money(gmv_delta)}; Orders changed by {orders_delta:+.0f}.")
    if gp_delta is not None:
        lines.append(f"- Gross Profit changed by {_fmt_money(gp_delta)}.")

    lines.append("")
    lines.append("## Why it changed (Price / Volume / Mix)")
    lines.append(f"- Volume effect: {_fmt_money(vol)}")
    lines.append(f"- Price effect: {_fmt_money(price)}")
    lines.append(f"- Mix effect: {_fmt_money(mix)} (residual, driven by SKU/channel composition)")

    lines.append("")
    lines.append("## Top Drivers")
    if sku_driver:
        lines.append(f"- Top SKU driver by GMV delta: **{sku_driver[0]}** ({_fmt_money(sku_driver[1])})")
    if ch_driver:
        lines.append(f"- Top Channel driver by GMV delta: **{ch_driver[0]}** ({_fmt_money(ch_driver[1])})")

    lines.append("")
    lines.append("## Recommended Actions (next 7 days)")
    lines.append("- Validate whether the change is driven by a few SKUs (stockouts, price changes, promo ending).")
    lines.append("- If Volume effect is negative: check inventory availability and fulfillment constraints by channel.")
    lines.append("- If Price effect is negative: review pricing changes, discounts, and shipping fees impacting net price.")
    lines.append("- If Mix effect is negative: shift traffic/supply toward higher-margin SKUs or channels.")

    lines.append("")
    lines.append("## Data Checks")
    lines.append("- Confirm order_date completeness and no missing days in the current window.")
    lines.append("- Verify SKU mapping consistency (no duplicate/renamed SKUs).")
    lines.append("- Validate unit_cost coverage if Gross Profit is used.")

    return "\n".join(lines)


def generate_ai_summary_with_openai(rule_summary: str, context: Dict[str, Any]) -> str:
    """
    Optional enhancement. Only runs if OPENAI_API_KEY is set.
    Keeps your app runnable without any key.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return rule_summary

    # Minimal, safe dependency: use requests (already common). If missing, fall back.
    try:
        import requests
    except Exception:
        return rule_summary

    prompt = f"""
You are an analytics copilot. Rewrite the following rule-based diagnosis into a concise, interview-ready narrative.
- Keep it factual and consistent with the numbers.
- Use bullet points, then a short paragraph recommendation.
- Do not invent data.

RULE SUMMARY:
{rule_summary}
"""

    # Use OpenAI Responses API style via HTTP (kept generic; you can swap later).
    # If this fails for any reason, we fall back to rule_summary.
    try:
        r = requests.post(
            "https://api.openai.com/v1/responses",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": "gpt-4.1-mini",
                "input": prompt,
                "temperature": 0.2,
            },
            timeout=20,
        )
        if r.status_code != 200:
            return rule_summary
        data = r.json()
        # Responses API: extract output text if available
        # Robust extraction:
        txt = ""
        for item in data.get("output", []):
            for c in item.get("content", []):
                if c.get("type") == "output_text":
                    txt += c.get("text", "")
        return txt.strip() or rule_summary
    except Exception:
        return rule_summary
def call_openai_text(prompt: str, model: str = "gpt-4.1-mini", temperature: float = 0.2) -> str:
    """
    Generic LLM call. Returns "" if no key or request fails.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return ""

    try:
        import requests
    except Exception:
        return ""

    try:
        r = requests.post(
            "https://api.openai.com/v1/responses",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "input": prompt,
                "temperature": temperature,
            },
            timeout=25,
        )
        if r.status_code != 200:
            return ""

        data = r.json()
        txt = ""
        for item in data.get("output", []):
            for c in item.get("content", []):
                if c.get("type") == "output_text":
                    txt += c.get("text", "")
        return txt.strip()
    except Exception:
        return ""
def call_openai_text(prompt: str, model: str = "gpt-4.1-mini", temperature: float = 0.2):
    """
    Returns: (text, debug_message)
    """
    import os

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "", "OPENAI_API_KEY is missing (env not loaded)."

    try:
        import requests
    except Exception as e:
        return "", f"requests not available: {e}"

    try:
        r = requests.post(
            "https://api.openai.com/v1/responses",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "input": prompt,
                "temperature": temperature,
            },
            timeout=25,
        )

        if r.status_code != 200:
            # show a short server message
            try:
                j = r.json()
                msg = j.get("error", {}).get("message", str(j))[:400]
            except Exception:
                msg = (r.text or "")[:400]
            return "", f"HTTP {r.status_code}: {msg}"

        data = r.json()
        txt = ""
        for item in data.get("output", []):
            for c in item.get("content", []):
                if c.get("type") == "output_text":
                    txt += c.get("text", "")
        txt = txt.strip()
        if not txt:
            return "", "Response OK but empty output_text."
        return txt, "OK"
    except Exception as e:
        return "", f"Request failed: {e}"

