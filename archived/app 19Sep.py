# -*- coding: utf-8 -*-
"""
CGT Tool for DEGIRO CSV
- Annual Summary with tabs (Shares CGT / ETFs Exit Tax / Combined / Dividends)
- Dividend handling (gross in Total; tax in Fee; no EUR)
- ETF Exit Tax at 41%
- Money Market fund price change rows excluded
- EUR fallbacks for trades when FX is 'EUR' or 1.0
- FIFO realised P/L (EUR) for sells
- CGT exemption toggle + value
- Loss carry-forward mechanics
- Optional summary columns (B/F Loss Used, Exemption Used, Carry Forward, Net Cashflow, Total Fees)
- Opening lots (manual import)
"""
from __future__ import annotations

import re
from typing import Optional, Tuple, List, Dict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from pandas.core.groupby import grouper
import streamlit as st

__isin_roll_map: dict[str, str] = {}

# ---------------- Page config ----------------
st.set_page_config(page_title="CGT Tool", layout="wide")
st.title("📈 Irish CGT Tool")

# ------------ Key differences (columns for Loss offset & When due) ------------
st.markdown("""
#### Key differences

| Regime | Tax basis | Standard rate | Annual exemption | Loss offset | When due |
|---|---|---|---|---|---|
| **Shares (CGT)** | Capital gains | **33%** | **€1,270** per person | **Allowed** (same-year or carried forward) | On disposal |
| **ETFs (Exit Tax)** | Deemed exit tax on gains | **41%** | **None** | **Not applicable** | On gain events (disposals) |
| **ETFs (Deemed Disposal)** | Deemed disposal every 8 years (Fair Market Value input) | **41%** | **None** | **Not applicable** | Every 8 years from acquisition |
| **Dividends** | Income tax rules (not CGT) | N/A here | N/A | N/A | On receipt (withholding may apply) |
""")

# ---------------- Sidebar: Import & settings ----------------
with st.sidebar:
    st.markdown("### 📤 Upload Transactions")

    uploads = st.file_uploader("CSV file(s)", type=["csv"], accept_multiple_files=True, label_visibility="collapsed")

    with st.expander("📥 Upload Missing Transactions", expanded=False):
        st.caption("Upload either minimal `ISIN,Quantity,UnitPrice` or rich `ISIN,Currency,Quantity,Price,Currency Value,EUR Value`.")
        ol_mode = st.radio("Input method", options=["Upload CSV", "Manual entry"], horizontal=True)
        opening_lots_df = None
        if ol_mode == "Upload CSV":
            ol_file = st.file_uploader("Opening lots file", type=["csv"], key="ol_csv")
            if ol_file is not None:
                try:
                    df_ol = pd.read_csv(ol_file)
                except Exception:
                    ol_file.seek(0)
                    df_ol = pd.read_csv(ol_file, sep=";")
                opening_lots_df = df_ol
        else:
            sample = pd.DataFrame([{"ISIN":"", "Quantity":"", "UnitPrice":""}])
            edited = st.data_editor(sample, num_rows="dynamic", key="ol_editor")
            if edited is not None and len(edited) > 0:
                mask = edited[["ISIN","Quantity","UnitPrice"]].apply(lambda s: s.astype(str).str.strip() != "")
                valid = edited[mask.all(axis=1)]
                opening_lots_df = valid if not valid.empty else None

    with st.expander("🧾 Summary columns", expanded=False):
        show_bf_used    = st.checkbox("B/F Loss Used (EUR)", value=False)
        show_ex_used    = st.checkbox("Exemption Used (EUR)", value=False)
        show_carry_fw   = st.checkbox("Carry Forward (EUR)", value=False)
        show_cashflow   = st.checkbox("Net Cashflow (EUR)", value=False)
        show_total_fees = st.checkbox("Total Fees (EUR)", value=False)

    with st.expander("💶 CGT settings", expanded=False):
        use_exemption    = st.checkbox("Apply annual CGT exemption (Shares only)", value=True)
        exemption_val    = st.number_input("Exemption amount (EUR)", min_value=0.0, value=1270.0, step=10.0)
        cgt_rate_shares  = st.number_input("Shares CGT rate", min_value=0.0, max_value=1.0, value=0.33, step=0.01)
        exit_tax_rate_etf= st.number_input("ETFs Exit Tax rate", min_value=0.0, max_value=1.0, value=0.41, step=0.01)

    with st.expander("🎛️ Display options", expanded=False):
        show_buys = st.checkbox("Show Buys", value=True)
        show_sells = st.checkbox("Show Sells", value=True)
        show_dividends = st.checkbox("Show Dividends", value=True)
        show_corp = st.checkbox("Show Corporate actions", value=True)
        show_fees_interest = st.checkbox("Show Fee & Interest rows", value=False)

# ---------------- Helpers ----------------
SPLIT_RE_LIST = [
    re.compile(r'(\d+(?:\.\d+)?)\s*for\s*(\d+(?:\.\d+)?)', re.I),
    re.compile(r'(\d+(?:\.\d+)?)[x×]\s*split', re.I),
    re.compile(r'split\s*ratio\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*[:/]\s*(\d+(?:\.\d+)?)', re.I),
    re.compile(r'(\d+)\s*:\s*(\d+)', re.I),
]

ISIN_RE = re.compile(r'\b[A-Z]{2}[A-Z0-9]{9}\d\b')

def _detect_isin_product_mappings(base: pd.DataFrame):
    """
    Returns:
      isin_maps: list of (old_isin, new_isin, dt) tuples
      prod_maps: list of (isin, new_product, dt) tuples
    """
    isin_maps, prod_maps = [], []

    ca = base[base["__Type"].isin(["ISIN change", "Product change"])].copy()
    if ca.empty:
        return isin_maps, prod_maps

    for _, r in ca.iterrows():
        dt   = pd.to_datetime(r["Date"])
        desc = str(r.get("Description", "") or "")
        row_isin = str(r.get("ISIN", "") or "")
        row_prod = str(r.get("Product", "") or "")

        # ISIN change: try to find OLD -> NEW in the description
        if r["__Type"] == "ISIN change":
            isins = ISIN_RE.findall(desc)
            if len(isins) >= 2:
                old_isin, new_isin = isins[0], isins[1]
                isin_maps.append((old_isin, new_isin, dt))
            elif len(isins) == 1 and isins[0] != row_isin:
                # assume desc shows OLD and the row ISIN is NEW
                isin_maps.append((isins[0], row_isin, dt))

        # Product change: normalize Product label for this ISIN from this time onward
        if r["__Type"] == "Product change":
            if row_prod and row_isin:
                prod_maps.append((row_isin, row_prod, dt))

    # De-dup & sort by time
    isin_maps = sorted(list({(o, n, dt) for (o, n, dt) in isin_maps}), key=lambda x: x[2])
    prod_maps = sorted(list({(i, p, dt) for (i, p, dt) in prod_maps}), key=lambda x: x[2])
    return isin_maps, prod_maps

def parse_split_factor(description: str) -> Optional[float]:
    if not isinstance(description, str):
        return None
    d = description.strip()
    for rx in SPLIT_RE_LIST:
        m = rx.search(d)
        if m:
            try:
                a = float(m.group(1)); b = float(m.group(2)) if m.lastindex and m.lastindex >= 2 else 1.0
                if a > 0 and b > 0:
                    return a / b
            except Exception:
                continue
    return None

def _safe_col(df: pd.DataFrame, name: str) -> str:
    for c in df.columns:
        if c.strip().lower() == name.strip().lower():
            return c
    raise KeyError(f"Column '{name}' not found. Found: {list(df.columns)}")

def _find_between(df: pd.DataFrame, left: str, right: str) -> Optional[str]:
    """
    Return the column between `left` and `right` if it exists, else try 'Cash Movements',
    otherwise None (pipeline will handle None).
    """
    cols = list(df.columns)
    try:
        li = cols.index(left); ri = cols.index(right)
        if ri - li == 2:
            return cols[li + 1]
        if "Cash Movements" in df.columns:
            return "Cash Movements"
    except ValueError:
        if "Cash Movements" in df.columns:
            return "Cash Movements"
    return None

TRADE_RE = re.compile(
    r'(?P<type>Buy|Sell)\s+(?P<qty>\d+(?:[\.,]\d+)?)\s+.*?(?:@|at)\s*(?P<price>\d+(?:[\.,]\d+)?)\s*(?P<ccy>[A-Z]{3})?\b',
    re.IGNORECASE
)

def parse_desc_numbers(desc) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """
    Parse qty, price, currency from DEGIRO-style description lines like:
      "Sell 2,300 Bank of Ireland Group PLC@1.923 EUR (IE00...)"
    Rules:
      - If both '.' and ',' appear in the number, treat ',' as thousands sep and remove it.
      - If only ',' appears, treat it as thousands sep if there are exactly 3 digits after it
        and no '.' elsewhere; otherwise treat it as decimal comma.
      - If only '.' appears, treat it as decimal point.
    """
    if not isinstance(desc, str):
        return (None, None, None)

    m = TRADE_RE.search(desc)
    if not m:
        return (None, None, None)

    def _to_float(s: str) -> Optional[float]:
        s = s.strip()
        if not s:
            return None
        has_dot = "." in s
        has_com = "," in s

        if has_dot and has_com:
            # e.g. "2,300.50" => remove commas, parse float
            s2 = s.replace(",", "")
            try: return float(s2)
            except: return None

        if has_com and not has_dot:
            # Could be "2,300" (thousands) or "1,23" (decimal comma).
            parts = s.split(",")
            if len(parts) == 2 and len(parts[1]) == 3 and parts[0].isdigit() and parts[1].isdigit():
                # looks like thousands sep: "2,300"
                try: return float(parts[0] + parts[1])
                except: return None
            # otherwise treat as decimal comma: "1,23" -> "1.23"
            try: return float(s.replace(",", "."))
            except: return None

        # only dot or plain digits
        try: return float(s)
        except: return None

    qty_str   = m.group("qty")
    price_str = m.group("price")
    ccy       = m.group("ccy").upper() if m.group("ccy") else None

    qty = _to_float(qty_str)
    px  = _to_float(price_str)

    return qty, px, ccy

ETF_PROVIDERS = [
    "vanguard", "ishares", "vaneck", "wisdomtree",
    "amundi", "invesco", "xtrackers", "spdr",
    "lyxor", "ubs", "schwab"
]

ETF_TOKENS = ("etf", "ucits", "etn", "etc")

def infer_asset(product, description) -> str:
    def _has_tokens(x):
        if not isinstance(x, str):
            return False
        lx = x.lower()
        if any(tok in lx for tok in ETF_TOKENS):
            return True
        if any(provider in lx for provider in ETF_PROVIDERS):
            return True
        return False

    if _has_tokens(product) or _has_tokens(description):
        return "ETF"
    return "Share"

def parse_type(desc: any) -> str:
    if not isinstance(desc, str):
        return "Other"
    d = desc.lower().strip()

    # Treat explicit “Opening lot” rows as buys for consistency
    if "opening lot" in d:
        return "Buy"

    # --- ignore-only: Money Market fund price change rows ---
    if "money market fund price change" in d:
        return "MMF price change"

    # --- corporate actions (order matters; check these before generic dividends/trades) ---
    if "stock split" in d or d.startswith("stock split:"):
        return "Stock split"
    if "product change" in d or d.startswith("product change:"):
        return "Product change"
    if "isin change" in d or d.startswith("isin change:"):
        return "ISIN change"
    # additional corporate actions
    if "spin-off" in d or "spinoff" in d:
        return "Spin-off"
    if "merger" in d or "merged" in d:
        return "Merger"
    # --- Corporate actions ---
    if re.search(r"\breturn of capital\b", d) or "capital repayment" in d:
        return "Return of capital"
    if "scrip dividend" in d or "scrip" in d:
        return "Scrip dividend"
        # delisting
    if "delisting" in d or "corporate action cash settlement" in d:
        return "Sell"
    # NEW: spin-offs
    if "spin-off" in d or "spinoff" in d or "spin off" in d or "demerger" in d:
        return "Spin-off"
    # NEW: mergers
    if "merger" in d or "merged into" in d or "merging into" in d:
        return "Merger"
    # NEW: scrip/stock dividends (before plain 'dividend')
    if ("scrip" in d and "dividend" in d) or "stock dividend" in d:
        return "Scrip dividend"
    # --- dividends ---
    if "dividend tax" in d or "withholding tax" in d:
        return "Dividend Tax"
    if "dividend" in d:
        return "Dividend"
    # --- trades ---
    if "buy" in d or "purchase" in d:
        return "Buy"
    if "sell" in d or "sale" in d:
        return "Sell"
    # --- fx helper rows ---
    if d.startswith("fx credit"):
        return "FX Credit"
    if d.startswith("fx debit"):
        return "FX Debit"
    # --- other ledger items ---
    if "coupon" in d:
        return "Coupon"
    if "interest" in d:
        return "Interest"
    if "fee" in d or "commission" in d or "transaction" in d or "exchange" in d:
        return "Fee"

    return "Other"

def _direct_eur_from_rate(row: pd.Series) -> float:
    """
    If adapter gave a true FX_Rate and we parsed qty/price from description,
    compute signed EUR notional directly: sign * qty * price * FX_Rate.
    Returns NaN when not applicable.
    """
    try:
        if row["Type"] not in ("Buy", "Sell"):
            return np.nan
        rate = row.get("FX_Rate")
        if pd.isna(rate) or float(rate) <= 0 or abs(float(rate) - 1.0) < 1e-9:
            return np.nan
        qty = row.get("__qty_desc"); px = row.get("__price_desc")
        if pd.isna(qty) or pd.isna(px):
            return np.nan
        eur = float(qty) * float(px) * float(rate)
        # sign: Buy negative cash, Sell positive cash
        return -eur if row["Type"] == "Buy" else eur
    except Exception:
        return np.nan


# ---------------- FIFO basis ----------------
def fifo_cost_for_sell(history: pd.DataFrame, sell_row: pd.Series) -> float:
    """
    Compute FIFO basis (EUR) for a single Sell row.
    Correctly consumes earlier sells so later sells in the same instrument/day
    don't reuse the same buy quantities.

    `history` must be ALL rows strictly BEFORE the current sell_row in the
    consolidated table (your code already passes `consolidated.iloc[:i]`).
    """
    isin = str(sell_row["ISIN"])
    sell_dt = sell_row["Date"]

    # Only rows for same ISIN and strictly before current sell time
    hist = history[(history["ISIN"] == isin) & (history["Date"] <= sell_dt)].copy()
    if hist.empty:
        return float("nan")

    # Build buy lots (qty, unit_eur) in time order
    buys = hist[hist["Type"] == "Buy"].sort_values(["Date", "Order ID"], kind="mergesort")
    if buys.empty:
        return float("nan")

    lots = []
    for _, b in buys.iterrows():
        b_qty = float(abs(b.get("Quantity_signed", 0.0)) or 0.0)
        if b_qty <= 0:
            continue

        unit_eur = b.get("Price_EUR", np.nan)
        if pd.isna(unit_eur) or unit_eur == 0:
            # fallback to total EUR / qty
            tot_eur = b.get("Total_EUR_FeeAdj", np.nan)
            if pd.isna(tot_eur):
                tot_eur = b.get("Total_EUR", np.nan)
            if pd.isna(tot_eur):
                tot_eur = b.get("_CashValue", np.nan)
            if not pd.isna(tot_eur) and b_qty != 0:
                unit_eur = abs(float(tot_eur)) / b_qty
        if pd.isna(unit_eur) or unit_eur <= 0:
            unit_eur = 0.0

        lots.append([b_qty, float(unit_eur)])

    if not lots:
        return float("nan")

    # Compute ALL quantity sold BEFORE this sell and consume from the lots
    prior_sold_qty = float(
        abs(
            hist.loc[
                hist["Type"] == "Sell",
                "Quantity_signed"
            ].fillna(0.0)
        ).sum()
    )

    # Consume prior sells from the front of the FIFO lots
    i = 0
    while prior_sold_qty > 0 and i < len(lots):
        take = min(prior_sold_qty, lots[i][0])
        lots[i][0] -= take
        prior_sold_qty -= take
        if lots[i][0] <= 1e-12:  # remove empty lot
            i += 1
    lots = [lot for lot in lots[i:] if lot[0] > 1e-12]
    if not lots:
        return float("nan")

    # Now price THIS sell quantity from the remaining lots
    qty_to_match = abs(float(sell_row.get("Quantity_signed", 0.0)) or 0.0)
    if qty_to_match <= 0:
        return float("nan")

    cost = 0.0
    for lot_qty, lot_unit in lots:
        if qty_to_match <= 0:
            break
        take = min(qty_to_match, lot_qty)
        cost += take * lot_unit
        qty_to_match -= take

    if qty_to_match > 1e-12:
        # Not enough historical buys to cover this sell
        return float("nan")

    return float(cost)

# ---------------- Build dataset ----------------
from typing import Dict

# Columns used for the stock-split audit table
AUDIT_COLS = [
    "ISIN","Product","Split date","Factor","Row kind","Trade date","Order ID",
    "Qty (before)","Qty (after)","Unit px (before)","Unit px (after)",
    "Unit px EUR (before)","Unit px EUR (after)"
]

def _aggregate_non_fx(base: pd.DataFrame) -> pd.DataFrame:
    return grouper

def _apply_corporate_actions_and_map_fx(base: pd.DataFrame, grouped: pd.DataFrame, opening_lots: Optional[pd.DataFrame]) -> Tuple[pd.DataFrame, List[Dict]]:
    non_fx = base[~(base["__is_fx_credit"] | base["__is_fx_debit"])].copy()

    def _agg_type(series: pd.Series) -> str:
        priorities = ["Buy","Sell","Dividend","Dividend Tax","Stock split","Product change","ISIN change","Coupon","Interest","Other","Fee"]
        types = [x for x in series.dropna().tolist() if isinstance(x, str)]
        for tpe in priorities:
            if tpe in types:
                return tpe
        return types[0] if types else "Other"

    grouped = non_fx.groupby(["ISIN", "__EffID"], dropna=False, as_index=False).agg({
        "Date": "min",
        "__minute_key": "min",
        "Product": "first",
        "Order ID": "first",
        "Description": lambda s: " | ".join(sorted(set([str(x) for x in s if isinstance(x, str)]))),
        "Change": "sum",
        "_CashValue": "sum",
        "__Type": _agg_type,
        "__is_fee": "sum",
        "__year": "min",
        "__qty_desc": "max",
        "__price_desc": "max",
        "FX_Rate": "max",
        "FXCCY": "first",
        "__ccy_desc": "first",
    })
    grouped["Type"] = grouped["__Type"]
    # Normalized trade/ticker currency (from the description if present)
    grouped["TradeCCY"] = grouped["__ccy_desc"].astype(str).str.upper().replace({"NONE": "", "NAN": "", "": np.nan})

    # Quantities for trades
    def _qty_signed(row):
        if row["Type"] in ("Buy", "Sell"):
            q = row["__qty_desc"]
            if not pd.isna(q):
                return -abs(q) if row["Type"] == "Sell" else abs(q)
            return 0.0
        return 0.0
    grouped["Quantity_signed"] = grouped.apply(_qty_signed, axis=1).astype(float)

    # Fee aggregation (from raw fee lines)
    fees_only = non_fx[non_fx["__is_fee"]]
    if not fees_only.empty:
        fees_grouped = fees_only.groupby(["ISIN", "__EffID"], dropna=False, as_index=False)["_CashValue"].sum()
        fees_grouped.rename(columns={"_CashValue": "Fee_signed"}, inplace=True)
        grouped = grouped.merge(fees_grouped, on=["ISIN", "__EffID"], how="left")
    if "Fee_signed" not in grouped.columns:
        grouped["Fee_signed"] = 0.0
    grouped["Fee_signed"] = grouped["Fee_signed"].fillna(0.0)

    # Cash ex fees & unit price
    def _sign_for_type(tpe: str) -> int:
        if tpe == "Buy": return -1
        if tpe == "Sell": return +1
        return 0
    sign = grouped["Type"].map(_sign_for_type).fillna(0)
    grouped["_CashExFees"] = grouped["_CashValue"] - (sign * grouped["Fee_signed"])
    grouped["Price_from_desc"] = grouped["__price_desc"]
    grouped["Price_calc"] = np.where(grouped["Quantity_signed"].abs() > 0,
                                     (grouped["_CashExFees"].abs() / grouped["Quantity_signed"].abs()),
                                     np.nan)
    grouped["Price"] = grouped["Price_from_desc"].where(~grouped["Price_from_desc"].isna(), grouped["Price_calc"])

    # --- Attach order-level fees to split trades by Order ID (pro-rata on cash) ---
    # 1) Gather fees by (ISIN, Order ID)
    fees_oid = (
        non_fx.loc[non_fx["__is_fee"], ["ISIN", "Order ID", "_CashValue"]]
            .dropna(subset=["Order ID"])
            .groupby(["ISIN", "Order ID"], dropna=False)["_CashValue"]
            .sum()  # negative EUR (e.g. -2.00)
            .rename("__Fee_by_oid")
            .reset_index()
    )

    # 2) For trades only, compute each split trade's share of the order cash (abs)
    trades_only = grouped[grouped["Type"].isin(["Buy", "Sell"])].copy()
    order_cash = (
        trades_only.groupby(["ISIN", "Order ID"], dropna=False)["_CashValue"]
                .apply(lambda s: s.abs().sum())
                .rename("__OrderAbsCash")
                .reset_index()
    )

    grouped = grouped.merge(order_cash, on=["ISIN","Order ID"], how="left")
    grouped = grouped.merge(fees_oid,  on=["ISIN","Order ID"], how="left")

    # 3) Pro-rate fee to each split trade row; keep non-trade rows at 0
    fee_prorata = np.where(
        grouped["Type"].isin(["Buy","Sell"]) & grouped["__OrderAbsCash"].gt(0) & grouped["__Fee_by_oid"].notna(),
        # distribute (negative) fee by absolute cash weight
        grouped["__Fee_by_oid"] * (grouped["_CashValue"].abs() / grouped["__OrderAbsCash"]),
        0.0
    ).astype(float)

    # 4) Add to any existing Fee_signed (e.g. dividend tax already present)
    grouped["Fee_signed"] = grouped.get("Fee_signed", 0.0).fillna(0.0) + fee_prorata

    # after fee_prorata is computed and you've merged fees_oid
    if not fees_oid.empty:
        fee_rows_mask = grouped["Type"].eq("Fee") & grouped["Order ID"].isin(fees_oid["Order ID"])
        # Keep them visible if you want, but zero their amounts so they don't double count
        grouped.loc[fee_rows_mask, ["Fee_signed", "_CashValue", "Total_signed"]] = 0.0

    # Dividend handling (keep gross in Total; tax in Fee; no EUR)
    if "Total_signed" not in grouped.columns:
        grouped["Total_signed"] = np.nan

    is_div     = grouped["Type"].eq("Dividend")
    is_div_tax = grouped["Type"].eq("Dividend Tax")

    # Gross: try Change first, but fall back to _CashValue if NaN OR zero (groupby(sum) of all-NaN becomes 0.0)
    grouped.loc[is_div, "Total_signed"] = grouped.loc[is_div, "Change"]
    missing_gross = is_div & (grouped["Total_signed"].isna() | (grouped["Total_signed"].abs() < 1e-12))
    grouped.loc[missing_gross, "Total_signed"] = grouped.loc[missing_gross, "_CashValue"].abs()

    # Dividend tax -> Fee_signed; prefer Change if positive, else _CashValue
    tax_from_change = grouped.loc[is_div_tax, "Change"].abs()
    tax_from_cash   = grouped.loc[is_div_tax, "_CashValue"].abs()
    grouped.loc[is_div_tax, "Fee_signed"] = np.where(
        tax_from_change.fillna(0) > 0, tax_from_change, tax_from_cash.fillna(0.0)
    )

    # Don't carry a Total for the tax line
    grouped.loc[is_div_tax, "Total_signed"] = np.nan

    # For non-dividend rows default Total to net cash if missing
    mask_other = ~(is_div | is_div_tax)
    grouped.loc[mask_other & grouped["Total_signed"].isna(), "Total_signed"] = grouped.loc[mask_other, "_CashValue"]

    # Asset inference
    grouped["Asset"] = grouped.apply(lambda r: infer_asset(r["Product"], r["Description"]), axis=1)

    # -------- Robust FX→EUR mapping (DEGIRO) — now that `grouped` exists --------
    # Build FX ledger maps from `base` (credits = EUR in, debits = EUR out)
    fx = base[(base["__is_fx_credit"] | base["__is_fx_debit"])].copy()
    fx["__amt"] = pd.to_numeric(
        fx["Change"].where(fx["Change"].notna(), fx["_CashValue"]),
        errors="coerce"
    )

    fx_pos = fx[fx["__amt"] > 0].copy()   # credits (EUR)
    fx_neg = fx[fx["__amt"] < 0].copy()   # debits  (EUR)

    # EUR totals by (ISIN, Order ID)
    eur_credit_by_oid = (
        fx_pos.groupby(["ISIN", "Order ID"], dropna=False)["__amt"]
             .sum().abs().rename("__EUR_credit_by_oid").reset_index()
    )
    eur_debit_by_oid = (
        fx_neg.groupby(["ISIN", "Order ID"], dropna=False)["__amt"]
             .sum().abs().rename("__EUR_debit_by_oid").reset_index()
    )

    # EUR totals by (ISIN, minute) as a fallback when Order ID is missing
    eur_credit_by_min = (
        fx_pos.groupby(["ISIN", "__minute_key"], dropna=False)["__amt"]
             .sum().abs().rename("__EUR_credit_by_minute").reset_index()
             .rename(columns={"__minute_key": "__minute_key_fx1"})
    )
    eur_debit_by_min = (
        fx_neg.groupby(["ISIN", "__minute_key"], dropna=False)["__amt"]
             .sum().abs().rename("__EUR_debit_by_minute").reset_index()
             .rename(columns={"__minute_key": "__minute_key_fx2"})
    )

    # Merge FX maps into grouped
    grouped = grouped.merge(eur_credit_by_oid, on=["ISIN","Order ID"], how="left")
    grouped = grouped.merge(eur_debit_by_oid,  on=["ISIN","Order ID"], how="left")
    grouped = grouped.merge(
        eur_credit_by_min,
        left_on=["ISIN","__minute_key"], right_on=["ISIN","__minute_key_fx1"], how="left"
    )
    grouped = grouped.merge(
        eur_debit_by_min,
        left_on=["ISIN","__minute_key"], right_on=["ISIN","__minute_key_fx2"], how="left"
    )

    # --- Allocate EUR per Order ID across split trades by absolute cash weight ---
    # We already built __OrderAbsCash earlier for fee pro-rating. If not present, compute it.
    if "__OrderAbsCash" not in grouped.columns:
        trades_tmp = grouped[grouped["Type"].isin(["Buy","Sell"])].copy()
        order_cash = (
            trades_tmp.groupby(["ISIN", "Order ID"], dropna=False)["_CashValue"]
                     .apply(lambda s: s.abs().sum())
                     .rename("__OrderAbsCash")
                     .reset_index()
        )
        grouped = grouped.merge(order_cash, on=["ISIN","Order ID"], how="left")

    # Order-level allocation (preferred)
    order_buy_eur = np.where(
        grouped["Type"].eq("Buy") &
        grouped["__OrderAbsCash"].gt(0) &
        grouped["__EUR_debit_by_oid"].notna(),
        grouped["__EUR_debit_by_oid"] * (grouped["_CashValue"].abs() / grouped["__OrderAbsCash"]),
        np.nan
    )
    order_sell_eur = np.where(
        grouped["Type"].eq("Sell") &
        grouped["__OrderAbsCash"].gt(0) &
        grouped["__EUR_credit_by_oid"].notna(),
        grouped["__EUR_credit_by_oid"] * (grouped["_CashValue"].abs() / grouped["__OrderAbsCash"]),
        np.nan
    )

    # --- Fallback: allocate EUR at the minute-bucket level when Order ID is missing ---
    # Build minute-bucket abs cash per (ISIN, minute, Type)
    trades_min = grouped[grouped["Type"].isin(["Buy","Sell"])].copy()
    min_cash = (
        trades_min.groupby(["ISIN","__minute_key","Type"], dropna=False)["_CashValue"]
                  .apply(lambda s: s.abs().sum())
                  .rename("__MinuteAbsCash")
                  .reset_index()
    )
    grouped = grouped.merge(min_cash, on=["ISIN","__minute_key","Type"], how="left")

    minute_buy_eur = np.where(
        grouped["Type"].eq("Buy") &
        grouped["__MinuteAbsCash"].gt(0) &
        grouped["__EUR_debit_by_minute"].notna() &
        grouped["Total_EUR"].isna() if "Total_EUR" in grouped.columns else True,
        grouped["__EUR_debit_by_minute"] * (grouped["_CashValue"].abs() / grouped["__MinuteAbsCash"]),
        np.nan
    )
    minute_sell_eur = np.where(
        grouped["Type"].eq("Sell") &
        grouped["__MinuteAbsCash"].gt(0) &
        grouped["__EUR_credit_by_minute"].notna() &
        grouped["Total_EUR"].isna() if "Total_EUR" in grouped.columns else True,
        grouped["__EUR_credit_by_minute"] * (grouped["_CashValue"].abs() / grouped["__MinuteAbsCash"]),
        np.nan
    )

    # --- Prefer adapter-supplied FX_Rate if present (qty * price * rate) ---
    direct_rate_eur = grouped.apply(_direct_eur_from_rate, axis=1)

    # When order-level EUR is missing, prefer direct rate EUR; otherwise fall back to minute-level EUR.
    buy_eur_pref = np.where(
        ~pd.isna(order_buy_eur),
        order_buy_eur,
        np.where(~pd.isna(direct_rate_eur), direct_rate_eur, minute_buy_eur)
    )
    sell_eur_pref = np.where(
        ~pd.isna(order_sell_eur),
        order_sell_eur,
        np.where(~pd.isna(direct_rate_eur), direct_rate_eur, minute_sell_eur)
    )

    # Compose final Total_EUR for trades: prefer order-level allocation, then minute-level, else NaN
    total_eur_trade = np.where(
        grouped["Type"].eq("Buy"),
        buy_eur_pref,
        np.where(grouped["Type"].eq("Sell"), sell_eur_pref, np.nan)
    )

    # Assign to Total_EUR; keep non-trade rows as NaN for EUR
    grouped["Total_EUR"] = total_eur_trade

    # Keep dividends blank in EUR explicitly
    grouped.loc[grouped["Type"].isin(["Dividend", "Dividend Tax"]), "Total_EUR"] = np.nan

    # Fallbacks for trades that still have no EUR (e.g., EUR-denominated or FX=1.0)
    mask_eur_no_fx = (
        grouped["Total_EUR"].isna()
        & ~grouped["Type"].isin(["Dividend", "Dividend Tax"])
        & (grouped["FXCCY"].astype(str).str.upper().str.strip().eq("EUR")
           | (grouped["FX_Rate"].round(6) == 1.0))
    )
    grouped.loc[mask_eur_no_fx, "Total_EUR"] = grouped.loc[mask_eur_no_fx, "Total_signed"].abs()

    # Final fallback for trades: use native cash abs
    mask_trade_final_fallback = grouped["Total_EUR"].isna() & grouped["Type"].isin(["Buy", "Sell"])
    grouped.loc[mask_trade_final_fallback, "Total_EUR"] = grouped.loc[mask_trade_final_fallback, "Total_signed"].abs()

    # --- Fee-adjusted EUR totals (apply fees as allowable costs) ---
    is_trade = grouped["Type"].isin(["Buy", "Sell"])
    fee_abs = grouped["Fee_signed"].abs().fillna(0.0)

    grouped["Total_EUR_FeeAdj"] = grouped["Total_EUR"]
    grouped.loc[is_trade, "Total_EUR_FeeAdj"] = (
        grouped.loc[is_trade, "Total_EUR"].fillna(0.0)
        + np.where(grouped.loc[is_trade, "Type"].eq("Buy"),
                fee_abs.loc[is_trade],          # Buys: add fee (cost ↑)
                -fee_abs.loc[is_trade])         # Sells: subtract fee (proceeds ↓)
    )

    # Price in EUR for FIFO (fee-adjusted)
    grouped["Price_EUR"] = np.where(
        grouped["Quantity_signed"].abs() > 0,
        (grouped["Total_EUR_FeeAdj"] / grouped["Quantity_signed"].abs()),
        np.nan
    )

    # Opening lots (optional)
    if opening_lots is not None and not opening_lots.empty:
        ol = opening_lots.copy().rename(columns={c: c.strip() for c in opening_lots.columns})
        rich_ok = {"ISIN","Currency","Quantity","Price","EUR Value"}.issubset(set(ol.columns))
        simple_ok = {"ISIN","Quantity","UnitPrice"}.issubset(set(ol.columns))

        earliest = pd.to_datetime(grouped["Date"]).min()
        open_date = (earliest - timedelta(seconds=1)) if pd.notna(earliest) else datetime(1900,1,1)
        rows: List[dict] = []

        # Map ISIN -> last seen Product name (nicer labels for opening lots)
        _product_map = (
            grouped.loc[grouped["Product"].notna(), ["ISIN","Product"]]
                .drop_duplicates(subset=["ISIN"], keep="last")
                .set_index("ISIN")["Product"]
                .to_dict()
        )

        if rich_ok:
            _ol = ol.copy()
            _ol["Quantity"] = pd.to_numeric(_ol["Quantity"], errors="coerce")
            _ol["Price"] = pd.to_numeric(_ol["Price"], errors="coerce")
            _ol["EUR Value"] = pd.to_numeric(_ol["EUR Value"], errors="coerce")
            _ol["UnitPrice"] = _ol["EUR Value"] / _ol["Quantity"]

            for _, r in _ol.iterrows():
                try:
                    isin = str(r["ISIN"]).strip()
                    qty = float(r["Quantity"])
                    px_trade = float(r["Price"]) if not pd.isna(r["Price"]) else np.nan
                    eur_val = float(r["EUR Value"])
                    if qty <= 0 or eur_val <= 0:
                        continue
                    unit_eur = eur_val / qty
                except Exception:
                    continue

                # Build friendly label
                name_guess = _product_map.get(isin)
                label = (f"Missing Import - {name_guess}"
                        if isinstance(name_guess, str) and name_guess.strip()
                        else f"Missing Import - {isin}")

                rows.append({
                    "Date": open_date, "__minute_key": pd.to_datetime(open_date).floor("min"),
                    "Product": label,
                    "ISIN": isin, "Order ID": f"OPENING-{isin}", "Description": "Opening lot",
                    "Change": qty, "_CashValue": -eur_val, "Type": "Buy", "__Type": "Buy",
                    "__is_fee": 0.0, "__year": open_date.year, "__qty_desc": qty,
                    "__price_desc": px_trade, "Quantity_signed": qty, "Fee_signed": 0.0,
                    "_CashExFees": -eur_val, "Price_from_desc": px_trade, "Price_calc": px_trade,
                    "Price": px_trade, "Total_signed": -eur_val, "Asset": "Share",
                    "__EUR_by_oid": eur_val, "__EUR_by_minute": np.nan, "Total_EUR": eur_val,
                    "Price_EUR": unit_eur,
                })

        elif simple_ok:
            _ol = ol.copy()
            _ol["Quantity"] = pd.to_numeric(_ol["Quantity"], errors="coerce")
            _ol["UnitPrice"] = pd.to_numeric(_ol["UnitPrice"], errors="coerce")

            for _, r in _ol.iterrows():
                try:
                    isin = str(r["ISIN"]).strip()
                    qty = float(r["Quantity"])
                    unit_eur = float(r["UnitPrice"])
                    if qty <= 0 or unit_eur <= 0:
                        continue
                    eur_val = qty * unit_eur
                except Exception:
                    continue

                # Build friendly label
                name_guess = _product_map.get(isin)
                label = (f"Missing Import - {name_guess}"
                        if isinstance(name_guess, str) and name_guess.strip()
                        else f"Missing Import - {isin}")

                rows.append({
                    "Date": open_date, "__minute_key": pd.to_datetime(open_date).floor("min"),
                    "Product": label,
                    "ISIN": isin, "Order ID": f"OPENING-{isin}", "Description": "Opening lot",
                    "Change": qty, "_CashValue": -eur_val, "Type": "Buy", "__Type": "Buy",
                    "__is_fee": 0.0, "__year": open_date.year, "__qty_desc": qty,
                    "__price_desc": np.nan, "Quantity_signed": qty, "Fee_signed": 0.0,
                    "_CashExFees": -eur_val, "Price_from_desc": np.nan, "Price_calc": np.nan,
                    "Price": np.nan, "Total_signed": -eur_val, "Asset": "Share",
                    "__EUR_by_oid": eur_val, "__EUR_by_minute": np.nan, "Total_EUR": eur_val,
                    "Price_EUR": unit_eur,
                })

        if rows:
            grouped = pd.concat([pd.DataFrame(rows), grouped], ignore_index=True)

        # ---- ISIN/Product roll-forward (identity-level CA effects; no qty/price change) ----
    isin_maps, prod_maps = _detect_isin_product_mappings(base)

    # For each ISIN change OLD->NEW at time dt, rewrite *pre-change* rows of OLD to NEW
    for old_isin, new_isin, dt in isin_maps:
        pre_mask = (base["ISIN"].astype(str) == old_isin) & (pd.to_datetime(base["Date"]) < pd.to_datetime(dt))
        if pre_mask.any():
            base.loc[pre_mask, "ISIN"] = new_isin

    # Product label normalization from the change time onward (display only)
    for isin, new_prod, dt in prod_maps:
        prod_mask = (base["ISIN"].astype(str) == isin) & (pd.to_datetime(base["Date"]) >= pd.to_datetime(dt))
        if prod_mask.any():
            base.loc[prod_mask, "Product"] = new_prod

    # --- Audit store for split adjustments (debug table) ---
    split_audit: List[Dict] = []

    # --- Detect stock splits (ISIN, date, factor) ---

    split_events = []

    # Enhanced detector: handle pairs (Sell old → Buy new) and ISIN changes
    __isin_roll_map = {}  # old ISIN -> new ISIN discovered from split pairs

    # Grab split-like rows (robust against case/spacing)
    spmask = grouped["Description"].astype(str).str.contains(r"\bSTOCK SPLIT:", case=False, na=False)
    splits = grouped.loc[spmask, ["Date","Product","ISIN","Description","Order ID"]].copy()
    if not splits.empty:
        splits["Date"] = pd.to_datetime(splits["Date"], errors="coerce")

        # Parse description: e.g. "STOCK SPLIT: Sell 32 Foo@1,363 DKK (DK0060534915)"
        import re
        _pat = re.compile(
            r"STOCK\s+SPLIT:\s*(Buy|Sell)\s+([\d.,]+)\s+(.+?)@([\d.,]+)\s+[A-Z]{3}\s+\(([A-Z0-9]{12})\)",
            re.IGNORECASE
        )

        def _parse_desc(desc: str):
            m = _pat.search(str(desc))
            if not m:
                return pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan],
                                index=["action","qty","prod_desc","unit_px","desc_isin"])
            action, qty, prod_desc, unit_px, desc_isin = m.groups()
            def _to_f(x):
                x = str(x).replace(",", "").strip()
                try:
                    return float(x)
                except Exception:
                    return np.nan
            return pd.Series([
                action.capitalize(),
                _to_f(qty),
                (prod_desc or "").strip(),
                _to_f(unit_px),
                desc_isin
            ], index=["action","qty","prod_desc","unit_px","desc_isin"])

        splits[["action","qty","prod_desc","unit_px","desc_isin"]] = splits["Description"].apply(_parse_desc)

        # Event key: same minute + normalized product name from description (fallback to row Product)
        splits["k_time"] = splits["Date"].dt.floor("5min")
        splits["k_name"] = (
            splits["prod_desc"].where(splits["prod_desc"].notna(), splits["Product"])
            .astype(str).str.replace(r"\s+", " ", regex=True).str.strip().str.lower()
        )

        for (kt, kn), grp in splits.groupby(["k_time","k_name"], sort=False):
            a = grp.dropna(subset=["action","qty","desc_isin"])
            if a.empty:
                continue
            sells = a[a["action"].eq("Sell")]
            buys  = a[a["action"].eq("Buy")]
            # We expect exactly one Sell and one Buy line per split event
            if len(sells) == 1 and len(buys) == 1:
                sell = sells.iloc[0]
                buy  = buys.iloc[0]
                old_isin = str(sell["desc_isin"] or sell["ISIN"])
                new_isin = str(buy["desc_isin"]  or buy["ISIN"])
                sell_qty = float(sell["qty"] or 0.0)
                buy_qty  = float(buy["qty"]  or 0.0)
                if sell_qty > 0 and buy_qty > 0:
                    factor = buy_qty / sell_qty
                    # Record traditional split event (by old ISIN)
                    split_events.append((old_isin, pd.to_datetime(sell["Date"]), float(factor)))
                    # If the ISIN changed, record a roll-forward mapping too
                    if old_isin != new_isin:
                        __isin_roll_map[old_isin] = new_isin
                    # Audit for both legs (keeps your existing columns style)
                    split_audit.append([
                        old_isin, sell.get("Product",""), sell["Date"], factor, "Sell(old)",
                        sell["Date"], sell.get("Order ID",""),
                        np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
                    ])
                    split_audit.append([
                        new_isin, buy.get("Product",""), buy["Date"], factor, "Buy(new)",
                        buy["Date"], buy.get("Order ID",""),
                        np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
                    ])

    # 1) Description-based ratios (e.g., "1 for 5", "4:1")
    if not base.empty:
        _spl = base[base["__Type"].eq("Stock split")].copy()
        for _, r in _spl.iterrows():
            f = parse_split_factor(r.get("Description", ""))
            if f and f > 0 and not np.isclose(f, 1.0):
                split_events.append((str(r["ISIN"]), pd.to_datetime(r["Date"]), float(f)))

    # 2) DEGIRO paired "Stock split: Sell ... / Buy ..." rows on same day (infers factor = buy_qty / sell_qty)
    if not base.empty:
        splits_raw = base[base["__Type"].eq("Stock split")].copy()
        if not splits_raw.empty:
            desc_lower = splits_raw["Description"].astype(str).str.lower()
            splits_raw["__is_buy_desc"]  = desc_lower.str.contains(r"\bbuy\b",  na=False)
            splits_raw["__is_sell_desc"] = desc_lower.str.contains(r"\bsell\b", na=False)
            splits_raw["__day"] = pd.to_datetime(splits_raw["Date"]).dt.floor("D")

            splits_raw["__day"] = pd.to_datetime(splits_raw["Date"]).dt.floor("D")

            buys  = splits_raw[splits_raw["__is_buy_desc"]].groupby(["ISIN","__day"])["__qty_desc"].sum().rename("buy_qty")
            sells = splits_raw[splits_raw["__is_sell_desc"]].groupby(["ISIN","__day"])["__qty_desc"].sum().rename("sell_qty")

            agg = (
                pd.concat([buys, sells], axis=1)
                .fillna(0.0)
                .reset_index()
            )

            # Add first date (earliest timestamp for that ISIN/day)
            first_dt = splits_raw.groupby(["ISIN","__day"])["Date"].min().rename("first_dt").reset_index()
            agg = agg.merge(first_dt, on=["ISIN","__day"], how="left")

            inferred = []
            for _, r in agg.iterrows():
                bq, sq = float(r["buy_qty"]), float(r["sell_qty"])
                if bq > 0 and sq > 0:
                    f = bq / sq
                    if f > 0 and not np.isclose(f, 1.0):
                        inferred.append((str(r["ISIN"]), pd.to_datetime(r["first_dt"]), float(f)))

            # De-dup with description-based ones (same ISIN within ±1 day)
            if inferred:
                existing = [(str(i), pd.to_datetime(d), float(f)) for (i, d, f) in split_events]
                def _is_dup(isin_new, dt_new):
                    for (isin_old, dt_old, _f) in existing:
                        if isin_new == isin_old and abs(dt_new - dt_old) <= pd.Timedelta("1D"):
                            return True
                    return False
                for (isin, dt, f) in inferred:
                    if not _is_dup(isin, dt):
                        split_events.append((isin, dt, f))

    # --- Adjust opening lots for any subsequent splits (and audit) ---
    if split_events:
        is_open = grouped["Order ID"].astype(str).str.startswith("OPENING-")
        for isin, split_dt, factor in sorted(split_events, key=lambda x: (x[0], x[1])):
            mask = (
                is_open
                & grouped["ISIN"].astype(str).eq(isin)
                & pd.to_datetime(grouped["Date"]).lt(split_dt)
            )
            if not mask.any():
                continue

            idx = grouped.index[mask]
            q_before  = grouped.loc[idx, "Quantity_signed"].copy()
            p_before  = grouped.loc[idx, "Price"].copy()
            pe_before = grouped.loc[idx, "Price_EUR"].copy()

            # Apply scaling: qty * factor, unit prices / factor
            grouped.loc[idx, "Quantity_signed"] = grouped.loc[idx, "Quantity_signed"] * factor
            for col in ["Price", "Price_EUR"]:
                if col in grouped.columns:
                    grouped.loc[idx, col] = grouped.loc[idx, col] / factor

            # Audit rows
            for i in idx:
                split_audit.append({
                    "ISIN": str(grouped.at[i, "ISIN"]),
                    "Product": str(grouped.at[i, "Product"]),
                    "Row kind": "Opening lot (pre-split)",
                    "Trade date": pd.to_datetime(grouped.at[i, "Date"]),
                    "Order ID": str(grouped.at[i, "Order ID"]),
                    "Split date": pd.to_datetime(split_dt),
                    "Factor": float(factor),
                    "Qty (before)": float(q_before.get(i) if pd.notna(q_before.get(i)) else np.nan),
                    "Qty (after)":  float(grouped.at[i, "Quantity_signed"] if pd.notna(grouped.at[i, "Quantity_signed"]) else np.nan),
                    "Unit px (before)": float(p_before.get(i) if pd.notna(p_before.get(i)) else np.nan),
                    "Unit px (after)":  float(grouped.at[i, "Price"] if pd.notna(grouped.at[i, "Price"]) else np.nan),
                    "Unit px EUR (before)": float(pe_before.get(i) if pd.notna(pe_before.get(i)) else np.nan),
                    "Unit px EUR (after)":  float(grouped.at[i, "Price_EUR"] if pd.notna(grouped.at[i, "Price_EUR"]) else np.nan),
                })

    is_corp = grouped["Type"].isin(["Stock split","Product change","ISIN change"])
    if is_corp.any():
        grouped.loc[is_corp, ["Quantity_signed","Price","Price_EUR","Fee_signed",
                            "Total_signed","Total_EUR","Total_EUR_FeeAdj"]] = np.nan

    # Combine Dividend + Dividend Tax (single line)
    con = grouped.copy()
    is_div2 = con["Type"].eq("Dividend"); is_tax2 = con["Type"].eq("Dividend Tax")
    if (is_div2 | is_tax2).any():
        combined_rows = []
        for (isin, mkey, product), g in con[is_div2 | is_tax2].groupby(["ISIN","__minute_key","Product"], dropna=False):
            date = pd.to_datetime(g["Date"]).min()
            tax_amt = float(g.loc[g["Type"].eq("Dividend Tax"), "Fee_signed"].abs().sum())
            if (tax_amt == 0.0 or pd.isna(tax_amt)) and "Change" in g:
                tax_amt = float(g.loc[g["Type"].eq("Dividend Tax"), "Change"].abs().sum())
            if (tax_amt == 0.0 or pd.isna(tax_amt)) and "_CashValue" in g:
                tax_amt = float(g.loc[g["Type"].eq("Dividend Tax"), "_CashValue"].abs().sum())

            gross_series = g.loc[g["Type"] == "Dividend", "Total_signed"].dropna()
            if not gross_series.empty:
                gross_val = float(gross_series.abs().sum())
            else:
                gross_val = float(g.loc[g["Type"] == "Dividend", "_CashValue"].abs().sum())

            oid_series = g["Order ID"].dropna().astype(str)
            order_id = oid_series.iloc[0] if not oid_series.empty and oid_series.iloc[0].strip() else ""

            combined_rows.append({
                "Date": date, "__minute_key": pd.to_datetime(date).floor("min"),
                "Product": product,"ISIN": isin,"Order ID": order_id,
                "Description":"Dividend","Change": np.nan,"_CashValue": np.nan,
                "Type":"Dividend","Quantity_signed":0.0,"Fee_signed": tax_amt,
                "_CashExFees": np.nan,"Price_from_desc": np.nan,"Price_calc": np.nan,"Price": np.nan,
                "Total_signed": gross_val,"Asset": g["Asset"].iloc[0] if "Asset" in g.columns and len(g["Asset"]) else "Share",
                "__year": pd.to_datetime(date).year, "Total_EUR": np.nan,"Price_EUR": np.nan,
            })
        combined_df = pd.DataFrame(combined_rows)
        con = con[~(is_div2 | is_tax2)]
        con = pd.concat([con, combined_df], ignore_index=True)

    # Sort
    type_sort = {
        "Buy": 0, "Sell": 1, "Dividend": 2,
        "Stock split": 3, "Product change": 3, "ISIN change": 3,
        "Coupon": 4, "Interest": 4, "Fee": 5,
        "Delisting (non-cash)": 6,  # put after trades
        "Other": 7
    }
    con["__type_sort"] = con["Type"].map(type_sort).fillna(7).astype(int)

    # Make sure we have a stable tiebreaker for sorting later
    if "__row_id" not in con.columns:
        con["__row_id"] = np.arange(len(con))

    grouped = con
    return grouped, split_audit

def _consolidate_fifo(grouped: pd.DataFrame) -> pd.DataFrame:
    # Defensive: ensure __row_id exists
    if "__row_id" not in grouped.columns:
        grouped = grouped.copy()
        grouped["__row_id"] = np.arange(len(grouped))

    consolidated = grouped.sort_values(
        ["ISIN", "Date", "Order ID", "__type_sort", "__row_id"],
        kind="mergesort"
    ).reset_index(drop=True)

    # --- FIFO Gain/Loss (EUR) for sells using a running lot ledger (per ISIN) ---
    # Uses fee-adjusted proceeds and fee-adjusted buy cost
    gl = np.full(len(consolidated), np.nan)  # index-aligned result array

    # Work instrument-by-instrument in chronological order
    for isin, idx in consolidated.groupby("ISIN", sort=False).groups.items():
        open_lots = []  # each item: [qty_remaining (positive), unit_cost_eur]

        for i in idx:
            row = consolidated.loc[i]
            t = str(row.get("Type", ""))

            if t == "Buy":
                qty = float(row.get("Quantity_signed", 0.0) or 0.0)  # buys are +ve in your pipeline
                if qty > 0:
                    unit = row.get("Price_EUR", np.nan)

                    # Fallback if unit EUR is missing
                    if pd.isna(unit) or unit == 0:
                        tefa = row.get("Total_EUR_FeeAdj", np.nan)
                        if pd.isna(tefa):
                            tefa = row.get("Total_EUR", np.nan)
                        if pd.isna(tefa):
                            tefa = abs(float(row.get("_CashValue", 0.0) or 0.0))
                        if not pd.isna(tefa) and abs(qty) > 0:
                            unit = float(tefa) / abs(qty)

                    if not pd.isna(unit) and unit > 0:
                        open_lots.append([qty, float(unit)])

            elif t == "Sell":
                qty_to_match = abs(float(row.get("Quantity_signed", 0.0) or 0.0))
                cost = 0.0

                # Consume from FIFO lots
                j = 0
                while qty_to_match > 0 and j < len(open_lots):
                    lot_qty, lot_unit = open_lots[j]
                    take = min(qty_to_match, lot_qty)
                    cost += take * lot_unit
                    lot_qty -= take
                    qty_to_match -= take

                    if lot_qty <= 1e-12:
                        # lot fully consumed
                        open_lots.pop(j)
                    else:
                        open_lots[j][0] = lot_qty
                        j += 1

                # Proceeds (fee-adjusted if available)
                proceeds = row.get("Total_EUR_FeeAdj", np.nan)
                if pd.isna(proceeds):
                    proceeds = row.get("Total_EUR", np.nan)
                if pd.isna(proceeds):
                    proceeds = abs(float(row.get("_CashValue", 0.0) or 0.0))

                gl[i] = float(proceeds) - float(cost) if not pd.isna(proceeds) else np.nan

            # other row types (dividends/fees/corp actions) leave gl[i] as NaN

    consolidated["Gain/Loss"] = gl

    # Expose currency for display
    consolidated["Currency"] = consolidated.get("TradeCCY")
    return consolidated

def _build_out_table(consolidated: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "Date": consolidated["Date"],
        "Ticker - Name": consolidated["Product"],
        "ISIN": consolidated["ISIN"],
        "Order ID": consolidated["Order ID"],
        "Type": consolidated["Type"],
        "Asset": consolidated["Asset"],
        "Currency": consolidated["Currency"],
        "Quantity": np.where(consolidated["Type"].isin(["Buy","Sell"]), consolidated["Quantity_signed"].abs(), np.nan),
        "Price": np.where(consolidated["Type"].isin(["Buy","Sell"]), consolidated["Price"], np.nan),
        "Fee": consolidated["Fee_signed"].abs(),
        "Total": np.where(consolidated["Type"].isin(["Buy","Sell"]), consolidated["Total_signed"].abs(), consolidated["Total_signed"]),
        "Total (EUR)": consolidated["Total_EUR"],
        "Total (EUR, fee-adj)": consolidated.get("Total_EUR_FeeAdj"),
        "Gain/Loss": consolidated["Gain/Loss"],
        "Description": consolidated["Description"],
        "__year": pd.to_datetime(consolidated["Date"]).dt.year,
    })
    for col in ["Fee","Total","Total (EUR)","Total (EUR, fee-adj)","Gain/Loss"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    # --- Ensure Dividends carry through Gross (Total) and Tax (Fee) cleanly ---
    div_mask = consolidated["Type"].eq("Dividend")
    if div_mask.any():
        gross = pd.to_numeric(consolidated.loc[div_mask, "Total_signed"], errors="coerce").abs()
        tax   = pd.to_numeric(consolidated.loc[div_mask, "Fee_signed"],   errors="coerce").abs()

        # Push into out (same row order)
        out.loc[div_mask.values, "Total"] = gross.values
        out.loc[div_mask.values, "Fee"]   = tax.values

    # --- Build audit dataframe (debug view) ---
    return out

# ---------------- Broker adapter registry (start with DEGIRO only) ----------------
from typing import Callable

# A broker adapter converts a raw CSV dataframe into the canonical schema
# expected by the rest of the pipeline:
#   Date, (optional Time/Value date), Product, ISIN, Description, FX, Change, Balance, Order ID, Currency
BrokerAdapter = Callable[[pd.DataFrame], pd.DataFrame]

def parse_degiros_csv(df_raw: pd.DataFrame) -> pd.DataFrame:
    df_norm = _canonicalize_headers(df_raw)
    _validate_required_columns(df_norm)

    # Degiro exports are D-M-Y — be explicit
    if "Date" in df_norm.columns:
        df_norm["Date"] = pd.to_datetime(df_norm["Date"], errors="coerce", dayfirst=True)

    return df_norm

def detect_broker_from_headers(df_head: pd.DataFrame) -> str:
    cols = {str(c).strip().lower() for c in df_head.columns}

    degiro_signals = {"date","product","isin","description","change"}
    if degiro_signals <= cols:
        return "DEGIRO"

    t212_signals = {
        "action","time","isin","ticker","name",
        "no. of shares","price / share","currency (price / share)",
        "exchange rate","total","currency (total)","withholding tax"
    }
    if len(t212_signals & cols) >= 3:
        return "TRADING212"

    return "DEGIRO"

# --- CSV header normalization & validation (DEGIRO) ---
import re
from typing import Dict, List, Optional, Tuple

# Canonical -> accepted aliases (case/spacing insensitive)
HEADER_ALIASES: Dict[str, List[str]] = {
    "Date":        ["Date", "Datum"],
    "Time":        ["Time", "Tijd"],
    "Value date":  ["Value date", "Valuta datum", "Valutadatum", "Value-date", "Value_date"],
    "Product":     ["Product"],
    "ISIN":        ["ISIN"],
    "Description": ["Description", "Omschrijving"],
    "FX":          ["FX", "Exchange rate"],
    "Change":      ["Change", "Mutatie", "Amount"],
    # We will canonicalize *to* 'Balance' (even if the file says 'Cash Movements')
    "Balance":     ["Balance", "Cash Movements", "Cash movements", "Cash"],
    # Order id variants are common
    "Order ID":    ["Order ID", "Order Id", "OrderId", "Order", "Order-ID"],
    # Optional but sometimes present
    "Currency":    ["Currency", "Valuta"],
}

REQUIRED_ALL = ["Date", "Product", "ISIN", "Description", "Change"]
REQUIRED_ONE_OF = [["Balance"]]  # after canonicalization we expect 'Balance' to exist

def _norm(s: str) -> str:
    return re.sub(r"[^\w]+", " ", str(s).strip().lower())

def _canonicalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    # Build reverse lookup: normalized alias -> canonical
    rev: Dict[str, str] = {}
    for canon, aliases in HEADER_ALIASES.items():
        for a in aliases:
            rev[_norm(a)] = canon
    # Map current columns to canonical where possible
    rename: Dict[str, str] = {}
    for col in df.columns:
        key = _norm(col)
        if key in rev:
            rename[col] = rev[key]
    out = df.rename(columns=rename).copy()
    return out

def _validate_required_columns(df: pd.DataFrame) -> None:
    missing: List[str] = []
    cols = set(df.columns)
    for c in REQUIRED_ALL:
        if c not in cols:
            # Tell the user acceptable aliases if we have them
            missing.append(f"{c} (aliases: {', '.join(HEADER_ALIASES.get(c, [c]))})")
    for group in REQUIRED_ONE_OF:
        if not any(c in cols for c in group):
            pretty = " or ".join(group)
            missing.append(pretty + f" (aliases: {', '.join(sum((HEADER_ALIASES.get(c,[c]) for c in group), []))})")
    if missing:
        raise ValueError("Missing required columns: " + "; ".join(missing))

# ---------- FX resolution helpers ----------

def _parse_fx_cell(val: object) -> tuple[float|None, str]:
    """
    Returns (rate, ccy_hint)
      - If val looks like a float -> (rate, "")
      - If 'EUR' -> (None, 'EUR')
      - If 3-letter code (USD/GBP/...) -> (None, 'CCY')
      - Else -> (None, '')
    """
    s = str(val).strip()
    if not s or s.lower() == "nan":
        return (None, "")
    # numeric rate?
    try:
        return (float(str(s).replace(",", ".")), "")
    except Exception:
        pass
    up = s.upper()
    if up == "EUR":
        return (None, "EUR")
    if len(up) == 3 and up.isalpha():
        return (None, up)
    return (None, "")

def _compute_eur_cash_from_fx(
    change_series: pd.Series,
    fx_series: pd.Series
) -> pd.Series:
    """
    Computes an EUR cash fallback from (Change, FX):
      - If FX is numeric -> assume Change is in instrument currency,
        and 'Total EUR' was Change / FX when FX is quoted 'CCY per EUR' or vice-versa.
        Because brokers vary, we conservatively choose:
           if |Change| is small and rate < 5, treat as EUR already (return NaN)
           else use Change / rate  (most DEGIRO/T212 exports give rate ~1.05 -> EUR=Change/Rate)
      - If FX=='EUR' -> Change already EUR (return Change)
      - If FX is some other 3-letter code without rate -> cannot compute (NaN)
    This is a fallback used only when Cash Movements is missing/NaN.
    """
    rates, hints = zip(*fx_series.map(_parse_fx_cell).tolist())
    rates = pd.Series(rates, index=fx_series.index, dtype="float64")
    hints = pd.Series(hints, index=fx_series.index, dtype="object")

    # If explicitly EUR -> take Change as EUR
    eur_mask = hints.eq("EUR")
    out = pd.Series(np.nan, index=fx_series.index, dtype="float64")
    out.loc[eur_mask] = pd.to_numeric(change_series, errors="coerce").loc[eur_mask]

    # Numeric FX rate fallback: prefer Change / rate
    rate_mask = rates.notna()
    chg = pd.to_numeric(change_series, errors="coerce")
    out.loc[rate_mask] = chg.loc[rate_mask] / rates.loc[rate_mask]

    # Otherwise (unknown CCY & no rate) -> leave NaN
    return out

# ---------------- Trading 212 adapter ----------------
def parse_trading212_csv(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Trading 212 adapter -> canonical schema used by the pipeline.
    Canonical cols produced (order matters for _find_between):
      Date, Time, Value date, Product, ISIN, Description, FX, Change, Cash Movements, Balance, Order ID, Currency
    """
    df = df_raw.copy()

    def _num(x):
        s = pd.Series([x]).astype(str)
        # (1) strip currency symbols and spaces/commas
        s = s.str.replace(r"[,\s€$£]", "", regex=True)
        # (2) handle accounting negatives like (123.45)
        s = s.str.replace(r"^\((.*)\)$", r"-\1", regex=True)
        # (3) normalize decimal comma if it slipped through
        s = s.str.replace(",", ".", regex=False)
        return pd.to_numeric(s, errors="coerce").iloc[0]

    # Column lookups
    def col(name: str, alts: list[str] = []):
        # exact, case-sensitive
        for c in [name] + alts:
            if c in df.columns:
                return c
        # case-insensitive
        lower_map = {c.lower(): c for c in df.columns}
        for c in [name] + alts:
            if c.lower() in lower_map:
                return lower_map[c.lower()]
        # startswith match (handles "Total (EUR)", "Currency (Total)", etc.)
        want = [name] + alts
        for w in want:
            for colname in df.columns:
                if colname.lower().startswith(w.lower()):
                    return colname
        return None

    # prefer a timestamp column; Trading212 usually calls it "Time" but sometimes "Date"
    c_action   = col("Action")
    c_time     = col("Time", alts=["Date"])
    c_isin     = col("ISIN")
    c_ticker   = col("Ticker")
    c_name     = col("Name")
    c_id       = col("ID")
    c_qty      = col("No. of shares", alts=["Quantity", "No. of Shares"])
    c_px       = col("Price / share", alts=["Price"])
    c_px_ccy   = col("Currency (Price / share)", alts=["Currency"])
    c_exrate   = col("Exchange rate", alts=["FX rate"])
    # totals in account currency often come as "Total (EUR)" or similar
    c_total    = col("Total", alts=["Total (EUR)", "Total (GBP)", "Total (USD)"])
    # withholding tax variants
    c_wht      = col("Withholding tax", alts=["Withholding Tax", "Dividend Tax", "Tax"])
    # conversion fee
    c_ccyfee   = col("Currency conversion fee", alts=["FX fee", "Currency Conversion Fee"])

    rows = []

    for _, r in df.iterrows():
        action = str(r.get(c_action, "")).lower()
        time_s = str(r.get(c_time, "")).strip()
        # If the file had only a Date (no time), keep it – downstream handles date-only too.
        dt = pd.to_datetime(time_s, errors="coerce")
        isin = str(r.get(c_isin, "")).strip()
        name = str(r.get(c_name, "")).strip()
        ticker = str(r.get(c_ticker, "")).strip()
        oid = str(r.get(c_id, "")).strip()
        qty = _num(r.get(c_qty))
        px = _num(r.get(c_px))
        px_ccy = str(r.get(c_px_ccy, "")).upper().strip()
        exrate = pd.to_numeric(r.get(c_exrate), errors="coerce")
        total_eur = _num(r.get(c_total))
        wht_eur = _num(r.get(c_wht))
        fee_eur = _num(r.get(c_ccyfee))

        product = name if name else (ticker or isin)

        # Trades
        if "buy" in action:
            desc = f"Buy {qty:g} {product}@{px:g} {px_ccy}"
            change = -(qty * px)
            cash_eur = total_eur
            currency = px_ccy
            fx_field = f"{exrate}" if pd.notna(exrate) else ("EUR" if currency == "EUR" else "")
        elif "sell" in action:
            desc = f"Sell {qty:g} {product}@{px:g} {px_ccy}"
            change = +(qty * px)
            cash_eur = total_eur
            currency = px_ccy
            fx_field = f"{exrate}" if pd.notna(exrate) else ("EUR" if currency == "EUR" else "")
        elif "dividend" in action:
            desc = "Dividend"
            gross_eur = (total_eur or 0) + (wht_eur or 0)
            change = gross_eur
            cash_eur = total_eur
            currency = "EUR"
            fx_field = "EUR"
            if not oid:
                oid = f"DIV-{isin}-{dt.strftime('%Y%m%d') if pd.notna(dt) else 'NA'}"
            rows.append({
                "Date": dt, "Time": time_s, "Value date": None,
                "Product": product, "ISIN": isin, "Description": desc,
                "FX": fx_field, "Change": change,
                "Cash Movements": cash_eur, "Balance": None,
                "Order ID": oid, "Currency": currency,
            })
            if pd.notna(wht_eur) and abs(wht_eur) > 0:
                rows.append({
                    "Date": dt, "Time": time_s, "Value date": None,
                    "Product": product, "ISIN": isin, "Description": "Dividend Tax",
                    "FX": "EUR", "Change": abs(wht_eur),
                    "Cash Movements": None, "Balance": None,
                    "Order ID": f"TAX-{isin}-{dt.strftime('%Y%m%d') if pd.notna(dt) else 'NA'}",
                    "Currency": "EUR",
                })
            continue
        elif "interest" in action:
            desc = "Interest on cash" if "cash" in action else "Lending interest"
            change = total_eur
            cash_eur = total_eur
            currency = "EUR"
            fx_field = "EUR"
        elif "deposit" in action or "withdrawal" in action:
            continue
        else:
            desc, change, cash_eur, currency, fx_field = "Other", None, None, "", ""

        # Fee
        if pd.notna(fee_eur) and abs(fee_eur) > 0:
            rows.append({
                "Date": dt, "Time": time_s, "Value date": None,
                "Product": product, "ISIN": isin, "Description": "Fee: Currency conversion fee",
                "FX": "EUR", "Change": -abs(fee_eur),
                "Cash Movements": -abs(fee_eur), "Balance": None,
                "Order ID": f"FEE-{oid or isin}-{dt.strftime('%Y%m%d') if pd.notna(dt) else 'NA'}",
                "Currency": "EUR",
            })

        rows.append({
            "Date": dt, "Time": time_s, "Value date": None,
            "Product": product, "ISIN": isin, "Description": desc,
            "FX": fx_field, "Change": change,
            "Cash Movements": cash_eur, "Balance": None,
            "Order ID": oid, "Currency": currency,
        })

    out = pd.DataFrame(rows, columns=[
        "Date", "Time", "Value date", "Product", "ISIN", "Description",
        "FX", "Change", "Cash Movements", "Balance", "Order ID", "Currency"
    ])
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce", utc=False, dayfirst=False)

    # Ensure required canonical columns exist
    for col in ["Balance", "Value date", "Currency"]:
        if col not in out.columns:
            out[col] = None

    # Reorder to canonical order
    out = out[[
        "Date","Time","Value date","Product","ISIN","Description",
        "FX","Change","Cash Movements","Balance","Order ID","Currency"
    ]]

    # Make Date dtype explicit (T212 Time is ISO; keep dayfirst False)
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce", dayfirst=False)

    return out

ETF_PROVIDER_KEYWORDS = {
    "ishares", "vanguard", "vaneck", "xtrackers",
    "spdr", "lyxor", "invesco", "wisdomtree", "amundi",
    "first trust", "global x", "ubs", "hsbc"
}

def _is_etf_by_name(product_name: str) -> bool:
    s = str(product_name or "").lower()
    return any(k in s for k in ETF_PROVIDER_KEYWORDS)

# Register adapters here. For now, only DEGIRO. Future brokers add a new function and an entry below.
BROKER_ADAPTERS: Dict[str, BrokerAdapter] = {
    "DEGIRO": parse_degiros_csv,
    "TRADING212": parse_trading212_csv
}

def build_output(df_norm: pd.DataFrame, opening_lots: Optional[pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:

    # Ensure split collectors exist before any use
    split_events: List[Tuple[str, pd.Timestamp, float]] = []
    split_audit: List[List[object]] = []
    date_col    = _safe_col(df_norm, "Date")
    try:
        time_col = _safe_col(df_norm, "Time"); has_time = True
    except Exception:
        time_col = None; has_time = False

    product_col = _safe_col(df_norm, "Product")
    isin_col    = _safe_col(df_norm, "ISIN")
    descr_col   = _safe_col(df_norm, "Description")
    change_col  = _safe_col(df_norm, "Change")
    balance_col = _safe_col(df_norm, "Balance")
    orderid_col = _safe_col(df_norm, "Order ID")
    between_cash_col = _find_between(df_norm, change_col, balance_col)

    df = df_norm.copy()
    # Resolve EUR cash value: prefer broker-provided "Cash Movements" (between Change and Balance),
    # otherwise derive from Change & FX when possible.
    fx_colname = None
    try:
        fx_colname = _safe_col(df_norm, "FX")
    except Exception:
        pass

    cash_between = pd.to_numeric(df[_find_between(df_norm, change_col, balance_col)], errors="coerce")
    if fx_colname:
        eur_fallback = _compute_eur_cash_from_fx(df[change_col], df[fx_colname])
    else:
        eur_fallback = pd.Series(np.nan, index=df.index, dtype="float64")

    df["__cash_eur"] = cash_between.combine_first(eur_fallback)

    # Stable tiebreaker for same-minute rows across all branches
    if "__row_id" not in df.columns:
        df["__row_id"] = np.arange(len(df))

    def _parse_dt(row):
        dstr = str(row[date_col]).strip()
        if has_time:
            tstr = str(row[time_col]).strip()

            # If Time looks like time-only (e.g., "08:32" or "08:32:15"), combine with Date
            if re.match(r"^\d{1,2}:\d{2}(:\d{2})?$", tstr):
                # Combine: Date + time-only
                # Support common date formats + ISO
                # Try pandas first
                dparsed = pd.to_datetime(dstr, errors="coerce")
                if pd.notna(dparsed):
                    # replace hour/min/sec on the parsed date
                    try:
                        tparts = [int(x) for x in tstr.split(":")]
                        h = tparts[0]; m = tparts[1]; s = tparts[2] if len(tparts) > 2 else 0
                        return dparsed.replace(hour=h, minute=m, second=s).to_pydatetime()
                    except Exception:
                        pass
                # fallback: explicit patterns
                for dfmt in ("%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d", "%d.%m.%Y"):
                    try:
                        base = datetime.strptime(dstr, dfmt)
                        tparts = [int(x) for x in tstr.split(":")]
                        h = tparts[0]; m = tparts[1]; s = tparts[2] if len(tparts) > 2 else 0
                        return base.replace(hour=h, minute=m, second=s)
                    except Exception:
                        continue
                return pd.NaT

            # Otherwise, Time might already be a full timestamp (Trading212)
            tparsed = pd.to_datetime(tstr, errors="coerce")
            if pd.notna(tparsed):
                return tparsed.to_pydatetime()

            # As a fallback, try concatenating Date + Time (covers odd exports)
            dt_str = f"{dstr} {tstr}".strip()
            for fmt in (
                "%d-%m-%Y %H:%M", "%d/%m/%Y %H:%M", "%Y-%m-%d %H:%M", "%d.%m.%Y %H:%M",
                "%d-%m-%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f"
            ):
                try:
                    return datetime.strptime(dt_str, fmt)
                except Exception:
                    continue
            # Last resort: parse date only
            dparsed = pd.to_datetime(dstr, errors="coerce")
            return dparsed.to_pydatetime() if pd.notna(dparsed) else pd.NaT

        # No Time column: parse Date only
        dparsed = pd.to_datetime(dstr, errors="coerce")
        if pd.notna(dparsed):
            return dparsed.to_pydatetime()
        for fmt in ("%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d", "%d.%m.%Y"):
            try:
                return datetime.strptime(dstr, fmt)
            except Exception:
                continue
        return pd.NaT

    df["__dt"] = df.apply(_parse_dt, axis=1)
    df["__minute_key"] = pd.to_datetime(df["__dt"]).dt.floor("min")
    df["__year"] = pd.to_datetime(df["__dt"]).dt.year
    df["__Type"] = df[descr_col].apply(parse_type)
    df["__is_fee"] = df["__Type"].eq("Fee")
    df["__is_fx_credit"] = df["__Type"].eq("FX Credit")
    df["__is_fx_debit"]  = df["__Type"].eq("FX Debit")

    # Parse qty, price, and currency from description
    q_desc, p_desc, c_desc = zip(*df[descr_col].apply(parse_desc_numbers).tolist())
    df["__qty_desc"] = pd.to_numeric(q_desc, errors="coerce")
    df["__price_desc"] = pd.to_numeric(p_desc, errors="coerce")
    df["__ccy_desc"] = pd.Series(c_desc, dtype="object")

    # Fallback: if description didn’t contain a quantity (e.g., “Opening lot”), and
    # the DataFrame has a real Quantity column (from opening-lots ingest),
    # use it so FIFO can see these shares.
    if "Quantity" in df.columns:
        df["__qty_desc"] = df["__qty_desc"].fillna(pd.to_numeric(df["Quantity"], errors="coerce"))


    # FX col parsing
    try:
        fx_colname = _safe_col(df_norm, "FX")
        fx_raw  = df[fx_colname].astype(str).str.strip()
        fx_rate = pd.to_numeric(fx_raw.str.replace(",", ".", regex=False), errors="coerce")
        fx_ccy  = np.where(fx_rate.notna(), "", fx_raw.str.upper())
        fx_ccy  = pd.Series(fx_ccy).replace({"NAN": "", "": ""})
        df["__fx_rate"] = fx_rate
        df["__fx_ccy"]  = fx_ccy.astype(str).str.upper().str.strip()
    except Exception:
        df["__fx_rate"] = np.nan
        df["__fx_ccy"]  = ""
        # If broker left FX empty, assume EUR
        df["__fx_ccy"] = df["__fx_ccy"].mask(df["__fx_ccy"].eq(""), "EUR")

    order_ids = df[orderid_col].astype(object)
    order_ids = order_ids.where(~pd.isna(order_ids), "")
    order_ids = order_ids.astype(str)

    base = pd.DataFrame({
        "Date": df["__dt"],
        "__name__": "base",
        "__minute_key": df["__minute_key"],
        "Product": df[product_col],
        "ISIN": df[isin_col],
        "Description": df[descr_col],
        "Change": pd.to_numeric(df[change_col], errors="coerce"),
        "_CashValue": df["__cash_eur"],
        "Order ID": order_ids,
        "__Type": df["__Type"],
        "__is_fee": df["__is_fee"],
        "__is_fx_credit": df["__is_fx_credit"],
        "__is_fx_debit": df["__is_fx_debit"],
        "__year": df["__year"],
        "__qty_desc": df["__qty_desc"],
        "__price_desc": df["__price_desc"],
        "__ccy_desc": df["__ccy_desc"],
        "FX_Rate": df["__fx_rate"],
        "FXCCY": df["__fx_ccy"],
        "__row_id": df["__row_id"],
    })

    # --- Fix: Corporate Action Cash Settlement (delisting) with missing quantity ---
    # Some brokers (e.g., DEGIRO) emit two rows at delist:
    #   1) "Corporate Action Cash Settlement ..." (cash comes in) but no qty on the row
    #   2) "DELISTING: Sell N ..." (qty present) but zero cash
    # FIFO needs the qty on the cash row so it can close lots. We borrow the N from the delisting line.
    cash_delist_mask = (
        base["__Type"].eq("Sell")
        & base["Description"].str.contains("cash settlement", case=False, na=False)
        & (base["__qty_desc"].isna() | (base["__qty_desc"] == 0))
    )

    if cash_delist_mask.any():
        # Prefer to tie rows by Order ID; if missing, fall back to minute bucket.
        key_series = np.where(
            base["Order ID"].astype(str).str.len() > 0,
            base["Order ID"].astype(str),
            base["__minute_key"].astype(str)
        )
        key_series = pd.Series(key_series, index=base.index)

        # Extract quantity from any sibling "delisting: sell <N>" description
        delist_qty = base["Description"].str.extract(r"delisting.*sell\s+(\d+)", flags=re.I)[0]
        delist_qty = pd.to_numeric(delist_qty, errors="coerce")

        for idx in base.index[cash_delist_mask]:
            isin = base.at[idx, "ISIN"]
            key  = key_series.at[idx]

            sib_candidates = base.index[
                (base["ISIN"] == isin)
                & (key_series == key)
                & delist_qty.notna()
            ]
            if len(sib_candidates) > 0:
                base.at[idx, "__qty_desc"] = float(delist_qty.loc[sib_candidates].iloc[0])

    # --- Delisting zero-cash guard (prevents double-count closes) ---
    # If a row is classified as Sell, description mentions delisting, and has zero EUR cash,
    # BUT there exists another row in the same minute/order with positive EUR cash,
    # then downgrade this line so only the cash-bearing settlement acts as the Sell.

    # Build a grouping key: prefer Order ID if present, else minute bucket
    order_id_str = base["Order ID"].astype(str)
    gkey = np.where(order_id_str.str.len() > 0, order_id_str, base["__minute_key"].astype(str))

    cash_mask = base["_CashValue"].fillna(0) > 0
    cash_keys = set(pd.Series(gkey)[cash_mask])

    mask_delist_zero = (
        base["__Type"].eq("Sell")
        & base["Description"].str.contains("delisting", case=False, na=False)
        & (base["_CashValue"].fillna(0) == 0)
    )

    to_downgrade = mask_delist_zero & pd.Series(gkey).isin(cash_keys)

    # Make the zero-cash delisting row inert; the cash-settlement row remains the true Sell
    base.loc[to_downgrade, "__Type"] = "Delisting (non-cash)"


    # Stable row id used for deterministic sorting & FIFO tie-breaks
    base["__row_id"] = base.index.astype(int)
    base["__row_id"] = np.arange(len(base))

    # Drop MMF / cash sweep rows (trades + price-change helpers) — Degiro only
    looks_degiro = "Value date" in df_norm.columns and "Product" in df_norm.columns

    if looks_degiro:
        MMF_KEYWORDS = ("liquidity fund", "money market", "cash fund", " mmf ")
        MMF_ISINS = {"LU0904783973"}  # Morgan Stanley EUR Liquidity Fund (extend if needed)

        _prod = base["Product"].astype(str).str.lower()
        _desc = base["Description"].astype(str).str.lower()

        _mmf_name_mask = pd.Series(False, index=base.index)
        for kw in MMF_KEYWORDS:
            _mmf_name_mask = _mmf_name_mask | _prod.str.contains(kw, na=False)

        _mmf_mask = (
            _mmf_name_mask |
            _desc.str.contains("money market fund price change", na=False) |
            base["ISIN"].isin(MMF_ISINS)
        )

        if _mmf_mask.any():
            base = base.loc[~_mmf_mask].copy()

    # --- Apply inferred ISIN roll-forward mappings from paired split rows ---
    # (Requires Patch A which builds __isin_roll_map after detecting paired split Buy/Sell rows)
    if '__isin_roll_map' in locals() and __isin_roll_map:
        # 1) Normalize base ISINs: replace any remaining OLD -> NEW
        for old_isin, new_isin in __isin_roll_map.items():
            mask_all = base["ISIN"].astype(str).eq(old_isin)
            if mask_all.any():
                base.loc[mask_all, "ISIN"] = new_isin

        # 2) Normalize split_events so adjustments hit the right ISIN after the rewrite
        split_events = [
            (__isin_roll_map.get(isin, isin), dt, f)
            for (isin, dt, f) in split_events
        ]

    # Apply split factors to historical BUY lots that occurred strictly BEFORE the split time.
    # We alter source fields used for Quantity_signed and Price (both in native and EUR), but keep total EUR the same.
    if split_events:
        for isin, split_dt, factor in sorted(split_events, key=lambda x: (x[0], x[1])):
            pre_mask = (
                base["ISIN"].astype(str).eq(isin)
                & base["__Type"].eq("Buy")
                & pd.to_datetime(base["Date"]).lt(split_dt)
            )
            if not pre_mask.any():
                continue

            idx = base.index[pre_mask]
            qty_before   = base.loc[idx, "__qty_desc"].copy()
            price_before = base.loc[idx, "__price_desc"].copy()

            # Apply scaling
            base.loc[idx, "__qty_desc"]   = base.loc[idx, "__qty_desc"] * factor
            base.loc[idx, "__price_desc"] = base.loc[idx, "__price_desc"] / factor

            # Log audit rows
            for i in idx:
                split_audit.append({
                    "ISIN": isin,
                    "Product": str(base.at[i, "Product"]),
                    "Row kind": "Buy (pre-split)",
                    "Trade date": pd.to_datetime(base.at[i, "Date"]),
                    "Order ID": str(base.at[i, "Order ID"]),
                    "Split date": pd.to_datetime(split_dt),
                    "Factor": float(factor),
                    "Qty (before)": float(qty_before.get(i) if pd.notna(qty_before.get(i)) else np.nan),
                    "Qty (after)":  float(base.at[i, "__qty_desc"] if pd.notna(base.at[i, "__qty_desc"]) else np.nan),
                    "Unit px (before)": float(price_before.get(i) if pd.notna(price_before.get(i)) else np.nan),
                    "Unit px (after)":  float(base.at[i, "__price_desc"] if pd.notna(base.at[i, "__price_desc"]) else np.nan),
                })

    base["__rowid"] = np.arange(len(base))
    def _effective_id(row):
        oid = str(row["Order ID"]) if "Order ID" in row else ""
        tpe = row["__Type"] if "__Type" in row else "Other"
        seq = row.get("__trade_seq", np.nan)

        # If it's a trade with an Order ID, append a sequence so multiple lines don't collapse
        if oid and oid.strip().lower() != "nan" and tpe in ("Buy", "Sell") and pd.notna(seq):
            return f"{oid}-{int(seq)}"

        # Fallback: timestamp-based synthetic id
        ts = row["Date"]; ts_key = ts.strftime("%Y%m%d%H%M") if pd.notna(ts) else "NA"
        return f"NOID-{tpe}-{ts_key}-{int(row['__rowid'])}"
    base["__EffID"] = base.apply(_effective_id, axis=1)

    # ---- Split multiple Buy/Sell rows under the same Order ID into separate effective IDs ----
    base["__trade_seq"] = np.nan

    # We'll sequence only the non-FX Buy/Sell rows, per (ISIN, Order ID, Type) in time order
    mask_trade_rows = (~(base["__is_fx_credit"] | base["__is_fx_debit"])) & base["__Type"].isin(["Buy", "Sell"])

    # Stable sort so cumcount is deterministic. Use 'Date' (exists in base), not '__dt'.
    trade_sorted = base.loc[mask_trade_rows].sort_values(
        ["ISIN", "Order ID", "Date", "Description"], kind="mergesort"
    )

    seq = (
        trade_sorted
        .groupby(["ISIN", "Order ID", "__Type"], dropna=False)
        .cumcount()
    )

    base.loc[trade_sorted.index, "__trade_seq"] = seq.values

    # Build effective ID AFTER we have __trade_seq
    base["__rowid"] = np.arange(len(base))

    def _effective_id(row):
        oid = str(row["Order ID"]) if "Order ID" in row else ""
        tpe = row.get("__Type", "Other")
        seq = row.get("__trade_seq", np.nan)

        # If it's a trade with an Order ID, append a sequence so multiple lines don't collapse
        if oid and oid.strip().lower() != "nan" and tpe in ("Buy", "Sell") and pd.notna(seq):
            return f"{oid}-{int(seq)}"

        # Fallback: timestamp-based synthetic id
        ts = row["Date"]
        ts_key = ts.strftime("%Y%m%d%H%M") if pd.notna(ts) else "NA"
        return f"NOID-{tpe}-{ts_key}-{int(row['__rowid'])}"

    base["__EffID"] = base.apply(_effective_id, axis=1)

    # -------- Aggregate non-FX first (so `grouped` exists) --------
    grouped = _aggregate_non_fx(base)

    grouped, split_audit = _apply_corporate_actions_and_map_fx(base, grouped, opening_lots)

    consolidated = _consolidate_fifo(grouped)
    
    out = _build_out_table(consolidated)
    split_audit_df = pd.DataFrame(split_audit, columns=AUDIT_COLS)
    # Clean audit rows + enforce dtypes for Arrow/Streamlit
    if not split_audit_df.empty:
        # Drop any header-like rows accidentally appended as data
        headerish = (
            split_audit_df["ISIN"].astype(str).str.upper().eq("ISIN") |
            split_audit_df["Product"].astype(str).str.upper().eq("PRODUCT")
        )
        split_audit_df = split_audit_df.loc[~headerish].copy()

        # Enforce proper dtypes
        for col in ["Split date", "Trade date"]:
            split_audit_df[col] = pd.to_datetime(split_audit_df[col], errors="coerce")

        split_audit_df["Factor"] = pd.to_numeric(split_audit_df["Factor"], errors="coerce")

        for col in [
            "Qty (before)", "Qty (after)",
            "Unit px (before)", "Unit px (after)",
            "Unit px EUR (before)", "Unit px EUR (after)"
        ]:
            if col in split_audit_df.columns:
                split_audit_df[col] = pd.to_numeric(split_audit_df[col], errors="coerce")

        split_audit_df = split_audit_df.sort_values(
            ["ISIN","Split date","Row kind","Trade date"], kind="mergesort"
        )

    return out, split_audit_df

# ===== Deemed Disposal (ETFs) — Planner + Estimator (no pipeline changes) =====
EXIT_TAX_RATE = 0.41

def _is_exit_tax_asset_row(row: pd.Series) -> bool:
    asset = str(row.get("Asset") or row.get("AssetInference") or "").lower()
    return ("etf" in asset) or ("fund" in asset) or ("gross" in asset)

def _eight_year_anniversary(d: pd.Timestamp) -> pd.Timestamp:
    try:
        return d + pd.DateOffset(years=8)
    except Exception:
        return (pd.Timestamp(d.year + 8, d.month, 1) + pd.offsets.MonthEnd(0))

def _last_trade_price_eur_before(out_df: pd.DataFrame, isin: str, when: pd.Timestamp) -> Optional[float]:
    df = out_df[out_df["ISIN"].astype(str).eq(isin)].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df[(df["Type"].isin(["Buy","Sell"])) & (df["Date"] <= when)].sort_values("Date", kind="mergesort")
    if df.empty:
        return None
    if "Price_EUR" in df.columns:
        price = pd.to_numeric(df["Price_EUR"], errors="coerce")
    else:
        # derive unit price from totals if needed (prefer fee-adjusted)
        tot = pd.to_numeric(df.get("Total (EUR, fee-adj)", df.get("Total (EUR)", np.nan)), errors="coerce")
        qty = pd.to_numeric(df.get("Quantity"), errors="coerce").replace(0, np.nan)
        price = tot / qty
    price = price.dropna()
    return float(price.iloc[-1]) if not price.empty else None

def _replay_fifo_lots_from_out(out_df: pd.DataFrame) -> Dict[str, List[Dict]]:
    """
    From the final `out` table, rebuild remaining lots per ISIN for Exit-Tax assets only.
    Each lot: {"acq": Timestamp, "qty": float, "unit_cost_eur": float}
    Uses Price_EUR if present; else derives from Total (EUR, fee-adj)/Quantity.
    """
    needed = {"ISIN", "Date", "Order ID", "Type", "Quantity"}
    missing = needed - set(out_df.columns)
    if missing:
        raise KeyError(f"`out` is missing required columns: {', '.join(sorted(missing))}")

    df = out_df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.sort_values(["ISIN", "Date", "Order ID"], kind="mergesort")

    # Ensure a EUR unit price
    if "Price_EUR" not in df.columns:
        tot = pd.to_numeric(df.get("Total (EUR, fee-adj)", df.get("Total (EUR)", np.nan)), errors="coerce")
        qty = pd.to_numeric(df.get("Quantity"), errors="coerce").replace(0, np.nan)
        df["Price_EUR"] = np.where((qty.abs() > 0), tot / qty, np.nan)

    lots_by_isin: Dict[str, List[Dict]] = {}
    exit_mask = df.apply(_is_exit_tax_asset_row, axis=1)
    trades = df[exit_mask & df["Type"].isin(["Buy", "Sell"])].copy()

    for isin, g in trades.groupby("ISIN", sort=False):
        lots: List[Dict] = []
        for _, r in g.iterrows():
            t = r["Type"]
            q = float(r.get("Quantity") or 0.0)
            if q <= 1e-12:
                continue
            if t == "Buy":
                unit_cost = float(r.get("Price_EUR") or 0.0)
                lots.append({"acq": r["Date"], "qty": q, "unit_cost_eur": unit_cost})
            else:  # Sell
                qty_to_sell = q
                j = 0
                while qty_to_sell > 1e-12 and j < len(lots):
                    take = min(lots[j]["qty"], qty_to_sell)
                    lots[j]["qty"] -= take
                    qty_to_sell -= take
                    if lots[j]["qty"] <= 1e-12:
                        j += 1
                lots = [L for L in lots if L["qty"] > 1e-12]
        lots_by_isin[isin] = lots
    return lots_by_isin

def _deemed_plan_and_estimates(out_df: pd.DataFrame, asof: Optional[pd.Timestamp]=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      planner_df: ISIN, AcquisitionDate, DeemedDate, QtyRemaining, __year
      estimator_df: planner columns + UnitCostEUR, ProposedFMV_UnitEUR, ProposedFMV_EUR, EstGain_EUR, EstExitTax_EUR, __ValNeeded
    """
    if asof is None:
        asof = pd.Timestamp.today().normalize()

    lots_by_isin = _replay_fifo_lots_from_out(out_df)
    rows: List[Dict] = []
    for isin, lots in lots_by_isin.items():
        for L in lots:
            acq = pd.to_datetime(L["acq"])
            dd = _eight_year_anniversary(acq)
            if dd > asof or L["qty"] <= 1e-12:
                continue
            unit_cost = float(L["unit_cost_eur"] or 0.0)
            unit_fmv = _last_trade_price_eur_before(out_df, isin, dd)  # heuristic (last known trade price in EUR)
            qty = float(L["qty"])
            fmv = unit_fmv * qty if unit_fmv is not None else np.nan
            base = unit_cost * qty
            gain = (fmv - base) if (fmv == fmv) else np.nan
            est_tax = max(gain, 0.0) * EXIT_TAX_RATE if (gain == gain) else np.nan
            rows.append({
                "ISIN": isin,
                "AcquisitionDate": acq.normalize(),
                "DeemedDate": dd.normalize(),
                "QtyRemaining": qty,
                "UnitCostEUR": unit_cost,
                "ProposedFMV_UnitEUR": unit_fmv,
                "ProposedFMV_EUR": fmv,
                "EstGain_EUR": gain,
                "EstExitTax_EUR": est_tax,
            })

    est = pd.DataFrame(rows)
    if est.empty:
        planner = est.copy()
    else:
        est["__ValNeeded"] = est["ProposedFMV_UnitEUR"].isna()
        est["__year"] = pd.to_datetime(est["DeemedDate"]).dt.year
        est = est.sort_values(["ISIN","DeemedDate","AcquisitionDate"], kind="mergesort")
        planner = est[["ISIN","AcquisitionDate","DeemedDate","QtyRemaining","__year"]].copy()

    return planner, est

# ---------------- Main ----------------
if not uploads:
    st.info("👆 Import a CSV to see results.")
    st.stop()

# Read CSV (robust) + normalize via adapter + clear errors
import traceback

# --- NEW: support multiple uploads & auto-detect broker per file ---
frames = []
debug_lines = []
for f in uploads:
    try:
        df_raw = pd.read_csv(f) if f.name.lower().endswith(".csv") else pd.read_excel(f)
    except Exception:
        # Try CSV with semicolon as fallback
        f.seek(0)
        df_raw = pd.read_csv(f, sep=";")
    broker = detect_broker_from_headers(df_raw.head(1))
    adapter = BROKER_ADAPTERS.get(broker, parse_degiros_csv)
    df_norm_one = adapter(df_raw)
    # tag for provenance
    df_norm_one["__Broker"] = broker
    df_norm_one["__SourceFile"] = getattr(f, "name", "upload")
    frames.append(df_norm_one)
    debug_lines.append(f"- {getattr(f,'name','upload')} -> {broker} -> {df_norm_one.shape}")
if not frames:
    raise ValueError("No valid files parsed.")
df_norm = pd.concat(frames, ignore_index=True)
# optional stable sort
if "Date" in df_norm.columns:
    df_norm = df_norm.sort_values("Date", kind="stable").reset_index(drop=True)
# Debug block
with st.expander("🔧 Debug: broker ingest (temporary)", expanded=False):
    st.write("Detected per-file:", "\n".join(debug_lines))
    st.write("Combined adapter output shape:", df_norm.shape)
    st.dataframe(df_norm.head(50))

# Build dataset
try:
    out, split_audit_df = build_output(df_norm, opening_lots_df)
    # --- Data Quality Checks (read-only diagnostics) ---
    with st.expander("🔎 Data quality checks (non-blocking)"):
        try:
            dfq = out.copy()

            # 1) Sells exceeding prior buys (simple cumulative check per ISIN)
            sells_issue = pd.DataFrame()
            if {"ISIN","Type","Quantity","Date","Order ID"}.issubset(dfq.columns):
                g = dfq.sort_values(["ISIN","Date","Order ID"], kind="mergesort")
                g["qty_signed"] = np.where(
                    g["Type"].eq("Buy"),
                    pd.to_numeric(g["Quantity"], errors="coerce"),
                    np.where(g["Type"].eq("Sell"), -pd.to_numeric(g["Quantity"], errors="coerce"), 0.0),
                )
                # Vectorized cumulative per ISIN (no apply; no deprecation warning)
                g["cum"] = g.groupby("ISIN", group_keys=False)["qty_signed"].cumsum()
                sells_issue = g.loc[(g["cum"] < -1e-8) & g["Type"].eq("Sell"), ["ISIN","Date","Order ID","Quantity"]]

            # 2) Possible FX mapping gaps: non-EUR trades with missing EUR total (and no derivable EUR)
            fx_gaps = pd.DataFrame()
            eur_cols = {"Total (EUR)", "Total (EUR, fee-adj)"}
            if {"Type","Quantity","Product"}.issubset(dfq.columns) and (eur_cols & set(dfq.columns)):

                eur_col = "Total (EUR, fee-adj)" if "Total (EUR, fee-adj)" in dfq.columns else "Total (EUR)"

                # Treat blank currency as EUR to avoid false positives on opening imports
                cur = dfq.get("Currency")
                cur = cur.fillna("EUR").astype(str).str.upper() if cur is not None else pd.Series("EUR", index=dfq.index)

                # These are synthetic opening/import rows; don't flag them
                is_opening = dfq["Product"].astype(str).str.startswith("OPENING-")
                is_incoming = dfq.get("__IncomingTransfer", False)

                # Only look at actual trades (Buy/Sell), non-EUR currency, excluding opening/imports
                mask_trade = dfq["Type"].isin(["Buy","Sell"])
                mask_non_eur = cur.ne("EUR")
                mask_ignore = is_opening | is_incoming

                # Try to derive EUR total from Price_EUR * Quantity if the total column is empty
                qty = pd.to_numeric(dfq.get("Quantity"), errors="coerce")
                price_eur = pd.to_numeric(dfq.get("Price_EUR"), errors="coerce")
                derived_eur = price_eur * qty

                # EUR considered present if either explicit total exists or we can derive it
                eur_present = dfq[eur_col].notna() | derived_eur.notna()

                fx_mask = mask_trade & ~mask_ignore & mask_non_eur & ~eur_present

                fx_gaps = dfq.loc[fx_mask, ["Date","ISIN","Order ID","Type","Quantity","Currency"]]

            # 3) Dividend tax lines without nearby matching dividend (same ISIN, ±1 day)
            orphan_divtax = pd.DataFrame()
            if {"Type","ISIN","Date"}.issubset(dfq.columns):
                d = dfq[["ISIN","Type","Date"]].copy()
                d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
                tax = d[d["Type"].eq("Dividend Tax")]
                div = d[d["Type"].eq("Dividend")]
                if not tax.empty and not div.empty:
                    div_isin = div.groupby("ISIN")["Date"].apply(list).to_dict()
                    rows = []
                    for _, r in tax.iterrows():
                        dates = div_isin.get(r["ISIN"], [])
                        if not any(abs((r["Date"] - pd.to_datetime(dt)).days) <= 1 for dt in dates):
                            rows.append({"ISIN": r["ISIN"], "TaxDate": r["Date"]})
                    orphan_divtax = pd.DataFrame(rows)

            # Render
            if sells_issue.empty and fx_gaps.empty and orphan_divtax.empty:
                st.success("No issues detected 🎉")
            else:
                if not sells_issue.empty:
                    st.markdown("**Sells exceeding prior buys (check FIFO inputs):**")
                    st.dataframe(sells_issue.head(200))
                if not fx_gaps.empty:
                    st.markdown("**Trades missing EUR mapping (check FX rows):**")
                    st.dataframe(fx_gaps.head(200))
                if not orphan_divtax.empty:
                    st.markdown("**Dividend Tax without nearby Dividend:**")
                    st.dataframe(orphan_divtax.head(200))

        except Exception as e:
            st.info(f"Quality checks skipped: {e}")

    def _normalize_manual(opening_lots_df):
        if opening_lots_df is None or opening_lots_df.empty:
            return pd.DataFrame(columns=["ISIN","Quantity","EUR_Value","Unit_EUR"])

        def _clean_num(series):
            # remove thousands separators, spaces, and €/$/£ signs before numeric cast
            return pd.to_numeric(
                series.astype(str).str.replace(r"[,\s€$£]", "", regex=True),
                errors="coerce"
            )

        ol = opening_lots_df.copy()
        ol.columns = [c.strip() for c in ol.columns]

        # Rich format: ISIN, Currency, Quantity, Price, EUR Value   (other extra columns are fine)
        if {"ISIN","Currency","Quantity","Price","EUR Value"}.issubset(ol.columns):
            ol["Quantity"]   = _clean_num(ol["Quantity"])
            ol["Price"]      = _clean_num(ol["Price"])
            ol["EUR Value"]  = _clean_num(ol["EUR Value"])
            ol = ol.dropna(subset=["ISIN","Quantity","EUR Value"])
            ol["Unit_EUR"] = np.where(ol["Quantity"].abs() > 0, ol["EUR Value"] / ol["Quantity"].abs(), np.nan)
            return ol[["ISIN","Quantity","EUR Value","Unit_EUR"]].rename(columns={"EUR Value":"EUR_Value"})

        # Simple format: ISIN, Quantity, UnitPrice
        if {"ISIN","Quantity","UnitPrice"}.issubset(ol.columns):
            ol["Quantity"]  = _clean_num(ol["Quantity"])
            ol["UnitPrice"] = _clean_num(ol["UnitPrice"])
            ol = ol.dropna(subset=["ISIN","Quantity","UnitPrice"])
            ol["EUR_Value"] = ol["Quantity"].abs() * ol["UnitPrice"]
            ol["Unit_EUR"]  = ol["UnitPrice"]
            return ol[["ISIN","Quantity","EUR_Value","Unit_EUR"]]

        return pd.DataFrame(columns=["ISIN","Quantity","EUR_Value","Unit_EUR"])

    _manual_norm = _normalize_manual(opening_lots_df)

    # STRICT detection: only rows whose description explicitly says INCOMING TRANSFER
    if "Description" in out.columns:
        out["__IncomingTransfer"] = (
            out["Type"].eq("Buy") &
            out["Description"].astype(str).str.contains("INCOMING TRANSFER", case=False, na=False)
        )
    else:
        out["__IncomingTransfer"] = False

    incoming_rows = out.loc[out["__IncomingTransfer"]].copy()

    matched = 0
    missing_isins = set()
    extra_manual_isins = set(_manual_norm["ISIN"].astype(str).unique())

    for i, r in incoming_rows.iterrows():
        isin = str(r["ISIN"])
        qty  = float(r.get("Quantity") or 0.0)
        cand = _manual_norm[_manual_norm["ISIN"].astype(str).eq(isin)].copy()
        if cand.empty:
            missing_isins.add(isin)
            continue
        cand["qdiff"] = (cand["Quantity"].abs() - abs(qty)).abs()
        if (cand["qdiff"] <= 1e-6).any():
            pick = cand.loc[cand["qdiff"].idxmin()]
        elif len(cand) == 1:
            pick = cand.iloc[0]
        else:
            missing_isins.add(isin)
            continue

        # Display-only fill of EUR columns; DO NOT touch anything used for FIFO math
        out.at[i, "Total (EUR)"] = float(pick["EUR_Value"])
        out.at[i, "Total (EUR, fee-adj)"] = float(pick["EUR_Value"])
        matched += 1
        if isin in extra_manual_isins:
            extra_manual_isins.remove(isin)

        # Ensure provenance columns exist (for older adapters)
        if "__Broker" not in out.columns:
            out["__Broker"] = "UNKNOWN"
        if "__SourceFile" not in out.columns:
            out["__SourceFile"] = "uploads"

except Exception as e:
    st.error(f"Could not parse CSV: {e}")
    st.stop()

# --------- Annual Summary (TOP with tabs) ---------
st.markdown("### 🧾 Annual Summary")

if 'out' in locals() and out is not None:
    df_sum = out.copy()
for col in ["Total (EUR)", "Fee", "Gain/Loss"]:
    if col in df_sum.columns:
        df_sum[col] = pd.to_numeric(df_sum[col], errors="coerce")

def _sum_mask(df, mask, col):
    if col not in df.columns:
        return 0.0
    s = pd.to_numeric(df.loc[mask, col], errors="coerce")
    return float(s.fillna(0).sum())

years_sorted = sorted([int(y) for y in df_sum["__year"].dropna().unique()])

# Build per-asset class helper (dividend columns removed from these summaries)
def build_summary(df_source: pd.DataFrame, asset_filter: Optional[str]) -> pd.DataFrame:
    rows = []
    # carry-forward is only relevant to shares (CGT)
    carry_forward_prev_shares = 0.0

    for yr in years_sorted:
        g_all = df_source[df_source["__year"].eq(yr)].copy()

        # Exclude INCOMING TRANSFER rows from all cash/tax calculations
        if "__IncomingTransfer" in g_all.columns:
            g_all = g_all[~g_all["__IncomingTransfer"].fillna(False)].copy()

        # Filters by asset
        g_shares = g_all[g_all["Asset"].astype(str).str.lower().eq("share")]
        g_etfs   = g_all[g_all["Asset"].astype(str).str.lower().eq("etf")]

        # Common helpers
        def _sum_mask_local(df, mask, col):
            if df.empty or col not in df.columns:
                return 0.0
            s = pd.to_numeric(df.loc[mask, col], errors="coerce")
            return float(s.fillna(0).sum())

        if asset_filter == "share":
            g = g_shares

            buys_eur  = _sum_mask_local(g, g["Type"].eq("Buy"),  "Total (EUR, fee-adj)")
            sells_eur = _sum_mask_local(g, g["Type"].eq("Sell"), "Total (EUR, fee-adj)")
            fees_eur  = _sum_mask_local(g, g["Type"].isin(["Fee","Interest","Dividend"]), "Fee")
            realised_pl = _sum_mask_local(g, g["Type"].eq("Sell"), "Gain/Loss")

            # CGT rules for shares
            bf_used = 0.0
            if realised_pl > 0 and carry_forward_prev_shares > 0:
                bf_used = min(carry_forward_prev_shares, realised_pl)
            remaining_gain_after_bf = realised_pl - bf_used
            ex_used = min(exemption_val, remaining_gain_after_bf) if (use_exemption and remaining_gain_after_bf > 0) else 0.0
            taxable_gain = max(0.0, remaining_gain_after_bf - ex_used)
            tax_due = taxable_gain * cgt_rate_shares

            carry_forward_new = max(0.0, carry_forward_prev_shares - bf_used)
            if realised_pl < 0:
                carry_forward_new += abs(realised_pl)
            carry_forward_prev_shares = carry_forward_new

            row = {
                "Year": int(yr),
                "Buys (EUR)": buys_eur,
                "Sells (EUR)": sells_eur,
                "Realised Profit / Loss (EUR)": realised_pl,
                "Taxable Gain (EUR)": taxable_gain,
                f"Tax @ {int(cgt_rate_shares*100)}% (EUR)": tax_due,
                "B/F Loss Used (EUR)": bf_used,
                "Exemption Used (EUR)": ex_used,
                "Carry Forward (EUR)": carry_forward_new,
                "Net Cashflow (EUR)": sells_eur - buys_eur - fees_eur,
                "Total Fees (EUR)": _sum_mask_local(g, g["Type"].isin(["Fee","Interest"]), "Fee"),
            }

        elif asset_filter == "etf":
            g = g_etfs

            buys_eur  = _sum_mask_local(g, g["Type"].eq("Buy"),  "Total (EUR, fee-adj)")
            sells_eur = _sum_mask_local(g, g["Type"].eq("Sell"), "Total (EUR, fee-adj)")
            fees_eur  = _sum_mask_local(g, g["Type"].isin(["Fee","Interest","Dividend"]), "Fee")
            realised_pl = _sum_mask_local(g, g["Type"].eq("Sell"), "Gain/Loss")

            taxable_gain = max(0.0, realised_pl)
            tax_due = taxable_gain * exit_tax_rate_etf

            row = {
                "Year": int(yr),
                "Buys (EUR)": buys_eur,
                "Sells (EUR)": sells_eur,
                "Realised Profit / Loss (EUR)": realised_pl,
                "Taxable Gain (EUR)": taxable_gain,
                f"Tax @ {int(exit_tax_rate_etf*100)}% (EUR)": tax_due,
                "Net Cashflow (EUR)": sells_eur - buys_eur - fees_eur,
                "Total Fees (EUR)": _sum_mask_local(g, g["Type"].isin(["Fee","Interest"]), "Fee"),
            }

        else:
            # COMBINED view: compute per-asset taxes, then sum
            buys_eur_all  = _sum_mask_local(g_all, g_all["Type"].eq("Buy"),  "Total (EUR, fee-adj)")
            sells_eur_all = _sum_mask_local(g_all, g_all["Type"].eq("Sell"), "Total (EUR, fee-adj)")
            fees_eur_all  = _sum_mask_local(g_all, g_all["Type"].isin(["Fee","Interest","Dividend"]), "Fee")

            # Realised per asset
            realised_shares = _sum_mask_local(g_shares, g_shares["Type"].eq("Sell"), "Gain/Loss")
            realised_etfs   = _sum_mask_local(g_etfs,   g_etfs["Type"].eq("Sell"),   "Gain/Loss")

            # Shares CGT (with carry-forward + exemption)
            bf_used = 0.0
            if realised_shares > 0 and carry_forward_prev_shares > 0:
                bf_used = min(carry_forward_prev_shares, realised_shares)
            remaining_gain_after_bf = realised_shares - bf_used
            ex_used = min(exemption_val, remaining_gain_after_bf) if (use_exemption and remaining_gain_after_bf > 0) else 0.0
            taxable_shares = max(0.0, remaining_gain_after_bf - ex_used)
            tax_shares = taxable_shares * cgt_rate_shares

            carry_forward_new = max(0.0, carry_forward_prev_shares - bf_used)
            if realised_shares < 0:
                carry_forward_new += abs(realised_shares)
            carry_forward_prev_shares = carry_forward_new  # carry forward to next year

            # ETFs exit tax
            taxable_etfs = max(0.0, realised_etfs)
            tax_etfs = taxable_etfs * exit_tax_rate_etf

            row = {
                "Year": int(yr),
                "Buys (EUR)": buys_eur_all,
                "Sells (EUR)": sells_eur_all,
                "Realised Profit / Loss (EUR)": realised_shares + realised_etfs,
                f"Tax @ Shares {int(cgt_rate_shares*100)}% (EUR)": tax_shares,
                f"Tax @ ETFs {int(exit_tax_rate_etf*100)}% (EUR)": tax_etfs,
                "Tax @ Combined (EUR)": tax_shares + tax_etfs,
                "B/F Loss Used (EUR)": bf_used,
                "Exemption Used (EUR)": ex_used,
                "Carry Forward (EUR)": carry_forward_new,
                "Net Cashflow (EUR)": sells_eur_all - buys_eur_all - fees_eur_all,
                "Total Fees (EUR)": _sum_mask_local(g_all, g_all["Type"].isin(["Fee","Interest"]), "Fee"),
            }

        rows.append(row)

    return pd.DataFrame(rows)

summary_shares    = build_summary(df_sum, "share")
summary_etfs      = build_summary(df_sum, "etf")
summary_combined  = build_summary(df_sum, None)

def _fmt_money_eur(x):
    if isinstance(x, str):  # keep "Total" in the Year column, etc.
        return x
    if pd.isna(x):
        return ""
    return f"€{x:,.2f}"

# --- renderer (re-usable) ---
def style_and_show_summary(df: pd.DataFrame):
    if df.empty:
        st.info("No data for this selection."); return

    base_cols = ["Year", "Buys (EUR)", "Sells (EUR)", "Realised Profit / Loss (EUR)", "Taxable Gain (EUR)"]
    tax_cols = [c for c in df.columns if c.startswith("Tax @")]

    optional_cols = []
    if show_bf_used:    optional_cols.append("B/F Loss Used (EUR)")
    if show_ex_used:    optional_cols.append("Exemption Used (EUR)")
    if show_carry_fw:   optional_cols.append("Carry Forward (EUR)")
    if show_cashflow:   optional_cols.append("Net Cashflow (EUR)")
    if show_total_fees: optional_cols.append("Total Fees (EUR)")

    ordered = [c for c in base_cols if c in df.columns] + tax_cols + [c for c in optional_cols if c in df.columns]
    df_v = df[ordered].copy()

    # Totals row
    totals = {}
    for col in df_v.columns:
        if col == "Year": 
            continue
        totals[col] = float(pd.to_numeric(df_v[col], errors="coerce").fillna(0).sum())
    df_v = pd.concat([df_v, pd.DataFrame([{"Year": "Total", **totals}])], ignore_index=True)

    # Ensure Year is string for the "Total" label
    df_v["Year"] = df_v["Year"].astype(str)

    # Apply € formatter to every numeric money column (everything except "Year")
    money_cols = [c for c in df_v.columns if c != "Year"]
    styler = df_v.style.format({c: _fmt_money_eur for c in money_cols})

    # Keep green/red styling for realised P/L if present
    def pl_color(val):
        if isinstance(val, str) or pd.isna(val): return ""
        if val > 0: return "color: green; font-weight: 600;"
        if val < 0: return "color: red; font-weight: 600;"
        return ""
    if "Realised Profit / Loss (EUR)" in df_v.columns:
        styler = styler.map(pl_color, subset=["Realised Profit / Loss (EUR)"])

    st.dataframe(styler, use_container_width=True)

# --- TABS UI (includes Dividends) ---
tabs = st.tabs([
    "📈 Shares (CGT)",
    "🧺 ETFs (Exit Tax)",
    "➕ Combined (Shares+ETFs)",
    "💸 Dividends",
    "⏳ ETFs (Deemed Disposal)"
])
with tabs[0]:
    style_and_show_summary(summary_shares)
with tabs[1]:
    style_and_show_summary(summary_etfs)
with tabs[2]:
    style_and_show_summary(summary_combined)
with tabs[3]:
    # Dividends tab content (keeps gross/tax here only)
    st.subheader("Dividend Summary")
    divs = out[out["Type"] == "Dividend"].copy()
    if divs.empty:
        st.info("No dividends found in this file.")
    else:
        # Ensure numeric and use sensible defaults
        divs["Gross"]  = pd.to_numeric(divs["Total"], errors="coerce").fillna(0).abs()
        divs["TaxAmt"] = pd.to_numeric(divs["Fee"],   errors="coerce").fillna(0)

        divs["Year"] = pd.to_datetime(divs["Date"]).dt.year

        # Per-year summary
        per_year = (
            divs.groupby("Year", dropna=False)
                .agg(Gross=("Gross","sum"), Tax=("TaxAmt","sum"))
                .reset_index()
        )
        per_year["Net"] = per_year["Gross"] - per_year["Tax"]
        per_year["Year"] = per_year["Year"].astype(str)

        # By ticker summary
        by_ticker = (
            divs.groupby(["Ticker - Name","ISIN"], dropna=False)
                .agg(Gross=("Gross","sum"), Tax=("TaxAmt","sum"), Payments=("Date","count"))
                .reset_index()
                .sort_values("Gross", ascending=False)
        )
        by_ticker["Net"] = by_ticker["Gross"] - by_ticker["Tax"]

        def fmt_money(x):
            if pd.isna(x): return ""
            return f"{x:,.2f}"

        st.markdown("**Per Year**")
        st.dataframe(
            per_year.style.format({"Gross":fmt_money,"Tax":fmt_money,"Net":fmt_money}),
            use_container_width=True
        )

        st.markdown("**By Ticker**")
        st.dataframe(
            by_ticker.style.format({"Gross":fmt_money,"Tax":fmt_money,"Net":fmt_money}),
            use_container_width=True
        )

        st.markdown("**Dividend Transactions**")
        tx_cols = ["Date","Ticker - Name","ISIN","Total","Fee","Order ID"]
        st.dataframe(
            divs.sort_values("Date").loc[:, tx_cols].style.format({
                "Total":fmt_money,"Fee":fmt_money
            }),
            use_container_width=True
        )
with tabs[4]:
    # --- Deemed disposal (runs after upload; no toggle) ---
    import io, zipfile, traceback

    planner = None
    est = None

    # Only attempt when a CSV is uploaded and the pipeline produced data
    if 'out' in locals() and out is not None and not out.empty:
        # Optional: skip if there are no ETFs
        has_etfs = out.get("Asset") is not None and out["Asset"].astype(str).str.lower().eq("etf").any()
        if has_etfs:
            try:
                with st.spinner("Building ETF deemed-disposal planner & estimate…"):
                    planner, est = _deemed_plan_and_estimates(out)
            except Exception as e:
                st.warning(f"Deemed-disposal generation failed: {e}")
                st.code(traceback.format_exc())
        else:
            st.info("No ETF positions found — deemed-disposal not applicable.")

    st.subheader("ETF Deemed Disposal — Planner & Estimator")
    if planner is None or est is None:
        st.caption("Upload a DEGIRO CSV with ETF transactions to generate the planner and estimate.")
    else:
        if planner.empty:
            st.info("No ETF lots currently held for ≥ 8 years — nothing to plan yet.")
        else:
            total = len(planner)
            by_year = planner["__year"].value_counts().sort_index()
            st.write(f"Lots hitting deemed disposal (8-year): **{total}**")
            st.dataframe(by_year.rename_axis("Year").reset_index(name="Lots"))

            with st.expander("Lots (planner)"):
                st.dataframe(planner[["ISIN","AcquisitionDate","DeemedDate","QtyRemaining"]], use_container_width=True)

        # Estimator view (single user input: today's price per ISIN, Unit EUR)
        if est is None or est.empty:
            st.info("No proposed valuations could be derived yet.")
        else:
            st.markdown("**Enter today’s price (per unit, EUR) for each ETF ISIN.**")

            # Prepare a per-ISIN editor with one editable column
            # Use the median of proposed unit FMVs as a sensible suggestion if multiple lots per ISIN
            per_isin = (
                est.groupby("ISIN", dropna=False)["ProposedFMV_UnitEUR"]
                   .median()
                   .rename("Suggested Price (Unit EUR)")
                   .reset_index()
            )

            # Session-persisted price table (only one user-editable column)
            price_key = "deemed_today_prices"
            if price_key not in st.session_state:
                st.session_state[price_key] = per_isin.assign(**{
                    "Today’s Price (Unit EUR)": per_isin["Suggested Price (Unit EUR)"]
                })

            # Merge new ISINs that may appear on rerun
            merged = per_isin.merge(
                st.session_state[price_key][["ISIN", "Today’s Price (Unit EUR)"]],
                on="ISIN", how="left"
            )
            merged["Today’s Price (Unit EUR)"] = merged["Today’s Price (Unit EUR)"].fillna(
                merged["Suggested Price (Unit EUR)"]
            )
            st.session_state[price_key] = merged

            # Render the minimalist editor (only ISIN + today's unit price editable)
            price_inputs = st.data_editor(
                st.session_state[price_key][["ISIN", "Suggested Price (Unit EUR)", "Today’s Price (Unit EUR)"]],
                use_container_width=True,
                key="deemed_today_prices_editor",
                column_config={
                    "ISIN": st.column_config.TextColumn("ISIN", disabled=True),
                    "Suggested Price (Unit EUR)": st.column_config.NumberColumn(
                        "Suggested Price (Unit EUR)", format="€%.4f", disabled=True
                    ),
                    "Today’s Price (Unit EUR)": st.column_config.NumberColumn(
                        "Today’s Price (Unit EUR)", format="€%.4f",
                        help="Enter the current unit price in EUR for this ETF."
                    ),
                }
            )

            # Map ISIN -> chosen unit price, fallback to suggested if blank
            price_inputs["Today’s Price (Unit EUR)"] = pd.to_numeric(
                price_inputs["Today’s Price (Unit EUR)"], errors="coerce"
            )
            price_inputs["__unit_price"] = price_inputs["Today’s Price (Unit EUR)"].where(
                price_inputs["Today’s Price (Unit EUR)"].notna(),
                price_inputs["Suggested Price (Unit EUR)"]
            )
            price_map = dict(zip(price_inputs["ISIN"], price_inputs["__unit_price"]))

            # Build a read-only results table using user price per ISIN
            est_view = est.copy().rename(columns={
                "UnitCostEUR": "Unit Cost (EUR)"
            })
            est_view["Fair Market Value (Unit EUR)"] = est_view["ISIN"].map(price_map)
            est_view["Fair Market Value (EUR)"] = est_view["Fair Market Value (Unit EUR)"] * est_view["QtyRemaining"]
            est_view["Estimated Gain (EUR)"] = est_view["Fair Market Value (EUR)"] - (
                est_view["Unit Cost (EUR)"] * est_view["QtyRemaining"]
            )
            est_view["Estimated Exit Tax (EUR)"] = est_view["Estimated Gain (EUR)"].clip(lower=0) * EXIT_TAX_RATE

            # Display results
            show_cols = [
                "ISIN","AcquisitionDate","DeemedDate","QtyRemaining",
                "Unit Cost (EUR)",
                "Fair Market Value (Unit EUR)","Fair Market Value (EUR)",
                "Estimated Gain (EUR)","Estimated Exit Tax (EUR)"
            ]

            def _fmt_money(x):
                return "" if pd.isna(x) else f"€{float(x):,.2f}"

            st.markdown("**Calculated results**")
            st.dataframe(
                est_view[show_cols].sort_values(["DeemedDate","ISIN","AcquisitionDate"]).style.format({
                    "QtyRemaining": lambda x: "" if pd.isna(x) else f"{float(x):.6f}".rstrip("0").rstrip("."),
                    "Unit Cost (EUR)": _fmt_money,
                    "Fair Market Value (Unit EUR)": _fmt_money,
                    "Fair Market Value (EUR)": _fmt_money,
                    "Estimated Gain (EUR)": _fmt_money,
                    "Estimated Exit Tax (EUR)": _fmt_money,
                }),
                use_container_width=True
            )

            # Roll-up by deemed year
            deemed_year = pd.to_datetime(est_view["DeemedDate"]).dt.year
            roll = (
                est_view.assign(__year=deemed_year)
                        .groupby("__year", dropna=False)[["Fair Market Value (EUR)","Estimated Gain (EUR)","Estimated Exit Tax (EUR)"]]
                        .sum(min_count=1)
                        .reset_index()
                        .rename(columns={"__year":"Year"})
            )
            if not roll.empty:
                st.markdown("**Summary by deemed year**")
                st.dataframe(
                    roll.style.format({
                        "Fair Market Value (EUR)": _fmt_money,
                        "Estimated Gain (EUR)": _fmt_money,
                        "Estimated Exit Tax (EUR)": _fmt_money,
                    }),
                    use_container_width=True
                )

            st.caption(
                "Fair Market Value = the value you use for deemed disposal. "
                "Enter **today’s unit price in EUR** per ETF above; it will be applied to all lots of that ISIN. "
                "Exit Tax is applied at 41% to gains only."
            )

# --------- Transaction History ---------
st.markdown("### 📜 Transaction History")

# Reuse years from earlier summary
years = years_sorted
year_options = ["All"] + years

# Build options safely (work if columns are missing)
asset_unique = sorted(
    out.get("Asset", pd.Series([], dtype="object"))
       .dropna()
       .astype(str)
       .str.lower()
       .unique()
    .tolist()
)
asset_options = ["All"] + [a.title() if a != "etf" else "ETF" for a in asset_unique]

broker_unique = []
if "__Broker" in out.columns:
    broker_unique = sorted(out["__Broker"].dropna().astype(str).unique().tolist())
broker_options = ["All"] + broker_unique

source_unique = []
if "__SourceFile" in out.columns:
    source_unique = sorted(out["__SourceFile"].dropna().astype(str).unique().tolist())
source_options = ["All"] + source_unique

# Controls layout
cols = st.columns([1, 1, 1, 2])

with cols[0]:
    year_choice = st.selectbox("Year", options=year_options, index=0)

with cols[1]:
    asset_choice = st.radio("Asset", options=asset_options, horizontal=True, index=0)

with cols[2]:
    broker_choice = st.selectbox("Broker", options=broker_options, index=0)

with cols[3]:
    source_choice = st.selectbox("Source file", options=source_options, index=0)

# Apply filters in sequence
filtered = out.copy() if year_choice == "All" else out[out["__year"].eq(year_choice)].copy()

if asset_choice != "All" and "Asset" in filtered.columns:
    filtered = filtered[filtered["Asset"].astype(str).str.lower() == asset_choice.lower()]

if broker_choice != "All" and "__Broker" in filtered.columns:
    filtered = filtered[filtered["__Broker"] == broker_choice]

if source_choice != "All" and "__SourceFile" in filtered.columns:
    filtered = filtered[filtered["__SourceFile"] == source_choice]

# Toggle types using your sidebar switches
hide_types = []
if not show_buys: hide_types += ["Buy"]
if not show_sells: hide_types += ["Sell"]
if not show_dividends: hide_types += ["Dividend"]
if not show_corp: hide_types += ["Stock split","Product change","ISIN change"]
if not show_fees_interest: hide_types += ["Fee","Interest","Other"]
if hide_types:
    filtered = filtered[~filtered["Type"].isin(hide_types)]

filtered = filtered.sort_values("Date", ascending=False, kind="mergesort")

# Render table
def fmt_date(d): 
    return "" if pd.isna(d) else d.strftime("%d %b %Y")

def fmt_money(x):  # generic (no currency symbol)
    if pd.isna(x): return ""
    try: return f"{x:,.2f}"
    except: return str(x)

def fmt_money_eur(x):  # EUR amounts with €
    if pd.isna(x): return ""
    try: return f"€{x:,.2f}"
    except: return str(x)

def fmt_qty(x):
    if pd.isna(x): return ""
    return f"{x:.6f}".rstrip("0").rstrip(".")

display_cols = [c for c in [
    "Date","Ticker - Name","ISIN","Type","Asset","Currency","Quantity","Price","Fee",
    "Total","Total (EUR)","Total (EUR, fee-adj)","Gain/Loss",
    "__Broker","__SourceFile","Order ID"
] if c in filtered.columns]

to_show = filtered.drop(columns=["__year"]).loc[:, display_cols].copy()
# Pretty asset label
if "Asset" in to_show.columns:
    to_show["Asset"] = to_show["Asset"].astype(str).str.title().replace({"Etf":"ETF"})

def pl_color(val):
    if isinstance(val, str) or pd.isna(val): return ""
    if val > 0: return "color: green; font-weight: 600;"
    if val < 0: return "color: red; font-weight: 600;"
    return ""

styler = to_show.style.format({
    "Date": fmt_date,
    "Quantity": fmt_qty,
    "Price": fmt_money,                 # native currency, leave without €
    "Fee": fmt_money_eur,               # fee is in EUR in your pipeline
    "Total": fmt_money,                 # native currency total (no €)
    "Total (EUR)": fmt_money_eur,       # add €
    "Total (EUR, fee-adj)": fmt_money_eur,  # add €
    "Gain/Loss": fmt_money_eur,         # realised P/L in EUR → add €
})
if "Gain/Loss" in to_show.columns:
    styler = styler.map(pl_color, subset=["Gain/Loss"])

st.dataframe(styler, use_container_width=True)

with st.expander("📊 Stock Split Adjustments (debug)"):
    if split_audit_df is None or split_audit_df.empty:
        st.info("No stock split adjustments were applied.")
    else:
        # Summary: one row per (ISIN, Split date, Factor)
        summary = (
            split_audit_df
            .groupby(["ISIN","Product","Split date","Factor"], dropna=False)
            .agg(Adjusted_rows=("Row kind","count"))
            .reset_index()
            .sort_values(["Split date","ISIN"])
        )

        def _fmt(x):
            if pd.isna(x): return ""
            try: return f"{x:,.6f}".rstrip("0").rstrip(".")
            except: return str(x)

        st.markdown("**Detected splits**")
        st.dataframe(summary, use_container_width=True)

        st.markdown("**Adjusted rows (before/after)**")
        st.dataframe(
            split_audit_df.sort_values(["Split date","ISIN","Trade date"]).style.format({
                "Qty (before)": _fmt, "Qty (after)": _fmt,
                "Unit px (before)": _fmt, "Unit px (after)": _fmt,
                "Unit px EUR (before)": _fmt, "Unit px EUR (after)": _fmt,
            }),
            use_container_width=True
        )
with st.expander("🔗 Incoming Transfers (display-only links)"):
    if "__IncomingTransfer" not in out.columns or not out["__IncomingTransfer"].any():
        st.caption("No INCOMING TRANSFER rows detected.")
    else:
        # --- Linked transfers table (same as before) ---
        transfers = out[out["__IncomingTransfer"]].copy()
        cols = [c for c in ["Date","Ticker - Name","ISIN","Quantity","Total (EUR)","Total (EUR, fee-adj)","Order ID"] if c in transfers.columns]
        st.dataframe(
            transfers.sort_values("Date").loc[:, cols].style.format({
                "Total (EUR)":       lambda x: "" if pd.isna(x) else f"€{x:,.2f}",
                "Total (EUR, fee-adj)": lambda x: "" if pd.isna(x) else f"€{x:,.2f}",
            }),
            use_container_width=True
        )

        # --- Summary lines moved here ---
        st.markdown(f"**Linked:** {matched} INCOMING TRANSFER trade(s) to Manual Import (display-only).")

        if missing_isins:
            st.markdown(
                "• **No Manual match for CSV transfer rows:** " +
                ", ".join(sorted(missing_isins))
            )
        if extra_manual_isins:
            st.markdown(
                "• **Manual ISINs without a CSV transfer row:** " +
                ", ".join(sorted(extra_manual_isins))
            )

        # ---------- Tiny Diagnostics ----------
        # Build a small triage table to understand why items didn't match
        def _safe_abs(v):
            try:
                return abs(float(v))
            except:
                return np.nan

        # Normalised manual imports we already built earlier
        manual_norm = _manual_norm.copy() if '_manual_norm' in globals() else pd.DataFrame(columns=["ISIN","Quantity","EUR_Value","Unit_EUR"])
        manual_norm["ISIN"] = manual_norm["ISIN"].astype(str)
        manual_norm["ManualQty"] = pd.to_numeric(manual_norm["Quantity"], errors="coerce")
        manual_norm["ManualEUR"] = pd.to_numeric(manual_norm["EUR_Value"], errors="coerce")

        # CSV buys & transfer flags by ISIN
        csv_buys = out[out["Type"].eq("Buy")].copy()
        csv_buys["ISIN"] = csv_buys["ISIN"].astype(str)
        csv_buys["BuyQty"] = pd.to_numeric(csv_buys.get("Quantity"), errors="coerce")

        # Summary per ISIN
        has_transfer = (
            out[out["__IncomingTransfer"]]
            .groupby("ISIN")["__IncomingTransfer"].size()
            .rename("CSV_TransferRows")
        )
        # Build per-ISIN stats using safe positional argmin
        def _closest_to_mean(series: pd.Series):
            sd = series.dropna()
            if sd.size == 0:
                return np.nan
            # find the POSITION of the min absolute deviation from the mean
            pos = (sd - sd.mean()).abs().values.argmin()
            return float(sd.iloc[pos])

        buy_stats = (
            csv_buys.groupby("ISIN", dropna=False)["BuyQty"]
                    .agg(
                        CSV_BuyRows=lambda x: int(x.notna().sum()),
                        ClosestBuyQty=_closest_to_mean
                    )
        )

        # Build diagnostics for two sets:
        # A) CSV transfer rows with no manual match (missing_isins)
        a_rows = []
        for isin in sorted(missing_isins):
            row = {
                "ISIN": isin,
                "Kind": "CSV transfer w/o Manual",
                "ManualQty": np.nan,
                "ManualEUR": np.nan,
                "CSV_TransferRows": int(has_transfer.get(isin, 0)),
                "CSV_BuyRows": 0,
                "ClosestBuyQty": np.nan,
                "QtyDiff": np.nan,
                "Note": "Manual not found / qty mismatch",
            }
            if isinstance(buy_stats, pd.DataFrame) and isin in buy_stats.index:
                row["CSV_BuyRows"] = int(buy_stats.at[isin, "CSV_BuyRows"])
                row["ClosestBuyQty"] = pd.to_numeric(buy_stats.at[isin, "ClosestBuyQty"], errors="coerce")
            a_rows.append(row)

        # B) Manual ISINs without a CSV transfer row (extra_manual_isins)
        b_rows = []
        for isin in sorted(extra_manual_isins):
            cand = manual_norm[manual_norm["ISIN"] == isin]
            if cand.empty:
                continue
            m_qty = float(cand["ManualQty"].iloc[0]) if pd.notna(cand["ManualQty"].iloc[0]) else np.nan
            m_eur = float(cand["ManualEUR"].iloc[0]) if pd.notna(cand["ManualEUR"].iloc[0]) else np.nan
            row = {
                "ISIN": isin,
                "Kind": "Manual w/o CSV transfer",
                "ManualQty": m_qty,
                "ManualEUR": m_eur,
                "CSV_TransferRows": int(has_transfer.get(isin, 0)),
                "CSV_BuyRows": 0,
                "ClosestBuyQty": np.nan,
                "QtyDiff": np.nan,
                "Note": "No CSV row with 'INCOMING TRANSFER'",
            }
            if isin in buy_stats.index:
                row["CSV_BuyRows"] = int(buy_stats.at[isin, "CSV_BuyRows"])
                row["ClosestBuyQty"] = pd.to_numeric(buy_stats.at[isin, "ClosestBuyQty"], errors="coerce")
                if pd.notna(row["ClosestBuyQty"]) and pd.notna(m_qty):
                    row["QtyDiff"] = abs(row["ClosestBuyQty"] - m_qty)
            b_rows.append(row)

        diag = pd.DataFrame(a_rows + b_rows, columns=[
            "ISIN","Kind","ManualQty","ManualEUR","CSV_TransferRows","CSV_BuyRows","ClosestBuyQty","QtyDiff","Note"
        ])

        if not diag.empty:
            def _fmt_qty(x):
                if pd.isna(x): return ""
                return f"{float(x):.6f}".rstrip("0").rstrip(".")
            def _fmt_money(x):
                if pd.isna(x): return ""
                return f"€{float(x):,.2f}"

            st.markdown("**Diagnostics**")
            st.dataframe(
                diag.sort_values(["Kind","ISIN"]).style.format({
                    "ManualQty": _fmt_qty,
                    "ManualEUR": _fmt_money,
                    "ClosestBuyQty": _fmt_qty,
                    "QtyDiff": _fmt_qty,
                }),
                use_container_width=True
            )
        else:
            st.caption("No outstanding items to diagnose ✅")

with st.expander("🧪 Reconciliation (debug) – identify FIFO differences"):
    if 'out' not in locals() or out.empty:
        st.caption("Load a file first.")
    else:
        df_dbg = out.copy()
        df_dbg["Date"] = pd.to_datetime(df_dbg["Date"], errors="coerce")
        df_dbg["Year"] = df_dbg["__year"]

        # Pick a year to inspect
        yrs = sorted(x for x in df_dbg["Year"].dropna().unique())
        yr_pick = st.selectbox("Year to inspect", yrs, index=len(yrs)-1 if yrs else 0) if yrs else None

        if yr_pick is not None:
            d = df_dbg[df_dbg["Year"].eq(yr_pick)].copy()

            # Top ISINs by realised P/L (sells only)
            by_isin = (
                d[d["Type"].eq("Sell")]
                .groupby(["ISIN","Ticker - Name"], dropna=False)["Gain/Loss"]
                .sum().sort_values(ascending=False).reset_index()
            )
            st.markdown("**Top realised P/L by ISIN**")
            st.dataframe(by_isin.head(20).style.format({"Gain/Loss": lambda x: f"€{x:,.2f}"}), use_container_width=True)

            # Time hygiene checks (blank/HH:MM only?) from the *raw-ish* columns if you still have them
            # We'll reconstruct a 'TimeText' best-effort from Description is not reliable; so use minute buckets
            d["Minute"] = d["Date"].dt.strftime("%Y-%m-%d %H:%M")
            same_minute_counts = (
                d[d["Type"].isin(["Buy","Sell"])]
                .groupby(["ISIN","Minute"])["Order ID"].count()
                .reset_index(name="rows_in_minute")
            )
            hot_minutes = same_minute_counts[same_minute_counts["rows_in_minute"] > 1]
            st.markdown("**Same-minute trade clusters (could change FIFO ordering if Time precision is limited):**")
            st.dataframe(hot_minutes.head(100), use_container_width=True)

            # Drill into a specific ISIN to see exact FIFO order we used
            opt = ["(select)"] + by_isin["ISIN"].head(50).tolist()
            pick_isin = st.selectbox("Drill into ISIN", options=opt, index=0)
            if pick_isin != "(select)":
                z = d[d["ISIN"] == pick_isin].copy()
                z = z.sort_values(["Date","Type","Order ID"], kind="mergesort")
                st.dataframe(
                    z.loc[:, [c for c in [
                        "Date","Type","Quantity","Price","Price_EUR","Total (EUR, fee-adj)","Gain/Loss",
                        "Order ID","Description"
                    ] if c in z.columns]],
                    use_container_width=True
                )

            # Quick counts that hint at time parsing issues:
            # rows with midnight times (00:00), which often mean date-only parse
            d["_hhmm"] = d["Date"].dt.strftime("%H:%M")
            midnight_rows = d[(d["_hhmm"] == "00:00") & d["Type"].isin(["Buy","Sell"])]
            st.markdown(f"**Trades at 00:00 in {yr_pick} (date-only parsed): {len(midnight_rows)}**")
            if len(midnight_rows):
                st.dataframe(midnight_rows[["Date","ISIN","Type","Order ID","Description"]].head(50), use_container_width=True)

# --------- Drilldown ---------
st.markdown("#### View full trade history")

# Build clean instrument list
ins = (
    out.loc[:, ["ISIN", "Ticker - Name"]]
       .dropna(subset=["ISIN", "Ticker - Name"])
       .astype({"ISIN": str, "Ticker - Name": str})
)
ins = ins[(ins["ISIN"].str.strip() != "") & (ins["Ticker - Name"].str.strip() != "")]
ins = ins.drop_duplicates().copy()

if ins.empty:
    st.info("No instruments found (ISIN or name missing in the data).")
else:
    ins["label"] = ins["Ticker - Name"] + " — " + ins["ISIN"]
    ins = ins.sort_values("label")
    labels = ["(none)"] + ins["label"].tolist()
    choice = st.selectbox("Select an instrument:", options=labels, index=0)

    if choice != "(none)":
        picked_isin = choice.rsplit(" — ", 1)[-1]
        detail = out[out["ISIN"] == picked_isin].drop(columns=["__year"], errors="ignore").copy()
        # show corporate actions rows too so you can see stock splits
        detail = detail.sort_values(["Date", "Type", "Order ID"], kind="mergesort")

        def fmt_date(d): return "" if pd.isna(d) else d.strftime("%d %b %Y")
        def fmt_money(x):
            if pd.isna(x): return ""
            try: return f"{x:,.2f}"
            except: return str(x)
        def fmt_qty(x):
            if pd.isna(x): return ""
            return f"{x:.6f}".rstrip("0").rstrip(".")

        # ensure the useful columns show if present
        cols = [c for c in [
            "Date","Ticker - Name","ISIN","Type","Asset","Currency","Quantity","Price",
            "Fee","Total","Total (EUR)","Total (EUR, fee-adj)","Gain/Loss","Order ID","Description"
        ] if c in detail.columns]
        detail = detail.loc[:, cols]

        styler = detail.style.format({
            "Date": fmt_date, "Quantity": fmt_qty, "Price": fmt_money,
            "Fee": fmt_money, "Total": fmt_money,
            "Total (EUR)": fmt_money, "Total (EUR, fee-adj)": fmt_money,
            "Gain/Loss": fmt_money,
        })
        st.dataframe(styler, use_container_width=True)

# Download
filename_year = "all" if year_choice == "All" else str(year_choice)
csv_bytes = filtered.drop(columns=["__year"]).to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download filtered table (CSV)", data=csv_bytes, file_name=f"trades_{filename_year}.csv", mime="text/csv")

st.caption(
    "Dividend rows show gross (Total) and withholding tax (Fee) in the original currency, with EUR left blank. "
    "FX-to-EUR mapping is applied to trades and eligible rows. FIFO realised P/L is computed in EUR. "
    "Annual Summary uses 33% for Shares (with optional €1,270 exemption & loss carry-forward) and 41% Exit Tax for ETFs."
)