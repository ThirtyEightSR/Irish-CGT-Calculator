from __future__ import annotations

import re
from typing import Dict, List

import numpy as np
import pandas as pd


ETF_PROVIDERS = [
    "vanguard",
    "ishares",
    "vaneck",
    "wisdomtree",
    "amundi",
    "invesco",
    "xtrackers",
    "spdr",
    "lyxor",
    "ubs",
    "schwab",
]
ETF_TOKENS = ("etf", "ucits", "etn", "etc")


def _is_exit_tax_asset_row(row: pd.Series) -> bool:
    asset = str(row.get("Asset", "")).lower()
    if asset == "etf":
        return True
    name = str(row.get("Ticker - Name", row.get("Product", ""))).lower()
    if any(tok in name for tok in ETF_TOKENS):
        return True
    if any(provider in name for provider in ETF_PROVIDERS):
        return True
    return False


def year_today() -> int:
    return pd.Timestamp.today().year


def replay_fifo_lots_all(out_df: pd.DataFrame) -> Dict[str, List[Dict]]:
    if out_df is None or out_df.empty:
        return {}

    required = {"ISIN", "Date", "Type"}
    if not required.issubset(out_df.columns):
        return {}

    df = out_df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df[df["Type"].isin(["Buy", "Sell"])].copy()
    if df.empty:
        return {}

    idx = df.index
    q_candidates: List[pd.Series] = []
    if "Quantity" in df.columns:
        q_candidates.append(pd.to_numeric(df["Quantity"], errors="coerce"))
    if "__qty_desc" in df.columns:
        q_candidates.append(pd.to_numeric(df["__qty_desc"], errors="coerce"))

    desc = (df["Description"] if "Description" in df.columns else pd.Series("", index=df.index)).astype(str)
    q_desc = pd.to_numeric(desc.str.extract(r"\b(?:buy|sell)\s+(\d+(?:\.\d+)?)", flags=re.I)[0], errors="coerce")
    q_candidates.append(q_desc)

    qty = pd.Series(np.nan, index=idx, dtype="float64")
    for cand in q_candidates:
        if cand is None:
            continue
        cand = pd.to_numeric(cand, errors="coerce").reindex(idx)
        qty = qty.where(qty.notna(), cand)

    tot_eur = pd.Series(np.nan, index=idx, dtype="float64")
    price_eur = pd.Series(np.nan, index=idx, dtype="float64")
    for cand in ["Total (EUR, fee-adj)", "Total (EUR)"]:
        if cand in df.columns:
            s = pd.to_numeric(df[cand], errors="coerce").reindex(idx)
            tot_eur = tot_eur.where(tot_eur.notna(), s)
    for cand in ["Price_EUR", "Unit_EUR", "UnitPrice"]:
        if cand in df.columns:
            s = pd.to_numeric(df[cand], errors="coerce").reindex(idx)
            price_eur = price_eur.where(price_eur.notna(), s)

    mask_q_from_tot = qty.isna() & tot_eur.notna() & price_eur.notna() & (price_eur > 0)
    if mask_q_from_tot.any():
        qty.loc[mask_q_from_tot] = (tot_eur.loc[mask_q_from_tot].abs() / price_eur.loc[mask_q_from_tot]).round(6)

    qty = pd.to_numeric(qty, errors="coerce").where(lambda s: s > 1e-12, np.nan)
    df["__qty_for_fifo"] = qty

    unit_cost = price_eur.copy()
    mask_uc = unit_cost.isna() & tot_eur.notna() & df["__qty_for_fifo"].gt(0)
    if mask_uc.any():
        unit_cost.loc[mask_uc] = (tot_eur.loc[mask_uc].abs() / df.loc[mask_uc, "__qty_for_fifo"]).values
    df["__unit_cost"] = pd.to_numeric(unit_cost, errors="coerce")

    order_cols = [c for c in ["ISIN", "Date", "Order ID", "__row_id"] if c in df.columns]
    df = df.sort_values(by=order_cols, kind="mergesort")

    lots_by_isin: Dict[str, List[Dict]] = {}
    for isin, g in df.groupby("ISIN", sort=False):
        lots: List[Dict] = []
        for _, r in g.iterrows():
            t = str(r["Type"])
            q = float(r["__qty_for_fifo"]) if pd.notna(r["__qty_for_fifo"]) else 0.0
            if q <= 1e-12:
                continue
            if t == "Buy":
                uc_eur = float(r["__unit_cost"]) if pd.notna(r["__unit_cost"]) else np.nan
                desc = str(r.get("Description", ""))
                m = re.search(r"@([\d,]*\.?\d+)\s*([A-Z]{3})", desc)
                unit_native = np.nan
                ccy = ""
                if m:
                    raw_num = m.group(1).replace(",", "")
                    try:
                        unit_native = float(raw_num)
                    except ValueError:
                        unit_native = np.nan
                    ccy = m.group(2).upper()
                else:
                    fx = pd.to_numeric(r.get("FX_Rate"), errors="coerce")
                    if pd.notna(fx) and fx > 0:
                        unit_native = uc_eur / fx
                    ccy = str(r.get("FXCCY") or "").strip().upper()

                lots.append(
                    {
                        "acq": r["Date"],
                        "qty": q,
                        "unit_cost_eur": uc_eur,
                        "unit_cost_native": float(unit_native) if pd.notna(unit_native) else np.nan,
                        "ccy": ccy,
                    }
                )
            else:
                qty_to_sell = q
                j = 0
                while qty_to_sell > 1e-12 and j < len(lots):
                    take = min(lots[j]["qty"], qty_to_sell)
                    lots[j]["qty"] -= take
                    qty_to_sell -= take
                    if lots[j]["qty"] <= 1e-12:
                        j += 1
                lots = [L for L in lots if L["qty"] > 1e-12]
        lots_by_isin[str(isin)] = lots
    return lots_by_isin


def asset_kind_for_isin(out_df: pd.DataFrame, isin: str) -> str:
    sub = out_df[out_df["ISIN"].astype(str).eq(isin)]
    if sub.empty:
        return "share"
    if "Asset" in sub.columns:
        a = sub["Asset"].astype(str).str.lower()
        if (a == "etf").any():
            return "etf"
        if (a == "share").any():
            return "share"
    any_etf = sub.apply(_is_exit_tax_asset_row, axis=1).any()
    return "etf" if any_etf else "share"


def last_known_unit_price_eur(out_df: pd.DataFrame, isin: str) -> float | None:
    df = out_df[out_df["ISIN"].astype(str).eq(isin)].copy()
    if df.empty:
        return None
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values(by="Date")
    if "Price_EUR" in df.columns:
        px = pd.to_numeric(df["Price_EUR"], errors="coerce").dropna()
        return float(px.iloc[-1]) if not px.empty else None
    tot = pd.to_numeric(df.get("Total (EUR, fee-adj)", df.get("Total (EUR)")), errors="coerce")
    qty = pd.to_numeric(df.get("Quantity"), errors="coerce").replace(0, np.nan)
    u = (tot / qty).dropna()
    return float(u.iloc[-1]) if not u.empty else None


def available_qty(out_df: pd.DataFrame, isin: str) -> float:
    lots = replay_fifo_lots_all(out_df).get(str(isin), [])
    return float(sum(float(L.get("qty", 0.0) or 0.0) for L in lots)) if lots else 0.0


def fifo_cost_for_sale(out_df: pd.DataFrame, isin: str, qty: float) -> float:
    lots = list(replay_fifo_lots_all(out_df).get(str(isin), []))
    lots.sort(key=lambda L: pd.to_datetime(L.get("acq"), errors="coerce"))
    need = float(qty)
    cost = 0.0
    for L in lots:
        if need <= 1e-12:
            break
        have = float(L.get("qty", 0.0) or 0.0)
        take = min(have, need)
        cost += take * float(L.get("unit_cost_eur", 0.0) or 0.0)
        need -= take
    return float(cost)


def ytd_realised_gains(out_df: pd.DataFrame, year: int) -> tuple[float, float]:
    df = out_df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df[df["Date"].dt.year.eq(year)]
    sells = df[df["Type"].eq("Sell")].copy()
    gl = pd.to_numeric(sells.get("Gain/Loss"), errors="coerce").fillna(0.0)
    is_etf = sells.apply(_is_exit_tax_asset_row, axis=1)
    shares_gl = float(gl[~is_etf].sum())
    etf_gl = float(gl[is_etf].sum())
    return shares_gl, etf_gl


def carry_forward_shares_to_year(out_df: pd.DataFrame, year: int, use_exemption: bool, exemption_val: float) -> float:
    if out_df.empty:
        return 0.0
    df = out_df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["__year"] = df["Date"].dt.year

    if "Gain/Loss" not in df.columns:
        return 0.0

    is_share = df["Asset"].astype(str).str.lower().eq("share") if "Asset" in df.columns else ~df.apply(_is_exit_tax_asset_row, axis=1)
    y = (
        df[df["Type"].eq("Sell") & is_share]
        .assign(__gl=pd.to_numeric(df["Gain/Loss"], errors="coerce"))
        .groupby("__year")["__gl"]
        .sum(min_count=1)
        .fillna(0.0)
        .to_dict()
    )

    carry = 0.0
    for yr in sorted(k for k in y.keys() if pd.notna(k) and k < year):
        realised = float(y[yr])
        if realised >= 0:
            used = min(carry, realised)
            remaining_gain = realised - used
            if use_exemption:
                _ = min(exemption_val, remaining_gain)
            carry = max(0.0, carry - used)
        else:
            carry += abs(realised)
    return float(carry)


def tax_shares_delta(
    ytd_gain_now: float,
    hypo_gain: float,
    carry_in: float,
    use_exemption: bool,
    exemption_val: float,
    rate: float,
) -> tuple[float, float, float]:
    def taxable(total_gain: float) -> float:
        if total_gain <= 0:
            return 0.0
        remaining = max(0.0, total_gain - carry_in)
        ex_used = min(exemption_val, remaining) if use_exemption else 0.0
        return max(0.0, remaining - ex_used)

    t_now = rate * taxable(ytd_gain_now)
    t_new = rate * taxable(ytd_gain_now + hypo_gain)
    return float(t_now), float(t_new), float(t_new - t_now)


def tax_etf_delta(ytd_gl_now: float, hypo_gl: float, rate: float) -> tuple[float, float, float]:
    taxable_now = max(0.0, ytd_gl_now)
    taxable_new = max(0.0, ytd_gl_now + hypo_gl)
    t_now = rate * taxable_now
    t_new = rate * taxable_new
    return float(t_now), float(t_new), float(t_new - t_now)
