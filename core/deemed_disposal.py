from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.settings import DEFAULT_EXIT_TAX_RATE_ETF

# ===== Deemed Disposal (ETFs) — Planner + Estimator (no pipeline changes) =====
EXIT_TAX_RATE = DEFAULT_EXIT_TAX_RATE_ETF


def _is_exit_tax_asset_row(row: pd.Series) -> bool:
    asset = str(row.get("Asset") or row.get("AssetInference") or "").lower()
    return ("etf" in asset) or ("fund" in asset) or ("gross" in asset)


def _eight_year_anniversary(d: pd.Timestamp) -> pd.Timestamp:
    try:
        return d + pd.DateOffset(years=8)
    except Exception:
        return pd.Timestamp(d.year + 8, d.month, 1) + pd.offsets.MonthEnd(0)


def _last_trade_price_eur_before(out_df: pd.DataFrame, isin: str, when: pd.Timestamp) -> Optional[float]:
    df = out_df[out_df["ISIN"].astype(str).eq(isin)].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df[(df["Type"].isin(["Buy", "Sell"])) & (df["Date"] <= when)].sort_values(by="Date", kind="mergesort")
    if df.empty:
        return None
    if "Price_EUR" in df.columns:
        price = pd.to_numeric(df["Price_EUR"], errors="coerce")
    else:
        tot = pd.to_numeric(df.get("Total (EUR, fee-adj)", df.get("Total (EUR)", np.nan)), errors="coerce")
        qty = pd.to_numeric(df.get("Quantity"), errors="coerce").replace(0, np.nan)
        price = tot / qty
    price = price.dropna()
    return float(price.iloc[-1]) if not price.empty else None


def _replay_fifo_lots_from_out(out_df: pd.DataFrame) -> Dict[str, List[Dict]]:
    needed = {"ISIN", "Date", "Order ID", "Type", "Quantity"}
    missing = needed - set(out_df.columns)
    if missing:
        raise KeyError(f"`out` is missing required columns: {', '.join(sorted(missing))}")

    df = out_df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.sort_values(by=["ISIN", "Date", "Order ID"], kind="mergesort")

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
                unit_cost = float(pd.to_numeric(r.get("Price_EUR"), errors="coerce"))
                if np.isnan(unit_cost) or unit_cost == 0.0:
                    tot_row = pd.to_numeric(r.get("Total (EUR, fee-adj)", r.get("Total (EUR)", np.nan)), errors="coerce")
                    if pd.notna(tot_row) and q > 0:
                        unit_cost = float(tot_row / q)
                    else:
                        unit_cost = np.nan

                lots.append({"acq": r["Date"], "qty": q, "unit_cost_eur": unit_cost})
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
        lots_by_isin[isin] = lots
    return lots_by_isin


def deemed_plan_and_estimates(out_df: pd.DataFrame, asof: Optional[pd.Timestamp] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
            unit_fmv = _last_trade_price_eur_before(out_df, isin, dd)
            qty = float(L["qty"])
            fmv = unit_fmv * qty if unit_fmv is not None else np.nan
            base = unit_cost * qty
            gain = (fmv - base) if (fmv == fmv) else np.nan
            est_tax = max(gain, 0.0) * EXIT_TAX_RATE if (gain == gain) else np.nan
            rows.append(
                {
                    "ISIN": isin,
                    "AcquisitionDate": acq.normalize(),
                    "DeemedDate": dd.normalize(),
                    "QtyRemaining": qty,
                    "UnitCostEUR": unit_cost,
                    "ProposedFMV_UnitEUR": unit_fmv,
                    "ProposedFMV_EUR": fmv,
                    "EstGain_EUR": gain,
                    "EstExitTax_EUR": est_tax,
                }
            )

    est = pd.DataFrame(rows)
    if est.empty:
        planner = est.copy()
    else:
        est["__ValNeeded"] = est["ProposedFMV_UnitEUR"].isna()
        est["__year"] = pd.to_datetime(est["DeemedDate"]).dt.year
        est = est.sort_values(by=["ISIN", "DeemedDate", "AcquisitionDate"], kind="mergesort")
        planner = est[["ISIN", "AcquisitionDate", "DeemedDate", "QtyRemaining", "__year"]].copy()

    return planner, est
