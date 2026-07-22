from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from core.settings import DEFAULT_EXIT_TAX_RATE_ETF


ETF_PROVIDER_KEYWORDS = {
    "ishares",
    "vanguard",
    "vaneck",
    "xtrackers",
    "spdr",
    "lyxor",
    "invesco",
    "wisdomtree",
    "amundi",
    "first trust",
    "global x",
    "ubs",
    "hsbc",
    "etf",
}


def _is_etf_by_name(product_name: str) -> bool:
    s = (product_name or "").strip().lower()
    return any(k in s for k in ETF_PROVIDER_KEYWORDS)


def build_cgt1_export(out: pd.DataFrame, split_audit_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    cols = [
        "CGT Period",
        "Date Acquired",
        "Date Disposed",
        "Ticker - Name",
        "Asset Type",
        "ISIN",
        "Quantity",
        "Sell Proceeds (EUR)",
        "Gain/Loss (EUR)",
        "Buys + Fees (EUR)",
        "Order ID",
        "Broker",
        "Source File",
    ]

    if out is None or out.empty:
        return pd.DataFrame(columns=cols)

    df = out.copy()
    disp_mask = df["Type"].isin(["Sell", "Deemed Disposal"]) & df["Asset"].astype(str).str.lower().ne("etf")
    df = df.loc[disp_mask].copy()
    if df.empty:
        return pd.DataFrame(columns=cols)

    proceeds_col = "Total (EUR, fee-adj)" if "Total (EUR, fee-adj)" in df.columns else "Total (EUR)"
    if proceeds_col not in df.columns:
        df[proceeds_col] = pd.to_numeric(df.get("Total (EUR)"), errors="coerce")

    proceeds = pd.to_numeric(df[proceeds_col], errors="coerce")
    gl = pd.to_numeric(df.get("Gain/Loss"), errors="coerce")
    costs = proceeds - gl
    d = pd.to_datetime(df["Date"], errors="coerce")

    year_series = d.dt.year
    latest_year = int(year_series.max()) if year_series.notna().any() else None
    is_latest = year_series.eq(latest_year) if latest_year is not None else pd.Series(False, index=df.index)
    bucket_latest = np.where(d.dt.month.eq(12), "Dec", "Jan–Nov")
    cgt_period = np.where(is_latest, bucket_latest, year_series.astype("Int64").astype("string"))
    cgt_period = pd.Series(cgt_period, index=df.index).fillna("Unknown")

    if "Ticker - Name" in df.columns:
        name_series = df["Ticker - Name"].astype(str)
    elif "Product" in df.columns:
        name_series = df["Product"].astype(str)
    else:
        name_series = pd.Series("", index=df.index, dtype="object").astype(str)

    is_etf_flag = pd.Series(False, index=df.index)
    if "Asset" in df.columns:
        is_etf_flag = is_etf_flag | df["Asset"].astype(str).str.upper().eq("ETF")
    is_etf_flag = is_etf_flag | name_series.map(lambda s: bool(_is_etf_by_name(s)))
    asset_type = np.where(is_etf_flag, "ETF", "Share")

    acquired_ser = pd.Series("Various", index=df.index, dtype="object")
    if split_audit_df is not None and not split_audit_df.empty:
        sad = split_audit_df.copy()
        possible_sell_keys = [c for c in sad.columns if "__row_id" in c.lower() or ("sell" in c.lower() and "row" in c.lower())]
        possible_buy_dates = [
            c
            for c in sad.columns
            if ("buy" in c.lower() and "date" in c.lower()) or c.lower() in {"lot_date", "buy_date", "acq_date"}
        ]
        if possible_sell_keys and possible_buy_dates:
            sell_key = possible_sell_keys[0]
            buy_date_col = possible_buy_dates[0]
            try:
                sad[sell_key] = pd.to_numeric(sad[sell_key], errors="coerce").astype("Int64")
            except Exception:
                pass
            sad[buy_date_col] = pd.to_datetime(sad[buy_date_col], errors="coerce")
            acq_map = sad.groupby(sell_key, dropna=True)[buy_date_col].min().to_dict()

            if "__row_id" in df.columns:
                sell_ids = pd.to_numeric(df["__row_id"], errors="coerce")
                mapped = []
                for sid in sell_ids:
                    dt0 = acq_map.get(int(sid)) if pd.notna(sid) else None
                    mapped.append(dt0.date() if (dt0 is not None and pd.notna(dt0)) else "Various")
                acquired_ser = pd.Series(mapped, index=df.index, dtype="object")

    if (acquired_ser == "Various").any():
        all_rows = out.copy()
        all_rows["Date"] = pd.to_datetime(all_rows["Date"], errors="coerce")
        buys = all_rows.loc[all_rows["Type"].eq("Buy"), ["ISIN", "Date"]].dropna().sort_values(by=["ISIN", "Date"])
        buy_dates_by_isin = {k: v["Date"].tolist() for k, v in buys.groupby("ISIN")}
        d_disp = pd.to_datetime(df["Date"], errors="coerce")

        def _fallback_acq(i: int):
            if acquired_ser.iat[i] != "Various":
                return acquired_ser.iat[i]
            isin = str(df["ISIN"].iat[i])
            disp = d_disp.iat[i]
            if pd.isna(disp):
                return "Various"
            candidates = [dt for dt in buy_dates_by_isin.get(isin, []) if dt <= disp]
            return min(candidates).date() if candidates else "Various"

        acquired_ser = pd.Series([_fallback_acq(i) for i in range(len(df))], index=df.index, dtype="object")

    broker = df.get("__Broker", "UNKNOWN")
    source = df.get("__SourceFile", "uploads")

    export = pd.DataFrame(
        {
            "CGT Period": cgt_period,
            "Date Acquired": acquired_ser.astype(str).replace({"NaT": "Various"}),
            "Date Disposed": d.astype("string").str[:10].fillna(""),
            "Ticker - Name": name_series,
            "Asset Type": asset_type,
            "ISIN": df.get("ISIN", ""),
            "Quantity": pd.to_numeric(df.get("Quantity"), errors="coerce"),
            "Buys + Fees (EUR)": costs.round(2),
            "Sell Proceeds (EUR)": proceeds.round(2),
            "Gain/Loss (EUR)": gl.round(2),
            "Order ID": df.get("Order ID", ""),
            "Broker": broker if isinstance(broker, pd.Series) else str(broker),
            "Source File": source if isinstance(source, pd.Series) else str(source),
        }
    )
    return export.sort_values(by=["CGT Period", "Date Disposed"], kind="stable").reset_index(drop=True)


def build_form12_export(out: pd.DataFrame, exit_tax_rate: float = DEFAULT_EXIT_TAX_RATE_ETF) -> pd.DataFrame:
    cols = [
        "Tax Year",
        "Date",
        "Chargeable Event",
        "Ticker - Name",
        "ISIN",
        "Asset",
        "Quantity",
        "Proceeds (EUR)",
        "Cost (EUR)",
        "Gain/Loss (EUR)",
        "Taxable Gain (EUR)",
        f"Tax @ {int(exit_tax_rate*100)}% (EUR)",
        "Order ID",
        "Broker",
        "Source File",
    ]

    if out is None or out.empty:
        return pd.DataFrame(columns=cols)

    df = out.copy()
    mask = df["Asset"].astype(str).str.lower().eq("etf") & df["Type"].isin(["Sell", "Deemed Disposal"])
    df = df.loc[mask].copy()
    if df.empty:
        return pd.DataFrame(columns=cols)

    d = pd.to_datetime(df["Date"], errors="coerce")
    tax_year = d.dt.year.astype("Int64").astype("string")
    date_str = d.dt.strftime("%Y-%m-%d").fillna("").astype("string")

    proceeds_col = "Total (EUR, fee-adj)" if "Total (EUR, fee-adj)" in df.columns else "Total (EUR)"
    proceeds = pd.to_numeric(df.get(proceeds_col), errors="coerce")
    gl = pd.to_numeric(df.get("Gain/Loss"), errors="coerce")
    cost = proceeds - gl
    taxable = gl.clip(lower=0)
    tax_due = taxable * float(exit_tax_rate)
    event = np.where(df["Type"].eq("Deemed Disposal"), "Deemed Disposal", "Disposal")

    export = pd.DataFrame(
        {
            "Tax Year": tax_year,
            "Date": date_str,
            "Chargeable Event": event,
            "Ticker - Name": df.get("Ticker - Name", df.get("Product", "")),
            "ISIN": df.get("ISIN", ""),
            "Asset": df.get("Asset", "ETF"),
            "Quantity": pd.to_numeric(df.get("Quantity"), errors="coerce"),
            "Proceeds (EUR)": proceeds.round(2),
            "Cost (EUR)": cost.round(2),
            "Gain/Loss (EUR)": gl.round(2),
            "Taxable Gain (EUR)": taxable.round(2),
            f"Tax @ {int(exit_tax_rate*100)}% (EUR)": tax_due.round(2),
            "Order ID": df.get("Order ID", ""),
            "Broker": df.get("__Broker", "UNKNOWN"),
            "Source File": df.get("__SourceFile", "uploads"),
        }
    )
    return export.sort_values(by=["Tax Year", "Date", "Ticker - Name"], kind="stable").reset_index(drop=True)
