from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def _is_rich_missing_transactions_file(df: Optional[pd.DataFrame]) -> bool:
    if df is None or df.empty:
        return False
    cols = {str(c).strip() for c in df.columns}
    return {"Date", "Type", "ISIN", "Quantity"}.issubset(cols)


def _normalize_manual(opening_lots_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    return pd.DataFrame(columns=["ISIN", "Quantity", "EUR_Value", "Unit_EUR"])


def merge_missing_transactions(out_df: pd.DataFrame, opening_lots_df: pd.DataFrame) -> pd.DataFrame:
    if opening_lots_df is None or opening_lots_df.empty:
        return out_df

    manual = opening_lots_df.copy()
    manual.columns = [c.strip() for c in manual.columns]
    if not _is_rich_missing_transactions_file(manual):
        return out_df

    manual["Type"] = manual.get("Type", "Buy").astype(str).str.strip()
    manual["ISIN"] = manual["ISIN"].astype(str).str.strip()
    manual["Date"] = pd.to_datetime(manual.get("Date"), errors="coerce", dayfirst=True).fillna(pd.Timestamp("2010-01-01"))
    manual["Quantity"] = pd.to_numeric(manual.get("Quantity"), errors="coerce")

    if "Price_EUR" in manual.columns:
        price_eur = pd.to_numeric(manual["Price_EUR"], errors="coerce")
    elif "Unit_EUR" in manual.columns:
        price_eur = pd.to_numeric(manual["Unit_EUR"], errors="coerce")
    else:
        price_eur = pd.Series(np.nan, index=manual.index)

    if "Total (EUR)" in manual.columns:
        total_eur = pd.to_numeric(manual["Total (EUR)"], errors="coerce")
    elif "Total_EUR" in manual.columns:
        total_eur = pd.to_numeric(manual["Total_EUR"], errors="coerce")
    else:
        total_eur = pd.Series(np.nan, index=manual.index)

    missing_price = price_eur.isna() & total_eur.notna() & manual["Quantity"].notna() & manual["Quantity"].ne(0)
    price_eur = price_eur.where(~missing_price, total_eur.abs() / manual["Quantity"].abs())
    missing_total = total_eur.isna() & manual["Quantity"].notna() & price_eur.notna()
    total_eur = total_eur.where(~missing_total, manual["Quantity"].abs() * price_eur)

    t_norm = manual["Type"].str.lower()
    is_buy = t_norm.eq("buy")
    is_sell = t_norm.eq("sell")
    keep = (
        manual["ISIN"].ne("")
        & (is_buy | is_sell)
        & manual["Quantity"].notna()
        & price_eur.notna()
        & total_eur.notna()
    )
    if not keep.any():
        return out_df

    signed_cash = np.where(is_buy, -total_eur.abs(), total_eur.abs())
    product_series = manual.get("Product", manual.get("Ticker - Name", manual["ISIN"])).astype(str).str.strip()
    product_series = product_series.mask(product_series.str.lower().isin(["", "nan", "none", "nat"]), manual["ISIN"])
    default_desc = np.where(
        is_buy,
        "Buy " + manual["Quantity"].abs().astype(str) + " " + product_series + "@" + price_eur.astype(str) + " EUR",
        "Sell " + manual["Quantity"].abs().astype(str) + " " + product_series + "@" + price_eur.astype(str) + " EUR",
    )
    desc_series = manual.get("Description", pd.Series(default_desc, index=manual.index)).astype(str).str.strip()
    desc_series = desc_series.mask(
        desc_series.str.lower().isin(["", "nan", "none", "nat"]),
        pd.Series(default_desc, index=manual.index),
    )
    order_id = manual.get("Order ID", pd.Series("", index=manual.index)).fillna("").astype(str).str.strip()
    fallback_oid = pd.Series([f"IMPORT-{i:04d}" for i in range(1, len(order_id) + 1)], index=order_id.index)
    order_id = order_id.mask(order_id.eq(""), fallback_oid)

    canonical = pd.DataFrame(
        {
            "Date": manual["Date"],
            "Time": None,
            "Value date": None,
            "Product": product_series,
            "ISIN": manual["ISIN"],
            "Description": desc_series,
            "FX": "EUR",
            "Change": signed_cash,
            "Cash Movements": signed_cash,
            "Balance": None,
            "Order ID": order_id,
            "Currency": "EUR",
            "__Broker": "MANUAL",
            "__SourceFile": "missing_transactions",
        }
    ).loc[keep]

    merged = pd.concat([out_df, canonical], ignore_index=True, sort=False)
    if "Date" in merged.columns:
        merged["Date"] = pd.to_datetime(merged["Date"], errors="coerce")
        merged = merged.sort_values("Date", kind="stable").reset_index(drop=True)
    return merged


def _apply_missing_precedence(out_df: pd.DataFrame, opening_lots_df: pd.DataFrame) -> pd.DataFrame:
    if out_df is None or out_df.empty or opening_lots_df is None or opening_lots_df.empty:
        return out_df
    if not _is_rich_missing_transactions_file(opening_lots_df):
        return out_df

    df = out_df.copy()
    if "ISIN" not in opening_lots_df.columns:
        return df

    manual_isins = (
        opening_lots_df["ISIN"].astype(str).str.strip().dropna().unique().tolist()
    )
    manual_isins = set(manual_isins)
    if not manual_isins:
        return df

    if "Product" in df.columns:
        mask_missing_import = (
            df["ISIN"].astype(str).isin(manual_isins)
            & df["Product"].astype(str).str.contains("Missing Import", case=False, na=False)
        )
        df = df.loc[~mask_missing_import].copy()

    if "Description" in df.columns:
        desc = df["Description"].astype(str)
        mask_incoming_buy = (
            df["ISIN"].astype(str).isin(manual_isins)
            & desc.str.contains("incoming transfer", case=False, na=False)
        )
        df = df.loc[~mask_incoming_buy].copy()

    return df.reset_index(drop=True)
