from __future__ import annotations

import re
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from services.parsing_helpers import (
    compute_eur_cash_from_fx as _compute_eur_cash_from_fx,
    detect_isin_product_mappings as _detect_isin_product_mappings,
    direct_eur_from_rate as _direct_eur_from_rate,
    find_between as _find_between,
    infer_asset,
    parse_desc_numbers,
    parse_split_factor,
    parse_type,
    safe_col as _safe_col,
    to_numeric_flexible as _to_numeric_flexible,
)
from services.output_builder import apply_corporate_actions_and_map_fx, build_out_table, consolidate_fifo

__isin_roll_map: dict[str, str] = {}

AUDIT_COLS = [
    "ISIN",
    "Product",
    "Row kind",
    "Trade date",
    "Order ID",
    "Split date",
    "Factor",
    "Qty (before)",
    "Qty (after)",
    "Unit px (before)",
    "Unit px (after)",
    "Unit px EUR (before)",
    "Unit px EUR (after)",
]


def build_output(df_norm: pd.DataFrame, opening_lots: Optional[pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:

    # Ensure split collectors exist before any use
    split_events: List[Tuple[str, pd.Timestamp, float]] = []
    split_audit: List[List[object]] = []
    date_col = _safe_col(df_norm, "Date")
    try:
        time_col = _safe_col(df_norm, "Time")
        has_time = True
    except Exception:
        time_col = None
        has_time = False

    product_col = _safe_col(df_norm, "Product")
    isin_col = _safe_col(df_norm, "ISIN")
    descr_col = _safe_col(df_norm, "Description")
    change_col = _safe_col(df_norm, "Change")
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

    cash_between = _to_numeric_flexible(df[_find_between(df_norm, change_col, balance_col)])
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
                        h = tparts[0]
                        m = tparts[1]
                        s = tparts[2] if len(tparts) > 2 else 0
                        return dparsed.replace(hour=h, minute=m, second=s).to_pydatetime()
                    except Exception:
                        pass
                # fallback: explicit patterns
                for dfmt in ("%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d", "%d.%m.%Y"):
                    try:
                        base = datetime.strptime(dstr, dfmt)
                        tparts = [int(x) for x in tstr.split(":")]
                        h = tparts[0]
                        m = tparts[1]
                        s = tparts[2] if len(tparts) > 2 else 0
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
                "%d-%m-%Y %H:%M",
                "%d/%m/%Y %H:%M",
                "%Y-%m-%d %H:%M",
                "%d.%m.%Y %H:%M",
                "%d-%m-%Y %H:%M:%S",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M:%S.%f",
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
    df["__is_fx_debit"] = df["__Type"].eq("FX Debit")

    # Parse qty, price, and currency from description
    q_desc, p_desc, c_desc = zip(*df[descr_col].apply(parse_desc_numbers).tolist())
    df["__qty_desc"] = pd.to_numeric(q_desc, errors="coerce")
    df["__price_desc"] = pd.to_numeric(p_desc, errors="coerce")
    df["__ccy_desc"] = pd.Series(c_desc, dtype="object")

    # Fallback: for trades without '@price' where quantity wasn't parsed,
    # extract plain "buy|sell <N>" from the Description (e.g., "Sell 140 ...")
    _missing_qty = df["__qty_desc"].isna() & df["__Type"].isin(["Buy", "Sell"])
    if _missing_qty.any():
        _qty_fallback = (
            df.loc[_missing_qty, descr_col].astype(str).str.extract(r"\b(?:buy|sell)\s+(\d+(?:\.\d+)?)", flags=re.I)[0]
        )
        df.loc[_missing_qty, "__qty_desc"] = pd.to_numeric(_qty_fallback, errors="coerce")

    # Fallback: if description didn’t contain a quantity (e.g., “Opening lot”), and
    # the DataFrame has a real Quantity column (from opening-lots ingest),
    # use it so FIFO can see these shares.
    if "Quantity" in df.columns:
        df["__qty_desc"] = df["__qty_desc"].fillna(pd.to_numeric(df["Quantity"], errors="coerce"))

    # FX col parsing
    try:
        fx_colname = _safe_col(df_norm, "FX")
        fx_raw = df[fx_colname].astype(str).str.strip()
        fx_rate = pd.to_numeric(fx_raw.str.replace(",", ".", regex=False), errors="coerce")
        fx_ccy = np.where(fx_rate.notna(), "", fx_raw.str.upper())
        fx_ccy = pd.Series(fx_ccy).replace({"NAN": "", "": ""})
        df["__fx_rate"] = fx_rate
        df["__fx_ccy"] = fx_ccy.astype(str).str.upper().str.strip()
    except Exception:
        df["__fx_rate"] = np.nan
        df["__fx_ccy"] = ""
        # If broker left FX empty, assume EUR
        df["__fx_ccy"] = df["__fx_ccy"].mask(df["__fx_ccy"].eq(""), "EUR")

    order_ids = df[orderid_col].astype(object)
    order_ids = order_ids.where(~pd.isna(order_ids), "")
    order_ids = order_ids.astype(str)
    
    # Extract Currency column from input if present (for dividends)
    input_currency = pd.Series(np.nan, index=df.index, dtype="object")


    base = pd.DataFrame(
        {
            "Date": df["__dt"],
            "__name__": "base",
            "__minute_key": df["__minute_key"],
            "Product": df[product_col],
            "ISIN": df[isin_col],
            "Description": df[descr_col],
            "Change": _to_numeric_flexible(df[change_col]),
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
            "__Broker": df.get("__Broker", "UNKNOWN"),
            "__SourceFile": df.get("__SourceFile", "uploads"),
            "__InputCurrency": input_currency,
        }
    )

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
            base["Order ID"].astype(str).str.len() > 0, base["Order ID"].astype(str), base["__minute_key"].astype(str)
        )
        key_series = pd.Series(key_series, index=base.index)

        # Extract quantity from any sibling "delisting: sell <N>" description
        delist_qty = base["Description"].str.extract(r"delisting.*sell\s+(\d+)", flags=re.I)[0]
        delist_qty = pd.to_numeric(delist_qty, errors="coerce")

        for idx in base.index[cash_delist_mask]:
            isin = base.at[idx, "ISIN"]
            key = key_series.at[idx]

            sib_candidates = base.index[(base["ISIN"] == isin) & (key_series == key) & delist_qty.notna()]
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
            _mmf_name_mask
            | _desc.str.contains("money market fund price change", na=False)
            | base["ISIN"].isin(MMF_ISINS)
        )

        if _mmf_mask.any():
            base = base.loc[~_mmf_mask].copy()

    # --- Apply inferred ISIN roll-forward mappings from paired split rows ---
    # (Requires Patch A which builds __isin_roll_map after detecting paired split Buy/Sell rows)
    if "__isin_roll_map" in locals() and __isin_roll_map:
        # 1) Normalize base ISINs: replace any remaining OLD -> NEW
        for old_isin, new_isin in __isin_roll_map.items():
            mask_all = base["ISIN"].astype(str).eq(old_isin)
            if mask_all.any():
                base.loc[mask_all, "ISIN"] = new_isin

        # 2) Normalize split_events so adjustments hit the right ISIN after the rewrite
        split_events = [(__isin_roll_map.get(isin, isin), dt, f) for (isin, dt, f) in split_events]

    # Apply split factors to historical BUY lots that occurred strictly BEFORE the split time.
    # We alter source fields used for Quantity_signed and Price (both in native and EUR), but keep total EUR the same.
    if split_events:
        for isin, split_dt, factor in sorted(split_events, key=lambda x: (x[0], x[1])):
            pre_mask = (
                base["ISIN"].astype(str).eq(isin) & base["__Type"].eq("Buy") & pd.to_datetime(base["Date"]).lt(split_dt)
            )
            if not pre_mask.any():
                continue

            idx = base.index[pre_mask]
            qty_before = base.loc[idx, "__qty_desc"].copy()
            price_before = base.loc[idx, "__price_desc"].copy()

            # Apply scaling
            base.loc[idx, "__qty_desc"] = base.loc[idx, "__qty_desc"] * factor
            base.loc[idx, "__price_desc"] = base.loc[idx, "__price_desc"] / factor

            # Log audit rows
            for i in idx:
                split_audit.append(
                    {
                        "ISIN": isin,
                        "Product": str(base.at[i, "Product"]),
                        "Row kind": "Buy (pre-split)",
                        "Trade date": pd.to_datetime(base.at[i, "Date"]),
                        "Order ID": str(base.at[i, "Order ID"]),
                        "Split date": pd.to_datetime(split_dt),
                        "Factor": float(factor),
                        "Qty (before)": float(qty_before.get(i) if pd.notna(qty_before.get(i)) else np.nan),
                        "Qty (after)": float(
                            base.at[i, "__qty_desc"] if pd.notna(base.at[i, "__qty_desc"]) else np.nan
                        ),
                        "Unit px (before)": float(price_before.get(i) if pd.notna(price_before.get(i)) else np.nan),
                        "Unit px (after)": float(
                            base.at[i, "__price_desc"] if pd.notna(base.at[i, "__price_desc"]) else np.nan
                        ),
                    }
                )

    base["__rowid"] = np.arange(len(base))

    # ---- Split multiple Buy/Sell rows under the same Order ID into separate effective IDs ----
    base["__trade_seq"] = np.nan

    # We'll sequence only the non-FX Buy/Sell rows, per (ISIN, Order ID, Type) in time order
    mask_trade_rows = (~(base["__is_fx_credit"] | base["__is_fx_debit"])) & base["__Type"].isin(["Buy", "Sell"])

    # Stable sort so cumcount is deterministic. Use 'Date' (exists in base), not '__dt'.
    trade_sorted = base.loc[mask_trade_rows].sort_values(by=["ISIN", "Order ID", "Date", "Description"], kind="mergesort")

    seq = trade_sorted.groupby(["ISIN", "Order ID", "__Type"], dropna=False).cumcount()

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

    grouped, split_audit = apply_corporate_actions_and_map_fx(
        base=base,
        opening_lots=opening_lots,
        infer_asset_fn=infer_asset,
        direct_eur_from_rate_fn=_direct_eur_from_rate,
        detect_isin_product_mappings_fn=_detect_isin_product_mappings,
        parse_split_factor_fn=parse_split_factor,
    )

    consolidated = consolidate_fifo(grouped)

    out = build_out_table(consolidated)
    split_audit_df = pd.DataFrame(split_audit, columns=AUDIT_COLS)
    # Clean audit rows + enforce dtypes for Arrow/Streamlit
    if not split_audit_df.empty:
        # Drop any header-like rows accidentally appended as data
        headerish = split_audit_df["ISIN"].astype(str).str.upper().eq("ISIN") | split_audit_df["Product"].astype(
            str
        ).str.upper().eq("PRODUCT")
        split_audit_df = split_audit_df.loc[~headerish].copy()

        # Enforce proper dtypes
        for col in ["Split date", "Trade date"]:
            split_audit_df[col] = pd.to_datetime(split_audit_df[col], errors="coerce")

        split_audit_df["Factor"] = pd.to_numeric(split_audit_df["Factor"], errors="coerce")

        for col in [
            "Qty (before)",
            "Qty (after)",
            "Unit px (before)",
            "Unit px (after)",
            "Unit px EUR (before)",
            "Unit px EUR (after)",
        ]:
            if col in split_audit_df.columns:
                split_audit_df[col] = pd.to_numeric(split_audit_df[col], errors="coerce")

        split_audit_df = split_audit_df.sort_values(by=["ISIN", "Split date", "Row kind", "Trade date"], kind="mergesort")

    return out, split_audit_df
