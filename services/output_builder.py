from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def consolidate_fifo(grouped: pd.DataFrame) -> pd.DataFrame:
    # Defensive: ensure __row_id exists
    if "__row_id" not in grouped.columns:
        grouped = grouped.copy()
        grouped["__row_id"] = np.arange(len(grouped))

    consolidated = grouped.sort_values(
        ["ISIN", "Date", "Order ID", "__type_sort", "__row_id"], kind="mergesort"
    ).reset_index(drop=True)

    # --- FIFO Gain/Loss (EUR) for sells using a running lot ledger (per ISIN) ---
    # Uses fee-adjusted proceeds and fee-adjusted buy cost
    gl = np.full(len(consolidated), np.nan)  # index-aligned result array

    # Work instrument-by-instrument in chronological order
    for _, idx in consolidated.groupby("ISIN", sort=False).groups.items():
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


def build_out_table(consolidated: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "Date": consolidated["Date"],
            "Ticker - Name": consolidated["Product"],
            "ISIN": consolidated["ISIN"],
            "Order ID": consolidated["Order ID"],
            "Type": consolidated["Type"],
            "Asset": consolidated["Asset"],
            "Currency": consolidated["Currency"],
            "FXCCY": consolidated.get("FXCCY"),
            "FX_Rate": consolidated.get("FX_Rate"),
            "Quantity": np.where(
                consolidated["Type"].isin(["Buy", "Sell"]), consolidated["Quantity_signed"].abs(), np.nan
            ),
            "Price": np.where(consolidated["Type"].isin(["Buy", "Sell"]), consolidated["Price"], np.nan),
            "Fee": consolidated["Fee_signed"].abs(),
            "Total": np.where(
                consolidated["Type"].isin(["Buy", "Sell"]),
                consolidated["Total_signed"].abs(),
                consolidated["Total_signed"],
            ),
            "Total (EUR)": consolidated["Total_EUR"],
            "Total (EUR, fee-adj)": consolidated.get("Total_EUR_FeeAdj"),
            "Gain/Loss": consolidated["Gain/Loss"],
            "Description": consolidated["Description"],
            "__year": pd.to_datetime(consolidated["Date"]).dt.year,
        }
    )

    for _c in ["__Broker", "__SourceFile", "__row_id"]:
        if _c not in out.columns and _c in consolidated.columns:
            out[_c] = consolidated[_c]

    for col in ["Fee", "Total", "Total (EUR)", "Total (EUR, fee-adj)", "Gain/Loss"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    # --- Ensure Dividends carry through Gross (Total) and Tax (Fee) cleanly ---
    div_mask = consolidated["Type"].eq("Dividend")
    if div_mask.any():
        gross = pd.to_numeric(consolidated.loc[div_mask, "Total_signed"], errors="coerce").abs()
        tax = pd.to_numeric(consolidated.loc[div_mask, "Fee_signed"], errors="coerce").abs()

        # Push into out (same row order)
        out.loc[div_mask.values, "Total"] = gross.values
        out.loc[div_mask.values, "Fee"] = tax.values

    return out


def apply_corporate_actions_and_map_fx(
    base: pd.DataFrame,
    opening_lots: Optional[pd.DataFrame],
    infer_asset_fn: Callable[[object, object], str],
    direct_eur_from_rate_fn: Callable[[pd.Series], float],
    detect_isin_product_mappings_fn: Callable[[pd.DataFrame], tuple[list[tuple[str, str, pd.Timestamp]], list[tuple[str, str, pd.Timestamp]]]],
    parse_split_factor_fn: Callable[[str], Optional[float]],
) -> Tuple[pd.DataFrame, List[Dict]]:
    non_fx = base[~(base["__is_fx_credit"] | base["__is_fx_debit"])].copy()

    def _agg_type(series: pd.Series) -> str:
        priorities = [
            "Buy",
            "Sell",
            "Dividend",
            "Dividend Tax",
            "Stock split",
            "Product change",
            "ISIN change",
            "Coupon",
            "Interest",
            "Other",
            "Fee",
        ]
        types = [x for x in series.dropna().tolist() if isinstance(x, str)]
        for tpe in priorities:
            if tpe in types:
                return tpe
        return types[0] if types else "Other"

    grouped = non_fx.groupby(["ISIN", "__EffID"], dropna=False, as_index=False).agg(
        {
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
            "__InputCurrency": lambda s: next(
                (
                    v
                    for v in s
                    if pd.notna(v) and str(v).strip() and str(v).upper() not in ["NAN", "NONE", ""]
                ),
                s.iloc[0] if len(s) > 0 else None,
            ),
            "__Broker": "first",
            "__SourceFile": "first",
            "__row_id": "first",
        }
    )
    grouped["Type"] = grouped["__Type"]
    grouped["TradeCCY"] = grouped["__ccy_desc"].astype(str).str.upper().replace({"NONE": "", "NAN": "", "": np.nan})

    if "__InputCurrency" in grouped.columns:
        input_ccy = grouped["__InputCurrency"].astype(str).str.upper().str.strip()
        input_ccy = input_ccy.where(~input_ccy.isin(["NAN", "NONE", ""]), np.nan)
        grouped["TradeCCY"] = grouped["TradeCCY"].where(grouped["TradeCCY"].notna(), input_ccy)

    grouped["TradeCCY"] = grouped["TradeCCY"].fillna("EUR")

    def _qty_signed(row):
        if row["Type"] in ("Buy", "Sell"):
            q = row["__qty_desc"]
            if not pd.isna(q):
                return -abs(q) if row["Type"] == "Sell" else abs(q)
            return 0.0
        return 0.0

    grouped["Quantity_signed"] = grouped.apply(_qty_signed, axis=1).astype(float)

    fees_only = non_fx[non_fx["__is_fee"]]
    if not fees_only.empty:
        fees_grouped = fees_only.groupby(["ISIN", "__EffID"], dropna=False, as_index=False)["_CashValue"].sum()
        fees_grouped.rename(columns={"_CashValue": "Fee_signed"}, inplace=True)
        grouped = grouped.merge(fees_grouped, on=["ISIN", "__EffID"], how="left")
    if "Fee_signed" not in grouped.columns:
        grouped["Fee_signed"] = 0.0
    grouped["Fee_signed"] = grouped["Fee_signed"].fillna(0.0)

    def _sign_for_type(tpe: str) -> int:
        if tpe == "Buy":
            return -1
        if tpe == "Sell":
            return +1
        return 0

    sign = grouped["Type"].map(_sign_for_type).fillna(0)
    grouped["_CashExFees"] = grouped["_CashValue"] - (sign * grouped["Fee_signed"])
    grouped["Price_from_desc"] = grouped["__price_desc"]
    grouped["Price_calc"] = np.where(
        grouped["Quantity_signed"].abs() > 0, (grouped["_CashExFees"].abs() / grouped["Quantity_signed"].abs()), np.nan
    )
    grouped["Price"] = grouped["Price_from_desc"].where(~grouped["Price_from_desc"].isna(), grouped["Price_calc"])

    fees_oid = (
        non_fx.loc[non_fx["__is_fee"], ["ISIN", "Order ID", "_CashValue"]]
        .dropna(subset=["Order ID"])
        .groupby(["ISIN", "Order ID"], dropna=False)["_CashValue"]
        .sum()
        .rename("__Fee_by_oid")
        .reset_index()
    )

    trades_only = grouped[grouped["Type"].isin(["Buy", "Sell"])].copy()
    order_cash = (
        trades_only.groupby(["ISIN", "Order ID"], dropna=False)["_CashValue"]
        .apply(lambda s: s.abs().sum())
        .rename("__OrderAbsCash")
        .reset_index()
    )

    grouped = grouped.merge(order_cash, on=["ISIN", "Order ID"], how="left")
    grouped = grouped.merge(fees_oid, on=["ISIN", "Order ID"], how="left")

    fee_prorata = np.where(
        grouped["Type"].isin(["Buy", "Sell"]) & grouped["__OrderAbsCash"].gt(0) & grouped["__Fee_by_oid"].notna(),
        grouped["__Fee_by_oid"] * (grouped["_CashValue"].abs() / grouped["__OrderAbsCash"]),
        0.0,
    ).astype(float)

    grouped["Fee_signed"] = grouped.get("Fee_signed", 0.0).fillna(0.0) + fee_prorata

    if not fees_oid.empty:
        # Once order-level fees are allocated onto matching buy/sell rows,
        # the original standalone fee rows are zeroed to avoid double counting.
        fee_rows_mask = grouped["Type"].eq("Fee") & grouped["Order ID"].isin(fees_oid["Order ID"])
        grouped.loc[fee_rows_mask, ["Fee_signed", "_CashValue", "Total_signed"]] = 0.0

    if "Total_signed" not in grouped.columns:
        grouped["Total_signed"] = np.nan

    is_div = grouped["Type"].eq("Dividend")
    is_div_tax = grouped["Type"].eq("Dividend Tax")

    grouped.loc[is_div, "Total_signed"] = grouped.loc[is_div, "Change"]
    missing_gross = is_div & (grouped["Total_signed"].isna() | (grouped["Total_signed"].abs() < 1e-12))
    grouped.loc[missing_gross, "Total_signed"] = grouped.loc[missing_gross, "_CashValue"].abs()

    tax_from_change = grouped.loc[is_div_tax, "Change"].abs()
    tax_from_cash = grouped.loc[is_div_tax, "_CashValue"].abs()
    grouped.loc[is_div_tax, "Fee_signed"] = np.where(
        tax_from_change.fillna(0) > 0, tax_from_change, tax_from_cash.fillna(0.0)
    )
    grouped.loc[is_div_tax, "Total_signed"] = np.nan

    mask_other = ~(is_div | is_div_tax)
    grouped.loc[mask_other & grouped["Total_signed"].isna(), "Total_signed"] = grouped.loc[mask_other, "_CashValue"]

    grouped["Asset"] = grouped.apply(lambda r: infer_asset_fn(r["Product"], r["Description"]), axis=1)

    fx = base[(base["__is_fx_credit"] | base["__is_fx_debit"])].copy()
    fx["__amt"] = pd.to_numeric(fx["Change"].where(fx["Change"].notna(), fx["_CashValue"]), errors="coerce")

    fx_pos = fx[fx["__amt"] > 0].copy()
    fx_neg = fx[fx["__amt"] < 0].copy()

    eur_credit_by_oid = (
        fx_pos.groupby(["ISIN", "Order ID"], dropna=False)["__amt"]
        .sum()
        .abs()
        .rename("__EUR_credit_by_oid")
        .reset_index()
    )
    eur_debit_by_oid = (
        fx_neg.groupby(["ISIN", "Order ID"], dropna=False)["__amt"]
        .sum()
        .abs()
        .rename("__EUR_debit_by_oid")
        .reset_index()
    )

    eur_credit_by_min = (
        fx_pos.groupby(["ISIN", "__minute_key"], dropna=False)["__amt"]
        .sum()
        .abs()
        .rename("__EUR_credit_by_minute")
        .reset_index()
        .rename(columns={"__minute_key": "__minute_key_fx1"})
    )
    eur_debit_by_min = (
        fx_neg.groupby(["ISIN", "__minute_key"], dropna=False)["__amt"]
        .sum()
        .abs()
        .rename("__EUR_debit_by_minute")
        .reset_index()
        .rename(columns={"__minute_key": "__minute_key_fx2"})
    )

    grouped = grouped.merge(eur_credit_by_oid, on=["ISIN", "Order ID"], how="left")
    grouped = grouped.merge(eur_debit_by_oid, on=["ISIN", "Order ID"], how="left")
    grouped = grouped.merge(
        eur_credit_by_min, left_on=["ISIN", "__minute_key"], right_on=["ISIN", "__minute_key_fx1"], how="left"
    )
    grouped = grouped.merge(
        eur_debit_by_min, left_on=["ISIN", "__minute_key"], right_on=["ISIN", "__minute_key_fx2"], how="left"
    )

    if "__OrderAbsCash" not in grouped.columns:
        trades_tmp = grouped[grouped["Type"].isin(["Buy", "Sell"])].copy()
        order_cash = (
            trades_tmp.groupby(["ISIN", "Order ID"], dropna=False)["_CashValue"]
            .apply(lambda s: s.abs().sum())
            .rename("__OrderAbsCash")
            .reset_index()
        )
        grouped = grouped.merge(order_cash, on=["ISIN", "Order ID"], how="left")

    order_buy_eur = np.where(
        grouped["Type"].eq("Buy") & grouped["__OrderAbsCash"].gt(0) & grouped["__EUR_debit_by_oid"].notna(),
        grouped["__EUR_debit_by_oid"] * (grouped["_CashValue"].abs() / grouped["__OrderAbsCash"]),
        np.nan,
    )
    order_sell_eur = np.where(
        grouped["Type"].eq("Sell") & grouped["__OrderAbsCash"].gt(0) & grouped["__EUR_credit_by_oid"].notna(),
        grouped["__EUR_credit_by_oid"] * (grouped["_CashValue"].abs() / grouped["__OrderAbsCash"]),
        np.nan,
    )

    trades_min = grouped[grouped["Type"].isin(["Buy", "Sell"])].copy()
    min_cash = (
        trades_min.groupby(["ISIN", "__minute_key", "Type"], dropna=False)["_CashValue"]
        .apply(lambda s: s.abs().sum())
        .rename("__MinuteAbsCash")
        .reset_index()
    )
    grouped = grouped.merge(min_cash, on=["ISIN", "__minute_key", "Type"], how="left")

    minute_buy_eur = np.where(
        (
            grouped["Type"].eq("Buy")
            & grouped["__MinuteAbsCash"].gt(0)
            & grouped["__EUR_debit_by_minute"].notna()
            & grouped["Total_EUR"].isna()
            if "Total_EUR" in grouped.columns
            else True
        ),
        grouped["__EUR_debit_by_minute"] * (grouped["_CashValue"].abs() / grouped["__MinuteAbsCash"]),
        np.nan,
    )
    minute_sell_eur = np.where(
        (
            grouped["Type"].eq("Sell")
            & grouped["__MinuteAbsCash"].gt(0)
            & grouped["__EUR_credit_by_minute"].notna()
            & grouped["Total_EUR"].isna()
            if "Total_EUR" in grouped.columns
            else True
        ),
        grouped["__EUR_credit_by_minute"] * (grouped["_CashValue"].abs() / grouped["__MinuteAbsCash"]),
        np.nan,
    )

    direct_rate_eur = grouped.apply(direct_eur_from_rate_fn, axis=1)

    buy_eur_pref = np.where(
        ~pd.isna(order_buy_eur), order_buy_eur, np.where(~pd.isna(direct_rate_eur), direct_rate_eur, minute_buy_eur)
    )
    sell_eur_pref = np.where(
        ~pd.isna(order_sell_eur), order_sell_eur, np.where(~pd.isna(direct_rate_eur), direct_rate_eur, minute_sell_eur)
    )

    total_eur_trade = np.where(
        grouped["Type"].eq("Buy"), buy_eur_pref, np.where(grouped["Type"].eq("Sell"), sell_eur_pref, np.nan)
    )
    grouped["Total_EUR"] = total_eur_trade

    grouped.loc[grouped["Type"].isin(["Dividend", "Dividend Tax"]), "Total_EUR"] = np.nan

    mask_eur_no_fx = (
        grouped["Total_EUR"].isna()
        & ~grouped["Type"].isin(["Dividend", "Dividend Tax"])
        & (grouped["FXCCY"].astype(str).str.upper().str.strip().eq("EUR") | (grouped["FX_Rate"].round(6) == 1.0))
    )
    grouped.loc[mask_eur_no_fx, "Total_EUR"] = grouped.loc[mask_eur_no_fx, "Total_signed"].abs()

    mask_trade_final_fallback = grouped["Total_EUR"].isna() & grouped["Type"].isin(["Buy", "Sell"])
    grouped.loc[mask_trade_final_fallback, "Total_EUR"] = grouped.loc[mask_trade_final_fallback, "Total_signed"].abs()

    is_trade = grouped["Type"].isin(["Buy", "Sell"])
    fee_abs = grouped["Fee_signed"].abs().fillna(0.0)
    grouped["Total_EUR_FeeAdj"] = grouped["Total_EUR"]
    grouped.loc[is_trade, "Total_EUR_FeeAdj"] = grouped.loc[is_trade, "Total_EUR"].fillna(0.0) + np.where(
        grouped.loc[is_trade, "Type"].eq("Buy"), fee_abs.loc[is_trade], -fee_abs.loc[is_trade]
    )

    try:
        is_incoming = grouped["Type"].eq("Buy") & grouped["Description"].astype(str).str.contains(
            "INCOMING TRANSFER", case=False, na=False
        )
        need_fee_adj = is_incoming & grouped["Total_EUR_FeeAdj"].isna() & grouped["Total_EUR"].notna()
        grouped.loc[need_fee_adj, "Total_EUR_FeeAdj"] = grouped.loc[need_fee_adj, "Total_EUR"]
        need_total = is_incoming & grouped["Total_EUR"].isna() & grouped["Total_EUR_FeeAdj"].notna()
        grouped.loc[need_total, "Total_EUR"] = grouped.loc[need_total, "Total_EUR_FeeAdj"]
    except Exception:
        pass

    grouped["Price_EUR"] = np.where(
        grouped["Quantity_signed"].abs() > 0, (grouped["Total_EUR_FeeAdj"] / grouped["Quantity_signed"].abs()), np.nan
    )

    isin_maps, prod_maps = detect_isin_product_mappings_fn(base)
    for old_isin, new_isin, dt in isin_maps:
        pre_mask = (base["ISIN"].astype(str) == old_isin) & (pd.to_datetime(base["Date"]) < pd.to_datetime(dt))
        if pre_mask.any():
            base.loc[pre_mask, "ISIN"] = new_isin
    for isin, new_prod, dt in prod_maps:
        prod_mask = (base["ISIN"].astype(str) == isin) & (pd.to_datetime(base["Date"]) >= pd.to_datetime(dt))
        if prod_mask.any():
            base.loc[prod_mask, "Product"] = new_prod

    split_audit: List[Dict] = []
    split_events = []
    __isin_roll_map = {}

    spmask = grouped["Description"].astype(str).str.contains(r"\bSTOCK SPLIT:", case=False, na=False)
    splits = grouped.loc[spmask, ["Date", "Product", "ISIN", "Description", "Order ID"]].copy()
    if not splits.empty:
        splits["Date"] = pd.to_datetime(splits["Date"], errors="coerce")
        _pat = re.compile(
            r"STOCK\s+SPLIT:\s*(Buy|Sell)\s+([\d.,]+)\s+(.+?)@([\d.,]+)\s+[A-Z]{3}\s+\(([A-Z0-9]{12})\)", re.IGNORECASE
        )

        def _parse_desc(desc: str):
            m = _pat.search(str(desc))
            if not m:
                return pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan], index=["action", "qty", "prod_desc", "unit_px", "desc_isin"])
            action, qty, prod_desc, unit_px, desc_isin = m.groups()
            def _to_f(x):
                x = str(x).replace(",", "").strip()
                try:
                    return float(x)
                except Exception:
                    return np.nan
            return pd.Series([action.capitalize(), _to_f(qty), (prod_desc or "").strip(), _to_f(unit_px), desc_isin], index=["action", "qty", "prod_desc", "unit_px", "desc_isin"])

        splits[["action", "qty", "prod_desc", "unit_px", "desc_isin"]] = splits["Description"].apply(_parse_desc)
        splits["k_time"] = splits["Date"].dt.floor("5min")
        splits["k_name"] = (
            splits["prod_desc"]
            .where(splits["prod_desc"].notna(), splits["Product"])
            .astype(str)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
            .str.lower()
        )
        for (_, _), grp in splits.groupby(["k_time", "k_name"], sort=False):
            a = grp.dropna(subset=["action", "qty", "desc_isin"])
            if a.empty:
                continue
            sells = a[a["action"].eq("Sell")]
            buys = a[a["action"].eq("Buy")]
            if len(sells) == 1 and len(buys) == 1:
                sell = sells.iloc[0]
                buy = buys.iloc[0]
                old_isin = str(sell["desc_isin"] or sell["ISIN"])
                new_isin = str(buy["desc_isin"] or buy["ISIN"])
                sell_qty = float(sell["qty"] or 0.0)
                buy_qty = float(buy["qty"] or 0.0)
                if sell_qty > 0 and buy_qty > 0:
                    factor = buy_qty / sell_qty
                    split_events.append((old_isin, pd.to_datetime(sell["Date"]), float(factor)))
                    if old_isin != new_isin:
                        __isin_roll_map[old_isin] = new_isin
                    split_audit.append([old_isin, sell.get("Product", ""), sell["Date"], factor, "Sell(old)", sell["Date"], sell.get("Order ID", ""), np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
                    split_audit.append([new_isin, buy.get("Product", ""), buy["Date"], factor, "Buy(new)", buy["Date"], buy.get("Order ID", ""), np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])

    if not base.empty:
        _spl = base[base["__Type"].eq("Stock split")].copy()
        for _, r in _spl.iterrows():
            f = parse_split_factor_fn(r.get("Description", ""))
            if f and f > 0 and not np.isclose(f, 1.0):
                split_events.append((str(r["ISIN"]), pd.to_datetime(r["Date"]), float(f)))

    if not base.empty:
        splits_raw = base[base["__Type"].eq("Stock split")].copy()
        if not splits_raw.empty:
            desc_lower = splits_raw["Description"].astype(str).str.lower()
            splits_raw["__is_buy_desc"] = desc_lower.str.contains(r"\bbuy\b", na=False)
            splits_raw["__is_sell_desc"] = desc_lower.str.contains(r"\bsell\b", na=False)
            splits_raw["__day"] = pd.to_datetime(splits_raw["Date"]).dt.floor("D")
            buys = splits_raw[splits_raw["__is_buy_desc"]].groupby(["ISIN", "__day"])["__qty_desc"].sum().rename("buy_qty")
            sells = splits_raw[splits_raw["__is_sell_desc"]].groupby(["ISIN", "__day"])["__qty_desc"].sum().rename("sell_qty")
            agg = pd.concat([buys, sells], axis=1).fillna(0.0).reset_index()
            first_dt = splits_raw.groupby(["ISIN", "__day"])["Date"].min().rename("first_dt").reset_index()
            agg = agg.merge(first_dt, on=["ISIN", "__day"], how="left")
            inferred = []
            for _, r in agg.iterrows():
                bq, sq = float(r["buy_qty"]), float(r["sell_qty"])
                if bq > 0 and sq > 0:
                    f = bq / sq
                    if f > 0 and not np.isclose(f, 1.0):
                        inferred.append((str(r["ISIN"]), pd.to_datetime(r["first_dt"]), float(f)))
            if inferred:
                existing = [(str(i), pd.to_datetime(d), float(f)) for (i, d, f) in split_events]
                def _is_dup(isin_new, dt_new):
                    for isin_old, dt_old, _f in existing:
                        if isin_new == isin_old and abs(dt_new - dt_old) <= pd.Timedelta("1D"):
                            return True
                    return False
                for isin, dt, f in inferred:
                    if not _is_dup(isin, dt):
                        split_events.append((isin, dt, f))

    if split_events:
        is_open = grouped["Order ID"].astype(str).str.startswith("OPENING-")
        for isin, split_dt, factor in sorted(split_events, key=lambda x: (x[0], x[1])):
            mask = is_open & grouped["ISIN"].astype(str).eq(isin) & pd.to_datetime(grouped["Date"]).lt(split_dt)
            if not mask.any():
                continue
            idx = grouped.index[mask]
            q_before = grouped.loc[idx, "Quantity_signed"].copy()
            p_before = grouped.loc[idx, "Price"].copy()
            pe_before = grouped.loc[idx, "Price_EUR"].copy()
            grouped.loc[idx, "Quantity_signed"] = grouped.loc[idx, "Quantity_signed"] * factor
            for col in ["Price", "Price_EUR"]:
                if col in grouped.columns:
                    grouped.loc[idx, col] = grouped.loc[idx, col] / factor
            for i in idx:
                split_audit.append(
                    {
                        "ISIN": str(grouped.at[i, "ISIN"]),
                        "Product": str(grouped.at[i, "Product"]),
                        "Row kind": "Opening lot (pre-split)",
                        "Trade date": pd.to_datetime(grouped.at[i, "Date"]),
                        "Order ID": str(grouped.at[i, "Order ID"]),
                        "Split date": pd.to_datetime(split_dt),
                        "Factor": float(factor),
                        "Qty (before)": float(q_before.get(i) if pd.notna(q_before.get(i)) else np.nan),
                        "Qty (after)": float(grouped.at[i, "Quantity_signed"] if pd.notna(grouped.at[i, "Quantity_signed"]) else np.nan),
                        "Unit px (before)": float(p_before.get(i) if pd.notna(p_before.get(i)) else np.nan),
                        "Unit px (after)": float(grouped.at[i, "Price"] if pd.notna(grouped.at[i, "Price"]) else np.nan),
                        "Unit px EUR (before)": float(pe_before.get(i) if pd.notna(pe_before.get(i)) else np.nan),
                        "Unit px EUR (after)": float(grouped.at[i, "Price_EUR"] if pd.notna(grouped.at[i, "Price_EUR"]) else np.nan),
                    }
                )

    is_corp = grouped["Type"].isin(["Stock split", "Product change", "ISIN change"])
    if is_corp.any():
        grouped.loc[
            is_corp,
            ["Quantity_signed", "Price", "Price_EUR", "Fee_signed", "Total_signed", "Total_EUR", "Total_EUR_FeeAdj"],
        ] = np.nan

    con = grouped.copy()
    is_div2 = con["Type"].eq("Dividend")
    is_tax2 = con["Type"].eq("Dividend Tax")
    if (is_div2 | is_tax2).any():
        combined_rows = []
        for (isin, _mkey, product, broker, source_file), g in con[is_div2 | is_tax2].groupby(
            ["ISIN", "__minute_key", "Product", "__Broker", "__SourceFile"], dropna=False
        ):
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
            ccy_series = g.get("TradeCCY", pd.Series(dtype="object")).dropna().astype(str).str.upper().str.strip()
            ccy_series = ccy_series[~ccy_series.isin(["", "NAN", "NONE"])]
            trade_ccy = ccy_series.iloc[0] if not ccy_series.empty else "EUR"
            combined_rows.append(
                {
                    "Date": date,
                    "__minute_key": pd.to_datetime(date).floor("min"),
                    "Product": product,
                    "ISIN": isin,
                    "Order ID": order_id,
                    "Description": "Dividend",
                    "Change": np.nan,
                    "_CashValue": np.nan,
                    "Type": "Dividend",
                    "Quantity_signed": 0.0,
                    "Fee_signed": tax_amt,
                    "_CashExFees": np.nan,
                    "Price_from_desc": np.nan,
                    "Price_calc": np.nan,
                    "Price": np.nan,
                    "Total_signed": gross_val,
                    "Asset": g["Asset"].iloc[0] if "Asset" in g.columns and len(g["Asset"]) else "Share",
                    "__year": pd.to_datetime(date).year,
                    "Total_EUR": np.nan,
                    "Price_EUR": np.nan,
                    "TradeCCY": trade_ccy,
                    "FXCCY": trade_ccy,
                    "__InputCurrency": trade_ccy,
                    "__Broker": broker if pd.notna(broker) else "UNKNOWN",
                    "__SourceFile": source_file if pd.notna(source_file) else "uploads",
                }
            )
        combined_df = pd.DataFrame(combined_rows)
        con = con[~(is_div2 | is_tax2)]
        con = pd.concat([con, combined_df], ignore_index=True)

    type_sort = {
        "Buy": 0,
        "Sell": 1,
        "Dividend": 2,
        "Stock split": 3,
        "Product change": 3,
        "ISIN change": 3,
        "Coupon": 4,
        "Interest": 4,
        "Fee": 5,
        "Delisting (non-cash)": 6,
        "Other": 7,
    }
    con["__type_sort"] = con["Type"].map(type_sort).fillna(7).astype(int)
    if "__row_id" not in con.columns:
        con["__row_id"] = np.arange(len(con))
    return con, split_audit
