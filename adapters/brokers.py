from __future__ import annotations

from typing import Callable, Dict, List

import pandas as pd
import re


BrokerAdapter = Callable[[pd.DataFrame], pd.DataFrame]


def _safe_stringify_series(series: pd.Series) -> pd.Series:
    return series.apply(lambda value: value if isinstance(value, str) else "" if pd.isna(value) else str(value))


def parse_degiros_csv(df_raw: pd.DataFrame) -> pd.DataFrame:
    # DEGIRO's CSV has a weird structure for dividends:
    # Columns: Date, Time, Value date, Product, ISIN, Description, FX, Change, Unnamed: 8, Balance, Unnamed: 10, Order ID
    # For dividends:
    #   Change column = currency code (USD/GBP/etc)
    #   Unnamed: 8 = dividend amount in that currency
    #   Unnamed: 10 = EUR equivalent amount

    df_norm = _canonicalize_headers(df_raw)
    _validate_required_columns(df_norm)

    description_series = _safe_stringify_series(df_norm["Description"])
    is_div_before = description_series.str.contains("Dividend", case=False, na=False)

    if is_div_before.any():
        dividend_amounts_col = None
        dividend_currencies_col = None
        for candidate in ["Unnamed: 8", "Unnamed_8", "Unnamed 8"]:
            if candidate in df_raw.columns:
                dividend_amounts_col = candidate
                break
        if dividend_amounts_col is None:
            dividend_amounts_col = df_raw.columns[8] if len(df_raw.columns) > 8 else None
        for candidate in ["Change", "Mutatie", "Amount"]:
            if candidate in df_raw.columns:
                dividend_currencies_col = candidate
                break
        if dividend_currencies_col is None:
            dividend_currencies_col = df_raw.columns[7] if len(df_raw.columns) > 7 else None

        if dividend_amounts_col is not None and dividend_currencies_col is not None:
            dividend_amounts = df_raw[dividend_amounts_col].copy()
            dividend_currencies = df_raw[dividend_currencies_col].copy()
        else:
            dividend_amounts = None
            dividend_currencies = None
    else:
        dividend_amounts = None
        dividend_currencies = None

    if "Date" in df_norm.columns:
        df_norm["Date"] = pd.to_datetime(df_norm["Date"], errors="coerce", dayfirst=True)

    if dividend_amounts is not None:
        description_for_dividend = _safe_stringify_series(df_norm["Description"])
        is_dividend = description_for_dividend.str.contains("Dividend", case=False, na=False)
        if is_dividend.any():
            df_norm.loc[is_dividend, "Change"] = pd.to_numeric(dividend_amounts[is_dividend.values], errors="coerce")
            if "Currency" in df_norm.columns and pd.api.types.is_numeric_dtype(df_norm["Currency"]):
                df_norm["Currency"] = df_norm["Currency"].astype(object)
            if "FX" in df_norm.columns and pd.api.types.is_numeric_dtype(df_norm["FX"]):
                df_norm["FX"] = df_norm["FX"].astype(object)
            df_norm.loc[is_dividend, "Currency"] = _safe_stringify_series(
                dividend_currencies[is_dividend.values]
            ).str.upper()
            df_norm.loc[is_dividend, "FX"] = df_norm.loc[is_dividend, "Currency"]

    if "Order ID" in df_norm.columns:
        df_norm["Order ID"] = df_norm["Order ID"].astype(object)
    else:
        df_norm["Order ID"] = None

    for col in ["Product", "ISIN", "Description", "FX", "Change", "Balance", "Currency"]:
        if col in df_norm.columns:
            df_norm[col] = df_norm[col].astype(object)

    return df_norm


def detect_broker_from_headers(df_head: pd.DataFrame) -> str:
    cols = {str(c).strip().lower() for c in df_head.columns}

    degiro_signals = {"date", "product", "isin", "description", "change"}
    if degiro_signals <= cols:
        return "DEGIRO"

    t212_signals = {
        "action",
        "time",
        "isin",
        "ticker",
        "name",
        "no. of shares",
        "price / share",
        "currency (price / share)",
        "exchange rate",
        "total",
        "currency (total)",
        "withholding tax",
    }
    if len(t212_signals & cols) >= 3:
        return "TRADING212"

    return "DEGIRO"


HEADER_ALIASES: Dict[str, List[str]] = {
    "Date": ["Date", "Datum"],
    "Time": ["Time", "Tijd"],
    "Value date": ["Value date", "Valuta datum", "Valutadatum", "Value-date", "Value_date"],
    "Product": ["Product"],
    "ISIN": ["ISIN"],
    "Description": ["Description", "Omschrijving"],
    "FX": ["FX", "Exchange rate"],
    "Change": ["Change", "Mutatie", "Amount"],
    "Balance": ["Balance", "Cash Movements", "Cash movements", "Cash"],
    "Order ID": ["Order ID", "Order Id", "OrderId", "Order", "Order-ID", "Order Id"],
    "Currency": ["Currency", "Valuta"],
}

REQUIRED_ALL = ["Date", "Product", "ISIN", "Description", "Change"]
REQUIRED_ONE_OF = [["Balance"]]


def _norm(s: str) -> str:
    return re.sub(r"[^\w]+", " ", str(s).strip().lower())


def _canonicalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    rev: Dict[str, str] = {}
    for canon, aliases in HEADER_ALIASES.items():
        for a in aliases:
            rev[_norm(a)] = canon
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
            missing.append(f"{c} (aliases: {', '.join(HEADER_ALIASES.get(c, [c]))})")
    for group in REQUIRED_ONE_OF:
        if not any(c in cols for c in group):
            pretty = " or ".join(group)
            missing.append(pretty + f" (aliases: {', '.join(sum((HEADER_ALIASES.get(c, [c]) for c in group), []))})")
    if missing:
        raise ValueError("Missing required columns: " + "; ".join(missing))


def parse_trading212_csv(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    def _num(x):
        s = pd.Series([x]).astype(str)
        s = s.str.replace(r"[,\s€$£]", "", regex=True)
        s = s.str.replace(r"^\((.*)\)$", r"-\1", regex=True)
        s = s.str.replace(",", ".", regex=False)
        return pd.to_numeric(s, errors="coerce").iloc[0]

    def col(name: str, alts: list[str] = []):
        for c in [name] + alts:
            if c in df.columns:
                return c
        lower_map = {c.lower(): c for c in df.columns}
        for c in [name] + alts:
            if c.lower() in lower_map:
                return lower_map[c.lower()]
        want = [name] + alts
        for w in want:
            for colname in df.columns:
                if colname.lower().startswith(w.lower()):
                    return colname
        return None

    c_action = col("Action")
    c_time = col("Time", alts=["Date"])
    c_isin = col("ISIN")
    c_ticker = col("Ticker")
    c_name = col("Name")
    c_id = col("ID")
    c_qty = col("No. of shares", alts=["Quantity", "No. of Shares"])
    c_px = col("Price / share", alts=["Price"])
    c_px_ccy = col("Currency (Price / share)", alts=["Currency"])
    c_exrate = col("Exchange rate", alts=["FX rate"])
    c_total = col("Total", alts=["Total (EUR)", "Total (GBP)", "Total (USD)"])
    c_total_ccy = col("Currency (Total)")
    c_wht = col("Withholding tax", alts=["Withholding Tax", "Dividend Tax", "Tax"])
    c_ccyfee = col("Currency conversion fee", alts=["FX fee", "Currency Conversion Fee"])

    rows = []

    for _, r in df.iterrows():
        action = str(r.get(c_action, "")).lower()
        time_s = str(r.get(c_time, "")).strip()
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
            total_ccy = str(r.get(c_total_ccy, "")).upper().strip() if c_total_ccy else ""
            if total_ccy in ["NAN", "NONE", ""]:
                total_ccy = ""
            currency = total_ccy if total_ccy else (px_ccy if px_ccy and px_ccy not in ["NAN", "NONE"] else "EUR")
            fx_field = "EUR" if currency == "EUR" else currency
            if not oid:
                oid = f"DIV-{isin}-{dt.strftime('%Y%m%d') if pd.notna(dt) else 'NA'}"
            rows.append(
                {
                    "Date": dt,
                    "Time": time_s,
                    "Value date": None,
                    "Product": product,
                    "ISIN": isin,
                    "Description": desc,
                    "FX": fx_field,
                    "Change": change,
                    "Cash Movements": cash_eur,
                    "Balance": None,
                    "Order ID": oid,
                    "Currency": currency,
                }
            )
            if pd.notna(wht_eur) and abs(wht_eur) > 0:
                rows.append(
                    {
                        "Date": dt,
                        "Time": time_s,
                        "Value date": None,
                        "Product": product,
                        "ISIN": isin,
                        "Description": "Dividend Tax",
                        "FX": fx_field,
                        "Change": abs(wht_eur),
                        "Cash Movements": None,
                        "Balance": None,
                        "Order ID": f"TAX-{isin}-{dt.strftime('%Y%m%d') if pd.notna(dt) else 'NA'}",
                        "Currency": currency,
                    }
                )
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

        if pd.notna(fee_eur) and abs(fee_eur) > 0:
            rows.append(
                {
                    "Date": dt,
                    "Time": time_s,
                    "Value date": None,
                    "Product": product,
                    "ISIN": isin,
                    "Description": "Fee: Currency conversion fee",
                    "FX": "EUR",
                    "Change": -abs(fee_eur),
                    "Cash Movements": -abs(fee_eur),
                    "Balance": None,
                    "Order ID": f"FEE-{oid or isin}-{dt.strftime('%Y%m%d') if pd.notna(dt) else 'NA'}",
                    "Currency": "EUR",
                }
            )

        rows.append(
            {
                "Date": dt,
                "Time": time_s,
                "Value date": None,
                "Product": product,
                "ISIN": isin,
                "Description": desc,
                "FX": fx_field,
                "Change": change,
                "Cash Movements": cash_eur,
                "Balance": None,
                "Order ID": oid,
                "Currency": currency,
            }
        )

    out = pd.DataFrame(
        rows,
        columns=[
            "Date",
            "Time",
            "Value date",
            "Product",
            "ISIN",
            "Description",
            "FX",
            "Change",
            "Cash Movements",
            "Balance",
            "Order ID",
            "Currency",
        ],
    )
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce", utc=False, dayfirst=False)

    for colname in ["Balance", "Value date", "Currency"]:
        if colname not in out.columns:
            out[colname] = None

    out = out[
        [
            "Date",
            "Time",
            "Value date",
            "Product",
            "ISIN",
            "Description",
            "FX",
            "Change",
            "Cash Movements",
            "Balance",
            "Order ID",
            "Currency",
        ]
    ]
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce", dayfirst=False)
    return out


BROKER_ADAPTERS: Dict[str, BrokerAdapter] = {"DEGIRO": parse_degiros_csv, "TRADING212": parse_trading212_csv}
