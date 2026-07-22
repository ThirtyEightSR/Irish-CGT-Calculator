from __future__ import annotations

import re
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd

SPLIT_RE_LIST = [
    re.compile(r"(\d+(?:\.\d+)?)\s*for\s*(\d+(?:\.\d+)?)", re.I),
    re.compile(r"(\d+(?:\.\d+)?)[x×]\s*split", re.I),
    re.compile(r"split\s*ratio\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*[:/]\s*(\d+(?:\.\d+)?)", re.I),
    re.compile(r"(\d+)\s*:\s*(\d+)", re.I),
]

ISIN_RE = re.compile(r"\b[A-Z]{2}[A-Z0-9]{9}\d\b")


TRADE_RE = re.compile(
    r"(?P<type>Buy|Sell)\s+(?P<qty>\d+(?:[\.,]\d+)?)\s+.*?(?:@|at)\s*"
    r"(?P<price>\d+(?:[\.,]\d+)?)\s*(?P<ccy>[A-Z]{3})?\b",
    re.IGNORECASE,
)


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


def detect_isin_product_mappings(base: pd.DataFrame):
    isin_maps, prod_maps = [], []

    ca = base[base["__Type"].isin(["ISIN change", "Product change"])].copy()
    if ca.empty:
        return isin_maps, prod_maps

    for _, r in ca.iterrows():
        dt = pd.to_datetime(r["Date"])
        desc = str(r.get("Description", "") or "")
        row_isin = str(r.get("ISIN", "") or "")
        row_prod = str(r.get("Product", "") or "")

        if r["__Type"] == "ISIN change":
            isins = ISIN_RE.findall(desc)
            if len(isins) >= 2:
                old_isin, new_isin = isins[0], isins[1]
                isin_maps.append((old_isin, new_isin, dt))
            elif len(isins) == 1 and isins[0] != row_isin:
                isin_maps.append((isins[0], row_isin, dt))

        if r["__Type"] == "Product change":
            if row_prod and row_isin:
                prod_maps.append((row_isin, row_prod, dt))

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
                a = float(m.group(1))
                b = float(m.group(2)) if m.lastindex and m.lastindex >= 2 else 1.0
                if a > 0 and b > 0:
                    return a / b
            except Exception:
                continue
    return None


def safe_col(df: pd.DataFrame, name: str) -> str:
    for c in df.columns:
        if c.strip().lower() == name.strip().lower():
            return c
    raise KeyError(f"Column '{name}' not found. Found: {list(df.columns)}")


def find_between(df: pd.DataFrame, left: str, right: str) -> Optional[str]:
    cols = list(df.columns)
    try:
        li = cols.index(left)
        ri = cols.index(right)
        if ri - li == 2:
            return cols[li + 1]
        if "Cash Movements" in df.columns:
            return "Cash Movements"
    except ValueError:
        if "Cash Movements" in df.columns:
            return "Cash Movements"
    return None


def to_numeric_flexible(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan, "<NA>": np.nan})

    def _norm(x: object) -> object:
        if pd.isna(x):
            return np.nan
        v = str(x).strip().replace(" ", "")
        v = re.sub(r"[€$£]", "", v)
        v = re.sub(r"^\((.*)\)$", r"-\1", v)
        if not v:
            return np.nan

        has_comma = "," in v
        has_dot = "." in v

        if has_comma and has_dot:
            if v.rfind(",") > v.rfind("."):
                v = v.replace(".", "").replace(",", ".")
            else:
                v = v.replace(",", "")
        elif has_comma:
            parts = v.split(",")
            if len(parts) == 2 and len(parts[1]) in (1, 2):
                v = v.replace(",", ".")
            elif len(parts) == 2 and len(parts[1]) == 3 and parts[0].lstrip("-").isdigit():
                v = v.replace(",", "")
            else:
                v = v.replace(",", "")

        return v

    out = pd.to_numeric(s.map(_norm), errors="coerce")
    return pd.Series(out, index=series.index)


def parse_desc_numbers(desc) -> Tuple[Optional[float], Optional[float], Optional[str]]:
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
            s2 = s.replace(",", "")
            try:
                return float(s2)
            except Exception:
                return None

        if has_com and not has_dot:
            parts = s.split(",")
            if len(parts) == 2 and len(parts[1]) == 3 and parts[0].isdigit() and parts[1].isdigit():
                try:
                    return float(parts[0] + parts[1])
                except Exception:
                    return None
            try:
                return float(s.replace(",", "."))
            except Exception:
                return None

        try:
            return float(s)
        except Exception:
            return None

    qty_str = m.group("qty")
    price_str = m.group("price")
    ccy = m.group("ccy").upper() if m.group("ccy") else None

    qty = _to_float(qty_str)
    px = _to_float(price_str)

    return qty, px, ccy


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


def parse_type(desc: Any) -> str:
    if not isinstance(desc, str):
        return "Other"
    d = desc.lower().strip()

    if "opening lot" in d:
        return "Buy"
    if "money market fund price change" in d:
        return "MMF price change"

    if "stock split" in d or d.startswith("stock split:"):
        return "Stock split"
    if "product change" in d or d.startswith("product change:"):
        return "Product change"
    if "isin change" in d or d.startswith("isin change:"):
        return "ISIN change"
    if "spin-off" in d or "spinoff" in d:
        return "Spin-off"
    if "merger" in d or "merged" in d:
        return "Merger"
    if re.search(r"\breturn of capital\b", d) or "capital repayment" in d:
        return "Return of capital"
    if "scrip dividend" in d or "scrip" in d:
        return "Scrip dividend"
    if "delisting" in d or "corporate action cash settlement" in d:
        return "Sell"
    if "spin-off" in d or "spinoff" in d or "spin off" in d or "demerger" in d:
        return "Spin-off"
    if "merger" in d or "merged into" in d or "merging into" in d:
        return "Merger"
    if ("scrip" in d and "dividend" in d) or "stock dividend" in d:
        return "Scrip dividend"
    if "dividend tax" in d or "withholding tax" in d:
        return "Dividend Tax"
    if "dividend" in d:
        return "Dividend"
    if "buy" in d or "purchase" in d:
        return "Buy"
    if "sell" in d or "sale" in d:
        return "Sell"
    if d.startswith("fx credit"):
        return "FX Credit"
    if d.startswith("fx debit"):
        return "FX Debit"
    if "coupon" in d:
        return "Coupon"
    if "interest" in d:
        return "Interest"
    if "fee" in d or "commission" in d or "transaction" in d or "exchange" in d:
        return "Fee"

    return "Other"


def direct_eur_from_rate(row: pd.Series) -> float:
    try:
        if row["Type"] not in ("Buy", "Sell"):
            return np.nan
        rate = pd.to_numeric(pd.Series([row.get("FX_Rate")]), errors="coerce").iloc[0]
        if pd.isna(rate) or float(rate) <= 0 or abs(float(rate) - 1.0) < 1e-9:
            return np.nan
        qty = pd.to_numeric(pd.Series([row.get("__qty_desc")]), errors="coerce").iloc[0]
        px = pd.to_numeric(pd.Series([row.get("__price_desc")]), errors="coerce").iloc[0]
        if pd.isna(qty) or pd.isna(px):
            return np.nan
        eur = float(qty) * float(px) * float(rate)
        return -eur if row["Type"] == "Buy" else eur
    except Exception:
        return np.nan


def parse_fx_cell(val: object) -> tuple[float | None, str]:
    s = str(val).strip()
    if not s or s.lower() == "nan":
        return (None, "")
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


def compute_eur_cash_from_fx(change_series: pd.Series, fx_series: pd.Series) -> pd.Series:
    rates, hints = zip(*fx_series.map(parse_fx_cell).tolist())
    rates = pd.Series(rates, index=fx_series.index, dtype="float64")
    hints = pd.Series(hints, index=fx_series.index, dtype="object")

    eur_mask = hints.eq("EUR")
    out = pd.Series(np.nan, index=fx_series.index, dtype="float64")
    out.loc[eur_mask] = pd.to_numeric(change_series, errors="coerce").loc[eur_mask]

    rate_mask = rates.notna()
    chg = pd.to_numeric(change_series, errors="coerce")
    out.loc[rate_mask] = chg.loc[rate_mask] / rates.loc[rate_mask]

    return out
