from __future__ import annotations

import pandas as pd


def format_eur(x):
    if isinstance(x, str):
        return x
    if pd.isna(x):
        return ""
    try:
        return f"€{float(x):,.2f}"
    except Exception:
        return str(x)


def format_number(x):
    if pd.isna(x):
        return ""
    try:
        return f"{float(x):,.2f}"
    except Exception:
        return str(x)


def format_qty(x):
    if pd.isna(x):
        return ""
    try:
        return f"{float(x):.6f}".rstrip("0").rstrip(".")
    except Exception:
        return str(x)


def format_date(d):
    return "" if pd.isna(d) else (d.strftime("%d %b %Y") if hasattr(d, "strftime") else str(d))
