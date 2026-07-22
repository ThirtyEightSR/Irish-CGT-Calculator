from __future__ import annotations

import pandas as pd


def _manual_transactions_to_canonical(manual_list: list) -> pd.DataFrame:
    """
    Convert manual transaction entries from session state to canonical CSV format.
    """
    if not manual_list:
        return pd.DataFrame()

    rows = []
    for i, trans in enumerate(manual_list):
        date_val = pd.to_datetime(trans["Date"])
        trans_type = trans["Type"]
        isin = str(trans["ISIN"]).strip()
        product = str(trans["Product"]).strip() or isin
        qty = float(trans["Quantity"])
        unit_price_eur = float(trans["Unit_Price_EUR"])
        total_eur = float(trans["Total_EUR"])

        desc = f"{trans_type} {qty:g} {product}@{unit_price_eur:g} EUR"
        change = -total_eur if trans_type == "Buy" else total_eur
        cash_movements = -total_eur if trans_type == "Buy" else total_eur

        rows.append(
            {
                "Date": date_val,
                "Time": None,
                "Value date": None,
                "Product": product,
                "ISIN": isin,
                "Description": desc,
                "FX": "EUR",
                "Change": change,
                "Cash Movements": cash_movements,
                "Balance": None,
                "Order ID": f"MANUAL-{isin}-{i:04d}",
                "Currency": "EUR",
                "__Broker": "MANUAL",
                "__SourceFile": "manual_entry",
            }
        )

    return pd.DataFrame(rows)
