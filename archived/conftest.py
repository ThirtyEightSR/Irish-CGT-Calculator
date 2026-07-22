# tests/conftest.py
import os
import sys
from pathlib import Path
import pytest

# Prevent the Streamlit UI from launching when importing app.py
os.environ["CGT_TEST_MODE"] = "1"

# Ensure the project root (where app.py lives) is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

@pytest.fixture
def degiro_min_df():
    # Minimal DEGIRO-like dataframe (already header-canonicalizable)
    return pd.DataFrame([
        {
            "Date": "13-09-2023", "Time": "08:32", "Value date": "13-09-2023",
            "Product": "ACME UCITS ETF", "ISIN": "IE00B4L5Y983",
            "Description": "Buy 10 ACME UCITS ETF@50.00 USD (IE00B4L5Y983)",
            "FX": "USD", "Change": -500.00, "Balance": 10000.00, "Order ID": "OID-1"
        },
        {
            "Date": "20-09-2023", "Time": "09:15", "Value date": "20-09-2023",
            "Product": "ACME UCITS ETF", "ISIN": "IE00B4L5Y983",
            "Description": "Sell 10 ACME UCITS ETF@55.00 USD (IE00B4L5Y983)",
            "FX": "USD", "Change": 550.00, "Balance": 10550.00, "Order ID": "OID-2"
        },
    ])

@pytest.fixture
def t212_min_df():
    return pd.DataFrame([
        {"Action":"Deposit","Time":"2023-05-01 10:00:00","ISIN":"","Ticker":"","Name":"","ID":"DEP1",
         "Total":100.00,"Currency (Total)":"EUR"},
        {"Action":"Market buy","Time":"2023-05-02 11:00:00","ISIN":"IE00ABCDEF01","Ticker":"ACME","Name":"ACME UCITS ETF","ID":"B1",
         "No. of shares":10,"Price / share":5.00,"Currency (Price / share)":"USD","Exchange rate":"1.10000000",
         "Total":50.00,"Currency (Total)":"USD","Currency conversion fee":0.10,"Currency (Currency conversion fee)":"EUR"},
        {"Action":"Dividend","Time":"2023-06-01 09:00:00","ISIN":"IE00ABCDEF01","Ticker":"ACME","Name":"ACME UCITS ETF","ID":"D1",
         "Total":2.00,"Currency (Total)":"USD","Withholding tax":0.30,"Currency (Withholding tax)":"USD"},
        {"Action":"Market sell","Time":"2023-06-10 14:00:00","ISIN":"IE00ABCDEF01","Ticker":"ACME","Name":"ACME UCITS ETF","ID":"S1",
         "No. of shares":10,"Price / share":5.50,"Currency (Price / share)":"USD","Exchange rate":"1.08000000",
         "Total":55.00,"Currency (Total)":"USD"}
    ])


@pytest.fixture
def ib_min_df():
    # Minimal IB-like CSV export shape (trade log style)
    return pd.DataFrame([
        {
            "Trade Date": "2023-09-13",
            "Symbol": "ACME",
            "Description": "ACME UCITS ETF",
            "Buy/Sell": "BUY",
            "Quantity": 10,
            "Price": 50.00,
            "Currency": "USD",
            # "Amount": -500.00,  # optional in some exports
        },
        {
            "Trade Date": "2023-09-20",
            "Symbol": "ACME",
            "Description": "ACME UCITS ETF",
            "Buy/Sell": "SELL",
            "Quantity": 10,
            "Price": 55.00,
            "Currency": "USD",
            # "Amount": 550.00,
        },
    ])
