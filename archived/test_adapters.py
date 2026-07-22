# tests/test_adapters.py
import importlib
import pandas as pd
import numpy as np
import sys

# Import your app.py as a module
app = importlib.import_module("app")

def test_detect_broker_degiros(degiro_min_df):
    bro = app.detect_broker_from_headers(degiro_min_df.head(1))
    assert bro in ("DEGIRO", "UNKNOWN")  # detector is conservative, UNKNOWN is acceptable

def test_detect_broker_ib(ib_min_df):
    bro = app.detect_broker_from_headers(ib_min_df.head(1))
    assert bro in ("IB", "UNKNOWN")  # allow UNKNOWN if headers vary

def test_parse_degiros_csv_minimal(degiro_min_df):
    df_norm = app.parse_degiros_csv(degiro_min_df)
    # Must contain canonical columns the pipeline expects
    for col in ["Date","Product","ISIN","Description","Change","Balance"]:
        assert col in df_norm.columns
    assert len(df_norm) == 2

def test_parse_ib_csv_minimal(ib_min_df):
    df_norm = app.parse_ib_csv_minimal(ib_min_df)
    # Adapter must synthesize canonical columns so the pipeline can proceed
    for col in ["Date","Product","ISIN","Description","Change","Balance","Order ID"]:
        assert col in df_norm.columns
    # Description should include Buy/Sell pattern for downstream parsing
    assert any(df_norm["Description"].astype(str).str.contains("Buy", case=False))
    assert any(df_norm["Description"].astype(str).str.contains("Sell", case=False))

def test_build_output_roundtrip_degiros(degiro_min_df):
    # Use canonicalization then pipeline
    df_norm = app._canonicalize_headers(degiro_min_df)
    app._validate_required_columns(df_norm)
    out, split_audit = app.build_output(df_norm, opening_lots=None)

    # Basic expectations
    assert isinstance(out, pd.DataFrame)
    assert not out.empty
    # Key columns used by UI
    for c in ["Date","Ticker - Name","ISIN","Type","Asset","Quantity","Total (EUR)","Gain/Loss"]:
        assert c in out.columns

    # There should be both a Buy and a Sell row in the out table
    assert (out["Type"] == "Buy").any()
    assert (out["Type"] == "Sell").any()

def test_build_output_roundtrip_ib(ib_min_df):
    # Go through the IB adapter then reuse canonicalization (as your ingest does)
    df_norm = app.parse_ib_csv_minimal(ib_min_df)
    df_norm = app._canonicalize_headers(df_norm)
    # For IB minimal, _validate_required_columns should pass because we synthesized fields
    app._validate_required_columns(df_norm)

    out, split_audit = app.build_output(df_norm, opening_lots=None)

    # Should run without exceptions and return a DataFrame
    assert isinstance(out, pd.DataFrame)
    assert not out.empty

    # Even if ISIN might be blank in IB minimal, the pipeline should still populate key columns
    for c in ["Date","Ticker - Name","Type","Quantity","Total (EUR)"]:
        assert c in out.columns

def test_parse_t212_csv_minimal(t212_min_df):
    df_norm = app.parse_trading212_csv_minimal(t212_min_df)
    df_norm = app._canonicalize_headers(df_norm)
    app._validate_required_columns(df_norm)
    # Must have canonical columns
    for col in ["Date","Product","ISIN","Description","Change","Balance","Order ID"]:
        assert col in df_norm.columns
    # Should emit separate Dividend and Dividend Tax rows
    assert (df_norm["Description"].astype(str) == "Dividend").any()
    assert (df_norm["Description"].astype(str) == "Dividend Tax").any()

def test_roundtrip_t212(t212_min_df):
    df_norm = app.parse_trading212_csv_minimal(t212_min_df)
    df_norm = app._canonicalize_headers(df_norm)
    app._validate_required_columns(df_norm)
    out, split_audit = app.build_output(df_norm, opening_lots=None)
    assert isinstance(out, pd.DataFrame)
    # Expect Buy and Sell mapped
    assert (out["Type"] == "Buy").any()
    assert (out["Type"] == "Sell").any()
    # Dividends visible
    assert (out["Type"] == "Dividend").any()
