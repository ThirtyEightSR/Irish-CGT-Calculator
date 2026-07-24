from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.tax import TaxConfig, build_annual_summary
from core.deemed_disposal import _eight_year_anniversary
from core.deemed_disposal import deemed_plan_and_estimates
from core.reports import build_cgt1_export, build_form12_export
from core.what_if import carry_forward_shares_to_year, fifo_cost_for_sale
from services.output_builder import build_out_table
from services.pipeline import run_output_pipeline
from services.corporate_actions import build_output
from ui.reconciliation import build_tax_reconciliation_frame
from ui.diagnostics import build_fx_diagnostics_frame



def test_tax_summary_falls_back_to_total_eur_when_fee_adj_missing() -> None:
    df = pd.DataFrame(
        {
            "__year": [2025, 2025],
            "Asset": ["share", "share"],
            "Type": ["Buy", "Sell"],
            "Total (EUR)": [1000.0, 1300.0],
            "Fee": [0.0, 0.0],
            "Total": [1000.0, 1300.0],
            "Gain/Loss": [None, 300.0],
        }
    )

    out = build_annual_summary(df, "share", [2025], TaxConfig(use_exemption=False))
    assert not out.empty
    assert float(out.loc[0, "Buys (EUR)"]) == 1000.0
    assert float(out.loc[0, "Sells (EUR)"]) == 1300.0



def test_carry_forward_handles_missing_gain_loss_column() -> None:
    df = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2024-01-01")],
            "Type": ["Sell"],
            "Asset": ["share"],
        }
    )
    assert carry_forward_shares_to_year(df, 2025, True, 1270.0) == 0.0



def test_fifo_cost_caps_to_available_lots() -> None:
    out = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2024-01-01")],
            "ISIN": ["IE00TEST12345"],
            "Type": ["Buy"],
            "Quantity": [10.0],
            "Price_EUR": [10.0],
            "Total (EUR, fee-adj)": [100.0],
            "Description": ["Buy 10 TEST @10 EUR"],
        }
    )
    # Asking for cost of 15 should only consume 10 held units.
    assert fifo_cost_for_sale(out, "IE00TEST12345", 15.0) == 100.0


def test_deemed_disposal_leap_day_anniversary_stays_in_february() -> None:
    assert _eight_year_anniversary(pd.Timestamp("2020-02-29")) == pd.Timestamp("2028-02-29")


def test_deemed_planner_includes_upcoming_lots_within_12_months() -> None:
    out = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2019-10-01")],
            "ISIN": ["IE00ETFUPCOMING1"],
            "Order ID": ["OID-ETF-1"],
            "Type": ["Buy"],
            "Asset": ["etf"],
            "Quantity": [10.0],
            "Price_EUR": [20.0],
            "Total (EUR, fee-adj)": [200.0],
            "Total (EUR)": [200.0],
            "Ticker - Name": ["Test ETF"],
            "Description": ["Buy 10 Test ETF @20 EUR"],
            "Currency": ["EUR"],
        }
    )

    planner, est = deemed_plan_and_estimates(out, asof=pd.Timestamp("2027-11-01"))

    assert not planner.empty
    assert not est.empty
    assert planner["ISIN"].astype(str).eq("IE00ETFUPCOMING1").any()


def test_pipeline_warns_on_missing_trade_valuations() -> None:
    df_norm = pd.DataFrame(
        {
            "Type": ["Buy", "Sell"],
            "Total (EUR)": [None, 1300.0],
            "Price_EUR": [10.0, None],
            "Gain/Loss": [None, None],
        }
    )

    def _build_output(df_work: pd.DataFrame, opening_lots_df):
        return df_work.copy(), pd.DataFrame()

    result = run_output_pipeline(
        df_norm=df_norm,
        opening_lots_df=None,
        is_rich_missing_file=False,
        merge_missing_transactions_fn=lambda a, b: a,
        apply_missing_precedence_fn=lambda a, b: a,
        build_output_fn=_build_output,
    )

    assert any("Total (EUR)" in msg for msg in result.warnings)
    assert any("Price_EUR" in msg for msg in result.warnings)
    assert any("Gain/Loss" in msg for msg in result.warnings)


def test_pipeline_warns_on_suspicious_fx_mapping() -> None:
    df_norm = pd.DataFrame({"Type": ["Buy"], "Total (EUR)": [100.0]})

    out = pd.DataFrame(
        {
            "Type": ["Buy"],
            "Currency": ["USD"],
            "FXCCY": ["USD"],
            "FX_Rate": [1.0],
            "Total (EUR)": [100.0],
            "Price_EUR": [10.0],
            "Gain/Loss": [None],
        }
    )

    def _build_output(df_work: pd.DataFrame, opening_lots_df):
        return out.copy(), pd.DataFrame()

    result = run_output_pipeline(
        df_norm=df_norm,
        opening_lots_df=None,
        is_rich_missing_file=False,
        merge_missing_transactions_fn=lambda a, b: a,
        apply_missing_precedence_fn=lambda a, b: a,
        build_output_fn=_build_output,
    )

    assert any("FX_Rate=1.0" in msg for msg in result.warnings)


def test_pipeline_fills_missing_fx_rate_from_nearest_date() -> None:
    """Test that missing FX rates are filled using nearest-date valid rate for same currency pair."""
    out = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-05"), pd.Timestamp("2025-01-10")],
            "Type": ["Buy", "Buy", "Buy"],
            "Currency": ["USD", "USD", "USD"],
            "FXCCY": ["USD", "USD", "USD"],
            "FX_Rate": [1.15, None, 1.18],  # Middle row missing rate
            "Total (EUR)": [100.0, 100.0, 100.0],
            "Price_EUR": [10.0, 10.0, 10.0],
            "Gain/Loss": [None, None, None],
        }
    )

    def _build_output(df_work: pd.DataFrame, opening_lots_df):
        return out.copy(), pd.DataFrame()

    result = run_output_pipeline(
        df_norm=out,
        opening_lots_df=None,
        is_rich_missing_file=False,
        merge_missing_transactions_fn=lambda a, b: a,
        apply_missing_precedence_fn=lambda a, b: a,
        build_output_fn=_build_output,
    )

    # Should fill missing FX_Rate with nearest-date rate (1.15 is closer than 1.18)
    assert float(result.out.loc[1, "FX_Rate"]) == 1.15
    # Should track that FX_Rate was sourced from 2025-01-01
    assert pd.to_datetime(result.out.loc[1, "FX_Rate_Source_Date"]).date() == pd.Timestamp("2025-01-01").date()
    assert any("Filled" in msg and "FX_Rate" in msg for msg in result.warnings)


def test_pipeline_derives_missing_fx_rate_from_trade_totals() -> None:
    out = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2025-01-05")],
            "Type": ["Buy"],
            "Currency": ["USD"],
            "FXCCY": [""],
            "FX_Rate": [None],
            "Total": [121.20],
            "Total (EUR)": [108.55],
            "Price_EUR": [27.1375],
            "Gain/Loss": [None],
        }
    )

    def _build_output(df_work: pd.DataFrame, opening_lots_df):
        return out.copy(), pd.DataFrame()

    result = run_output_pipeline(
        df_norm=out,
        opening_lots_df=None,
        is_rich_missing_file=False,
        merge_missing_transactions_fn=lambda a, b: a,
        apply_missing_precedence_fn=lambda a, b: a,
        build_output_fn=_build_output,
    )

    expected_rate = 121.20 / 108.55
    assert float(result.out.loc[0, "FX_Rate"]) == expected_rate
    assert pd.to_datetime(result.out.loc[0, "FX_Rate_Source_Date"]).date() == pd.Timestamp("2025-01-05").date()
    assert any("Derived" in msg and "FX_Rate" in msg for msg in result.warnings)


def test_build_out_table_preserves_fx_columns_for_diagnostics() -> None:
    consolidated = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2025-01-05")],
            "Product": ["Test USD Asset"],
            "ISIN": ["US00TESTFX123"],
            "Order ID": ["OID-1"],
            "Type": ["Buy"],
            "Asset": ["share"],
            "Currency": ["USD"],
            "FXCCY": ["USD"],
            "FX_Rate": [1.15],
            "Quantity_signed": [10.0],
            "Price": [12.0],
            "Fee_signed": [0.0],
            "Total_signed": [120.0],
            "Total_EUR": [104.35],
            "Total_EUR_FeeAdj": [104.35],
            "Gain/Loss": [None],
            "Description": ["Buy 10 Test USD Asset @12 USD"],
        }
    )

    out = build_out_table(consolidated)

    assert "FXCCY" in out.columns
    assert "FX_Rate" in out.columns
    assert out.loc[0, "FXCCY"] == "USD"
    assert float(out.loc[0, "FX_Rate"]) == 1.15


def test_fx_diagnostics_flags_row_level_anomaly() -> None:
    out = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2025-01-01")],
            "Ticker - Name": ["Test USD Asset"],
            "ISIN": ["US00TESTFX123"],
            "Type": ["Buy"],
            "Currency": ["USD"],
            "FXCCY": ["USD"],
            "FX_Rate": [1.0],
            "Total (EUR)": [100.0],
            "Price_EUR": [10.0],
            "Description": ["Buy 10 Test USD Asset @10 USD"],
        }
    )

    fx_diag = build_fx_diagnostics_frame(out)
    assert not fx_diag.empty
    assert any("Suspicious FX_Rate=1.0" in issue for issue in fx_diag["Issue"].tolist())


def test_fx_diagnostics_prioritizes_missing_fxccy_over_suspicious_one() -> None:
    out = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2025-01-01")],
            "Ticker - Name": ["Test USD Asset"],
            "ISIN": ["US00TESTFX124"],
            "Type": ["Buy"],
            "Currency": ["USD"],
            "FXCCY": [""],
            "FX_Rate": [1.0],
            "Total (EUR)": [100.0],
            "Price_EUR": [10.0],
            "Description": ["Buy 10 Test USD Asset @10 USD"],
        }
    )

    fx_diag = build_fx_diagnostics_frame(out)
    assert not fx_diag.empty
    assert fx_diag.iloc[0]["Issue"] == "Missing FXCCY code for non-EUR trade"


def test_exports_align_on_simple_trade_row() -> None:
    out = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2025-03-01")],
            "Ticker - Name": ["Test ETF"],
            "ISIN": ["IE00TEST12345"],
            "Type": ["Sell"],
            "Asset": ["share"],
            "Quantity": [10.0],
            "Total (EUR)": [1500.0],
            "Total (EUR, fee-adj)": [1500.0],
            "Gain/Loss": [500.0],
            "Order ID": ["OID-1"],
            "__Broker": ["BROKER"],
            "__SourceFile": ["sample.csv"],
            "__row_id": [7],
        }
    )

    cgt1 = build_cgt1_export(out)
    assert float(cgt1.loc[0, "Sell Proceeds (EUR)"]) == 1500.0
    assert float(cgt1.loc[0, "Gain/Loss (EUR)"]) == 500.0
    assert float(cgt1.loc[0, "Buys + Fees (EUR)"]) == 1000.0

    f12 = build_form12_export(out, exit_tax_rate=0.41)
    assert f12.empty


def _minimal_norm_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": ["2025-01-01", "2025-01-02", "2026-01-01", "2026-01-01"],
            "Time": ["10:00", "10:00", "10:00", "10:00"],
            "Product": ["Test Split Asset", "Test Split Asset", "Test Split Asset", "Test Split Asset"],
            "ISIN": ["IE00SPLIT123", "IE00SPLIT123", "IE00SPLIT123", "IE00SPLIT123"],
            "Description": [
                "Buy 10 Test Split Asset @10 EUR",
                "Buy 5 Test Split Asset @12 EUR",
                "STOCK SPLIT: Sell 10 Test Split Asset @5 EUR (IE00SPLIT123)",
                "STOCK SPLIT: Buy 20 Test Split Asset @5 EUR (IE00SPLIT123)",
            ],
            "Change": [-100.0, -60.0, 0.0, 0.0],
            "Cash Movements": [-100.0, -60.0, 0.0, 0.0],
            "Balance": [900.0, 840.0, 840.0, 840.0],
            "Order ID": ["OPENING-1", "OPENING-2", "SPLIT-1", "SPLIT-1"],
            "FX": ["EUR", "EUR", "EUR", "EUR"],
        }
    )


def test_corporate_action_split_adjusts_pre_split_buy_rows() -> None:
    out, split_audit_df = build_output(_minimal_norm_rows(), None)

    buy_rows = out[out["Type"].eq("Buy")].sort_values(by="Date")
    assert len(buy_rows) == 2
    assert float(buy_rows.iloc[0]["Quantity"]) == 20.0
    assert float(buy_rows.iloc[0]["Price"]) == 5.0
    assert float(buy_rows.iloc[1]["Quantity"]) == 10.0
    assert float(buy_rows.iloc[1]["Price"]) == 6.0
    assert not split_audit_df.empty


def test_missing_order_ids_keep_same_minute_rows_separate() -> None:
    df_norm = pd.DataFrame(
        {
            "Date": ["2025-03-01", "2025-03-01"],
            "Time": ["10:00", "10:00"],
            "Product": ["Asset A", "Asset B"],
            "ISIN": ["IE00MISSORD1", "IE00MISSORD2"],
            "Description": ["Buy 3 Asset A @10 EUR", "Buy 4 Asset B @11 EUR"],
            "Change": [-30.0, -44.0],
            "Cash Movements": [-30.0, -44.0],
            "Balance": [970.0, 926.0],
            "Order ID": ["", ""],
            "FX": ["EUR", "EUR"],
        }
    )

    out, _ = build_output(df_norm, None)
    buy_rows = out[out["Type"].eq("Buy")]
    assert len(buy_rows) == 2
    assert sorted(buy_rows["ISIN"].astype(str).tolist()) == ["IE00MISSORD1", "IE00MISSORD2"]


def test_tax_reconciliation_frame_reports_zero_delta() -> None:
    summary_shares = pd.DataFrame(
        {
            "Year": [2025],
            "Realised Profit / Loss (EUR)": [1000.0],
            "Taxable Gain (EUR)": [1000.0],
            "Tax @ 33% (EUR)": [330.0],
        }
    )
    summary_etfs = pd.DataFrame(columns=["Year", "Realised Profit / Loss (EUR)", "Taxable Gain (EUR)", "Tax @ 41% (EUR)"])
    summary_combined = pd.DataFrame(
        {
            "Year": [2025],
            "Tax @ Combined (EUR)": [330.0],
        }
    )

    rec = build_tax_reconciliation_frame(summary_shares, summary_etfs, summary_combined, 0.33, 0.41)
    assert not rec.empty
    assert float(rec.loc[2025, "Delta (Reported - Recomputed) (EUR)"]) == 0.0


def test_export_totals_match_annual_summaries() -> None:
    out = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2025-03-01"), pd.Timestamp("2025-06-01")],
            "Ticker - Name": ["Test Share", "Test ETF"],
            "ISIN": ["IE00TESTS1234", "IE00TESTE1234"],
            "Type": ["Sell", "Sell"],
            "Asset": ["share", "etf"],
            "Quantity": [10.0, 5.0],
            "Total (EUR)": [1500.0, 800.0],
            "Total (EUR, fee-adj)": [1500.0, 800.0],
            "Gain/Loss": [500.0, 200.0],
            "Order ID": ["OID-1", "OID-2"],
            "__Broker": ["BROKER", "BROKER"],
            "__SourceFile": ["sample.csv", "sample.csv"],
            "__row_id": [7, 8],
            "__year": [2025, 2025],
        }
    )

    tax_cfg = TaxConfig(use_exemption=False, exit_tax_rate_etf=0.41)
    summary_shares = build_annual_summary(out, "share", [2025], tax_cfg)
    summary_etfs = build_annual_summary(out, "etf", [2025], tax_cfg)

    cgt1 = build_cgt1_export(out)
    assert float(cgt1["Gain/Loss (EUR)"].sum()) == float(summary_shares["Realised Profit / Loss (EUR)"].sum())

    f12 = build_form12_export(out, exit_tax_rate=0.41)
    assert float(f12["Taxable Gain (EUR)"].sum()) == float(summary_etfs["Taxable Gain (EUR)"].sum())
    assert float(f12[f"Tax @ 41% (EUR)"].sum()) == float(summary_etfs["Tax @ 41% (EUR)"].sum())
