# -*- coding: utf-8 -*-
"""
CGT Tool for DEGIRO CSV
- Annual Summary with tabs (Shares CGT / ETFs Exit Tax / Combined / Dividends)
- Dividend handling (gross in Total; tax in Fee; no EUR)
- ETF Exit Tax at 41%
- Money Market fund price change rows excluded
- EUR fallbacks for trades when FX is 'EUR' or 1.0
- FIFO realised P/L (EUR) for sells
- CGT exemption toggle + value
- Loss carry-forward mechanics
- Optional summary columns (B/F Loss Used, Exemption Used, Carry Forward, Net Cashflow, Total Fees)
- Missing transactions import
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

# Ensure the repository root is on sys.path so local packages (e.g. `core`) can be
# imported when the app runs in hosted environments with a different CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import streamlit as st

from adapters.brokers import BROKER_ADAPTERS, detect_broker_from_headers
from adapters.manual_transactions import _manual_transactions_to_canonical
from adapters.missing_transactions import (
    _apply_missing_precedence,
    _is_rich_missing_transactions_file,
    _normalize_manual,
    merge_missing_transactions,
)
from core.deemed_disposal import EXIT_TAX_RATE, deemed_plan_and_estimates as _deemed_plan_and_estimates
from core.reports import build_cgt1_export, build_form12_export
from core.tax import TaxConfig, build_annual_summary
from core.what_if import (
    asset_kind_for_isin,
    available_qty,
    carry_forward_shares_to_year,
    fifo_cost_for_sale,
    last_known_unit_price_eur,
    replay_fifo_lots_all,
    tax_etf_delta,
    tax_shares_delta,
    year_today,
    ytd_realised_gains,
)
from services.corporate_actions import build_output
from services.preprocess import prepare_input_dataframe
from services.pipeline import run_output_pipeline
from utils.formatting import format_date, format_eur, format_number, format_qty


def _load_ui_module(module_name: str, relative_path: str):
    repo_root = Path(__file__).resolve().parent
    module_path = repo_root / relative_path

    if module_name == "ui":
        package = types.ModuleType(module_name)
        package.__file__ = str(repo_root / "ui" / "__init__.py")
        package.__path__ = [str(repo_root / "ui")]
        package.__package__ = module_name
        sys.modules[module_name] = package
        return package

    if module_name not in sys.modules:
        package = sys.modules.get("ui")
        if package is None:
            _load_ui_module("ui", "ui/__init__.py")
            package = sys.modules["ui"]

        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load {module_name} from {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    return sys.modules[module_name]

_load_ui_module("ui", "ui/__init__.py")

components_module = _load_ui_module("ui.components", "ui/components.py")
render_cgt1_export_expander = components_module.render_cgt1_export_expander
render_dividend_summary_expander = components_module.render_dividend_summary_expander
render_dividend_tax_sidebar = components_module.render_dividend_tax_sidebar
render_form12_export_expander = components_module.render_form12_export_expander
render_main_sidebar = components_module.render_main_sidebar
render_welcome_banner = components_module.render_welcome_banner

annual_summary_module = _load_ui_module("ui.annual_summary", "ui/annual_summary.py")
render_annual_summary_tabs = annual_summary_module.render_annual_summary_tabs

diagnostics_module = _load_ui_module("ui.diagnostics", "ui/diagnostics.py")
render_fx_diagnostics = diagnostics_module.render_fx_diagnostics
render_incoming_transfer_diagnostics = diagnostics_module.render_incoming_transfer_diagnostics
render_manual_missing_diagnostics = diagnostics_module.render_manual_missing_diagnostics

history_module = _load_ui_module("ui.history", "ui/history.py")
render_transaction_history = history_module.render_transaction_history

positions_module = _load_ui_module("ui.positions", "ui/positions.py")
render_open_positions = positions_module.render_open_positions

reconciliation_module = _load_ui_module("ui.reconciliation", "ui/reconciliation.py")
render_tax_reconciliation_debug = reconciliation_module.render_tax_reconciliation_debug

what_if_module = _load_ui_module("ui.what_if", "ui/what_if.py")
render_what_if = what_if_module.render_what_if


# Backwards-compatible aliases used throughout the file
fmt_money = format_number
fmt_money_eur = format_eur
fmt_qty = format_qty
fmt_date = format_date


# ---------------- Page config ----------------
st.set_page_config(page_title="CGT Tool", layout="wide")
st.sidebar.markdown("## 📈 Irish CGT Tool")
st.sidebar.caption("Review CGT, ETF exit tax, dividends, and trade history in one place.")
render_welcome_banner()

# ---------------- Sidebar: Import & settings ----------------
sidebar_state = render_main_sidebar()
uploads = sidebar_state.uploads
opening_lots_df = sidebar_state.opening_lots_df
show_bf_used = sidebar_state.show_bf_used
show_ex_used = sidebar_state.show_ex_used
show_carry_fw = sidebar_state.show_carry_fw
show_cashflow = sidebar_state.show_cashflow
show_total_fees = sidebar_state.show_total_fees
use_exemption = sidebar_state.use_exemption
exemption_val = sidebar_state.exemption_val
cgt_rate_shares = sidebar_state.cgt_rate_shares
exit_tax_rate_etf = sidebar_state.exit_tax_rate_etf

# Initialize FX-rate state once (used by dividend calculator)
if "fx_rates_manual" not in st.session_state:
    st.session_state.fx_rates_manual = {}






# ---------------- Main ----------------
has_manual = "manual_transactions" in st.session_state and len(st.session_state.manual_transactions) > 0

if not uploads and not has_manual:
    st.info("👈 Import a CSV or add manual transactions to see results.")
    st.stop()

# Read CSV + normalize + enrich pre-pipeline input
try:
    prep = prepare_input_dataframe(
        uploads=uploads,
        manual_transactions=st.session_state.get("manual_transactions", []),
        opening_lots_df=opening_lots_df,
        detect_broker_from_headers_fn=detect_broker_from_headers,
        broker_adapters=BROKER_ADAPTERS,
        manual_transactions_to_canonical_fn=_manual_transactions_to_canonical,
        is_rich_missing_transactions_file_fn=_is_rich_missing_transactions_file,
        normalize_manual_fn=_normalize_manual,
    )
    df_norm = prep.df_norm
    _is_rich_missing_file = prep.is_rich_missing_file
    _manual_norm = prep.manual_norm
    if prep.merged_manual_count > 0:
        st.info(f"✅ Merged {prep.merged_manual_count} manual transaction(s) into analysis")
except Exception as e:
    st.error(f"Could not prepare input data: {e}")
    st.stop()

# --------- Annual Summary (TOP with tabs) ---------
try:
    pipeline_result = run_output_pipeline(
        df_norm=df_norm,
        opening_lots_df=opening_lots_df,
        is_rich_missing_file=_is_rich_missing_file,
        merge_missing_transactions_fn=merge_missing_transactions,
        apply_missing_precedence_fn=_apply_missing_precedence,
        build_output_fn=build_output,
    )
    df_norm = pipeline_result.df_norm
    out = pipeline_result.out
    split_audit_df = pipeline_result.split_audit_df
    # Keep FX-related warnings in Diagnostics tab only.
    top_level_warnings = [
        msg
        for msg in pipeline_result.warnings
        if ("FX_Rate" not in msg and "FX mapping" not in msg)
    ]
    for msg in top_level_warnings:
        st.warning(msg)

except Exception as e:
    st.error(f"Could not parse CSV: {e}")
    st.stop()

# ---- Dividend Tax Settings (rates + FX inputs) ----
div_tax_state = render_dividend_tax_sidebar(out)
tax_bracket = div_tax_state.tax_bracket
usc_rate = div_tax_state.usc_rate
prsi_rate = div_tax_state.prsi_rate

# Build the full export AFTER merging
cgt1_df_full = None
try:
    cgt1_df_full = build_cgt1_export(out, split_audit_df)
except Exception as _e:
    st.info(f"CGT1 export unavailable: {_e}")

try:
    cgt1_df = build_cgt1_export(out, split_audit_df)
    for _col in ["CGT Period", "Date Acquired", "Date Disposed"]:
        if _col in cgt1_df.columns:
            cgt1_df[_col] = cgt1_df[_col].astype("string")
    if "Date Acquired" in cgt1_df.columns:
        cgt1_df["Date Acquired"] = cgt1_df["Date Acquired"].fillna("Various")
except Exception as _e:
    st.info(f"CGT1 export unavailable: {_e}")


# ===================== Navigation panels =====================
df_sum = out.copy()
for col in ["Total (EUR)", "Fee", "Gain/Loss"]:
    if col in df_sum.columns:
        df_sum[col] = pd.to_numeric(df_sum[col], errors="coerce")

years_sorted = sorted([int(y) for y in df_sum["__year"].dropna().unique()])

tax_cfg = TaxConfig(
    use_exemption=use_exemption,
    exemption_val=exemption_val,
    cgt_rate_shares=cgt_rate_shares,
    exit_tax_rate_etf=exit_tax_rate_etf,
)

summary_shares = build_annual_summary(df_sum, "share", years_sorted, tax_cfg)
summary_etfs = build_annual_summary(df_sum, "etf", years_sorted, tax_cfg)
summary_combined = build_annual_summary(df_sum, None, years_sorted, tax_cfg)

section_labels = ["Overview", "Transactions", "Open Positions", "What-if", "Diagnostics", "Exports"]
section_help = {
    "Overview": "A quick snapshot of key totals before you drill into the detailed tables.",
    "Transactions": "Browse the full trade history and filter by year, asset, broker, or source file.",
    "Open Positions": "Inspect the current holdings and their cost basis without the noise of past trades.",
    "What-if": "Model a sale to understand how it may affect this year’s tax position.",
    "Diagnostics": "Check imported lots, incoming transfers, and any data-matching issues.",
    "Exports": "Review export-ready views for CGT1, Form 12, and dividend summaries.",
}

section_tabs = st.tabs(section_labels)

with section_tabs[0]:
    st.caption(section_help["Overview"])
    if isinstance(out, pd.DataFrame) and not out.empty:
        try:
            rows_analyzed = int(len(out))
            if "ISIN" in out.columns:
                isin_series = pd.Series(out["ISIN"].astype(str).str.strip().replace({"nan": "", "None": ""}))
                unique_isins = int(len({str(value).strip() for value in isin_series.tolist() if str(value).strip()}))
            else:
                unique_isins = 0
            years_covered = int(len(years_sorted))
            dividend_rows = out[out["Type"].eq("Dividend")].copy() if "Type" in out.columns else pd.DataFrame()
            if not dividend_rows.empty and "Total" in dividend_rows.columns:
                dividend_series = pd.Series(pd.to_numeric(dividend_rows["Total"], errors="coerce"))
                dividend_gross = float(dividend_series.fillna(0).abs().sum())
            else:
                dividend_gross = 0.0
            if "Gain/Loss" in out.columns:
                realised_series = pd.Series(pd.to_numeric(out["Gain/Loss"], errors="coerce"))
                realised_total = float(realised_series.fillna(0).sum())
            else:
                realised_total = 0.0

            lots_map = replay_fifo_lots_all(out)
            open_positions = 0
            for _, lots in lots_map.items():
                qty = sum(float(L.get("qty", 0.0)) for L in lots)
                if qty > 1e-12:
                    open_positions += 1

            transaction_mix = (
                out["Type"].astype(str).str.strip().replace({"nan": "Unknown", "None": "Unknown"}).replace("", "Unknown").value_counts().head(5).reset_index()
                if "Type" in out.columns
                else pd.DataFrame(columns=["Type", "Rows"])
            )
            if not transaction_mix.empty:
                transaction_mix.columns = ["Type", "Rows"]

            top_holdings_rows = []
            if "ISIN" in out.columns:
                for isin, lots in lots_map.items():
                    qty = float(sum(float(L.get("qty", 0.0)) for L in lots))
                    if qty <= 1e-12:
                        continue
                    total_cost_eur = float(sum(float(L.get("qty", 0.0)) * float(L.get("unit_cost_eur", 0.0)) for L in lots))
                    isin_mask = out["ISIN"].astype(str).eq(str(isin))
                    name_series = out.loc[isin_mask, "Ticker - Name"].dropna().astype(str) if "Ticker - Name" in out.columns else pd.Series(dtype=str)
                    latest_name = name_series.iloc[-1] if not name_series.empty else str(isin)
                    top_holdings_rows.append(
                        {
                            "Holding": latest_name,
                            "ISIN": str(isin),
                            "Units": qty,
                            "Cost (EUR)": total_cost_eur,
                        }
                    )

            top_holdings = pd.DataFrame(top_holdings_rows).sort_values(by="Cost (EUR)", ascending=False).head(5) if top_holdings_rows else pd.DataFrame(columns=["Holding", "ISIN", "Units", "Cost (EUR)"])

            annual_preview_cols = ["Year", "Buys (EUR)", "Sells (EUR)", "Realised Profit / Loss (EUR)", "Taxable Gain (EUR)"]
            annual_preview_tax_cols = [c for c in summary_combined.columns if c.startswith("Tax @")]
            annual_preview_optional = []
            if show_bf_used:
                annual_preview_optional.append("B/F Loss Used (EUR)")
            if show_ex_used:
                annual_preview_optional.append("Exemption Used (EUR)")
            if show_carry_fw:
                annual_preview_optional.append("Carry Forward (EUR)")
            if show_cashflow:
                annual_preview_optional.append("Net Cashflow (EUR)")
            if show_total_fees:
                annual_preview_optional.append("Total Fees (EUR)")

            annual_preview_ordered = [c for c in annual_preview_cols if c in summary_combined.columns]
            annual_preview_ordered += [c for c in annual_preview_tax_cols if c in summary_combined.columns]
            annual_preview_ordered += [c for c in annual_preview_optional if c in summary_combined.columns]
            annual_preview = summary_combined.loc[:, annual_preview_ordered].copy() if annual_preview_ordered else summary_combined.copy()

            if not annual_preview.empty:
                annual_totals = {}
                for col in annual_preview.columns:
                    if col == "Year":
                        continue
                    annual_totals[col] = float(pd.to_numeric(annual_preview[col], errors="coerce").fillna(0).sum())
                annual_preview = pd.concat([annual_preview, pd.DataFrame([{"Year": "Total", **annual_totals}])], ignore_index=True)
                annual_preview["Year"] = annual_preview["Year"].astype(str)

            current_year = int(max(years_sorted)) if years_sorted else None
            current_year_buys = 0.0
            current_year_sells = 0.0
            current_year_tax = 0.0

            def _sum_numeric_col(df_local: pd.DataFrame, col_name: str) -> float:
                if col_name not in df_local.columns:
                    return 0.0
                series_local = pd.Series(pd.to_numeric(df_local[col_name], errors="coerce"))
                return float(series_local.fillna(0).sum())

            if current_year is not None and not summary_combined.empty and "Year" in summary_combined.columns:
                current_year_values = pd.Series(pd.to_numeric(summary_combined["Year"], errors="coerce"))
                current_year_mask = current_year_values.eq(float(current_year))
                current_year_row = summary_combined.loc[current_year_mask, :].copy()
                if not current_year_row.empty:
                    current_year_buys = _sum_numeric_col(current_year_row, "Buys (EUR)")
                    current_year_sells = _sum_numeric_col(current_year_row, "Sells (EUR)")

                    if "Tax @ Combined (EUR)" in current_year_row.columns:
                        current_year_tax = _sum_numeric_col(current_year_row, "Tax @ Combined (EUR)")
                    elif {
                        f"Tax @ Shares {int(cgt_rate_shares*100)}% (EUR)",
                        f"Tax @ ETFs {int(exit_tax_rate_etf*100)}% (EUR)",
                    }.issubset(current_year_row.columns):
                        current_year_tax = _sum_numeric_col(
                            current_year_row, f"Tax @ Shares {int(cgt_rate_shares*100)}% (EUR)"
                        ) + _sum_numeric_col(current_year_row, f"Tax @ ETFs {int(exit_tax_rate_etf*100)}% (EUR)")
                    else:
                        fallback_tax_cols = [c for c in current_year_row.columns if c.startswith("Tax @")]
                        current_year_tax = float(sum(_sum_numeric_col(current_year_row, c) for c in fallback_tax_cols))

            overview_cols = st.columns(4)
            with overview_cols[0]:
                st.metric("Rows analysed", f"{rows_analyzed:,}")
            with overview_cols[1]:
                st.metric("Unique ISINs", f"{unique_isins:,}")
            with overview_cols[2]:
                st.metric("Dividend gross", fmt_money_eur(dividend_gross))
            with overview_cols[3]:
                st.metric("Open positions", f"{open_positions}")

            overview_cols_2 = st.columns(4)
            with overview_cols_2[0]:
                st.metric("Years covered", f"{years_covered}")
            with overview_cols_2[1]:
                st.metric("Realised P/L", fmt_money_eur(realised_total))
            with overview_cols_2[2]:
                if "Fee" in out.columns:
                    fee_series = pd.Series(pd.to_numeric(out["Fee"], errors="coerce"))
                    fee_total = float(fee_series.fillna(0).sum())
                else:
                    fee_total = 0.0
                st.metric("Fees & tax (EUR)", fmt_money_eur(fee_total))
            with overview_cols_2[3]:
                st.metric("Tx rows", f"{rows_analyzed:,}")

            overview_cols_3 = st.columns(3)
            year_suffix = f" ({current_year})" if current_year is not None else ""
            with overview_cols_3[0]:
                st.metric(f"Buys{year_suffix}", fmt_money_eur(current_year_buys))
            with overview_cols_3[1]:
                st.metric(f"Sells{year_suffix}", fmt_money_eur(current_year_sells))
            with overview_cols_3[2]:
                st.metric(f"Tax owed{year_suffix}", fmt_money_eur(current_year_tax))

            st.markdown("#### Annual Summary Details")
            st.caption("Detailed annual summaries, dividend breakdowns, and deemed-disposal projections.")
            render_annual_summary_tabs(
                summary_shares=summary_shares,
                summary_etfs=summary_etfs,
                summary_combined=summary_combined,
                out=out,
                show_bf_used=show_bf_used,
                show_ex_used=show_ex_used,
                show_carry_fw=show_carry_fw,
                show_cashflow=show_cashflow,
                show_total_fees=show_total_fees,
                fmt_money=fmt_money,
                fmt_money_eur=fmt_money_eur,
                deemed_plan_and_estimates_fn=_deemed_plan_and_estimates,
                deemed_exit_tax_rate=EXIT_TAX_RATE,
            )

            snap_cols = st.columns(2)
            with snap_cols[0]:
                st.markdown("#### Transaction Mix")
                st.caption("How the loaded file breaks down by transaction type.")
                if transaction_mix.empty:
                    st.info("No transaction type breakdown available.")
                else:
                    st.dataframe(transaction_mix, use_container_width=True, hide_index=True)
            with snap_cols[1]:
                st.markdown("#### Top Open Holdings")
                st.caption("Largest current holdings by fee-adjusted cost basis.")
                if top_holdings.empty:
                    st.info("No open holdings available yet.")
                else:
                    st.dataframe(
                        top_holdings.style.format({"Units": lambda x: f"{float(x):.6f}".rstrip("0").rstrip("."), "Cost (EUR)": fmt_money_eur}),
                        use_container_width=True,
                    )
        except Exception as overview_error:
            st.caption(f"Overview metrics unavailable: {overview_error}")
    else:
        st.info("Upload and process data to see the overview summary.")

with section_tabs[1]:
    st.caption(section_help["Transactions"])
    render_transaction_history(
        out=out,
        years_sorted=years_sorted,
        fmt_date=fmt_date,
        fmt_qty=fmt_qty,
        fmt_money=fmt_money,
        fmt_money_eur=fmt_money_eur,
    )

with section_tabs[2]:
    st.caption(section_help["Open Positions"])
    render_open_positions(out=out, replay_fifo_lots_all_fn=replay_fifo_lots_all)

with section_tabs[3]:
    st.caption(section_help["What-if"])
    render_what_if(
        out=out,
        cgt_rate_shares=cgt_rate_shares,
        exit_tax_rate_etf=exit_tax_rate_etf,
        use_exemption=use_exemption,
        exemption_val=exemption_val,
        replay_fifo_lots_all_fn=replay_fifo_lots_all,
        available_qty_fn=available_qty,
        last_known_unit_price_eur_fn=last_known_unit_price_eur,
        asset_kind_for_isin_fn=asset_kind_for_isin,
        fifo_cost_for_sale_fn=fifo_cost_for_sale,
        year_today_fn=year_today,
        ytd_realised_gains_fn=ytd_realised_gains,
        carry_forward_shares_to_year_fn=carry_forward_shares_to_year,
        tax_shares_delta_fn=tax_shares_delta,
        tax_etf_delta_fn=tax_etf_delta,
    )

with section_tabs[4]:
    st.caption(section_help["Diagnostics"])
    render_fx_diagnostics(out=out)
    render_manual_missing_diagnostics(opening_lots_df=opening_lots_df, out=out)
    render_incoming_transfer_diagnostics(out=out, manual_norm=_manual_norm)
    render_tax_reconciliation_debug(
        summary_shares=summary_shares,
        summary_etfs=summary_etfs,
        summary_combined=summary_combined,
        cgt_rate_shares=cgt_rate_shares,
        exit_tax_rate_etf=exit_tax_rate_etf,
        fmt_money_eur=fmt_money_eur,
    )

with section_tabs[5]:
    st.caption(section_help["Exports"])
    if cgt1_df_full is not None:
        render_cgt1_export_expander(cgt1_df_full, summary_shares=summary_shares)
    else:
        st.info("CGT1 export is unavailable for the current data.")
    render_form12_export_expander(out, exit_tax_rate_etf, build_form12_export, summary_etfs=summary_etfs)
    render_dividend_summary_expander(out, tax_bracket, usc_rate, prsi_rate)
