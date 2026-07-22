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

import importlib
import sys
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


def _import_ui_module(module_name: str):
    return importlib.import_module(module_name)

components_module = _import_ui_module("ui.components")
render_cgt1_export_expander = components_module.render_cgt1_export_expander
render_dividend_summary_expander = components_module.render_dividend_summary_expander
render_dividend_tax_sidebar = components_module.render_dividend_tax_sidebar
render_form12_export_expander = components_module.render_form12_export_expander
render_main_sidebar = components_module.render_main_sidebar

annual_summary_module = _import_ui_module("ui.annual_summary")
render_annual_summary_tabs = annual_summary_module.render_annual_summary_tabs

diagnostics_module = _import_ui_module("ui.diagnostics")
render_incoming_transfer_diagnostics = diagnostics_module.render_incoming_transfer_diagnostics
render_manual_missing_diagnostics = diagnostics_module.render_manual_missing_diagnostics

history_module = _import_ui_module("ui.history")
render_transaction_history = history_module.render_transaction_history

positions_module = _import_ui_module("ui.positions")
render_open_positions = positions_module.render_open_positions

reconciliation_module = _import_ui_module("ui.reconciliation")
render_tax_reconciliation_debug = reconciliation_module.render_tax_reconciliation_debug

what_if_module = _import_ui_module("ui.what_if")
render_what_if = what_if_module.render_what_if


# Backwards-compatible aliases used throughout the file
fmt_money = format_number
fmt_money_eur = format_eur
fmt_qty = format_qty
fmt_date = format_date


# ---------------- Page config ----------------
st.set_page_config(page_title="CGT Tool", layout="wide")
st.title("📈 Irish CGT Tool")

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
    for msg in pipeline_result.warnings:
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
cgt1_df_full = build_cgt1_export(out, split_audit_df)
render_cgt1_export_expander(cgt1_df_full)

try:
    cgt1_df = build_cgt1_export(out, split_audit_df)
    for _col in ["CGT Period", "Date Acquired", "Date Disposed"]:
        if _col in cgt1_df.columns:
            cgt1_df[_col] = cgt1_df[_col].astype("string")
    if "Date Acquired" in cgt1_df.columns:
        cgt1_df["Date Acquired"] = cgt1_df["Date Acquired"].fillna("Various")
except Exception as _e:
    st.info(f"CGT1 export unavailable: {_e}")

render_form12_export_expander(out, exit_tax_rate_etf, build_form12_export)
render_dividend_summary_expander(out, tax_bracket, usc_rate, prsi_rate)


# ===================== Annual Summary =====================
st.markdown("### 🧾 Annual Summary")

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

# module-level formatters defined above; local duplicate removed


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

render_tax_reconciliation_debug(
    summary_shares=summary_shares,
    summary_etfs=summary_etfs,
    summary_combined=summary_combined,
    cgt_rate_shares=cgt_rate_shares,
    exit_tax_rate_etf=exit_tax_rate_etf,
    fmt_money_eur=fmt_money_eur,
)


# --------- Transaction History ---------
render_transaction_history(
    out=out,
    years_sorted=years_sorted,
    fmt_date=fmt_date,
    fmt_qty=fmt_qty,
    fmt_money=fmt_money,
    fmt_money_eur=fmt_money_eur,
)

# ===================== MANUAL / MISSING TRANSACTIONS DIAGNOSTICS =====================
render_manual_missing_diagnostics(opening_lots_df=opening_lots_df, out=out)

# ===================== INCOMING TRANSFER HANDLING =====================
render_incoming_transfer_diagnostics(out=out, manual_norm=_manual_norm)

# ===================== OPEN POSITIONS (Cost Basis — current holdings only) =====================
render_open_positions(out=out, replay_fifo_lots_all_fn=replay_fifo_lots_all)

# ===================== WHAT-IF: UI =====================
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
