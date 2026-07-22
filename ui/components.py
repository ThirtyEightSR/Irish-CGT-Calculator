from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

import numpy as np
import pandas as pd
import streamlit as st

from core.settings import DEFAULT_CGT_EXEMPTION_EUR, DEFAULT_CGT_RATE_SHARES, DEFAULT_EXIT_TAX_RATE_ETF


@dataclass
class SidebarState:
    uploads: list[Any]
    opening_lots_df: pd.DataFrame | None
    show_bf_used: bool
    show_ex_used: bool
    show_carry_fw: bool
    show_cashflow: bool
    show_total_fees: bool
    use_exemption: bool
    exemption_val: float
    cgt_rate_shares: float
    exit_tax_rate_etf: float


@dataclass
class DividendTaxState:
    tax_bracket: float
    usc_rate: float
    prsi_rate: float


def render_main_sidebar() -> SidebarState:
    with st.sidebar:
        st.markdown("### 📤 Upload Transactions")

        uploads = st.file_uploader("CSV file(s)", type=["csv"], accept_multiple_files=True, label_visibility="collapsed")

        with st.expander("📥 Upload Missing Transactions", expanded=True):
            st.caption(
                "Upload a rich transaction CSV with `Date`, `Type`, `ISIN`, `Quantity`, "
                "and `Price_EUR`/`Unit_EUR` or `Total (EUR)`. `Type` supports Buy and Sell."
            )
            opening_lots_df = None
            ol_file = st.file_uploader("Missing transactions file", type=["csv"], key="ol_csv")
            if ol_file is not None:
                try:
                    df_ol = pd.read_csv(ol_file)
                except Exception:
                    ol_file.seek(0)
                    df_ol = pd.read_csv(ol_file, sep=";")
                required_cols = {"Date", "Type", "ISIN", "Quantity"}
                found_cols = {str(c).strip() for c in df_ol.columns}
                value_cols = {"Price_EUR", "Unit_EUR", "Total (EUR)", "Total_EUR"}
                if required_cols.issubset(found_cols) and found_cols.intersection(value_cols):
                    opening_lots_df = df_ol
                else:
                    missing_cols = sorted(required_cols - found_cols)
                    if missing_cols:
                        st.error(f"Missing transactions CSV is missing required column(s): {', '.join(missing_cols)}")
                    else:
                        st.error("Missing transactions CSV must include `Price_EUR`, `Unit_EUR`, `Total (EUR)`, or `Total_EUR`.")

        with st.expander("🧾 Summary columns", expanded=False):
            show_bf_used = st.checkbox("B/F Loss Used (EUR)", value=False)
            show_ex_used = st.checkbox("Exemption Used (EUR)", value=False)
            show_carry_fw = st.checkbox("Carry Forward (EUR)", value=False)
            show_cashflow = st.checkbox("Net Cashflow (EUR)", value=False)
            show_total_fees = st.checkbox("Total Fees (EUR)", value=False)

        with st.expander("💶 CGT settings", expanded=False):
            use_exemption = st.checkbox("Apply annual CGT exemption (Shares only)", value=True)
            exemption_val = st.number_input(
                "Exemption amount (EUR)", min_value=0.0, value=DEFAULT_CGT_EXEMPTION_EUR, step=10.0
            )
            cgt_rate_shares = st.number_input(
                "Shares CGT rate", min_value=0.0, max_value=1.0, value=DEFAULT_CGT_RATE_SHARES, step=0.01
            )
            exit_tax_rate_etf = st.number_input(
                "ETFs Exit Tax rate", min_value=0.0, max_value=1.0, value=DEFAULT_EXIT_TAX_RATE_ETF, step=0.01
            )

        with st.expander("✏️ Add Manual Transactions", expanded=False):
            st.caption(
                "Add individual buy/sell transactions here. They will be merged with uploaded files "
                "and included in all calculations (Annual Summary, CGT1 export, etc.)."
            )

            if "manual_transactions" not in st.session_state:
                st.session_state.manual_transactions = []

            col1, col2 = st.columns(2)
            with col1:
                trans_date = st.date_input("Transaction Date", value=datetime.today())
                trans_type = st.selectbox("Type", options=["Buy", "Sell"], index=0)
            with col2:
                isin = st.text_input("ISIN", placeholder="e.g., IE00B4L5Y983").strip().upper()
                product_name = st.text_input("Product Name", placeholder="e.g., Vanguard FTSE 100")

            col3, col4, col5 = st.columns(3)
            with col3:
                quantity = st.number_input("Quantity", min_value=0.0, step=0.01, format="%.6f")
            with col4:
                unit_price = st.number_input("Unit Price (EUR)", min_value=0.0, step=0.01, format="%.4f")
            with col5:
                fees = st.number_input("Fees (EUR)", min_value=0.0, step=0.01, value=0.0, format="%.2f")

            if st.button("➕ Add Transaction", use_container_width=True):
                if not isin:
                    st.error("❌ ISIN is required")
                elif quantity <= 0:
                    st.error("❌ Quantity must be > 0")
                elif unit_price <= 0:
                    st.error("❌ Unit Price must be > 0")
                else:
                    trans = {
                        "Date": trans_date,
                        "Type": trans_type,
                        "ISIN": isin,
                        "Product": product_name or isin,
                        "Quantity": quantity,
                        "Unit_Price_EUR": unit_price,
                        "Fees": fees,
                        "Total_EUR": quantity * unit_price + (fees if trans_type == "Buy" else -fees),
                    }
                    st.session_state.manual_transactions.append(trans)
                    st.success(f"✅ Added {trans_type} transaction for {isin}")

            if st.session_state.manual_transactions:
                st.markdown("**Added Transactions:**")

                manual_df_display = pd.DataFrame(st.session_state.manual_transactions)
                date_series = pd.Series(pd.to_datetime(manual_df_display["Date"], errors="coerce"), index=manual_df_display.index)
                manual_df_display["Date"] = date_series.dt.strftime("%Y-%m-%d")
                manual_df_display["Total_EUR"] = manual_df_display["Total_EUR"].apply(lambda x: f"€{x:,.2f}")
                manual_df_display["Unit_Price_EUR"] = manual_df_display["Unit_Price_EUR"].apply(lambda x: f"€{x:,.4f}")
                manual_df_display["Quantity"] = manual_df_display["Quantity"].apply(lambda x: f"{x:.6f}".rstrip("0").rstrip("."))
                manual_df_display["Fees"] = manual_df_display["Fees"].apply(lambda x: f"€{x:,.2f}")

                st.dataframe(
                    manual_df_display[["Date", "Type", "ISIN", "Quantity", "Unit_Price_EUR", "Fees"]],
                    use_container_width=True,
                )

                if st.button("🗑️ Clear All Manual Transactions", use_container_width=True):
                    st.session_state.manual_transactions = []
                    st.rerun()

    return SidebarState(
        uploads=uploads,
        opening_lots_df=opening_lots_df,
        show_bf_used=show_bf_used,
        show_ex_used=show_ex_used,
        show_carry_fw=show_carry_fw,
        show_cashflow=show_cashflow,
        show_total_fees=show_total_fees,
        use_exemption=use_exemption,
        exemption_val=float(exemption_val),
        cgt_rate_shares=float(cgt_rate_shares),
        exit_tax_rate_etf=float(exit_tax_rate_etf),
    )


def render_dividend_tax_sidebar(out: pd.DataFrame | None) -> DividendTaxState:
    tax_bracket = 40.0
    usc_rate = 0.08
    prsi_rate = 0.04

    detected_currencies: list[str] = []
    if out is not None and not out.empty:
        div_rows = out[out["Type"] == "Dividend"].copy()
        if not div_rows.empty:
            detected_currencies = div_rows.get("Currency", pd.Series(dtype=object)).dropna().astype(str).str.upper().str.strip().tolist()
            detected_currencies = [c for c in detected_currencies if c and c not in ["NAN", "NONE"]]
            detected_currencies = sorted(set(detected_currencies))

    with st.sidebar:
        with st.expander("💰 Dividend Tax Settings", expanded=False):
            st.caption("Set your own rates for estimation. This is not tax advice.")
            preset_options = {
                "High-rate (40/8/4)": (40.0, 8.0, 4.0),
                "Standard (20/4/4)": (20.0, 4.0, 4.0),
                "Custom": None,
            }
            preset = st.selectbox("Rate preset", options=list(preset_options.keys()), index=0, key="div_tax_preset")

            if "div_tax_income_pct" not in st.session_state:
                st.session_state.div_tax_income_pct = 40.0
            if "div_tax_usc_pct" not in st.session_state:
                st.session_state.div_tax_usc_pct = 8.0
            if "div_tax_prsi_pct" not in st.session_state:
                st.session_state.div_tax_prsi_pct = 4.0

            if preset != "Custom":
                p_income, p_usc, p_prsi = preset_options[preset]
                st.session_state.div_tax_income_pct = p_income
                st.session_state.div_tax_usc_pct = p_usc
                st.session_state.div_tax_prsi_pct = p_prsi

            tax_bracket = st.number_input(
                "Income tax rate (%)",
                min_value=0.0,
                max_value=60.0,
                value=float(st.session_state.div_tax_income_pct),
                step=0.5,
                key="div_tax_income_input",
            )
            usc_rate_pct = st.number_input(
                "USC rate (%)",
                min_value=0.0,
                max_value=20.0,
                value=float(st.session_state.div_tax_usc_pct),
                step=0.1,
                key="div_tax_usc_input",
            )
            prsi_rate_pct = st.number_input(
                "PRSI rate (%)",
                min_value=0.0,
                max_value=20.0,
                value=float(st.session_state.div_tax_prsi_pct),
                step=0.1,
                key="div_tax_prsi_input",
            )
            st.session_state.div_tax_income_pct = tax_bracket
            st.session_state.div_tax_usc_pct = usc_rate_pct
            st.session_state.div_tax_prsi_pct = prsi_rate_pct
            usc_rate = usc_rate_pct / 100.0
            prsi_rate = prsi_rate_pct / 100.0

            st.markdown("### 💱 FX Rates for Dividends")
            st.caption("Enter exchange rates to convert non-EUR dividends to EUR")

            non_eur_detected = [c for c in detected_currencies if c != "EUR"]
            for curr in non_eur_detected:
                default_rate = st.session_state.fx_rates_manual.get(curr, 1.0)
                fx_input = st.number_input(
                    f"{curr} → EUR",
                    min_value=0.01,
                    value=default_rate,
                    step=0.01,
                    format="%.4f",
                    key=f"fx_rate_sidebar_{curr}",
                )
                st.session_state.fx_rates_manual[curr] = fx_input

            if not non_eur_detected:
                st.caption("No non-EUR dividends detected in current data.")

            custom_ccy = st.text_input("Add custom currency code", value="", placeholder="e.g. USD", key="fx_custom_ccy").strip().upper()
            if custom_ccy and re.fullmatch(r"[A-Z]{3}", custom_ccy) and custom_ccy != "EUR":
                default_rate = st.session_state.fx_rates_manual.get(custom_ccy, 1.0)
                fx_input_custom = st.number_input(
                    f"{custom_ccy} → EUR (manual)",
                    min_value=0.01,
                    value=default_rate,
                    step=0.01,
                    format="%.4f",
                    key=f"fx_rate_sidebar_custom_{custom_ccy}",
                )
                st.session_state.fx_rates_manual[custom_ccy] = fx_input_custom

    return DividendTaxState(
        tax_bracket=float(tax_bracket),
        usc_rate=float(usc_rate),
        prsi_rate=float(prsi_rate),
    )


def render_cgt1_export_expander(cgt1_df_full: pd.DataFrame) -> None:
    with st.expander("📄 CGT1 export", expanded=False):
        if cgt1_df_full.empty:
            st.info("No disposals to export.")
            return

        _disp = pd.to_datetime(cgt1_df_full["Date Disposed"], errors="coerce")
        years = sorted({d.year for d in _disp.dropna()}, reverse=True)
        year_choice = st.selectbox("Filter by tax year", options=( ["All years"] + years), index=0)

        if year_choice == "All years":
            cgt1_df = cgt1_df_full.copy()
            y_token = "ALL"
        else:
            cgt1_df = cgt1_df_full[_disp.dt.year.eq(int(year_choice))].copy()
            y_token = str(year_choice)

        cgt1_df = cgt1_df.sort_values(by=["CGT Period", "Date Disposed", "Asset Type", "Ticker - Name"], kind="stable")

        totals = {
            "CGT Period": "",
            "Date Acquired": "Totals",
            "Date Disposed": "",
            "Ticker - Name": "",
            "Asset Type": "",
            "ISIN": "",
            "Quantity": pd.to_numeric(cgt1_df["Quantity"], errors="coerce").sum(),
            "Buys + Fees (EUR)": pd.to_numeric(cgt1_df["Buys + Fees (EUR)"], errors="coerce").sum(),
            "Sell Proceeds (EUR)": pd.to_numeric(cgt1_df["Sell Proceeds (EUR)"], errors="coerce").sum(),
            "Gain/Loss (EUR)": pd.to_numeric(cgt1_df["Gain/Loss (EUR)"], errors="coerce").sum(),
            "Order ID": "",
            "Broker": "",
            "Source File": "",
        }
        cgt1_preview = pd.concat([cgt1_df, pd.DataFrame([totals])], ignore_index=True)

        st.dataframe(cgt1_preview, use_container_width=True)
        st.download_button(
            label=f"⬇️ Download CGT1 ({y_token})",
            data=cgt1_df.to_csv(index=False).encode("utf-8"),
            file_name=f"CGT1_{y_token}.csv",
            mime="text/csv",
            use_container_width=True,
        )


def render_form12_export_expander(out: pd.DataFrame, exit_tax_rate_etf: float, build_form12_export_fn: Callable[..., pd.DataFrame]) -> None:
    with st.expander("📄 Form 12 export (ETF Exit Tax)", expanded=False):
        f12_full = build_form12_export_fn(out, exit_tax_rate=exit_tax_rate_etf)

        if f12_full.empty:
            st.info("No ETF disposals or deemed disposals found.")
            return

        d = pd.to_datetime(f12_full["Date"], errors="coerce")
        years = sorted({dt.year for dt in d.dropna()}, reverse=True)
        year_choice = st.selectbox("Filter by tax year", options=( ["All years"] + years), index=0)

        if year_choice == "All years":
            f12_df = f12_full.copy()
            y_token = "ALL"
        else:
            f12_df = f12_full[d.dt.year.eq(int(year_choice))].copy()
            y_token = str(year_choice)

        tax_col = [c for c in f12_df.columns if c.startswith("Tax @ ") and c.endswith("% (EUR)")]
        tax_col = tax_col[0] if tax_col else None

        totals = {
            "Tax Year": "",
            "Date": "",
            "Chargeable Event": "Totals",
            "Ticker - Name": "",
            "ISIN": "",
            "Asset": "",
            "Quantity": pd.to_numeric(f12_df["Quantity"], errors="coerce").sum(),
            "Proceeds (EUR)": pd.to_numeric(f12_df["Proceeds (EUR)"], errors="coerce").sum(),
            "Cost (EUR)": pd.to_numeric(f12_df["Cost (EUR)"], errors="coerce").sum(),
            "Gain/Loss (EUR)": pd.to_numeric(f12_df["Gain/Loss (EUR)"], errors="coerce").sum(),
            "Taxable Gain (EUR)": pd.to_numeric(f12_df["Taxable Gain (EUR)"], errors="coerce").sum(),
            (tax_col or f"Tax @ {int(exit_tax_rate_etf*100)}% (EUR)"): pd.to_numeric(f12_df.get(tax_col, 0), errors="coerce").sum(),
            "Order ID": "",
            "Broker": "",
            "Source File": "",
        }
        preview = pd.concat([f12_df, pd.DataFrame([totals])], ignore_index=True)

        st.dataframe(preview, use_container_width=True)
        st.download_button(
            label=f"⬇️ Download Form 12 (ETF Exit Tax) — {y_token}",
            data=f12_df.to_csv(index=False).encode("utf-8"),
            file_name=f"Form12_ETF_ExitTax_{y_token}.csv",
            mime="text/csv",
            use_container_width=True,
        )


def render_dividend_summary_expander(out: pd.DataFrame, tax_bracket: float, usc_rate: float, prsi_rate: float) -> None:
    with st.expander("💵 Dividend Summary & Tax Calculator", expanded=False):
        divs = out[out["Type"] == "Dividend"].copy()

        if divs.empty:
            st.info("No dividends recorded.")
            return

        divs["Date"] = pd.to_datetime(divs["Date"], errors="coerce")
        divs["Year"] = divs["Date"].dt.year
        divs["Currency"] = divs.get("Currency", "EUR").fillna("EUR")

        divs["FX_Rate"] = divs["Currency"].apply(lambda c: st.session_state.fx_rates_manual.get(c, 1.0) if c != "EUR" else 1.0)

        divs["Gross_Native"] = pd.to_numeric(divs["Total"], errors="coerce").fillna(0.0).abs()
        divs["Gross_EUR"] = divs["Gross_Native"] * divs["FX_Rate"]

        divs["WHT_Native"] = pd.to_numeric(divs["Fee"], errors="coerce").fillna(0.0)
        divs["WHT_Native"] = divs["WHT_Native"].apply(lambda x: -x if x < 0 else x)
        divs["WHT_EUR"] = divs["WHT_Native"] * divs["FX_Rate"]

        divs["Net_EUR"] = divs["Gross_EUR"] - divs["WHT_EUR"]

        isin_prefix = divs.get("ISIN", pd.Series("", index=divs.index)).astype(str).str[:2].str.upper()
        is_uk = isin_prefix.eq("GB")
        divs["Tax_Base_EUR"] = np.where(is_uk, divs["Net_EUR"], divs["Gross_EUR"])
        income_rate = tax_bracket / 100.0
        divs["Income_Tax_EUR"] = divs["Tax_Base_EUR"] * income_rate
        divs["USC_EUR"] = divs["Tax_Base_EUR"] * usc_rate
        divs["PRSI_EUR"] = divs["Tax_Base_EUR"] * prsi_rate
        divs["Credit"] = np.where(is_uk, 0.0, np.minimum(divs["Income_Tax_EUR"], divs["WHT_EUR"]))
        divs["Tax_Due_Ireland"] = (divs["Income_Tax_EUR"] - divs["Credit"]) + divs["USC_EUR"] + divs["PRSI_EUR"]

        summary = divs.groupby(["Year", "Currency"]).agg(
            Gross_EUR=("Gross_EUR", "sum"),
            WHT_EUR=("WHT_EUR", "sum"),
            Net_EUR=("Net_EUR", "sum"),
            Income_Tax_EUR=("Income_Tax_EUR", "sum"),
            USC_EUR=("USC_EUR", "sum"),
            PRSI_EUR=("PRSI_EUR", "sum"),
            Credit=("Credit", "sum"),
            Tax_Due_Ireland=("Tax_Due_Ireland", "sum"),
        ).reset_index().sort_values(by=["Year", "Currency"], ascending=[False, True])

        st.dataframe(
            summary.style.format(
                {
                    "Gross_EUR": "€{:,.2f}".format,
                    "WHT_EUR": "€{:,.2f}".format,
                    "Net_EUR": "€{:,.2f}".format,
                    "Income_Tax_EUR": "€{:,.2f}".format,
                    "USC_EUR": "€{:,.2f}".format,
                    "PRSI_EUR": "€{:,.2f}".format,
                    "Credit": "€{:,.2f}".format,
                    "Tax_Due_Ireland": "€{:,.2f}".format,
                }
            ),
            use_container_width=True,
        )

        total_tax = float(summary["Tax_Due_Ireland"].sum())
        st.success(f"👉 Total Irish dividend tax due: **€{total_tax:,.2f}**")
