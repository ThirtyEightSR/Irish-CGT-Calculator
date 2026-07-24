from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Sequence

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
    compact_mode: bool


@dataclass
class DividendTaxState:
    tax_bracket: float
    usc_rate: float
    prsi_rate: float


def inject_global_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --cgt-bg: #f5f7fb;
            --cgt-card: #ffffff;
            --cgt-border: #d9e0ea;
            --cgt-text: #1f2937;
            --cgt-muted: #5c6b80;
            --cgt-pos: #0f7b45;
            --cgt-neg: #a61b1b;
            --cgt-accent: #0c5da5;
        }

        .stApp {
            background: radial-gradient(circle at 5% 0%, #f2f6fd 0%, #f7f9fc 38%, #fbfcfe 100%);
        }

        .cgt-section-intro {
            background: linear-gradient(135deg, #f8fbff 0%, #eef4fb 100%);
            border: 1px solid var(--cgt-border);
            border-radius: 14px;
            padding: 0.65rem 0.9rem;
            margin: 0.25rem 0 0.8rem 0;
        }

        .cgt-section-intro p {
            color: var(--cgt-muted);
            margin: 0;
            font-size: 0.92rem;
            line-height: 1.35;
        }

        .cgt-card {
            background: var(--cgt-card);
            border: 1px solid var(--cgt-border);
            border-radius: 14px;
            padding: 0.75rem 0.8rem;
            min-height: 90px;
            margin-bottom: 0.55rem;
            box-shadow: 0 1px 0 rgba(10, 30, 60, 0.03);
        }

        .cgt-card-label {
            color: var(--cgt-muted);
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 0.28rem;
            font-weight: 600;
        }

        .cgt-card-value {
            color: var(--cgt-text);
            font-size: 1.2rem;
            font-weight: 700;
            line-height: 1.2;
        }

        .cgt-card-value.pos {
            color: var(--cgt-pos);
        }

        .cgt-card-value.neg {
            color: var(--cgt-neg);
        }

        .cgt-card-note {
            color: var(--cgt-muted);
            font-size: 0.76rem;
            margin-top: 0.35rem;
        }

        .cgt-chip-wrap {
            display: flex;
            flex-wrap: wrap;
            gap: 0.38rem;
            margin: 0.15rem 0 0.45rem 0;
        }

        .cgt-chip {
            display: inline-block;
            border: 1px solid #c9d7e8;
            border-radius: 999px;
            padding: 0.16rem 0.54rem;
            font-size: 0.78rem;
            color: #1d4d7b;
            background: #f4f8ff;
            font-weight: 500;
        }

        @media (max-width: 900px) {
            .cgt-section-intro {
                padding: 0.55rem 0.7rem;
                border-radius: 12px;
                margin-bottom: 0.65rem;
            }

            .cgt-section-intro p {
                font-size: 0.86rem;
            }

            .cgt-card {
                min-height: auto;
                padding: 0.62rem 0.65rem;
                border-radius: 12px;
            }

            .cgt-card-label {
                font-size: 0.72rem;
            }

            .cgt-card-value {
                font-size: 1.02rem;
            }

            .cgt-card-note {
                font-size: 0.72rem;
            }

            .cgt-chip-wrap {
                gap: 0.25rem;
                margin-bottom: 0.35rem;
            }

            .cgt-chip {
                font-size: 0.69rem;
                padding: 0.14rem 0.45rem;
            }

            .stTabs [data-baseweb="tab-list"] {
                flex-wrap: wrap;
                gap: 0.28rem;
            }

            .stTabs [data-baseweb="tab"] {
                height: auto;
                min-height: 0;
                padding: 0.3rem 0.55rem;
            }

            .stTabs [data-baseweb="tab"] p {
                font-size: 0.8rem;
            }

            div[data-testid="stHorizontalBlock"] {
                gap: 0.5rem;
            }

            div[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] {
                gap: 0.35rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def inject_density_mode_styles(compact_mode: bool) -> None:
    if not compact_mode:
        return

    st.markdown(
        """
        <style>
        div[data-testid="stDataFrame"] table {
            font-size: 0.84rem;
        }

        div[data-testid="stDataFrame"] th,
        div[data-testid="stDataFrame"] td {
            padding-top: 0.2rem !important;
            padding-bottom: 0.2rem !important;
        }

        .cgt-card {
            min-height: auto;
            padding: 0.58rem 0.62rem;
        }

        .cgt-card-label {
            font-size: 0.72rem;
        }

        .cgt-card-value {
            font-size: 1.02rem;
        }

        .cgt-card-note {
            font-size: 0.7rem;
        }

        .cgt-chip {
            font-size: 0.7rem;
            padding: 0.12rem 0.4rem;
        }

        div[data-testid="stHorizontalBlock"] {
            gap: 0.4rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_section_intro(text: str) -> None:
    st.markdown(f'<div class="cgt-section-intro"><p>{text}</p></div>', unsafe_allow_html=True)


def render_stat_cards(cards: Sequence[dict[str, str]], columns: int = 4) -> None:
    if not cards:
        return

    cols = st.columns(columns)
    for idx, card in enumerate(cards):
        with cols[idx % columns]:
            label = str(card.get("label", "")).strip()
            value = str(card.get("value", "")).strip()
            note = str(card.get("note", "")).strip()
            tone = str(card.get("tone", "neutral")).strip().lower()
            tone_cls = ""
            if tone == "positive":
                tone_cls = " pos"
            elif tone == "negative":
                tone_cls = " neg"
            note_html = f'<div class="cgt-card-note">{note}</div>' if note else ""
            st.markdown(
                (
                    '<div class="cgt-card">'
                    f'<div class="cgt-card-label">{label}</div>'
                    f'<div class="cgt-card-value{tone_cls}">{value}</div>'
                    f"{note_html}"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )


def render_filter_chips(chips: Sequence[str]) -> None:
    chip_list = [c for c in chips if str(c).strip()]
    if not chip_list:
        return
    chips_html = "".join(f'<span class="cgt-chip">{str(chip)}</span>' for chip in chip_list)
    st.markdown(f'<div class="cgt-chip-wrap">{chips_html}</div>', unsafe_allow_html=True)


def render_welcome_banner() -> None:
    st.markdown("## 🧭 Your CGT workspace")
    st.caption("Upload your trades, review the summary, and explore the details without losing any of the underlying analysis.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("1. **Import your data**")
        st.caption("Add broker CSVs and optional missing-lot files.")
    with col2:
        st.markdown("2. **Review the overview**")
        st.caption("Check the headline figures before diving into the tables.")
    with col3:
        st.markdown("3. **Explore the details**")
        st.caption("Use positions, history, and what-if analysis for deeper review.")


def render_main_sidebar() -> SidebarState:
    with st.sidebar:
        st.markdown("### 1️⃣ Import your data")
        st.caption("Upload one or more broker CSVs to get started.")
        uploads = st.file_uploader("Broker CSV file(s)", type=["csv"], accept_multiple_files=True, label_visibility="collapsed")

        with st.expander("2️⃣ Add missing lots", expanded=True):
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

        with st.expander("3️⃣ Choose summary columns", expanded=False):
            show_bf_used = st.checkbox("B/F Loss Used (EUR)", value=False)
            show_ex_used = st.checkbox("Exemption Used (EUR)", value=False)
            show_carry_fw = st.checkbox("Carry Forward (EUR)", value=False)
            show_cashflow = st.checkbox("Net Cashflow (EUR)", value=False)
            show_total_fees = st.checkbox("Total Fees (EUR)", value=False)
            compact_mode = st.checkbox("Compact table density", value=False)

        with st.expander("4️⃣ Adjust tax settings", expanded=False):
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

        with st.expander("5️⃣ Add manual transactions", expanded=False):
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
        compact_mode=compact_mode,
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


def render_cgt1_export_expander(cgt1_df_full: pd.DataFrame, summary_shares: pd.DataFrame | None = None) -> None:
    with st.expander("📄 CGT1 export", expanded=False):
        render_filter_chips(["Disposals only", "CGT1-ready columns", "CSV download"])
        if cgt1_df_full.empty:
            st.info("No disposals to export.")
            return

        _disp = pd.to_datetime(cgt1_df_full["Date Disposed"], errors="coerce")
        years = sorted({d.year for d in _disp.dropna()}, reverse=True)
        year_choice = st.selectbox(
            "Filter by tax year",
            options=( ["All years"] + years),
            index=0,
            key="cgt1_export_year_filter",
        )

        if year_choice == "All years":
            cgt1_df = cgt1_df_full.copy()
            y_token = "ALL"
        else:
            cgt1_df = cgt1_df_full[_disp.dt.year.eq(int(year_choice))].copy()
            y_token = str(year_choice)

        if summary_shares is not None and not summary_shares.empty and "Year" in summary_shares.columns:
            summary_year = pd.to_numeric(summary_shares["Year"], errors="coerce")
            if year_choice == "All years":
                summary_slice = summary_shares.copy()
            else:
                summary_slice = summary_shares.loc[summary_year.eq(float(year_choice))].copy()

            if not summary_slice.empty:
                export_gain = pd.to_numeric(cgt1_df["Gain/Loss (EUR)"], errors="coerce").sum()
                summary_gain = pd.to_numeric(summary_slice.get("Realised Profit / Loss (EUR)"), errors="coerce").sum()
                gain_delta = float(export_gain - summary_gain)
                if abs(gain_delta) > 0.01:
                    st.warning(
                        f"CGT1 export gain/loss differs from annual summary by €{gain_delta:,.2f} for the selected year."
                    )
                else:
                    st.success("CGT1 export matches annual summary gain/loss within a €0.01 tolerance.")

        cgt1_df = cgt1_df.sort_values(by=["CGT Period", "Date Disposed", "Asset Type", "Ticker - Name"], kind="stable")

        total_disposals = int(len(cgt1_df))
        total_gain = float(pd.to_numeric(cgt1_df["Gain/Loss (EUR)"], errors="coerce").fillna(0).sum())
        render_stat_cards(
            [
                {"label": "Rows in export", "value": f"{total_disposals:,}"},
                {
                    "label": "Total gain/loss",
                    "value": f"€{total_gain:,.2f}",
                    "tone": "positive" if total_gain > 0 else ("negative" if total_gain < 0 else "neutral"),
                },
                {"label": "Year filter", "value": str(year_choice)},
            ],
            columns=3,
        )

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

        def _highlight_total_row(row: pd.Series) -> list[str]:
            if str(row.get("Date Acquired", "")).strip() == "Totals":
                return ["background-color: #edf3fb; font-weight: 700;"] * len(row)
            return [""] * len(row)

        st.dataframe(cgt1_preview.style.apply(_highlight_total_row, axis=1), use_container_width=True)
        st.download_button(
            label=f"⬇️ Download CGT1 ({y_token})",
            data=cgt1_df.to_csv(index=False).encode("utf-8"),
            file_name=f"CGT1_{y_token}.csv",
            mime="text/csv",
            use_container_width=True,
        )


def render_form12_export_expander(
    out: pd.DataFrame,
    exit_tax_rate_etf: float,
    build_form12_export_fn: Callable[..., pd.DataFrame],
    summary_etfs: pd.DataFrame | None = None,
) -> None:
    with st.expander("📄 Form 12 export (ETF Exit Tax)", expanded=False):
        render_filter_chips(["ETF events", "Exit tax view", "CSV download"])
        f12_full = build_form12_export_fn(out, exit_tax_rate=exit_tax_rate_etf)

        if f12_full.empty:
            st.info("No ETF disposals or deemed disposals found.")
            return

        d = pd.to_datetime(f12_full["Date"], errors="coerce")
        years = sorted({dt.year for dt in d.dropna()}, reverse=True)
        year_choice = st.selectbox(
            "Filter by tax year",
            options=( ["All years"] + years),
            index=0,
            key="form12_export_year_filter",
        )

        if year_choice == "All years":
            f12_df = f12_full.copy()
            y_token = "ALL"
        else:
            f12_df = f12_full[d.dt.year.eq(int(year_choice))].copy()
            y_token = str(year_choice)

        if summary_etfs is not None and not summary_etfs.empty and "Year" in summary_etfs.columns:
            summary_year = pd.to_numeric(summary_etfs["Year"], errors="coerce")
            if year_choice == "All years":
                summary_slice = summary_etfs.copy()
            else:
                summary_slice = summary_etfs.loc[summary_year.eq(float(year_choice))].copy()

            if not summary_slice.empty:
                export_taxable = pd.to_numeric(f12_df["Taxable Gain (EUR)"], errors="coerce").sum()
                export_tax = pd.to_numeric(f12_df[f"Tax @ {int(exit_tax_rate_etf*100)}% (EUR)"], errors="coerce").sum()
                summary_taxable = pd.to_numeric(summary_slice.get("Taxable Gain (EUR)"), errors="coerce").sum()
                tax_col = f"Tax @ {int(exit_tax_rate_etf*100)}% (EUR)"
                summary_tax = pd.to_numeric(summary_slice.get(tax_col), errors="coerce").sum()
                delta_taxable = float(export_taxable - summary_taxable)
                delta_tax = float(export_tax - summary_tax)
                if abs(delta_taxable) > 0.01 or abs(delta_tax) > 0.01:
                    st.warning(
                        f"Form 12 export differs from ETF summary by {delta_taxable:+.2f} taxable / €{delta_tax:,.2f} tax for the selected year."
                    )
                else:
                    st.success("Form 12 export matches ETF summary within a €0.01 tolerance.")

        tax_col = [c for c in f12_df.columns if c.startswith("Tax @ ") and c.endswith("% (EUR)")]
        tax_col = tax_col[0] if tax_col else None

        total_rows = int(len(f12_df))
        total_tax_due = float(pd.to_numeric(f12_df.get(tax_col or "", 0), errors="coerce").fillna(0).sum()) if tax_col else 0.0
        render_stat_cards(
            [
                {"label": "Rows in export", "value": f"{total_rows:,}"},
                {"label": "Total tax", "value": f"€{total_tax_due:,.2f}"},
                {"label": "Year filter", "value": str(year_choice)},
            ],
            columns=3,
        )

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

        def _highlight_total_row(row: pd.Series) -> list[str]:
            if str(row.get("Chargeable Event", "")).strip() == "Totals":
                return ["background-color: #edf3fb; font-weight: 700;"] * len(row)
            return [""] * len(row)

        st.dataframe(preview.style.apply(_highlight_total_row, axis=1), use_container_width=True)
        st.download_button(
            label=f"⬇️ Download Form 12 (ETF Exit Tax) — {y_token}",
            data=f12_df.to_csv(index=False).encode("utf-8"),
            file_name=f"Form12_ETF_ExitTax_{y_token}.csv",
            mime="text/csv",
            use_container_width=True,
        )


def render_dividend_summary_expander(out: pd.DataFrame, tax_bracket: float, usc_rate: float, prsi_rate: float) -> None:
    with st.expander("💵 Dividend Summary & Tax Calculator", expanded=False):
        render_filter_chips(["Dividend roll-up", "Manual FX aware", "Estimated Irish tax"])
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

        total_div_eur = float(pd.to_numeric(summary["Gross_EUR"], errors="coerce").fillna(0).sum()) if not summary.empty else 0.0
        total_tax = float(summary["Tax_Due_Ireland"].sum())
        render_stat_cards(
            [
                {"label": "Year-currency rows", "value": f"{len(summary):,}"},
                {"label": "Gross dividends (EUR)", "value": f"€{total_div_eur:,.2f}"},
                {"label": "Estimated tax due", "value": f"€{total_tax:,.2f}"},
            ],
            columns=3,
        )

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

        st.success(f"👉 Total Irish dividend tax due: **€{total_tax:,.2f}**")
