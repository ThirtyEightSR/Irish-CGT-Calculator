from __future__ import annotations

import re
import base64
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
import streamlit as st

from core.settings import DEFAULT_CGT_EXEMPTION_EUR, DEFAULT_CGT_RATE_SHARES, DEFAULT_DIRT_RATE_DEPOSIT, DEFAULT_EXIT_TAX_RATE_ETF


@dataclass
class SidebarState:
    uploads: list[Any]
    opening_lots_df: pd.DataFrame | None


@dataclass
class DividendTaxState:
    tax_bracket: float
    usc_rate: float
    prsi_rate: float


@dataclass
class DisplaySettingsState:
    show_bf_used: bool
    show_ex_used: bool
    show_carry_fw: bool
    show_cashflow: bool
    show_total_fees: bool
    compact_mode: bool


@dataclass
class TaxSettingsState:
    use_exemption: bool
    exemption_val: float
    cgt_rate_shares: float
    exit_tax_rate_etf: float
    dirt_rate_deposit: float


@dataclass
class TierState:
    tier: str
    is_paid: bool


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

        .cgt-card.variant-blue {
            background: linear-gradient(180deg, #f3f8ff 0%, #edf5ff 100%);
            border-color: #c7dbf7;
            box-shadow: 0 2px 8px rgba(25, 78, 145, 0.08);
        }

        .cgt-card.variant-green {
            background: linear-gradient(180deg, #f2fbf7 0%, #eaf8f1 100%);
            border-color: #bde8d3;
            box-shadow: 0 2px 8px rgba(21, 120, 76, 0.08);
        }

        .cgt-card.variant-red {
            background: linear-gradient(180deg, #fff4f4 0%, #ffeded 100%);
            border-color: #f0c5c5;
            box-shadow: 0 2px 8px rgba(166, 27, 27, 0.08);
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

        .cgt-hero {
            background: linear-gradient(120deg, #0f3f74 0%, #1b5e99 52%, #1f7aae 100%);
            border-radius: 16px;
            padding: 1rem 1rem 0.9rem 1rem;
            margin: 0.15rem 0 0.85rem 0;
            box-shadow: 0 8px 24px rgba(11, 39, 78, 0.18);
            border: 1px solid rgba(255, 255, 255, 0.16);
        }

        .cgt-hero-title {
            color: #ffffff;
            font-size: 1.15rem;
            font-weight: 800;
            letter-spacing: 0.01em;
            margin: 0 0 0.65rem 0;
        }

        .cgt-hero-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.5rem;
        }

        .cgt-hero-pill {
            background: rgba(255, 255, 255, 0.17);
            border: 1px solid rgba(255, 255, 255, 0.24);
            border-radius: 12px;
            padding: 0.55rem 0.62rem;
        }

        .cgt-hero-pill strong {
            color: #ffffff;
            font-size: 1.18rem;
        }

        .cgt-hero-pill span {
            display: block;
            color: #e7f4ff;
            font-size: 1rem;
            margin-top: 0.08rem;
            line-height: 1.25;
        }

        .cgt-platform-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.7rem;
            margin: 0.2rem 0 0.45rem 0;
        }

        .cgt-platform-card {
            background: #ffffff;
            border: 1px solid #d8e1ec;
            border-radius: 12px;
            padding: 0.75rem 0.82rem;
            box-shadow: 0 1px 0 rgba(10, 30, 60, 0.03);
        }

        .cgt-logo-card {
            background: #ffffff;
            border: 1px solid #d8e1ec;
            border-radius: 12px;
            padding: 0.6rem;
            height: 132px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 0.55rem;
        }

        .cgt-logo-card img {
            max-width: 94%;
            max-height: 104px;
            width: auto;
            height: auto;
            object-fit: contain;
            display: block;
        }

        .cgt-logo-fallback {
            color: #354861;
            font-size: 0.9rem;
            font-weight: 700;
            letter-spacing: 0.03em;
            text-transform: uppercase;
        }

        .cgt-guide-card {
            background: #ffffff;
            border: 1px solid #d8e1ec;
            border-radius: 12px;
            padding: 0.75rem 0.82rem;
            box-shadow: 0 1px 0 rgba(10, 30, 60, 0.03);
        }

        .cgt-platform-card ol,
        .cgt-guide-card ul {
            margin: 0;
            padding-left: 1.05rem;
            color: #334a62;
            font-size: 0.82rem;
            line-height: 1.34;
        }

        .cgt-guide-checklist {
            margin-top: 0.2rem;
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

            .cgt-hero {
                border-radius: 14px;
                padding: 0.78rem 0.75rem 0.72rem 0.75rem;
                margin-bottom: 0.65rem;
            }

            .cgt-hero-title {
                font-size: 1rem;
                margin-bottom: 0.52rem;
            }

            .cgt-hero-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 0.42rem;
            }

            .cgt-hero-pill {
                padding: 0.5rem 0.52rem;
            }

            .cgt-hero-pill strong {
                font-size: 1.02rem;
            }

            .cgt-hero-pill span {
                font-size: 0.88rem;
            }

            .cgt-platform-grid {
                gap: 0.5rem;
                grid-template-columns: 1fr;
            }

            .cgt-platform-card {
                padding: 0.62rem 0.66rem;
            }

            .cgt-logo-card {
                height: 104px;
                padding: 0.45rem;
            }

            .cgt-logo-card img {
                max-height: 78px;
                max-width: 95%;
            }

            .cgt-platform-card ol,
            .cgt-guide-card ul {
                font-size: 0.77rem;
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


def render_stat_cards(cards: Sequence[dict[str, Any]], columns: int = 4) -> None:
    if not cards:
        return

    cols = st.columns(columns)
    for idx, card in enumerate(cards):
        with cols[idx % columns]:
            label = str(card.get("label", "")).strip()
            value = str(card.get("value", "")).strip()
            note = str(card.get("note", "")).strip()
            tone = str(card.get("tone", "neutral")).strip().lower()
            variant = str(card.get("variant", "")).strip().lower()
            tone_cls = ""
            if tone == "positive":
                tone_cls = " pos"
            elif tone == "negative":
                tone_cls = " neg"

            variant_map = {
                "blue": " variant-blue",
                "green": " variant-green",
                "red": " variant-red",
            }
            variant_cls = variant_map.get(variant, "")

            note_html = f'<div class="cgt-card-note">{note}</div>' if note else ""
            st.markdown(
                (
                    f'<div class="cgt-card{variant_cls}">'
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


def _logo_data_uri(path: Path) -> str:
    raw = path.read_bytes()
    suffix = path.suffix.lower()
    if suffix == ".svg":
        mime = "image/svg+xml"
    elif suffix == ".png":
        mime = "image/png"
    elif suffix in {".jpg", ".jpeg"}:
        mime = "image/jpeg"
    elif suffix == ".webp":
        mime = "image/webp"
    else:
        mime = "application/octet-stream"
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _first_existing_logo(repo_root: Path, stem: str) -> Path | None:
    for ext in [".svg", ".png", ".jpg", ".jpeg", ".webp"]:
        candidate = repo_root / "assets" / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def render_welcome_banner(expand_guide: bool = True) -> None:
    st.markdown(
        (
            '<div class="cgt-hero">'
            '<div class="cgt-hero-title">⚡ What you can do in 60 seconds</div>'
            '<div class="cgt-hero-grid">'
            '<div class="cgt-hero-pill"><strong>📥 Import fast</strong><span>Drop broker CSVs and go.</span></div>'
            '<div class="cgt-hero-pill"><strong>📊 See tax now</strong><span>Instant current-year summary.</span></div>'
            '<div class="cgt-hero-pill"><strong>🧾 Prep filings</strong><span>Generate export-ready tax views.</span></div>'
            '<div class="cgt-hero-pill"><strong>🧠 Model outcomes</strong><span>Run what-if sale scenarios.</span></div>'
            "</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    with st.expander("Step-by-step export guide (DEGIRO + Trading 212)", expanded=expand_guide):
        repo_root = Path(__file__).resolve().parent.parent
        degiro_logo = _first_existing_logo(repo_root, "degiro_logo")
        t212_logo = _first_existing_logo(repo_root, "trading212_logo")

        degiro_html = '<div class="cgt-logo-fallback">DEGIRO</div>'
        t212_html = '<div class="cgt-logo-fallback">TRADING 212</div>'
        if degiro_logo is not None:
            degiro_html = f'<img src="{_logo_data_uri(degiro_logo)}" alt="DEGIRO logo" />'
        if t212_logo is not None:
            t212_html = f'<img src="{_logo_data_uri(t212_logo)}" alt="Trading 212 logo" />'

        st.markdown(
            (
                '<div class="cgt-platform-grid">'
                '<div class="cgt-platform-card">'
                f'<div class="cgt-logo-card">{degiro_html}</div>'
                '<ol>'
                '<li>Open DEGIRO and go to Activity &gt; Account statement.</li>'
                '<li>Set the date range to include all years you want to analyse.</li>'
                '<li>Export CSV from the statement view.</li>'
                '<li>Repeat for any additional accounts if needed.</li>'
                '<li>Upload the CSV in this app under Import your data.</li>'
                '</ol>'
                '</div>'
                '<div class="cgt-platform-card">'
                f'<div class="cgt-logo-card">{t212_html}</div>'
                '<ol>'
                '<li>Open Trading 212 and go to History.</li>'
                '<li>Use Export to download your account statement as CSV.</li>'
                '<li>Include all available dates for best tax continuity.</li>'
                '<li>Keep the original columns unchanged.</li>'
                '<li>Upload the CSV in this app under Import your data.</li>'
                '</ol>'
                '</div>'
                '</div>'
                '<div class="cgt-guide-card cgt-guide-checklist">'
                '<strong>Before uploading</strong>'
                '<ul>'
                '<li>Do not edit column names in the CSV.</li>'
                '<li>Keep decimal separators exactly as exported.</li>'
                '<li>If you transferred holdings in, add a missing-lots CSV too.</li>'
                '</ul>'
                '</div>'
            ),
            unsafe_allow_html=True,
        )

        st.info("After upload: start in Overview to verify current-year values first, then review detailed tabs.")


def render_main_sidebar() -> SidebarState:
    with st.sidebar:
        st.markdown("### 1️⃣ Import your data")
        st.caption("Upload one or more broker CSVs to get started.")
        st.caption("Need help? Follow the step-by-step broker guide on the home page.")
        uploads = st.file_uploader("Broker CSV file(s)", type=["csv"], accept_multiple_files=True, label_visibility="collapsed")

        with st.expander("2️⃣ Add missing lots", expanded=False):
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

        with st.expander("3️⃣ Add manual transactions", expanded=False):
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

            st.markdown("---")
            st.markdown("**RSU / ESPP guided entry**")
            st.caption(
                "Use this for vest/purchase events. Entries are treated as Buy lots for cost-basis tracking. "
                "Cash payroll withholding is not modeled separately."
            )

            rsu_col1, rsu_col2 = st.columns(2)
            with rsu_col1:
                equity_event = st.selectbox(
                    "Equity event",
                    options=["RSU Vest", "ESPP Purchase"],
                    index=0,
                    key="equity_event_kind",
                )
                equity_date = st.date_input("Event date", value=datetime.today(), key="equity_event_date")
                equity_isin = st.text_input("ISIN (RSU/ESPP)", placeholder="e.g., US0378331005", key="equity_event_isin").strip().upper()
            with rsu_col2:
                equity_product = st.text_input(
                    "Product / Company",
                    placeholder="e.g., Apple Inc",
                    key="equity_event_product",
                )
                equity_qty = st.number_input(
                    "Units vested/purchased",
                    min_value=0.0,
                    step=0.01,
                    format="%.6f",
                    key="equity_event_qty",
                )
                equity_unit = st.number_input(
                    "Cost basis per unit (EUR)",
                    min_value=0.0,
                    step=0.01,
                    format="%.4f",
                    key="equity_event_unit",
                )

            equity_fee = st.number_input(
                "Broker fees (EUR)",
                min_value=0.0,
                step=0.01,
                value=0.0,
                format="%.2f",
                key="equity_event_fee",
            )

            if st.button("➕ Add RSU/ESPP Entry", use_container_width=True):
                if not equity_isin:
                    st.error("❌ ISIN is required")
                elif equity_qty <= 0:
                    st.error("❌ Units must be > 0")
                elif equity_unit <= 0:
                    st.error("❌ Cost basis per unit must be > 0")
                else:
                    total_eur = equity_qty * equity_unit + equity_fee
                    trans = {
                        "Date": equity_date,
                        "Type": "Buy",
                        "ISIN": equity_isin,
                        "Product": equity_product or equity_isin,
                        "Quantity": equity_qty,
                        "Unit_Price_EUR": equity_unit,
                        "Fees": equity_fee,
                        "Total_EUR": total_eur,
                        "Manual_Label": equity_event,
                    }
                    st.session_state.manual_transactions.append(trans)
                    st.success(f"✅ Added {equity_event} entry for {equity_isin}")

    return SidebarState(
        uploads=uploads,
        opening_lots_df=opening_lots_df,
    )


def render_tier_and_settings_menu() -> tuple[TierState, DisplaySettingsState, TaxSettingsState, DividendTaxState]:
    if "tier_mode" not in st.session_state:
        st.session_state.tier_mode = "free"

    if "show_bf_used" not in st.session_state:
        st.session_state.show_bf_used = False
    if "show_ex_used" not in st.session_state:
        st.session_state.show_ex_used = False
    if "show_carry_fw" not in st.session_state:
        st.session_state.show_carry_fw = False
    if "show_cashflow" not in st.session_state:
        st.session_state.show_cashflow = False
    if "show_total_fees" not in st.session_state:
        st.session_state.show_total_fees = False
    if "compact_mode" not in st.session_state:
        st.session_state.compact_mode = False

    if "use_exemption" not in st.session_state:
        st.session_state.use_exemption = True
    if "exemption_val" not in st.session_state:
        st.session_state.exemption_val = float(DEFAULT_CGT_EXEMPTION_EUR)
    if "cgt_rate_shares" not in st.session_state:
        st.session_state.cgt_rate_shares = float(DEFAULT_CGT_RATE_SHARES)
    if "exit_tax_rate_etf" not in st.session_state:
        st.session_state.exit_tax_rate_etf = float(DEFAULT_EXIT_TAX_RATE_ETF)
    if "dirt_rate_deposit" not in st.session_state:
        st.session_state.dirt_rate_deposit = float(DEFAULT_DIRT_RATE_DEPOSIT)

    if "div_tax_income_pct" not in st.session_state:
        st.session_state.div_tax_income_pct = 40.0
    if "div_tax_usc_pct" not in st.session_state:
        st.session_state.div_tax_usc_pct = 8.0
    if "div_tax_prsi_pct" not in st.session_state:
        st.session_state.div_tax_prsi_pct = 4.0

    with st.sidebar:
        st.markdown("---")
        with st.expander("⚙️ Settings", expanded=False):
            st.markdown("### Access tier")
            st.session_state.tier_mode = st.radio(
                "Tier",
                options=["free", "paid"],
                index=0 if st.session_state.tier_mode == "free" else 1,
                horizontal=True,
                key="tier_mode_picker",
                label_visibility="collapsed",
            )

            st.markdown("### Summary display")
            st.session_state.show_bf_used = st.checkbox("B/F Loss Used (EUR)", value=bool(st.session_state.show_bf_used))
            st.session_state.show_ex_used = st.checkbox("Exemption Used (EUR)", value=bool(st.session_state.show_ex_used))
            st.session_state.show_carry_fw = st.checkbox("Carry Forward (EUR)", value=bool(st.session_state.show_carry_fw))
            st.session_state.show_cashflow = st.checkbox("Net Cashflow (EUR)", value=bool(st.session_state.show_cashflow))
            st.session_state.show_total_fees = st.checkbox("Total Fees (EUR)", value=bool(st.session_state.show_total_fees))
            st.session_state.compact_mode = st.checkbox("Compact table density", value=bool(st.session_state.compact_mode))

            st.markdown("### Tax settings")
            st.session_state.use_exemption = st.checkbox(
                "Apply annual CGT exemption (Shares only)", value=bool(st.session_state.use_exemption)
            )
            st.session_state.exemption_val = st.number_input(
                "Exemption amount (EUR)",
                min_value=0.0,
                value=float(st.session_state.exemption_val),
                step=10.0,
            )
            st.session_state.cgt_rate_shares = st.number_input(
                "Shares CGT rate",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state.cgt_rate_shares),
                step=0.01,
            )
            st.session_state.exit_tax_rate_etf = st.number_input(
                "ETFs Exit Tax rate",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state.exit_tax_rate_etf),
                step=0.01,
            )
            st.session_state.dirt_rate_deposit = st.number_input(
                "DIRT rate (deposit interest)",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state.dirt_rate_deposit),
                step=0.01,
            )

            st.markdown("### Dividend tax")
            preset_options = {
                "High-rate (40/8/4)": (40.0, 8.0, 4.0),
                "Standard (20/4/4)": (20.0, 4.0, 4.0),
                "Custom": None,
            }
            preset = st.selectbox("Rate preset", options=list(preset_options.keys()), index=0, key="div_tax_preset")
            if preset != "Custom":
                p_income, p_usc, p_prsi = preset_options[preset]
                st.session_state.div_tax_income_pct = p_income
                st.session_state.div_tax_usc_pct = p_usc
                st.session_state.div_tax_prsi_pct = p_prsi

            st.session_state.div_tax_income_pct = st.number_input(
                "Income tax rate (%)",
                min_value=0.0,
                max_value=60.0,
                value=float(st.session_state.div_tax_income_pct),
                step=0.5,
                key="div_tax_income_input",
            )
            st.session_state.div_tax_usc_pct = st.number_input(
                "USC rate (%)",
                min_value=0.0,
                max_value=20.0,
                value=float(st.session_state.div_tax_usc_pct),
                step=0.1,
                key="div_tax_usc_input",
            )
            st.session_state.div_tax_prsi_pct = st.number_input(
                "PRSI rate (%)",
                min_value=0.0,
                max_value=20.0,
                value=float(st.session_state.div_tax_prsi_pct),
                step=0.1,
                key="div_tax_prsi_input",
            )

    tier_mode = str(st.session_state.tier_mode).strip().lower()
    if tier_mode not in {"free", "paid"}:
        tier_mode = "free"
    tier_state = TierState(tier=tier_mode, is_paid=(tier_mode == "paid"))

    display_state = DisplaySettingsState(
        show_bf_used=bool(st.session_state.show_bf_used),
        show_ex_used=bool(st.session_state.show_ex_used),
        show_carry_fw=bool(st.session_state.show_carry_fw),
        show_cashflow=bool(st.session_state.show_cashflow),
        show_total_fees=bool(st.session_state.show_total_fees),
        compact_mode=bool(st.session_state.compact_mode),
    )
    tax_state = TaxSettingsState(
        use_exemption=bool(st.session_state.use_exemption),
        exemption_val=float(st.session_state.exemption_val),
        cgt_rate_shares=float(st.session_state.cgt_rate_shares),
        exit_tax_rate_etf=float(st.session_state.exit_tax_rate_etf),
        dirt_rate_deposit=float(st.session_state.dirt_rate_deposit),
    )
    div_state = DividendTaxState(
        tax_bracket=float(st.session_state.div_tax_income_pct),
        usc_rate=float(st.session_state.div_tax_usc_pct) / 100.0,
        prsi_rate=float(st.session_state.div_tax_prsi_pct) / 100.0,
    )
    return tier_state, display_state, tax_state, div_state


def render_locked_feature(feature_name: str) -> None:
    st.info(f"{feature_name} is available on the paid tier.")
    st.markdown(
        "Upgrade to paid to unlock full history, exports, diagnostics, positions, and what-if analysis."
    )


def mask_historical_years_for_free(df: pd.DataFrame, current_year: int | None) -> pd.DataFrame:
    if df.empty or current_year is None or "Year" not in df.columns:
        return df

    masked = df.copy()
    years = pd.to_numeric(masked["Year"], errors="coerce")
    mask_rows = years.notna() & years.ne(float(current_year))
    for col in masked.columns:
        if col == "Year":
            continue
        masked.loc[mask_rows, col] = "Blurred on free tier"
    return masked


def render_dividend_fx_menu(out: pd.DataFrame | None) -> None:
    detected_currencies: list[str] = []
    if out is not None and not out.empty:
        div_rows = out[out["Type"] == "Dividend"].copy()
        if not div_rows.empty:
            detected_currencies = div_rows.get("Currency", pd.Series(dtype=object)).dropna().astype(str).str.upper().str.strip().tolist()
            detected_currencies = [c for c in detected_currencies if c and c not in ["NAN", "NONE"]]
            detected_currencies = sorted(set(detected_currencies))

    fx_cols = st.columns([8, 2])
    with fx_cols[1]:
        with st.popover("💱 FX Rates", use_container_width=True):
            st.caption("Used by dividend tax estimation for non-EUR dividends.")
            non_eur_detected = [c for c in detected_currencies if c != "EUR"]

            for curr in non_eur_detected:
                default_rate = float(st.session_state.fx_rates_manual.get(curr, 1.0))
                fx_input = st.number_input(
                    f"{curr} → EUR",
                    min_value=0.01,
                    value=default_rate,
                    step=0.01,
                    format="%.4f",
                    key=f"fx_rate_menu_{curr}",
                )
                st.session_state.fx_rates_manual[curr] = fx_input

            if not non_eur_detected:
                st.caption("No non-EUR dividends detected in current data.")

            custom_ccy = st.text_input(
                "Add custom currency code",
                value="",
                placeholder="e.g. USD",
                key="fx_menu_custom_ccy",
            ).strip().upper()
            if custom_ccy and re.fullmatch(r"[A-Z]{3}", custom_ccy) and custom_ccy != "EUR":
                default_rate = float(st.session_state.fx_rates_manual.get(custom_ccy, 1.0))
                fx_input_custom = st.number_input(
                    f"{custom_ccy} → EUR (manual)",
                    min_value=0.01,
                    value=default_rate,
                    step=0.01,
                    format="%.4f",
                    key=f"fx_rate_menu_custom_{custom_ccy}",
                )
                st.session_state.fx_rates_manual[custom_ccy] = fx_input_custom


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
