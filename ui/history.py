from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
import streamlit as st

_CASH_ISIN_BLOCKLIST = {"NLFLATEXACNT"}
_CASH_NAME_KEYWORDS = ("flatex euro bankaccount", "flatexdegiro bank")


def render_transaction_history(
    out: pd.DataFrame,
    years_sorted: list[int],
    fmt_date: Callable[[Any], str],
    fmt_qty: Callable[[Any], str],
    fmt_money: Callable[[Any], str],
    fmt_money_eur: Callable[[Any], str],
) -> None:
    st.markdown("### 📜 Transaction History")

    years = years_sorted
    year_options = ["All"] + years

    asset_unique = sorted(out.get("Asset", pd.Series([], dtype="object")).dropna().astype(str).str.lower().unique().tolist())
    asset_options = ["All"] + [a.title() if a != "etf" else "ETF" for a in asset_unique]

    broker_unique = []
    if "__Broker" in out.columns:
        broker_unique = sorted(out["__Broker"].dropna().astype(str).unique().tolist())
    broker_options = ["All"] + broker_unique

    source_unique = []
    if "__SourceFile" in out.columns:
        source_unique = sorted(out["__SourceFile"].dropna().astype(str).unique().tolist())
    source_options = ["All"] + source_unique

    cols = st.columns([1, 1, 1, 2])
    with cols[0]:
        year_choice = st.selectbox("Year", options=year_options, index=0)
    with cols[1]:
        asset_choice = st.radio("Asset", options=asset_options, horizontal=True, index=0)
    with cols[2]:
        broker_choice = st.selectbox("Broker", options=broker_options, index=0)
    with cols[3]:
        source_choice = st.selectbox("Source file", options=source_options, index=0)

    filtered = out.copy() if year_choice == "All" else out[out["__year"].eq(year_choice)].copy()

    if asset_choice != "All" and "Asset" in filtered.columns:
        filtered = filtered[filtered["Asset"].astype(str).str.lower() == asset_choice.lower()]
    if broker_choice != "All" and "__Broker" in filtered.columns:
        filtered = filtered[filtered["__Broker"] == broker_choice]
    if source_choice != "All" and "__SourceFile" in filtered.columns:
        filtered = filtered[filtered["__SourceFile"] == source_choice]

    defaults = {
        "show_buys": st.session_state.get("show_buys", True),
        "show_sells": st.session_state.get("show_sells", True),
        "show_dividends": st.session_state.get("show_dividends", False),
        "show_corp": st.session_state.get("show_corp", False),
        "show_fees_interest": st.session_state.get("show_fees_interest", False),
    }

    c1, c2, c3, c4, c5, spacer = st.columns([1.1, 1.1, 1.4, 1.6, 1.8, 2.5])
    with c1:
        show_buys = st.toggle("Buys", value=defaults["show_buys"])
    with c2:
        show_sells = st.toggle("Sells", value=defaults["show_sells"])
    with c3:
        show_dividends = st.toggle("Dividends", value=defaults["show_dividends"])
    with c4:
        show_corp = st.toggle("Corp actions", value=defaults["show_corp"])
    with c5:
        show_fees_interest = st.toggle("Fees & Interest", value=defaults["show_fees_interest"])

    st.caption("Fee shows explicit fees and dividend tax. Interest amounts are shown in Total.")

    st.session_state.show_buys = show_buys
    st.session_state.show_sells = show_sells
    st.session_state.show_dividends = show_dividends
    st.session_state.show_corp = show_corp
    st.session_state.show_fees_interest = show_fees_interest

    hide_types = []
    if not show_buys:
        hide_types += ["Buy"]
    if not show_sells:
        hide_types += ["Sell"]
    if not show_dividends:
        hide_types += ["Dividend", "Dividend Tax", "Scrip dividend"]
    if not show_corp:
        hide_types += ["Stock split", "Product change", "ISIN change"]
    if not show_fees_interest:
        hide_types += ["Fee", "Interest", "Other"]
    if hide_types:
        filtered = filtered[~filtered["Type"].isin(hide_types)]

    show_fee_interest_amount_col = show_fees_interest and {"Type", "Fee", "Total"}.issubset(filtered.columns)
    if show_fee_interest_amount_col:
        fee_num = pd.to_numeric(filtered["Fee"], errors="coerce")
        total_num = pd.to_numeric(filtered["Total"], errors="coerce")
        amount_col = pd.Series(np.nan, index=filtered.index, dtype="float64")
        mask_interest = filtered["Type"].eq("Interest")
        mask_fee_like = filtered["Type"].isin(["Fee", "Dividend Tax"])
        amount_col.loc[mask_interest] = total_num.loc[mask_interest]
        amount_col.loc[mask_fee_like] = fee_num.loc[mask_fee_like]
        filtered["Fee/Interest Amount"] = amount_col

    filtered = filtered.sort_values(by="Date", ascending=False, kind="mergesort")

    display_cols = [
        c
        for c in [
            "Date",
            "Ticker - Name",
            "ISIN",
            "Type",
            "Asset",
            "Currency",
            "Quantity",
            "Price",
            "Fee",
            *(["Fee/Interest Amount"] if show_fee_interest_amount_col else []),
            "Total",
            "Total (EUR)",
            "Total (EUR, fee-adj)",
            "Gain/Loss",
            "__Broker",
            "__SourceFile",
            "Order ID",
        ]
        if c in filtered.columns
    ]

    to_show = filtered.drop(columns=["__year"]).loc[:, display_cols].copy()
    if "Asset" in to_show.columns:
        to_show["Asset"] = to_show["Asset"].astype(str).str.title().replace({"Etf": "ETF"})

    def pl_color(val: Any) -> str:
        if isinstance(val, str) or pd.isna(val):
            return ""
        if val > 0:
            return "color: green; font-weight: 600;"
        if val < 0:
            return "color: red; font-weight: 600;"
        return ""

    styler = to_show.style.format(
        {
            "Date": fmt_date,
            "Quantity": fmt_qty,
            "Price": fmt_money,
            "Fee": fmt_money_eur,
            "Fee/Interest Amount": fmt_money,
            "Total": fmt_money,
            "Total (EUR)": fmt_money_eur,
            "Total (EUR, fee-adj)": fmt_money_eur,
            "Gain/Loss": fmt_money_eur,
        }
    )
    if "Gain/Loss" in to_show.columns:
        styler = styler.map(pl_color, subset=["Gain/Loss"])

    st.dataframe(styler, use_container_width=True)

    st.markdown("#### View full trade history")

    if isinstance(out, pd.DataFrame) and not out.empty:
        name_col = "Ticker - Name" if "Ticker - Name" in out.columns else ("Product" if "Product" in out.columns else None)

        if name_col and "ISIN" in out.columns:
            ins = out.loc[:, ["ISIN", name_col]].dropna(subset=["ISIN", name_col]).astype({"ISIN": str, name_col: str})
            ins = ins[(ins["ISIN"].str.strip() != "") & (ins[name_col].str.strip() != "")]
            # Exclude broker cash-account pseudo instruments from drilldown.
            ins = ins[~ins["ISIN"].str.upper().isin(_CASH_ISIN_BLOCKLIST)]
            ins = ins[~ins[name_col].str.lower().str.contains("|".join(_CASH_NAME_KEYWORDS), na=False)]
            ins = ins.drop_duplicates().copy().rename(columns={name_col: "Ticker - Name"})

            if ins.empty:
                st.info("No instruments found (ISIN or name missing in the data).")
            else:
                ins["label"] = ins["Ticker - Name"] + " — " + ins["ISIN"]
                ins = ins.sort_values(by="label")
                labels = ["(none)"] + ins["label"].tolist()
                choice = st.selectbox("Select an instrument:", options=labels, index=0)

                if choice != "(none)":
                    picked_isin = choice.rsplit(" — ", 1)[-1]
                    detail = out[out["ISIN"] == picked_isin].drop(columns=["__year"], errors="ignore").copy()
                    if detail.empty:
                        st.info("No trade rows found for the selected instrument.")
                        return

                    sort_keys = [c for c in ["Date", "Type", "Order ID"] if c in detail.columns]
                    if sort_keys:
                        detail = detail.sort_values(by=sort_keys, kind="mergesort")

                    cols_pref = [
                        "Date",
                        "Ticker - Name",
                        "ISIN",
                        "Type",
                        "Asset",
                        "Currency",
                        "Quantity",
                        "Price",
                        "Fee",
                        "Total",
                        "Total (EUR)",
                        "Total (EUR, fee-adj)",
                        "Gain/Loss",
                        "Order ID",
                        "Description",
                    ]
                    cols = [c for c in cols_pref if c in detail.columns]
                    if cols:
                        detail = detail.loc[:, cols]

                    detail_show_fee_interest_amount_col = {"Type", "Fee", "Total"}.issubset(detail.columns) and detail[
                        "Type"
                    ].isin(["Fee", "Interest", "Dividend Tax"]).any()
                    if detail_show_fee_interest_amount_col:
                        fee_num = pd.to_numeric(detail["Fee"], errors="coerce")
                        total_num = pd.to_numeric(detail["Total"], errors="coerce")
                        amount_col = pd.Series(np.nan, index=detail.index, dtype="float64")
                        mask_interest = detail["Type"].eq("Interest")
                        mask_fee_like = detail["Type"].isin(["Fee", "Dividend Tax"])
                        amount_col.loc[mask_interest] = total_num.loc[mask_interest]
                        amount_col.loc[mask_fee_like] = fee_num.loc[mask_fee_like]
                        detail["Fee/Interest Amount"] = amount_col
                        cols_pref = [
                            "Date",
                            "Ticker - Name",
                            "ISIN",
                            "Type",
                            "Asset",
                            "Currency",
                            "Quantity",
                            "Price",
                            "Fee",
                            *(["Fee/Interest Amount"] if detail_show_fee_interest_amount_col else []),
                            "Total",
                            "Total (EUR)",
                            "Total (EUR, fee-adj)",
                            "Gain/Loss",
                            "Order ID",
                            "Description",
                        ]
                        cols = [c for c in cols_pref if c in detail.columns]
                        detail = detail.loc[:, cols]

                    styler = detail.style.format(
                        {
                            "Date": fmt_date,
                            "Quantity": fmt_qty,
                            "Price": fmt_money,
                            "Fee": fmt_money,
                            "Fee/Interest Amount": fmt_money,
                            "Total": fmt_money,
                            "Total (EUR)": fmt_money,
                            "Total (EUR, fee-adj)": fmt_money,
                            "Gain/Loss": fmt_money,
                        }
                    )
                    st.dataframe(styler, use_container_width=True)
        else:
            st.info("Your data is missing either ISIN or an instrument name column.")
    else:
        st.info("Upload a CSV to enable trade history.")
