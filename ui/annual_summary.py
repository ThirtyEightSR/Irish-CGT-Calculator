from __future__ import annotations

import html
import traceback
from typing import Any, Callable

import pandas as pd
import streamlit as st

from ui.components import render_filter_chips


def render_annual_summary_tabs(
    summary_shares: pd.DataFrame,
    summary_etfs: pd.DataFrame,
    summary_combined: pd.DataFrame,
    out: pd.DataFrame,
    show_bf_used: bool,
    show_ex_used: bool,
    show_carry_fw: bool,
    show_cashflow: bool,
    show_total_fees: bool,
    fmt_money: Callable[[Any], str],
    fmt_money_eur: Callable[[Any], str],
    deemed_plan_and_estimates_fn: Callable[[pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame]],
    deemed_exit_tax_rate: float,
    is_paid: bool,
    current_year: int | None,
) -> None:
    def style_and_show_summary(df: pd.DataFrame):
        if df.empty:
            st.info("No data for this selection.")
            return

        base_cols = ["Year", "Buys (EUR)", "Sells (EUR)", "Realised Profit / Loss (EUR)", "Taxable Gain (EUR)"]
        tax_cols = [c for c in df.columns if c.startswith("Tax @")]

        optional_cols = []
        if show_bf_used:
            optional_cols.append("B/F Loss Used (EUR)")
        if show_ex_used:
            optional_cols.append("Exemption Used (EUR)")
        if show_carry_fw:
            optional_cols.append("Carry Forward (EUR)")
        if show_cashflow:
            optional_cols.append("Net Cashflow (EUR)")
        if show_total_fees:
            optional_cols.append("Total Fees (EUR)")

        ordered = [c for c in base_cols if c in df.columns] + tax_cols + [c for c in optional_cols if c in df.columns]
        df_v = df[ordered].copy()

        totals = {}
        for col in df_v.columns:
            if col == "Year":
                continue
            totals[col] = float(pd.to_numeric(df_v[col], errors="coerce").fillna(0).sum())
        df_v = pd.concat([df_v, pd.DataFrame([{"Year": "Total", **totals}])], ignore_index=True)
        df_v["Year"] = df_v["Year"].astype(str)

        # Streamlit's styled dataframe does not consistently apply CSS blur filters.
        # For free tier, render a lightweight HTML table so historical values are visibly blurred.
        if not is_paid and current_year is not None:
            df_display = df_v.copy()
            money_cols = [c for c in df_display.columns if c != "Year"]
            for col in money_cols:
                num_series = pd.to_numeric(df_display[col], errors="coerce")
                df_display[col] = num_series.apply(lambda v: fmt_money_eur(v) if pd.notna(v) else "")

            year_num = pd.to_numeric(df_display["Year"], errors="coerce")
            free_locked_rows = year_num.notna() & year_num.ne(float(current_year))

            headers_html = "".join(f"<th>{html.escape(str(col))}</th>" for col in df_display.columns)
            rows_html: list[str] = []
            for idx, row in df_display.iterrows():
                is_total = str(row.get("Year", "")).strip().lower() == "total"
                row_classes = []
                if is_total:
                    row_classes.append("cgt-row-total")
                elif idx % 2 == 1:
                    row_classes.append("cgt-row-zebra")
                row_class_attr = f' class="{" ".join(row_classes)}"' if row_classes else ""

                cell_html: list[str] = []
                is_locked = bool(free_locked_rows.iloc[idx]) if idx < len(free_locked_rows) else False
                for col in df_display.columns:
                    raw_val = "" if pd.isna(row[col]) else str(row[col])
                    safe_val = html.escape(raw_val)
                    if is_locked and col != "Year":
                        safe_val = f'<span class="cgt-blur-cell">{safe_val}</span>'
                    cell_html.append(f"<td>{safe_val}</td>")

                rows_html.append(f"<tr{row_class_attr}>{''.join(cell_html)}</tr>")

            st.markdown(
                (
                    "<style>"
                    ".cgt-free-summary-table{width:100%;border-collapse:collapse;font-size:0.9rem;}"
                    ".cgt-free-summary-table th,.cgt-free-summary-table td{border:1px solid #e2e8f0;padding:0.46rem 0.55rem;text-align:left;}"
                    ".cgt-free-summary-table thead th{background:#f8fafc;color:#334155;font-weight:700;}"
                    ".cgt-free-summary-table .cgt-row-zebra td{background:#fafcfe;}"
                    ".cgt-free-summary-table .cgt-row-total td{background:#edf3fb;font-weight:700;}"
                    ".cgt-blur-cell{display:inline-block;filter:blur(4px);-webkit-filter:blur(4px);user-select:none;}"
                    "</style>"
                    f"<table class='cgt-free-summary-table'><thead><tr>{headers_html}</tr></thead><tbody>{''.join(rows_html)}</tbody></table>"
                ),
                unsafe_allow_html=True,
            )
            return

        money_cols = [c for c in df_v.columns if c != "Year"]
        styler = df_v.style.format({c: fmt_money_eur for c in money_cols})

        def _highlight_total_row(row: pd.Series) -> list[str]:
            if str(row.get("Year", "")).strip().lower() == "total":
                return ["background-color: #edf3fb; font-weight: 700;"] * len(row)
            return [""] * len(row)

        def _zebra_rows(row: pd.Series) -> list[str]:
            if str(row.get("Year", "")).strip().lower() == "total":
                return [""] * len(row)
            base = "background-color: #fafcfe;" if row.name % 2 else ""
            return [base] * len(row)

        def pl_color(val):
            if isinstance(val, str) or pd.isna(val):
                return ""
            if val > 0:
                return "color: green; font-weight: 600;"
            if val < 0:
                return "color: red; font-weight: 600;"
            return ""

        if "Realised Profit / Loss (EUR)" in df_v.columns:
            styler = styler.map(pl_color, subset=["Realised Profit / Loss (EUR)"])

        styler = styler.apply(_zebra_rows, axis=1)
        styler = styler.apply(_highlight_total_row, axis=1)
        st.dataframe(styler, use_container_width=True)

    tab_labels = ["➕ Combined (Shares+ETFs)", "📈 Shares (CGT)", "🧺 ETFs (Exit Tax)", "💸 Dividends", "⏳ ETFs (Deemed Disposal)"]
    if not is_paid:
        tab_labels = [
            "➕ Combined (Shares+ETFs)",
            "🔒 Shares (Paid)",
            "🔒 ETFs (Paid)",
            "🔒 Dividends (Paid)",
            "🔒 Deemed Disposal (Paid)",
        ]
    tabs = st.tabs(tab_labels)
    with tabs[0]:
        render_filter_chips(["Combined annual view", "Shares + ETFs", "Totals row highlighted"])
        style_and_show_summary(summary_combined)
    with tabs[1]:
        if not is_paid:
            st.info("Shares annual detail is available on the paid tier.")
        else:
            render_filter_chips(["Shares-only annual view", "CGT regime", "Totals row highlighted"])
            style_and_show_summary(summary_shares)
    with tabs[2]:
        if not is_paid:
            st.info("ETF annual detail is available on the paid tier.")
        else:
            render_filter_chips(["ETF annual view", "Exit tax regime", "Totals row highlighted"])
            style_and_show_summary(summary_etfs)
    with tabs[3]:
        if not is_paid:
            st.info("Dividend breakdown is available on the paid tier.")
        else:
            render_filter_chips(["Dividend aggregates", "By year / ticker / broker"])
            st.subheader("Dividend Summary")
            divs = out[out["Type"].eq("Dividend")].copy()
            if divs.empty:
                st.info("No dividends found in this file.")
            else:
                divs["Gross"] = pd.to_numeric(divs["Total"], errors="coerce").fillna(0).abs()
                divs["TaxAmt"] = pd.to_numeric(divs["Fee"], errors="coerce").fillna(0)
                divs["Currency"] = divs.get("Currency", "EUR").fillna("EUR").astype(str).str.upper().str.strip()
                divs["Currency"] = divs["Currency"].replace({"": "EUR", "NAN": "EUR", "NONE": "EUR"})
                divs["Year"] = pd.to_datetime(divs["Date"]).dt.year

                per_year = (
                    divs.groupby(["Year", "Currency"], dropna=False)
                    .agg(Gross=("Gross", "sum"), Tax=("TaxAmt", "sum"))
                    .reset_index()
                    .sort_values(by=["Year", "Currency"], ascending=[False, True])
                )
                per_year["Net"] = per_year["Gross"] - per_year["Tax"]
                per_year["Year"] = per_year["Year"].astype(str)

                by_ticker = (
                    divs.groupby(["Ticker - Name", "ISIN", "Currency"], dropna=False)
                    .agg(Gross=("Gross", "sum"), Tax=("TaxAmt", "sum"), Payments=("Date", "count"))
                    .reset_index()
                    .sort_values(by="Gross", ascending=False)
                )
                by_ticker["Net"] = by_ticker["Gross"] - by_ticker["Tax"]

                broker_col = "__Broker" if "__Broker" in divs.columns else None
                if broker_col is None:
                    divs["Broker"] = "UNKNOWN"
                    broker_col = "Broker"
                else:
                    divs["Broker"] = divs[broker_col].fillna("UNKNOWN").astype(str).str.strip().replace({"": "UNKNOWN"})
                by_broker_year = (
                    divs.groupby(["Year", "Broker"], dropna=False)
                    .agg(Gross=("Gross", "sum"), Tax=("TaxAmt", "sum"), Payments=("Date", "count"))
                    .reset_index()
                    .sort_values(by=["Year", "Gross"], ascending=[False, False])
                )
                by_broker_year["Net"] = by_broker_year["Gross"] - by_broker_year["Tax"]
                by_broker_year["Year"] = by_broker_year["Year"].astype(str)

                st.markdown("**Per Year**")
                st.dataframe(
                    per_year.style.format({"Gross": fmt_money, "Tax": fmt_money, "Net": fmt_money}),
                    use_container_width=True,
                )

                st.markdown("**By Ticker**")
                st.dataframe(
                    by_ticker.style.format({"Gross": fmt_money, "Tax": fmt_money, "Net": fmt_money}),
                    use_container_width=True,
                )

                st.markdown("**By Broker (Per Year)**")
                st.dataframe(
                    by_broker_year.style.format({"Gross": fmt_money, "Tax": fmt_money, "Net": fmt_money}),
                    use_container_width=True,
                )

                st.markdown("**Dividend Transactions**")
                tx_cols = ["Date", "Ticker - Name", "ISIN", "Currency", "Total", "Fee", "Order ID"]
                st.dataframe(divs.sort_values(by="Date").loc[:, tx_cols].style.format({"Total": fmt_money, "Fee": fmt_money}), use_container_width=True)

    with tabs[4]:
        if not is_paid:
            st.info("ETF deemed-disposal planner is available on the paid tier.")
            return

        render_filter_chips(["8-year ETF rule", "Upcoming lots highlighted", "Manual FMV input"])
        planner = None
        est = None
        today = pd.Timestamp.today().normalize()
        upcoming_cutoff = today + pd.Timedelta(days=365)

        def _deemed_status(deemed_date: object) -> str:
            dt = pd.to_datetime(deemed_date, errors="coerce")
            if pd.isna(dt):
                return "Unknown"
            dt = pd.Timestamp(dt).normalize()
            if dt < today:
                return "Past due"
            if dt <= upcoming_cutoff:
                return "Upcoming (12m)"
            return "Later"

        def _highlight_upcoming_row(row: pd.Series) -> list[str]:
            if row.get("Status") == "Upcoming (12m)":
                return ["background-color: #fff3cd"] * len(row)
            return [""] * len(row)

        if out is not None and not out.empty:
            has_etfs = out.get("Asset") is not None and out["Asset"].astype(str).str.lower().eq("etf").any()
            if has_etfs:
                try:
                    with st.spinner("Building ETF deemed-disposal planner & estimate…"):
                        planner, est = deemed_plan_and_estimates_fn(out)
                except Exception as e:
                    st.warning(f"Deemed-disposal generation failed: {e}")
                    st.code(traceback.format_exc())
            else:
                st.info("No ETF positions found — deemed-disposal not applicable.")

        st.subheader("ETF Deemed Disposal — Planner & Estimator")
        if planner is None or est is None:
            st.caption("Upload a DEGIRO CSV with ETF transactions to generate the planner and estimate.")
        else:
            if planner.empty:
                st.info("No ETF lots currently open for deemed-disposal planning.")
            else:
                total = len(planner)
                by_year = planner["__year"].value_counts().sort_index()
                status_counts = planner.assign(Status=planner["DeemedDate"].apply(_deemed_status))["Status"].value_counts()
                upcoming_count = int(status_counts.get("Upcoming (12m)", 0))
                st.write(f"Open ETF lots in deemed-disposal schedule: **{total}**  |  Upcoming in 12 months: **{upcoming_count}**")
                st.dataframe(by_year.rename_axis("Year").reset_index(name="Lots"))
                st.caption("Status legend: Upcoming (12m) is highlighted in yellow.")

                with st.expander("Lots (planner)"):
                    planner_view = planner[["ISIN", "AcquisitionDate", "DeemedDate", "QtyRemaining"]].copy()
                    planner_view["Status"] = planner_view["DeemedDate"].apply(_deemed_status)
                    st.dataframe(
                        planner_view.style.apply(_highlight_upcoming_row, axis=1).format(
                            {
                                "QtyRemaining": lambda x: "" if pd.isna(x) else f"{float(x):.6f}".rstrip("0").rstrip("."),
                            }
                        ),
                        use_container_width=True,
                    )

            if est is None or est.empty:
                st.info("No proposed valuations could be derived yet.")
            else:
                st.markdown("**Enter today’s price (per unit, EUR) for each ETF ISIN.**")

                per_isin = (
                    est.groupby("ISIN", dropna=False)["ProposedFMV_UnitEUR"]
                    .median()
                    .rename("Suggested Price (Unit EUR)")
                    .reset_index()
                )

                price_key = "deemed_today_prices"
                if price_key not in st.session_state:
                    st.session_state[price_key] = per_isin.assign(**{"Today’s Price (Unit EUR)": per_isin["Suggested Price (Unit EUR)"]})

                merged = per_isin.merge(st.session_state[price_key][["ISIN", "Today’s Price (Unit EUR)"]], on="ISIN", how="left")
                merged["Today’s Price (Unit EUR)"] = merged["Today’s Price (Unit EUR)"].fillna(merged["Suggested Price (Unit EUR)"])
                st.session_state[price_key] = merged

                price_inputs = st.data_editor(
                    st.session_state[price_key][["ISIN", "Suggested Price (Unit EUR)", "Today’s Price (Unit EUR)"]],
                    use_container_width=True,
                    key="deemed_today_prices_editor",
                    column_config={
                        "ISIN": st.column_config.TextColumn("ISIN", disabled=True),
                        "Suggested Price (Unit EUR)": st.column_config.NumberColumn("Suggested Price (Unit EUR)", format="€%.4f", disabled=True),
                        "Today’s Price (Unit EUR)": st.column_config.NumberColumn(
                            "Today’s Price (Unit EUR)",
                            format="€%.4f",
                            help="Enter the current unit price in EUR for this ETF.",
                        ),
                    },
                )

                price_inputs["Today’s Price (Unit EUR)"] = pd.to_numeric(price_inputs["Today’s Price (Unit EUR)"], errors="coerce")
                price_inputs["__unit_price"] = price_inputs["Today’s Price (Unit EUR)"].where(
                    price_inputs["Today’s Price (Unit EUR)"].notna(), price_inputs["Suggested Price (Unit EUR)"]
                )
                price_map = dict(zip(price_inputs["ISIN"], price_inputs["__unit_price"]))

                est_view = est.copy().rename(columns={"UnitCostEUR": "Unit Cost (EUR)"})
                est_view["Status"] = est_view["DeemedDate"].apply(_deemed_status)
                est_view["Fair Market Value (Unit EUR)"] = est_view["ISIN"].map(price_map)
                est_view["Fair Market Value (EUR)"] = est_view["Fair Market Value (Unit EUR)"] * est_view["QtyRemaining"]
                est_view["Estimated Gain (EUR)"] = est_view["Fair Market Value (EUR)"] - (
                    est_view["Unit Cost (EUR)"] * est_view["QtyRemaining"]
                )
                est_view["Estimated Exit Tax (EUR)"] = est_view["Estimated Gain (EUR)"].clip(lower=0) * deemed_exit_tax_rate

                show_cols = [
                    "ISIN",
                    "AcquisitionDate",
                    "DeemedDate",
                    "Status",
                    "QtyRemaining",
                    "Unit Cost (EUR)",
                    "Fair Market Value (Unit EUR)",
                    "Fair Market Value (EUR)",
                    "Estimated Gain (EUR)",
                    "Estimated Exit Tax (EUR)",
                ]

                st.markdown("**Calculated results**")
                st.dataframe(
                    est_view[show_cols]
                    .sort_values(by=["DeemedDate", "ISIN", "AcquisitionDate"])
                    .style.apply(_highlight_upcoming_row, axis=1)
                    .format(
                        {
                            "QtyRemaining": lambda x: "" if pd.isna(x) else f"{float(x):.6f}".rstrip("0").rstrip("."),
                            "Unit Cost (EUR)": fmt_money,
                            "Fair Market Value (Unit EUR)": fmt_money,
                            "Fair Market Value (EUR)": fmt_money,
                            "Estimated Gain (EUR)": fmt_money,
                            "Estimated Exit Tax (EUR)": fmt_money,
                        }
                    ),
                    use_container_width=True,
                )

                deemed_year = pd.to_datetime(est_view["DeemedDate"]).dt.year
                roll = (
                    est_view.assign(__year=deemed_year)
                    .groupby("__year", dropna=False)[["Fair Market Value (EUR)", "Estimated Gain (EUR)", "Estimated Exit Tax (EUR)"]]
                    .sum(min_count=1)
                    .reset_index()
                    .rename(columns={"__year": "Year"})
                )
                if not roll.empty:
                    st.markdown("**Summary by deemed year**")
                    st.dataframe(
                        roll.style.format(
                            {
                                "Fair Market Value (EUR)": fmt_money,
                                "Estimated Gain (EUR)": fmt_money,
                                "Estimated Exit Tax (EUR)": fmt_money,
                            }
                        ),
                        use_container_width=True,
                    )

                st.caption(
                    "Fair Market Value = the value you use for deemed disposal. "
                    "Enter **today’s unit price in EUR** per ETF above; it will be applied to all lots of that ISIN. "
                    "Exit Tax is applied at 41% to gains only."
                )
