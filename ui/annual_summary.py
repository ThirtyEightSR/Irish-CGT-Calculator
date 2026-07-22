from __future__ import annotations

import traceback
from typing import Any, Callable

import pandas as pd
import streamlit as st


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

        money_cols = [c for c in df_v.columns if c != "Year"]
        styler = df_v.style.format({c: fmt_money_eur for c in money_cols})

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

        st.dataframe(styler, use_container_width=True)

    tabs = st.tabs(["📈 Shares (CGT)", "🧺 ETFs (Exit Tax)", "➕ Combined (Shares+ETFs)", "💸 Dividends", "⏳ ETFs (Deemed Disposal)"])
    with tabs[0]:
        style_and_show_summary(summary_shares)
    with tabs[1]:
        style_and_show_summary(summary_etfs)
    with tabs[2]:
        style_and_show_summary(summary_combined)
    with tabs[3]:
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
            st.dataframe(per_year.style.format({"Gross": fmt_money, "Tax": fmt_money, "Net": fmt_money}), use_container_width=True)

            st.markdown("**By Ticker**")
            st.dataframe(by_ticker.style.format({"Gross": fmt_money, "Tax": fmt_money, "Net": fmt_money}), use_container_width=True)

            st.markdown("**By Broker (Per Year)**")
            st.dataframe(by_broker_year.style.format({"Gross": fmt_money, "Tax": fmt_money, "Net": fmt_money}), use_container_width=True)

            st.markdown("**Dividend Transactions**")
            tx_cols = ["Date", "Ticker - Name", "ISIN", "Currency", "Total", "Fee", "Order ID"]
            st.dataframe(divs.sort_values(by="Date").loc[:, tx_cols].style.format({"Total": fmt_money, "Fee": fmt_money}), use_container_width=True)

    with tabs[4]:
        planner = None
        est = None

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
                st.info("No ETF lots currently held for ≥ 8 years — nothing to plan yet.")
            else:
                total = len(planner)
                by_year = planner["__year"].value_counts().sort_index()
                st.write(f"Lots hitting deemed disposal (8-year): **{total}**")
                st.dataframe(by_year.rename_axis("Year").reset_index(name="Lots"))

                with st.expander("Lots (planner)"):
                    st.dataframe(planner[["ISIN", "AcquisitionDate", "DeemedDate", "QtyRemaining"]], use_container_width=True)

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
                    .style.format(
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
