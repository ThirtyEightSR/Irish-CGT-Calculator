from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
import pandas as pd
import streamlit as st


def render_open_positions(
    out: pd.DataFrame | None,
    replay_fifo_lots_all_fn: Callable[[pd.DataFrame], dict[str, list[dict[str, Any]]]],
) -> None:
    st.markdown("### 📊 Open Positions (Cost Basis — current holdings only)")

    if isinstance(out, pd.DataFrame) and not out.empty:
        try:
            out_for_positions = out.copy()
            incoming_mask = pd.Series(False, index=out_for_positions.index)
            if "__IncomingTransfer" in out_for_positions.columns:
                incoming_mask = incoming_mask | out_for_positions["__IncomingTransfer"].fillna(False)
            if "Description" in out_for_positions.columns:
                incoming_mask = incoming_mask | out_for_positions["Description"].astype(str).str.contains(
                    "INCOMING TRANSFER", case=False, na=False
                )
            if incoming_mask.any():
                out_for_positions = out_for_positions.loc[~incoming_mask].copy()

            lots_map = replay_fifo_lots_all_fn(out_for_positions)

            if not lots_map:
                st.info("No open positions at the moment.")
            else:
                latest_names = (
                    out.sort_values(by="Date", kind="mergesort")
                    .groupby("ISIN", as_index=False)
                    .last()[["ISIN", "Ticker - Name"]]
                    .rename(columns={"Ticker - Name": "Company"})
                )

                rows = []
                for isin, lots in lots_map.items():
                    qty = float(sum(float(L.get("qty", 0.0)) for L in lots))
                    if qty <= 1e-12:
                        continue

                    total_cost_eur = float(sum(float(L.get("qty", 0.0)) * float(L.get("unit_cost_eur", 0.0)) for L in lots))
                    avg_cost_eur = total_cost_eur / qty if qty > 0 else np.nan

                    raw_ccys = {str(L.get("ccy", "") or "").upper() for L in lots}
                    norm_map = {"GBX": "GBP"}
                    bad_ccys = {"", "NAN", "NONE", "NULL"}
                    ccys = {norm_map.get(c, c) for c in raw_ccys if c not in bad_ccys}

                    native_cost_str = ""
                    if ccys and len(ccys) == 1:
                        native_ccy = ccys.pop()
                        native_vals = []
                        for L in lots:
                            qL = float(L.get("qty", 0.0))
                            uL = float(L.get("unit_cost_native", np.nan))
                            if qL > 0 and not math.isnan(uL):
                                native_vals.append(qL * uL)

                        if native_vals:
                            total_cost_native = float(sum(native_vals))
                            avg_cost_native = total_cost_native / qty if qty > 0 else np.nan
                        else:
                            avg_cost_native = np.nan

                        if not pd.isna(avg_cost_native):
                            native_cost_str = f"{native_ccy} {avg_cost_native:.2f}"

                    nm = latest_names[latest_names["ISIN"].astype(str).eq(isin)]
                    company = nm["Company"].iloc[0] if not nm.empty else isin

                    rows.append(
                        {
                            "Company": company,
                            "ISIN": isin,
                            "Units": qty,
                            "Avg Cost / Unit (Native)": native_cost_str,
                            "Degiro BEP (Native)": native_cost_str,
                            "Avg Cost / Unit (EUR)": avg_cost_eur,
                            "Total Cost (EUR)": total_cost_eur,
                        }
                    )

                if not rows:
                    st.info("No open positions at the moment.")
                else:
                    view = pd.DataFrame(rows).sort_values(by=["Company", "ISIN"])

                    if "Avg Cost / Unit (Native)" in view.columns:
                        s_native = view["Avg Cost / Unit (Native)"]
                        empty_mask = s_native.isna() | s_native.astype(str).str.strip().eq("")
                        if empty_mask.all():
                            view = view.drop(columns=["Avg Cost / Unit (Native)"])

                    def _fmt_qty(x: Any) -> str:
                        if pd.isna(x):
                            return ""
                        return f"{float(x):.6f}".rstrip("0").rstrip(".")

                    def _fmt_eur(x: Any) -> str:
                        if pd.isna(x):
                            return ""
                        return f"€{float(x):,.2f}"

                    styler = view.style
                    if "Units" in view.columns:
                        styler = styler.format({"Units": _fmt_qty})

                    money_cols_eur = [c for c in ["Avg Cost / Unit (EUR)", "Total Cost (EUR)"] if c in view.columns]
                    for c in money_cols_eur:
                        styler = styler.format({c: _fmt_eur})

                    st.dataframe(styler, use_container_width=True)

                    invested = float(pd.to_numeric(view["Total Cost (EUR)"], errors="coerce").fillna(0).sum())
                    st.caption(
                        "Only positions with a positive remaining quantity are shown. "
                        "Average cost is your fee-adjusted cost per unit in EUR. "
                        "Where available, the native-cost column will approximate your "
                        "broker’s BEP (subject to minor rounding). "
                        f"Total invested cost (EUR): **€{invested:,.2f}**"
                    )

        except Exception as e:
            st.warning(f"Open positions view failed: {e}")
    else:
        st.info("Upload and process a CSV to view open positions.")
