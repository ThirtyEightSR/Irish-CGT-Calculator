from __future__ import annotations

from typing import Callable

import pandas as pd
import streamlit as st


def render_what_if(
    out: pd.DataFrame | None,
    cgt_rate_shares: float,
    exit_tax_rate_etf: float,
    use_exemption: bool,
    exemption_val: float,
    replay_fifo_lots_all_fn: Callable[[pd.DataFrame], dict],
    available_qty_fn: Callable[[pd.DataFrame, str], float],
    last_known_unit_price_eur_fn: Callable[[pd.DataFrame, str], float | None],
    asset_kind_for_isin_fn: Callable[[pd.DataFrame, str], str],
    fifo_cost_for_sale_fn: Callable[[pd.DataFrame, str, float], float],
    year_today_fn: Callable[[], int],
    ytd_realised_gains_fn: Callable[[pd.DataFrame, int], tuple[float, float]],
    carry_forward_shares_to_year_fn: Callable[[pd.DataFrame, int, bool, float], float],
    tax_shares_delta_fn: Callable[[float, float, float, bool, float, float], tuple[float, float, float]],
    tax_etf_delta_fn: Callable[[float, float, float], tuple[float, float, float]],
) -> None:
    st.markdown("### 🧮 What-if: sell to reduce this year’s tax")

    if out is not None and not out.empty:
        lots_map = replay_fifo_lots_all_fn(out)
        if not lots_map:
            st.info("No current holdings found — nothing to simulate.")
        else:
            latest_names = out.sort_values(by="Date").groupby("ISIN", as_index=False).last()[["ISIN", "Ticker - Name"]]
            latest_names = latest_names.rename(columns={"Ticker - Name": "Name"})
            holding_rows = []
            for isin, lots in lots_map.items():
                held = sum(L["qty"] for L in lots)
                if held <= 1e-12:
                    continue
                nm = latest_names[latest_names["ISIN"].astype(str).eq(str(isin))]
                name = nm["Name"].iloc[0] if not nm.empty else str(isin)
                holding_rows.append({"ISIN": str(isin), "Name": name, "HeldQty": float(held)})

            if not holding_rows:
                st.info("All positions are flat — nothing to simulate.")
            else:
                holdings_df = pd.DataFrame(holding_rows).sort_values(by=["Name", "ISIN"])
                holdings_df["label"] = holdings_df.apply(
                    lambda r: f'{r["Name"]} — {r["ISIN"]} (Qty {r["HeldQty"]:.6f})', axis=1
                )
                choice = str(st.selectbox("Pick a holding:", holdings_df["label"].tolist()))
                picked_isin = choice.split(" — ")[-1].split(" (Qty")[0]

                avail = available_qty_fn(out, picked_isin)
                last_px = last_known_unit_price_eur_fn(out, picked_isin)

                colA, colB = st.columns(2)
                with colA:
                    qty = st.number_input("Units to sell", min_value=0.0, value=float(avail), step=1.0, format="%.6f")
                with colB:
                    price_eur = st.number_input("Price per unit (EUR)", min_value=0.0, value=float(last_px or 0.0), step=0.01)

                if qty <= 0 or price_eur <= 0:
                    st.caption("Enter a positive quantity and price to simulate.")
                else:
                    qty_sim = min(float(qty), float(avail))
                    kind = asset_kind_for_isin_fn(out, picked_isin)
                    cost = fifo_cost_for_sale_fn(out, picked_isin, qty_sim)
                    proceeds = qty_sim * price_eur
                    hypo_gl = proceeds - cost

                    year_now = year_today_fn()
                    shares_ytd_gl, etfs_ytd_gl = ytd_realised_gains_fn(out, year_now)

                    if qty > avail + 1e-9:
                        st.warning(
                            (
                                f"Selected quantity exceeds current holding ({avail:.6f}). "
                                f"Simulation is capped to available holdings ({qty_sim:.6f} units)."
                            )
                        )

                    if kind == "share":
                        carry_in = carry_forward_shares_to_year_fn(out, year_now, use_exemption, exemption_val)
                        tax_now, tax_new, delta = tax_shares_delta_fn(
                            shares_ytd_gl,
                            hypo_gl,
                            carry_in,
                            use_exemption,
                            exemption_val,
                            cgt_rate_shares,
                        )
                        tax_title = f"CGT @ {int(cgt_rate_shares*100)}%"
                        regime = "Shares (CGT)"
                    else:
                        tax_now, tax_new, delta = tax_etf_delta_fn(etfs_ytd_gl, hypo_gl, exit_tax_rate_etf)
                        tax_title = f"Exit Tax @ {int(exit_tax_rate_etf*100)}%"
                        regime = "ETF (Exit Tax)"

                    def fmt(x: float) -> str:
                        return f"€{x:,.2f}"

                    res = pd.DataFrame(
                        [
                            ["Instrument", f"{choice}"],
                            ["Regime", regime],
                            ["Proceeds (EUR)", fmt(proceeds)],
                            ["Cost basis (EUR)", fmt(cost)],
                            ["Hypothetical Gain/Loss (EUR)", fmt(hypo_gl)],
                            [f"YTD {tax_title} (before)", fmt(tax_now)],
                            [f"YTD {tax_title} (after)", fmt(tax_new)],
                            ["Δ Tax from this sale", fmt(delta)],
                        ],
                        columns=["Metric", "Value"],
                    )

                    st.dataframe(res, use_container_width=True, hide_index=True)
    else:
        st.info("Upload and process a CSV first to use the What-if tool.")

    st.markdown(
        (
            "#### Key differences\n\n"
            "| Regime | Tax basis | Standard rate | Annual exemption | Loss offset | When due |\n"
            "|---|---|---|---|---|---|\n"
            "| **Shares (CGT)** | Capital gains | **33%** | **€1,270** per person | "
            "**Allowed** (same-year or carried forward) | On disposal |\n"
            "| **ETFs (Exit Tax)** | Deemed exit tax on gains | **41%** | **None** | "
            "**Not applicable** | On gain events (disposals) |\n"
            "| **ETFs (Deemed Disposal)** | Deemed disposal every 8 years (Fair Market Value input) | "
            "**41%** | **None** | **Not applicable** | Every 8 years from acquisition |\n"
            "| **Dividends** | Income tax rules (not CGT) | N/A here | N/A | N/A | "
            "On receipt (withholding may apply) |\n"
        )
    )
