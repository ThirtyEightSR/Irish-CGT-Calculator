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
    st.markdown('<div class="cgt-chip-wrap"><span class="cgt-chip">Hypothetical sale simulator</span><span class="cgt-chip">No changes are applied to source data</span></div>', unsafe_allow_html=True)

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

                    top_cards = st.columns(3)
                    with top_cards[0]:
                        st.markdown(
                            (
                                '<div class="cgt-card">'
                                '<div class="cgt-card-label">Hypothetical gain/loss</div>'
                                f'<div class="cgt-card-value{" pos" if hypo_gl > 0 else (" neg" if hypo_gl < 0 else "")}">{fmt(hypo_gl)}</div>'
                                f'<div class="cgt-card-note">{regime}</div>'
                                "</div>"
                            ),
                            unsafe_allow_html=True,
                        )
                    with top_cards[1]:
                        st.markdown(
                            (
                                '<div class="cgt-card">'
                                '<div class="cgt-card-label">YTD tax before</div>'
                                f'<div class="cgt-card-value">{fmt(tax_now)}</div>'
                                f'<div class="cgt-card-note">{tax_title}</div>'
                                "</div>"
                            ),
                            unsafe_allow_html=True,
                        )
                    with top_cards[2]:
                        st.markdown(
                            (
                                '<div class="cgt-card">'
                                '<div class="cgt-card-label">Tax delta from this sale</div>'
                                f'<div class="cgt-card-value{" pos" if delta > 0 else (" neg" if delta < 0 else "")}">{fmt(delta)}</div>'
                                '<div class="cgt-card-note">Difference between before and after</div>'
                                "</div>"
                            ),
                            unsafe_allow_html=True,
                        )

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

    st.markdown("#### Key differences")
    regime_df = pd.DataFrame(
        [
            {
                "Regime": "Shares (CGT)",
                "Tax basis": "Capital gains",
                "Standard rate": "33%",
                "Annual exemption": "€1,270 per person",
                "Loss offset": "Allowed (same-year or carried forward)",
                "When due": "On disposal",
            },
            {
                "Regime": "ETFs (Exit Tax)",
                "Tax basis": "Deemed exit tax on gains",
                "Standard rate": "41%",
                "Annual exemption": "None",
                "Loss offset": "Not applicable",
                "When due": "On gain events (disposals)",
            },
            {
                "Regime": "ETFs (Deemed Disposal)",
                "Tax basis": "Deemed disposal every 8 years",
                "Standard rate": "41%",
                "Annual exemption": "None",
                "Loss offset": "Not applicable",
                "When due": "Every 8 years from acquisition",
            },
            {
                "Regime": "Dividends",
                "Tax basis": "Income tax rules (not CGT)",
                "Standard rate": "N/A here",
                "Annual exemption": "N/A",
                "Loss offset": "N/A",
                "When due": "On receipt (withholding may apply)",
            },
        ]
    )
    st.dataframe(regime_df, use_container_width=True, hide_index=True)
