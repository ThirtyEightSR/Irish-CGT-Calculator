from __future__ import annotations

from typing import Callable

import pandas as pd
import streamlit as st


def _as_year_int(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def _by_year(df: pd.DataFrame, col: str) -> pd.Series:
    if df.empty or "Year" not in df.columns or col not in df.columns:
        return pd.Series(dtype="float64")
    temp = df.loc[:, ["Year", col]].copy()
    temp["Year"] = _as_year_int(temp["Year"])
    temp[col] = pd.to_numeric(temp[col], errors="coerce").fillna(0.0)
    temp = temp.dropna(subset=["Year"])
    if temp.empty:
        return pd.Series(dtype="float64")
    return temp.groupby("Year")[col].sum()


def render_tax_reconciliation_debug(
    summary_shares: pd.DataFrame,
    summary_etfs: pd.DataFrame,
    summary_combined: pd.DataFrame,
    cgt_rate_shares: float,
    exit_tax_rate_etf: float,
    fmt_money_eur: Callable[[object], str],
) -> None:
    with st.expander("🧮 Tax Reconciliation (Debug)", expanded=False):
        share_tax_col = f"Tax @ {int(cgt_rate_shares * 100)}% (EUR)"
        etf_tax_col = f"Tax @ {int(exit_tax_rate_etf * 100)}% (EUR)"

        shares_pl = _by_year(summary_shares, "Realised Profit / Loss (EUR)")
        shares_bf = _by_year(summary_shares, "B/F Loss Used (EUR)")
        shares_ex = _by_year(summary_shares, "Exemption Used (EUR)")
        shares_taxable = _by_year(summary_shares, "Taxable Gain (EUR)")
        shares_tax = _by_year(summary_shares, share_tax_col)

        etf_pl = _by_year(summary_etfs, "Realised Profit / Loss (EUR)")
        etf_taxable = _by_year(summary_etfs, "Taxable Gain (EUR)")
        etf_tax = _by_year(summary_etfs, etf_tax_col)

        combined_tax_reported = _by_year(summary_combined, "Tax @ Combined (EUR)")

        year_index = sorted(
            set(shares_pl.index.tolist())
            | set(shares_bf.index.tolist())
            | set(shares_ex.index.tolist())
            | set(shares_taxable.index.tolist())
            | set(shares_tax.index.tolist())
            | set(etf_pl.index.tolist())
            | set(etf_taxable.index.tolist())
            | set(etf_tax.index.tolist())
            | set(combined_tax_reported.index.tolist())
        )
        if not year_index:
            st.info("No annual summary rows available for reconciliation.")
            return

        rec = pd.DataFrame(index=year_index)
        rec.index.name = "Year"
        rec["Shares Realised P/L (EUR)"] = shares_pl.reindex(year_index, fill_value=0.0).values
        rec["B/F Loss Used (EUR)"] = shares_bf.reindex(year_index, fill_value=0.0).values
        rec["Exemption Used (EUR)"] = shares_ex.reindex(year_index, fill_value=0.0).values
        rec["Shares Taxable Gain (EUR)"] = shares_taxable.reindex(year_index, fill_value=0.0).values
        rec[f"Tax @ Shares {int(cgt_rate_shares * 100)}% (EUR)"] = shares_tax.reindex(year_index, fill_value=0.0).values
        rec["ETFs Realised P/L (EUR)"] = etf_pl.reindex(year_index, fill_value=0.0).values
        rec["ETFs Taxable Gain (EUR)"] = etf_taxable.reindex(year_index, fill_value=0.0).values
        rec[f"Tax @ ETFs {int(exit_tax_rate_etf * 100)}% (EUR)"] = etf_tax.reindex(year_index, fill_value=0.0).values
        rec["Tax @ Combined (recomputed) (EUR)"] = (
            rec[f"Tax @ Shares {int(cgt_rate_shares * 100)}% (EUR)"] + rec[f"Tax @ ETFs {int(exit_tax_rate_etf * 100)}% (EUR)"]
        )
        rec["Tax @ Combined (reported) (EUR)"] = combined_tax_reported.reindex(year_index, fill_value=0.0).values
        rec["Delta (Reported - Recomputed) (EUR)"] = (
            rec["Tax @ Combined (reported) (EUR)"] - rec["Tax @ Combined (recomputed) (EUR)"]
        )

        st.caption(
            "Per-year bridge from realised gains/losses to taxable gain and tax due. "
            "Delta should normally be zero."
        )
        st.dataframe(
            rec.reset_index().style.format({c: fmt_money_eur for c in rec.columns}),
            use_container_width=True,
        )
