from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from core.settings import DEFAULT_CGT_EXEMPTION_EUR, DEFAULT_CGT_RATE_SHARES, DEFAULT_EXIT_TAX_RATE_ETF


@dataclass(frozen=True)
class TaxConfig:
    use_exemption: bool = True
    exemption_val: float = DEFAULT_CGT_EXEMPTION_EUR
    cgt_rate_shares: float = DEFAULT_CGT_RATE_SHARES
    exit_tax_rate_etf: float = DEFAULT_EXIT_TAX_RATE_ETF


def _sum_mask_local(df: pd.DataFrame, mask: pd.Series, col: str) -> float:
    if df.empty or col not in df.columns:
        return 0.0
    s = pd.to_numeric(df.loc[mask, col], errors="coerce")
    return float(s.fillna(0).sum())


def _sum_mask_first_available(df: pd.DataFrame, mask: pd.Series, candidates: list[str]) -> float:
    for col in candidates:
        if col in df.columns:
            return _sum_mask_local(df, mask, col)
    return 0.0


def build_annual_summary(
    df_source: pd.DataFrame,
    asset_filter: Optional[str],
    years_sorted: list[int],
    config: TaxConfig,
) -> pd.DataFrame:
    rows = []
    carry_forward_prev_shares = 0.0

    for yr in years_sorted:
        g_all = df_source[df_source["__year"].eq(yr)].copy()

        if "__IncomingTransfer" in g_all.columns:
            g_all = g_all[~g_all["__IncomingTransfer"].fillna(False)].copy()

        g_shares = g_all[g_all["Asset"].astype(str).str.lower().eq("share")]
        g_etfs = g_all[g_all["Asset"].astype(str).str.lower().eq("etf")]

        if asset_filter == "share":
            g = g_shares

            buys_eur = _sum_mask_first_available(g, g["Type"].eq("Buy"), ["Total (EUR, fee-adj)", "Total (EUR)"])
            sells_eur = _sum_mask_first_available(g, g["Type"].eq("Sell"), ["Total (EUR, fee-adj)", "Total (EUR)"])
            fees_eur = _sum_mask_local(g, g["Type"].eq("Fee"), "Fee")
            div_net_eur = _sum_mask_local(g, g["Type"].eq("Dividend"), "Total") - _sum_mask_local(
                g, g["Type"].eq("Dividend"), "Fee"
            )
            interest_eur = _sum_mask_local(g, g["Type"].eq("Interest"), "Total")
            realised_pl = _sum_mask_local(g, g["Type"].eq("Sell"), "Gain/Loss")

            bf_used = 0.0
            if realised_pl > 0 and carry_forward_prev_shares > 0:
                bf_used = min(carry_forward_prev_shares, realised_pl)
            remaining_gain_after_bf = realised_pl - bf_used
            ex_used = (
                min(config.exemption_val, remaining_gain_after_bf)
                if (config.use_exemption and remaining_gain_after_bf > 0)
                else 0.0
            )
            taxable_gain = max(0.0, remaining_gain_after_bf - ex_used)
            tax_due = taxable_gain * config.cgt_rate_shares

            carry_forward_new = max(0.0, carry_forward_prev_shares - bf_used)
            if realised_pl < 0:
                carry_forward_new += abs(realised_pl)
            carry_forward_prev_shares = carry_forward_new

            row = {
                "Year": int(yr),
                "Buys (EUR)": buys_eur,
                "Sells (EUR)": sells_eur,
                "Realised Profit / Loss (EUR)": realised_pl,
                "Taxable Gain (EUR)": taxable_gain,
                f"Tax @ {int(config.cgt_rate_shares*100)}% (EUR)": tax_due,
                "B/F Loss Used (EUR)": bf_used,
                "Exemption Used (EUR)": ex_used,
                "Carry Forward (EUR)": carry_forward_new,
                "Net Cashflow (EUR)": sells_eur - buys_eur + div_net_eur + interest_eur - fees_eur,
                "Total Fees (EUR)": _sum_mask_local(g, g["Type"].isin(["Fee", "Interest"]), "Fee"),
            }

        elif asset_filter == "etf":
            g = g_etfs

            buys_eur = _sum_mask_first_available(g, g["Type"].eq("Buy"), ["Total (EUR, fee-adj)", "Total (EUR)"])
            sells_eur = _sum_mask_first_available(g, g["Type"].eq("Sell"), ["Total (EUR, fee-adj)", "Total (EUR)"])
            fees_eur = _sum_mask_local(g, g["Type"].eq("Fee"), "Fee")
            div_net_eur = _sum_mask_local(g, g["Type"].eq("Dividend"), "Total") - _sum_mask_local(
                g, g["Type"].eq("Dividend"), "Fee"
            )
            interest_eur = _sum_mask_local(g, g["Type"].eq("Interest"), "Total")
            realised_pl = _sum_mask_local(g, g["Type"].eq("Sell"), "Gain/Loss")

            taxable_gain = max(0.0, realised_pl)
            tax_due = taxable_gain * config.exit_tax_rate_etf

            row = {
                "Year": int(yr),
                "Buys (EUR)": buys_eur,
                "Sells (EUR)": sells_eur,
                "Realised Profit / Loss (EUR)": realised_pl,
                "Taxable Gain (EUR)": taxable_gain,
                f"Tax @ {int(config.exit_tax_rate_etf*100)}% (EUR)": tax_due,
                "Net Cashflow (EUR)": sells_eur - buys_eur + div_net_eur + interest_eur - fees_eur,
                "Total Fees (EUR)": _sum_mask_local(g, g["Type"].isin(["Fee", "Interest"]), "Fee"),
            }

        else:
            buys_eur_all = _sum_mask_first_available(
                g_all, g_all["Type"].eq("Buy"), ["Total (EUR, fee-adj)", "Total (EUR)"]
            )
            sells_eur_all = _sum_mask_first_available(
                g_all, g_all["Type"].eq("Sell"), ["Total (EUR, fee-adj)", "Total (EUR)"]
            )
            fees_eur_all = _sum_mask_local(g_all, g_all["Type"].eq("Fee"), "Fee")
            div_net_eur_all = _sum_mask_local(g_all, g_all["Type"].eq("Dividend"), "Total") - _sum_mask_local(
                g_all, g_all["Type"].eq("Dividend"), "Fee"
            )
            interest_eur_all = _sum_mask_local(g_all, g_all["Type"].eq("Interest"), "Total")

            realised_shares = _sum_mask_local(g_shares, g_shares["Type"].eq("Sell"), "Gain/Loss")
            realised_etfs = _sum_mask_local(g_etfs, g_etfs["Type"].eq("Sell"), "Gain/Loss")

            bf_used = 0.0
            if realised_shares > 0 and carry_forward_prev_shares > 0:
                bf_used = min(carry_forward_prev_shares, realised_shares)
            remaining_gain_after_bf = realised_shares - bf_used
            ex_used = (
                min(config.exemption_val, remaining_gain_after_bf)
                if (config.use_exemption and remaining_gain_after_bf > 0)
                else 0.0
            )
            taxable_shares = max(0.0, remaining_gain_after_bf - ex_used)
            tax_shares = taxable_shares * config.cgt_rate_shares

            carry_forward_new = max(0.0, carry_forward_prev_shares - bf_used)
            if realised_shares < 0:
                carry_forward_new += abs(realised_shares)
            carry_forward_prev_shares = carry_forward_new

            taxable_etfs = max(0.0, realised_etfs)
            tax_etfs = taxable_etfs * config.exit_tax_rate_etf

            row = {
                "Year": int(yr),
                "Buys (EUR)": buys_eur_all,
                "Sells (EUR)": sells_eur_all,
                "Realised Profit / Loss (EUR)": realised_shares + realised_etfs,
                f"Tax @ Shares {int(config.cgt_rate_shares*100)}% (EUR)": tax_shares,
                f"Tax @ ETFs {int(config.exit_tax_rate_etf*100)}% (EUR)": tax_etfs,
                "Tax @ Combined (EUR)": tax_shares + tax_etfs,
                "B/F Loss Used (EUR)": bf_used,
                "Exemption Used (EUR)": ex_used,
                "Carry Forward (EUR)": carry_forward_new,
                "Net Cashflow (EUR)": sells_eur_all - buys_eur_all + div_net_eur_all + interest_eur_all - fees_eur_all,
                "Total Fees (EUR)": _sum_mask_local(g_all, g_all["Type"].isin(["Fee", "Interest"]), "Fee"),
            }

        rows.append(row)

    return pd.DataFrame(rows)
