from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd


@dataclass
class PipelineResult:
    df_norm: pd.DataFrame
    out: pd.DataFrame
    split_audit_df: pd.DataFrame
    warnings: list[str]


def _fill_missing_fx_rates(out: pd.DataFrame, warnings: list[str]) -> pd.DataFrame:
    """Fill missing FX rates using nearest-date rate for same currency pair."""
    if out is None or out.empty or "Type" not in out.columns:
        return out

    out = out.copy()
    
    # Only process trades
    trade_mask = out["Type"].isin(["Buy", "Sell"])
    if not trade_mask.any():
        return out
    
    # Required columns for interpolation
    if "Date" not in out.columns or "Currency" not in out.columns or "FXCCY" not in out.columns or "FX_Rate" not in out.columns:
        return out
    
    # Initialize FX_Rate_Source_Date column (defaults to transaction date)
    out["Date_dt"] = pd.to_datetime(out["Date"], errors="coerce")
    out["FX_Rate_Source_Date"] = out["Date_dt"]
    
    currency = out["Currency"].astype(str).str.upper().str.strip()
    fx_ccy = out["FXCCY"].astype(str).str.upper().str.strip()
    fx_rate = pd.to_numeric(out["FX_Rate"], errors="coerce")
    
    # Only non-EUR trades need FX rate
    non_eur_mask = trade_mask & ~currency.eq("EUR") & ~fx_ccy.eq("EUR")
    if not non_eur_mask.any():
        out = out.drop(columns=["Date_dt"], errors="ignore")
        return out
    
    missing_fx_mask = non_eur_mask & fx_rate.isna()
    if not missing_fx_mask.any():
        out = out.drop(columns=["Date_dt"], errors="ignore")
        return out
    
    # Build lookup of valid (Currency, FXCCY) -> [(Date, Rate)] sorted by date
    valid_mask = non_eur_mask & fx_rate.notna()
    
    filled_count = 0
    for idx in out.index[missing_fx_mask]:
        curr = out.loc[idx, "Currency"]
        fccy = out.loc[idx, "FXCCY"]
        row_date = out.loc[idx, "Date_dt"]
        
        # Find valid rates for same currency pair
        pair_mask = valid_mask & (out["Currency"] == curr) & (out["FXCCY"] == fccy)
        if not pair_mask.any():
            continue
        
        pair_rows = out.loc[pair_mask].copy()
        pair_rows["Date_Diff"] = (pair_rows["Date_dt"] - row_date).abs()
        nearest_idx = pair_rows["Date_Diff"].idxmin()
        nearest = pair_rows.loc[nearest_idx]
        
        if nearest["FX_Rate"] == nearest["FX_Rate"]:  # not NaN
            out.loc[idx, "FX_Rate"] = nearest["FX_Rate"]
            out.loc[idx, "FX_Rate_Source_Date"] = nearest["Date_dt"]
            filled_count += 1
    
    if filled_count > 0:
        warnings.append(
            f"Filled {filled_count} missing FX_Rate(s) using nearest-date rates for same currency pair."
        )
    
    return out.drop(columns=["Date_dt"], errors="ignore")


def _append_trade_validation_warnings(out: pd.DataFrame, warnings: list[str]) -> None:
    if out is None or out.empty:
        return

    trade_mask = out["Type"].isin(["Buy", "Sell"]) if "Type" in out.columns else pd.Series(False, index=out.index)
    if not trade_mask.any():
        return

    if "Total (EUR)" in out.columns:
        missing_total = trade_mask & pd.to_numeric(out["Total (EUR)"], errors="coerce").isna()
        if missing_total.any():
            warnings.append(
                f"{int(missing_total.sum())} trade row(s) are missing Total (EUR) after normalization; EUR valuation may be incomplete."
            )

    if "Price_EUR" in out.columns:
        missing_price = trade_mask & pd.to_numeric(out["Price_EUR"], errors="coerce").isna()
        if missing_price.any():
            warnings.append(
                f"{int(missing_price.sum())} trade row(s) are missing Price_EUR; FIFO cost basis may rely on fallbacks."
            )

    if "Gain/Loss" in out.columns:
        sell_mask = trade_mask & out["Type"].eq("Sell")
        missing_gl = sell_mask & pd.to_numeric(out["Gain/Loss"], errors="coerce").isna()
        if missing_gl.any():
            warnings.append(
                f"{int(missing_gl.sum())} sell row(s) are missing Gain/Loss; tax summaries and exports may be incomplete."
            )


def _append_fx_validation_warnings(out: pd.DataFrame, warnings: list[str]) -> None:
    if out is None or out.empty or "Type" not in out.columns:
        return

    trade_mask = out["Type"].isin(["Buy", "Sell"])
    if not trade_mask.any():
        return

    currency = out["Currency"].astype(str).str.upper().str.strip() if "Currency" in out.columns else pd.Series("", index=out.index)
    fx_rate = pd.to_numeric(out["FX_Rate"], errors="coerce") if "FX_Rate" in out.columns else pd.Series(np.nan, index=out.index, dtype="float64")
    fx_ccy = out["FXCCY"].astype(str).str.upper().str.strip() if "FXCCY" in out.columns else pd.Series("", index=out.index)

    non_eur_mask = trade_mask & ~currency.eq("EUR") & ~fx_ccy.eq("EUR")
    if non_eur_mask.any():
        missing_fx = non_eur_mask & fx_rate.isna()
        if missing_fx.any():
            warnings.append(
                f"{int(missing_fx.sum())} non-EUR trade row(s) are missing FX_Rate; EUR conversion may be incomplete."
            )

        suspicious_one = non_eur_mask & fx_rate.notna() & fx_rate.round(6).eq(1.0)
        if suspicious_one.any():
            warnings.append(
                f"{int(suspicious_one.sum())} non-EUR trade row(s) have FX_Rate=1.0; check whether FX parsing inverted or dropped the currency code."
            )

    eur_mask = trade_mask & (currency.eq("EUR") | fx_ccy.eq("EUR"))
    if eur_mask.any() and "FX_Rate" in out.columns:
        suspicious_non_one = eur_mask & fx_rate.notna() & ~fx_rate.round(6).eq(1.0)
        if suspicious_non_one.any():
            warnings.append(
                f"{int(suspicious_non_one.sum())} EUR trade row(s) carry a non-1.0 FX_Rate; broker FX mapping may be inconsistent."
            )


def run_output_pipeline(
    df_norm: pd.DataFrame,
    opening_lots_df: Optional[pd.DataFrame],
    is_rich_missing_file: bool,
    merge_missing_transactions_fn: Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame],
    apply_missing_precedence_fn: Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame],
    build_output_fn: Callable[[pd.DataFrame, Optional[pd.DataFrame]], tuple[pd.DataFrame, pd.DataFrame]],
) -> PipelineResult:
    warnings: list[str] = []
    df_work = df_norm

    if opening_lots_df is not None and not opening_lots_df.empty and is_rich_missing_file:
        try:
            df_work = merge_missing_transactions_fn(df_work, opening_lots_df)
            df_work = apply_missing_precedence_fn(df_work, opening_lots_df)
        except Exception as e:  # keep app behavior: non-fatal warning, continue
            warnings.append(f"Failed to merge missing/manual trades: {e}")

    out, split_audit_df = build_output_fn(df_work, opening_lots_df)
    out = _fill_missing_fx_rates(out, warnings)
    _append_trade_validation_warnings(out, warnings)
    _append_fx_validation_warnings(out, warnings)
    return PipelineResult(df_norm=df_work, out=out, split_audit_df=split_audit_df, warnings=warnings)

