from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import pandas as pd


@dataclass
class PipelineResult:
    df_norm: pd.DataFrame
    out: pd.DataFrame
    split_audit_df: pd.DataFrame
    warnings: list[str]


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
    return PipelineResult(df_norm=df_work, out=out, split_audit_df=split_audit_df, warnings=warnings)

