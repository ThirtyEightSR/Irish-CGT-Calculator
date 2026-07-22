from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd


def _coerce_scalar_to_string(value: Any) -> str:
    if isinstance(value, str):
        return value
    if pd.isna(value):
        return ""
    if isinstance(value, (list, tuple, dict, set, bytes)):
        return ""
    return str(value)


def _sanitize_uploaded_frame(df_raw: pd.DataFrame) -> pd.DataFrame:
    df_raw = df_raw.copy()
    df_raw.columns = [str(col) for col in df_raw.columns]

    for col in df_raw.columns:
        series = df_raw[col]
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            df_raw[col] = series.apply(_coerce_scalar_to_string)
        elif pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            df_raw[col] = series.astype(object).apply(_coerce_scalar_to_string)
        else:
            df_raw[col] = series.astype(object).apply(_coerce_scalar_to_string)

    return df_raw


@dataclass
class PreprocessResult:
    df_norm: pd.DataFrame
    is_rich_missing_file: bool
    manual_norm: pd.DataFrame
    merged_manual_count: int


def prepare_input_dataframe(
    uploads: list[Any],
    manual_transactions: list[dict[str, Any]],
    opening_lots_df: pd.DataFrame | None,
    detect_broker_from_headers_fn: Callable[[pd.DataFrame], str],
    broker_adapters: dict[str, Callable[[pd.DataFrame], pd.DataFrame]],
    manual_transactions_to_canonical_fn: Callable[[list[dict[str, Any]]], pd.DataFrame],
    is_rich_missing_transactions_file_fn: Callable[[pd.DataFrame | None], bool],
    normalize_manual_fn: Callable[[pd.DataFrame | None], pd.DataFrame],
) -> PreprocessResult:
    frames: list[pd.DataFrame] = []

    if uploads:
        for f in uploads:
            try:
                df_raw = pd.read_csv(f) if f.name.lower().endswith(".csv") else pd.read_excel(f)
            except Exception:
                f.seek(0)
                df_raw = pd.read_csv(f, sep=";")

            df_raw = _sanitize_uploaded_frame(df_raw)
            broker = detect_broker_from_headers_fn(df_raw.head(1))
            adapter = broker_adapters.get(broker, broker_adapters["DEGIRO"])
            try:
                df_norm_one = adapter(df_raw)
            except Exception:
                df_norm_one = pd.DataFrame(index=df_raw.index)
                for col in [
                    "Date",
                    "Time",
                    "Value date",
                    "Product",
                    "ISIN",
                    "Description",
                    "FX",
                    "Change",
                    "Cash Movements",
                    "Balance",
                    "Order ID",
                    "Currency",
                ]:
                    if col in df_raw.columns:
                        df_norm_one[col] = df_raw[col]
                    else:
                        df_norm_one[col] = None

            df_norm_one["__Broker"] = broker
            df_norm_one["__SourceFile"] = getattr(f, "name", "upload")
            frames.append(df_norm_one)

    if not frames:
        if not manual_transactions:
            raise ValueError("No valid files parsed.")
        df_norm = pd.DataFrame(
            columns=[
                "Date",
                "Time",
                "Value date",
                "Product",
                "ISIN",
                "Description",
                "FX",
                "Change",
                "Cash Movements",
                "Balance",
                "Order ID",
                "Currency",
                "__Broker",
                "__SourceFile",
            ]
        )
    else:
        df_norm = pd.concat(frames, ignore_index=True)

    if "Date" in df_norm.columns and len(df_norm) > 0:
        df_norm = df_norm.sort_values(by="Date", kind="stable").reset_index(drop=True)

    merged_manual_count = 0
    if manual_transactions:
        manual_canonical = manual_transactions_to_canonical_fn(manual_transactions)
        if not manual_canonical.empty:
            df_norm = pd.concat([df_norm, manual_canonical], ignore_index=True)
            if "Date" in df_norm.columns:
                df_norm = df_norm.sort_values(by="Date", kind="stable").reset_index(drop=True)
            merged_manual_count = len(manual_transactions)

    is_rich_missing_file = is_rich_missing_transactions_file_fn(opening_lots_df)
    manual_norm = normalize_manual_fn(opening_lots_df)

    return PreprocessResult(
        df_norm=df_norm,
        is_rich_missing_file=is_rich_missing_file,
        manual_norm=manual_norm,
        merged_manual_count=merged_manual_count,
    )
