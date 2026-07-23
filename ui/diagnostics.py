from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st


def build_fx_diagnostics_frame(out: Optional[pd.DataFrame]) -> pd.DataFrame:
    if not isinstance(out, pd.DataFrame) or out.empty:
        return pd.DataFrame()

    df = out.copy()
    if "Type" not in df.columns:
        return pd.DataFrame()

    df["Type"] = df["Type"].astype(str).str.strip()
    trade_mask = df["Type"].isin(["Buy", "Sell"])
    if not trade_mask.any():
        return pd.DataFrame()

    currency = df["Currency"].astype(str).str.upper().str.strip() if "Currency" in df.columns else pd.Series("", index=df.index)
    fx_rate = pd.to_numeric(df["FX_Rate"], errors="coerce") if "FX_Rate" in df.columns else pd.Series(np.nan, index=df.index, dtype="float64")
    fx_ccy = df["FXCCY"].astype(str).str.upper().str.strip() if "FXCCY" in df.columns else pd.Series("", index=df.index)

    issues = pd.Series("", index=df.index, dtype="object")

    non_eur_mask = trade_mask & ~currency.eq("EUR") & ~fx_ccy.eq("EUR")
    issues.loc[non_eur_mask & fx_rate.isna()] = "Missing FX_Rate for non-EUR trade"
    issues.loc[non_eur_mask & fx_rate.notna() & fx_rate.round(6).eq(1.0)] = "Suspicious FX_Rate=1.0 for non-EUR trade"

    eur_mask = trade_mask & (currency.eq("EUR") | fx_ccy.eq("EUR"))
    issues.loc[eur_mask & fx_rate.notna() & ~fx_rate.round(6).eq(1.0)] = "EUR trade carries non-1.0 FX_Rate"

    missing_fx_code = trade_mask & fx_ccy.eq("") & currency.ne("EUR")
    issues.loc[missing_fx_code & issues.eq("")] = "Missing FXCCY code for non-EUR trade"

    suspicious = df.loc[trade_mask & issues.ne(""), [c for c in ["Date", "Ticker - Name", "ISIN", "Type", "Currency", "FXCCY", "FX_Rate", "Total (EUR)", "Price_EUR", "Description"] if c in df.columns]].copy()
    if suspicious.empty:
        return pd.DataFrame()

    suspicious["Issue"] = issues.loc[suspicious.index].values
    if "Date" in suspicious.columns:
        suspicious["Date"] = pd.to_datetime(suspicious["Date"], errors="coerce")
    return suspicious.sort_values(by=[c for c in ["Date", "ISIN"] if c in suspicious.columns], kind="mergesort")


def render_fx_diagnostics(out: Optional[pd.DataFrame]) -> None:
    st.markdown("### 💱 FX Mapping Checks")

    fx_diag = build_fx_diagnostics_frame(out)
    if fx_diag.empty:
        st.success("No obvious FX mapping anomalies found in the current output.")
        return

    issue_counts = fx_diag["Issue"].value_counts().reset_index()
    issue_counts.columns = ["Issue", "Rows"]
    st.warning(f"Detected {len(fx_diag)} row(s) with FX mapping anomalies.")
    st.dataframe(issue_counts, use_container_width=True, hide_index=True)
    st.dataframe(fx_diag.head(100), use_container_width=True, hide_index=True)


def render_manual_missing_diagnostics(opening_lots_df: Optional[pd.DataFrame], out: Optional[pd.DataFrame]) -> None:
    st.markdown("### 📦 Imported Manual / Missing Transactions")

    if opening_lots_df is not None and not opening_lots_df.empty:
        manual_df = opening_lots_df.copy()
        df_show = manual_df.copy()
        df_show.columns = [c.strip() for c in df_show.columns]

        if "ISIN" not in df_show.columns:
            st.warning("Manual file has no ISIN column — cannot identify holdings.")
            return

        df_show["ISIN"] = df_show["ISIN"].astype(str).str.strip()
        if {"Type", "Quantity", "ISIN"}.issubset(df_show.columns):
            price_col_for_desc = next((c for c in ["Price_EUR", "Unit_EUR", "Total (EUR)", "Total_EUR"] if c in df_show.columns), None)
            if price_col_for_desc is not None:
                product_for_desc = (
                    df_show["Product"].astype(str).str.strip()
                    if "Product" in df_show.columns
                    else df_show["ISIN"].astype(str).str.strip()
                )
                default_desc = (
                    df_show["Type"].astype(str).str.strip()
                    + " "
                    + df_show["Quantity"].astype(str).str.strip()
                    + " "
                    + product_for_desc
                    + "@"
                    + df_show[price_col_for_desc].astype(str).str.strip()
                    + " EUR"
                )
                if "Description" not in df_show.columns:
                    df_show["Description"] = default_desc
                else:
                    desc = df_show["Description"].astype(str).str.strip()
                    df_show["Description"] = desc.mask(desc.str.lower().isin(["", "nan", "none", "nat"]), default_desc)

        ticker_col = next((c for c in df_show.columns if c.lower() in ["ticker", "name", "product", "description"]), None)
        qty_col = next((c for c in df_show.columns if "qty" in c.lower() or "quantity" in c.lower()), None)
        price_col = next((c for c in df_show.columns if "price" in c.lower()), None)
        eur_col = next((c for c in df_show.columns if "eur" in c.lower()), None)

        display_cols = ["ISIN"]
        if ticker_col:
            display_cols.append(ticker_col)
        if qty_col:
            display_cols.append(qty_col)
        if price_col:
            display_cols.append(price_col)
        if eur_col:
            display_cols.append(eur_col)

        unique_isins = sorted(df_show["ISIN"].dropna().unique().tolist())
        st.markdown(f"**{len(unique_isins)} ISIN(s)** detected in your uploaded manual file:")

        st.dataframe(df_show[display_cols].head(100), use_container_width=True, hide_index=True)

        if out is not None and isinstance(out, pd.DataFrame) and not out.empty:
            out_isins = set(out["ISIN"].astype(str).str.strip().dropna())
            missing_in_out = [i for i in unique_isins if i not in out_isins]
            if missing_in_out:
                st.warning(f"⚠️ {len(missing_in_out)} ISIN(s) not found in your DEGIRO export:")
                st.write(", ".join(missing_in_out))
            else:
                st.success("✅ All ISINs from your manual file exist in your DEGIRO data.")
        else:
            st.info("Upload your DEGIRO CSV to compare ISINs against existing trades.")
    else:
        st.caption("No manual or missing-transactions file has been uploaded yet.")


def render_incoming_transfer_diagnostics(out: Optional[pd.DataFrame], manual_norm: Optional[pd.DataFrame]) -> None:
    st.markdown("### 🔄 Incoming Transfers (promotion check)")

    if not isinstance(out, pd.DataFrame) or out.empty:
        st.caption("Upload and process a CSV first to enable incoming transfer checks.")
        return

    tmp = out.copy()

    if "Type" not in tmp.columns:
        tmp["Type"] = ""
    tmp["Type"] = tmp["Type"].astype(str).str.strip()

    desc_series = (tmp["Description"] if "Description" in tmp.columns else pd.Series("", index=tmp.index)).astype(str)
    incoming_mask = tmp["Type"].str.contains(r"\b(?:incoming|transfer\s*in|inbound)\b", case=False, na=False) | desc_series.str.contains(
        r"\b(?:incoming|transfer\s*in|inbound)\b", case=False, na=False
    )

    incoming_rows = tmp[incoming_mask].copy()

    if incoming_rows.empty:
        st.caption("No incoming transfer rows found in your broker data.")
        return

    st.markdown(f"Found **{len(incoming_rows)} incoming transfer(s)** in your data.")
    preview_cols = [c for c in ["Date", "Ticker - Name", "ISIN", "Quantity", "Order ID", "Type"] if c in incoming_rows.columns]
    st.dataframe(incoming_rows[preview_cols].head(50), use_container_width=True)

    if isinstance(manual_norm, pd.DataFrame) and not manual_norm.empty:
        man = manual_norm.copy()
        man["ISIN"] = man["ISIN"].astype(str).str.strip()
        man["Quantity"] = pd.to_numeric(man["Quantity"], errors="coerce")
        man["EUR_Value"] = pd.to_numeric(man["EUR_Value"], errors="coerce")
        man["Unit_EUR"] = pd.to_numeric(man["Unit_EUR"], errors="coerce")

        manual_by_isin = man.groupby("ISIN", dropna=False).agg(ManualQty=("Quantity", "sum"), ManualEUR=("EUR_Value", "sum")).reset_index()

        matched, unmatched = [], []
        for _, inc in incoming_rows.iterrows():
            isin = str(inc.get("ISIN", "")).strip()
            inc_qty = float(pd.to_numeric(inc.get("Quantity"), errors="coerce") or 0.0)
            mrow = manual_by_isin[manual_by_isin["ISIN"].eq(isin)]
            if not mrow.empty:
                matched.append(
                    {
                        "ISIN": isin,
                        "IncomingQty": inc_qty,
                        "ManualQty (sum)": float(mrow["ManualQty"].iloc[0] or 0.0),
                        "ManualEUR (sum)": float(mrow["ManualEUR"].iloc[0] or 0.0),
                    }
                )
            else:
                unmatched.append({"ISIN": isin, "IncomingQty": inc_qty})

        if matched:
            st.success(
                f"✅ Matched **{len(matched)} incoming transfer{'s' if len(matched) != 1 else ''}** "
                "to your uploaded manual lots file. These transfers have corresponding entries in "
                "your manual upload, so no further action is needed for them."
            )
            st.caption("Each row below shows the ISIN, quantity received, and the matching manual-lot details:")
            st.dataframe(pd.DataFrame(matched), use_container_width=True)

        if unmatched:
            st.warning(
                f"⚠️ {len(unmatched)} incoming transfer{'s' if len(unmatched) != 1 else ''} "
                "could not be matched to any entry in your uploaded manual lots file."
            )
            st.caption(
                "If you transferred these holdings from another broker, please add them to your manual file "
                "so their acquisition cost and quantity are recognised for capital gains tracking. "
                "Otherwise, they’ll remain unmatched and appear without a cost basis."
            )
            st.dataframe(pd.DataFrame(unmatched), use_container_width=True)
    else:
        st.info("Upload an opening-lots file to allow matching of incoming transfers.")
