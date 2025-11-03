#!/usr/bin/env python3
"""
Streamlit app to forecast cotton prices and recommend top vendors for a selected future year.
"""

from __future__ import annotations

from pathlib import Path
import os

import pandas as pd
import streamlit as st

from vendor_forecast import load_data, forecast_vendor_prices


MARKET_PATH = Path("vendor_market_data.csv")
MASTER_PATH = Path("vendor_master.csv")


@st.cache_data(show_spinner=False)
def get_market_data() -> pd.DataFrame:
    return load_data(MARKET_PATH, MASTER_PATH)


@st.cache_data(show_spinner=False)
def load_cached_forecasts() -> pd.DataFrame | None:
    precomputed_path = Path("vendor_forecasts.csv")
    if precomputed_path.exists():
        return pd.read_csv(precomputed_path)
    return None


def main() -> None:
    st.set_page_config(page_title="Cotton Vendor Forecasts", layout="wide")
    st.title("Cotton Vendor Price Forecasts")
    st.markdown(
        "Predict future cotton prices per vendor using Prophet models and identify the best suppliers."
    )

    with st.sidebar:
        st.header("Forecast settings")
        market = get_market_data()
        latest_year = int(market["year"].max())

        default_target = min(latest_year + 1, latest_year + 5)
        target_year = st.number_input(
            "Target year",
            min_value=latest_year + 1,
            max_value=latest_year + 20,
            value=default_target,
            step=1,
        )
        horizon = st.slider(
            "Minimum forecast horizon (years)",
            min_value=1,
            max_value=10,
            value=5,
        )
        top_k = st.slider("Vendors to display", min_value=1, max_value=20, value=5)

    market = get_market_data()
    running_on_streamlit_cloud = os.environ.get("STREAMLIT_RUNTIME") == "streamlit-runtime"
    fallback_used = False

    if running_on_streamlit_cloud:
        precomputed = load_cached_forecasts()
        if precomputed is None:
            st.error(
                "This hosted app relies on precomputed forecasts, but none were found. "
                "Please compute forecasts locally (see README) or provide an updated "
                "`vendor_forecasts.csv`."
            )
            st.stop()
        predictions = (
            precomputed.loc[precomputed["year"] == target_year]
            .rename(
                columns={
                    "yhat": "predicted_price",
                    "yhat_lower": "predicted_low",
                    "yhat_upper": "predicted_high",
                }
            )
            .copy()
        )
        predictions["forecast_year"] = target_year
        predictions = predictions.merge(
            market[["vendor_id", "vendor_name", "vendor_country", "region"]]
            .drop_duplicates("vendor_id"),
            on="vendor_id",
            how="left",
        )
        predictions = predictions.sort_values("predicted_price", ascending=True)
        fallback_used = True
        if predictions.empty:
            st.error(
                "Forecasting is unavailable for the selected year in this hosted environment."
                " Please choose a year covered by the precomputed dataset or run the CLI locally."
            )
            st.stop()
    else:
        try:
            predictions, _ = forecast_vendor_prices(
                market=market, target_year=target_year, min_horizon=horizon
            )
        except Exception as exc:  # noqa: BLE001
            precomputed = load_cached_forecasts()
            if precomputed is None:
                st.error(f"Forecast failed: {exc}")
                st.stop()
            predictions = (
                precomputed.loc[precomputed["year"] == target_year]
                .rename(
                    columns={
                        "yhat": "predicted_price",
                        "yhat_lower": "predicted_low",
                        "yhat_upper": "predicted_high",
                    }
                )
                .copy()
            )
            predictions["forecast_year"] = target_year
            predictions = predictions.merge(
                market[["vendor_id", "vendor_name", "vendor_country", "region"]]
                .drop_duplicates("vendor_id"),
                on="vendor_id",
                how="left",
            )
            predictions = predictions.sort_values("predicted_price", ascending=True)
            fallback_used = True
            if predictions.empty:
                st.error(
                    "Forecasting is unavailable for the selected year. "
                    "Please choose a year covered by the precomputed dataset."
                )
                st.stop()

    if predictions.empty:
        st.warning("No forecasts available for the selected parameters.")
        st.stop()

    top_vendors = predictions.head(top_k).copy()
    top_vendors = top_vendors.rename(
        columns={
            "predicted_price": "Predicted price",
            "predicted_low": "Lower bound",
            "predicted_high": "Upper bound",
            "vendor_name": "Vendor name",
            "vendor_country": "Country",
        }
    )

    st.subheader(f"Top {len(top_vendors)} vendors for {target_year}")
    st.dataframe(
        top_vendors[
            [
                col
                for col in [
                    "vendor_id",
                    "Vendor name",
                    "Country",
                    "region",
                    "Predicted price",
                    "Lower bound",
                    "Upper bound",
                ]
                if col in top_vendors.columns
            ]
        ].style.format(
            {
                "Predicted price": "{:.2f}",
                "Lower bound": "{:.2f}",
                "Upper bound": "{:.2f}",
            }
        ),
        use_container_width=True,
    )

    best_vendor = top_vendors.iloc[0]
    st.success(
        f"Recommended Vendor: "
        f"{best_vendor.get('Vendor name', best_vendor['vendor_id'])} "
        f"({best_vendor['vendor_id']}) at "
        f"{best_vendor['Predicted price']:.2f} per kg "
        f"(range {best_vendor['Lower bound']:.2f} – {best_vendor['Upper bound']:.2f})"
    )

    with st.expander("Raw prediction data"):
        st.dataframe(predictions, use_container_width=True)


if __name__ == "__main__":
    main()
