#!/usr/bin/env python3
"""
Train Prophet models per vendor to forecast cotton prices and recommend the cheapest supplier
for a specified future year.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from functools import lru_cache
import os
import subprocess
import sys
from typing import Iterable

import pandas as pd
from prophet import Prophet


REGRESSORS: tuple[str, ...] = (
    "rainfall_index",
    "temperature_avg",
    "fertilizer_cost_index",
    "government_incentive_score",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train Prophet models per vendor and forecast cotton prices to identify the best supplier "
            "for a target year."
        )
    )
    parser.add_argument(
        "--market-data",
        default="vendor_market_data.csv",
        type=Path,
        help="Path to the historical vendor market data (CSV).",
    )
    parser.add_argument(
        "--vendor-master",
        default="vendor_master.csv",
        type=Path,
        help="Path to the vendor master data (CSV) with vendor names/countries.",
    )
    parser.add_argument(
        "--target-year",
        default=2025,
        type=int,
        help="Calendar year (YYYY) for which to generate price recommendations.",
    )
    parser.add_argument(
        "--horizon",
        default=5,
        type=int,
        help=(
            "Minimum number of additional years each model should forecast. "
            "Useful when the target year is close to the end of the historic data."
        ),
    )
    parser.add_argument(
        "--top-k",
        default=10,
        type=int,
        help="Number of cheapest vendors to report.",
    )
    parser.add_argument(
        "--forecast-output",
        type=Path,
        help="Optional path to write the consolidated per-vendor forecasts as CSV.",
    )
    return parser.parse_args()


def load_data(market_path: Path, vendor_path: Path | None) -> pd.DataFrame:
    if not market_path.exists():
        raise FileNotFoundError(f"Market data not found: {market_path}")

    market = pd.read_csv(market_path)
    if vendor_path and vendor_path.exists():
        master = pd.read_csv(vendor_path)
        market = market.merge(master, on="vendor_id", how="left")

    return market


def _ensure_regressors(frame: pd.DataFrame) -> Iterable[str]:
    missing = [col for col in REGRESSORS if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing required regressors: {missing}")
    return REGRESSORS


@lru_cache(maxsize=1)
def ensure_cmdstan() -> str:
    """Install CmdStan if needed and point Prophet to it."""
    import cmdstanpy

    try:
        current = cmdstanpy.cmdstan_path()
        if current and Path(current, "bin", "stanc").exists():
            os.environ.setdefault("CMDSTAN", current)
            return current
    except Exception:
        pass

    try:
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import cmdstanpy; cmdstanpy.install_cmdstan()",
            ],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("CmdStan installation failed") from exc

    current = cmdstanpy.cmdstan_path()
    if not current or not Path(current, "bin", "stanc").exists():
        raise RuntimeError("CmdStan installation failed; Prophet cannot run.")

    os.environ["CMDSTAN"] = current
    return current


def forecast_vendor_prices(
    market: pd.DataFrame,
    target_year: int,
    min_horizon: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_cmdstan()

    required_regressors = tuple(_ensure_regressors(market))

    market = market.copy()
    market["year"] = market["year"].astype(int)
    market = market.sort_values(["vendor_id", "year"])

    max_year = int(market["year"].max())
    additional_years = max(0, target_year - max_year)
    future_periods = max(additional_years, min_horizon)

    per_vendor_results: list[dict[str, float | str | int]] = []
    forecast_frames: list[pd.DataFrame] = []

    for vendor_id, vendor_frame in market.groupby("vendor_id"):
        vendor_frame = vendor_frame.sort_values("year")
        model = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode="additive",
        )

        for reg in required_regressors:
            model.add_regressor(reg)

        train = vendor_frame.rename(
            columns={"year": "ds", "avg_price_per_kg": "y"}
        ).copy()
        train["ds"] = pd.to_datetime(train["ds"], format="%Y")

        model.fit(train[["ds", "y", *required_regressors]])

        future = model.make_future_dataframe(
            periods=future_periods,
            freq="Y",
            include_history=True,
        )
        future["year"] = future["ds"].dt.year
        future = future.merge(
            vendor_frame[["year", *required_regressors]],
            on="year",
            how="left",
        )

        for reg in required_regressors:
            future[reg] = future[reg].fillna(vendor_frame[reg].iloc[-1])

        forecast = model.predict(future)
        forecast = forecast.assign(
            vendor_id=vendor_id,
            year=future["year"].values,
        )[["vendor_id", "year", "ds", "yhat", "yhat_lower", "yhat_upper"]]

        forecast_frames.append(forecast)

        target_row = forecast.loc[forecast["year"] == target_year]
        if target_row.empty:
            continue

        record = {
            "vendor_id": vendor_id,
            "forecast_year": target_year,
            "predicted_price": float(target_row["yhat"].iloc[0]),
            "predicted_low": float(target_row["yhat_lower"].iloc[0]),
            "predicted_high": float(target_row["yhat_upper"].iloc[0]),
        }

        for extra_col in ("vendor_name", "vendor_country", "region"):
            if extra_col in vendor_frame.columns:
                record[extra_col] = vendor_frame[extra_col].iloc[-1]

        per_vendor_results.append(record)

    predictions = pd.DataFrame(per_vendor_results).sort_values(
        "predicted_price", ascending=True
    )
    all_forecasts = pd.concat(forecast_frames, ignore_index=True)

    return predictions, all_forecasts


def main() -> None:
    args = parse_args()

    market = load_data(args.market_data, args.vendor_master)
    predictions, forecasts = forecast_vendor_prices(
        market=market,
        target_year=args.target_year,
        min_horizon=args.horizon,
    )

    if predictions.empty:
        raise RuntimeError("No forecasts were generated. Check the input data and parameters.")

    top_k = predictions.head(args.top_k)
    print(f"Top {len(top_k)} vendors for {args.target_year}:")
    display_cols = [
        col
        for col in ["vendor_id", "vendor_name", "region", "vendor_country", "predicted_price"]
        if col in top_k.columns
    ]
    print(top_k[display_cols].to_string(index=False, float_format=lambda x: f"{x:0.2f}"))

    best_vendor = top_k.iloc[0]
    best_name = best_vendor.get("vendor_name", best_vendor["vendor_id"])
    print(
        f"\nRecommended vendor for {args.target_year}: {best_name} "
        f"({best_vendor['vendor_id']}) at {best_vendor['predicted_price']:.2f} per kg"
    )

    if args.forecast_output:
        forecasts.to_csv(args.forecast_output, index=False)
        print(f"Full forecast data written to {args.forecast_output}")


if __name__ == "__main__":
    main()
