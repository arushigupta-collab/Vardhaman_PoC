#!/usr/bin/env python3
"""
Plot forecast trajectories for the cheapest vendors in a given target year.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot forecast trends for top vendors.")
    parser.add_argument(
        "--forecast-csv",
        default="vendor_forecasts.csv",
        type=Path,
        help="CSV created by vendor_forecast.py containing per-vendor forecasts.",
    )
    parser.add_argument(
        "--vendor-master",
        default="vendor_master.csv",
        type=Path,
        help="Vendor master data to attach readable names (optional).",
    )
    parser.add_argument(
        "--target-year",
        default=2025,
        type=int,
        help="Target year used to pick the cheapest vendors.",
    )
    parser.add_argument(
        "--top-k",
        default=5,
        type=int,
        help="Number of vendors to include in the plot.",
    )
    parser.add_argument(
        "--output",
        default=Path("vendor_price_trends.png"),
        type=Path,
        help="Where to write the plot.",
    )
    parser.add_argument(
        "--start-year",
        default=2015,
        type=int,
        help="First year to display in the plot.",
    )
    parser.add_argument(
        "--end-year",
        default=2030,
        type=int,
        help="Last year to display in the plot.",
    )
    return parser.parse_args()


def load_forecasts(forecast_path: Path) -> pd.DataFrame:
    if not forecast_path.exists():
        raise FileNotFoundError(f"Forecast CSV not found: {forecast_path}")
    return pd.read_csv(forecast_path)


def load_vendor_names(master_path: Path) -> pd.DataFrame | None:
    if master_path.exists():
        return pd.read_csv(master_path)[["vendor_id", "vendor_name"]]
    return None


def select_top_vendors(
    forecasts: pd.DataFrame, target_year: int, top_k: int
) -> list[str]:
    target = forecasts.loc[forecasts["year"] == target_year]
    if target.empty:
        raise ValueError(f"No forecast rows found for target year {target_year}.")
    sorted_rows = target.sort_values("yhat")
    return sorted_rows["vendor_id"].head(top_k).tolist()


def prepare_plot_data(
    forecasts: pd.DataFrame,
    vendor_ids: list[str],
    start_year: int,
    end_year: int,
    vendor_names: pd.DataFrame | None,
) -> pd.DataFrame:
    subset = forecasts[
        forecasts["vendor_id"].isin(vendor_ids) & forecasts["year"].between(start_year, end_year)
    ].copy()
    if vendor_names is not None:
        subset = subset.merge(vendor_names, on="vendor_id", how="left")
    else:
        subset["vendor_name"] = subset["vendor_id"]
    return subset


def plot_trends(
    data: pd.DataFrame,
    vendor_order: list[str],
    output_path: Path,
    start_year: int,
    end_year: int,
) -> None:
    plt.figure(figsize=(10, 6))

    for vendor_id in vendor_order:
        vendor_data = data.loc[data["vendor_id"] == vendor_id]
        if vendor_data.empty:
            continue

        name = vendor_data["vendor_name"].iloc[0]
        plt.plot(
            vendor_data["year"],
            vendor_data["yhat"],
            marker="o",
            label=name,
        )
        plt.fill_between(
            vendor_data["year"],
            vendor_data["yhat_lower"],
            vendor_data["yhat_upper"],
            alpha=0.15,
        )

    plt.xlim(start_year, end_year)
    plt.xlabel("Year")
    plt.ylabel("Predicted price per kg")
    plt.title("Forecasted Cotton Prices for Top Vendors")
    plt.legend()
    plt.grid(True, which="major", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()

    forecasts = load_forecasts(args.forecast_csv)
    vendor_names = load_vendor_names(args.vendor_master)

    top_vendors = select_top_vendors(forecasts, args.target_year, args.top_k)
    plot_data = prepare_plot_data(
        forecasts, top_vendors, args.start_year, args.end_year, vendor_names
    )

    plot_trends(plot_data, top_vendors, args.output, args.start_year, args.end_year)
    print(f"Saved trend plot for top {len(top_vendors)} vendors to {args.output}")


if __name__ == "__main__":
    main()
