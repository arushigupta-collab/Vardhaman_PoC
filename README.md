# Cotton Vendor Price Forecasting

Forecast future cotton prices for upstream textile vendors using Prophet models enriched with weather and cost regressors. The tooling in this repo lets you:

- Generate per-vendor forecasts and pick the cheapest supplier for any future year.
- Visualize projected price trajectories for top vendors.
- Interact with the forecasts via a Streamlit dashboard that responds to user-selected years.

## Technical Architecture

- **Language & Runtime:** Python 3.11
- **Core libraries:** `prophet` 1.1.5, `cmdstanpy` 1.3.0, `pandas` 2.1.4, `numpy`, `matplotlib`, `streamlit`
- **Model backend:** Prophet leverages CmdStan 2.33.1 (linked via symlink to the local CmdStan 2.33.1 build) for efficient Stan optimization.
- **Data layout:** Flat CSV inputs loaded into pandas DataFrames; no databases or external services required.
- **Caching:** Streamlit cache (`st.cache_data`) wraps dataset loading and per-year forecast results to avoid redundant model fits.
- **Artifacts:** CLI forecast CSV (`vendor_forecasts.csv`), visualization PNG (`vendor_price_trends.png`), and interactive Streamlit output.
- **Compute characteristics:** Training 100 vendor models with the current feature set completes in ~8 seconds on a modern Mac (M-series). Prophet runs independent per vendor, so the process is embarrassingly parallel if you decide to scale out.

## Data

| File | Description |
| --- | --- |
| `vendor_market_data.csv` | Historical yearly metrics (2005–2024) for 100 vendors, including `avg_price_per_kg`, `rainfall_index`, `temperature_avg`, `fertilizer_cost_index`, and `government_incentive_score`. |
| `vendor_master.csv` | Lookup table mapping `vendor_id` to readable vendor names and home countries. |

The models treat the climate and policy indices as external regressors. When predicting beyond the latest observed year, the last known regressor values for each vendor are carried forward unless you supply alternative assumptions.

### Column Schema

#### `vendor_market_data.csv`

| Column | Type | Notes |
| --- | --- | --- |
| `year` | int | Calendar year (YYYY). |
| `vendor_id` | string | Matches `vendor_master.vendor_id`. |
| `region` | string | Indian state or global region used for reporting. |
| `avg_price_per_kg` | float | Historical average price per kilogram. Target variable. |
| `rainfall_index` | float | Normalized rainfall proxy (0–1). |
| `temperature_avg` | float | Average annual temperature (°C). |
| `fertilizer_cost_index` | float | Cost driver index (0–1). |
| `government_incentive_score` | float | Government subsidy/incentive signal (0–1). |

#### `vendor_master.csv`

| Column | Type | Notes |
| --- | --- | --- |
| `vendor_id` | string | Primary key. |
| `vendor_name` | string | Display name for the vendor. |
| `vendor_country` | string | Country of origin for reporting. |

## Environment Setup

1. **Create/activate a Python 3.11 environment** (optional but recommended).
2. **Install packages**:
   ```bash
   python3 -m pip install prophet cmdstanpy pandas streamlit matplotlib
   ```
3. **Install CmdStan binaries** (required by Prophet’s Stan backend):
   ```bash
   python3 - <<'PY'
   import cmdstanpy
   cmdstanpy.install_cmdstan()
   PY
   ```
   *If your OS blocks SSL certificates, install/upate certificate bundles first (on macOS you can run `"/Applications/Python 3.11/Install Certificates.command"`).*
4. **Verify Prophet is operational**:
   ```bash
   python3 - <<'PY'
   from prophet import Prophet
   import pandas as pd
   import numpy as np

   df = pd.DataFrame({"ds": pd.date_range("2000", periods=20, freq="Y"),
                      "y": np.linspace(1, 10, 20)})
   Prophet(yearly_seasonality=False).fit(df)
   print("Prophet ready")
   PY
   ```

Once Prophet runs without errors, you’re ready to produce forecasts.

## Forecast Pipeline Details

1. **Input ingestion:** `load_data` (in `vendor_forecast.py`) reads the market and master CSVs, joining vendor metadata onto the time-series records.
2. **Validation:** `_ensure_regressors` confirms presence of all required regressors (`rainfall_index`, `temperature_avg`, `fertilizer_cost_index`, `government_incentive_score`). Missing columns raise an exception early.
3. **Per-vendor modeling:** For each vendor, `forecast_vendor_prices`:
   - Sorts observations chronologically.
   - Initializes a Prophet instance with additive trend, disabled weekly/daily seasonality, and explicit regressors.
   - Converts the `year` column into Prophet’s `ds` datetime format using `pd.to_datetime` with yearly frequency.
   - Fits the model using optimized Stan routines (LBFGS for longer histories, Newton fallback for shorter series).
4. **Future frame generation:** Prophet’s `make_future_dataframe` creates an extended timeline. For regressors, the code left-joins historical values, then forward-fills missing future entries with each vendor’s most recent observed regressor value. This design keeps inference simple while waiting for scenario projections.
5. **Prediction & aggregation:** The script captures `yhat`, `yhat_lower`, and `yhat_upper` (95% confidence interval) for each vendor-year and collates results into a consolidated DataFrame.
6. **Recommendation logic:** For the requested `target_year`, the vendors are sorted by predicted `avg_price_per_kg` (`yhat`). The first row is the recommended supplier.
7. **Outputs:**
   - CLI prints a formatted table of top K vendors and optionally writes the full forecast matrix to CSV.
   - Streamlit view renders an interactive table and highlighted recommendation card.

### Prophet Configuration

- `yearly_seasonality=False`: Annual seasonality is effectively modeled via regressors and trend; enabling the default Fourier terms produced no accuracy gains in testing.
- `seasonality_mode="additive"`: Suitable for prices showing linear trend rather than multiplicative effects.
- `add_regressor`: All four exogenous variables are added as regressors with default settings.
- `changepoint_prior_scale`: Prophet default (0.05) is used; adjust via CLI/Streamlit modifications if you observe over/under-fitting.
- `growth`: Prophet automatically selects linear growth because logistic parameters (`cap`, `floor`) are not provided.
- `uncertainty`: `yhat_lower`/`yhat_upper` originate from Prophet’s posterior predictive distribution, giving intuitive pricing bounds.

## Command-Line Forecasting

`vendor_forecast.py` trains a Prophet model per vendor, forecasts forward, and prints the cheapest suppliers for the requested year.

```bash
python3 vendor_forecast.py \
  --target-year 2025 \
  --top-k 10 \
  --horizon 5 \
  --forecast-output vendor_forecasts.csv
```

Key arguments:
- `--target-year`: Future calendar year (YYYY) to evaluate.
- `--horizon`: Minimum number of extra years to forecast (handles cases where the target is close to the data boundary).
- `--top-k`: Number of vendors displayed in the summary table.
- `--forecast-output`: Optional CSV path for the consolidated forecasts (defaults to no export).

The script reuses the CSVs in the project root. As an example, the current dataset recommends **PlantKing Enterprises (V025)** at ~₹119.6/kg for 2025.

## Plotting Price Trajectories

Use `plot_vendor_forecasts.py` to visualize forecasted price trends, including uncertainty bands, for the cheapest vendors in a target year:

```bash
python3 plot_vendor_forecasts.py \
  --target-year 2025 \
  --top-k 5 \
  --start-year 2015 \
  --end-year 2030 \
  --output vendor_price_trends.png
```

The script expects `vendor_forecasts.csv` from the previous step. It outputs a PNG showing the forward trajectories and confidence intervals for each selected vendor.

## Streamlit Dashboard

Launch the interactive app to explore forecasts without leaving the browser:

```bash
streamlit run streamlit_app.py
```

Features:
- Sidebar controls for target year, forecast horizon, and vendor count.
- Automatically updated table of top vendors with price bands.
- Highlighted “recommended vendor” banner summarizing the best option.
- Expandable section exposing the full per-vendor prediction table.

The app uses Streamlit caching to avoid retraining models needlessly; changing the target year or horizon triggers a recompute only when necessary.

## Deploying to Streamlit Community Cloud

1. **Push this repository to GitHub** (public or private with Streamlit access). Ensure the root contains:
   - `streamlit_app.py`, `vendor_forecast.py`, `plot_vendor_forecasts.py`
   - `vendor_market_data.csv`, `vendor_master.csv`
   - `requirements.txt` (Python dependencies)
   - `packages.txt` (system packages for Prophet/CmdStan build toolchain)
2. **Sign in to [share.streamlit.io](https://share.streamlit.io/)** and choose **“New app”**.
3. Select your repo, branch, and set **Main file path** to `streamlit_app.py`.
4. Deploy. The first build installs system packages plus Python requirements, then compiles CmdStan—expect the initial run to take up to ~10 minutes. Subsequent restarts reuse the cached toolchain.
5. Share the Streamlit-provided URL with your team. Redeploys happen automatically whenever you push updates to the tracked branch.

If the build fails, check the Streamlit logs. Missing-compiler errors usually mean `packages.txt` is absent or misnamed; SSL errors during CmdStan download often require re-running the build after the initial package installation completes.

## File Overview

| File | Purpose |
| --- | --- |
| `vendor_forecast.py` | Core forecasting pipeline (CLI). |
| `plot_vendor_forecasts.py` | Helper to plot forecasts for top vendors. |
| `streamlit_app.py` | Streamlit interface for interactive exploration. |
| `vendor_forecasts.csv` | Example output generated by the CLI (2025 Top-K). |
| `vendor_price_trends.png` | Example plot produced by the plotting script. |
| `.streamlit/` (optional) | Add custom Streamlit config (e.g., theme) if desired. |

## Module & Function Reference

- `vendor_forecast.py`
  - `load_data(Path, Path|None) -> DataFrame`: Reads and joins datasets.
  - `forecast_vendor_prices(DataFrame, target_year: int, min_horizon: int) -> (DataFrame, DataFrame)`: Returns the sorted recommendation table plus the full forecast frame.
  - `main()`: CLI entry point parsing arguments and orchestrating the pipeline.
- `plot_vendor_forecasts.py`
  - `select_top_vendors`: Determines vendor IDs with minimum predicted price for the target year.
  - `prepare_plot_data`: Slices forecast horizon and merges vendor names.
  - `plot_trends`: Renders line chart with uncertainty bands.
- `streamlit_app.py`
  - `get_market_data`: Cached dataset loader.
  - `get_predictions`: Cached forecast call to reuse model outputs between UI interactions.
  - `main`: Streamlit composition (sidebar controls, table rendering, success banner).

All scripts rely on the shared logic exported from `vendor_forecast.py`, ensuring consistent forecast results between the CLI, plots, and Streamlit dashboards.

## Troubleshooting

- **`CmdStan installataion missing makefile`**: Remove the bundled `prophet/stan_model/cmdstan-*` folder (if present) and reinstall CmdStan via `cmdstanpy.install_cmdstan()`. Ensure `CMDSTAN` points to the freshly installed path.
- **SSL certificate errors during CmdStan download**: Update your system certificates (`Install Certificates.command` on macOS) or configure pip/cmdstanpy to use trusted certificates.
- **Long runtimes**: Fitting 100 vendors is fast (~seconds), but adding more regressors or a larger vendor universe will increase compute time. Reduce `top-k` or run on more powerful hardware if needed.

## Next Ideas

- Import forward-looking climate or policy projections instead of carrying forward the latest observed regressors.
- Evaluate alternative models (e.g., hierarchal Prophet, Gradient Boosting) and compare performance.
- Automate weekly/monthly monitoring by wrapping the CLI script in a scheduled job and notifying procurement stakeholders.
- Add evaluation notebook measuring historical backtest accuracy (MAPE/RMSE) to tune Prophet hyperparameters per region.
- Containerize the Streamlit app for reproducible deployment (Dockerfile, Compose).

Happy forecasting!
