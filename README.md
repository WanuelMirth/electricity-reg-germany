# electricity-reg-germany

Forecast German hourly and daily electricity grid load using SMARD data.

## First milestone: inspect the raw data

Open `notebooks/01_raw_data_audit.ipynb` and run its cells in order. The notebook
needs pandas, numpy, matplotlib, and IPython. It locates `data/` relative to the
repository, so it can run locally or inside a Databricks Git folder. On Databricks,
select notebook compute and ensure those packages are available first.

The audit checks numeric parsing, missing values and intervals, daylight saving,
hourly/daily consistency, and development-period distributions. It does not modify
the source CSV files. Structural checks cover the whole dataset; exploratory plots
use 2019–2023, preserving 2024 for final evaluation.

Review the notebook's audit tables and final section before deciding on cleaning.

## Consumption analysis

Run `notebooks/02_electricity_eda.ipynb` from top to bottom (and `notebooks/03_external_data_eda.ipynb` for weather, German holidays, and feature correlations):

1. Prepare hourly and daily data.
2. Compare distributions with boxplots.
3. Inspect trend with rolling and monthly means.
4. Compare hourly, weekly and annual seasonal profiles.
5. Separate daily load into trend, weekly/annual components and remainder using MSTL.

Uses 2019–2023; 2024 is excluded. Daily MWh is divided by the actual 23/24/25-hour
length to compare average load in MW. The 365-day seasonal cycle is approximate.
This is retrospective analysis, not forecasting. Raw CSVs remain unchanged.

To use local changes in Databricks, commit and push them to your GitHub branch,
then pull that branch in your Databricks Git folder. A local save alone does not
update the workspace.

Data attribution: Bundesnetzagentur | SMARD.de. Refer to the bundled SMARD user
guide for data definitions and time conventions.
