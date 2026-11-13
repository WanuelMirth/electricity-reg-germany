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

## Second milestone: contextual outliers

Run `notebooks/02_outlier_analysis.ipynb` independently, from top to bottom. It
compares calendar-group distributions, ranks the 20 largest adjacent and weekly
changes, flags contextual IQR candidates, and plots event windows. It uses only
2019–2023 and leaves the raw data unchanged. Daily energy is also divided by the
actual 23/24/25-hour day length to compare average load fairly.

Comparison helpers live in `src/outliers.py`. Weekly comparisons match German
local calendar time and exclude ambiguous autumn-hour matches. Tests cover clock
changes, missing intervals, and candidate flags; run `uv run pytest`.

To use local changes in Databricks, commit and push them to your GitHub branch,
then pull that branch in your Databricks Git folder. A local save alone does not
update the workspace.

Data attribution: Bundesnetzagentur | SMARD.de. Refer to the bundled SMARD user
guide for data definitions and time conventions.
