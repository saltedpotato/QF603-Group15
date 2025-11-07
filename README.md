# QF603-Group15 — Volatility Forecasting

This repository contains code used for volatility forecasting experiments (HAR, HAR-X, ML models and a Temporal Fusion Transformer) and associated plotting/reporting utilities. This README highlights the main user-facing scripts, where outputs are written, and how to generate the final PDF report.

## Main scripts (training / prediction)
Run these from the `QF603-Group15/` folder.

- `python HAR.py`
  - Trains/evaluates HAR models (multiple estimators & rolling windows). Produces ensemble predictions and saves consolidated CSV(s) used by plotting.
  - Produces `yhat_log_har.csv` in the working directory (contains ensemble predictions and other HAR outputs).

- `python HARX.py`
  - Same as `HAR.py` but includes exogenous variables (HAR-X). Produces `yhat_log_harx.csv`.

- `python Ml.py`
  - Trains various machine-learning models and saves predictions to `ml_predictions.csv`.

- `python TFT.py`
  - Runs the Temporal Fusion Transformer model(s) and writes `tft_predictions.csv`.

- `python plot_all_predictions.py`
  - Loads the CSVs above and creates interactive plots. The script also saves an HTML plot file (Plotly) and static image(s) used in the report.

Notes:
- The plotting script expects files with these names in the working directory: `yhat_log_har.csv`, `yhat_log_harx.csv`, `ml_predictions.csv`, `tft_predictions.csv` (adjust paths inside `plot_all_predictions.py` if your CSVs are elsewhere).
- Some legacy CSVs exist in the repository (e.g., `yhat_var_har.csv`, `yhat_var_harx.csv`). These are older/variance-scale outputs and may be empty or unused by `plot_all_predictions.py`.

## Generated outputs / report
- The generated report(s) and auxiliary outputs are placed in `report_output_v6/`.
  - Example report markdown: `report_output_v6/volatility_forecast_report_20251107_150925.md` (your timestamp may vary).
  - Images used by the report are in `report_output_v6/images/`.
  - The Plotly HTML created by `plot_all_predictions.py` is also stored in the report folder (check the script's save path).

- CSVs created by the training scripts include (common names):
  - `yhat_log_har.csv`
  - `yhat_log_harx.csv`
  - `ml_predictions.csv`
  - `tft_predictions.csv`
  - (`yhat_var_har.csv`, `yhat_var_harx.csv` — legacy; may be empty)

## Quick start — create a virtual environment and run
(From `QF603-Group15/`)

Create and activate a venv (zsh):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install commonly-used packages (suggested list; adjust as needed):

```bash
pip install --upgrade pip
pip install pandas numpy scipy matplotlib seaborn plotly scikit-learn statsmodels catboost lightgbm
# If you plan to run the TFT model, also install the required deep learning framework (PyTorch or TensorFlow)
# Example (PyTorch):
pip install torch torchvision torchaudio pytorch-lightning
```

Tip: If the repo contains a `requirements.txt` or `environment.yml`, prefer using that. If not, you can create one after confirming all packages used by the scripts.

## Run the full pipeline
1. Train/evaluate HAR (and HAR-X):

```bash
python HAR.py
python HARX.py
```

2. Train ML/TFT models (optional):

```bash
python Ml.py
python TFT.py
```

3. Produce plots and interactive HTML:

```bash
python plot_all_predictions.py
```

4. Generated CSVs and plot files will be available in the working directory or in `report_output_v6/` depending on script config.

## Generate a PDF report from the markdown (pandoc)
If you want to convert the generated markdown report to PDF using pandoc and XeLaTeX, run (from `QF603-Group15/`):

```bash
pandoc report_output_v6/volatility_forecast_report_20251107_150925.md -o report.pdf --pdf-engine=xelatex
```

Replace the markdown filename with the one you want to convert. Make sure `pandoc` and a TeX engine (e.g., `xelatex`) are installed on your system.

## Where to look for key code sections
- `HAR.py` and `HARX.py` — model fitting, ensemble logic, and CSV saving (look near the end of each file for the CSV consolidation & `to_csv` calls).
- `stat_model_r6.py` / `stat_model_r7.py` — older experiment notebooks/scripts; these files still write `yhat_var_*.csv` if you run them.
- `plot_all_predictions.py` — script that aggregates model outputs and creates the final interactive plot and static plots used in the report.

## Notes & troubleshooting
- If a CSV is unexpectedly empty, inspect the corresponding script to confirm whether that file is still written (legacy code may create empty outputs). We removed/disabled some variance-scale CSV outputs in recent cleanup to avoid empty files.
- If plotting script fails to find a CSV, either re-run the producing script or edit the `plot_all_predictions.py` to point to the correct file path.
- For reproducibility, consider exporting exact package versions into `requirements.txt` after you get a working environment:

```bash
pip freeze > requirements.txt
```

## Next steps (optional)
- I can add a minimal `requirements.txt` with the packages you use most.
- I can also run a quick smoke test to verify the plotting script finds the CSVs and produces an HTML output (if you want me to run scripts locally, confirm and I'll execute them).

If you'd like any edits to this README (different wording, more/less detail, or inclusion of a `requirements.txt`), tell me which part to expand and I'll update it.
