# %%
from sklearn.linear_model import Ridge

from sklearn.linear_model import LinearRegression, ElasticNet, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as mse
from statsmodels.tsa.arima.model import ARIMA

import yfinance as yf

from lets_plot import *
LetsPlot.setup_html()

import plotly.graph_objects as go

from vol_models.VolatilityReportGenerator import VolatilityReportGenerator
from vol_models.VolatilityEstimator import *
from vol_models.VolEstCheck import *

from vol_models.HARModel import *
from vol_models.EnsembleModel import *

from vol_models.Metrics import *

# %%
report = VolatilityReportGenerator(report_name="volatility_forecast_report", append=True)
print("Report generator initialized!")

data_folder = 'data'
# Read into DataFrame
IV_y_values = pd.read_csv(f'{data_folder}/MOVE_index.csv')
Fed_funds = pd.read_csv(f'{data_folder}/FedFunds.csv')
UST_10Y = pd.read_csv(f'{data_folder}/UST10Y.csv')
HYOAS = pd.read_csv(f'{data_folder}/HYOAS.csv')
NFCI = pd.read_csv(f'{data_folder}/NFCI.csv')
Termspread = pd.read_csv(f'{data_folder}/TermSpread_10Y_2Y.csv')
vix = pd.read_csv(f'{data_folder}/VIX.csv')
Breakeven_10Y = pd.read_csv(f'{data_folder}/Breakeven10Y.csv')

starting = "2003-01-01"
ending = "2025-09-30"

tlt =\
( # one ticker
    yf
    .download("TLT", # ticker
              start = starting, # starting date
              end = ending,
             auto_adjust = False)
    .droplevel("Ticker",
                axis = 1)
    # [["Close", "Volume"]]
)

tlt_data = tlt.loc[:'2024-12-30']


eps = 1e-12

y_true =\
(
    252
    *
    (np.log(tlt_data["Close"]
           .shift(-1)
            /
           tlt_data["Close"]))**2

)
y_true_log = np.log(y_true.clip(lower=eps))
y_true_log =\
(
    y_true_log
    .replace([np.inf, -np.inf], np.nan)
    .dropna()
    .iloc[1:]
)
# %%
vol_calc = volatility_estimator(add_log=True)
vol_results = vol_calc.compute_all(tlt_data, lag_for_predictors=True)
vol_results.isna().sum()
vol_results_adj = vol_results.dropna()
vol_estimator_check = vol_results[['square_est_log',
                                  'parkinson_est_log',
                                    'gk_est_log',
                                    'rs_est_log']]

y_predictors = vol_estimator_check.dropna()
y_predictors.describe()
vol_check = Vol_Est_Check(
    alpha=0.05,
    lb_lags=(10, 20),
    kpss_reg='c',
    kpss_nlags='auto',
    acf_pacf_nlags=40
)

for col in vol_estimator_check.columns:
    print(f"=== Diagnostics for {col} ===")
    result = vol_check.summarize_series(vol_estimator_check[col], name=col)
    print(result, "\n")

# %%
exo_variables = [UST_10Y, HYOAS, Termspread, vix, Breakeven_10Y]

for i, df in enumerate(exo_variables):
  df['Date'] = pd.to_datetime(df['Date'])
  df.set_index('Date', inplace=True)

exo_variable_all = pd.concat(exo_variables, axis=1, join = 'outer')
exo_var_adj = exo_variable_all.copy()
exo_var_adj.isna().sum()
master_idx = vol_results_adj.index
exo_adj =\
(
   exo_var_adj
   .reindex(index = master_idx)
   .ffill()
)
exo_adj.isna().sum()

exo_label = ['UST10Y', 'HYOAS', 'TermSpread_10Y_2Y', 'VIX', 'Breakeven10Y']

def Stdize_ExoVariables(df):
  df = df.copy()
  out = pd.DataFrame(index = df.index)

  for exo in exo_label:
        mean_series = df[exo].expanding().mean().shift(1)
        std_series  = df[exo].expanding().std(ddof=1).shift(1)
        z_series = (df[exo] - mean_series) / std_series

        out[f'std_mean_{exo}'] = mean_series
        out[f'std_dev_{exo}'] = std_series
        out[f'{exo}'] = z_series

  return out
exo_std_df = Stdize_ExoVariables(exo_adj)
exo_std_df = exo_std_df.dropna()
exo_label = ['UST10Y', 'HYOAS', 'TermSpread_10Y_2Y', 'VIX', 'Breakeven10Y']
exo_std_harx = exo_std_df[exo_label]
exo_std_harx_adj = exo_std_harx.loc[:'2024-12-27']
vol_adj_harx = y_predictors


# %%
vol_check = Vol_Est_Check(
    alpha=0.05,
    lb_lags=(10, 20),
    kpss_reg='c',
    kpss_nlags='auto',
    acf_pacf_nlags=40
)

for col in exo_std_harx_adj.columns:
    print(f"=== Diagnostics for {col} ===")
    result = vol_check.summarize_series(exo_std_harx_adj[col], name=col)
    print(result, "\n")

# diagnotics check before HAR modelling
summary_rows = []
for col in exo_std_harx_adj.columns:
    res = vol_check.summarize_series(exo_std_harx_adj[col], name=col)
    summary_rows.append({
        "Estimator": col,
        "ADF stat": res.get("adf_stat"),
        "ADF p": res.get("adf_p"),
        "ADF pass (p≤α)": res.get("adf_p") is not None and res["adf_p"] <= vol_check.alpha,
        "KPSS stat": res.get("kpss_stat"),
        "KPSS p": res.get("kpss_p"),
        "KPSS pass (p>α)": res.get("kpss_p") is not None and res["kpss_p"] > vol_check.alpha,
        "LB p @10": res.get("lb_lb_p_10"),
        "LB p @20": res.get("lb_lb_p_20"),
        "White noise (LB)": res.get("lb_white_noise_flag"),
    })

diag_tbl = pd.DataFrame(summary_rows).set_index("Estimator")

# Convenience column: both stationarity tests agree
diag_tbl["Stationary (ADF∩KPSS)"] = diag_tbl["ADF pass (p≤α)"] & diag_tbl["KPSS pass (p>α)"]


exo_std_harx_r1 = exo_std_harx_adj.copy()
exo_std_harx_r1['TermSpread_10Y_2Y'] = exo_std_harx_r1['TermSpread_10Y_2Y'].diff()
exo_std_harx_r1 = exo_std_harx_r1.dropna()

common_idx = vol_adj_harx.index.intersection(exo_std_harx_r1.index)

vol_adj_harx = vol_adj_harx.loc[common_idx]
exo_std_harx_r1 =exo_std_harx_r1.loc[common_idx]

print(vol_adj_harx )
print(exo_std_harx_r1)

for col in exo_std_harx_r1.columns:
    print(f"=== Diagnostics for {col} ===")
    result = vol_check.summarize_series(exo_std_harx_r1[col], name=col)
    print(result, "\n")

# diagnotics check before HAR modelling
summary_rows = []
for col in exo_std_harx_r1.columns:
    res = vol_check.summarize_series(exo_std_harx_r1[col], name=col)
    summary_rows.append({
        "Estimator": col,
        "ADF stat": res.get("adf_stat"),
        "ADF p": res.get("adf_p"),
        "ADF pass (p≤α)": res.get("adf_p") is not None and res["adf_p"] <= vol_check.alpha,
        "KPSS stat": res.get("kpss_stat"),
        "KPSS p": res.get("kpss_p"),
        "KPSS pass (p>α)": res.get("kpss_p") is not None and res["kpss_p"] > vol_check.alpha,
        "LB p @10": res.get("lb_lb_p_10"),
        "LB p @20": res.get("lb_lb_p_20"),
        "White noise (LB)": res.get("lb_white_noise_flag"),
    })

diag_tbl = pd.DataFrame(summary_rows).set_index("Estimator")

# Convenience column: both stationarity tests agree
diag_tbl["Stationary (ADF∩KPSS)"] = diag_tbl["ADF pass (p≤α)"] & diag_tbl["KPSS pass (p>α)"]

# %%
# Add HARX exogenous variables diagnostics to report
report.add_section("HAR-X Model Results", level=2)
report.add_section("Exogenous Variables", level=3)
report.add_text("""
The HAR-X model extends the HAR model by incorporating exogenous variables:
- **UST10Y**: 10-Year US Treasury Yield
- **HYOAS**: High Yield Option-Adjusted Spread
- **TermSpread_10Y_2Y**: Term Spread (10Y - 2Y)
- **VIX**: CBOE Volatility Index
- **Breakeven10Y**: 10-Year Breakeven Inflation Rate

All exogenous variables were standardized using expanding window standardization to prevent look-ahead bias.
""")

report.add_table(diag_tbl, caption="Table 7: Diagnostic Tests for Exogenous Variables (After Differencing)")
# %%
y_true_log_harx = y_true_log.loc[common_idx]
split_date = '2023-01-01'

print("\n" + "="*80)
print("HAR-X TRAIN/TEST SPLIT (Calendar-based)")
print("="*80)
print(f"Split date: {split_date}")

# x_variables
train_x = vol_adj_harx[vol_adj_harx.index < split_date].copy()
test_x = vol_adj_harx[vol_adj_harx.index >= split_date].copy()

# exogenous variables
exo_harx_train = exo_std_harx_r1[exo_std_harx_r1.index < split_date].copy()
exo_harx_test = exo_std_harx_r1[exo_std_harx_r1.index >= split_date].copy()

# y_variables
train_y = y_true_log_harx[y_true_log_harx.index < split_date].copy()
test_y = y_true_log_harx[y_true_log_harx.index >= split_date].copy()

print("Train X shape:", train_x.shape)
print("Test  X shape:", test_x.shape)
print("Train y shape:", train_y.shape)
print("Test  y shape:", test_y.shape)
print('Train Exo shape:' , exo_harx_train.shape)
print('Test Exo shape:' , exo_harx_test.shape)
print(f"Training period: {train_x.index.min()} to {train_x.index.max()}")
print(f"Test period: {test_x.index.min()} to {test_x.index.max()}")
print("="*80 + "\n")


# %%
window = [252, 504, 756, 1008, 1260]
estimators = ['square_est_log', 'parkinson_est_log', 'gk_est_log', 'rs_est_log']
per_est = {w: {} for w in window}
per_pred = {w: {} for w in window}
per_residual = {w: {} for w in window}
pred_raw_residual = {w: {} for w in window}

df_pred = {}
df_pred_adj = {}
df_residual = {}
df_residual_adj = {}
qlike_loss_df = {}
mspe_loss_df = {}
yhat_var = {}
summary_df = {}
ljung_box_df = {}

exo_cols = ['UST10Y', 'HYOAS', 'TermSpread_10Y_2Y', 'VIX', 'Breakeven10Y']

for w in window:
  print(f"\n[Window {w}] Training HARX models on FULL dataset (rolling window)...")

  for est in estimators:
    # Use FULL dataset (not just training set) for rolling window predictions
    # This will generate predictions for ALL time periods including 2023-2024
    df_in = pd.concat([vol_adj_harx[[est]], exo_std_harx_r1[exo_cols]], axis=1)
    har = HAR_Model(y_log_col=est, exo_col=exo_cols, lags=[1,5,22])
    x_est = har.features(df_in)
    # Use full y_true_log_harx (not just train_y)
    y_adj = y_true_log_harx.loc[x_est.index]
    per_est[w][est] = x_est

    y_pred, resid_pred, residual_raw = har.fit_predict(x_est, y_adj, window=w)

    per_pred[w][est] = y_pred
    per_residual[w][est] = resid_pred
    pred_raw_residual[w][est] = residual_raw
    
    print(f"  {est}: {len(y_pred)} predictions (from {y_pred.index.min()} to {y_pred.index.max()})")

  df_pred[w] = pd.DataFrame(per_pred[w])
  df_pred_adj[w] = df_pred[w].dropna()
  df_residual[w] = pd.DataFrame(pred_raw_residual[w])
  df_residual_adj[w] = df_residual[w].dropna()
  residual_input = df_residual_adj[w]

  #variance scale
  yhat_var[w] = np.exp(df_pred_adj[w])
  # Use FULL y_true_log_harx (not just train_y) to evaluate against predictions
  ytrue_var = np.exp(y_true_log_harx)
  common_idx = yhat_var[w].index.intersection(ytrue_var.index)
  yhat = yhat_var[w].loc[common_idx]
  ytrue = ytrue_var.loc[common_idx]
  
  print(f"  Evaluation: {len(yhat)} samples from {common_idx.min()} to {common_idx.max()}")

  qlike_loss_df[w] = pd.DataFrame({col: Metric_Evaluation.qlike(ytrue, yhat[col])
                                for col in yhat.columns})
  mspe_loss_df[w]  = pd.DataFrame({col: Metric_Evaluation.mspe(ytrue, yhat[col])
                                for col in yhat.columns})
  summary_df[w] = pd.DataFrame({
    'QLIKE_mean': qlike_loss_df[w].mean(),
    'QLIKE_std':  qlike_loss_df[w].std(),
    'MSPE_mean':  mspe_loss_df[w].mean(),
    'MSPE_std':   mspe_loss_df[w].std()
  }).round(4)

  vol_check = Vol_Est_Check(
      alpha=0.05,
      lb_lags=(10, 20),
      kpss_reg='c',
      kpss_nlags='auto',
      acf_pacf_nlags=40
  )
  ljung_box_df[w] = pd.DataFrame({col: vol_check.ljung_box(residual_input[col])
                              for col in residual_input.columns})

final_summary = pd.concat(summary_df, axis=0)
final_summary.index.name = 'Window'

ljung_box_summary = pd.concat(ljung_box_df, axis=0)
ljung_box_summary.index.name = 'Window'

print(final_summary)
print(ljung_box_summary)
# %%
# Add HARX model performance to report
report.add_section("HARX Model Performance", level=3)
report.add_text("""
The HAR-X model performance across different rolling window sizes is presented below.
""")
report.add_table(final_summary, caption="Table 8: HAR-X Model Performance Summary")
report.add_table(ljung_box_summary, caption="Table 9: HAR-X Model Ljung-Box Test Results")

# %%
# Save HARX prediction plots to report
report.add_section("HAR-X Model Predictions vs True RV", level=3)
report.add_text("The following plots compare the predicted volatility from each estimator with exogenous variables against the true realized volatility.")

for w in window:
    common_idx = df_pred_adj[w].index.intersection(train_y.index)
    yhat_plot = df_pred_adj[w].loc[common_idx]
    yhat_plot.columns = [f"{col}_pred" for col in yhat_plot.columns]
    ytrue_plot = train_y.loc[common_idx].to_frame(name='true_RV')
    
    fig = plt.figure(figsize=[16,7])
    yhat_plot.plot(ax=plt.gca(), alpha=0.9)
    ytrue_plot.plot(ax=plt.gca(), color='black', linewidth=2, alpha=0.3, label='True RV')
    plt.xlabel("Date")
    plt.ylabel("Log variance")
    plt.legend()
    plt.title(f"HAR-X prediction vs true RV for window {w}")
    plt.tight_layout()
    
    report.save_and_add_plot(fig, f"harx_prediction_w{w}", 
                            caption=f"HAR-X Model: Predictions vs True RV (Window={w})")
    plt.close()

print("✓ HAR-X prediction plots saved to report")

report.add_section("Loss Metrics Over Time", level=3)
report.add_text("""
QLIKE (Quasi-Likelihood) and MSPE (Mean Squared Prediction Error) are computed over time for each window.
These metrics help assess forecast calibration and error magnitude for the HAR-X model with exogenous variables.
""")

# QLIKE plots
for w in window:
    fig = plt.figure(figsize=[16,7])
    qlike_loss_df[w][['square_est_log', 'parkinson_est_log', 'gk_est_log']].plot(ax=plt.gca())
    plt.xlabel("Date")
    plt.ylabel("QLIKE")
    plt.legend()
    plt.title(f"HAR-X QLIKE Loss for window {w}")
    plt.tight_layout()
    report.save_and_add_plot(fig, f"harx_qlike_loss_w{w}", caption=f"HAR-X QLIKE Loss Over Time (Window={w})")
    plt.close()

# MSPE plots
for w in window:
    fig = plt.figure(figsize=[16,7])
    mspe_loss_df[w][['square_est_log', 'parkinson_est_log', 'gk_est_log', 'rs_est_log']].plot(ax=plt.gca())
    plt.xlabel("Date")
    plt.ylabel("MSPE")
    plt.legend()
    plt.title(f"HAR-X MSPE Loss for window {w}")
    plt.tight_layout()
    report.save_and_add_plot(fig, f"harx_mspe_loss_w{w}", caption=f"HAR-X MSPE Loss Over Time (Window={w})")
    plt.close()

print("✓ HAR-X loss metric plots saved to report")

# %%
# Ensemble model for HARX
qlike_ensemble_harx = {}
wts_harx = {}
weight_ensemble_harx = {}
yhat_ensemble_harx = {}
yhat_enfinal_harx = {}
log_yhat_enfinal_harx = {}
log_yhat_ensemble_harx = {}
residual_ensemble_harx = {}
qlike_loss_ensemble_harx = {}
mspe_loss_ensemble_harx = {}
summary_ensemble_harx = {}
ljung_box_ensemble_harx = {}

for w in window:
    # Compute weightage
    ensemble_model = EnsembleModel(estimators=None)
    qlike_ensemble_harx[w] = summary_df[w]['QLIKE_mean']
    weight_ensemble_harx[w] = ensemble_model.compute_weightage(qlike_ensemble_harx[w])
    yhat_ensemble_harx[w] = (np.exp(df_pred_adj[w]))
    
    wts_harx[w] = pd.Series(weight_ensemble_harx[w], index=yhat_ensemble_harx[w].columns, dtype=float)
    
    yhat_enfinal_harx[w] = yhat_ensemble_harx[w].dot(wts_harx[w])
    log_yhat_enfinal_harx[w] = np.log(yhat_enfinal_harx[w])
    
    common_idx = log_yhat_enfinal_harx[w].index.intersection(train_y.index)
    log_yhat_ensemble_harx[w] = log_yhat_enfinal_harx[w].loc[common_idx]  # log-variance
    log_ytrue_ensemble_harx = train_y.loc[common_idx]  # log-variance
    ytrue_ensemble_harx = ytrue_var.loc[common_idx]  # variance
    
    residual_ensemble_harx[w] = log_yhat_ensemble_harx[w] - log_ytrue_ensemble_harx
    
    qlike_loss_ensemble_harx[w] = pd.DataFrame(Metric_Evaluation.qlike(ytrue_ensemble_harx, yhat_enfinal_harx[w]))
    mspe_loss_ensemble_harx[w] = pd.DataFrame(Metric_Evaluation.mspe(ytrue_ensemble_harx, yhat_enfinal_harx[w]))
    
    summary_ensemble_harx[w] = pd.DataFrame({
        'QLIKE_mean': qlike_loss_ensemble_harx[w].mean(),
        'QLIKE_std': qlike_loss_ensemble_harx[w].std(),
        'MSPE_mean': mspe_loss_ensemble_harx[w].mean(),
        'MSPE_std': mspe_loss_ensemble_harx[w].std()
    }).round(4)
    
    vol_check = Vol_Est_Check(
        alpha=0.05,
        lb_lags=(10, 20),
        kpss_reg='c',
        kpss_nlags='auto',
        acf_pacf_nlags=40
    )
    ljung_box_ensemble_harx[w] = pd.DataFrame(vol_check.ljung_box(residual_ensemble_harx[w]))

# %%
final_summary_ensemble_harx = pd.concat(summary_ensemble_harx, axis=0)
final_summary_ensemble_harx.index.name = 'Window'

lb_ensemble_final_harx = pd.concat(ljung_box_ensemble_harx, axis=0)
lb_ensemble_final_harx.index.name = 'Window'

print(final_summary_ensemble_harx)
print(lb_ensemble_final_harx)

# %%
# Add ensemble model results to report
report.add_section("HAR-X Ensemble Model Results", level=2)
report.add_section("Ensemble Weights", level=3)
report.add_text("""
The HAR-X ensemble model combines predictions from multiple estimators using inverse QLIKE weighting.
Below are the weights assigned to each estimator for different window sizes.
""")

weights_df_harx = pd.DataFrame({w: wts_harx[w] for w in window}).T
weights_df_harx.index.name = 'Window'
report.add_table(weights_df_harx.round(4), caption="Table 10: HAR-X Ensemble Model Weights by Window")

report.add_section("HAR-X Ensemble Performance Summary", level=3)
report.add_table(final_summary_ensemble_harx, caption="Table 11: HAR-X Ensemble Model Performance Metrics")
report.add_table(lb_ensemble_final_harx, caption="Table 12: HAR-X Ensemble Model Ljung-Box Test Results")

report.add_section("HAR-X Ensemble Predictions and Loss Metrics", level=3)

# Ensemble predictions
for w in window:
    common_idx = log_yhat_enfinal_harx[w].index.intersection(train_y.index)
    yhat_plot = log_yhat_enfinal_harx[w].loc[common_idx].to_frame(name='Ensemble_HARX_RV')
    ytrue_plot = train_y.loc[common_idx].to_frame(name='true_RV')
    
    fig, ax = plt.subplots(figsize=(16, 7))
    yhat_plot.plot(ax=ax, color='green', linewidth=2, label='Ensemble_HARX_RV')
    ytrue_plot.plot(ax=ax, color='orange', linewidth=1.5, alpha=0.5, label='true_RV')
    plt.xlabel("Date")
    plt.ylabel("Log variance")
    plt.legend()
    plt.title(f"Ensemble HAR-X prediction vs true RV for window {w}")
    plt.tight_layout()
    report.save_and_add_plot(fig, f"harx_ensemble_pred_w{w}", 
                            caption=f"HAR-X Ensemble Model: Predictions vs True RV (Window={w})")
    plt.close()

# Ensemble QLIKE
for w in window:
    fig = plt.figure(figsize=[16,7])
    qlike_loss_ensemble_harx[w].plot(ax=plt.gca())
    plt.xlabel("Date")
    plt.ylabel("QLIKE")
    plt.title(f"HAR-X Ensemble QLIKE Loss for window {w}")
    plt.tight_layout()
    report.save_and_add_plot(fig, f"harx_ensemble_qlike_w{w}", 
                            caption=f"HAR-X Ensemble QLIKE Loss (Window={w})")
    plt.close()

# Ensemble MSPE
for w in window:
    fig = plt.figure(figsize=[16,7])
    mspe_loss_ensemble_harx[w].plot(ax=plt.gca())
    plt.xlabel("Date")
    plt.ylabel("MSPE")
    plt.legend()
    plt.title(f"HAR-X Ensemble MSPE Loss for window {w}")
    plt.tight_layout()
    report.save_and_add_plot(fig, f"harx_ensemble_mspe_w{w}", 
                            caption=f"HAR-X Ensemble MSPE Loss (Window={w})")
    plt.close()

print("✓ HAR-X Ensemble plots saved to report")

# %%
# Diebold-Mariano test comparing HAR-X windows
loss1_harx = qlike_loss_ensemble_harx[504]
loss2_harx = qlike_loss_ensemble_harx[756]
common_idx_harx = loss2_harx.index.intersection(loss1_harx.index)
loss2_adj_harx = loss2_harx.loc[common_idx_harx]
loss1_adj_harx = loss1_harx.loc[common_idx_harx]

DM_test_results_harx = Metric_Evaluation.DM_test(loss1_adj_harx,
                                                  loss2_adj_harx,
                                                  model1_name='HARX_Window_504',
                                                  model2_name='HARX_Window_756')
print(DM_test_results_harx)

report.add_section("HAR-X Diebold-Mariano Test", level=3)
report.add_text("""
The Diebold-Mariano (DM) test evaluates whether there is a statistically significant difference 
in predictive accuracy between two HAR-X models. Here we compare Windows 504 and 756.
""")

dm_stat_harx, p_val_harx, decision_harx = DM_test_results_harx
dm_results_harx = {
    "DM Statistic": dm_stat_harx,
    "P-value": p_val_harx,
    "Better Model": decision_harx['Better model'],
    "Significant?": decision_harx['Significant'],
    "Alpha": decision_harx['Alpha'],
    "Observations": decision_harx['Observations']
}
report.add_metrics_summary(dm_results_harx, title="HAR-X DM Test Results: Window 504 vs Window 756")

report.add_text(f"""
**Interpretation:** With p-value = {p_val_harx:.4f}, we {'reject' if p_val_harx < 0.05 else 'fail to reject'} the null hypothesis of equal 
predictive accuracy. This indicates {'a statistically significant difference' if p_val_harx < 0.05 else 'no statistically significant difference'} between Window 504 and 756 for the HAR-X model.
""")

# %%
# Finalize report
report.finalize_report()
print("✓ HAR-X report complete!")

# %%
# =============================================================================
# Save Predictions to CSV
# =============================================================================
print("Consolidating and saving HAR-X predictions to CSV...")

# Start with full dataset index (not just training set)
# ytrue_var only contains training set, so we need to use the full y_true_log_harx
full_ytrue_var = np.exp(y_true_log_harx)
results_df_harx = pd.DataFrame(index=full_ytrue_var.index)
results_df_harx['RV_true'] = full_ytrue_var

# Add HAR-X model predictions (yhat_var is a dict of window -> df)
for w in window:
    df = yhat_var[w].rename(columns=lambda c: f"HARX_{c.replace('_est_log', '')}_w{w}")
    results_df_harx = results_df_harx.join(df, how='outer')

# Add HAR-X Ensemble predictions (yhat_enfinal_harx is a dict of window -> series)
for w in window:
    series = yhat_enfinal_harx[w].rename(f"HARX_Ensemble_w{w}")
    results_df_harx = results_df_harx.join(series, how='outer')

# Save to CSV with the date index as a column named 'Date'
results_df_harx.to_csv('yhat_log_harx.csv', index_label='Date')

print("✓ HAR-X predictions saved to yhat_log_harx.csv")



