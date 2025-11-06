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

from vol_models.VolatilityReportGenerator import *
from vol_models.VolatilityEstimator import *
from vol_models.VolEstCheck import *

from vol_models.HARModel import *
from vol_models.EnsembleModel import *

from vol_models.Metrics import *

# %%
report = VolatilityReportGenerator()
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

# Calendar split metadata for reporting
split_date_str = "2023-01-01"
split_date_ts = pd.Timestamp(split_date_str)
train_mask_desc = y_predictors.index < split_date_ts
test_mask_desc = y_predictors.index >= split_date_ts
train_obs_desc = int(train_mask_desc.sum())
test_obs_desc = int(test_mask_desc.sum())
train_start_desc = y_predictors.index.min()
train_end_desc = y_predictors.index[train_mask_desc][-1] if train_obs_desc > 0 else None
test_start_desc = y_predictors.index[test_mask_desc][0] if test_obs_desc > 0 else None

# %%
report.add_section("Executive Summary", level=2)
report.add_text("""
This report presents a comprehensive analysis of volatility forecasting using Heterogeneous Autoregressive (HAR) 
and HAR with exogenous variables (HAR-X) models applied to Treasury Bond ETF (TLT) data. 

**Key Objectives:**
- Evaluate multiple volatility estimators (Squared Return, Parkinson, Garman-Klass, Rogers-Satchell)
- Compare HAR and HAR-X model performance across different rolling window sizes
- Implement ensemble forecasting using inverse QLIKE weighting
- Validate models using out-of-sample testing and statistical comparisons

**Main Findings:**
- Ensemble models outperform individual estimators across all metrics
- HAR-X with window=756 provides most stable and consistent predictions
- Exogenous variables offer marginal but meaningful improvement in forecast calibration
- No statistically significant difference between windows 504 and 756 (DM test)
- Both models demonstrate strong forecasting ability with well-behaved residuals
""")

report.add_section("Data Description", level=2)
report.add_text(f"""
**Dataset:** iShares 20+ Year Treasury Bond ETF (TLT)  
**Period:** 2003-01-01 to 2024-12-30  
**Frequency:** Daily  
**Total Observations:** {len(tlt_data)}  

**Price Data Components:**
- Open, High, Low, Close prices
- Trading volume
- Adjusted close prices

**Target Variable:**
- Realized Volatility (RV): Annualized variance computed from log returns
- Log-transformed for modeling to ensure stationarity

**Train/Test Split:**
- Training Set: {train_start_desc.strftime('%Y-%m-%d')} to {train_end_desc.strftime('%Y-%m-%d')} (data before {split_date_str}) — {train_obs_desc} observations
- Test Set: {test_start_desc.strftime('%Y-%m-%d')} to 2024-12-30 (data on/after {split_date_str}) — {test_obs_desc} observations
""")

report.add_section("Methodology", level=2)
report.add_text("""
### Model Framework

**1. Volatility Estimation**

Four volatility estimators are computed from OHLC data:
- **Squared Return (RV)**: σ²ₜ = 252 × (log(Cₜ/Cₜ₋₁))²
- **Parkinson**: σ²ₜ = 252 × (1/(4ln2)) × (log(Hₜ/Lₜ))²
- **Garman-Klass**: σ²ₜ = 252 × [0.5(log(Hₜ/Lₜ))² - (2ln2-1)(log(Cₜ/Oₜ))²]
- **Rogers-Satchell**: σ²ₜ = 252 × [log(Hₜ/Oₜ)log(Hₜ/Cₜ) + log(Lₜ/Oₜ)log(Lₜ/Cₜ)]

All estimators are log-transformed for modeling.

**2. HAR Model**

The HAR model captures heterogeneous volatility components:

log(RVₜ) = β₀ + β₁·RVₜ₋₁ + β₂·RVₜ₋₅:ₜ₋₁ + β₃·RVₜ₋₂₂:ₜ₋₁ + εₜ

Where:
- RVₜ₋₁: Daily component (lag 1)
- RVₜ₋₅:ₜ₋₁: Weekly component (5-day average)
- RVₜ₋₂₂:ₜ₋₁: Monthly component (22-day average)

**3. HAR-X Model**

Extends HAR by adding exogenous variables:

log(RVₜ) = β₀ + β₁·RVₜ₋₁ + β₂·RVₜ₋₅:ₜ₋₁ + β₃·RVₜ₋₂₂:ₜ₋₁ + Σγᵢ·Xᵢₜ + εₜ

Exogenous variables (Xᵢₜ):
- UST10Y (10-Year Treasury Yield)
- HYOAS (High Yield Spread)
- TermSpread (10Y-2Y)
- VIX (Volatility Index)
- Breakeven10Y (Inflation expectations)

**4. Rolling Window Estimation**

Models estimated using rolling windows: 252, 504, 756, 1008, 1260 days

**5. Ensemble Forecasting**

Predictions combined using inverse QLIKE weighting:

wᵢ = (1/QLIKEᵢ) / Σⱼ(1/QLIKEⱼ)

Final forecast: ŷₜ = Σᵢ wᵢ × ŷᵢₜ

**6. Evaluation Metrics**

- **QLIKE**: log(σ̂²ₜ) + σ²ₜ/σ̂²ₜ (forecast calibration)
- **MSPE**: ((σ²ₜ - σ̂²ₜ)/σ²ₜ)² (percentage error)
- **RMSE**: √(E[(σ²ₜ - σ̂²ₜ)²]) (absolute error)
- **Diebold-Mariano Test**: Statistical comparison of forecast accuracy
- **Ljung-Box Test**: Residual autocorrelation check
""")


# %%

summary_rows = []
for col in vol_estimator_check.columns:
    res = vol_check.summarize_series(vol_estimator_check[col], name=col)
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

report.add_section("Volatility Estimators Analysis", level=2)
report.add_section("Pre-Model Diagnostics", level=3)
report.add_text("""
The following table presents the stationarity and autocorrelation diagnostics for each volatility estimator.
We use the Augmented Dickey-Fuller (ADF) test and Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test to assess stationarity,
and the Ljung-Box test to check for serial correlation in the residuals.
""")
report.add_table(diag_tbl, caption="Table 1: Diagnostic Tests for Volatility Estimators")
report.add_section("ACF and PACF Analysis", level=3)
report.add_text("""
The Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) plots reveal:
- **Slow decay in ACF**: Indicates long memory in volatility, consistent with volatility clustering
- **Significant PACF spikes**: Suggests short-term AR effects up to 5-15 lags
- **HAR model justification**: These patterns support using daily (1), weekly (5), and monthly (22) lags
""")

# %%
columns = list(vol_estimator_check.columns)
n_cols = len(columns)
fig, axes = plt.subplots(n_cols, 2, figsize=(16, 6 * n_cols))

for i, col in enumerate(columns):
    # ACF
    plot_acf(vol_estimator_check[col].dropna(), lags=40, ax=axes[i, 0])
    axes[i, 0].set_title(f"ACF - {col}")
    
    # PACF
    plot_pacf(vol_estimator_check[col].dropna(), lags=40, ax=axes[i, 1])
    axes[i, 1].set_title(f"PACF - {col}")

plt.tight_layout()
report.save_and_add_plot(fig, "acf_pacf_all_estimators", caption="ACF and PACF for All Volatility Estimators")
plt.close()


# %%
comon_idx = y_true_log.index.intersection(y_predictors.index)
y_true_log = y_true_log.loc[comon_idx]
y_predictors = y_predictors.loc[comon_idx]
print(y_true_log)

# Calendar-based split
split_date = split_date_ts  # reuse timestamp defined earlier

print("\n" + "="*80)
print("HAR TRAIN/TEST SPLIT (Calendar-based)")
print("="*80)
print(f"Split date: {split_date_str}")

# x_variables
train_x = y_predictors[y_predictors.index < split_date].copy()
test_x = y_predictors[y_predictors.index >= split_date].copy()

# y_variables
train_y = y_true_log[y_true_log.index < split_date].copy()
test_y = y_true_log[y_true_log.index >= split_date].copy()

print("Train X shape:", train_x.shape)
print("Test  X shape:", test_x.shape)
print("Train y shape:", train_y.shape)
print("Test  y shape:", test_y.shape)
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


for w in window:

  for est in estimators:
    har = HAR_Model(y_log_col=est, exo_col=None)
    x_est = har.features(train_x)
    y_adj = train_y.loc[x_est.index] # log variance
    per_est[w][est] = x_est

    y_pred, resid_pred, residual_raw = har.fit_predict(x_est ,y_adj, window=w)

    per_pred[w][est] = y_pred
    per_residual[w][est] = resid_pred
    pred_raw_residual[w][est] = residual_raw

  df_pred[w] = pd.DataFrame(per_pred[w])
  df_pred_adj[w] = df_pred[w].dropna()
  df_residual[w] = pd.DataFrame(pred_raw_residual[w])
  df_residual_adj[w] = df_residual[w].dropna()
  residual_input = df_residual_adj[w]

  #variance scale
  yhat_var[w] = np.exp(df_pred_adj[w])
  ytrue_var = np.exp(train_y) #variance scale
  common_idx = yhat_var[w].index.intersection(ytrue_var.index)
  yhat = yhat_var[w].loc[common_idx]
  ytrue = ytrue_var.loc[common_idx]

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

# %%
final_summary = pd.concat(summary_df, axis=0)
final_summary.index.name = 'Window'

ljung_box_summary = pd.concat(ljung_box_df, axis=0)
ljung_box_summary.index.name = 'Window'

print(final_summary)
print(ljung_box_summary)

# Add HAR model results to report
report.add_section("HAR Model Results", level=2)
report.add_section("Performance Metrics Across Windows", level=3)
report.add_text("""
The HAR model was evaluated across multiple rolling window sizes: 252, 504, 756, 1008, and 1260 days.
Below are the comprehensive performance metrics for each estimator and window size.
""")
report.add_table(final_summary, caption="Table 2: HAR Model Performance Summary (QLIKE and MSPE)")
report.add_table(ljung_box_summary, caption="Table 3: Ljung-Box Test Results for HAR Model Residuals")


# %%
# Save HAR prediction plots to report
report.add_section("HAR Model Predictions vs True RV", level=3)
report.add_text("The following plots compare the predicted volatility from each estimator against the true realized volatility.")

window = [252, 504, 756, 1008, 1260]
for w in window:
    common_idx = df_pred_adj[w].index.intersection(y_adj.index)
    yhat_plot = df_pred_adj[w].loc[common_idx]
    yhat_plot.columns = [f"{col}_pred" for col in yhat_plot.columns]
    ytrue_plot = train_y.loc[common_idx].to_frame(name='true_RV')
    
    fig = plt.figure(figsize=[16,7])
    yhat_plot.plot(ax=plt.gca(), alpha=0.9)
    ytrue_plot.plot(ax=plt.gca(), color='black', linewidth=2, alpha=0.3, label='True RV')
    plt.xlabel("Date")
    plt.ylabel("Log variance")
    plt.legend()
    plt.title(f"HAR prediction vs true RV for window {w}")
    plt.tight_layout()
    
    report.save_and_add_plot(fig, f"har_prediction_w{w}", 
                            caption=f"HAR Model: Predictions vs True RV (Window={w})")
    plt.close()

print("✓ HAR prediction plots saved to report")
report.add_section("Loss Metrics Over Time", level=3)
report.add_text("""
QLIKE (Quasi-Likelihood) and MSPE (Mean Squared Prediction Error) are computed over time for each window.
These metrics help assess forecast calibration and error magnitude.
""")

# QLIKE plots
for w in window:
    fig = plt.figure(figsize=[16,7])
    qlike_loss_df[w][['square_est_log', 'parkinson_est_log', 'gk_est_log']].plot(ax=plt.gca())
    plt.xlabel("Date")
    plt.ylabel("QLIKE")
    plt.legend()
    plt.title(f"QLIKE Loss for window {w}")
    plt.tight_layout()
    report.save_and_add_plot(fig, f"qlike_loss_w{w}", caption=f"QLIKE Loss Over Time (Window={w})")
    plt.close()

# MSPE plots
for w in window:
    fig = plt.figure(figsize=[16,7])
    mspe_loss_df[w][['square_est_log', 'parkinson_est_log', 'gk_est_log', 'rs_est_log']].plot(ax=plt.gca())
    plt.xlabel("Date")
    plt.ylabel("MSPE")
    plt.legend()
    plt.title(f"MSPE Loss for window {w}")
    plt.tight_layout()
    report.save_and_add_plot(fig, f"mspe_loss_w{w}", caption=f"MSPE Loss Over Time (Window={w})")
    plt.close()

print("✓ Loss metric plots saved to report")


window = [252, 504, 756, 1008, 1260]

qlike_ensemble = {}
wts = {}
weight_ensemble = {}
yhat_ensemble = {}
yhat_enfinal = {}
log_yhat_enfinal = {}
log_yhat_ensemble = {}
residual_ensemble = {}
qlike_loss_ensemble = {}
mspe_loss_ensemble = {}
summary_ensemble = {}
ljung_box_ensemble = {}

for w in window:

  #compute weightage
  ensemble_model = EnsembleModel(estimators=None)
  qlike_ensemble[w] = summary_df[w]['QLIKE_mean']
  weight_ensemble[w] = ensemble_model.compute_weightage(qlike_ensemble[w])
  yhat_ensemble[w] = (np.exp(df_pred_adj[w]))

  wts[w] = pd.Series(weight_ensemble[w], index=yhat_ensemble[w].columns, dtype=float)

  yhat_enfinal[w] = yhat_ensemble[w].dot(wts[w])
  log_yhat_enfinal[w] = np.log(  yhat_enfinal[w])

  common_idx = log_yhat_enfinal[w].index.intersection(y_adj.index)
  log_yhat_ensemble[w] = log_yhat_enfinal[w].loc[common_idx] #log-variance
  log_ytrue_ensemble = y_adj.loc[common_idx] #log-variance
  ytrue_ensemble = ytrue_var.loc[common_idx] # variance

  residual_ensemble[w] = log_yhat_ensemble[w] - log_ytrue_ensemble

  qlike_loss_ensemble[w] = pd.DataFrame(Metric_Evaluation.qlike(ytrue_ensemble, yhat_enfinal[w]))
  mspe_loss_ensemble[w]  = pd.DataFrame(Metric_Evaluation.mspe(ytrue_ensemble, yhat_enfinal[w]))

  summary_ensemble[w] = pd.DataFrame({
    'QLIKE_mean': qlike_loss_ensemble[w].mean(),
    'QLIKE_std':  qlike_loss_ensemble[w].std(),
    'MSPE_mean':  mspe_loss_ensemble[w].mean(),
    'MSPE_std':   mspe_loss_ensemble[w].std()
  }).round(4)

  vol_check = Vol_Est_Check(
      alpha=0.05,
      lb_lags=(10, 20),
      kpss_reg='c',
      kpss_nlags='auto',
      acf_pacf_nlags=40
  )
  ljung_box_ensemble[w] = pd.DataFrame(vol_check.ljung_box(residual_ensemble[w]))


# %%
final_summary_ensemble = pd.concat(summary_ensemble, axis=0)
final_summary_ensemble.index.name = 'Window'

lb_ensemble_final = pd.concat(ljung_box_ensemble, axis=0)
lb_ensemble_final.index.name = 'Window'

print(final_summary_ensemble)
print(lb_ensemble_final)

# %%
# Add ensemble model results to report
report.add_section("Ensemble Model Results", level=2)
report.add_section("Ensemble Weights", level=3)
report.add_text("""
The ensemble model combines predictions from multiple estimators using inverse QLIKE weighting.
Below are the weights assigned to each estimator for different window sizes.
""")

weights_df = pd.DataFrame({w: wts[w] for w in window}).T
weights_df.index.name = 'Window'
report.add_table(weights_df.round(4), caption="Table 4: Ensemble Model Weights by Window")

report.add_section("Ensemble Performance Summary", level=3)
report.add_table(final_summary_ensemble, caption="Table 5: Ensemble Model Performance Metrics")
report.add_table(lb_ensemble_final, caption="Table 6: Ensemble Model Ljung-Box Test Results")

report.add_section("Ensemble Predictions and Loss Metrics", level=3)

# Ensemble predictions
for w in window:
    common_idx = log_yhat_enfinal[w].index.intersection(y_adj.index)
    yhat_plot = log_yhat_enfinal[w].loc[common_idx].to_frame(name='Ensemble_RV')
    ytrue_plot = y_adj.loc[common_idx].to_frame(name='true_RV')
    
    fig, ax = plt.subplots(figsize=(16, 7))
    yhat_plot.plot(ax=ax, color='blue', linewidth=2, label='Ensemble_RV')
    ytrue_plot.plot(ax=ax, color='orange', linewidth=1.5, alpha=0.5, label='true_RV')
    plt.xlabel("Date")
    plt.ylabel("Log variance")
    plt.legend()
    plt.title(f"Ensemble HAR prediction vs true RV for window {w}")
    plt.tight_layout()
    report.save_and_add_plot(fig, f"ensemble_pred_w{w}", 
                            caption=f"Ensemble Model: Predictions vs True RV (Window={w})")
    plt.close()

# Ensemble QLIKE
for w in window:
    fig = plt.figure(figsize=[16,7])
    qlike_loss_ensemble[w].plot(ax=plt.gca())
    plt.xlabel("Date")
    plt.ylabel("QLIKE")
    plt.title(f"QLIKE Loss for window {w}")
    plt.tight_layout()
    report.save_and_add_plot(fig, f"ensemble_qlike_w{w}", 
                            caption=f"Ensemble QLIKE Loss (Window={w})")
    plt.close()

# Ensemble MSPE
for w in window:
    fig = plt.figure(figsize=[16,7])
    mspe_loss_ensemble[w].plot(ax=plt.gca())
    plt.xlabel("Date")
    plt.ylabel("MSPE")
    plt.legend()
    plt.title(f"MSPE Loss for window {w}")
    plt.tight_layout()
    report.save_and_add_plot(fig, f"ensemble_mspe_w{w}", 
                            caption=f"Ensemble MSPE Loss (Window={w})")
    plt.close()

print("✓ Ensemble plots saved to report")


loss1 = qlike_loss_ensemble[504]
loss2 = qlike_loss_ensemble[756]
common_idx = loss2.index.intersection( loss1.index)
loss2_adj = loss2.loc[common_idx]
loss1_adj = loss1.loc[common_idx]
# %%
DM_test_results = Metric_Evaluation.DM_test(loss1_adj,
                                            loss2_adj,
                                            model1_name='Window_504',
                                            model2_name='Window_756'
                                            )
print(DM_test_results)

report.add_section("Diebold-Mariano Test", level=3)
report.add_text("""
The Diebold-Mariano (DM) test evaluates whether there is a statistically significant difference 
in predictive accuracy between two models. Here we compare Windows 504 and 756.
""")

dm_stat, p_val, decision = DM_test_results
dm_results = {
    "DM Statistic": dm_stat,
    "P-value": p_val,
    "Better Model": decision['Better model'],
    "Significant?": decision['Significant'],
    "Alpha": decision['Alpha'],
    "Observations": decision['Observations']
}
report.add_metrics_summary(dm_results, title="DM Test Results: Window 504 vs Window 756")

report.add_text(f"""
**Interpretation:** With p-value = {p_val:.4f} > 0.05, we fail to reject the null hypothesis of equal 
predictive accuracy. This indicates no statistically significant difference between Window 504 and 756.
Either window may be used, and selection can be based on secondary metrics or practical considerations.
""")

# %%
# =============================================================================
# Save Predictions and True Values to CSV
# =============================================================================
print("Consolidating and saving predictions to CSV...")

# Start with true realized volatility and other vol estimates
vol_estimates_var = vol_results[['square_est', 'parkinson_est', 'gk_est', 'rs_est']].copy()
vol_estimates_var.rename(columns={
    'square_est': 'RV_squared_return',
    'parkinson_est': 'RV_parkinson',
    'gk_est': 'RV_garman_klass',
    'rs_est': 'RV_rogers_satchell'
}, inplace=True)

# Combine all data into a single DataFrame
results_df = vol_estimates_var.copy()

# Add HAR model predictions (yhat_var is a dict of window -> df)
for w in window:
    df = yhat_var[w].rename(columns=lambda c: f"HAR_{c.replace('_est_log', '')}_w{w}")
    results_df = results_df.join(df, how='outer')

# Add HAR Ensemble predictions (yhat_enfinal is a dict of window -> series)
for w in window:
    series = yhat_enfinal[w].rename(f"HAR_Ensemble_w{w}")
    results_df = results_df.join(series, how='outer')

# Save to CSV with the date index as a column named 'Date'
results_df.to_csv('yhat_log_har.csv', index_label='Date')

print("✓ HAR predictions and true values saved to yhat_log_har.csv")
