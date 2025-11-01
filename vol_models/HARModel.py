from vol_models.model_load_packages import *
import statsmodels.api as sm

class HAR_Model:
  def __init__(self, y_log_col, exo_col, lags =[1,5,22]):
    self.y_log_col = y_log_col
    self.exo_col = exo_col
    self.lags = lags #daily, weekly, monthly

  def features(self, df):
    y_pred = df[self.y_log_col]
    out = pd.DataFrame(index=df.index)
    out['rv_d'] = y_pred
    out['rv_w'] = y_pred.rolling(self.lags[1], min_periods = self.lags[1]).mean()
    out['rv_m'] = y_pred.rolling(self.lags[2], min_periods = self.lags[2]).mean()

    if self.exo_col:
      for col in self.exo_col:
        out[f'x_{col}'] =df[col]

    return out.dropna()

  def fit_predict(self,
                  x_train,
                  y_train,
                  window):

    resid_full = pd.Series(index=y_train.index, data=np.nan)
    yhat_full = pd.Series(index=y_train.index, data=np.nan)
    residual_raw = pd.Series(index=y_train.index, data=np.nan)
    for t in range(window, len(y_train)):
      y_slice = y_train.iloc[t-window:t]
      x_slice = x_train.iloc[t-window:t]

      common_idx = x_slice.index.intersection(y_slice.index)
      y_slice = y_train.loc[common_idx]
      x_slice = x_train.loc[common_idx]

      model = sm.OLS(y_slice, sm.add_constant(x_slice)).fit()

      x_next = pd.DataFrame([x_train.iloc[t]])
      x_next = sm.add_constant(x_next, has_constant='add')

      yhat_full.iloc[t] = model.predict(x_next).iloc[0]
      resid_full.iloc[t] = model.resid.var(ddof=x_slice.shape[1]) #ddof degree of freedom correction for unbiased variance
      residual_raw.iloc[t] = yhat_full.iloc[t] - y_train.iloc[t]

    return yhat_full, resid_full, residual_raw

