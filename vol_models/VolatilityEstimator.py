from vol_models.model_load_packages import *

class volatility_estimator:
    def __init__(self, add_log):
        self.add_log = add_log

    def _check(self, df):
        required = ['High', 'Low', 'Open', 'Close']
        if not set(required).issubset (df.columns):
            raise ValueError(f"Dataframe needs columns {required}.")
        if (df[required]<=0).any().any():
            raise ValueError(f"Dataframe contains nonpositive values")
        return df

    def compute_square_return(self,df):
        df = self._check(df)
        log_return =  np.log(df['Close'] / df['Close'].shift(1))
        return 252*(log_return ** 2)

    def compute_parkinson_estimator(self,df):
        df = self._check(df)
        log_par_var = (np.log(df['High'] / df['Low']))**2
        return 252*((1/(4*np.log(2))) * log_par_var)

    def compute_gk_estimator(self,df):
        df = self._check(df)
        gk_var_1 = (1/2)*(np.log(df['High']/df['Low']))**2
        gk_var_2 = (2*np.log(2)-1)*(np.log(df['Close']/df['Open']))**2
        return 252*(gk_var_1 - gk_var_2)

    def compute_rs_estimator(self, df):
        df = self._check(df)
        rs_var_1 = (np.log(df['High']/df['Open']))*(np.log(df['High']/df['Close']))
        rs_var_2 = (np.log(df['Low']/df['Open']))*(np.log(df['Low']/df['Close']))
        return 252*(rs_var_1 + rs_var_2)

    def compute_all(self,df, lag_for_predictors:bool=False):
          df = self._check(df).copy()
          eps = 1e-12

          out = pd.DataFrame(index = df.index)
          out['square_est'] = self.compute_square_return(df)
          out['parkinson_est']=self.compute_parkinson_estimator(df)
          out['gk_est'] = self.compute_gk_estimator(df)
          out['rs_est'] = self.compute_rs_estimator(df)

          if self.add_log:
              for col in ['square_est', 'parkinson_est', 'gk_est', 'rs_est']:
                  x = out[col].astype(float).replace([np.inf, -np.inf], np.nan)
                  out[col + '_log'] = np.log(x.clip(lower=eps))
          if lag_for_predictors:
            out = out.shift(1)

          return out