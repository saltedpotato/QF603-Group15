# premodel diagnotics on the data assumptions.
from statsmodels.tsa.stattools import adfuller, kpss, acf as sm_acf, pacf as sm_pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings

from vol_models.model_load_packages import *

class Vol_Est_Check:

    def __init__(self,
                 alpha,
                 lb_lags,
                 kpss_reg,
                 kpss_nlags,
                 acf_pacf_nlags):
        # alpha: significant level
        # lb_lags: lags to report to Ljung box
        # kpss: reg[c - level, ct - trend], nlags: auto or int
        #ADF passed stationary when p<0.05 (reject H_0 of unit root)
        self.alpha = alpha
        self.lb_lags = tuple(lb_lags)
        self.kpss_reg = kpss_reg
        self.kpss_nlags = kpss_nlags
        self.acf_pacf_nlags = acf_pacf_nlags

    def ADF(self, df, name):
        df = df.dropna()
        series_name = name or getattr (df, 'name', 'series')

        stat, p, lags, nobs, crit, icbest = adfuller(df, autolag = 'AIC')
        stationary_flag = p <= self.alpha
        return {"adf_stat": stat,
                "adf_p": p,
                "adf_lags": lags,
                "adf_nobs": nobs,
                "adf_crit": crit,
                'adf_icbest': icbest,
                'adf_stationary_flag': stationary_flag,
                'adf_decision': (f'{series_name}: Reject H0 -> stationary'
                    if stationary_flag
                    else 'Fail to reject H0 -> non-stationary'
                )}


    def KPSS(self,df, name, nlags):
        df = df.dropna()
        series_name = name or getattr (df, 'name', 'series')

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            stat, p, lags, crit = kpss(df,
                                       regression=self.kpss_reg,
                                       nlags=nlags)

        stationary_flag = (p > self.alpha)
        return {
            "kpss_stat": stat, # stat < crit -> series is stationary
            "kpss_p": p, # p >0.05  -> series is stationary
            "kpss_lags": lags,
            "kpss_crit": crit,
            'kpss_reg': self.kpss_reg,
            'kpss_stationary_flag': stationary_flag,
            'kpss_decision': (f'{series_name}: Fail to reject H0 ->stationary'
            if stationary_flag
            else f"{series_name}: Reject H0 -> Non-stationary")
            }


    def ljung_box(self, df): # reject H0 -> serial correlation
        df = df.dropna()
        lb = acorr_ljungbox(df, lags=list(self.lb_lags), return_df=True)
        out={}
        for L in self.lb_lags:
            out[f'lb_stat_{L}'] = float(lb.loc[L, "lb_stat"]) # stat for each lag h
            out[f"lb_p_{L}"]   = float(lb.loc[L, "lb_pvalue"])  # p-value

        out['white_noise_flag'] = all(out[f'lb_p_{L}'] > self.alpha for L in self.lb_lags)
        out["lb_lags_used"] = self.lb_lags
        out['n_obs'] = len(df)
        out['name'] = getattr(df, 'name', 'series')
        return out

    def compute_acf(self, df, nlags, alpha):
        df=df.dropna()
        nlags = nlags or self.acf_pacf_nlags
        vals, conf = sm_acf(df,
                            nlags = nlags,
                            alpha = alpha,
                            fft = True,
                            adjusted = False)
        return {'acf_vals': vals,
                'acf_confint': conf} #shape(nlags+1,2)


    def compute_pacf(self, df, nlags, alpha, method: str = 'ywmle'):
        df=df.dropna()
        nlags = nlags or self.acf_pacf_nlags
        vals, conf = sm_pacf(df,
                             nlags=nlags,
                             alpha=alpha,
                             method=method)
        return {
            'pacf_vals': vals,
            'pacf_confint': conf,
            'pacf_method': method
            }

    def plot_acf(self, df, nlags, title:str = None):
        df = df.dropna()
        nlags = nlags or self.acf_pacf_nlags
        plot_acf(df, lags=nlags)
        plt.title(title or f"ACF ({getattr(df,'name','series')})")
        plt.show()

    def plot_pacf(self, df, nlags, title: str = None, method: str = 'ywmle'):
        df = df.dropna()
        nlags = nlags or self.acf_pacf_nlags
        plot_pacf(df, lags=nlags)
        plt.title(title or f"PACF ({getattr(df,'name','series')})")
        plt.show()

    def summarize_series(self, df, name = None): #for 1 estimator
        series_name = name or getattr (df, 'name', 'series')
        results = {}
        results.update(self.ADF(df, name = name))
        results.update(self.KPSS(df, name = name, nlags=self.kpss_nlags))

        lb = self.ljung_box(df)
        results.update({f"lb_{k}": v for k, v in lb.items()})

        return results
