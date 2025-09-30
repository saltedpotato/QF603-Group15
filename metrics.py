import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import norm, chi2
import warnings
warnings.filterwarnings('ignore')

# Statistical Metrics
def mspe(y_true, y_pred):
    """
    Mean Squared Percentage Error
    """
    return np.mean(((y_true - y_pred) / y_true) ** 2)

def qlike(y_true, y_pred):
    """
    QLIKE loss function (commonly used for volatility forecasting)
    """
    return np.mean(np.log(y_pred) + y_true / y_pred)

# Diebold-Mariano Test
def diebold_mariano_test(y_true, pred1, pred2, h=1, loss_type='MSE'):
    """
    Diebold-Mariano test for equal predictive accuracy
    
    Parameters:
    y_true: actual values
    pred1: predictions from model 1
    pred2: predictions from model 2  
    h: forecast horizon
    loss_type: 'MSE' or 'QLIKE'
    """
    if loss_type == 'MSE':
        loss1 = (y_true - pred1) ** 2
        loss2 = (y_true - pred2) ** 2
    elif loss_type == 'QLIKE':
        loss1 = np.log(pred1) + y_true / pred1
        loss2 = np.log(pred2) + y_true / pred2
    else:
        raise ValueError("loss_type must be 'MSE' or 'QLIKE'")
    
    d = loss1 - loss2
    d_mean = np.mean(d)
    
    # HAC estimator for variance
    n = len(d)
    gamma_0 = np.var(d, ddof=1)
    
    # Calculate autocovariances
    max_lags = h - 1
    for lag in range(1, max_lags + 1):
        gamma_0 += 2 * np.cov(d[lag:], d[:-lag])[0, 1]
    
    dm_stat = d_mean / np.sqrt(gamma_0 / n)
    p_value = 2 * (1 - norm.cdf(abs(dm_stat)))
    
    return dm_stat, p_value

# Model Confidence Set (Simplified Implementation)
def model_confidence_set(y_true, predictions_dict, alpha=0.05, B=1000):
    """
    Simplified Model Confidence Set implementation
    
    Parameters:
    y_true: actual values
    predictions_dict: dictionary of model_name: predictions
    alpha: significance level
    B: number of bootstrap samples
    """
    models = list(predictions_dict.keys())
    n_models = len(models)
    n_obs = len(y_true)
    
    # Calculate losses for each model
    losses = {}
    for model_name, pred in predictions_dict.items():
        losses[model_name] = np.log(pred) + y_true / pred  # QLIKE loss
    
    # Create loss matrix
    loss_matrix = np.column_stack([losses[model] for model in models])
    
    # Calculate relative performance
    best_model = min(losses, key=lambda x: np.mean(losses[x]))
    
    # Simplified MCS procedure (this is a basic implementation)
    mcs_results = {}
    for model in models:
        if model == best_model:
            mcs_results[model] = {'in_mcs': True, 'p_value': 1.0}
        else:
            # Simple test: check if significantly worse than best model
            loss_diff = losses[model] - losses[best_model]
            t_stat = np.mean(loss_diff) / (np.std(loss_diff) / np.sqrt(n_obs))
            p_val = 2 * (1 - norm.cdf(abs(t_stat)))
            mcs_results[model] = {'in_mcs': p_val > alpha, 'p_value': p_val}
    
    return mcs_results, best_model

# Value-at-Risk Backtesting
def kupiec_test(returns, var, alpha=0.05):
    """
    Kupiec's unconditional coverage test for VaR
    """
    violations = returns < -var
    n = len(returns)
    x = np.sum(violations)  # Number of violations
    p_hat = x / n  # Empirical violation rate
    
    likelihood_ratio = -2 * np.log(
        ((1 - alpha) ** (n - x) * alpha ** x) / 
        ((1 - p_hat) ** (n - x) * p_hat ** x)
    )
    
    p_value = 1 - chi2.cdf(likelihood_ratio, 1)
    
    return {
        'violations': x,
        'violation_rate': p_hat,
        'expected_rate': alpha,
        'LR_stat': likelihood_ratio,
        'p_value': p_value
    }

def christoffersen_test(returns, var, alpha=0.05):
    """
    Christoffersen's conditional coverage test for VaR
    """
    violations = returns < -var
    n = len(violations)
    
    # Transition counts
    n00 = n01 = n10 = n11 = 0
    for i in range(1, n):
        if violations[i-1] == 0 and violations[i] == 0:
            n00 += 1
        elif violations[i-1] == 0 and violations[i] == 1:
            n01 += 1
        elif violations[i-1] == 1 and violations[i] == 0:
            n10 += 1
        elif violations[i-1] == 1 and violations[i] == 1:
            n11 += 1
    
    # Unconditional coverage (same as Kupiec)
    x = np.sum(violations)
    p_hat = x / n
    LR_uc = -2 * np.log(
        ((1 - alpha) ** (n - x) * alpha ** x) / 
        ((1 - p_hat) ** (n - x) * p_hat ** x)
    )
    
    # Independence test
    if n00 + n01 > 0 and n10 + n11 > 0:
        pi0 = n01 / (n00 + n01)
        pi1 = n11 / (n10 + n11)
        pi = (n01 + n11) / (n00 + n01 + n10 + n11)
        
        LR_ind = -2 * np.log(
            ((1 - pi) ** (n00 + n10) * pi ** (n01 + n11)) / 
            ((1 - pi0) ** n00 * pi0 ** n01 * (1 - pi1) ** n10 * pi1 ** n11)
        )
    else:
        LR_ind = 0
    
    # Conditional coverage
    LR_cc = LR_uc + LR_ind
    p_value_cc = 1 - chi2.cdf(LR_cc, 2)
    
    return {
        'LR_conditional': LR_cc,
        'p_value_conditional': p_value_cc,
        'LR_independence': LR_ind,
        'LR_unconditional': LR_uc
    }

# Economic Utility via Volatility Targeting
def volatility_targeting_portfolio(returns, volatility_forecast, target_volatility=0.15):
    """
    Calculate economic utility through volatility targeting
    
    Parameters:
    returns: asset returns
    volatility_forecast: predicted volatility
    target_volatility: annual target volatility
    """
    # Scale returns based on volatility forecast
    scaling_factor = target_volatility / (volatility_forecast * np.sqrt(252))
    scaled_returns = returns * np.minimum(scaling_factor, 1)  # Cap at 1 (no leverage)
    
    # Calculate performance metrics
    annual_return = np.mean(scaled_returns) * 252
    annual_volatility = np.std(scaled_returns) * np.sqrt(252)
    sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
    
    # Certainty equivalent return (assuming CRRA utility with gamma=2)
    ce_return = np.mean(scaled_returns) - 0.5 * np.var(scaled_returns)
    
    return {
        'annual_return': annual_return,
        'annual_volatility': annual_volatility,
        'sharpe_ratio': sharpe_ratio,
        'certainty_equivalent': ce_return,
        'scaled_returns': scaled_returns
    }

# Main Evaluation Function
def comprehensive_evaluation(y_test, y_pred, benchmark_pred=None, returns=None, 
                           volatility_forecast=None, alpha=0.05):
    """
    Comprehensive evaluation of forecasting model
    
    Parameters:
    y_test: actual values
    y_pred: model predictions
    benchmark_pred: benchmark model predictions (for DM test)
    returns: return series (for VaR backtesting)
    volatility_forecast: volatility predictions (for economic utility)
    alpha: significance level for VaR
    """
    
    results = {}
    
    # Statistical Metrics
    print("=== STATISTICAL METRICS ===")
    results['MSPE'] = mspe(y_test, y_pred)
    results['QLIKE'] = qlike(y_test, y_pred)
    
    print(f"MSPE: {results['MSPE']:.6f}")
    print(f"QLIKE: {results['QLIKE']:.6f}")
    
    # Comparative Tests
    if benchmark_pred is not None:
        print("\n=== COMPARATIVE TESTS ===")
        
        # Diebold-Mariano Test
        dm_stat, dm_pvalue = diebold_mariano_test(y_test, y_pred, benchmark_pred, loss_type='QLIKE')
        results['DM_statistic'] = dm_stat
        results['DM_pvalue'] = dm_pvalue
        
        print(f"Diebold-Mariano Test (QLIKE):")
        print(f"  Statistic: {dm_stat:.4f}, p-value: {dm_pvalue:.4f}")
        
        # Model Confidence Set
        predictions_dict = {
            'Model': y_pred,
            'Benchmark': benchmark_pred
        }
        mcs_results, best_model = model_confidence_set(y_test, predictions_dict)
        results['MCS'] = mcs_results
        results['Best_Model'] = best_model
        
        print(f"Model Confidence Set:")
        for model, result in mcs_results.items():
            status = "IN" if result['in_mcs'] else "OUT"
            print(f"  {model}: {status} (p-value: {result['p_value']:.4f})")
    
    # Economic Metrics
    if returns is not None and volatility_forecast is not None:
        print("\n=== ECONOMIC METRICS ===")
        
        # VaR Backtesting
        var_level = np.percentile(returns, alpha * 100)
        var_forecast = -volatility_forecast * norm.ppf(alpha)
        
        kupiec_results = kupiec_test(returns, var_forecast, alpha)
        christoffersen_results = christoffersen_test(returns, var_forecast, alpha)
        
        results['VaR_Kupiec'] = kupiec_results
        results['VaR_Christoffersen'] = christoffersen_results
        
        print("VaR Backtesting Results:")
        print(f"  Violations: {kupiec_results['violations']}/{len(returns)} "
              f"({kupiec_results['violation_rate']:.3%})")
        print(f"  Kupiec Test p-value: {kupiec_results['p_value']:.4f}")
        print(f"  Christoffersen Test p-value: {christoffersen_results['p_value_conditional']:.4f}")
        
        # Economic Utility
        utility_results = volatility_targeting_portfolio(returns, volatility_forecast)
        results['Economic_Utility'] = utility_results
        
        print("Economic Utility (Volatility Targeting):")
        print(f"  Annual Return: {utility_results['annual_return']:.3%}")
        print(f"  Annual Volatility: {utility_results['annual_volatility']:.3%}")
        print(f"  Sharpe Ratio: {utility_results['sharpe_ratio']:.4f}")
        print(f"  Certainty Equivalent: {utility_results['certainty_equivalent']:.6f}")
    
    return results

# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    n = 1000
    
    # True volatility process (GARCH-like)
    true_vol = np.zeros(n)
    true_vol[0] = 0.01
    for t in range(1, n):
        true_vol[t] = 0.1 + 0.8 * true_vol[t-1] + 0.1 * np.random.normal(0, 0.01)
    
    # Returns and predictions
    returns = np.random.normal(0, true_vol)
    y_test = returns ** 2  # Realized variance
    y_pred = true_vol ** 2 * 0.9  # Model predictions (slightly biased)
    benchmark_pred = true_vol ** 2 * 1.1  # Benchmark predictions
    
    # Run comprehensive evaluation
    results = comprehensive_evaluation(
        y_test=y_test,
        y_pred=y_pred,
        benchmark_pred=benchmark_pred,
        returns=returns,
        volatility_forecast=np.sqrt(y_pred)
    )