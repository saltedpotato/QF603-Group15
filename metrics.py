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


# Main Evaluation Function
def comprehensive_evaluation(y_test, y_pred, benchmark_pred=None):
    """
    Comprehensive evaluation of forecasting model
    
    Parameters:
    y_test: actual values
    y_pred: model predictions
    benchmark_pred: benchmark model predictions (for DM test)
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