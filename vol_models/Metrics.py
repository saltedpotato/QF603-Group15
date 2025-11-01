from vol_models.model_load_packages import *
from scipy.stats import norm

class Metric_Evaluation:
  def __init__(self, ytrue, y_pred, alpha):
    self.y_pred = y_pred
    self.ytrue = ytrue
    self.alpha = alpha

  def mspe(ytrue, ypred):
    return ((ytrue - ypred) / ytrue) ** 2

  def qlike(ytrue, ypred):
    return np.log(ypred) + ytrue / ypred

  def rmse(ytrue, ypred, window = 22):
    errors = (ytrue - ypred)**2
    rolling_mean = pd.Series(errors).rolling(window = window).mean()
    return np.sqrt(rolling_mean)

  def diebold_mariano_test(y_true, pred1, pred2, h=1, loss_type='MSE'):
    # """
    # Diebold-Mariano test for equal predictive accuracY
    # Parameters:
    # y_true: actual values
    # pred1: predictions from model 1
    # pred2: predictions from model 2
    # h: forecast horizon
    # loss_type: 'MSE' or 'QLIKE'
    # """
    if loss_type == 'MSE':
        loss1 = (y_true - pred1) ** 2
        loss2 = (y_true - pred2) ** 2
    elif loss_type == 'QLIKE':
        loss1 = np.log(pred1) + y_true / pred1
        loss2 = np.log(pred2) + y_true / pred2
    else:
        raise ValueError("loss_type must be 'MSE' or 'QLIKE'")

    d = loss1 - loss2
    d = np.asarray(d).flatten()
    d_mean = np.mean(d)
    n = len(d)
    d_centered = d - d_mean

    # HAC variance estimator
    L = np.floor(1.5*n**(1/3))
    gamma_0 = np.mean(d_centered**2)
    for lag in range(1, L+1):
      gamma_lag = np.mean(d_centered[:-lag] * d_centered[lag:])
      w = 1 - lag / (L + 1)
      gamma_0 += 2 * w * gamma_lag

    dm_stat = d_mean / np.sqrt(gamma_0 / n)
    p_value = 2 * (1 - norm.cdf(abs(dm_stat)))

    return dm_stat, p_value


  def DM_test(loss1, loss2, alpha = 0.05, model1_name = 'Model 1', model2_name = 'Model_2'): #confirm using qlike
    d = loss1 - loss2
    d = np.asarray(d).flatten()
    d_mean = np.mean(d)
    n = len(d)
    d_centered = d - d_mean

    # HAC variance estimator
    L = int(np.floor(1.5*n**(1/3)))
    gamma_0 = np.mean(d_centered**2)
    for lag in range(1, L+1):
      gamma_lag = np.mean(d_centered[:-lag] * d_centered[lag:])
      w = 1 - lag / (L + 1)
      gamma_0 += 2 * w * gamma_lag

    dm_stat = d_mean / np.sqrt(gamma_0 / n)
    p_value = 2 * (1 - norm.cdf(abs(dm_stat)))

    #decision logic
    if p_value < alpha:
       winner = model1_name if dm_stat < 0 else model2_name
       significant = True
    else:
       winner = 'None (No significant difference)'
       significant = False
    decision ={
       'Better model': winner,
       'Significant': significant,
       'Alpha': alpha,
       'Observations': n
    }
    return dm_stat, p_value, decision

