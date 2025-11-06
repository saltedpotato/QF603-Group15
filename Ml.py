# %%
# Core imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from pathlib import Path
from datetime import datetime
import os
import yfinance as yf
from joblib import Parallel, delayed
import multiprocessing

# ML model imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# TFT imports
import torch
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import RMSE, MAE, SMAPE, QuantileLoss

# ADF test
from statsmodels.tsa.stattools import adfuller

# Train test split
from sklearn.model_selection import train_test_split

# Metrics
from sklearn.metrics import mean_squared_error, r2_score

# ACF/PACF plots
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

try:
    # Try new Lightning 2.x import structure
    from lightning.pytorch import Trainer
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
except ImportError:
    # Fall back to old pytorch_lightning import
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

import warnings
warnings.filterwarnings('ignore')

# %%
# Import volatility models from vol_models package
from vol_models.VolatilityReportGenerator import VolatilityReportGenerator
from vol_models.VolatilityEstimator import volatility_estimator
from vol_models.VolEstCheck import *
from vol_models.HARModel import HAR_Model
from vol_models.EnsembleModel import EnsembleModel
from vol_models.Metrics import Metric_Evaluation

print("✓ All libraries imported successfully")



# %%
def load_data():
    """
    Load all necessary data for ML/TFT training.
    
    Returns:
    --------
    dict : Dictionary containing all loaded data
    """
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    data_dir = Path("./data")

    # Load TLT OHLC data: prefer local CSV, fallback to yfinance if required columns missing
    csv_path = data_dir / "TLT_2007-01-01_to_2025-08-30.csv"
    required_cols = {"Open", "High", "Low", "Close"}
    if csv_path.exists():
        try:
            print(f"Loading TLT OHLC data from CSV ({csv_path})...", end=" ")
            tlt_ohlc = pd.read_csv(csv_path)
            # Accept common date column names
            if 'Date' in tlt_ohlc.columns:
                tlt_ohlc['Date'] = pd.to_datetime(tlt_ohlc['Date'])
                tlt_ohlc = tlt_ohlc.set_index('Date')
            elif 'date' in tlt_ohlc.columns:
                tlt_ohlc['date'] = pd.to_datetime(tlt_ohlc['date'])
                tlt_ohlc = tlt_ohlc.set_index('date')
            # If CSV provides a single 'Price' column, treat as Close
            if 'Price' in tlt_ohlc.columns and 'Close' not in tlt_ohlc.columns:
                tlt_ohlc = tlt_ohlc.rename(columns={'Price': 'Close'})
            # If Close exists but Open missing, create Open from previous Close
            if 'Close' in tlt_ohlc.columns and 'Open' not in tlt_ohlc.columns:
                tlt_ohlc['Open'] = tlt_ohlc['Close'].shift(1)
            # Check for required OHLC columns; otherwise fallback to yfinance
            missing_required = required_cols.difference(tlt_ohlc.columns)
            if missing_required:
                print(f"✗ missing columns {missing_required}; falling back to yfinance")
                csv_path = None
            else:
                tlt_ohlc = tlt_ohlc.sort_index()
                tlt_ohlc = tlt_ohlc.loc[:'2024-12-30']
                print(f"✓ {tlt_ohlc.shape}")
        except Exception as e:
            print(f"✗ CSV load failed ({e}), falling back to yfinance")
            csv_path = None
    else:
        csv_path = None

    if csv_path is None:
        print("Downloading TLT OHLC data from Yahoo Finance...", end=" ")
        try:
            start_date  = "2003-01-01"
            end_date = "2025-08-30"
            tlt_df = yf.download(
                "TLT",
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=False,
                group_by="ticker"
            )
            if tlt_df.empty:
                raise RuntimeError("Yahoo Finance returned an empty dataframe for TLT")

            # Handle possible MultiIndex columns (e.g., ('TLT', 'Open'))
            if isinstance(tlt_df.columns, pd.MultiIndex):
                if 'Ticker' in tlt_df.columns.names:
                    tlt_df = tlt_df.droplevel('Ticker', axis=1)
                else:
                    tlt_df = tlt_df.droplevel(0, axis=1)

            tlt_df.index.name = 'Date'
            tlt_ohlc = tlt_df.copy()

            # yfinance sometimes names adjusted close differently
            if 'Close' not in tlt_ohlc.columns and 'Adj Close' in tlt_ohlc.columns:
                tlt_ohlc = tlt_ohlc.rename(columns={'Adj Close': 'Close'})

            missing_required = required_cols.difference(tlt_ohlc.columns)
            if missing_required:
                raise RuntimeError(f"Yahoo Finance data missing columns {missing_required}")

            # If Open missing, create from previous Close
            if 'Open' not in tlt_ohlc.columns and 'Close' in tlt_ohlc.columns:
                tlt_ohlc['Open'] = tlt_ohlc['Close'].shift(1)

            tlt_ohlc = tlt_ohlc.sort_index()
            tlt_ohlc = tlt_ohlc.loc[:'2024-12-30']
            print(f"✓ {tlt_ohlc.shape}")
        except Exception as e:
            raise RuntimeError(f"Failed to load TLT data from CSV and yfinance: {e}")
    
    # Compute volatility estimators using the same method as stat_model_r6.py
    print("Computing volatility estimators...", end=" ")
    vol_calc = volatility_estimator(add_log=True)
    vol_results = vol_calc.compute_all(tlt_ohlc, lag_for_predictors=True)
    vol_results = vol_results.dropna()
    print(f"✓ {vol_results.shape}")
    
    # Rename vol_results to tlt_rv for consistency with rest of code
    tlt_rv = vol_results
    
    # Load exogenous variables
    print("Loading exogenous variables...")
    exo_data = {}
    exo_files = {
        'UST10Y': 'UST10Y.csv',
        'HYOAS': 'HYOAS.csv',
        'TermSpread_10Y_2Y': 'TermSpread_10Y_2Y.csv',
        'VIX': 'VIX.csv',
        'Breakeven10Y': 'Breakeven10Y.csv'
    }
    
    for name, filename in exo_files.items():
        df = pd.read_csv(data_dir / filename)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        exo_data[name] = df.iloc[:, 0]  # Get first column after Date
        print(f"  {name:20s} ✓ {df.shape}")
    
    # Combine exogenous data
    exo_combined = pd.concat([exo_data[name] for name in exo_files.keys()], axis=1)
    exo_combined.columns = list(exo_files.keys())
    
    # Split data (same as main analysis)
    split_date = '2023-01-01'
    
    # Training data
    train_rv = tlt_rv[tlt_rv.index < split_date].copy()
    train_exo = exo_combined[exo_combined.index < split_date].copy()
    
    # Test data
    test_rv = tlt_rv[tlt_rv.index >= split_date].copy()
    test_exo = exo_combined[exo_combined.index >= split_date].copy()
    
    print(f"\n✓ Data split at {split_date}")
    print(f"  Training period: {train_rv.index.min()} to {train_rv.index.max()}")
    print(f"  Test period: {test_rv.index.min()} to {test_rv.index.max()}")
    print(f"  Training samples: {len(train_rv)}")
    print(f"  Test samples: {len(test_rv)}")
    
    # Prepare features (log variance estimators)
    estimators = ['square_est_log', 'parkinson_est_log', 'gk_est_log', 'rs_est_log']
    
    train_x = train_rv[estimators].copy()
    train_y = train_rv['square_est_log'].copy()  # Target is square estimator
    
    test_x = test_rv[estimators].copy()
    test_y = test_rv['square_est_log'].copy()  # Target is square estimator
    
    print("\n✓ All data loaded successfully")
    print("="*80)
    
    return {
        'train_x': train_x,
        'train_y': train_y,
        'train_exo': train_exo,
        'test_x': test_x,
        'test_y': test_y,
        'test_exo': test_exo,
        'test_rv': test_rv,
        'estimators': estimators,
        'exo_cols': list(exo_files.keys())
    }

print("✓ Data loading functions ready")


# %%
class ML_Volatility_Model:
    """
    Machine Learning model wrapper for volatility forecasting.
    Supports multiple ML algorithms with rolling window prediction.
    """
    
    def __init__(self, model_type='xgboost', model_params=None):
        """
        Initialize ML model.
        
        Parameters:
        -----------
        model_type : str
            Type of model: 'rf', 'gbm', 'xgboost', 'lightgbm', 'catboost'
        model_params : dict
            Model-specific hyperparameters
        """
        self.model_type = model_type
        self.model_params = model_params or {}
        self.model = None
        
    def _get_model(self):
        """Initialize the appropriate model based on model_type with GPU support"""
        # Detect GPU availability (check both hardware and environment variable)
        gpu_available = torch.cuda.is_available() and os.environ.get('ML_USE_GPU', '1') == '1'
        
        if self.model_type == 'rf':
            return RandomForestRegressor(
                n_estimators=self.model_params.get('n_estimators', 100),
                max_depth=self.model_params.get('max_depth', 10),
                min_samples_split=self.model_params.get('min_samples_split', 5),
                random_state=42,
                n_jobs=-1  # Use all CPU cores
            )
        elif self.model_type == 'gbm':
            return GradientBoostingRegressor(
                n_estimators=self.model_params.get('n_estimators', 100),
                max_depth=self.model_params.get('max_depth', 5),
                learning_rate=self.model_params.get('learning_rate', 0.1),
                random_state=42
            )
        elif self.model_type == 'xgboost':
            # XGBoost with GPU support (XGBoost 3.1+ uses 'device' parameter)
            xgb_params = {
                'n_estimators': self.model_params.get('n_estimators', 100),
                'max_depth': self.model_params.get('max_depth', 5),
                'learning_rate': self.model_params.get('learning_rate', 0.1),
                'subsample': self.model_params.get('subsample', 0.8),
                'colsample_bytree': self.model_params.get('colsample_bytree', 0.8),
                'random_state': 42,
                'n_jobs': -1,
                'tree_method': 'hist',  # Fast histogram-based algorithm
            }
            # Enable GPU if available (use 'device' parameter for XGBoost 3.1+)
            if gpu_available:
                xgb_params['device'] = 'cuda:0'  # Use first GPU device
                xgb_params['tree_method'] = 'hist'  # 'hist' works on both CPU and GPU
            return xgb.XGBRegressor(**xgb_params)
            
        elif self.model_type == 'lightgbm':
            # LightGBM with GPU support
            lgb_params = {
                'n_estimators': self.model_params.get('n_estimators', 100),
                'max_depth': self.model_params.get('max_depth', 5),
                'learning_rate': self.model_params.get('learning_rate', 0.1),
                'subsample': self.model_params.get('subsample', 0.8),
                'colsample_bytree': self.model_params.get('colsample_bytree', 0.8),
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1,
            }
            # Enable GPU if available
            if gpu_available:
                lgb_params['device'] = 'gpu'
                lgb_params['gpu_platform_id'] = 0
                lgb_params['gpu_device_id'] = 0
            return lgb.LGBMRegressor(**lgb_params)
            
        elif self.model_type == 'catboost':
            # CatBoost with GPU support
            cb_params = {
                'iterations': self.model_params.get('n_estimators', 100),
                'depth': self.model_params.get('max_depth', 5),
                'learning_rate': self.model_params.get('learning_rate', 0.1),
                'random_state': 42,
                'verbose': False,
                'thread_count': -1,  # Use all CPU cores
            }
            # Enable GPU if available
            if gpu_available:
                cb_params['task_type'] = 'GPU'
                cb_params['devices'] = '0'
            return cb.CatBoostRegressor(**cb_params)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit_predict_rolling(self, X_data, y_data, window, train_every_k=20):
        """
        Perform rolling window prediction - train every K days instead of every day.
        
        This creates a massive speedup by training fewer models while still capturing
        rolling window dynamics. Instead of training a model for every single time step,
        we train every K days and use that model to predict the next K days.
        
        Parameters:
        -----------
        X_data : pd.DataFrame
            FULL feature matrix (not split - we use all data with rolling window)
        y_data : pd.Series
            FULL target variable (log variance) - we use all data with rolling window
        window : int
            Rolling window size (e.g., 252, 504, 756)
        train_every_k : int
            Train a new model every K days (default=5 for 5x speedup)
            
        Returns:
        --------
        yhat_full : pd.Series
            Predictions (log variance) for all time points >= window
        residual_raw : pd.Series
            Raw residuals (predicted - actual)
        """
        # Initialize prediction arrays with NaN (same as stat model)
        yhat_full = pd.Series(index=y_data.index, data=np.nan)
        residual_raw = pd.Series(index=y_data.index, data=np.nan)
        
        # Rolling window loop - train every K days instead of every day
        for t in range(window, len(y_data), train_every_k):
            # Extract training window [t-window:t]
            y_slice = y_data.iloc[t-window:t]
            x_slice = X_data.iloc[t-window:t]
            
            # Align indices (defensive programming)
            common_idx = x_slice.index.intersection(y_slice.index)
            y_slice = y_data.loc[common_idx]
            x_slice = X_data.loc[common_idx]
            
            # Train fresh model on this window
            self.model = self._get_model()
            self.model.fit(x_slice, y_slice)
            
            # Predict for the NEXT K days (or until end of data)
            predict_end = min(t + train_every_k, len(y_data))
            for pred_t in range(t, predict_end):
                x_next = X_data.iloc[pred_t:pred_t+1]
                yhat_full.iloc[pred_t] = self.model.predict(x_next)[0]
                
                # Calculate residual
                residual_raw.iloc[pred_t] = yhat_full.iloc[pred_t] - y_data.iloc[pred_t]
        
        return yhat_full, residual_raw

# %%
def _train_single_model(model_name, model_params, X_combined, y_combined, window, use_gpu=True, train_every_k=5):
    """
    Helper function to train a single model - designed for parallel execution.
    
    Parameters:
    -----------
    model_name : str
        Model type ('rf', 'gbm', 'xgboost', 'lightgbm', 'catboost')
    model_params : dict
        Model hyperparameters
    X_combined : pd.DataFrame
        Combined feature matrix
    y_combined : pd.Series
        Target variable
    window : int
        Rolling window size
    use_gpu : bool
        Whether to attempt GPU usage (can be disabled for parallel CPU training)
    train_every_k : int
        Train a new model every K days (default=5 for 5x speedup)
        
    Returns:
    --------
    tuple : (model_name, window, results_dict, training_time, n_predictions)
    """
    model_start_time = time.time()
    
    try:
        # Train ONE model with ALL features
        ml_model = ML_Volatility_Model(
            model_type=model_name,
            model_params=model_params
        )
        
        y_pred, residual_raw = ml_model.fit_predict_rolling(
            X_combined, 
            y_combined, 
            window=window
        )
        
        # Store results
        results = {
            'predictions': y_pred.dropna(),      # Predictions (log scale)
            'residuals': residual_raw.dropna(),  # Residuals
            'y_true': y_combined                 # Full target
        }
        
        model_time = time.time() - model_start_time
        n_predictions = y_pred.notna().sum()
        
        return model_name, window, results, model_time, n_predictions
        
    except Exception as e:
        # If GPU training fails in parallel, this provides better error context
        print(f"\n⚠ Error training {model_name} (window={window}): {str(e)}")
        raise

def train_ml_models_rolling_window(full_x, full_y, full_exo, report, windows=[252, 504, 756], n_jobs=-1, use_gpu=True, train_every_k=5):
    """
    Train ML models using ROLLING WINDOW approach with ALL features.
    Uses parallel processing to train multiple models simultaneously.
    
    Key difference from stat models:
    - ONE model per algorithm per window size
    - Each model uses ALL volatility estimators + ALL exogenous variables as features
    - Uses HAR lags (1, 5, 22) for each estimator
    - PARALLEL TRAINING: Multiple models trained simultaneously for speed
    
    Parameters:
    -----------
    full_x : pd.DataFrame
        FULL volatility estimators (all data, not split)
    full_y : pd.Series
        FULL target variable (log variance - square_est_log)
    full_exo : pd.DataFrame
        FULL exogenous variables
    report : VolatilityReportGenerator
        Report generator instance
    windows : list
        List of rolling window sizes (e.g., [252, 504, 756])
    n_jobs : int
        Number of parallel jobs (-1 = use all cores, 1 = sequential)
    use_gpu : bool
        Whether to use GPU acceleration (if available). 
        Note: Set to False if experiencing GPU conflicts in parallel mode
    train_every_k : int
        Train a new model every K days (default=5 for 5x speedup)
        
    Returns:
    --------
    ml_results : dict
        Dictionary: ml_results[window][model_name] = {predictions, residuals, y_true}
    ml_training_times : dict
        Training times for each window-model combination
    """
    # Detect GPU and CPU configuration
    gpu_available = torch.cuda.is_available() and use_gpu
    n_cpus = multiprocessing.cpu_count()
    actual_jobs = n_cpus if n_jobs == -1 else min(n_jobs, n_cpus)
    
    # Set environment variable to control GPU usage in child processes
    if not use_gpu:
        os.environ['ML_USE_GPU'] = '0'
    else:
        os.environ['ML_USE_GPU'] = '1'
    
    # Warn about GPU + parallel mode
    if n_jobs != 1 and gpu_available:
        print("\n⚠ Note: GPU + parallel training may cause conflicts with some setups.")
        print("   If you see GPU errors, re-run with: use_gpu=False or n_jobs=1")
        print("   Continuing with GPU enabled...\n")
    
    print("="*80)
    print("TRAINING ML MODELS WITH ROLLING WINDOW APPROACH")
    print("="*80)
    print("\nConfiguration:")
    print("  • ONE model per algorithm per window size")
    print("  • Each model uses ALL estimators + exogenous variables")
    print("  • HAR lags (1, 5, 22) computed for each estimator")
    print(f"  • Window sizes: {windows}")
    print(f"  • Parallel jobs: {actual_jobs} (out of {n_cpus} CPU cores)")
    print(f"  • GPU acceleration: {'✓ ENABLED' if gpu_available else '✗ Not available'}")
    if gpu_available:
        print(f"    - XGBoost: device='cuda:0' with hist tree_method")
        print(f"    - LightGBM: device='gpu'")
        print(f"    - CatBoost: task_type='GPU'")
    print("="*80)
    
    # Configuration
    estimators = ['square_est_log', 'parkinson_est_log', 'gk_est_log', 'rs_est_log']
    exo_cols = ['UST10Y', 'HYOAS', 'TermSpread_10Y_2Y', 'VIX', 'Breakeven10Y']
    ml_model_types = ['rf', 'gbm', 'xgboost', 'lightgbm', 'catboost']
    model_params = {'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.05}
    
    # Build combined feature matrix with ALL estimators and their lags
    print("\n" + "="*60)
    print("BUILDING COMBINED FEATURE MATRIX")
    print("="*60)
    print("Including:")
    print(f"  • {len(estimators)} volatility estimators with HAR lags (1, 5, 22)")
    print(f"  • {len(exo_cols)} exogenous variables")
    
    # Compute HAR features for ALL estimators
    all_features = []
    for est in estimators:
        print(f"  Computing HAR lags for {est}...", end=" ")
        df_in = full_x[[est]].copy()
        har = HAR_Model(y_log_col=est, exo_col=[], lags=[1, 5, 22])
        x_est = har.features(df_in)
        # Rename columns to include estimator name
        x_est.columns = [f"{est}_{col}" for col in x_est.columns]
        all_features.append(x_est)
        print(f"✓ {x_est.shape[1]} features")
    
    # Add exogenous variables
    print(f"  Adding exogenous variables...", end=" ")
    all_features.append(full_exo[exo_cols])
    print(f"✓ {len(exo_cols)} features")
    
    # Combine all features
    X_combined = pd.concat(all_features, axis=1)
    X_combined = X_combined.dropna()
    y_combined = full_y.loc[X_combined.index]
    
    print(f"\n✓ Combined feature matrix: {X_combined.shape}")
    print(f"  Total features: {X_combined.shape[1]}")
    print(f"  Total samples: {X_combined.shape[0]}")
    print(f"  Feature breakdown:")
    print(f"    - HAR features: {X_combined.shape[1] - len(exo_cols)} (from {len(estimators)} estimators)")
    print(f"    - Exogenous: {len(exo_cols)}")
    print("="*60)
    
    # Results storage - nested dict: [window][model_name]
    ml_results = {w: {} for w in windows}
    ml_training_times = {}
    
    # Create list of all (model, window) combinations for parallel processing
    training_jobs = [(model_name, w) for w in windows for model_name in ml_model_types]
    total_jobs = len(training_jobs)
    
    print(f"\n{'='*80}")
    print(f"TRAINING {total_jobs} MODELS IN PARALLEL")
    print(f"{'='*80}")
    print(f"Combinations: {len(windows)} windows × {len(ml_model_types)} algorithms")
    print(f"Jobs per batch: {min(actual_jobs, total_jobs)}")
    print(f"{'='*80}\n")
    
    overall_start = time.time()
    
    # Train models in parallel using joblib
    # Note: n_jobs=1 for sequential (easier debugging), n_jobs=-1 for full parallelization
    if n_jobs == 1:
        # Sequential training (for debugging or GPU conflicts)
        print("Running SEQUENTIAL training (n_jobs=1)...\n")
        results_list = []
        for job_idx, (model_name, w) in enumerate(training_jobs, 1):
            print(f"[{job_idx}/{total_jobs}] Training {model_name.upper()} (window={w})...", end=" ")
            result = _train_single_model(model_name, model_params, X_combined, y_combined, w, use_gpu, train_every_k)
            results_list.append(result)
            print(f"✓ ({result[3]:.2f}s) Predictions: {result[4]}")
    else:
        # Parallel training
        print(f"Running PARALLEL training ({actual_jobs} jobs)...\n")
        results_list = Parallel(n_jobs=actual_jobs, verbose=10)(
            delayed(_train_single_model)(model_name, model_params, X_combined, y_combined, w, use_gpu, train_every_k)
            for model_name, w in training_jobs
        )
    
    # Organize results
    for model_name, w, results, model_time, n_predictions in results_list:
        ml_results[w][model_name] = results
        ml_training_times[f"{model_name}_w{w}"] = model_time
    
    overall_time = time.time() - overall_start
    
    print("\n" + "="*80)
    print("✓ ALL ML MODELS TRAINED WITH ROLLING WINDOWS")
    print("="*80)
    print(f"\nTotal models trained: {len(windows)} windows × {len(ml_model_types)} algorithms = {total_jobs} models")
    print(f"Parallel execution: {actual_jobs} jobs")
    print(f"GPU acceleration: {'✓ ENABLED' if gpu_available else '✗ Not available'}")
    print("\nTraining Time Summary:")
    for key, elapsed in sorted(ml_training_times.items(), key=lambda x: x[1]):
        print(f"  {key:20s}: {elapsed:7.2f}s")
    cumulative_time = sum(ml_training_times.values())
    print(f"\n  {'CUMULATIVE':20s}: {cumulative_time:7.2f}s (sum of all models)")
    print(f"  {'WALL-CLOCK':20s}: {overall_time:7.2f}s (actual time elapsed)")
    if overall_time > 0:
        speedup = cumulative_time / overall_time
        print(f"  {'SPEEDUP':20s}: {speedup:.2f}x")
    print(f"  {'AVERAGE':20s}: {cumulative_time/len(ml_training_times):.2f}s per model")
    print("="*80)
    
    return ml_results, ml_training_times, ml_model_types, windows


# %%
def evaluate_ml_models(ml_results_by_window, windows, ml_model_types):
    """
    Evaluate ML model performance across different window sizes.
    
    Since each model now uses ALL features (not per-estimator), we directly
    evaluate the predictions against the target.
    
    Parameters:
    -----------
    ml_results_by_window : dict
        Nested dict: ml_results[window][model_name] = {metrics and predictions}
    windows : list
        List of window sizes
    ml_model_types : list
        List of model names
        
    Returns:
    --------
    ml_evaluation_results : dict
        Nested dict: ml_evaluation_results[window][model_name] = {metrics and predictions}
    """
    print("\n" + "="*80)
    print("EVALUATING ML MODEL PERFORMANCE")
    print("="*80)
    
    ml_evaluation_results = {w: {} for w in windows}
    
    for w_idx, w in enumerate(windows, 1):
        print(f"\n[Window {w_idx}/{len(windows)}] Window size: {w} days")
        
        for model_idx, model_name in enumerate(ml_model_types, 1):
            print(f"  [{model_idx}/{len(ml_model_types)}] {model_name.upper()}...", end=' ')
            
            # Extract predictions (LOG scale)
            y_pred_log = ml_results_by_window[w][model_name]['predictions']
            y_true_log = ml_results_by_window[w][model_name]['y_true'].loc[y_pred_log.index]
            
            # Convert to VARIANCE scale
            y_pred_var = np.exp(y_pred_log)
            y_true_var = np.exp(y_true_log)
            
            # Calculate performance metrics
            qlike_scores = pd.Series(Metric_Evaluation.qlike(y_true_var, y_pred_var), index=y_pred_var.index)
            mspe_scores_raw = pd.Series(Metric_Evaluation.mspe(y_true_var, y_pred_var), index=y_pred_var.index)
            
            # Filter MSPE for numerical stability
            valid_mask = y_true_var > 1e-6
            mspe_scores = mspe_scores_raw[valid_mask]
            
            # Calculate RMSE (note: vol_models.Metrics.rmse has rolling window, we need simple RMSE)
            rmse_score = np.sqrt(np.mean((y_true_var - y_pred_var) ** 2))
            
            # Calculate directional accuracy
            directional_acc = Metric_Evaluation.calculate_directional_accuracy(y_true_var, y_pred_var)
            
            # Store results
            ml_evaluation_results[w][model_name] = {
                'y_true_var': y_true_var,
                'y_pred_var': y_pred_var,
                'y_true_log': y_true_log,
                'y_pred_log': y_pred_log,
                'qlike': qlike_scores,
                'mspe': mspe_scores,
                'rmse': rmse_score,
                'directional_accuracy': directional_acc,
                'qlike_mean': qlike_scores.mean(),
                'qlike_std': qlike_scores.std(),
                'mspe_mean': mspe_scores.mean(),
                'mspe_std': mspe_scores.std()
            }
            
            print(f"✓ QLIKE: {qlike_scores.mean():.4f}, MSPE: {mspe_scores.mean():.4f}, RMSE: {rmse_score:.4f}, Dir.Acc: {directional_acc:.4f} ({directional_acc*100:.2f}%)")
    
    print("\n✓ ML model evaluation completed for all windows")
    return ml_evaluation_results


# %%
def add_ml_results_to_report(ml_evaluation_results, windows, ml_model_types, report, plt):
    """
    Add comprehensive ML model results to report with detailed sections, tables, and plots.
    
    For each ML model, creates:
    1. Model description section
    2. Summary metrics table for all windows
    3. Individual prediction plots for each window
    4. QLIKE and MSPE loss time series plots for each window
    
    Parameters:
    -----------
    ml_evaluation_results : dict
        Nested dict: ml_evaluation_results[window][model_name] = {metrics and predictions}
    windows : list
        List of window sizes
    ml_model_types : list
        List of model names
    report : VolatilityReportGenerator
        Report generator instance
    plt : matplotlib.pyplot
        Matplotlib pyplot module
    """
    print("\n" + "="*80)
    print("ADDING ML MODEL RESULTS TO REPORT")
    print("="*80)
    
    # Model descriptions
    model_descriptions = {
        'rf': '''
**Random Forest (RF)** is an ensemble method that builds multiple decision trees and averages their predictions. 
It's robust to outliers and naturally captures non-linear relationships in volatility forecasting without requiring feature scaling.
        ''',
        'gbm': '''
**Gradient Boosting Machine (GBM)** sequentially builds decision trees, with each tree correcting the errors of previous ones. 
This incremental approach often provides superior performance for forecasting tasks by explicitly learning residual patterns.
        ''',
        'xgboost': '''
**XGBoost** (eXtreme Gradient Boosting) is an optimized gradient boosting implementation with GPU acceleration support. 
It includes regularization to prevent overfitting and has proven highly effective for time series forecasting competitions.
        ''',
        'lightgbm': '''
**LightGBM** (Light Gradient Boosting Machine) is a fast, distributed gradient boosting framework with GPU support. 
It uses leaf-wise tree growth for better accuracy and is especially efficient with large feature sets.
        ''',
        'catboost': '''
**CatBoost** (Categorical Boosting) is designed to handle categorical features effectively with GPU acceleration. 
It uses ordered boosting and symmetric trees to reduce overfitting, making it ideal for heterogeneous financial data.
        '''
    }
    
    # Add main ML section
    report.add_section("Machine Learning Models", level=2)
    report.add_text("""
This section evaluates five state-of-the-art machine learning algorithms for volatility forecasting:
- **Random Forest (RF)**: Ensemble of decision trees
- **Gradient Boosting Machine (GBM)**: Sequential tree learning with error correction
- **XGBoost**: Optimized gradient boosting with GPU support
- **LightGBM**: Fast gradient boosting with leaf-wise growth
- **CatBoost**: Boosting optimized for categorical features

Each model is trained on a rolling window of historical volatility estimators and exogenous variables, 
then evaluated on out-of-sample predictions across three window sizes (252, 504, 756 days).
    """)
    
    # Process each model
    for model_name in ml_model_types:
        print(f"\n[{ml_model_types.index(model_name)+1}/{len(ml_model_types)}] Processing {model_name.upper()}...", end=" ")
        
        # Add model section
        report.add_section(f"{model_name.upper()} Model Performance", level=3)
        report.add_text(model_descriptions.get(model_name, f"The {model_name.upper()} model"))
        
        # Create summary table across windows
        summary_data = []
        for w in windows:
            results = ml_evaluation_results[w][model_name]
            summary_data.append({
                'Window': w,
                'QLIKE': results['qlike_mean'],
                'QLIKE Std': results['qlike_std'],
                'MSPE': results['mspe_mean'],
                'MSPE Std': results['mspe_std'],
                'RMSE': results['rmse'],
                'Dir. Accuracy': f"{results['directional_accuracy']*100:.2f}%"
            })
        
        summary_df = pd.DataFrame(summary_data)
        report.add_table(summary_df, caption=f"Table: {model_name.upper()} Performance Summary Across Windows")
        
        # Add predictions section
        report.add_section(f"{model_name.upper()} Predictions vs True RV", level=4)
        report.add_text("The following plots compare predicted volatility against true realized volatility for each rolling window.")
        
        for w_idx, w in enumerate(windows):
            results = ml_evaluation_results[w][model_name]
            y_pred_var = results['y_pred_var']
            y_true_var = results['y_true_var']
            
            fig = plt.figure(figsize=(16, 7))
            plt.plot(y_pred_var.index, y_pred_var.values, label=f'{model_name.upper()} Prediction', alpha=0.8, linewidth=2)
            plt.plot(y_true_var.index, y_true_var.values, label='True RV', color='black', alpha=0.4, linewidth=2)
            plt.xlabel("Date")
            plt.ylabel("Realized Volatility")
            plt.legend(fontsize=11)
            plt.title(f"{model_name.upper()} Predictions vs True RV (Window={w})")
            plt.tight_layout()
            
            report.save_and_add_plot(fig, f"ml_{model_name}_prediction_w{w}", 
                                    caption=f"{model_name.upper()}: Predictions vs True RV (Window={w})")
            plt.close()
        
        print("✓ Predictions plotted", end=" ")
        
        # Add loss metrics section
        report.add_section(f"{model_name.upper()} Loss Metrics Over Time", level=4)
        report.add_text("""
QLIKE (Quasi-Likelihood) and MSPE (Mean Squared Prediction Error) are computed over time for each window.
These metrics help assess forecast calibration and error magnitude for the ML model across different training periods.
        """)
        
        # QLIKE plots
        for w_idx, w in enumerate(windows):
            results = ml_evaluation_results[w][model_name]
            qlike_scores = results['qlike']
            
            fig = plt.figure(figsize=(16, 7))
            plt.plot(qlike_scores.index, qlike_scores.values, label=f'QLIKE', color='#1f77b4', linewidth=1.5, alpha=0.8)
            plt.axhline(qlike_scores.mean(), color='#1f77b4', linestyle='--', alpha=0.5, label=f'Mean: {qlike_scores.mean():.4f}')
            plt.xlabel("Date")
            plt.ylabel("QLIKE")
            plt.legend(fontsize=11)
            plt.title(f"{model_name.upper()} QLIKE Loss Over Time (Window={w})")
            plt.tight_layout()
            
            report.save_and_add_plot(fig, f"ml_{model_name}_qlike_loss_w{w}", 
                                    caption=f"{model_name.upper()} QLIKE Loss Over Time (Window={w})")
            plt.close()
        
        # MSPE plots
        for w_idx, w in enumerate(windows):
            results = ml_evaluation_results[w][model_name]
            mspe_scores = results['mspe']
            
            fig = plt.figure(figsize=(16, 7))
            plt.plot(mspe_scores.index, mspe_scores.values, label=f'MSPE', color='#ff7f0e', linewidth=1.5, alpha=0.8)
            plt.axhline(mspe_scores.mean(), color='#ff7f0e', linestyle='--', alpha=0.5, label=f'Mean: {mspe_scores.mean():.4f}')
            plt.xlabel("Date")
            plt.ylabel("MSPE")
            plt.legend(fontsize=11)
            plt.title(f"{model_name.upper()} MSPE Loss Over Time (Window={w})")
            plt.tight_layout()
            
            report.save_and_add_plot(fig, f"ml_{model_name}_mspe_loss_w{w}", 
                                    caption=f"{model_name.upper()} MSPE Loss Over Time (Window={w})")
            plt.close()
        
        print("✓ Loss metrics plotted")
    
    print("\n✓ ML model results added to report")
    print("="*80)


# %%
# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Load data
    data = load_data()
    
    # Create report generator
    report = VolatilityReportGenerator(report_name="volatility_forecast_report", append=True)
    
    print("\n" + "="*80)
    print("STARTING ML ANALYSIS")
    print("="*80)
    
    # PHASE 1: PREPARE DATA
    print("\n" + "="*80)
    print("PHASE 1: PREPARING TRAIN/TEST SPLIT")
    print("="*80)
    
    # Combine train and test data to get full dataset
    full_x = pd.concat([data['train_x'], data['test_x']], axis=0)
    full_y = pd.concat([data['train_y'], data['test_y']], axis=0)
    full_exo = pd.concat([data['train_exo'], data['test_exo']], axis=0)
    
    print(f"\nFull dataset size:")
    print(f"  X: {full_x.shape}")
    print(f"  y: {full_y.shape}")
    print(f"  exo: {full_exo.shape}")
    
    # Define split date (matching statistical models)
    split_date = '2023-01-01'
    
    print(f"\n✓ Using split date: {split_date}")
    print(f"  Training period: {full_x.index.min()} to {full_x[full_x.index < split_date].index.max()}")
    print(f"  Test period: {full_x[full_x.index >= split_date].index.min()} to {full_x.index.max()}")
    
    # Split into train and test
    train_x_split = full_x[full_x.index < split_date]
    test_x_split = full_x[full_x.index >= split_date]
    
    train_y_split = full_y[full_y.index < split_date]
    test_y_split = full_y[full_y.index >= split_date]
    
    train_exo_split = full_exo[full_exo.index < split_date]
    test_exo_split = full_exo[full_exo.index >= split_date]
    
    print(f"\n✓ Data split completed:")
    print(f"  Training samples: {len(train_x_split)}")
    print(f"  Test samples: {len(test_x_split)}")
    
    # Define windows (matching stat model)
    rolling_windows = [252, 756]
    max_window = max(rolling_windows)
    
    # For test evaluation, augment test data with trailing training data (for rolling window)
    print(f"\n✓ Augmenting test data with {max_window} days of trailing training data for rolling window...")
    test_x_aug = pd.concat([train_x_split.tail(max_window), test_x_split])
    test_y_aug = pd.concat([train_y_split.tail(max_window), test_y_split])
    test_exo_aug = pd.concat([train_exo_split.tail(max_window), test_exo_split])
    
    print(f"  Augmented test size: {len(test_x_aug)} samples")
    print(f"  First {max_window} samples used for initial rolling window")
    print(f"  Predictions will be generated for {len(test_x_split)} test samples")
    
    # PHASE 2: TRAIN ML MODELS
    print("\n" + "="*80)
    print("PHASE 2: TRAINING ML MODELS ON TEST SET WITH ROLLING WINDOWS")
    print("="*80)
    print("Note: Models are trained on PAST data only (rolling window), predictions on TEST data")
    
    # Train with rolling windows ON TEST SET
    ml_results_by_window, ml_training_times, ml_model_types, windows = train_ml_models_rolling_window(
        full_x=test_x_aug,      # Use augmented test data (includes trailing training for rolling window)
        full_y=test_y_aug,      # Use augmented test targets
        full_exo=test_exo_aug,  # Use augmented test exogenous variables
        report=report,
        windows=rolling_windows,
        n_jobs=-1,          # Use all CPU cores for parallel training
        use_gpu=False,      # Disable GPU to avoid parallel conflicts (faster overall)
        train_every_k=5     # Train every 5 days for 5x speedup
    )
    
    # PHASE 3: FILTER PREDICTIONS TO TEST PERIOD
    print("\n" + "="*80)
    print("PHASE 3: FILTERING PREDICTIONS TO TEST PERIOD ONLY")
    print("="*80)
    
    for w in rolling_windows:
        for model_name in ml_model_types:
            # Get predictions
            predictions = ml_results_by_window[w][model_name]['predictions']
            residuals = ml_results_by_window[w][model_name]['residuals']
            y_true = ml_results_by_window[w][model_name]['y_true']
            
            # Filter to test period only (>= split_date)
            test_mask = predictions.index >= split_date
            
            # Align y_true with predictions index BEFORE filtering to test period
            y_true_aligned = y_true.loc[predictions.index]
            
            ml_results_by_window[w][model_name]['predictions'] = predictions[test_mask]
            ml_results_by_window[w][model_name]['residuals'] = residuals[test_mask]
            ml_results_by_window[w][model_name]['y_true'] = y_true_aligned[test_mask]
            
            print(f"  {model_name.upper()} (w={w}): {test_mask.sum()} test predictions (from {len(predictions)} total)")
    
    print("\n✓ All predictions filtered to test period")
    
    # PHASE 4: EVALUATE ML MODELS
    print("\n" + "="*80)
    print("PHASE 4: EVALUATING ML MODEL PERFORMANCE ON TEST SET")
    print("="*80)
    
    ml_evaluation_results = evaluate_ml_models(
        ml_results_by_window=ml_results_by_window,
        windows=rolling_windows,
        ml_model_types=ml_model_types
    )
    
    # PHASE 5: ADD RESULTS TO REPORT
    print("\n" + "="*80)
    print("PHASE 5: ADDING ML RESULTS TO REPORT")
    print("="*80)
    
    # Print summary for each window
    for w in rolling_windows:
        print(f"\nWindow {w} days:")
        for model_name in ml_model_types:
            res = ml_evaluation_results[w][model_name]
            print(f"  {model_name.upper():12s}: QLIKE={res['qlike_mean']:.4f}, MSPE={res['mspe_mean']:.4f}, RMSE={res['rmse']:.4f}, Predictions={len(res['y_pred_var'])}")
    
    # Add comprehensive ML results to report using our new function
    add_ml_results_to_report(
        ml_evaluation_results=ml_evaluation_results,
        windows=rolling_windows,
        ml_model_types=ml_model_types,
        report=report,
        plt=plt
    )
    
    print("\n✓ ML models analysis complete with rolling windows!")
    
    # =============================================================================
    # Save ML Predictions to CSV
    # =============================================================================
    print("\n" + "="*80)
    print("SAVING ML PREDICTIONS TO CSV")
    print("="*80)

    # Get the true values (they are the same for all models in the test set)
    y_true_var = ml_evaluation_results[rolling_windows[0]][ml_model_types[0]]['y_true_var']
    
    # Start DataFrame with true values
    ml_predictions_df = pd.DataFrame({'RV_true': y_true_var})

    # Add predictions from each model and window
    for w in rolling_windows:
        for model_name in ml_model_types:
            # Get the predictions for the current model and window
            y_pred_var = ml_evaluation_results[w][model_name]['y_pred_var']
            
            # Define a clear column name
            col_name = f"ML_{model_name}_w{w}"
            
            # Join the predictions to the main DataFrame
            ml_predictions_df = ml_predictions_df.join(y_pred_var.rename(col_name), how='outer')

    # Save to CSV
    ml_predictions_df.to_csv('ml_predictions.csv', index_label='Date')
    print("✓ ML predictions saved to ml_predictions.csv")
    print("="*80)


    # PHASE 6: FINALIZE REPORT
    print("\n" + "="*80)
    print("PHASE 6: FINALIZING REPORT")
    print("="*80)
    
    report.finalize_report(final_message="ML Models Analysis Completed Successfully")
    
    print("\n" + "="*80)
    print("✅ ALL PHASES COMPLETE!")
    print("="*80)
    print("\nSummary:")
    print(f"  • Trained {len(ml_model_types)} ML models")
    print(f"  • Evaluated on {len(rolling_windows)} window sizes")
    print(f"  • Total combinations: {len(ml_model_types) * len(rolling_windows)}")
    print(f"  • Test period predictions: {len(test_x_split)} days")
    print(f"  • Report saved to: {report.report_file}")
    print("="*80)


