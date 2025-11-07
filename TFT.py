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
# Custom Weighted Quantile Loss for TFT
# This allows us to penalize underpredictions more heavily for risk management
import torch
from pytorch_forecasting.metrics import QuantileLoss

class WeightedQuantileLoss(QuantileLoss):
    """
    Quantile loss with custom weights for each quantile.
    Inherits from pytorch_forecasting's QuantileLoss to ensure compatibility.
    
    Higher weights = more influence on training.
    
    For volatility forecasting, we prioritize higher quantiles (0.75, 0.9, 0.95)
    to be more conservative and avoid underpredicting risk.
    
    Parameters:
    -----------
    quantiles : list of float
        Quantile levels to predict (e.g., [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
    quantile_weights : list of float
        Weights for each quantile. Higher weights prioritize that quantile.
        Example: [0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0] heavily penalizes underpredictions
    """
    
    def __init__(self, quantiles=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95], 
                 quantile_weights=[0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0]):
        # Initialize parent QuantileLoss
        super().__init__(quantiles=quantiles)
        
        # Register weights as a buffer so they automatically move with the model to GPU/CPU
        self.register_buffer('quantile_weights', torch.tensor(quantile_weights, dtype=torch.float32))
        
    def loss(self, y_pred, y_true):
        """
        Calculate weighted quantile loss.
        
        Parameters:
        -----------
        y_pred : torch.Tensor
            Predicted quantiles, shape [batch_size, n_quantiles] or [..., n_quantiles]
        y_true : torch.Tensor
            Actual values (will be broadcast to match predictions)
        
        Returns:
        --------
        loss : torch.Tensor
            Weighted quantile loss per sample
        """
        # Get the standard quantile loss per quantile
        # Shape: [..., n_quantiles]
        losses = []
        
        # quantile_weights is a buffer, so it's already on the correct device
        weights = self.quantile_weights
        
        # Ensure y_true has the right shape for broadcasting
        if y_true.ndim < y_pred.ndim:
            # Add dimensions to match y_pred shape
            for _ in range(y_pred.ndim - y_true.ndim):
                y_true = y_true.unsqueeze(-1)
        
        for i, (q, w) in enumerate(zip(self.quantiles, weights)):
            pred = y_pred[..., i]
            actual = y_true[..., 0] if y_true.shape[-1] == 1 else y_true
            
            error = actual - pred
            
            # Standard quantile loss components
            loss_over = (1 - q) * torch.clamp(-error, min=0)  # Overprediction penalty
            loss_under = q * torch.clamp(error, min=0)        # Underprediction penalty
            
            # Apply quantile-specific weight
            loss = (loss_over + loss_under) * w
            losses.append(loss)
        
        # Stack and return mean across quantiles
        # Shape: [batch_size] or original shape without quantile dimension
        return torch.stack(losses, dim=-1).mean(dim=-1)

print("✓ WeightedQuantileLoss class defined")

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


def preprocess_predictors_for_tft(data_dict):
    """
    Apply robust preprocessing to handle outliers and scale features.
    This addresses:
    - Extreme outliers in HYOAS (spikes at ~20 from 2008 crisis)
    - Heavy left tail in log-variance estimators (extreme negative values)
    - Multi-modal distributions in exogenous variables
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary containing train_x, train_y, train_exo, estimators, exo_cols
    
    Returns:
    --------
    data_dict : dict
        Updated dictionary with preprocessed features
    scalers : dict
        Dictionary containing fitted scalers for inverse transform if needed
    """
    from sklearn.preprocessing import RobustScaler
    
    print("\n" + "="*80)
    print("PREPROCESSING PREDICTORS FOR TFT")
    print("="*80)
    
    # Store original statistics for reporting
    orig_stats = {}
    
    # 1. Clip extreme outliers in HYOAS (cap at 99th percentile)
    hyoas_99 = data_dict['train_exo']['HYOAS'].quantile(0.99)
    hyoas_max_before = data_dict['train_exo']['HYOAS'].max()
    data_dict['train_exo']['HYOAS'] = data_dict['train_exo']['HYOAS'].clip(upper=hyoas_99)
    print(f"\n1. HYOAS outlier clipping:")
    print(f"   Max before: {hyoas_max_before:.2f}")
    print(f"   Clipped at 99th percentile: {hyoas_99:.2f}")
    print(f"   Max after: {data_dict['train_exo']['HYOAS'].max():.2f}")
    
    # 2. Clip extreme log-variance values (floor at 1st percentile, cap at 99th)
    print(f"\n2. Log-variance estimators clipping:")
    for col in data_dict['estimators']:
        p01 = data_dict['train_x'][col].quantile(0.01)
        p99 = data_dict['train_x'][col].quantile(0.99)
        min_before = data_dict['train_x'][col].min()
        max_before = data_dict['train_x'][col].max()
        data_dict['train_x'][col] = data_dict['train_x'][col].clip(lower=p01, upper=p99)
        print(f"   {col:20s}: [{min_before:7.2f}, {max_before:7.2f}] → [{p01:7.2f}, {p99:7.2f}]")
    
    # Also clip target
    p01_target = data_dict['train_y'].quantile(0.01)
    p99_target = data_dict['train_y'].quantile(0.99)
    min_target_before = data_dict['train_y'].min()
    max_target_before = data_dict['train_y'].max()
    data_dict['train_y'] = data_dict['train_y'].clip(lower=p01_target, upper=p99_target)
    print(f"   {'target':20s}: [{min_target_before:7.2f}, {max_target_before:7.2f}] → [{p01_target:7.2f}, {p99_target:7.2f}]")
    
    # 3. Apply RobustScaler to exogenous variables (resistant to outliers)
    print(f"\n3. Scaling exogenous variables with RobustScaler:")
    scaler_exo = RobustScaler()
    exo_scaled = pd.DataFrame(
        scaler_exo.fit_transform(data_dict['train_exo']),
        index=data_dict['train_exo'].index,
        columns=data_dict['train_exo'].columns
    )
    for col in data_dict['train_exo'].columns:
        print(f"   {col:20s}: mean={exo_scaled[col].mean():7.3f}, std={exo_scaled[col].std():7.3f}")
    data_dict['train_exo'] = exo_scaled
    
    # 4. Apply RobustScaler to volatility estimators
    print(f"\n4. Scaling volatility estimators with RobustScaler:")
    scaler_vol = RobustScaler()
    vol_scaled = pd.DataFrame(
        scaler_vol.fit_transform(data_dict['train_x']),
        index=data_dict['train_x'].index,
        columns=data_dict['train_x'].columns
    )
    for col in data_dict['train_x'].columns:
        print(f"   {col:20s}: mean={vol_scaled[col].mean():7.3f}, std={vol_scaled[col].std():7.3f}")
    data_dict['train_x'] = vol_scaled
    
    # Scale the target
    scaler_target = RobustScaler()
    target_scaled = pd.Series(
        scaler_target.fit_transform(data_dict['train_y'].values.reshape(-1, 1)).flatten(),
        index=data_dict['train_y'].index
    )
    print(f"\n5. Scaling target:")
    print(f"   {'train_y':20s}: mean={target_scaled.mean():7.3f}, std={target_scaled.std():7.3f}")
    data_dict['train_y'] = target_scaled
    
    print("\n" + "="*80)
    print("✓ PREPROCESSING COMPLETE")
    print("="*80)
    print("\nKey improvements:")
    print("  ✓ Outliers clipped to 1st-99th percentile range")
    print("  ✓ All features scaled with RobustScaler (resistant to outliers)")
    print("  ✓ Distributions normalized for stable neural network training")
    print("="*80)
    
    # Return scalers for potential inverse transform
    scalers = {
        'exo': scaler_exo,
        'vol': scaler_vol,
        'target': scaler_target
    }
    
    return data_dict, scalers
def prepare_tft_data(vol_data, exo_data, y_true, max_encoder_length=22, max_prediction_length=1):
    """
    Prepare time series data for TFT model.
    
    Parameters:
    -----------
    vol_data : pd.DataFrame
        Volatility estimators (lagged)
    exo_data : pd.DataFrame
        Exogenous variables
    y_true : pd.Series
        Target variable (log variance)
    max_encoder_length : int
        Length of encoder sequence (lookback window)
    max_prediction_length : int
        Length of prediction horizon
        
    Returns:
    --------
    pd.DataFrame : Prepared data for TFT
    """
    # Combine all features
    df_combined = pd.concat([vol_data, exo_data, y_true.rename('target')], axis=1)
    df_combined = df_combined.dropna()
    
    # Add time index
    df_combined = df_combined.reset_index()
    df_combined['time_idx'] = range(len(df_combined))
    df_combined['group'] = 'TLT'  # Single time series group
    
    # Ensure all columns are numeric
    for col in df_combined.columns:
        if col not in ['Date', 'group']:
            df_combined[col] = pd.to_numeric(df_combined[col], errors='coerce')
    
    df_combined = df_combined.dropna()
    
    return df_combined
def add_tft_results_to_report(tft_qlike, tft_mspe, tft_pred_var, tft_actual_var,
                                tft_pred_series, tft_actual_series,
                                n_train, n_val, report, plt, tft_pred_q10=None, tft_pred_q90=None):
    """
    Add comprehensive TFT results to report with detailed visualizations.
    """
    report.add_section("Temporal Fusion Transformer (TFT) Results", level=2)
    report.add_text("""
The Temporal Fusion Transformer is a state-of-the-art deep learning architecture
for multi-horizon time series forecasting. It combines:

- **Multi-head attention mechanism**: Captures complex temporal dependencies
- **Variable selection networks**: Automatic feature importance learning
- **Gated residual networks**: Non-linear processing with skip connections
- **Quantile forecasting**: Provides prediction intervals (10th, 50th, 90th percentiles)

**Key Improvements Implemented:**
- **Multiple quantiles**: Generates prediction intervals for uncertainty quantification
- **Increased model capacity**: Larger hidden sizes and attention heads for better learning
- **Enhanced regularization**: Higher dropout to prevent overfitting
- **Extended lookback**: 90-day encoder length for capturing longer-term patterns

**TFT Architecture Details:**
- Hidden size: 64 (optimized for dataset size)
- Attention heads: 4
- Encoder length: 90 days (quarterly lookback)
- Dropout: 0.1 (enhanced regularization)
- Quantiles: 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95 (7 quantiles)
- Early stopping: Patience of 20 epochs
""")
    
    # Calculate additional metrics for TFT
    tft_rmse = np.sqrt(np.mean((tft_actual_var - tft_pred_var) ** 2))
    tft_accuracy = Metric_Evaluation.calculate_directional_accuracy(tft_actual_var, tft_pred_var)
    
    # Filter out infinite and NaN values from tft_mspe for plotting
    tft_mspe_clean = tft_mspe.replace([np.inf, -np.inf], np.nan).dropna()
    if len(tft_mspe_clean) == 0:
        # If all values are invalid, create clean mspe by recalculating with safeguards
        tft_mspe_safe = np.where(tft_actual_var > 0, ((tft_actual_var - tft_pred_var) / tft_actual_var) ** 2, np.nan)
        tft_mspe_clean = pd.Series(tft_mspe_safe, index=tft_actual_var.index).dropna()
    
    tft_metrics = {
        "QLIKE Mean": f"{tft_qlike.mean():.6f}",
        "QLIKE Std": f"{tft_qlike.std():.6f}",
        "QLIKE Min": f"{tft_qlike.min():.6f}",
        "QLIKE Max": f"{tft_qlike.max():.6f}",
        "MSPE Mean": f"{tft_mspe_clean.mean():.6f}",
        "MSPE Std": f"{tft_mspe_clean.std():.6f}",
        "MSPE Min": f"{tft_mspe_clean.min():.6f}",
        "MSPE Max": f"{tft_mspe_clean.max():.6f}",
        "RMSE": f"{tft_rmse:.6f}",
        "Directional Accuracy": f"{tft_accuracy:.4f} ({tft_accuracy*100:.2f}%)",
        "Training Samples": str(n_train),
        "Validation Samples": str(n_val)
    }
    report.add_metrics_summary(tft_metrics, title="TFT Model Performance (Validation Set)")
    
    # Add TFT main visualization plots (4 panels)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('TFT Model: Main Performance Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Actual vs Predicted with Prediction Intervals (Log Variance)
    axes[0, 0].plot(tft_actual_series.index, tft_actual_series, 
                    label='Actual Log Variance', color='black', linewidth=1.5, alpha=0.7)
    axes[0, 0].plot(tft_pred_series.index, tft_pred_series, 
                    label='TFT Predictions (Median)', color='red', linewidth=1.5, alpha=0.7)
    # Add prediction intervals if available
    if tft_pred_q10 is not None and tft_pred_q90 is not None:
        axes[0, 0].fill_between(tft_pred_series.index, 
                               tft_pred_q10[:len(tft_pred_series)], 
                               tft_pred_q90[:len(tft_pred_series)], 
                               alpha=0.3, color='red', label='80% Prediction Interval')
    axes[0, 0].set_title('Actual vs Predicted with Intervals (Log Variance)', fontweight='bold')
    axes[0, 0].set_xlabel('Sample Index')
    axes[0, 0].set_ylabel('Log Variance')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Actual vs Predicted (Variance Scale)
    axes[0, 1].plot(tft_actual_var.index, tft_actual_var, 
                    label='Actual Variance', color='black', linewidth=1.5, alpha=0.7)
    axes[0, 1].plot(tft_pred_var.index, tft_pred_var, 
                    label='TFT Predictions', color='red', linewidth=1.5, alpha=0.7)
    axes[0, 1].set_title('Actual vs Predicted Variance', fontweight='bold')
    axes[0, 1].set_xlabel('Sample Index')
    axes[0, 1].set_ylabel('Variance')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Prediction Errors
    tft_errors = tft_pred_var - tft_actual_var
    axes[1, 0].plot(tft_errors.index, tft_errors, color='purple', linewidth=1, alpha=0.7)
    axes[1, 0].axhline(y=0, color='black', linestyle='--', linewidth=1)
    axes[1, 0].set_title('Prediction Errors', fontweight='bold')
    axes[1, 0].set_xlabel('Sample Index')
    axes[1, 0].set_ylabel('Error (Predicted - Actual)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Scatter plot (Actual vs Predicted)
    axes[1, 1].scatter(tft_actual_var, tft_pred_var, alpha=0.5, s=20, color='steelblue')
    min_val = min(tft_actual_var.min(), tft_pred_var.min())
    max_val = max(tft_actual_var.max(), tft_pred_var.max())
    axes[1, 1].plot([min_val, max_val], [min_val, max_val], 
                    'r--', linewidth=2, label='Perfect Prediction')
    axes[1, 1].set_title(f'Actual vs Predicted Scatter (RMSE: {tft_rmse:.4f})', fontweight='bold')
    axes[1, 1].set_xlabel('Actual Variance')
    axes[1, 1].set_ylabel('Predicted Variance')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    report.save_and_add_plot(fig, "tft_predictions_analysis", 
                            caption="Figure: TFT Main Performance Analysis (Validation Set)")
    plt.close()
    
    # TFT QLIKE Distribution Analysis
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('TFT Model: QLIKE Distribution Analysis', fontsize=14, fontweight='bold')
    
    axes[0].hist(tft_qlike, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].axvline(tft_qlike.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {tft_qlike.mean():.4f}')
    axes[0].axvline(tft_qlike.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {tft_qlike.median():.4f}')
    axes[0].set_title('QLIKE Distribution (Histogram)', fontweight='bold')
    axes[0].set_xlabel('QLIKE Value')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    axes[1].boxplot(tft_qlike, vert=True, patch_artist=True)
    axes[1].set_title('QLIKE Distribution (Box Plot)', fontweight='bold')
    axes[1].set_ylabel('QLIKE Value')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    report.save_and_add_plot(fig, "tft_qlike_distribution", 
                            caption="Figure: TFT QLIKE Distribution Analysis")
    plt.close()
    
    # TFT MSPE Distribution Analysis
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('TFT Model: MSPE Distribution Analysis', fontsize=14, fontweight='bold')
    
    axes[0].hist(tft_mspe_clean, bins=50, color='coral', alpha=0.7, edgecolor='black')
    axes[0].axvline(tft_mspe_clean.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {tft_mspe_clean.mean():.4f}')
    axes[0].axvline(tft_mspe_clean.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {tft_mspe_clean.median():.4f}')
    axes[0].set_title('MSPE Distribution (Histogram)', fontweight='bold')
    axes[0].set_xlabel('MSPE Value')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    axes[1].boxplot(tft_mspe_clean, vert=True, patch_artist=True)
    axes[1].set_title('MSPE Distribution (Box Plot)', fontweight='bold')
    axes[1].set_ylabel('MSPE Value')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    report.save_and_add_plot(fig, "tft_mspe_distribution", 
                            caption="Figure: TFT MSPE Distribution Analysis")
    plt.close()
    
    # TFT RMSE Analysis
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('TFT Model: RMSE Analysis', fontsize=14, fontweight='bold')
    
    # RMSE over time (rolling window)
    rmse_rolling = pd.Series(tft_errors ** 2, index=tft_errors.index).rolling(window=20).mean().apply(np.sqrt)
    axes[0].plot(rmse_rolling.index, rmse_rolling, color='darkblue', linewidth=1.5, label='Rolling RMSE (20-sample window)')
    axes[0].axhline(tft_rmse, color='red', linestyle='--', linewidth=2, label=f'Overall RMSE: {tft_rmse:.4f}')
    axes[0].set_title('RMSE Over Time', fontweight='bold')
    axes[0].set_xlabel('Sample Index')
    axes[0].set_ylabel('RMSE')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Residuals histogram
    axes[1].hist(tft_errors, bins=50, color='mediumpurple', alpha=0.7, edgecolor='black')
    axes[1].axvline(tft_errors.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {tft_errors.mean():.4f}')
    axes[1].axvline(0, color='black', linestyle='-', linewidth=1)
    axes[1].set_title('Prediction Error Distribution', fontweight='bold')
    axes[1].set_xlabel('Error Value')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    report.save_and_add_plot(fig, "tft_rmse_analysis", 
                            caption="Figure: TFT RMSE Analysis")
    plt.close()
    
    print("\n✓ TFT comprehensive results and visualizations added to report")

print("✓ Preprocessing functions ready")
# %%
data = load_data()
exo_cols = ['UST10Y', 'HYOAS', 'TermSpread_10Y_2Y', 'VIX', 'Breakeven10Y']
estimators = ['square_est_log', 'parkinson_est_log', 'gk_est_log', 'rs_est_log']

# %%

# Create report generator
report = VolatilityReportGenerator(report_name="volatility_forecast_report", append=True)


# Prepare data for TFT
print("Preparing TFT dataset...")
tft_train_data = prepare_tft_data(
    vol_data=data['train_x'][estimators],
    exo_data=data['train_exo'][exo_cols],
    y_true=data['train_y'],
    max_encoder_length=22,
    max_prediction_length=1
)
# %%
print(f"✓ TFT training data prepared: {tft_train_data.shape}")
print(f"  Columns: {list(tft_train_data.columns)}")
print(f"  Date range: {tft_train_data['Date'].min()} to {tft_train_data['Date'].max()}")
# %%
# Define validation split (use 90% of training data for model training, 10% for early stopping)
# This matches the approach used for GBM/XGBoost models
training_cutoff = int(tft_train_data['time_idx'].max() * 0.9)
# Time-varying features (change over time)
time_varying_known_reals = exo_cols  # Exogenous variables
time_varying_unknown_reals = estimators + ['target']

print(f"Training cutoff: {training_cutoff} (90% of pre-2023 data for training)")
print(f"Time-varying known reals: {time_varying_known_reals}")
print(f"Time-varying unknown reals: {time_varying_unknown_reals}")

# Create TimeSeriesDataSet
training_tft = TimeSeriesDataSet(
    tft_train_data[tft_train_data['time_idx'] <= training_cutoff],
    time_idx='time_idx',
    target='target',
    group_ids=['group'],
    min_encoder_length=30,  # Reduced to keep more training samples
    max_encoder_length=90,  # Keep longer lookback for volatility patterns
    min_prediction_length=1,
    max_prediction_length=1,
    time_varying_known_reals=time_varying_known_reals,
    time_varying_unknown_reals=time_varying_unknown_reals,
    target_normalizer=GroupNormalizer(groups=["group"]),  # Re-enabled for better training
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

# Create validation dataset (10% of pre-2023 data for early stopping during training)
validation_tft = TimeSeriesDataSet.from_dataset(
    training_tft,
    tft_train_data[tft_train_data['time_idx'] > training_cutoff],
    predict=False,
    stop_randomization=True
)

print("\n✓ Preparing TRUE TEST SET (2023+) - matching GBM/XGBoost split...")
# CRITICAL FIX: Augment test data with trailing training data for encoder context
# TFT needs 90 days of history to make predictions, so we prepend the last 90 days of training data
max_encoder = 90

# Get trailing training data (last 90 days before test period)
train_x_tail = data['train_x'][estimators].tail(max_encoder)
train_exo_tail = data['train_exo'][exo_cols].tail(max_encoder)
train_y_tail = data['train_y'].tail(max_encoder)

# Concatenate training tail + test data
test_x_augmented = pd.concat([train_x_tail, data['test_x'][estimators]])
test_exo_augmented = pd.concat([train_exo_tail, data['test_exo'][exo_cols]])
test_y_augmented = pd.concat([train_y_tail, data['test_y']])

print(f"  Augmented test data with {max_encoder} days of trailing training data")
print(f"  Original test size: {len(data['test_x'])}")
print(f"  Augmented test size: {len(test_x_augmented)} (includes {max_encoder}-day encoder context)")

# This uses the SAME test set as all ML models for fair comparison
tft_test_data = prepare_tft_data(
    vol_data=test_x_augmented,      # Augmented with encoder context
    exo_data=test_exo_augmented,    # Augmented with encoder context
    y_true=test_y_augmented,        # Augmented with encoder context
    max_encoder_length=90,
    max_prediction_length=1
)

# Create test dataset - CRITICAL: Use different approach for predictions
# When predict=True, TimeSeriesDataSet.from_dataset treats each row as a separate prediction
# This causes very few samples when max_encoder_length is large relative to data size

# Instead, create test dataset directly with looser constraints for predictions
print(f"\n✓ Creating test dataset (critical fix for prediction generation)...")

# Use MUCH lower encoder constraints for test prediction phase
# The model was trained with max_encoder=90, but for predictions we can be more flexible
test_tft = TimeSeriesDataSet(
    tft_test_data,
    time_idx='time_idx',
    target='target',
    group_ids=['group'],
    min_encoder_length=1,  # REDUCED: Minimum encoder length for predictions
    max_encoder_length=90,  # Keep max to match training, but flexibility at prediction time
    min_prediction_length=1,
    max_prediction_length=1,
    time_varying_known_reals=exo_cols,
    time_varying_unknown_reals=estimators + ['target'],
    target_normalizer=training_tft.target_normalizer,  # Use training normalizer
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

print(f"✓ Direct test dataset creation: {len(test_tft)} samples")

# Create dataloaders
batch_size = 64  # Increased for more stable gradients with preprocessed features
train_dataloader = training_tft.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader = validation_tft.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
test_dataloader = test_tft.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

print(f"✓ TFT datasets created (matching GBM/XGBoost split):")
print(f"  Training samples: {len(training_tft)} (90% of pre-2023 for model training)")
print(f"  Validation samples: {len(validation_tft)} (10% of pre-2023 for early stopping)")
print(f"  TEST samples: {len(test_tft)} (2023+ TRUE test set - SAME as ML models)")
print(f"  Batch size: {batch_size}")
print(f"\n✓ TFT will be evaluated on SAME test period as GBM/XGBoost/LightGBM/CatBoost!")
# %%
# Configure TFT model
print("\n" + "="*60)
print("TRAINING TEMPORAL FUSION TRANSFORMER")
print("="*60)

tft = TemporalFusionTransformer.from_dataset(
    training_tft,
    learning_rate=0.001,  # Slightly higher LR for faster convergence with larger model
    hidden_size=128,      # DOUBLED: More capacity for complex patterns
    attention_head_size=4,  # More attention heads for richer representations
    dropout=0.2,         # Increased dropout to prevent overfitting with larger model
    hidden_continuous_size=64,  # DOUBLED: More processing capacity for continuous features
    lstm_layers=2,       # Add second LSTM layer for deeper temporal modeling
    output_size=7,       # Multiple quantiles (keeping as requested)
    loss=WeightedQuantileLoss(quantiles=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95], 
                              quantile_weights=[0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0]),  # RE-ENABLED: Weighted loss for conservative predictions
    log_interval=5,
    reduce_on_plateau_patience=4,
)

print(f"✓ TFT model configured - ENHANCED CAPACITY FOR RVOL PREDICTION")
print(f"  Hidden size: 128 (2x increase for complex volatility patterns)")
print(f"  Hidden continuous: 64 (2x increase for better feature processing)")
print(f"  LSTM layers: 2 (deeper temporal modeling)")
print(f"  Attention heads: 4 (multi-head attention)")
print(f"  Dropout: 0.2 (regularization for larger model)")
print(f"  Learning rate: 0.001")
print(f"  Loss: WeightedQuantileLoss (penalizes underpredictions for conservative forecasts)")
print(f"  Loss quantiles: {tft.loss.quantiles}")
print(f"  Quantile weights: {tft.loss.quantile_weights.tolist()} (higher = more conservative)")
print(f"  Estimated parameters: ~150K+ (vs ~40K in baseline model)")

# Custom PyTorch Lightning Callback for logging RMSE and Accuracy
try:
    from lightning.pytorch.callbacks import Callback
except ImportError:
    from pytorch_lightning.callbacks import Callback

class TFTMetricsCallback(Callback):
    """PyTorch Lightning callback to log RMSE and accuracy during training."""
    
    def __init__(self, val_dataloader, model, log_every_n_epochs=1):
        super().__init__()
        self.val_dataloader = val_dataloader
        self.model = model
        self.log_every_n_epochs = log_every_n_epochs
        
    def on_validation_epoch_end(self, trainer, pl_module):
        """Called at the end of each validation epoch."""
        if (trainer.current_epoch + 1) % self.log_every_n_epochs == 0:
            try:
                # Put model in eval mode
                pl_module.eval()
                
                # Collect predictions and actuals manually batch by batch
                all_predictions = []
                all_actuals = []
                
                with torch.no_grad():
                    for batch_idx, batch in enumerate(self.val_dataloader):
                        # Get batch data - handle device placement
                        x, y = batch
                        
                        # Move to same device as model if needed
                        if hasattr(pl_module, 'device'):
                            device = pl_module.device
                            # Move x dict tensors to device
                            x = {k: v.to(device) if torch.is_tensor(v) else v for k, v in x.items()}
                        
                        # Forward pass to get predictions
                        output = pl_module(x)
                        
                        # Extract predictions (median quantile)
                        if hasattr(output, 'prediction'):
                            preds = output.prediction
                        else:
                            preds = output
                        
                        # Handle quantile outputs
                        if preds.ndim == 3:  # [batch, time, quantiles]
                            preds = preds[:, 0, 3]  # Get first timestep, median quantile
                        elif preds.ndim == 2 and preds.shape[1] == 7:
                            preds = preds[:, 3]  # Get median quantile
                        elif preds.ndim == 2 and preds.shape[1] == 1:
                            preds = preds[:, 0]
                        else:
                            preds = preds[:, 0] if preds.ndim > 1 else preds
                        
                        # Get actual values
                        if isinstance(y, tuple):
                            actuals = y[0][:, 0]  # Get first timestep
                        else:
                            actuals = y[:, 0] if y.ndim > 1 else y
                        
                        # Move to CPU and store
                        all_predictions.extend(preds.detach().cpu().numpy())
                        all_actuals.extend(actuals.detach().cpu().numpy())
                
                # Convert to numpy arrays
                pred_values = np.array(all_predictions)
                actual_values = np.array(all_actuals)
                
                # Convert to variance scale
                pred_var = np.exp(pred_values)
                actual_var = np.exp(actual_values)
                
                # Filter valid values
                valid_mask = (actual_var > 1e-8) & (pred_var > 1e-8) & np.isfinite(actual_var) & np.isfinite(pred_var)
                pred_var_valid = pred_var[valid_mask]
                actual_var_valid = actual_var[valid_mask]
                
                if len(pred_var_valid) > 0:
                    # Calculate metrics (simple RMSE, not rolling window)
                    rmse_val = np.sqrt(np.mean((actual_var_valid - pred_var_valid) ** 2))
                    accuracy_val = Metric_Evaluation.calculate_directional_accuracy(actual_var_valid, pred_var_valid)
                    
                    # Log metrics
                    epoch_num = trainer.current_epoch + 1
                    print(f"\n  [Epoch {epoch_num:3d}] RMSE={rmse_val:.4f} | Directional Accuracy={accuracy_val:.4f} ({accuracy_val*100:.2f}%)")
                
                # Return model to training mode
                pl_module.train()
                
            except Exception as e:
                # Silent fail to avoid interrupting training
                pl_module.train()  # Make sure model is back in training mode
                pass

# %%
# Function to save TFT predictions at each epoch
def save_tft_epoch_predictions(tft_model, test_dataloader, tft_test_data, epoch_num):
    """
    Generate TFT predictions on test set and save to CSV with epoch number.
    
    Parameters:
    -----------
    tft_model : TemporalFusionTransformer
        Trained TFT model
    test_dataloader : DataLoader
        Test data loader
    tft_test_data : pd.DataFrame
        Raw test data with dates
    epoch_num : int
        Current epoch number (used for filename)
    
    Returns:
    --------
    None (saves CSV file)
    """
    import torch
    from torch.utils.data import DataLoader
    
    print(f"\n[EPOCH {epoch_num}] Saving predictions...")
    
    try:
        # Set model to eval mode and ensure it stays on the correct device
        tft_model.eval()
        device = next(tft_model.parameters()).device
        
        # Get predictions using TFT's predict method
        with torch.no_grad():
            raw_predictions = tft_model.predict(test_dataloader, mode="raw", return_x=True)
        
        # Extract prediction tensor - handle different output structures
        if hasattr(raw_predictions, 'output'):
            # Newer pytorch-forecasting structure
            if isinstance(raw_predictions.output, dict):
                prediction_tensor = raw_predictions.output['prediction']
            else:
                prediction_tensor = raw_predictions.output
        else:
            # Older structure - raw_predictions IS the output
            if isinstance(raw_predictions, dict):
                prediction_tensor = raw_predictions['prediction']
            else:
                prediction_tensor = raw_predictions
        
        # Convert to regular tensor if it's a special pytorch-forecasting object
        if hasattr(prediction_tensor, 'detach'):
            prediction_tensor = prediction_tensor.detach()
        elif not isinstance(prediction_tensor, torch.Tensor):
            # It might be a tuple or list - try to extract the actual tensor
            if isinstance(prediction_tensor, (tuple, list)):
                prediction_tensor = prediction_tensor[0]
            # Convert to tensor if needed
            if not isinstance(prediction_tensor, torch.Tensor):
                prediction_tensor = torch.tensor(prediction_tensor)
        
        # Extract median predictions (index 3 for [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
        # Use indexing compatible with both regular tensors and special objects
        if prediction_tensor.ndim == 3:
            # Shape: [batch, time, quantiles]
            tft_pred_log_values = prediction_tensor[:, 0, 3].cpu().numpy()
        elif prediction_tensor.ndim == 2:
            # Shape: [batch, quantiles]
            tft_pred_log_values = prediction_tensor[:, 3].cpu().numpy()
        else:
            raise ValueError(f"Unexpected prediction tensor shape: {prediction_tensor.shape}")
        
        # Get actual values from test dataset
        tft_actual_log_values = []
        for i in range(len(test_dataloader.dataset)):
            x, y = test_dataloader.dataset[i]
            if torch.is_tensor(y):
                actual_val = y[0].item() if y.dim() > 0 else y.item()
            else:
                actual_val = float(y[0]) if isinstance(y, (list, tuple, np.ndarray)) else float(y)
            tft_actual_log_values.append(actual_val)
        
        tft_actual_log_values = np.array(tft_actual_log_values)
        
        # Align lengths
        min_length = min(len(tft_pred_log_values), len(tft_actual_log_values))
        tft_pred_log_values = tft_pred_log_values[:min_length]
        tft_actual_log_values = tft_actual_log_values[:min_length]
        
        # Filter to TRUE TEST PERIOD (2023+)
        split_date = pd.Timestamp('2023-01-01')
        all_test_dates = tft_test_data['Date'].values
        test_period_mask = pd.to_datetime(all_test_dates) >= split_date
        test_dates_2023plus = all_test_dates[test_period_mask]
        
        # Take last N predictions matching test period
        if len(tft_pred_log_values) >= len(test_dates_2023plus):
            tft_pred_values_test = tft_pred_log_values[-len(test_dates_2023plus):]
            tft_actual_values_test = tft_actual_log_values[-len(test_dates_2023plus):]
            test_dates_filtered = test_dates_2023plus
        else:
            tft_pred_values_test = tft_pred_log_values
            tft_actual_values_test = tft_actual_log_values
            test_dates_filtered = test_dates_2023plus[-len(tft_pred_log_values):]
        
        # Create series with date indices
        tft_pred_series = pd.Series(
            tft_pred_values_test,
            index=pd.to_datetime(test_dates_filtered),
            name='TFT_predictions_log'
        )
        
        tft_actual_series = pd.Series(
            tft_actual_values_test,
            index=pd.to_datetime(test_dates_filtered),
            name='Actual_log'
        )
        
        # Convert to variance scale
        tft_pred_var = np.exp(tft_pred_series)
        tft_actual_var = np.exp(tft_actual_series)
        
        # Create DataFrame and save
        tft_predictions_df = pd.DataFrame({
            'RV_true': tft_actual_var.values,
            'TFT_prediction': tft_pred_var.values
        }, index=tft_pred_var.index)
        
        # Save with epoch number in filename
        filename = f'tft_predictions_{epoch_num}.csv'
        tft_predictions_df.to_csv(filename, index_label='Date')
        
        print(f"  ✓ Saved {filename} ({len(tft_predictions_df)} predictions)")
        
    except Exception as e:
        print(f"  ✗ Error saving predictions at epoch {epoch_num}: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # ALWAYS set model back to training mode on correct device
        tft_model.train()
        # Ensure model is on the correct device (GPU if available)
        if torch.cuda.is_available():
            tft_model = tft_model.cuda()


# Custom callback to save predictions every 3 epochs
class EpochPredictionCallback(Callback):
    """Save TFT predictions every 3 epochs during training."""
    
    def __init__(self, tft_model, test_dataloader, tft_test_data, save_every_n_epochs=3):
        super().__init__()
        self.tft_model = tft_model
        self.test_dataloader = test_dataloader
        self.tft_test_data = tft_test_data
        self.save_every_n_epochs = save_every_n_epochs
        
    def on_train_epoch_end(self, trainer, pl_module):
        """Called when a training epoch ends."""
        current_epoch = trainer.current_epoch + 1  # Current epoch (1-indexed)
        
        # Save predictions every N epochs
        if current_epoch % self.save_every_n_epochs == 0:
            save_tft_epoch_predictions(
                self.tft_model,
                self.test_dataloader,
                self.tft_test_data,
                current_epoch
            )

# Import Callback base class for custom callback
try:
    from lightning.pytorch.callbacks import Callback
except ImportError:
    from pytorch_lightning.callbacks import Callback

# Configure trainer
early_stop_callback = EarlyStopping(
    monitor='val_loss',
    min_delta=1e-4,
    patience=12,
    verbose=False,
    mode='min'
)

# Create metrics callback
metrics_callback = TFTMetricsCallback(val_dataloader, None, log_every_n_epochs=1)  # Will set model after instantiation

# Initialize callbacks list
trainer_callbacks = [early_stop_callback, metrics_callback]

# %%
trainer = Trainer(
    max_epochs=500,
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',  # Use CUDA if available, else CPU
    devices=1,  # Use single GPU
    enable_model_summary=True,
    gradient_clip_val=0.1,
    callbacks=trainer_callbacks,
    enable_progress_bar=True,
    enable_checkpointing=False,
    logger=False,
)

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Combine all predictors (volatility estimators + exogenous variables)
predictors = pd.concat([data['train_x'], data['train_exo']], axis=1)

# Create a 3x3 grid for 9 predictors
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 10))
axes = axes.flatten()  # Flatten for easy indexing

for i, col in enumerate(predictors.columns):
    sns.histplot(predictors[col], ax=axes[i], kde=True, bins=50)
    axes[i].set_title(f'Distribution of {col}', fontsize=12)
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
# plt.show()
print("✓ Trainer configured")
print(f"  Device: {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}")
print("  Max epochs: 500")
print("  Early stopping patience: 20")
print("  Live metrics logging: RMSE and Directional Accuracy (every epoch)")
# %%
# Train model with live metric logging
print("\nTraining TFT model...")

# Initialize epoch prediction callback NOW that TFT model is created
epoch_prediction_callback = EpochPredictionCallback(
    tft_model=tft,
    test_dataloader=test_dataloader,
    tft_test_data=tft_test_data,
    save_every_n_epochs=3
)
trainer.callbacks.append(epoch_prediction_callback)

print("\n✓ Epoch prediction callback initialized (saves every 3 epochs)")

# Override training_step to ensure proper loss return format for PyTorch Lightning
def custom_training_step(self, batch, batch_idx):
    """Custom training_step that ensures proper loss format for PyTorch Lightning."""
    x, y = batch
    output = self(x)
    
    # Extract prediction tensor from Output object
    if hasattr(output, 'prediction'):
        y_hat = output.prediction
    else:
        y_hat = output
    
    # Handle different y formats from TimeSeriesDataSet
    if isinstance(y, tuple):
        # y is (target, target_scale) tuple
        target, target_scale = y
        if target_scale is not None:
            y_hat = self.transform_output(y_hat, target_scale=target_scale)
        loss = self.loss(y_hat, target)
    else:
        # y is a tensor - use original logic
        try:
            y_hat = self.transform_output(y_hat, target_scale=y[..., 1:])
            loss = self.loss(y_hat, y[..., 0])
        except (IndexError, TypeError):
            # Fallback: assume y is just the target
            loss = self.loss(y_hat, y)
    
    # Log loss
    self.log("train_loss", loss, prog_bar=True)
    
    # Return loss in format expected by PyTorch Lightning
    return {"loss": loss}

# Monkey patch the training_step method
tft.training_step = custom_training_step.__get__(tft, tft.__class__)

print("✓ Custom training_step method applied to ensure proper loss format")

# Train with PyTorch Lightning
print("\nStarting TFT training with live progress tracking...")
print("Progress bar will show validation loss each epoch")
print("Metrics callback will log RMSE and Directional Accuracy")
print("Prediction callback will save predictions every 3 epochs\n")

trainer.fit(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)

# After training, calculate residual variance from the validation set
print("\nCalculating residual variance from validation set for log-normal correction...")
# Use the model's predict method with mode="raw" to get detailed outputs
val_raw_predictions = tft.predict(val_dataloader, mode="raw", return_x=True)

# Extract predictions and actuals from the raw output
val_pred_tensor = val_raw_predictions.output['prediction']
# Shape: [n_samples, n_timesteps, n_quantiles]
# We want timestep=0, quantile index 3 (median)
val_preds_log = val_pred_tensor[:, 0, 3].cpu().numpy()

# Extract actual values from validation dataset
val_actuals_log = []
for i in range(len(validation_tft)):
    x, y = validation_tft[i]
    if torch.is_tensor(y):
        actual_val = y[0].item() if y.dim() > 0 else y.item()
    else:
        actual_val = float(y[0]) if isinstance(y, (list, tuple, np.ndarray)) else float(y)
    val_actuals_log.append(actual_val)

val_actuals_log = np.array(val_actuals_log)

# Align lengths
min_val_length = min(len(val_preds_log), len(val_actuals_log))
val_preds_log = val_preds_log[:min_val_length]
val_actuals_log = val_actuals_log[:min_val_length]

# Calculate residuals and variance
log_residuals = val_preds_log - val_actuals_log
residual_variance = np.var(log_residuals)
print(f"✓ Calculated residual variance for correction: {residual_variance:.4f}")

print("\n✓ TFT model training completed")


# %%
# Generate predictions on TRUE TEST SET (2023+) - matching GBM/XGBoost evaluation
print("\n" + "="*80)
print("GENERATING TFT PREDICTIONS ON TRUE TEST SET (2023+)")
print("="*80)
print("Using SAME test period as GBM/XGBoost/LightGBM/CatBoost for fair comparison")

# Use the PROPER pytorch-forecasting API for batch prediction
print("\nUsing TFT.predict() method for efficient batch inference...")

# Create dataloader for prediction (use all samples, no shuffling)
test_dataloader = test_tft.to_dataloader(train=False, batch_size=128, num_workers=0, shuffle=False)

# Generate predictions using the model's predict method
raw_predictions = tft.predict(test_dataloader, mode="raw", return_x=True)

print(f"✓ Raw predictions generated")

# The output is a dictionary-like object with 'prediction' key containing the tensor
prediction_tensor = raw_predictions.output['prediction']
print(f"  Prediction tensor shape: {prediction_tensor.shape}")
print(f"  Output has {len(prediction_tensor)} predictions")

# Extract predictions (in log scale since model was trained on log-transformed targets)
# prediction_tensor shape: [n_samples, n_timesteps, n_quantiles]
# We want timestep=0 (one-step ahead)
# 
# QUANTILE INDEX MAPPING (with weighted loss):
# [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
# Weights: [0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0]
# 
# Index 3 = 0.5 quantile (median) - balanced, weight=1.5
# Index 4 = 0.75 quantile - conservative, weight=2.0
# Index 5 = 0.9 quantile - very conservative, weight=3.0
# Index 6 = 0.95 quantile - extremely conservative, weight=4.0
#
# For risk management, consider using index 4 (0.75) or 5 (0.9) instead of median
tft_pred_log_values = prediction_tensor[:, 0, 3].cpu().numpy()  # Median predictions (log scale)
tft_pred_q10_log = prediction_tensor[:, 0, 1].cpu().numpy()  # 0.1 quantile
tft_pred_q90_log = prediction_tensor[:, 0, 5].cpu().numpy()  # 0.9 quantile

# Get actual values from the test dataset
print(f"\nExtracting actual values from test dataset...")
tft_actual_log_values = []

for i in range(len(test_tft)):
    x, y = test_tft[i]
    # y is the target value (already in log scale from the dataset)
    if torch.is_tensor(y):
        actual_val = y[0].item() if y.dim() > 0 else y.item()
    else:
        actual_val = float(y[0]) if isinstance(y, (list, tuple, np.ndarray)) else float(y)
    tft_actual_log_values.append(actual_val)

tft_actual_log_values = np.array(tft_actual_log_values)

print(f"✓ Collected {len(tft_pred_log_values)} predictions")
print(f"✓ Collected {len(tft_actual_log_values)} actual values")

# Align lengths (predictions may be fewer due to TimeSeriesDataSet windowing)
min_length = min(len(tft_pred_log_values), len(tft_actual_log_values))
tft_pred_values = tft_pred_log_values[:min_length]
tft_pred_q10 = tft_pred_q10_log[:min_length]
tft_pred_q90 = tft_pred_q90_log[:min_length]
tft_actual_values = tft_actual_log_values[:min_length]

print(f"✓ Aligned to {min_length} samples")
print(f"  Prediction intervals available: Yes (10th and 90th percentiles)")

# Filter to TRUE TEST PERIOD (2023+) only
split_date = pd.Timestamp('2023-01-01')

print(f"\n✓ Filtering predictions to TRUE TEST PERIOD (2023+):")
print(f"  Total predictions collected: {len(tft_pred_values)}")
print(f"  Total actual values: {len(tft_actual_values)}")

# Get all dates from augmented test data
all_test_dates = tft_test_data['Date'].values

# Find where 2023+ starts
test_period_mask = pd.to_datetime(all_test_dates) >= split_date
test_dates_2023plus = all_test_dates[test_period_mask]

print(f"\nDate range analysis:")
print(f"  Test data dates: {pd.to_datetime(all_test_dates).min()} to {pd.to_datetime(all_test_dates).max()}")
print(f"  2023+ samples: {len(test_dates_2023plus)} (from {test_dates_2023plus[0]} to {test_dates_2023plus[-1]})")

# Take the last N predictions where N = len(test_dates_2023plus)
# This corresponds to the 2023+ period
if len(tft_pred_values) >= len(test_dates_2023plus):
    tft_pred_values_test = tft_pred_values[-len(test_dates_2023plus):]
    tft_actual_values_test = tft_actual_values[-len(test_dates_2023plus):]
    tft_pred_q10_test = tft_pred_q10[-len(test_dates_2023plus):]
    tft_pred_q90_test = tft_pred_q90[-len(test_dates_2023plus):]
    test_dates_filtered = test_dates_2023plus
else:
    print(f"\nWARNING: Only {len(tft_pred_values)} predictions for {len(test_dates_2023plus)} expected dates")
    tft_pred_values_test = tft_pred_values
    tft_actual_values_test = tft_actual_values
    tft_pred_q10_test = tft_pred_q10
    tft_pred_q90_test = tft_pred_q90
    test_dates_filtered = test_dates_2023plus[-len(tft_pred_values):]

print(f"\n  Final result:")
print(f"    Predictions: {len(tft_pred_values_test)}")
print(f"    Date range: {test_dates_filtered[0]} to {test_dates_filtered[-1]}")

# Create series with proper date indices (only test period)
tft_pred_series = pd.Series(
    tft_pred_values_test,
    index=pd.to_datetime(test_dates_filtered),
    name='TFT_predictions_log'
)

tft_actual_series = pd.Series(
    tft_actual_values_test,
    index=pd.to_datetime(test_dates_filtered),
    name='Actual_log'
)

print(f"\n✓ Created prediction series with date index:")
print(f"  Index: {tft_pred_series.index.min()} to {tft_pred_series.index.max()}")

# Calculate metrics (convert to variance scale)
print(f"\nConverting from log scale to variance scale:")
print(f"  Log predictions (first 3): {tft_pred_series.values[:3]}")
print(f"  Log actuals (first 3): {tft_actual_series.values[:3]}")

tft_pred_var = np.exp(tft_pred_series + 0.5 * residual_variance)
tft_actual_var = np.exp(tft_actual_series)

print(f"  Variance predictions (first 3): {tft_pred_var.values[:3]}")
print(f"  Variance actuals (first 3): {tft_actual_var.values[:3]}")

print(f"\n✓ TFT predictions generated on TEST SET (2023+)")
print(f"  SAME evaluation period as GBM/XGBoost/LightGBM/CatBoost")
print(f"  Ready for fair comparison!")

# Filter out extreme values that cause MSPE explosion
# Keep only reasonable variance values (> 1e-8 to avoid division by near-zero)
valid_mask = (tft_actual_var > 1e-8) & (tft_pred_var > 1e-8) & np.isfinite(tft_actual_var) & np.isfinite(tft_pred_var)

tft_qlike = Metric_Evaluation.qlike(tft_actual_var[valid_mask], tft_pred_var[valid_mask])
tft_mspe = Metric_Evaluation.mspe(tft_actual_var[valid_mask], tft_pred_var[valid_mask])

print(f"\n{'='*80}")
print("TFT MODEL PERFORMANCE ON TRUE TEST SET (2023+)")
print("="*80)
print(f"✓ Evaluated on SAME test period as GBM/XGBoost/LightGBM/CatBoost")
print(f"  Test period: {tft_pred_series.index.min()} to {tft_pred_series.index.max()}")
print(f"  Test samples: {len(tft_pred_series)}")
print(f"\nMetrics:")
print(f"  QLIKE Mean: {tft_qlike.mean():.4f} ± {tft_qlike.std():.4f}")
print(f"  MSPE Mean:  {tft_mspe.mean():.4f} ± {tft_mspe.std():.4f}")

# Calculate and log RMSE and Accuracy (simple RMSE, not rolling)
tft_rmse = np.sqrt(np.mean((tft_actual_var - tft_pred_var) ** 2))
tft_accuracy = Metric_Evaluation.calculate_directional_accuracy(tft_actual_var, tft_pred_var)

print(f"  RMSE:       {tft_rmse:.4f}")
print(f"  Directional Accuracy: {tft_accuracy:.4f} ({tft_accuracy*100:.2f}%)")
print(f"="*80)

# Add TFT results to report
add_tft_results_to_report(tft_qlike, tft_mspe, tft_pred_var, tft_actual_var, 
                           tft_pred_series, tft_actual_series, 
                           len(training_tft), len(validation_tft), report, plt,
                           tft_pred_q10_test, tft_pred_q90_test)

print("\n" + "="*80)
print("✓ TFT EVALUATION COMPLETE")
print("="*80)
print("✓ TFT predictions generated on 2023+ test set")
print("✓ SAME evaluation period as GBM/XGBoost/LightGBM/CatBoost")
print("✓ Ready for fair comparison - all models evaluated on same test data!")
print("="*80)

# =============================================================================
# Save TFT Predictions to CSV
# =============================================================================
print("\n" + "="*80)
print("SAVING TFT PREDICTIONS TO CSV")
print("="*80)

# Create a DataFrame with the true values and the TFT predictions (in variance scale)
# tft_pred_var and tft_actual_var are already exponentiated Series with date index
tft_predictions_df = pd.DataFrame({
    'RV_true': tft_actual_var.values,
    'TFT_prediction': tft_pred_var.values
}, index=tft_pred_var.index)

print(f"\nDataFrame info:")
print(f"  Shape: {tft_predictions_df.shape}")
print(f"  RV_true range: {tft_predictions_df['RV_true'].min():.6f} to {tft_predictions_df['RV_true'].max():.6f}")
print(f"  TFT_prediction range: {tft_predictions_df['TFT_prediction'].min():.6f} to {tft_predictions_df['TFT_prediction'].max():.6f}")
print(f"  Sample values (first 3):")
print(tft_predictions_df.head(3))

# Save to CSV
tft_predictions_df.to_csv('tft_predictions.csv', index_label='Date')
print("✓ TFT predictions saved to tft_predictions.csv")
print("="*80)


print("\n" + "="*80)
print("PHASE 7: COMPREHENSIVE MODEL COMPARISON")
print("="*80)


print("\n" + "="*80)
print("PHASE 6: FINALIZING REPORT")
print("="*80)

report.finalize_report()

print("\n" + "="*80)
print("✓ ANALYSIS COMPLETE!")
print("="*80)
print(f"\nResults summary:")
print(f"  • TFT model trained: ✓")
print(f"  • TFT QLIKE: {tft_qlike.mean():.6f} ± {tft_qlike.std():.6f}")
print(f"  • TFT MSPE: {tft_mspe.mean():.6f} ± {tft_mspe.std():.6f}")
print(f"  • TFT RMSE: {np.sqrt(np.mean((tft_actual_var - tft_pred_var) ** 2)):.6f}")
print(f"\nReport saved to: {report.report_file}")
print("="*80)
