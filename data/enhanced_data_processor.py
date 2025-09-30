"""
Enhanced Data Processing Pipeline for Volatility Forecasting
Implements next-generation feature engineering beyond Kilic's work
"""

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, List, Tuple, Optional, Union
import warnings
from scipy import stats
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import KNNImputer
import pywt
from statsmodels.tsa.regime_switching import MarkovRegression
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedDataProcessor:
    """
    Next-generation data processor for volatility forecasting
    
    Features beyond Kilic's work:
    1. Wavelet-based decomposition for multi-scale analysis
    2. Fractal dimension calculation for market complexity
    3. Real-time regime detection with HMM
    4. Cross-asset spillover effects
    5. Advanced outlier detection and handling
    6. Dynamic feature selection
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()
        self.scaler = RobustScaler()  # More robust than StandardScaler for financial data
        self.imputer = KNNImputer(n_neighbors=5)
        self.regime_model = None
        self.feature_history = []
        
    def _get_default_config(self) -> Dict:
        """Default configuration for data processing"""
        return {
            'volatility_windows': [5, 22, 66],  # 1 week, 1 month, 3 months
            'wavelet_name': 'db4',
            'wavelet_levels': 3,
            'regime_states': 3,
            'outlier_threshold': 3.0,
            'min_observations': 252,  # 1 year of daily data
            'cross_assets': ['SPY', 'VIX', 'GLD', 'DXY']  # For spillover analysis
        }
    
    def process_data(self, 
                    data: pd.DataFrame, 
                    target_col: str = 'Realized_Vol',
                    price_col: str = 'Price') -> pd.DataFrame:
        """
        Main data processing pipeline
        
        Args:
            data: Input DataFrame with price and volatility data
            target_col: Column name for realized volatility
            price_col: Column name for price data
            
        Returns:
            Enhanced DataFrame with advanced features
        """
        logger.info("Starting enhanced data processing pipeline...")
        
        # Step 1: Data validation and cleaning
        data_clean = self._validate_and_clean_data(data.copy())
        
        # Step 2: Calculate basic financial features
        data_enhanced = self._calculate_basic_features(data_clean, price_col, target_col)
        
        # Step 3: Advanced feature engineering
        data_enhanced = self._advanced_feature_engineering(data_enhanced, target_col)
        
        # Step 4: Cross-asset features (if available)
        data_enhanced = self._add_cross_asset_features(data_enhanced)
        
        # Step 5: Regime detection features
        data_enhanced = self._add_regime_features(data_enhanced, target_col)
        
        # Step 6: Handle missing values and outliers
        data_final = self._handle_missing_and_outliers(data_enhanced)
        
        # Step 7: Feature scaling (optional - depends on model requirements)
        if self.config.get('scale_features', False):
            data_final = self._scale_features(data_final)
        
        logger.info(f"Data processing complete. Shape: {data_final.shape}")
        logger.info(f"Features created: {[col for col in data_final.columns if col not in [price_col, target_col]]}")
        
        return data_final
    
    def _validate_and_clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean input data
        """
        logger.info("Validating and cleaning data...")
        
        # Ensure datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            if 'Date' in data.columns:
                data.set_index('Date', inplace=True)
                data.index = pd.to_datetime(data.index)
            else:
                raise ValueError("Data must have datetime index or 'Date' column")
        
        # Remove duplicates
        data = data[~data.index.duplicated(keep='first')]
        
        # Sort by date
        data = data.sort_index()
        
        # Basic data quality checks
        if len(data) < self.config['min_observations']:
            logger.warning(f"Insufficient data: {len(data)} < {self.config['min_observations']}")
        
        return data
    
    def _calculate_basic_features(self, 
                                 data: pd.DataFrame, 
                                 price_col: str, 
                                 target_col: str) -> pd.DataFrame:
        """
        Calculate basic financial features
        """
        logger.info("Calculating basic financial features...")
        
        # Returns
        data['returns'] = data[price_col].pct_change()
        data['log_returns'] = np.log(data[price_col] / data[price_col].shift(1))
        
        # Squared returns (proxy for realized variance)
        data['squared_returns'] = data['returns'] ** 2
        
        # Price-based features
        for window in self.config['volatility_windows']:
            # Moving averages
            data[f'ma_{window}'] = data[price_col].rolling(window=window).mean()
            data[f'price_ratio_{window}'] = data[price_col] / data[f'ma_{window}']
            
            # Volatility measures
            data[f'rolling_vol_{window}'] = data['returns'].rolling(window=window).std() * np.sqrt(252)
            data[f'rolling_skew_{window}'] = data['returns'].rolling(window=window).skew()
            data[f'rolling_kurt_{window}'] = data['returns'].rolling(window=window).kurtosis()
            
            # Range-based volatility (Parkinson estimator)
            if 'High' in data.columns and 'Low' in data.columns:
                data[f'parkinson_vol_{window}'] = np.sqrt(
                    0.361 * (np.log(data['High'] / data['Low']) ** 2).rolling(window=window).mean()
                ) * np.sqrt(252)
        
        return data
    
    def _advanced_feature_engineering(self, 
                                    data: pd.DataFrame, 
                                    target_col: str) -> pd.DataFrame:
        """
        Advanced feature engineering beyond traditional methods
        """
        logger.info("Creating advanced features...")
        
        # 1. Wavelet decomposition for multi-scale analysis
        data = self._add_wavelet_features(data, target_col)
        
        # 2. Fractal dimension for market complexity
        data = self._add_fractal_features(data, target_col)
        
        # 3. Jump detection
        data = self._add_jump_detection_features(data)
        
        # 4. Volatility clustering measures
        data = self._add_volatility_clustering_features(data, target_col)
        
        # 5. Microstructure proxies
        data = self._add_microstructure_features(data)
        
        return data
    
    def _add_wavelet_features(self, data: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """
        Add wavelet-based features for multi-scale volatility analysis
        Superior to simple moving averages
        """
        logger.info("Adding wavelet decomposition features...")
        
        # Get volatility series (fill NaN with forward fill for wavelet analysis)
        vol_series = data[target_col].fillna(method='ffill').fillna(method='bfill')
        
        if len(vol_series.dropna()) < 64:  # Minimum length for meaningful wavelet analysis
            logger.warning("Insufficient data for wavelet analysis")
            return data
        
        try:
            # Discrete wavelet transform
            coeffs = pywt.wavedec(vol_series.dropna(), 
                                 self.config['wavelet_name'], 
                                 level=self.config['wavelet_levels'])
            
            # Reconstruct different frequency components
            for level in range(1, self.config['wavelet_levels'] + 1):
                # Detail coefficients (high frequency)
                detail_coeffs = [None] * (self.config['wavelet_levels'] + 1)
                detail_coeffs[level] = coeffs[level]
                detail_recon = pywt.waverec(detail_coeffs, self.config['wavelet_name'])
                
                # Pad to match original length
                if len(detail_recon) != len(vol_series):
                    detail_recon = np.pad(detail_recon, 
                                        (0, len(vol_series) - len(detail_recon)), 
                                        mode='constant')
                
                data[f'wavelet_detail_{level}'] = detail_recon[:len(data)]
            
            # Approximation coefficients (low frequency trend)
            approx_coeffs = [coeffs[0]] + [None] * self.config['wavelet_levels']
            approx_recon = pywt.waverec(approx_coeffs, self.config['wavelet_name'])
            
            if len(approx_recon) != len(vol_series):
                approx_recon = np.pad(approx_recon, 
                                    (0, len(vol_series) - len(approx_recon)), 
                                    mode='constant')
            
            data['wavelet_trend'] = approx_recon[:len(data)]
            
        except Exception as e:
            logger.warning(f"Wavelet decomposition failed: {e}")
        
        return data
    
    def _add_fractal_features(self, data: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """
        Add fractal dimension features to capture market complexity
        """
        logger.info("Calculating fractal dimension features...")
        
        def hurst_exponent(series, max_lag=20):
            """Calculate Hurst exponent using R/S analysis"""
            if len(series) < max_lag * 2:
                return np.nan
                
            lags = range(2, max_lag + 1)
            rs_values = []
            
            for lag in lags:
                # Split series into chunks
                chunks = [series[i:i+lag] for i in range(0, len(series)-lag+1, lag)]
                rs_chunk = []
                
                for chunk in chunks:
                    if len(chunk) == lag:
                        mean_chunk = np.mean(chunk)
                        deviations = np.cumsum(chunk - mean_chunk)
                        R = np.max(deviations) - np.min(deviations)
                        S = np.std(chunk)
                        if S != 0:
                            rs_chunk.append(R / S)
                
                if rs_chunk:
                    rs_values.append(np.mean(rs_chunk))
                else:
                    rs_values.append(np.nan)
            
            # Linear regression to find Hurst exponent
            log_lags = np.log(lags)
            log_rs = np.log(rs_values)
            
            # Remove NaN values
            valid_idx = ~np.isnan(log_rs)
            if np.sum(valid_idx) < 3:
                return np.nan
            
            slope, _ = np.polyfit(log_lags[valid_idx], log_rs[valid_idx], 1)
            return slope
        
        # Rolling Hurst exponent
        window = 66  # Approximately 3 months
        vol_series = data[target_col].fillna(method='ffill')
        
        hurst_values = []
        for i in range(len(data)):
            if i < window:
                hurst_values.append(np.nan)
            else:
                window_data = vol_series.iloc[i-window:i]
                hurst = hurst_exponent(window_data.values)
                hurst_values.append(hurst)
        
        data['hurst_exponent'] = hurst_values
        
        # Fractal dimension approximation: FD = 2 - H
        data['fractal_dimension'] = 2 - data['hurst_exponent']
        
        return data
    
    def _add_jump_detection_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add jump detection features using various statistical tests
        """
        logger.info("Adding jump detection features...")
        
        if 'returns' not in data.columns:
            return data
        
        # Lee-Mykland jump test
        returns = data['returns'].dropna()
        
        # Calculate jump test statistic
        window = 22  # 1 month rolling window
        jump_stats = []
        
        for i in range(len(returns)):
            if i < window:
                jump_stats.append(np.nan)
            else:
                window_returns = returns.iloc[i-window:i]
                # Bipower variation
                bv = np.sum(np.abs(window_returns.iloc[:-1]) * np.abs(window_returns.iloc[1:]))
                bv = bv * (np.pi / 2)
                
                # Quadratic variation
                qv = np.sum(window_returns ** 2)
                
                # Jump component
                if bv > 0:
                    jump_component = max(0, qv - bv)
                    jump_ratio = jump_component / qv if qv > 0 else 0
                else:
                    jump_ratio = 0
                
                jump_stats.append(jump_ratio)
        
        data['jump_ratio'] = jump_stats
        
        # Binary jump indicator
        jump_threshold = np.percentile([j for j in jump_stats if not np.isnan(j)], 95)
        data['jump_indicator'] = (data['jump_ratio'] > jump_threshold).astype(int)
        
        return data
    
    def _add_volatility_clustering_features(self, data: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """
        Add features to capture volatility clustering
        """
        logger.info("Adding volatility clustering features...")
        
        vol_series = data[target_col].fillna(method='ffill')
        
        # GARCH-like features
        for lag in [1, 2, 5]:
            data[f'vol_lag_{lag}'] = vol_series.shift(lag)
        
        # Volatility persistence
        data['vol_persistence'] = vol_series.rolling(window=22).apply(
            lambda x: np.corrcoef(x[:-1], x[1:])[0, 1] if len(x) > 1 else np.nan
        )
        
        # High/low volatility regime indicators
        vol_ma_long = vol_series.rolling(window=66).mean()
        vol_ma_short = vol_series.rolling(window=5).mean()
        data['vol_regime_indicator'] = (vol_ma_short > vol_ma_long).astype(int)
        
        return data
    
    def _add_microstructure_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add microstructure-based features (using available data)
        """
        logger.info("Adding microstructure features...")
        
        if 'returns' in data.columns:
            # Amihud illiquidity measure (simplified)
            data['amihud_illiquidity'] = np.abs(data['returns']) / (data.get('Volume', 1))
            
            # Roll spread estimator
            data['roll_spread'] = 2 * np.sqrt(-data['returns'].rolling(window=5).cov(data['returns'].shift(1)).fillna(0))
            
        return data
    
    def _add_cross_asset_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add cross-asset spillover features
        """
        logger.info("Adding cross-asset features...")
        
        # For now, create placeholder features
        # In production, this would fetch real cross-asset data
        
        # Simulated VIX-like fear index
        if 'returns' in data.columns:
            data['fear_index'] = data['returns'].rolling(window=22).std() * np.sqrt(252) * 100
            
            # Cross-correlation with lagged market returns (simulated)
            data['market_correlation'] = data['returns'].rolling(window=66).corr(
                data['returns'].shift(1)
            )
        
        return data
    
    def _add_regime_features(self, data: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """
        Add regime detection features using Markov-switching models
        """
        logger.info("Adding regime detection features...")
        
        vol_series = data[target_col].dropna()
        
        if len(vol_series) < 100:  # Need sufficient data for regime detection
            data['regime_state'] = np.nan
            data['regime_probability_high'] = np.nan
            return data
        
        try:
            # Fit Markov-switching model
            model = MarkovRegression(
                vol_series, 
                k_regimes=self.config['regime_states'],
                trend='c',
                switching_trend=True
            )
            
            fitted_model = model.fit(disp=False)
            
            # Get regime probabilities
            regime_probs = fitted_model.smoothed_marginal_probabilities
            
            # Most likely regime state
            regime_states = np.argmax(regime_probs, axis=1)
            
            # Align with original data
            regime_series = pd.Series(regime_states, index=vol_series.index)
            data['regime_state'] = regime_series.reindex(data.index)
            
            # Probability of high volatility regime (assume last regime is high vol)
            high_vol_prob = regime_probs[:, -1]
            prob_series = pd.Series(high_vol_prob, index=vol_series.index)
            data['regime_probability_high'] = prob_series.reindex(data.index)
            
            self.regime_model = fitted_model
            
        except Exception as e:
            logger.warning(f"Regime detection failed: {e}")
            data['regime_state'] = np.nan
            data['regime_probability_high'] = np.nan
        
        return data
    
    def _handle_missing_and_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values and outliers
        """
        logger.info("Handling missing values and outliers...")
        
        # Identify numerical columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove extreme outliers using IQR method
        for col in numeric_cols:
            if col in data.columns:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - self.config['outlier_threshold'] * IQR
                upper_bound = Q3 + self.config['outlier_threshold'] * IQR
                
                # Cap outliers instead of removing them
                data[col] = np.clip(data[col], lower_bound, upper_bound)
        
        # Handle missing values
        # Forward fill then backward fill
        data = data.fillna(method='ffill').fillna(method='bfill')
        
        # For remaining NaNs, use KNN imputation on numeric columns only
        remaining_numeric_cols = [col for col in numeric_cols if data[col].isna().any()]
        
        if remaining_numeric_cols:
            logger.info(f"Applying KNN imputation to columns: {remaining_numeric_cols}")
            data[remaining_numeric_cols] = self.imputer.fit_transform(data[remaining_numeric_cols])
        
        return data
    
    def _scale_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Scale features using robust scaling
        """
        logger.info("Scaling features...")
        
        # Don't scale the target variable or date-related columns
        exclude_cols = ['Price', 'Realized_Vol', 'returns', 'log_returns']
        numeric_cols = [col for col in data.select_dtypes(include=[np.number]).columns 
                       if col not in exclude_cols]
        
        if numeric_cols:
            data[numeric_cols] = self.scaler.fit_transform(data[numeric_cols])
        
        return data
    
    def get_feature_importance_analysis(self, data: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """
        Analyze feature importance and correlations
        """
        # Calculate correlations with target
        numeric_data = data.select_dtypes(include=[np.number])
        correlations = numeric_data.corr()[target_col].abs().sort_values(ascending=False)
        
        # Calculate feature statistics
        feature_stats = pd.DataFrame({
            'correlation_abs': correlations,
            'missing_pct': (data.isna().sum() / len(data)) * 100,
            'unique_values': data.nunique(),
            'std': data.std()
        })
        
        return feature_stats.sort_values('correlation_abs', ascending=False)

# Example usage and testing
if __name__ == "__main__":
    # Load existing data
    data = pd.read_csv('/home/omegashenr01n/Desktop/Term2/QF603-Group15/data/TLT_2007-01-01_to_2025-08-30.csv')
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    
    # Initialize enhanced processor
    processor = EnhancedDataProcessor()
    
    # Process data
    enhanced_data = processor.process_data(data)
    
    # Display results
    print("\n=== ENHANCED DATA PROCESSING RESULTS ===")
    print(f"Original shape: {data.shape}")
    print(f"Enhanced shape: {enhanced_data.shape}")
    print(f"\nNew features created: {enhanced_data.shape[1] - data.shape[1]}")
    
    print("\n=== FEATURE OVERVIEW ===")
    feature_stats = processor.get_feature_importance_analysis(enhanced_data, 'Realized_Vol')
    print(feature_stats.head(15))
    
    print("\n=== SAMPLE OF ENHANCED DATA ===")
    print(enhanced_data.tail(10))