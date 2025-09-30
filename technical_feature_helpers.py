import talib

def calculate_rsi_talib(close_prices, period=14):
    """
    Calculate RSI using TA-Lib
    """
    return talib.RSI(close_prices, timeperiod=period)

def calculate_macd_talib(prices, fast=12, slow=26, signal=9):
    """
    Calculate MACD using TA-Lib
    """
    macd, macd_signal, macd_hist = talib.MACD(prices, 
                                             fastperiod=fast, 
                                             slowperiod=slow, 
                                             signalperiod=signal)
    
    return macd

def calculate_bull_bear_power(df, period=13):
    """
    Calculate Bull Bear Power (Elder-Ray Index)
    
    Formula:
    Bull Power = High - EMA(Close, period)
    Bear Power = Low - EMA(Close, period)
    
    Parameters:
    df: DataFrame with columns ['high', 'low', 'close']
    period: EMA period (default 13)
    
    Returns:
    DataFrame with Bull Power and Bear Power columns
    """
    # Calculate EMA of closing prices
    df['EMA'] = df['Price'].ewm(span=period, adjust=False).mean()
    
    # Calculate Bull Power and Bear Power
    df['Bull_Power'] = df['High'] - df['EMA']
    df['Bear_Power'] = df['Low'] - df['EMA']
    
    return df