import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import glob

def plot_predictions_dashboard():
    """
    Loads prediction data from multiple CSV files and creates an interactive
    Plotly dashboard to compare model forecasts against true volatility.
    Includes multiple volatility estimators and ensemble predictions across different windows.
    """
    print("="*80)
    print("Creating Interactive Volatility Predictions Dashboard")
    print("Includes multiple volatility estimators and ensemble predictions")
    print("="*80)

    # --- 1. Load Data ---
    print("Loading data from CSV files...")
    
    # Define file paths
    files = {
        'har': 'yhat_log_har.csv',
        'harx': 'yhat_log_harx.csv'
    }
    
    data = {}
    for key, filename in files.items():
        if os.path.exists(filename):
            print(f"  ✓ Loading {filename}...")
            df = pd.read_csv(filename, index_col='Date', parse_dates=True)
            # Filter to test set dates (2023 onwards)
            df = df[df.index >= '2023-01-01']
            data[key] = df
        else:
            print(f"  ✗ WARNING: {filename} not found. Skipping.")
            data[key] = None

    # Load individual ML model prediction files
    ml_model_types = ['rf', 'gbm', 'xgboost', 'lightgbm', 'catboost']
    ml_data = {}
    for model_name in ml_model_types:
        filename = f'ml_{model_name}_predictions.csv'
        if os.path.exists(filename):
            print(f"  ✓ Loading {filename}...")
            df = pd.read_csv(filename, index_col='Date', parse_dates=True)
            # Filter to test set dates (2023 onwards)
            df = df[df.index >= '2023-01-01']
            ml_data[model_name] = df
        else:
            print(f"  ✗ WARNING: {filename} not found. Skipping.")
            ml_data[model_name] = None

    # Load all TFT prediction files
    tft_files = glob.glob('tft_predictions_*.csv')
    tft_data = {}
    for filename in sorted(tft_files):
        print(f"  ✓ Loading {filename}...")
        df = pd.read_csv(filename, index_col='Date', parse_dates=True)
        # Filter to test set dates (2023 onwards)
        df = df[df.index >= '2023-01-01']
        # Extract epoch number from filename
        epoch = filename.replace('tft_predictions_', '').replace('.csv', '')
        tft_data[f'tft_{epoch}'] = df

    if not any(val is not None for val in data.values()) and not ml_data and not tft_data:
        print("\nError: No data files found. Please run the model scripts first.")
        print("="*80)
        return

    # --- 2. Initialize Figure ---
    fig = go.Figure()

    # --- 3. Add True Volatility Trace ---
    # Find the true volatility column (could have different names)
    y_true = None
    if data.get('har') is not None and 'RV_squared_return' in data['har'].columns:
        y_true = data['har']['RV_squared_return'].dropna()
        true_col_name = 'RV_squared_return'
    elif ml_data and any('RV_true' in df.columns for df in ml_data.values() if df is not None):
        # Use RV_true from ML files
        y_true = next(df['RV_true'].dropna() for df in ml_data.values() if df is not None and 'RV_true' in df.columns)
        true_col_name = 'RV_true (from ML)'
    
    if y_true is not None:
        fig.add_trace(go.Scatter(
            x=y_true.index,
            y=y_true,
            mode='lines',
            name='True RV (Squared Return)',
            line=dict(color='black', width=2.5),
            visible=True  # Always visible
        ))
        print(f"  ✓ Added 'True Realized Volatility' trace from '{true_col_name}'.")

    # Add other true volatility estimates from HAR data
    if data.get('har') is not None:
        rv_columns = ['RV_parkinson', 'RV_garman_klass', 'RV_rogers_satchell']
        rv_colors = ['gray', 'dimgray', 'darkgray']
        rv_names = ['Parkinson', 'Garman-Klass', 'Rogers-Satchell']
        
        for col, color, name in zip(rv_columns, rv_colors, rv_names):
            if col in data['har'].columns:
                rv_series = data['har'][col].dropna()
                fig.add_trace(go.Scatter(
                    x=rv_series.index,
                    y=rv_series,
                    mode='lines',
                    name=f'True RV ({name})',
                    line=dict(color=color, width=1.5, dash='dot'),
                    visible='legendonly'  # Hidden by default, can be toggled
                ))
                print(f"  ✓ Added '{col}' trace.")

    # --- 4. Add Model Prediction Traces ---
    # Define key models to plot by default for clarity
    key_models = {
        'HAR': ('har', ['HAR_Ensemble_w252', 'HAR_Ensemble_w504', 'HAR_Ensemble_w756', 'HAR_Ensemble_w1008', 'HAR_Ensemble_w1260']),
        'HARX': ('harx', ['HARX_Ensemble_w252', 'HARX_Ensemble_w504', 'HARX_Ensemble_w756', 'HARX_Ensemble_w1008', 'HARX_Ensemble_w1260'])
    }

    model_colors = {
        'HAR': ['royalblue', 'blue', 'darkblue', 'navy', 'midnightblue'],
        'HARX': ['firebrick', 'red', 'darkred', 'crimson', 'maroon']
    }

    for model_family, (data_key, col_names) in key_models.items():
        if data.get(data_key) is not None:
            colors = model_colors.get(model_family, ['gray'] * len(col_names))
            for col_name, color in zip(col_names, colors):
                if col_name in data[data_key].columns:
                    df = data[data_key]
                    # Extract window number for display
                    window_num = col_name.split('_w')[-1]
                    visibility = 'legendonly' if window_num != '756' else True  # Show w756 by default
                    
                    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df[col_name],
                        mode='lines',
                        name=f'{model_family} Ensemble w{window_num}',
                        line=dict(color=color, width=1.5, dash='dash'),
                        visible=visibility
                    ))
                    print(f"  ✓ Added '{col_name}' trace.")

    # Add individual ML model predictions
    ml_colors = ['green', 'lime', 'seagreen', 'mediumseagreen', 'darkgreen']
    ml_model_names = {
        'rf': 'Random Forest',
        'gbm': 'Gradient Boosting',
        'xgboost': 'XGBoost',
        'lightgbm': 'LightGBM',
        'catboost': 'CatBoost'
    }
    
    for i, (model_key, df) in enumerate(ml_data.items()):
        if df is not None:
            col_name = f'ML_{model_key}_w756'  # Using w756 as we saved with best_window
            if col_name in df.columns:
                color = ml_colors[i % len(ml_colors)]
                model_display_name = ml_model_names.get(model_key, model_key.upper())
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df[col_name],
                    mode='lines',
                    name=f'{model_display_name}',
                    line=dict(color=color, width=1.5, dash='solid'),
                    visible=True
                ))
                print(f"  ✓ Added {model_display_name} trace.")

    # Add TFT epoch predictions
    tft_colors = ['darkorange', 'orange', 'gold', 'yellow', 'lightyellow']
    for i, (key, df) in enumerate(tft_data.items()):
        epoch = key.replace('tft_', '')
        color = tft_colors[i % len(tft_colors)]
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['TFT_prediction'],
            mode='lines',
            name=f'TFT Epoch {epoch}',
            line=dict(color=color, width=1.5, dash='dot'),
            visible=True
        ))
        print(f"  ✓ Added TFT Epoch {epoch} trace.")

    # --- 5. Configure Layout and Interactivity ---
    print("\nConfiguring plot layout and interactivity...")
    
    fig.update_layout(
        title_text='<b>Volatility Forecast Dashboard: Model Comparison with Multiple Parameters</b>',
        title_x=0.5,
        xaxis_title='Date',
        yaxis_title='Realized Volatility (Variance)',
        legend_title_text='Models & Parameters',
        template='plotly_white',
        height=700,
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        )
    )

    # --- 6. Save to HTML ---
    output_filename = 'volatility_predictions_dashboard.html'
    fig.write_html(output_filename)
    
    print("\n" + "="*80)
    print(f"✓ Successfully created interactive dashboard!")
    print(f"  Saved to: {os.path.abspath(output_filename)}")
    print("  Features: Multiple volatility estimators, ensemble predictions across windows,")
    print("            ML models, and TFT predictions with interactive controls.")
    print("  Open this HTML file in your browser to view the chart.")
    print("="*80)

if __name__ == '__main__':
    plot_predictions_dashboard()
