import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

def plot_predictions_dashboard():
    """
    Loads prediction data from multiple CSV files and creates an interactive
    Plotly dashboard to compare model forecasts against true volatility.
    """
    print("="*80)
    print("Creating Interactive Volatility Predictions Dashboard")
    print("="*80)

    # --- 1. Load Data ---
    print("Loading data from CSV files...")
    
    # Define file paths
    files = {
        'har': 'yhat_log_har.csv',
        'harx': 'yhat_log_harx.csv',
        'ml': 'ml_predictions.csv',
        'tft': 'tft_predictions.csv'
    }
    
    data = {}
    for key, filename in files.items():
        if os.path.exists(filename):
            print(f"  ✓ Loading {filename}...")
            df = pd.read_csv(filename, index_col='Date', parse_dates=True)
            data[key] = df
        else:
            print(f"  ✗ WARNING: {filename} not found. Skipping.")
            data[key] = None

    if not any(data.values()):
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
    elif data.get('ml') is not None and 'RV_true' in data['ml'].columns:
        y_true = data['ml']['RV_true'].dropna()
        true_col_name = 'RV_true'
    
    if y_true is not None:
        fig.add_trace(go.Scatter(
            x=y_true.index,
            y=y_true,
            mode='lines',
            name='True Realized Volatility',
            line=dict(color='black', width=2.5),
            visible=True  # Always visible
        ))
        print(f"  ✓ Added 'True Realized Volatility' trace from '{true_col_name}'.")

    # --- 4. Add Model Prediction Traces ---
    # Define key models to plot by default for clarity
    key_models = {
        'HAR': ('har', 'HAR_Ensemble_w756'),
        'HARX': ('harx', 'HARX_Ensemble_w756'),
        'ML (XGB)': ('ml', 'ML_xgboost_w756'),
        'TFT': ('tft', 'TFT_prediction')
    }

    model_colors = {
        'HAR': 'royalblue',
        'HARX': 'firebrick',
        'ML': 'green',
        'TFT': 'darkorange'
    }

    for model_family, (data_key, col_name) in key_models.items():
        if data.get(data_key) is not None and col_name in data[data_key].columns:
            df = data[data_key]
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[col_name],
                mode='lines',
                name=f'{model_family} Forecast',
                line=dict(color=model_colors.get(model_family, 'gray'), width=1.5, dash='dash'),
                visible=True  # Start with key models visible
            ))
            print(f"  ✓ Added '{col_name}' trace.")

    # --- 5. Configure Layout and Interactivity ---
    print("\nConfiguring plot layout and interactivity...")
    
    fig.update_layout(
        title_text='<b>Volatility Forecast Dashboard: Model Comparison</b>',
        title_x=0.5,
        xaxis_title='Date',
        yaxis_title='Realized Volatility (Variance)',
        legend_title_text='Models',
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
    print("  Open this HTML file in your browser to view the chart.")
    print("="*80)

if __name__ == '__main__':
    plot_predictions_dashboard()
