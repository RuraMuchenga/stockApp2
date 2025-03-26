import matplotlib
matplotlib.use('Agg')  # Thread-safe backend
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
import numpy as np
import io
import base64
from prophet import Prophet
import xgboost as xgb
import ta
import joblib
import os
from datetime import datetime, timedelta
import time

app = Flask(__name__)

CACHE_DIR = 'cache'
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

def plot_to_base64():
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    plt.close()
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf8')

def get_stock_data(symbol):
    cache_file = os.path.join(CACHE_DIR, f"{symbol}.pkl")
    today = datetime.now().date()
    
    if os.path.exists(cache_file):
        cache_time = datetime.fromtimestamp(os.path.getmtime(cache_file)).date()
        if cache_time == today:
            return joblib.load(cache_file)
    
    start_date = (today - timedelta(days=5*365)).strftime('%Y-%m-%d')  # 5 years of data
    end_date = today.strftime('%Y-%m-%d')  # Today
    data = yf.download(symbol, start=start_date, end=end_date, auto_adjust=True)
    
    if data.empty or 'Close' not in data.columns:
        return None
    
    joblib.dump(data, cache_file)
    return data

def prophet_prediction(data):
    start_time = time.time()
    try:
        df = data.reset_index()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] if col[1] == '' else f"{col[0]}_{col[1]}" for col in df.columns]
        
        date_col = None
        for col in df.columns:
            if 'date' in col.lower() or 'index' in col.lower():
                date_col = col
                break
        if date_col is None:
            date_col = df.columns[0]
        
        close_col = None
        for col in df.columns:
            if 'close' in col.lower():
                close_col = col
                break
        if close_col is None:
            return "Prophet Error: No 'Close' column found", [], []
        
        df = df[[date_col, close_col]].rename(columns={date_col: 'ds', close_col: 'y'})
        
        model = Prophet(
            changepoint_prior_scale=0.01,
            yearly_seasonality=False,
            weekly_seasonality=True,
            daily_seasonality=True
        )
        model.fit(df)
        
        future = model.make_future_dataframe(periods=5, freq='B')
        forecast = model.predict(future)
        predictions = forecast[['ds', 'yhat']].tail(5)
        
        last_date = data.index[-1]
        past_dates = pd.date_range(end=last_date, periods=10, freq='B')
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=5, freq='B')
        all_dates = list(past_dates) + list(future_dates)
        date_labels = [d.strftime('%d/%m/%y') for d in all_dates]
        
        plt.figure(figsize=(10, 5))
        actual_last_10 = df['y'].tail(10).values
        predicted_5 = predictions['yhat'].values
        x_indices = range(15)
        plt.plot(x_indices[:10], actual_last_10, label='Actual', marker='o', color='#007bff')
        plt.plot(x_indices[10:], predicted_5, label='Predicted', marker='o', color='#ff9900')
        plt.title(f'Prophet: 5-Day Forecast', fontsize=14, pad=15)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price', fontsize=12)
        plt.xticks(x_indices, date_labels, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plot_data = plot_to_base64()
        
        print(f"Prophet - Total time: {time.time() - start_time:.2f}s")
        return plot_data, predictions['yhat'].tolist(), [d.strftime('%d/%m/%y') for d in future_dates]
    except Exception as e:
        print(f"Prophet - Failed with error: {str(e)} at {time.time() - start_time:.2f}s")
        plt.figure(figsize=(10, 5))
        plt.text(0.5, 0.5, 'Prophet Plot Failed', ha='center', va='center')
        plot_data = plot_to_base64()
        return plot_data, [], []

def xgboost_prediction(data):
    start_time = time.time()
    try:
        df = data.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] if col[1] == '' else f"{col[0]}_{col[1]}" for col in df.columns]
        
        close_col = None
        open_col = None
        high_col = None
        low_col = None
        volume_col = None
        for col in df.columns:
            if 'close' in col.lower():
                close_col = col
            elif 'open' in col.lower():
                open_col = col
            elif 'high' in col.lower():
                high_col = col
            elif 'low' in col.lower():
                low_col = col
            elif 'volume' in col.lower():
                volume_col = col
        
        if not all([close_col, open_col, high_col, low_col, volume_col]):
            return "XGBoost Error: Missing required columns", [], []
        
        # Calculate indicators and minimize dropna
        df['MA5'] = df[close_col].rolling(window=5).mean()
        df['MA10'] = df[close_col].rolling(window=10).mean()
        df['RSI'] = ta.momentum.RSIIndicator(df[close_col], window=14).rsi()
        df['MACD'] = ta.trend.MACD(df[close_col]).macd()
        
        # Drop NaN values once after all indicators are calculated
        df = df.dropna()
        if len(df) < 20:  # Ensure enough data for training
            return "XGBoost Error: Not enough data after dropping NaNs", [], []
        
        # Create target
        df['Target'] = df[close_col].shift(-1)
        df = df.dropna()  # Drop rows where Target is NaN
        
        if len(df) < 20:
            return "XGBoost Error: Not enough data after creating target", [], []
        
        # Prepare features and target
        features = [open_col, high_col, low_col, close_col, volume_col, 'MA5', 'MA10', 'RSI', 'MACD']
        X = df[features].values
        y = df['Target'].values
        
        # Adjust train-test split to ensure enough test data
        train_size = max(1, int(len(df) * 0.8))  # At least 1 row for training
        if len(df) - train_size < 5:  # Ensure at least 5 rows for testing
            train_size = len(df) - 5
        
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Train XGBoost
        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=200, learning_rate=0.05)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test[-5:])
        
        # Plotting: Last 10 actual + 5 predicted with dates
        last_date = data.index[-1]
        past_dates = pd.date_range(end=last_date, periods=10, freq='B')
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=5, freq='B')
        all_dates = list(past_dates) + list(future_dates)
        date_labels = [d.strftime('%d/%m/%y') for d in all_dates]
        
        plt.figure(figsize=(10, 5))
        actual_last_10 = y_test[-10:] if len(y_test) >= 10 else y_test  # Handle case where y_test < 10
        predicted_5 = predictions
        x_indices = range(15)
        plt.plot(x_indices[:len(actual_last_10)], actual_last_10, label='Actual', marker='o', color='#007bff')
        plt.plot(x_indices[10:15], predicted_5, label='Predicted', marker='o', color='#ff9900')
        plt.title(f'XGBoost: 5-Day Forecast', fontsize=14, pad=15)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price', fontsize=12)
        plt.xticks(x_indices, date_labels, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plot_data = plot_to_base64()
        
        print(f"XGBoost - Total time: {time.time() - start_time:.2f}s")
        return plot_data, predictions.tolist(), [d.strftime('%d/%m/%y') for d in future_dates]
    except Exception as e:
        print(f"XGBoost - Failed with error: {str(e)} at {time.time() - start_time:.2f}s")
        plt.figure(figsize=(10, 5))
        plt.text(0.5, 0.5, 'XGBoost Plot Failed', ha='center', va='center')
        plot_data = plot_to_base64()
        return plot_data, [], []

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        stock_symbol = request.form['stock_symbol'].upper()
        data = get_stock_data(stock_symbol)
        if data is None:
            return "Error: Failed to fetch stock data. Check the symbol."

        prophet_result, prophet_preds, prophet_dates = prophet_prediction(data)
        xgboost_result, xgboost_preds, xgboost_dates = xgboost_prediction(data)

        latest_data = data.iloc[-1]
        stock_info = {col: round(latest_data[col], 2) for col in ['Open', 'High', 'Low', 'Close', 'Volume']}

        return render_template('results.html', stock_symbol=stock_symbol, stock_info=stock_info,
                               prophet_plot=prophet_result, xgboost_plot=xgboost_result,
                               prophet_preds=prophet_preds, xgboost_preds=xgboost_preds,
                               pred_dates=prophet_dates)
    return render_template('checkPrediction.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, threaded=False)