import matplotlib
matplotlib.use('Agg')
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
    if os.path.exists(cache_file):
        cache_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        if datetime.now() - cache_time < timedelta(days=1):
            return joblib.load(cache_file)
    data = yf.download(symbol, start='2020-01-01', end='2025-12-31', auto_adjust=True)
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
        print(f"Prophet - Reset index took: {time.time() - start_time:.2f}s")
        
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
            return "Prophet Error: No 'Close' column found", []
        
        df = df[[date_col, close_col]].rename(columns={date_col: 'ds', close_col: 'y'})
        print(f"Prophet - Data prep took: {time.time() - start_time:.2f}s")
        
        model = Prophet()
        model.fit(df)
        print(f"Prophet - Fit took: {time.time() - start_time:.2f}s")
        
        future = model.make_future_dataframe(periods=5)
        forecast = model.predict(future)
        predictions = forecast[['ds', 'yhat']].tail(5)
        print(f"Prophet - Predict took: {time.time() - start_time:.2f}s")
        
        # Skip plotting for now
        print(f"Prophet - Total time: {time.time() - start_time:.2f}s")
        return "prophet_plot_placeholder", predictions['yhat'].tolist()
    except Exception as e:
        print(f"Prophet - Failed with detailed error: {str(e)} at {time.time() - start_time:.2f}s")
        return f"Prophet Error: {str(e)}", []

def xgboost_prediction(data):
    start_time = time.time()
    try:
        df = data.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] if col[1] == '' else f"{col[0]}_{col[1]}" for col in df.columns]
        print(f"XGBoost - Reset index took: {time.time() - start_time:.2f}s")
        
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
            print(f"XGBoost - Missing columns at: {time.time() - start_time:.2f}s")
            return "XGBoost Error: Missing required columns", []
        
        df['MA5'] = df[close_col].rolling(window=5).mean()
        df['MA10'] = df[close_col].rolling(window=10).mean()
        df['RSI'] = ta.momentum.RSIIndicator(df[close_col], window=14).rsi()
        df['MACD'] = ta.trend.MACD(df[close_col]).macd()
        df = df.dropna()
        print(f"XGBoost - Indicators added took: {time.time() - start_time:.2f}s")
        
        df['Target'] = df[close_col].shift(-1)
        df = df.dropna()
        print(f"XGBoost - Target prep took: {time.time() - start_time:.2f}s")
        
        X = df[[open_col, high_col, low_col, close_col, volume_col, 'MA5', 'MA10', 'RSI', 'MACD']]
        y = df['Target']
        
        train_size = int(len(df) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        print(f"XGBoost - Data split took: {time.time() - start_time:.2f}s")
        
        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
        model.fit(X_train, y_train)
        print(f"XGBoost - Model fit took: {time.time() - start_time:.2f}s")
        
        predictions = model.predict(X_test[-5:])
        print(f"XGBoost - Predict took: {time.time() - start_time:.2f}s")
        
        # Skip plotting for now
        print(f"XGBoost - Total time: {time.time() - start_time:.2f}s")
        return "xgboost_plot_placeholder", predictions.tolist()
    except Exception as e:
        print(f"XGBoost - Failed with detailed error: {str(e)} at {time.time() - start_time:.2f}s")
        return f"XGBoost Error: {str(e)}", []

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        stock_symbol = request.form['stock_symbol'].upper()
        data = get_stock_data(stock_symbol)
        if data is None:
            return "Error: Failed to fetch stock data. Check the symbol."

        data.index = pd.date_range(start=data.index[0], periods=len(data), freq='B')

        prophet_result, prophet_preds = prophet_prediction(data)
        xgboost_result, xgboost_preds = xgboost_prediction(data)

        if isinstance(prophet_result, str) and prophet_result.startswith("Prophet Error"):
            return f"Prophet failed: {prophet_result}"
        if isinstance(xgboost_result, str) and xgboost_result.startswith("XGBoost Error"):
            return f"XGBoost failed: {xgboost_result}"

        latest_data = data.iloc[-1]
        stock_info = {col: round(latest_data[col], 2) for col in ['Open', 'High', 'Low', 'Close', 'Volume']}

        return render_template('results.html', stock_symbol=stock_symbol, stock_info=stock_info,
                               prophet_plot=prophet_result, xgboost_plot=xgboost_result,
                               prophet_preds=prophet_preds, xgboost_preds=xgboost_preds)
    return render_template('checkPrediction.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, threaded=False)