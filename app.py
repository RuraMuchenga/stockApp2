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
import lightgbm as lgb  # Add LightGBM
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
        
        # Add volatility as a regressor
        df['volatility'] = df['y'].rolling(window=5).std()
        df = df.dropna()
        
        # Add cap and floor (2% above/below last price)
        last_price = df['y'].iloc[-1]
        df['cap'] = last_price * 1.02
        df['floor'] = last_price * 0.98
        
        # Fine-tune Prophet
        model = Prophet(
            changepoint_prior_scale=0.001,  # Reduce trend flexibility
            yearly_seasonality=False,       # Stocks don't have yearly seasonality
            weekly_seasonality=True,
            daily_seasonality=True,
            seasonality_prior_scale=1.0,    # Reduce seasonality impact
            growth='logistic'               # Use cap/floor
        )
        model.add_regressor('volatility')
        model.fit(df)
        
        future = model.make_future_dataframe(periods=5, freq='B')
        future['volatility'] = df['volatility'].iloc[-1]  # Use last known volatility
        future['cap'] = last_price * 1.02
        future['floor'] = last_price * 0.98
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
        
        # Add more features
        df['MA5'] = df[close_col].rolling(window=5).mean()
        df['MA10'] = df[close_col].rolling(window=10).mean()
        df['RSI'] = ta.momentum.RSIIndicator(df[close_col], window=14).rsi()
        df['MACD'] = ta.trend.MACD(df[close_col]).macd()
        df['Lag1'] = df[close_col].shift(1)
        df['Lag2'] = df[close_col].shift(2)
        df['Volatility'] = df[close_col].rolling(window=5).std()
        
        # Single dropna after all features
        df = df.dropna()
        if len(df) < 30:
            return f"XGBoost Error: Not enough data after dropping NaNs (only {len(df)} rows)", [], []
        
        df['Target'] = df[close_col].shift(-1)
        df = df.dropna()
        
        if len(df) < 30:
            return f"XGBoost Error: Not enough data after creating target (only {len(df)} rows)", [], []
        
        # Normalize features
        features = [open_col, high_col, low_col, close_col, volume_col, 'MA5', 'MA10', 'RSI', 'MACD', 'Lag1', 'Lag2', 'Volatility']
        for feature in features:
            df[feature] = (df[feature] - df[feature].mean()) / df[feature].std()
        
        X = df[features].values
        y = df['Target'].values
        
        # Adjust train-test split
        train_size = max(1, int(len(df) * 0.9))  # Use more data for training
        if len(df) - train_size < 5:
            train_size = len(df) - 5
        
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        if len(X_test) < 5:
            return f"XGBoost Error: Not enough test data (only {len(X_test)} rows)", [], []
        
        # Fine-tune XGBoost
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=200,
            learning_rate=0.01,
            max_depth=3,
            early_stopping_rounds=10
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        
        predictions = model.predict(X_test[-5:])
        
        # Denormalize predictions and y_test
        predictions = predictions * df[close_col].std() + df[close_col].mean()
        y_test = y_test * df[close_col].std() + df[close_col].mean()
        
        last_date = data.index[-1]
        past_dates = pd.date_range(end=last_date, periods=10, freq='B')
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=5, freq='B')
        all_dates = list(past_dates) + list(future_dates)
        date_labels = [d.strftime('%d/%m/%y') for d in all_dates]
        
        plt.figure(figsize=(10, 5))
        actual_last_10 = y_test[-10:] if len(y_test) >= 10 else y_test
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

def lightgbm_prediction(data):
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
            return "LightGBM Error: Missing required columns", [], []
        
        # Add more features (same as XGBoost for consistency)
        df['MA5'] = df[close_col].rolling(window=5).mean()
        df['MA10'] = df[close_col].rolling(window=10).mean()
        df['RSI'] = ta.momentum.RSIIndicator(df[close_col], window=14).rsi()
        df['MACD'] = ta.trend.MACD(df[close_col]).macd()
        df['Lag1'] = df[close_col].shift(1)
        df['Lag2'] = df[close_col].shift(2)
        df['Volatility'] = df[close_col].rolling(window=5).std()
        
        # Single dropna after all features
        df = df.dropna()
        if len(df) < 30:
            return f"LightGBM Error: Not enough data after dropping NaNs (only {len(df)} rows)", [], []
        
        df['Target'] = df[close_col].shift(-1)
        df = df.dropna()
        
        if len(df) < 30:
            return f"LightGBM Error: Not enough data after creating target (only {len(df)} rows)", [], []
        
        # Normalize features
        features = [open_col, high_col, low_col, close_col, volume_col, 'MA5', 'MA10', 'RSI', 'MACD', 'Lag1', 'Lag2', 'Volatility']
        for feature in features:
            df[feature] = (df[feature] - df[feature].mean()) / df[feature].std()
        
        X = df[features].values
        y = df['Target'].values
        
        # Adjust train-test split
        train_size = max(1, int(len(df) * 0.9))  # Use more data for training
        if len(df) - train_size < 5:
            train_size = len(df) - 5
        
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        if len(X_test) < 5:
            return f"LightGBM Error: Not enough test data (only {len(X_test)} rows)", [], []
        
        # Fine-tune LightGBM
        model = lgb.LGBMRegressor(
            objective='regression',
            n_estimators=200,
            learning_rate=0.01,
            max_depth=3,
            num_leaves=31,  # LightGBM-specific parameter
            early_stopping_rounds=10,
            verbose=-1
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='rmse')
        
        predictions = model.predict(X_test[-5:])
        
        # Denormalize predictions and y_test
        predictions = predictions * df[close_col].std() + df[close_col].mean()
        y_test = y_test * df[close_col].std() + df[close_col].mean()
        
        last_date = data.index[-1]
        past_dates = pd.date_range(end=last_date, periods=10, freq='B')
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=5, freq='B')
        all_dates = list(past_dates) + list(future_dates)
        date_labels = [d.strftime('%d/%m/%y') for d in all_dates]
        
        plt.figure(figsize=(10, 5))
        actual_last_10 = y_test[-10:] if len(y_test) >= 10 else y_test
        predicted_5 = predictions
        x_indices = range(15)
        plt.plot(x_indices[:len(actual_last_10)], actual_last_10, label='Actual', marker='o', color='#007bff')
        plt.plot(x_indices[10:15], predicted_5, label='Predicted', marker='o', color='#ff9900')
        plt.title(f'LightGBM: 5-Day Forecast', fontsize=14, pad=15)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price', fontsize=12)
        plt.xticks(x_indices, date_labels, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plot_data = plot_to_base64()
        
        print(f"LightGBM - Total time: {time.time() - start_time:.2f}s")
        return plot_data, predictions.tolist(), [d.strftime('%d/%m/%y') for d in future_dates]
    except Exception as e:
        print(f"LightGBM - Failed with error: {str(e)} at {time.time() - start_time:.2f}s")
        plt.figure(figsize=(10, 5))
        plt.text(0.5, 0.5, 'LightGBM Plot Failed', ha='center', va='center')
        plot_data = plot_to_base64()
        return plot_data, [], []

def combined_prediction_plot(data, prophet_preds, xgboost_preds, lightgbm_preds):
    try:
        df = data.reset_index()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] if col[1] == '' else f"{col[0]}_{col[1]}" for col in df.columns]
        
        close_col = None
        for col in df.columns:
            if 'close' in col.lower():
                close_col = col
                break
        if close_col is None:
            return "Combined Plot Error: No 'Close' column found"
        
        last_date = data.index[-1]
        past_dates = pd.date_range(end=last_date, periods=10, freq='B')
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=5, freq='B')
        all_dates = list(past_dates) + list(future_dates)
        date_labels = [d.strftime('%d/%m/%y') for d in all_dates]
        
        plt.figure(figsize=(10, 5))
        actual_last_10 = df[close_col].tail(10).values
        x_indices = range(15)
        plt.plot(x_indices[:10], actual_last_10, label='Actual', marker='o', color='#007bff')
        plt.plot(x_indices[10:], prophet_preds, label='Prophet Predicted', marker='o', color='#ff9900')
        plt.plot(x_indices[10:], xgboost_preds, label='XGBoost Predicted', marker='o', color='#28a745')
        plt.plot(x_indices[10:], lightgbm_preds, label='LightGBM Predicted', marker='o', color='#dc3545')
        plt.title('Combined: 5-Day Forecast', fontsize=14, pad=15)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price', fontsize=12)
        plt.xticks(x_indices, date_labels, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        return plot_to_base64()
    except Exception as e:
        print(f"Combined Plot - Failed with error: {str(e)}")
        plt.figure(figsize=(10, 5))
        plt.text(0.5, 0.5, 'Combined Plot Failed', ha='center', va='center')
        return plot_to_base64()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        stock_symbol = request.form['stock_symbol'].upper()
        data = get_stock_data(stock_symbol)
        if data is None:
            return "Error: Failed to fetch stock data. Check the symbol."

        prophet_result, prophet_preds, prophet_dates = prophet_prediction(data)
        xgboost_result, xgboost_preds, xgboost_dates = xgboost_prediction(data)
        lightgbm_result, lightgbm_preds, lightgbm_dates = lightgbm_prediction(data)
        combined_result = combined_prediction_plot(data, prophet_preds, xgboost_preds, lightgbm_preds)

        # Calculate combined predictions (average of Prophet, XGBoost, and LightGBM)
        combined_preds = [(p + x + l) / 3 for p, x, l in zip(prophet_preds, xgboost_preds, lightgbm_preds)]

        # Get the latest data and clean up stock_info
        latest_data = data.iloc[-1]
        stock_info = {col: float(latest_data[col]) for col in ['Open', 'High', 'Low', 'Close', 'Volume']}

        return render_template('results.html', stock_symbol=stock_symbol, stock_info=stock_info,
                               prophet_plot=prophet_result, xgboost_plot=xgboost_result,
                               lightgbm_plot=lightgbm_result, combined_plot=combined_result,
                               prophet_preds=prophet_preds, xgboost_preds=xgboost_preds,
                               lightgbm_preds=lightgbm_preds, combined_preds=combined_preds,
                               pred_dates=prophet_dates)
    return render_template('checkPrediction.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, threaded=False)