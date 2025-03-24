import os
from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import xgboost as xgb

# Force TensorFlow to run on CPU only
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

app = Flask(__name__)

# Helper function to convert plots to base64
def plot_to_base64():
    img = io.BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf8')

# ARIMA Model
def arima_prediction(data):
    try:
        data.index.freq = data.index.inferred_freq
        model = ARIMA(data['Close'], order=(2, 1, 0))
        model_fit = model.fit()
        predictions = model_fit.forecast(steps=5)
        plt.figure()
        plt.plot(data.index[-10:], data['Close'][-10:], label='Actual Price')
        plt.plot(data.index[-5:], predictions, label='ARIMA Predicted Price', color='orange')
        plt.title('ARIMA Stock Price Prediction')
        plt.legend()
        return plot_to_base64(), predictions.tolist()
    except Exception as e:
        return f"ARIMA Error: {e}", []

# LSTM Model
def lstm_prediction(data):
    try:
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
        X, y = [], []
        for i in range(10, len(data_scaled)):
            X.append(data_scaled[i-10:i, 0])
            y.append(data_scaled[i, 0])
        X, y = np.array(X), np.array(y)
        X = X.reshape(X.shape[0], X.shape[1], 1)
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
        model.add(LSTM(units=50))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=10, batch_size=32, verbose=0)
        predictions = model.predict(X[-5:])
        predictions = scaler.inverse_transform(predictions)
        plt.figure()
        plt.plot(data.index[-len(predictions):], data['Close'].iloc[-len(predictions):], label='Real Price')
        plt.plot(data.index[-5:], predictions, label='LSTM Predicted Price', color='orange')
        plt.title('LSTM Stock Price Prediction')
        plt.legend()
        return plot_to_base64(), predictions.flatten().tolist()
    except Exception as e:
        return f"LSTM Error: {e}", []

# XGBoost Model
def xgboost_prediction(data):
    try:
        data['Target'] = data['Close'].shift(-1)
        data = data.dropna()
        X = data[['Open', 'High', 'Low', 'Close', 'Volume']]
        y = data['Target']
        train_size = int(len(data) * 0.8)
        X_train, X_test, y_train, y_test = X[:train_size], X[train_size:], y[:train_size], y[train_size:]
        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test[-5:])
        plt.figure()
        plt.plot(data.index[-len(y_test):], y_test, label='Real Price')
        plt.plot(data.index[-5:], predictions, label='XGBoost Predicted Price', color='orange')
        plt.title('XGBoost Stock Price Prediction')
        plt.legend()
        return plot_to_base64(), predictions.tolist()
    except Exception as e:
        return f"XGBoost Error: {e}", []

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        stock_symbol = request.form['stock_symbol'].upper()
        try:
            data = yf.download(stock_symbol, start='2020-01-01', end='2025-12-31', auto_adjust=False)
            if data.empty or 'Close' not in data.columns:
                return "Error: Failed to fetch stock data. Check the symbol."
            
            arima_plot, arima_preds = arima_prediction(data)
            lstm_plot, lstm_preds = lstm_prediction(data)
            xgboost_plot, xgboost_preds = xgboost_prediction(data)

            latest_data = data.iloc[-1]
            stock_info = {
                'Open': latest_data['Open'],
                'High': latest_data['High'],
                'Low': latest_data['Low'],
                'Close': latest_data['Close'],
                'Volume': latest_data['Volume']
            }

            return render_template('results.html', stock_symbol=stock_symbol, stock_info=stock_info,
                                   arima_plot=arima_plot, lstm_plot=lstm_plot, xgboost_plot=xgboost_plot,
                                   arima_preds=arima_preds, lstm_preds=lstm_preds, xgboost_preds=xgboost_preds)
        except Exception as e:
            return f"Error: {e}"
    return render_template('checkPrediction.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)

# Gunicorn command suggestion (extend timeout)
# gunicorn app:app --timeout 120
