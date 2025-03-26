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
from keras.layers import Dense, LSTM, Input
import xgboost as xgb
import os
import tensorflow as tf
from keras import backend as K

# Force TensorFlow to run on CPU only
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Control TensorFlow memory growth
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

app = Flask(__name__)

# Helper function to convert plots to base64
def plot_to_base64():
    img = io.BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf8')

# ARIMA Model
#predicts the next 5 days of stock prices, plots the results alongside the last 10 days of actual data, and returns the plot + predictions as a list.
def arima_prediction(data):
    try:
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



#lstm
def lstm_prediction(data):
    try:
        # Scale data
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
        
        # Prepare training data
        X, y = [], []
        for i in range(10, len(data_scaled)):
            X.append(data_scaled[i-10:i, 0])
            y.append(data_scaled[i, 0])

        X, y = np.array(X), np.array(y)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        # Define the LSTM model
        model = Sequential([
            Input(shape=(X.shape[1], 1)),
            LSTM(units=50, return_sequences=True),
            LSTM(units=50),
            Dense(1)
        ])

        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=5, batch_size=64, verbose=0)

        # Save the model
        model.save('lstm_model.h5')
        print("✅ LSTM model saved as 'lstm_model.h5'!")

        # Make predictions
        predictions = model.predict(X[-5:])
        predictions = scaler.inverse_transform(predictions)

        # Plot results
        plt.figure()
        plt.plot(data.index[-len(predictions):], data['Close'].iloc[-len(predictions):], label='Real Price')
        plt.plot(data.index[-5:], predictions, label='LSTM Predicted Price', color='orange')
        plt.title('LSTM Stock Price Prediction')
        plt.legend()

        K.clear_session()  # Clear TensorFlow session to prevent memory buildup
        return plot_to_base64(), predictions.flatten().tolist()

    except Exception as e:
        return f"LSTM Error: {e}", []
    



def convert_model_to_tflite():
    try:
        # Load the trained Keras model
        model = tf.keras.models.load_model('lstm_model.h5')

        # Convert the model to TensorFlow Lite format
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()

        # Save the converted model
        with open('lstm_model.tflite', 'wb') as f:
            f.write(tflite_model)

        print("✅ Model successfully converted to TensorFlow Lite!")
    except Exception as e:
        print(f"Error converting model: {e}")


def lstm_tflite_prediction(data):
    try:
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

        # Prepare test data
        X = np.array([data_scaled[-10:].flatten()])
        X = X.reshape(X.shape[0], X.shape[1], 1)

        # Load TensorFlow Lite model
        interpreter = tf.lite.Interpreter(model_path="lstm_model.tflite")
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Feed input data to the model
        interpreter.set_tensor(input_details[0]['index'], X.astype(np.float32))

        # Run prediction
        interpreter.invoke()

        # Get output
        predictions = interpreter.get_tensor(output_details[0]['index'])
        predictions = scaler.inverse_transform(predictions)

        # Plot results
        plt.figure()
        plt.plot(data.index[-len(predictions):], data['Close'].iloc[-len(predictions):], label='Real Price')
        plt.plot(data.index[-5:], predictions, label='TFLite LSTM Predicted Price', color='orange')
        plt.title('TensorFlow Lite LSTM Stock Price Prediction')
        plt.legend()

        return plot_to_base64(), predictions.flatten().tolist()

    except Exception as e:
        return f"LSTM Lite Error: {e}", []



# XGBoost Model
#Creates a Target column that shifts the closing price one day forward (to predict the next day's price).
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
            data = yf.download(stock_symbol, start='2020-01-01', end='2025-12-31', auto_adjust=True)
            if data.empty or 'Close' not in data.columns:
                return "Error: Failed to fetch stock data. Check the symbol."

            # Ensure DataFrame has proper datetime index with frequency
            data.index = pd.date_range(start=data.index[0], periods=len(data), freq='B')

            arima_plot, arima_preds = arima_prediction(data)
            lstm_plot, lstm_preds = lstm_prediction(data)
            xgboost_plot, xgboost_preds = xgboost_prediction(data)

            latest_data = data.iloc[-1]
            stock_info = {col: round(latest_data[col], 2) for col in ['Open', 'High', 'Low', 'Close', 'Volume']}

            return render_template('results.html', stock_symbol=stock_symbol, stock_info=stock_info,
                                   arima_plot=arima_plot, lstm_plot=lstm_plot, xgboost_plot=xgboost_plot,
                                   arima_preds=arima_preds, lstm_preds=lstm_preds, xgboost_preds=xgboost_preds)
        except Exception as e:
            return f"Error: {e}"
    return render_template('checkPrediction.html')

if __name__ == "__main__":
    # Increase Gunicorn timeout
    app.run(host='0.0.0.0', port=8000, threaded=True)
