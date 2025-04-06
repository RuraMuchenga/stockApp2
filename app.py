import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory
import yfinance as yf
import pandas as pd
import numpy as np
import io
import base64
import os
from datetime import datetime, timedelta
import firebase_admin
from firebase_admin import credentials, auth as firebase_auth  # Alias to avoid confusion
from prophet import Prophet
import xgboost as xgb
import lightgbm as lgb
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
from bs4 import BeautifulSoup
import tweepy
from flask_mail import Mail, Message

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# Firebase setup
cred_path = os.getenv("FIREBASE_CRED_PATH")
if cred_path and os.path.exists(cred_path):
    cred = credentials.Certificate(cred_path)
else:
    firebase_key = os.getenv("FIREBASE_SERVICE_ACCOUNT_KEY")
    if firebase_key:
        cred = credentials.Certificate(json.loads(firebase_key))
    else:
        raise FileNotFoundError("Firebase credentials not found.")
firebase_admin.initialize_app(cred)

# Twitter API setup
TWITTER_API_KEY = 'GqcU2thHL681ECyMfbWsd5COn'
TWITTER_API_SECRET = '9CbIzwMZSaIKgjpwwaSDQ35VXDST4LpCYeLvIf7OAF052mwCd2'
TWITTER_ACCESS_TOKEN = '834439269607415809-KNVPB5cIigJDEXJjpUYchrOw5RrFVQ0'
TWITTER_ACCESS_SECRET = '4bLpdmgY9cwokuObhA7OrWO869IpzoNZeYjbeq7uPRmec'
twitter_auth = tweepy.OAuthHandler(TWITTER_API_KEY, TWITTER_API_SECRET)  # Separate name for clarity
twitter_auth.set_access_token(TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET)
twitter_api = tweepy.API(twitter_auth)

CACHE_DIR = 'cache'
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

def check_auth():
    id_token = session.get('id_token')
    if not id_token:
        return False
    try:
        firebase_auth.verify_id_token(id_token, check_revoked=True)
        return True
    except Exception as e:
        print(f"Check Auth Error: {str(e)}")
        session.pop('id_token', None)
        return False

def plot_to_base64():
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    plt.close()
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf8')

def get_stock_data(symbol):
    today = datetime.now().date()
    start_date = (today - timedelta(days=5*365)).strftime('%Y-%m-%d')
    end_date = today.strftime('%Y-%m-%d')
    data = yf.download(symbol, start=start_date, end=end_date, auto_adjust=True)
    if data.empty:
        return None
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]
    print(f"get_stock_data - Columns after fetch: {data.columns.tolist()}")
    return data

def create_features(df):
    df = df.copy()
    print(f"create_features - Columns: {df.columns.tolist()}")
    
    df['ret'] = df['Close'].pct_change()
    df['lag1'] = df['Close'].shift(1)
    df['lag5'] = df['Close'].shift(5)
    df['ma5'] = df['Close'].rolling(window=5).mean()
    df['ma20'] = df['Close'].rolling(window=20).mean()
    df['vol'] = df['ret'].rolling(window=20).std()
    df['dow'] = df.index.dayofweek
    df['mon'] = df.index.month
    df['trend_5d'] = df['Close'].pct_change(5)
    df['vol_spike'] = df['vol'] / df['vol'].shift(1)
    df['ma5_ma20_ratio'] = df['ma5'] / df['ma20']
    tariff_date = pd.to_datetime('2025-04-02')
    df['tariff_shock'] = (df.index >= tariff_date).astype(int) * -0.09
    return df.dropna()

def fetch_news_sentiment(symbol):
    try:
        # Skip Twitter (403'd)
        url = f"https://news.google.com/rss/search?q={symbol}+stock"
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'lxml-xml')  # Fixed parser
        articles = soup.find_all('item')[:10]
        news_texts = [article.title.text + " " + article.description.text for article in articles if article.description]
        all_texts = news_texts
        if not all_texts:
            return -0.5
        analyzer = SentimentIntensityAnalyzer()
        scores = [analyzer.polarity_scores(text)['compound'] for text in all_texts]
        avg_sentiment = np.mean(scores)
        print(f"Sentiment - {symbol}: {avg_sentiment} from {len(all_texts)} news sources")
        return avg_sentiment
    except Exception as e:
        print(f"Sentiment - Failed: {str(e)}")
        return -0.5

def news_bias(symbol):
    sentiment = fetch_news_sentiment(symbol)
    return sentiment  # -1 to 1, scaled later as you had it

def prophet_prediction(data, symbol):
    try:
        print(f"Prophet - Input data shape: {data.shape}")
        train_data = data.tail(50)[:-10]
        test_data = data.tail(10)
        
        df = pd.DataFrame({'ds': train_data.index, 'y': train_data['Close'].values})
        model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False,
                        growth='linear', changepoint_prior_scale=0.0005, seasonality_prior_scale=0.05)
        model.fit(df)
        
        past_future = pd.DataFrame({'ds': test_data.index})
        past_forecast = model.predict(past_future)
        past_preds = past_forecast['yhat'].values.tolist()
        
        last_date = data.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=5, freq='B')
        future = pd.DataFrame({'ds': future_dates})
        forecast = model.predict(future)
        future_preds = forecast['yhat'].values.tolist()
        
        bias = news_bias(symbol)
        prophet_preds = [p * (1 + bias * 0.1) * (1 - 0.10) for p in future_preds]

        last_10 = test_data['Close'].values.tolist()
        past_dates = test_data.index
        all_dates = list(past_dates) + list(future_dates)
        date_labels = [d.strftime('%d/%m/%y') for d in all_dates]

        print(f"Prophet - Past dates: {past_dates}")
        print(f"Prophet - Last 10 length: {len(last_10)}, Past preds length: {len(past_preds)}")
        
        plt.figure(figsize=(10, 5))
        x_indices = range(15)
        plt.plot(x_indices[:10], last_10, label='Actual Past 10', marker='o', color='#007bff')
        plt.plot(x_indices[:10], past_preds, label='Predicted Past 10', marker='x', color='#ff9900')
        plt.plot(x_indices[10:], prophet_preds, label='Predicted Next 5', marker='o', color='#ff9900')
        plt.title(f'Prophet: {symbol} Past 10 & Next 5 (News + Tariff Adjusted)', fontsize=14, pad=15)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price', fontsize=12)
        plt.xticks(x_indices, date_labels, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plot_data = plot_to_base64()

        print(f"Prophet - Last 10 actual: {last_10}")
        print(f"Prophet - Predicted Past 10: {past_preds}")
        print(f"Prophet - Predicted Next 5: {prophet_preds}")
        return plot_data, past_preds, prophet_preds, [d.strftime('%d/%m/%y') for d in past_dates], [d.strftime('%d/%m/%y') for d in future_dates]
    except Exception as e:
        print(f"Prophet - Failed with error: {str(e)}")
        plt.figure(figsize=(10, 5))
        plt.text(0.5, 0.5, 'Prophet Plot Failed', ha='center', va='center')
        plot_data = plot_to_base64()
        return plot_data, [0] * 10, [0] * 5, [], []

def xgboost_prediction(data, symbol):
    try:
        df = create_features(data.tail(100))
        train_df = df.iloc[:-10]
        test_df = df.tail(10)
        
        features = ['lag1', 'lag5', 'ma5', 'ma20', 'vol', 'dow', 'mon', 'trend_5d', 'vol_spike', 'ma5_ma20_ratio', 'tariff_shock']
        X = train_df[features]
        y = train_df['Close']
        model = xgb.XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=4, min_child_weight=0.1, reg_lambda=0.05, random_state=42)
        model.fit(X, y)
        
        X_test = test_df[features]
        past_preds = model.predict(X_test).tolist()

        last_df = df.tail(5).copy()
        xgboost_preds = []
        bias = news_bias(symbol)
        last_date = data.index[-1]
        for i in range(5):
            last_features = last_df[features].iloc[-1:].copy()
            pred = model.predict(last_features)[0] * (1 + bias * 0.1) * (1 - 0.10)
            xgboost_preds.append(float(pred))
            
            new_row = last_df.iloc[-1].copy()
            new_row['Close'] = pred
            new_row['lag1'] = pred
            new_row['lag5'] = last_df['Close'].iloc[-5] if i < 4 else xgboost_preds[i-4]
            new_row['ret'] = (pred - last_df['Close'].iloc[-1]) / last_df['Close'].iloc[-1]  # Fixed 'Close Hours' -> 'Close'
            new_row['ma5'] = (last_df['Close'].tail(4).sum() + pred) / 5
            new_row['ma20'] = (last_df['Close'].tail(19).sum() + pred) / 20
            new_row['vol'] = last_df['ret'].tail(19).std() if i == 0 else pd.concat([last_df['ret'].tail(19), pd.Series([new_row['ret']])]).std()
            new_row['dow'] = ((last_date + timedelta(days=i+1)).weekday()) % 7
            new_row['mon'] = (last_date + timedelta(days=i+1)).month
            new_row['trend_5d'] = (pred - last_df['Close'].iloc[-5]) / last_df['Close'].iloc[-5] if i < 4 else (pred - xgboost_preds[i-4]) / xgboost_preds[i-4]
            new_row['vol_spike'] = new_row['vol'] / last_df['vol'].iloc[-1] if last_df['vol'].iloc[-1] != 0 else 1.0
            new_row['ma5_ma20_ratio'] = new_row['ma5'] / new_row['ma20']
            new_row['tariff_shock'] = -0.09
            
            last_df = pd.concat([last_df, pd.DataFrame([new_row], index=[last_date + timedelta(days=i+1)])])

        last_10 = test_df['Close'].values.tolist()
        past_dates = test_df.index
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=5, freq='B')
        all_dates = list(past_dates) + list(future_dates)
        date_labels = [d.strftime('%d/%m/%y') for d in all_dates]

        print(f"XGBoost - Past dates: {past_dates}")
        print(f"XGBoost - Last 10 length: {len(last_10)}, Past preds length: {len(past_preds)}")
        
        plt.figure(figsize=(10, 5))
        x_indices = range(15)
        plt.plot(x_indices[:10], last_10, label='Actual Past 10', marker='o', color='#007bff')
        plt.plot(x_indices[:10], past_preds, label='Predicted Past 10', marker='x', color='#28a745')
        plt.plot(x_indices[10:], xgboost_preds, label='Predicted Next 5', marker='o', color='#28a745')
        plt.title(f'XGBoost: {symbol} Past 10 & Next 5 (News + Tariff Adjusted)', fontsize=14, pad=15)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price', fontsize=12)
        plt.xticks(x_indices, date_labels, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plot_data = plot_to_base64()

        print(f"XGBoost - Last 10 actual: {last_10}")
        print(f"XGBoost - Predicted Past 10: {past_preds}")
        print(f"XGBoost - Predicted Next 5: {xgboost_preds}")
        return plot_data, past_preds, xgboost_preds, [d.strftime('%d/%m/%y') for d in past_dates], [d.strftime('%d/%m/%y') for d in future_dates]
    except Exception as e:
        print(f"XGBoost - Failed with error: {str(e)}")
        plt.figure(figsize=(10, 5))
        plt.text(0.5, 0.5, 'XGBoost Plot Failed', ha='center', va='center')
        plot_data = plot_to_base64()
        return plot_data, [0] * 10, [0] * 5, [], []

def lightgbm_prediction(data, symbol):
    try:
        print(f"LightGBM - Input data shape: {data.shape}")
        df = create_features(data.tail(100))  # More data
        train_df = df.iloc[:-10]
        test_df = df.tail(10)
        
        features = ['lag1', 'lag5', 'ma5', 'ma20', 'vol', 'dow', 'mon', 'trend_5d', 'vol_spike', 'ma5_ma20_ratio', 'tariff_shock']
        X = train_df[features]
        y = train_df['Close']
        model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, max_depth=4, min_data_in_leaf=1, reg_lambda=0.05, random_state=42)  # Deeper
        model.fit(X, y)
        
        X_test = test_df[features]
        past_preds = model.predict(X_test).tolist()

        last_df = df.tail(5).copy()
        lightgbm_preds = []
        bias = news_bias(symbol)
        last_date = data.index[-1]
        for i in range(5):
            last_features = last_df[features].iloc[-1:].copy()
            pred = model.predict(last_features)[0] * (1 + bias * 0.1) * (1 - 0.10)
            lightgbm_preds.append(float(pred))
            
            new_row = last_df.iloc[-1].copy()
            new_row['Close'] = pred
            new_row['lag1'] = pred
            new_row['lag5'] = last_df['Close'].iloc[-5] if i < 4 else lightgbm_preds[i-4]
            new_row['ret'] = (pred - last_df['Close'].iloc[-1]) / last_df['Close'].iloc[-1]
            new_row['ma5'] = (last_df['Close'].tail(4).sum() + pred) / 5
            new_row['ma20'] = (last_df['Close'].tail(19).sum() + pred) / 20
            new_row['vol'] = last_df['ret'].tail(19).std() if i == 0 else pd.concat([last_df['ret'].tail(19), pd.Series([new_row['ret']])]).std()
            new_row['dow'] = ((last_date + timedelta(days=i+1)).weekday()) % 7
            new_row['mon'] = (last_date + timedelta(days=i+1)).month
            new_row['trend_5d'] = (pred - last_df['Close'].iloc[-5]) / last_df['Close'].iloc[-5] if i < 4 else (pred - lightgbm_preds[i-4]) / lightgbm_preds[i-4]
            new_row['vol_spike'] = new_row['vol'] / last_df['vol'].iloc[-1] if last_df['vol'].iloc[-1] != 0 else 1.0
            new_row['ma5_ma20_ratio'] = new_row['ma5'] / new_row['ma20']
            new_row['tariff_shock'] = -0.09
            
            last_df = pd.concat([last_df, pd.DataFrame([new_row], index=[last_date + timedelta(days=i+1)])])

        last_10 = test_df['Close'].values.tolist()
        past_dates = test_df.index
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=5, freq='B')
        all_dates = list(past_dates) + list(future_dates)
        date_labels = [d.strftime('%d/%m/%y') for d in all_dates]

        print(f"LightGBM - Past dates: {past_dates}")
        print(f"LightGBM - Last 10 length: {len(last_10)}, Past preds length: {len(past_preds)}")
        
        plt.figure(figsize=(10, 5))
        x_indices = range(15)
        plt.plot(x_indices[:10], last_10, label='Actual Past 10', marker='o', color='#007bff')
        plt.plot(x_indices[:10], past_preds, label='Predicted Past 10', marker='x', color='#dc3545')
        plt.plot(x_indices[10:], lightgbm_preds, label='Predicted Next 5', marker='o', color='#dc3545')
        plt.title(f'LightGBM: {symbol} Past 10 & Next 5 (News + Tariff Adjusted)', fontsize=14, pad=15)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price', fontsize=12)
        plt.xticks(x_indices, date_labels, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plot_data = plot_to_base64()

        print(f"LightGBM - Last 10 actual: {last_10}")
        print(f"LightGBM - Predicted Past 10: {past_preds}")
        print(f"LightGBM - Predicted Next 5: {lightgbm_preds}")
        return plot_data, past_preds, lightgbm_preds, [d.strftime('%d/%m/%y') for d in past_dates], [d.strftime('%d/%m/%y') for d in future_dates]
    except Exception as e:
        print(f"LightGBM - Failed with error: {str(e)}")
        plt.figure(figsize=(10, 5))
        plt.text(0.5, 0.5, 'LightGBM Plot Failed', ha='center', va='center')
        plot_data = plot_to_base64()
        return plot_data, [0] * 10, [0] * 5, [], []

def combined_prediction_plot(data, prophet_past, prophet_future, xgboost_past, xgboost_future, lightgbm_past, lightgbm_future, symbol):
    try:
        last_10 = data['Close'].tail(10).values
        last_date = data.index[-1]
        past_dates = data.index[-10:]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=5, freq='B')
        all_dates = list(past_dates) + list(future_dates)
        date_labels = [d.strftime('%d/%m/%y') for d in all_dates]

        plt.figure(figsize=(10, 5))
        x_indices = range(15)
        plt.plot(x_indices[:10], last_10, label='Actual Past 10', marker='o', color='#007bff')
        plt.plot(x_indices[:10], prophet_past, label='Prophet Past', marker='x', color='#ff9900')
        plt.plot(x_indices[:10], xgboost_past, label='XGBoost Past', marker='x', color='#28a745')
        plt.plot(x_indices[:10], lightgbm_past, label='LightGBM Past', marker='x', color='#dc3545')
        plt.plot(x_indices[10:], prophet_future, label='Prophet Future', marker='o', color='#ff9900')
        plt.plot(x_indices[10:], xgboost_future, label='XGBoost Future', marker='o', color='#28a745')
        plt.plot(x_indices[10:], lightgbm_future, label='LightGBM Future', marker='o', color='#dc3545')
        plt.title(f'Combined: {symbol} Past 10 & Next 5 (News + Tariff Adjusted)', fontsize=14, pad=15)
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

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/login', methods=['GET'])
def login_page():
    if check_auth():
        return redirect(url_for('index'))
    return render_template('login.html')

@app.route('/profile', methods=['GET'])
def profile_page():
    return render_template('checkProfile.html')

@app.route('/register', methods=['GET'])
def register_page():
    if check_auth():
        return redirect(url_for('index'))
    return render_template('register.html')

@app.route('/auth/login', methods=['POST'])
def login():
    id_token = request.form.get('id_token')
    try:
        firebase_auth.verify_id_token(id_token, check_revoked=True)  # Explicit Firebase auth
        session['id_token'] = id_token
        return redirect(url_for('index'))
    except Exception as e:
        print(f"Login Error: {str(e)}")
        return render_template('login.html', error="Login failed: " + str(e))

@app.route('/dashboard', methods=['GET', 'POST'])
def index():
    if not check_auth():
        return redirect(url_for('login_page'))
    
    if request.method == 'POST':
        stock_symbol = request.form['stock_symbol'].upper()
        data = get_stock_data(stock_symbol)
        if data is None:
            return render_template('checkPrediction.html', error="Error: Failed to fetch stock data.")

        prophet_plot, prophet_past, prophet_preds, prophet_past_dates, pred_dates = prophet_prediction(data, stock_symbol)
        xgboost_plot, xgboost_past, xgboost_preds, xgboost_past_dates, xgboost_future_dates = xgboost_prediction(data, stock_symbol)
        lightgbm_plot, lightgbm_past, lightgbm_preds, lightgbm_past_dates, lightgbm_future_dates = lightgbm_prediction(data, stock_symbol)
        combined_plot = combined_prediction_plot(data, prophet_past, prophet_preds, xgboost_past, xgboost_preds, lightgbm_past, lightgbm_preds, stock_symbol)

        combined_preds = [(p + x + l) / 3 for p, x, l in zip(prophet_preds, xgboost_preds, lightgbm_preds)]

        latest_data = data.iloc[-1]
        print(f"latest_data columns: {latest_data.index.tolist()}")
        stock_info = {
            'Open': float(latest_data['Open']),
            'High': float(latest_data['High']),
            'Low': float(latest_data['Low']),
            'Close': float(latest_data['Close']),
            'Volume': float(latest_data['Volume'])
        }

        return render_template('results.html', stock_symbol=stock_symbol, stock_info=stock_info,
                              prophet_plot=prophet_plot, xgboost_plot=xgboost_plot,
                              lightgbm_plot=lightgbm_plot, combined_plot=combined_plot,
                              prophet_past=prophet_past, prophet_preds=prophet_preds,
                              xgboost_past=xgboost_past, xgboost_preds=xgboost_preds,
                              lightgbm_past=lightgbm_past, lightgbm_preds=lightgbm_preds,
                              past_dates=prophet_past_dates, pred_dates=pred_dates,
                              combined_preds=combined_preds)
    return render_template('checkPrediction.html')

@app.route('/logout', methods=['GET'])
def logout():
    session.pop('id_token', None)
    return redirect(url_for('home'))

app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'ruramaimuchenga@gmail.com'
app.config['MAIL_PASSWORD'] = 'your-app-password'
mail = Mail(app)

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        subject = request.form['subject']
        message = request.form['message']
        
        msg = Message(subject, sender=email, recipients=['ruramaimuchenga@gmail.com'])
        msg.body = f"From: {name} ({email})\n\n{message}"
        mail.send(msg)
        return redirect(url_for('contact'))
    return render_template('contact.html')

@app.route('/test_static/<path:filename>')
def test_static(filename):
    return send_from_directory('static', filename)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, threaded=False, debug=True)