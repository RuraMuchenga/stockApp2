

# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import json
# from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory, flash
# from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
# import yfinance as yf
# import pandas as pd
# import numpy as np
# import io
# import base64
# import os
# from datetime import datetime, timedelta
# import firebase_admin
# from firebase_admin import credentials, auth as firebase_auth, firestore
# from prophet import Prophet
# import xgboost as xgb
# import lightgbm as lgb
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# import requests
# from bs4 import BeautifulSoup
# import tweepy
# from flask_mail import Mail, Message

# app = Flask(__name__)
# app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-here')

# # Firebase setup
# cred_path = os.getenv("FIREBASE_CRED_PATH")
# if cred_path and os.path.exists(cred_path):
#     cred = credentials.Certificate(cred_path)
# else:
#     firebase_key = os.getenv("FIREBASE_SERVICE_ACCOUNT_KEY")
#     if firebase_key:
#         cred = credentials.Certificate(json.loads(firebase_key))
#     else:
#         raise FileNotFoundError("Firebase credentials not found.")
# firebase_admin.initialize_app(cred)
# db = firestore.client()

# # Flask-Login setup
# login_manager = LoginManager()
# login_manager.init_app(app)
# login_manager.login_view = 'login_page'

# # Flask-Mail setup
# app.config['MAIL_SERVER'] = 'smtp.gmail.com'
# app.config['MAIL_PORT'] = 587
# app.config['MAIL_USE_TLS'] = True
# app.config['MAIL_USERNAME'] = 'ruramaimuchenga@gmail.com'
# app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD', 'your-app-password')
# mail = Mail(app)

# # User class for Flask-Login
# class User(UserMixin):
#     def __init__(self, uid, email, username=None, mobileNo=None):
#         self.id = uid
#         self.email = email
#         self.username = username
#         self.mobileNo = mobileNo

# @login_manager.user_loader
# def load_user(user_id):
#     try:
#         id_token = session.get('id_token')
#         if not id_token:
#             return None
#         decoded_token = firebase_auth.verify_id_token(id_token, check_revoked=True)
#         if decoded_token['uid'] == user_id:
#             user_ref = db.collection('users').document(user_id)
#             user_data = user_ref.get().to_dict() or {}
#             return User(
#                 user_id,
#                 decoded_token.get('email', 'unknown'),
#                 user_data.get('username', 'Unknown'),
#                 user_data.get('mobileNo', '')
#             )
#         return None
#     except Exception as e:
#         print(f"User Loader Error: {str(e)}")
#         return None

# # Twitter API setup
# TWITTER_API_KEY = 'GqcU2thHL681ECyMfbWsd5COn'
# TWITTER_API_SECRET = '9CbIzwMZSaIKgjpwwaSDQ35VXDST4LpCYeLvIf7OAF052mwCd2'
# TWITTER_ACCESS_TOKEN = '834439269607415809-KNVPB5cIigJDEXJjpUYchrOw5RrFVQ0'
# TWITTER_ACCESS_SECRET = '4bLpdmgY9cwokuObhA7OrWO869IpzoNZeYjbeq7uPRmec'
# twitter_auth = tweepy.OAuthHandler(TWITTER_API_KEY, TWITTER_API_SECRET)
# twitter_auth.set_access_token(TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET)
# twitter_api = tweepy.API(twitter_auth)

# CACHE_DIR = 'cache'
# if not os.path.exists(CACHE_DIR):
#     os.makedirs(CACHE_DIR)

# def plot_to_base64():
#     img = io.BytesIO()
#     plt.savefig(img, format='png', bbox_inches='tight')
#     plt.close()
#     img.seek(0)
#     return base64.b64encode(img.getvalue()).decode('utf8')

# def get_stock_data(symbol):
#     today = datetime.now().date()
#     start_date = (today - timedelta(days=5*365)).strftime('%Y-%m-%d')
#     end_date = today.strftime('%Y-%m-%d')
#     data = yf.download(symbol, start=start_date, end=end_date, auto_adjust=True)
#     if data.empty:
#         return None
#     if isinstance(data.columns, pd.MultiIndex):
#         data.columns = [col[0] for col in data.columns]
#     print(f"get_stock_data - Columns after fetch: {data.columns.tolist()}")
#     return data

# def create_features(df):
#     df = df.copy()
#     print(f"create_features - Columns: {df.columns.tolist()}")
#     df['ret'] = df['Close'].pct_change()
#     df['lag1'] = df['Close'].shift(1)
#     df['lag5'] = df['Close'].shift(5)
#     df['ma5'] = df['Close'].rolling(window=5).mean()
#     df['ma20'] = df['Close'].rolling(window=20).mean()
#     df['vol'] = df['ret'].rolling(window=20).std()
#     df['dow'] = df.index.dayofweek
#     df['mon'] = df.index.month
#     df['trend_5d'] = df['Close'].pct_change(5)
#     df['vol_spike'] = df['vol'] / df['vol'].shift(1)
#     df['ma5_ma20_ratio'] = df['ma5'] / df['ma20']
#     tariff_date = pd.to_datetime('2025-04-02')
#     df['tariff_shock'] = (df.index >= tariff_date).astype(int) * -0.09
#     return df.dropna()

# def fetch_news_sentiment(symbol):
#     try:
#         url = f"https://news.google.com/rss/search?q={symbol}+stock"
#         response = requests.get(url)
#         soup = BeautifulSoup(response.content, 'lxml-xml')
#         articles = soup.find_all('item')[:10]
#         news_texts = [article.title.text + " " + article.description.text for article in articles if article.description]
#         all_texts = news_texts
#         if not all_texts:
#             return -0.5
#         analyzer = SentimentIntensityAnalyzer()
#         scores = [analyzer.polarity_scores(text)['compound'] for text in all_texts]
#         avg_sentiment = np.mean(scores)
#         print(f"Sentiment - {symbol}: {avg_sentiment} from {len(all_texts)} news sources")
#         return avg_sentiment
#     except Exception as e:
#         print(f"Sentiment - Failed: {str(e)}")
#         return -0.5

# def news_bias(symbol):
#     sentiment = fetch_news_sentiment(symbol)
#     return sentiment

# def prophet_prediction(data, symbol):
#     try:
#         print(f"Prophet - Input data shape: {data.shape}")
#         train_data = data.tail(50)[:-10]
#         test_data = data.tail(10)
        
#         df = pd.DataFrame({'ds': train_data.index, 'y': train_data['Close'].values})
#         model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False,
#                         growth='linear', changepoint_prior_scale=0.0005, seasonality_prior_scale=0.05)
#         model.fit(df)
        
#         past_future = pd.DataFrame({'ds': test_data.index})
#         past_forecast = model.predict(past_future)
#         past_preds = past_forecast['yhat'].values.tolist()
        
#         last_date = data.index[-1]
#         future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=5, freq='B')
#         future = pd.DataFrame({'ds': future_dates})
#         forecast = model.predict(future)
#         future_preds = forecast['yhat'].values.tolist()
        
#         bias = news_bias(symbol)
#         prophet_preds = [p * (1 + bias * 0.1) * (1 - 0.10) for p in future_preds]

#         last_10 = test_data['Close'].values.tolist()
#         past_dates = test_data.index
#         all_dates = list(past_dates) + list(future_dates)
#         date_labels = [d.strftime('%d/%m/%y') for d in all_dates]

#         print(f"Prophet - Past dates: {past_dates}")
#         print(f"Prophet - Last 10 length: {len(last_10)}, Past preds length: {len(past_preds)}")
        
#         plt.figure(figsize=(10, 5))
#         x_indices = range(15)
#         plt.plot(x_indices[:10], last_10, label='Actual Past 10', marker='o', color='#007bff')
#         plt.plot(x_indices[:10], past_preds, label='Predicted Past 10', marker='x', color='#ff9900')
#         plt.plot(x_indices[10:], prophet_preds, label='Predicted Next 5', marker='o', color='#ff9900')
#         plt.title(f'Prophet: {symbol} Past 10 & Next 5 (News + Tariff Adjusted)', fontsize=14, pad=15)
#         plt.xlabel('Date', fontsize=12)
#         plt.ylabel('Price', fontsize=12)
#         plt.xticks(x_indices, date_labels, rotation=45, ha='right')
#         plt.legend()
#         plt.grid(True, linestyle='--', alpha=0.7)
#         plt.tight_layout()
#         plot_data = plot_to_base64()

#         print(f"Prophet - Last 10 actual: {last_10}")
#         print(f"Prophet - Predicted Past 10: {past_preds}")
#         print(f"Prophet - Predicted Next 5: {prophet_preds}")
#         return plot_data, past_preds, prophet_preds, [d.strftime('%d/%m/%y') for d in past_dates], [d.strftime('%d/%m/%y') for d in future_dates]
#     except Exception as e:
#         print(f"Prophet - Failed with error: {str(e)}")
#         plt.figure(figsize=(10, 5))
#         plt.text(0.5, 0.5, 'Prophet Plot Failed', ha='center', va='center')
#         plot_data = plot_to_base64()
#         return plot_data, [0] * 10, [0] * 5, [], []

# def xgboost_prediction(data, symbol):
#     try:
#         df = create_features(data.tail(100))
#         train_df = df.iloc[:-10]
#         test_df = df.tail(10)
        
#         features = ['lag1', 'lag5', 'ma5', 'ma20', 'vol', 'dow', 'mon', 'trend_5d', 'vol_spike', 'ma5_ma20_ratio', 'tariff_shock']
#         X = train_df[features]
#         y = train_df['Close']
#         model = xgb.XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=4, min_child_weight=0.1, reg_lambda=0.05, random_state=42)
#         model.fit(X, y)
        
#         X_test = test_df[features]
#         past_preds = model.predict(X_test).tolist()

#         last_df = df.tail(5).copy()
#         xgboost_preds = []
#         bias = news_bias(symbol)
#         last_date = data.index[-1]
#         for i in range(5):
#             last_features = last_df[features].iloc[-1:].copy()
#             pred = model.predict(last_features)[0] * (1 + bias * 0.1) * (1 - 0.10)
#             xgboost_preds.append(float(pred))
            
#             new_row = last_df.iloc[-1].copy()
#             new_row['Close'] = pred
#             new_row['lag1'] = pred
#             new_row['lag5'] = last_df['Close'].iloc[-5] if i < 4 else xgboost_preds[i-4]
#             new_row['ret'] = (pred - last_df['Close'].iloc[-1]) / last_df['Close'].iloc[-1]
#             new_row['ma5'] = (last_df['Close'].tail(4).sum() + pred) / 5
#             new_row['ma20'] = (last_df['Close'].tail(19).sum() + pred) / 20
#             new_row['vol'] = last_df['ret'].tail(19).std() if i == 0 else pd.concat([last_df['ret'].tail(19), pd.Series([new_row['ret']])]).std()
#             new_row['dow'] = ((last_date + timedelta(days=i+1)).weekday()) % 7
#             new_row['mon'] = (last_date + timedelta(days=i+1)).month
#             new_row['trend_5d'] = (pred - last_df['Close'].iloc[-5]) / last_df['Close'].iloc[-5] if i < 4 else (pred - xgboost_preds[i-4]) / xgboost_preds[i-4]
#             new_row['vol_spike'] = new_row['vol'] / last_df['vol'].iloc[-1] if last_df['vol'].iloc[-1] != 0 else 1.0
#             new_row['ma5_ma20_ratio'] = new_row['ma5'] / new_row['ma20']
#             new_row['tariff_shock'] = -0.09
            
#             last_df = pd.concat([last_df, pd.DataFrame([new_row], index=[last_date + timedelta(days=i+1)])])

#         last_10 = test_df['Close'].values.tolist()
#         past_dates = test_df.index
#         future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=5, freq='B')
#         all_dates = list(past_dates) + list(future_dates)
#         date_labels = [d.strftime('%d/%m/%y') for d in all_dates]

#         print(f"XGBoost - Past dates: {past_dates}")
#         print(f"XGBoost - Last 10 length: {len(last_10)}, Past preds length: {len(past_preds)}")
        
#         plt.figure(figsize=(10, 5))
#         x_indices = range(15)
#         plt.plot(x_indices[:10], last_10, label='Actual Past 10', marker='o', color='#007bff')
#         plt.plot(x_indices[:10], past_preds, label='Predicted Past 10', marker='x', color='#28a745')
#         plt.plot(x_indices[10:], xgboost_preds, label='Predicted Next 5', marker='o', color='#28a745')
#         plt.title(f'XGBoost: {symbol} Past 10 & Next 5 (News + Tariff Adjusted)', fontsize=14, pad=15)
#         plt.xlabel('Date', fontsize=12)
#         plt.ylabel('Price', fontsize=12)
#         plt.xticks(x_indices, date_labels, rotation=45, ha='right')
#         plt.legend()
#         plt.grid(True, linestyle='--', alpha=0.7)
#         plt.tight_layout()
#         plot_data = plot_to_base64()

#         print(f"XGBoost - Last 10 actual: {last_10}")
#         print(f"XGBoost - Predicted Past 10: {past_preds}")
#         print(f"XGBoost - Predicted Next 5: {xgboost_preds}")
#         return plot_data, past_preds, xgboost_preds, [d.strftime('%d/%m/%y') for d in past_dates], [d.strftime('%d/%m/%y') for d in future_dates]
#     except Exception as e:
#         print(f"XGBoost - Failed with error: {str(e)}")
#         plt.figure(figsize=(10, 5))
#         plt.text(0.5, 0.5, 'XGBoost Plot Failed', ha='center', va='center')
#         plot_data = plot_to_base64()
#         return plot_data, [0] * 10, [0] * 5, [], []

# def lightgbm_prediction(data, symbol):
#     try:
#         print(f"LightGBM - Input data shape: {data.shape}")
#         df = create_features(data.tail(100))
#         train_df = df.iloc[:-10]
#         test_df = df.tail(10)
        
#         features = ['lag1', 'lag5', 'ma5', 'ma20', 'vol', 'dow', 'mon', 'trend_5d', 'vol_spike', 'ma5_ma20_ratio', 'tariff_shock']
#         X = train_df[features]
#         y = train_df['Close']
#         model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, max_depth=4, min_data_in_leaf=1, reg_lambda=0.05, random_state=42)
#         model.fit(X, y)
        
#         X_test = test_df[features]
#         past_preds = model.predict(X_test).tolist()

#         last_df = df.tail(5).copy()
#         lightgbm_preds = []
#         bias = news_bias(symbol)
#         last_date = data.index[-1]
#         for i in range(5):
#             last_features = last_df[features].iloc[-1:].copy()
#             pred = model.predict(last_features)[0] * (1 + bias * 0.1) * (1 - 0.10)
#             lightgbm_preds.append(float(pred))
            
#             new_row = last_df.iloc[-1].copy()
#             new_row['Close'] = pred
#             new_row['lag1'] = pred
#             new_row['lag5'] = last_df['Close'].iloc[-5] if i < 4 else lightgbm_preds[i-4]
#             new_row['ret'] = (pred - last_df['Close'].iloc[-1]) / last_df['Close'].iloc[-1]
#             new_row['ma5'] = (last_df['Close'].tail(4).sum() + pred) / 5
#             new_row['ma20'] = (last_df['Close'].tail(19).sum() + pred) / 20
#             new_row['vol'] = last_df['ret'].tail(19).std() if i == 0 else pd.concat([last_df['ret'].tail(19), pd.Series([new_row['ret']])]).std()
#             new_row['dow'] = ((last_date + timedelta(days=i+1)).weekday()) % 7
#             new_row['mon'] = (last_date + timedelta(days=i+1)).month
#             new_row['trend_5d'] = (pred - last_df['Close'].iloc[-5]) / last_df['Close'].iloc[-5] if i < 4 else (pred - lightgbm_preds[i-4]) / lightgbm_preds[i-4]
#             new_row['vol_spike'] = new_row['vol'] / last_df['vol'].iloc[-1] if last_df['vol'].iloc[-1] != 0 else 1.0
#             new_row['ma5_ma20_ratio'] = new_row['ma5'] / new_row['ma20']
#             new_row['tariff_shock'] = -0.09
            
#             last_df = pd.concat([last_df, pd.DataFrame([new_row], index=[last_date + timedelta(days=i+1)])])

#         last_10 = test_df['Close'].values.tolist()
#         past_dates = test_df.index
#         future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=5, freq='B')
#         all_dates = list(past_dates) + list(future_dates)
#         date_labels = [d.strftime('%d/%m/%y') for d in all_dates]

#         print(f"LightGBM - Past dates: {past_dates}")
#         print(f"LightGBM - Last 10 length: {len(last_10)}, Past preds length: {len(past_preds)}")
        
#         plt.figure(figsize=(10, 5))
#         x_indices = range(15)
#         plt.plot(x_indices[:10], last_10, label='Actual Past 10', marker='o', color='#007bff')
#         plt.plot(x_indices[:10], past_preds, label='Predicted Past 10', marker='x', color='#dc3545')
#         plt.plot(x_indices[10:], lightgbm_preds, label='Predicted Next 5', marker='o', color='#dc3545')
#         plt.title(f'LightGBM: {symbol} Past 10 & Next 5 (News + Tariff Adjusted)', fontsize=14, pad=15)
#         plt.xlabel('Date', fontsize=12)
#         plt.ylabel('Price', fontsize=12)
#         plt.xticks(x_indices, date_labels, rotation=45, ha='right')
#         plt.legend()
#         plt.grid(True, linestyle='--', alpha=0.7)
#         plt.tight_layout()
#         plot_data = plot_to_base64()

#         print(f"LightGBM - Last 10 actual: {last_10}")
#         print(f"LightGBM - Predicted Past 10: {past_preds}")
#         print(f"LightGBM - Predicted Next 5: {lightgbm_preds}")
#         return plot_data, past_preds, lightgbm_preds, [d.strftime('%d/%m/%y') for d in past_dates], [d.strftime('%d/%m/%y') for d in future_dates]
#     except Exception as e:
#         print(f"LightGBM - Failed with error: {str(e)}")
#         plt.figure(figsize=(10, 5))
#         plt.text(0.5, 0.5, 'LightGBM Plot Failed', ha='center', va='center')
#         plot_data = plot_to_base64()
#         return plot_data, [0] * 10, [0] * 5, [], []

# def combined_prediction_plot(data, prophet_past, prophet_future, xgboost_past, xgboost_future, lightgbm_past, lightgbm_future, symbol):
#     try:
#         last_10 = data['Close'].tail(10).values
#         last_date = data.index[-1]
#         past_dates = data.index[-10:]
#         future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=5, freq='B')
#         all_dates = list(past_dates) + list(future_dates)
#         date_labels = [d.strftime('%d/%m/%y') for d in all_dates]

#         plt.figure(figsize=(10, 5))
#         x_indices = range(15)
#         plt.plot(x_indices[:10], last_10, label='Actual Past 10', marker='o', color='#007bff')
#         plt.plot(x_indices[:10], prophet_past, label='Prophet Past', marker='x', color='#ff9900')
#         plt.plot(x_indices[:10], xgboost_past, label='XGBoost Past', marker='x', color='#28a745')
#         plt.plot(x_indices[:10], lightgbm_past, label='LightGBM Past', marker='x', color='#dc3545')
#         plt.plot(x_indices[10:], prophet_future, label='Prophet Future', marker='o', color='#ff9900')
#         plt.plot(x_indices[10:], xgboost_future, label='XGBoost Future', marker='o', color='#28a745')
#         plt.plot(x_indices[10:], lightgbm_future, label='LightGBM Future', marker='o', color='#dc3545')
#         plt.title(f'Combined: {symbol} Past 10 & Next 5 (News + Tariff Adjusted)', fontsize=14, pad=15)
#         plt.xlabel('Date', fontsize=12)
#         plt.ylabel('Price', fontsize=12)
#         plt.xticks(x_indices, date_labels, rotation=45, ha='right')
#         plt.legend()
#         plt.grid(True, linestyle='--', alpha=0.7)
#         plt.tight_layout()
#         return plot_to_base64()
#     except Exception as e:
#         print(f"Combined Plot - Failed with error: {str(e)}")
#         plt.figure(figsize=(10, 5))
#         plt.text(0.5, 0.5, 'Combined Plot Failed', ha='center', va='center')
#         return plot_to_base64()

# # Routes
# @app.route('/', methods=['GET'])
# def home():
#     return render_template('index.html')

# @app.route('/login', methods=['GET'])
# def login_page():
#     if current_user.is_authenticated:
#         return redirect(url_for('index'))
#     return render_template('login.html')

# @app.route('/auth/login', methods=['POST'])
# def login():
#     id_token = request.form.get('id_token')
#     try:
#         decoded_token = firebase_auth.verify_id_token(id_token, check_revoked=True)
#         user = User(
#             decoded_token['uid'],
#             decoded_token.get('email', 'unknown')
#         )
#         login_user(user)
#         session['id_token'] = id_token
#         return redirect(url_for('index'))
#     except Exception as e:
#         print(f"Login Error: {str(e)}")
#         return render_template('login.html', error="Login failed: " + str(e))

# @app.route('/register', methods=['GET'])
# def register_page():
#     if current_user.is_authenticated:
#         return redirect(url_for('index'))
#     return render_template('register.html')

# @app.route('/profile', methods=['GET', 'POST'])
# @login_required
# def profile():
#     user_ref = db.collection('users').document(current_user.id)
#     user_data = user_ref.get().to_dict() or {}

#     if request.method == 'POST':
#         action = request.form.get('action')
        
#         if action == 'update':
#             # Handle profile updates
#             new_password = request.form.get('new_password')
#             confirm_password = request.form.get('confirm_password')
#             updated_data = {
#                 'username': request.form.get('username'),
#                 'mobileNo': request.form.get('mobileNo'),
#                 'updatedAt': firestore.SERVER_TIMESTAMP
#             }

#             try:
#                 # Update Firestore data
#                 user_ref.set(updated_data, merge=True)

#                 # Update password if provided
#                 if new_password:
#                     if new_password != confirm_password:
#                         flash("Passwords do not match!", 'danger')
#                         return redirect(url_for('profile'))
#                     if len(new_password) < 6:
#                         flash("Password must be at least 6 characters long!", 'danger')
#                         return redirect(url_for('profile'))
#                     firebase_auth.update_user(current_user.id, password=new_password)
#                     flash("Profile and password updated successfully! Please log in again.", 'success')
#                     logout_user()
#                     session.pop('id_token', None)
#                     return redirect(url_for('login_page'))
#                 else:
#                     flash("Profile updated successfully!", 'success')
#                 return redirect(url_for('profile'))
#             except Exception as e:
#                 flash(f"Update failed: {str(e)}", 'danger')

#         elif action == 'delete':
#             try:
#                 user_ref.delete()
#                 firebase_auth.delete_user(current_user.id)
#                 logout_user()
#                 session.pop('id_token', None)
#                 flash('Your account has been deleted.', 'success')
#                 return redirect(url_for('home'))
#             except Exception as e:
#                 flash(f"Deletion failed: {str(e)}", 'danger')

#     return render_template('profile.html', user=current_user, user_data=user_data)

# @app.route('/dashboard', methods=['GET', 'POST'])
# @login_required
# def index():
#     if request.method == 'POST':
#         stock_symbol = request.form['stock_symbol'].upper()
#         data = get_stock_data(stock_symbol)
#         if data is None:
#             return render_template('checkPrediction.html', error="Error: Failed to fetch stock data.")

#         prophet_plot, prophet_past, prophet_preds, prophet_past_dates, pred_dates = prophet_prediction(data, stock_symbol)
#         xgboost_plot, xgboost_past, xgboost_preds, xgboost_past_dates, xgboost_future_dates = xgboost_prediction(data, stock_symbol)
#         lightgbm_plot, lightgbm_past, lightgbm_preds, lightgbm_past_dates, lightgbm_future_dates = lightgbm_prediction(data, stock_symbol)
#         combined_plot = combined_prediction_plot(data, prophet_past, prophet_preds, xgboost_past, xgboost_preds, lightgbm_past, lightgbm_preds, stock_symbol)

#         combined_preds = [(p + x + l) / 3 for p, x, l in zip(prophet_preds, xgboost_preds, lightgbm_preds)]

#         latest_data = data.iloc[-1]
#         stock_info = {
#             'Open': float(latest_data['Open']),
#             'High': float(latest_data['High']),
#             'Low': float(latest_data['Low']),
#             'Close': float(latest_data['Close']),
#             'Volume': float(latest_data['Volume'])
#         }

#         return render_template('results.html', stock_symbol=stock_symbol, stock_info=stock_info,
#                               prophet_plot=prophet_plot, xgboost_plot=xgboost_plot,
#                               lightgbm_plot=lightgbm_plot, combined_plot=combined_plot,
#                               prophet_past=prophet_past, prophet_preds=prophet_preds,
#                               xgboost_past=xgboost_past, xgboost_preds=xgboost_preds,
#                               lightgbm_past=lightgbm_past, lightgbm_preds=lightgbm_preds,
#                               past_dates=prophet_past_dates, pred_dates=pred_dates,
#                               combined_preds=combined_preds)
#     return render_template('checkPrediction.html')

# @app.route('/logout', methods=['GET'])
# @login_required
# def logout():
#     session.pop('id_token', None)
#     logout_user()
#     return redirect(url_for('home'))

# @app.route('/contact', methods=['GET', 'POST'])
# def contact():
#     if request.method == 'POST':
#         name = request.form['name']
#         email = request.form['email']
#         subject = request.form['subject']
#         message = request.form['message']
        
#         msg = Message(subject, sender=email, recipients=['ruramaimuchenga@gmail.com'])
#         msg.body = f"From: {name} ({email})\n\n{message}"
#         try:
#             mail.send(msg)
#             flash('Message sent successfully!', 'success')
#         except Exception as e:
#             flash(f"Failed to send message: {str(e)}", 'danger')
#         return redirect(url_for('contact'))
#     return render_template('contact.html')

# @app.route('/test_static/<path:filename>')
# def test_static(filename):
#     return send_from_directory('static', filename)

# if __name__ == "__main__":
#     app.run(host='0.0.0.0', port=8000, threaded=False, debug=True)






























import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import yfinance as yf
import pandas as pd
import numpy as np
import io
import base64
import os
from datetime import datetime, timedelta
import firebase_admin
from firebase_admin import credentials, auth as firebase_auth, firestore
from prophet import Prophet
import xgboost as xgb
import lightgbm as lgb
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
from bs4 import BeautifulSoup
from flask_mail import Mail, Message

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-here')

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
db = firestore.client()

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login_page'

# Flask-Mail setup
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'ruramaimuchenga@gmail.com'
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD', 'your-app-password')
mail = Mail(app)

# User class for Flask-Login
class User(UserMixin):
    def __init__(self, uid, email, username=None, mobileNo=None):
        self.id = uid
        self.email = email
        self.username = username
        self.mobileNo = mobileNo

@login_manager.user_loader
def load_user(user_id):
    try:
        id_token = session.get('id_token')
        if not id_token:
            return None
        decoded_token = firebase_auth.verify_id_token(id_token, check_revoked=True)
        if decoded_token['uid'] == user_id:
            user_ref = db.collection('users').document(user_id)
            user_data = user_ref.get().to_dict() or {}
            return User(
                user_id,
                decoded_token.get('email', 'unknown'),
                user_data.get('username', 'Unknown'),
                user_data.get('mobileNo', '')
            )
        return None
    except Exception as e:
        print(f"User Loader Error: {str(e)}")
        return None

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
    today = datetime(2025, 4, 10).date()  # Hardcoded for testing
    start_date = (today - timedelta(days=5*365)).strftime('%Y-%m-%d')
    end_date = today.strftime('%Y-%m-%d')
    data = yf.download(symbol, start=start_date, end=end_date, auto_adjust=True)
    if data.empty:
        return None
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]
    print(f"get_stock_data - {symbol} - Columns: {data.columns.tolist()}")
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
    df['tariff_shock'] = ((df.index >= tariff_date).astype(int) * -0.05)
    return df.dropna()

def fetch_news_sentiment(symbol):
    try:
        url = f"https://news.google.com/rss/search?q={symbol}+stock"
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'lxml-xml')
        articles = soup.find_all('item')[:10]
        news_texts = [article.title.text + " " + article.description.text for article in articles if article.description]
        if not news_texts:
            return -0.5, "No recent news available"
        analyzer = SentimentIntensityAnalyzer()
        scores = [analyzer.polarity_scores(text)['compound'] for text in news_texts]
        avg_sentiment = np.mean(scores)
        news_summary = "Positive sentiment" if avg_sentiment > 0 else "Negative sentiment" if avg_sentiment < 0 else "Neutral sentiment"
        print(f"Sentiment - {symbol}: {avg_sentiment} from {len(news_texts)} news sources")
        return avg_sentiment, news_summary
    except Exception as e:
        print(f"Sentiment - {symbol} - Failed: {str(e)}")
        return -0.5, "Failed to fetch news"

def get_market_adjustment():
    sp500 = get_stock_data('^GSPC')
    if sp500 is None:
        return 0.0
    recent_change = sp500['Close'].pct_change(5).iloc[-1]
    adjustment = min(max(recent_change, -0.2), 0.2) * 2
    print(f"Market Adjustment based on S&P 500 5-day change: {adjustment:.2%}")
    return adjustment

def prophet_prediction(data, symbol):
    try:
        tariff_date = pd.to_datetime('2025-04-02')
        today = pd.to_datetime('2025-04-10')
        # Train only up to March 31st (before tariff), then predict forward
        train_end = pd.to_datetime('2025-03-31')
        train_data = data[data.index <= train_end].tail(200)
        test_data = data.tail(10)  # Last 10 days (April 1-10)
        
        df = pd.DataFrame({'ds': train_data.index, 'y': train_data['Close'].values})
        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True,
            growth='linear',
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0
        )
        model.fit(df)
        
        past_future = pd.DataFrame({'ds': test_data.index})
        past_forecast = model.predict(past_future)
        last_10_actual = test_data['Close'].values.tolist()
        past_preds = []
        
        for i, (date, pred) in enumerate(zip(test_data.index, past_forecast['yhat'].values)):
            if date <= pd.to_datetime('2025-04-01'):
                # Up to April 1st, use raw prediction with noise (as before)
                noise = np.random.uniform(-0.02, 0.02)
                past_preds.append(pred * (1 + noise))
            else:
                # From April 2nd, honest model prediction without forcing to actuals
                past_preds.append(pred)

        last_date = data.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=2, freq='B')
        future = pd.DataFrame({'ds': future_dates})
        forecast = model.predict(future)
        prophet_preds = forecast['yhat'].values.tolist()[:2]  # Next 2 days
        
        past_dates = test_data.index
        all_dates = list(past_dates) + list(future_dates)
        date_labels = [d.strftime('%d/%m/%y') for d in all_dates]

        trend = "upward" if prophet_preds[0] > last_10_actual[-1] else "downward"
        reasons = f"Prophet predicts a {trend} trend based on training up to 31/03/25. No tariff knowledge included."
        bias, news_summary = fetch_news_sentiment(symbol)

        plt.figure(figsize=(10, 5))
        x_indices = range(len(last_10_actual) + len(prophet_preds))
        plt.plot(x_indices[:len(last_10_actual)], last_10_actual, label='Actual Past 10', marker='o', color='#007bff')
        plt.plot(x_indices[:len(last_10_actual)], past_preds, label='Predicted Past 10', marker='x', color='#ff9900')
        plt.plot(x_indices[len(last_10_actual):], prophet_preds, label='Predicted Next', marker='o', color='#ff9900')
        plt.title(f'Prophet: {symbol} Past & Future', fontsize=14, pad=15)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price', fontsize=12)
        plt.axvline(x=1, color='r', linestyle='--', label='Tariff Start (02/04)')
        plt.xticks(x_indices, date_labels, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plot_data = plot_to_base64()

        print(f"Prophet - {symbol} - Last 10 actual: {last_10_actual}")
        print(f"Prophet - {symbol} - Predicted Past 10: {past_preds}")
        print(f"Prophet - {symbol} - Predicted Next: {prophet_preds}")
        return (plot_data, past_preds, prophet_preds, 
                [d.strftime('%d/%m/%y') for d in past_dates], [d.strftime('%d/%m/%y') for d in future_dates], 
                reasons, news_summary, last_10_actual)
    except Exception as e:
        print(f"Prophet - {symbol} - Failed: {str(e)}")
        plt.figure(figsize=(10, 5))
        plt.text(0.5, 0.5, 'Prophet Plot Failed', ha='center', va='center')
        plot_data = plot_to_base64()
        return plot_data, [0] * 10, [0] * 2, [], [], "Prediction failed", "No news", [0] * 10

def xgboost_prediction(data, symbol):
    try:
        df = create_features(data.tail(210))  # Extra buffer for feature creation
        train_end = pd.to_datetime('2025-03-31')
        train_df = df[df.index <= train_end].iloc[-190:]  # Last 190 of pre-tariff
        test_df = df.tail(10)  # April 1-10
        
        features = ['lag1', 'lag5', 'ma5', 'ma20', 'vol', 'dow', 'mon', 'trend_5d', 'vol_spike', 'ma5_ma20_ratio', 'tariff_shock']
        X = train_df[features]
        y = train_df['Close']
        model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.01, max_depth=5, min_child_weight=1, reg_lambda=1.0, random_state=42)
        model.fit(X, y)
        
        X_test = test_df[features]
        past_raw_preds = model.predict(X_test).tolist()
        last_10_actual = test_df['Close'].values.tolist()
        past_preds = []
        
        for i, (date, pred) in enumerate(zip(test_df.index, past_raw_preds)):
            if date <= pd.to_datetime('2025-04-01'):
                noise = np.random.uniform(-0.015, 0.015)
                past_preds.append(pred * (1 + noise))
            else:
                past_preds.append(pred)

        last_df = df.tail(5).copy()
        last_date = data.index[-1]
        xgboost_preds = []
        for i in range(2):
            last_features = last_df[features].iloc[-1:].copy()
            pred = model.predict(last_features)[0]
            xgboost_preds.append(float(pred))
            
            new_row = last_df.iloc[-1].copy()
            new_row['Close'] = pred
            new_row['lag1'] = pred
            new_row['lag5'] = last_df['Close'].iloc[-5] if i < 4 else xgboost_preds[i-4]
            new_row['ret'] = (pred - last_df['Close'].iloc[-1]) / last_df['Close'].iloc[-1]
            new_row['ma5'] = (last_df['Close'].tail(4).sum() + pred) / 5
            new_row['ma20'] = (last_df['Close'].tail(19).sum() + pred) / 20
            new_row['vol'] = last_df['ret'].tail(19).std() if i == 0 else pd.concat([last_df['ret'].tail(19), pd.Series([new_row['ret']])]).std()
            new_row['dow'] = ((last_date + timedelta(days=i+1)).weekday()) % 7
            new_row['mon'] = (last_date + timedelta(days=i+1)).month
            new_row['trend_5d'] = (pred - last_df['Close'].iloc[-5]) / last_df['Close'].iloc[-5]
            new_row['vol_spike'] = new_row['vol'] / last_df['vol'].iloc[-1] if last_df['vol'].iloc[-1] != 0 else 1.0
            new_row['ma5_ma20_ratio'] = new_row['ma5'] / new_row['ma20']
            new_row['tariff_shock'] = 0  # Model doesn't know tariff yet
            last_df = pd.concat([last_df, pd.DataFrame([new_row], index=[last_date + timedelta(days=i+1)])])

        past_dates = test_df.index
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=2, freq='B')
        all_dates = list(past_dates) + list(future_dates)
        date_labels = [d.strftime('%d/%m/%y') for d in all_dates]

        trend = "upward" if xgboost_preds[0] > last_10_actual[-1] else "downward"
        reasons = f"XGBoost predicts a {trend} trend based on training up to 31/03/25. No tariff knowledge included."
        bias, news_summary = fetch_news_sentiment(symbol)

        plt.figure(figsize=(10, 5))
        x_indices = range(len(last_10_actual) + len(xgboost_preds))
        plt.plot(x_indices[:len(last_10_actual)], last_10_actual, label='Actual Past 10', marker='o', color='#007bff')
        plt.plot(x_indices[:len(last_10_actual)], past_preds, label='Predicted Past 10', marker='x', color='#28a745')
        plt.plot(x_indices[len(last_10_actual):], xgboost_preds, label='Predicted Next', marker='o', color='#28a745')
        plt.axvline(x=1, color='r', linestyle='--', label='Tariff Start (02/04)')
        plt.title(f'XGBoost: {symbol} Past & Future', fontsize=14, pad=15)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price', fontsize=12)
        plt.xticks(x_indices, date_labels, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plot_data = plot_to_base64()

        print(f"XGBoost - {symbol} - Last 10 actual: {last_10_actual}")
        print(f"XGBoost - {symbol} - Predicted Past 10: {past_preds}")
        print(f"XGBoost - {symbol} - Predicted Next: {xgboost_preds}")
        return (plot_data, past_preds, xgboost_preds, 
                [d.strftime('%d/%m/%y') for d in past_dates], [d.strftime('%d/%m/%y') for d in future_dates], 
                reasons, news_summary, last_10_actual)
    except Exception as e:
        print(f"XGBoost - {symbol} - Failed: {str(e)}")
        plt.figure(figsize=(10, 5))
        plt.text(0.5, 0.5, 'XGBoost Plot Failed', ha='center', va='center')
        plot_data = plot_to_base64()
        return plot_data, [0] * 10, [0] * 2, [], [], "Prediction failed", "No news", [0] * 10

def lightgbm_prediction(data, symbol):
    try:
        df = create_features(data.tail(210))
        train_end = pd.to_datetime('2025-03-31')
        train_df = df[df.index <= train_end].iloc[-190:]
        test_df = df.tail(10)
        
        features = ['lag1', 'lag5', 'ma5', 'ma20', 'vol', 'dow', 'mon', 'trend_5d', 'vol_spike', 'ma5_ma20_ratio', 'tariff_shock']
        X = train_df[features]
        y = train_df['Close']
        model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.01, max_depth=5, min_data_in_leaf=5, reg_lambda=1.0, random_state=42)
        model.fit(X, y)
        
        X_test = test_df[features]
        past_raw_preds = model.predict(X_test).tolist()
        last_10_actual = test_df['Close'].values.tolist()
        past_preds = []
        
        for i, (date, pred) in enumerate(zip(test_df.index, past_raw_preds)):
            if date <= pd.to_datetime('2025-04-01'):
                noise = np.random.uniform(-0.015, 0.015)
                past_preds.append(pred * (1 + noise))
            else:
                past_preds.append(pred)

        last_df = df.tail(5).copy()
        last_date = data.index[-1]
        lightgbm_preds = []
        for i in range(2):
            last_features = last_df[features].iloc[-1:].copy()
            pred = model.predict(last_features)[0]
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
            new_row['trend_5d'] = (pred - last_df['Close'].iloc[-5]) / last_df['Close'].iloc[-5]
            new_row['vol_spike'] = new_row['vol'] / last_df['vol'].iloc[-1] if last_df['vol'].iloc[-1] != 0 else 1.0
            new_row['ma5_ma20_ratio'] = new_row['ma5'] / new_row['ma20']
            new_row['tariff_shock'] = 0  # Model doesn't know tariff yet
            last_df = pd.concat([last_df, pd.DataFrame([new_row], index=[last_date + timedelta(days=i+1)])])

        past_dates = test_df.index
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=2, freq='B')
        all_dates = list(past_dates) + list(future_dates)
        date_labels = [d.strftime('%d/%m/%y') for d in all_dates]

        trend = "upward" if lightgbm_preds[0] > last_10_actual[-1] else "downward"
        reasons = f"LightGBM predicts a {trend} trend based on training up to 31/03/25. No tariff knowledge included."
        bias, news_summary = fetch_news_sentiment(symbol)

        plt.figure(figsize=(10, 5))
        x_indices = range(len(last_10_actual) + len(lightgbm_preds))
        plt.plot(x_indices[:len(last_10_actual)], last_10_actual, label='Actual Past 10', marker='o', color='#007bff')
        plt.plot(x_indices[:len(last_10_actual)], past_preds, label='Predicted Past 10', marker='x', color='#dc3545')
        plt.plot(x_indices[len(last_10_actual):], lightgbm_preds, label='Predicted Next', marker='o', color='#dc3545')
        plt.axvline(x=1, color='r', linestyle='--', label='Tariff Start (02/04)')
        plt.title(f'LightGBM: {symbol} Past & Future', fontsize=14, pad=15)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price', fontsize=12)
        plt.xticks(x_indices, date_labels, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plot_data = plot_to_base64()

        print(f"LightGBM - {symbol} - Last 10 actual: {last_10_actual}")
        print(f"LightGBM - {symbol} - Predicted Past 10: {past_preds}")
        print(f"LightGBM - {symbol} - Predicted Next: {lightgbm_preds}")
        return (plot_data, past_preds, lightgbm_preds, 
                [d.strftime('%d/%m/%y') for d in past_dates], [d.strftime('%d/%m/%y') for d in future_dates], 
                reasons, news_summary, last_10_actual)
    except Exception as e:
        print(f"LightGBM - {symbol} - Failed: {str(e)}")
        plt.figure(figsize=(10, 5))
        plt.text(0.5, 0.5, 'LightGBM Plot Failed', ha='center', va='center')
        plot_data = plot_to_base64()
        return plot_data, [0] * 10, [0] * 2, [], [], "Prediction failed", "No news", [0] * 10

def combined_prediction_plot(data, prophet_past, prophet_future, xgboost_past, xgboost_future, lightgbm_past, lightgbm_future, symbol):
    try:
        last_10 = data['Close'].tail(10).values
        last_date = data.index[-1]
        past_dates = data.index[-10:]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=2, freq='B')
        all_dates = list(past_dates) + list(future_dates)
        date_labels = [d.strftime('%d/%m/%y') for d in all_dates]

        plt.figure(figsize=(10, 5))
        x_indices = range(len(last_10) + len(prophet_future))
        plt.plot(x_indices[:len(last_10)], last_10, label='Actual Past 10', marker='o', color='#007bff')
        plt.plot(x_indices[:len(last_10)], prophet_past, label='Prophet Past', marker='x', color='#ff9900')
        plt.plot(x_indices[:len(last_10)], xgboost_past, label='XGBoost Past', marker='x', color='#28a745')
        plt.plot(x_indices[:len(last_10)], lightgbm_past, label='LightGBM Past', marker='x', color='#dc3545')
        plt.plot(x_indices[len(last_10):], prophet_future, label='Prophet Future', marker='o', color='#ff9900')
        plt.plot(x_indices[len(last_10):], xgboost_future, label='XGBoost Future', marker='o', color='#28a745')
        plt.plot(x_indices[len(last_10):], lightgbm_future, label='LightGBM Future', marker='o', color='#dc3545')
        plt.axvline(x=1, color='r', linestyle='--', label='Tariff Start (02/04)')
        plt.title(f'Combined: {symbol} Past & Future', fontsize=14, pad=15)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price', fontsize=12)
        plt.xticks(x_indices, date_labels, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        return plot_to_base64()
    except Exception as e:
        print(f"Combined Plot - {symbol} - Failed: {str(e)}")
        plt.figure(figsize=(10, 5))
        plt.text(0.5, 0.5, 'Combined Plot Failed', ha='center', va='center')
        return plot_to_base64()

def get_cached_predictions(stock_symbol, cache_duration_hours=24):
    cache_ref = db.collection('predictions').document(stock_symbol)
    data = get_stock_data(stock_symbol)
    if data is None:
        return None
    latest_date = data.index[-1].date()

    prophet_plot, prophet_past, prophet_preds, prophet_past_dates, pred_dates, prophet_reasons, prophet_news, prophet_actual = prophet_prediction(data, stock_symbol)
    xgboost_plot, xgboost_past, xgboost_preds, xgboost_past_dates, xgboost_future_dates, xgboost_reasons, xgboost_news, xgboost_actual = xgboost_prediction(data, stock_symbol)
    lightgbm_plot, lightgbm_past, lightgbm_preds, lightgbm_past_dates, lightgbm_future_dates, lightgbm_reasons, lightgbm_news, lightgbm_actual = lightgbm_prediction(data, stock_symbol)

    prophet_preds = prophet_preds[:2]
    xgboost_preds = xgboost_preds[:2]
    lightgbm_preds = lightgbm_preds[:2]
    combined_preds = [(p + x + l) / 3 for p, x, l in zip(prophet_preds, xgboost_preds, lightgbm_preds)]

    past_predictions_ref = cache_ref.collection('past_predictions')
    for date_str, actual, prophet_pred, xgboost_pred, lightgbm_pred in zip(
        prophet_past_dates, prophet_actual, prophet_past, xgboost_past, lightgbm_past
    ):
        past_predictions_ref.document(date_str).set({
            'date': date_str,
            'actual_price': float(actual),
            'prophet_pred': float(prophet_pred),
            'xgboost_pred': float(xgboost_pred),
            'lightgbm_pred': float(lightgbm_pred),
            'timestamp': firestore.SERVER_TIMESTAMP
        })

    combined_plot = combined_prediction_plot(data, prophet_past, prophet_preds, xgboost_past, xgboost_preds, lightgbm_past, lightgbm_preds, stock_symbol)
    latest_data = data.iloc[-1]
    stock_info = {
        'Open': float(latest_data['Open']),
        'High': float(latest_data['High']),
        'Low': float(latest_data['Low']),
        'Close': float(latest_data['Close']),
        'Volume': float(latest_data['Volume'])
    }

    cache_ref.set({
        'timestamp': firestore.SERVER_TIMESTAMP,
        'prophet_plot': prophet_plot, 'prophet_past': prophet_past, 'prophet_preds': prophet_preds,
        'prophet_past_dates': prophet_past_dates, 'pred_dates': pred_dates,
        'xgboost_plot': xgboost_plot, 'xgboost_past': xgboost_past, 'xgboost_preds': xgboost_preds,
        'lightgbm_plot': lightgbm_plot, 'lightgbm_past': lightgbm_past, 'lightgbm_preds': lightgbm_preds,
        'combined_plot': combined_plot, 'combined_preds': combined_preds, 'stock_info': stock_info,
        'prophet_reasons': prophet_reasons, 'xgboost_reasons': xgboost_reasons, 'lightgbm_reasons': lightgbm_reasons,
        'prophet_news': prophet_news, 'xgboost_news': xgboost_news, 'lightgbm_news': lightgbm_news
    })

    return (prophet_plot, prophet_past, prophet_preds, prophet_past_dates, pred_dates,
            xgboost_plot, xgboost_past, xgboost_preds,
            lightgbm_plot, lightgbm_past, lightgbm_preds,
            combined_plot, combined_preds, stock_info,
            prophet_reasons, xgboost_reasons, lightgbm_reasons,
            prophet_news, xgboost_news, lightgbm_news)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/login', methods=['GET'])
def login_page():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    return render_template('login.html')

@app.route('/auth/login', methods=['POST'])
def login():
    id_token = request.form.get('id_token')
    try:
        decoded_token = firebase_auth.verify_id_token(id_token, check_revoked=True)
        user = User(
            decoded_token['uid'],
            decoded_token.get('email', 'unknown')
        )
        login_user(user)
        session['id_token'] = id_token
        return redirect(url_for('index'))
    except Exception as e:
        print(f"Login Error: {str(e)}")
        return render_template('login.html', error="Login failed: " + str(e))

@app.route('/register', methods=['GET'])
def register_page():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    return render_template('register.html')

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    user_ref = db.collection('users').document(current_user.id)
    user_data = user_ref.get().to_dict() or {}

    if request.method == 'POST':
        action = request.form.get('action')
        
        if action == 'update':
            new_password = request.form.get('new_password')
            confirm_password = request.form.get('confirm_password')
            updated_data = {
                'username': request.form.get('username'),
                'mobileNo': request.form.get('mobileNo'),
                'updatedAt': firestore.SERVER_TIMESTAMP
            }

            try:
                user_ref.set(updated_data, merge=True)
                if new_password:
                    if new_password != confirm_password:
                        flash("Passwords do not match!", 'danger')
                        return redirect(url_for('profile'))
                    if len(new_password) < 6:
                        flash("Password must be at least 6 characters long!", 'danger')
                        return redirect(url_for('profile'))
                    firebase_auth.update_user(current_user.id, password=new_password)
                    flash("Profile and password updated successfully! Please log in again.", 'success')
                    logout_user()
                    session.pop('id_token', None)
                    return redirect(url_for('login_page'))
                else:
                    flash("Profile updated successfully!", 'success')
                return redirect(url_for('profile'))
            except Exception as e:
                flash(f"Update failed: {str(e)}", 'danger')

        elif action == 'delete':
            try:
                user_ref.delete()
                firebase_auth.delete_user(current_user.id)
                logout_user()
                session.pop('id_token', None)
                flash('Your account has been deleted.', 'success')
                return redirect(url_for('home'))
            except Exception as e:
                flash(f"Deletion failed: {str(e)}", 'danger')

    return render_template('profile.html', user=current_user, user_data=user_data)

@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def index():
    if request.method == 'POST':
        stock_symbol = request.form['stock_symbol'].upper()
        
        result = get_cached_predictions(stock_symbol, cache_duration_hours=24)
        if result is None:
            return render_template('checkPrediction.html', error="Error: Failed to fetch stock data.")
        
        (prophet_plot, prophet_past, prophet_preds, prophet_past_dates, pred_dates,
         xgboost_plot, xgboost_past, xgboost_preds,
         lightgbm_plot, lightgbm_past, lightgbm_preds,
         combined_plot, combined_preds, stock_info,
         prophet_reasons, xgboost_reasons, lightgbm_reasons,
         prophet_news, xgboost_news, lightgbm_news) = result

        data = get_stock_data(stock_symbol)
        past_predictions = [
            {
                'date': date_str,
                'actual_price': float(actual),
                'prophet_pred': float(prophet_p),
                'xgboost_pred': float(xgboost_p),
                'lightgbm_pred': float(lightgbm_p)
            }
            for date_str, actual, prophet_p, xgboost_p, lightgbm_p in zip(
                prophet_past_dates, data.tail(10)['Close'].values, prophet_past, xgboost_past, lightgbm_past
            )
        ]

        return render_template('results.html', stock_symbol=stock_symbol, stock_info=stock_info,
                              prophet_plot=prophet_plot, xgboost_plot=xgboost_plot,
                              lightgbm_plot=lightgbm_plot, combined_plot=combined_plot,
                              prophet_past=prophet_past, prophet_preds=prophet_preds,
                              xgboost_past=xgboost_past, xgboost_preds=xgboost_preds,
                              lightgbm_past=lightgbm_past, lightgbm_preds=lightgbm_preds,
                              past_dates=prophet_past_dates, pred_dates=pred_dates,
                              combined_preds=combined_preds,
                              prophet_reasons=prophet_reasons, xgboost_reasons=xgboost_reasons,
                              lightgbm_reasons=lightgbm_reasons,
                              prophet_news=prophet_news, xgboost_news=xgboost_news,
                              lightgbm_news=lightgbm_news,
                              past_predictions=past_predictions)
    return render_template('checkPrediction.html')

@app.route('/logout', methods=['GET'])
@login_required
def logout():
    session.pop('id_token', None)
    logout_user()
    return redirect(url_for('home'))

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        subject = request.form['subject']
        message = request.form['message']
        
        msg = Message(subject, sender=email, recipients=['ruramaimuchenga@gmail.com'])
        msg.body = f"From: {name} ({email})\n\n{message}"
        try:
            mail.send(msg)
            flash('Message sent successfully!', 'success')
        except Exception as e:
            flash(f"Failed to send message: {str(e)}", 'danger')
        return redirect(url_for('contact'))
    return render_template('contact.html')

@app.route('/test_static/<path:filename>')
def test_static(filename):
    return send_from_directory('static', filename)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, threaded=False, debug=True)