import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from stats import binomial_test, generate_verdict
from backtest import run_backtest, compute_metrics
from tabs import (
    render_dashboard_tab, render_luck_test_tab, 
    render_breakeven_tab, render_honesty_report_tab
)

st.set_page_config(page_title="Market Forecast Terminal", page_icon="📈", layout="wide")
st.title("📈 Market Forecast Terminal")
st.markdown("*Educational use only. Not investment advice.*")

# --- DATA PIPELINE ---
@st.cache_data(ttl=3600) 
def fetch_live_data(ticker):
    df = yf.download(ticker, start="2010-01-01", progress=False, auto_adjust=True)
    if df.empty: return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df.reset_index(inplace=True)
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    
    close = df["close"]
    df["log_return"] = np.log(close / close.shift(1))
    df["target"] = df["log_return"].shift(-1)
    
    for lag in [1, 2, 5, 10, 20]: df[f"ret_lag_{lag}"] = df["log_return"].rolling(lag).sum()
    for window in [5, 10, 20]: df[f"vol_{window}"] = df["log_return"].rolling(window).std()
        
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["rsi_14"] = 100 - (100 / (1 + rs))
    
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df["bb_position"] = (close - sma20) / (2 * std20 + 1e-9)
    df["vol_ratio"] = df["volume"] / (df["volume"].rolling(20).mean() + 1e-9)
    
    return df.dropna().reset_index(drop=True)

@st.cache_resource
def train_model(df):
    FEATURES = ["ret_lag_1", "ret_lag_2", "ret_lag_5", "ret_lag_10", "ret_lag_20", "vol_5", "vol_10", "vol_20", "rsi_14", "macd", "bb_position", "vol_ratio"]
    train = df[df["date"] <= "2019-12-31"]
    if len(train) < 100: return None, None, FEATURES
    X_train, y_train = train[FEATURES].values, train["target"].values
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = Ridge(alpha=10.0).fit(X_train_scaled, y_train)
    return model, scaler, FEATURES

# --- SIDEBAR ---
st.sidebar.header("⚙️ Configuration")
STOCK_DICT = {"AAPL": "Apple Inc.", "MSFT": "Microsoft Corp.", "GOOGL": "Alphabet Inc.", "AMZN": "Amazon.com Inc.", "NVDA": "NVIDIA Corp.", "TSLA": "Tesla Inc."}
formatted_options = [f"{t} - {n}" for t, n in STOCK_DICT.items()]
selected_option = st.sidebar.selectbox("Select a Stock Ticker", formatted_options)
selected_ticker = selected_option.split(" - ")[0]
TICKERS = list(STOCK_DICT.keys())

cost_bps = st.sidebar.slider("Transaction Cost (bps)", 0, 20, 5)
min_confidence = st.sidebar.slider("Min Confidence", 0.0, 0.005, 0.001, format="%.3f")

# --- EXECUTE ---
stock_data = fetch_live_data(selected_ticker)
if stock_data.empty:
    st.error("Could not fetch data.")
    st.stop()

model, scaler, FEATURES = train_model(stock_data)
test_data = stock_data[stock_data["date"] > "2019-12-31"].copy()
bt_data = run_backtest(test_data, model.predict(scaler.transform(test_data[FEATURES].values)), cost_bps, min_confidence)
metrics = compute_metrics(bt_data)

hit_rate, num_trades = metrics["win_rate"], metrics["num_trades"]
p_val = binomial_test(int(hit_rate * num_trades), num_trades)
verdict_title, verdict_desc = generate_verdict(hit_rate, p_val, num_trades, metrics["ann_strat"], metrics["ann_mkt"])

# --- UI ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Dashboard", "🎲 Random Luck Test", "📉 Break-Even Cost", "🔍 Screener", "📄 Honesty Report", "ℹ️ How It Works"])

with tab1:
    render_dashboard_tab(stock_data, model, scaler, FEATURES, bt_data, metrics, verdict_title, verdict_desc, hit_rate, num_trades, cost_bps, min_confidence)
with tab2:
    render_luck_test_tab(bt_data, cost_bps)
with tab3:
    render_breakeven_tab(test_data, model, scaler, FEATURES, min_confidence, cost_bps)
with tab4:
    st.info("To use the screener, run the code from your original app.py")
with tab5:
    render_honesty_report_tab(selected_ticker, cost_bps, hit_rate, p_val, num_trades, verdict_title, model, FEATURES)
with tab6:
    st.markdown("### How it works\nThis app uses Ridge Regression trained on 2010-2019 data. See the report for details.")