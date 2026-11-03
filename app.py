import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import yfinance as yf

from stats import wilson_ci, binomial_test, generate_verdict
from backtest import run_backtest, compute_metrics

# --- PAGE CONFIG ---
st.set_page_config(page_title="Market Forecast Terminal", page_icon="📈", layout="wide")
st.title("📈 Market Forecast Terminal")
st.markdown("*Educational use only. Not investment advice. Past performance does not predict future results.*")

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

# Dictionary mapping Tickers to Full Names
STOCK_DICT = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corp.",
    "GOOGL": "Alphabet Inc.",
    "AMZN": "Amazon.com Inc.",
    "NVDA": "NVIDIA Corp.",
    "META": "Meta Platforms",
    "TSLA": "Tesla Inc.",
    "JPM": "JPMorgan Chase",
    "V": "Visa Inc.",
    "UNH": "UnitedHealth Group",
    "XOM": "Exxon Mobil",
    "JNJ": "Johnson & Johnson",
    "WMT": "Walmart Inc.",
    "MA": "Mastercard Inc.",
    "PG": "Procter & Gamble",
    "HD": "Home Depot"
}

# Create the formatted options: "AAPL - Apple Inc."
formatted_options = [f"{ticker} - {name}" for ticker, name in STOCK_DICT.items()]
selected_option = st.sidebar.selectbox("Select a Stock Ticker", formatted_options)

# Extract just the ticker symbol for the model/API
selected_ticker = selected_option.split(" - ")[0]
TICKERS = list(STOCK_DICT.keys()) # For the screener

st.sidebar.markdown("### 💰 Backtest Settings")
cost_bps = st.sidebar.slider("Transaction Cost (bps)", 0, 20, 5)
min_confidence = st.sidebar.slider("Min Confidence", 0.0, 0.005, 0.001, format="%.3f")

# --- DATA ---
stock_data = fetch_live_data(selected_ticker)
if stock_data.empty:
    st.error("Could not fetch data.")
    st.stop()

model, scaler, FEATURES = train_model(stock_data)
test_data = stock_data[stock_data["date"] > "2019-12-31"].copy()

# --- MAIN UI TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🎲 Random Luck Test", "📉 Break-Even Cost", "🔍 Multi-Stock Screener"])

# ==========================================
# TAB 1: DASHBOARD
# ==========================================
with tab1:
    col1, col2 = st.columns([2, 1])
    with col1:
        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(x=stock_data.tail(250)["date"], y=stock_data.tail(250)["close"], name="Close", line=dict(color="blue")))
        fig_price.update_layout(height=300, template="plotly_dark", margin=dict(l=0, r=0, t=0, b=0), yaxis_title="Price (USD)")
        st.plotly_chart(fig_price, use_container_width=True)
    
    with col2:
        st.subheader("🔮 Next-Day Prediction")
        latest = stock_data.iloc[-1]
        latest_scaled = scaler.transform(latest[FEATURES].values.reshape(1, -1).astype(float))
        prediction = model.predict(latest_scaled)[0]
        
        st.metric("Current Close", f"${latest['close']:.2f}")
        st.metric("Current RSI", f"{latest['rsi_14']:.2f}")
        st.markdown("---")
        
        if abs(prediction) < min_confidence:
            st.warning(f"⚠️ **UNCERTAIN**: {prediction:.5f}")
        elif prediction > 0:
            st.success(f"📈 **UP**: +{prediction:.5f}")
        else:
            st.error(f"📉 **DOWN**: {prediction:.5f}")

    st.markdown("---")
    st.subheader("💰 Honest Backtest (2020-Present)")
    
    bt_data = run_backtest(test_data, model.predict(scaler.transform(test_data[FEATURES].values)), cost_bps, min_confidence)
    metrics = compute_metrics(bt_data)
    
    hit_rate, num_trades = metrics["win_rate"], metrics["num_trades"]
    p_val = binomial_test(int(hit_rate * num_trades), num_trades)
    verdict_title, verdict_desc = generate_verdict(hit_rate, p_val, num_trades, metrics["ann_strat"], metrics["ann_mkt"])
    
    if "Real" in verdict_title: st.success(f"### {verdict_title}\n{verdict_desc}")
    elif "Inconclusive" in verdict_title: st.warning(f"### {verdict_title}\n{verdict_desc}")
    else: st.error(f"### {verdict_title}\n{verdict_desc}")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Net Strategy (Ann.)", f"{metrics['ann_strat']:.2f}%")
    c2.metric("Buy & Hold (Ann.)", f"{metrics['ann_mkt']:.2f}%")
    c3.metric("Max Drawdown", f"{metrics['max_dd']:.2f}%")
    ci_low, ci_high = wilson_ci(int(hit_rate * num_trades), num_trades)
    c4.metric(f"Win Rate (n={num_trades})", f"{hit_rate*100:.1f}%", f"95% CI: {ci_low*100:.1f}% - {ci_high*100:.1f}%", delta_color="off")

    fig_bt = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
    fig_bt.add_trace(go.Scatter(x=bt_data["date"], y=(bt_data["cum_strategy"] - 1)*100, name=f"Model ({cost_bps} bps)", line=dict(color="green")), row=1, col=1)
    fig_bt.add_trace(go.Scatter(x=bt_data["date"], y=(bt_data["cum_market"] - 1)*100, name="Buy & Hold", line=dict(color="grey", dash="dash")), row=1, col=1)
    fig_bt.add_trace(go.Scatter(x=bt_data["date"], y=bt_data["drawdown"]*100, name="Drawdown", fill="tozeroy", line=dict(color="red")), row=2, col=1)
    fig_bt.update_layout(height=450, template="plotly_dark", hovermode="x unified", margin=dict(l=0, r=0, t=20, b=0))
    fig_bt.update_yaxes(title_text="Return (%)", row=1, col=1)
    fig_bt.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
    st.plotly_chart(fig_bt, use_container_width=True)

# ==========================================
# TAB 2: RANDOM LUCK TEST
# ==========================================
with tab2:
    st.subheader("🎲 Monte Carlo Random Luck Test")
    st.markdown("Is the model actually finding a signal, or did it just get lucky? We simulate 1,000 random trading strategies with the exact same number of trades and costs. If the model is just luck, it will fall in the middle of this bell curve.")
    
    if st.button("Run 1,000 Random Simulations"):
        with st.spinner("Simulating random strategies..."):
            actual_model_return = (bt_data["cum_strategy"].iloc[-1] - 1) * 100
            n_trades = int(bt_data["trade"].sum())
            
            random_returns = []
            np.random.seed(42)
            for _ in range(1000):
                random_signals = np.random.choice([1, 0, -1], size=len(bt_data), p=[0.45, 0.1, 0.45])
                random_ret = random_signals * bt_data["target"].values
                random_trades = np.abs(np.diff(random_signals, prepend=0))
                random_ret -= (random_trades * (cost_bps / 10000))
                random_returns.append(((1 + random_ret).prod() - 1) * 100)
            
            fig_mc = go.Figure()
            fig_mc.add_trace(go.Histogram(x=random_returns, nbinsx=50, name="Random Strategies", marker_color='grey', opacity=0.7))
            fig_mc.add_vline(x=actual_model_return, line_dash="dash", line_color="green", annotation_text="Our Model", annotation_position="top right")
            fig_mc.update_layout(title="Distribution of 1,000 Random Strategies", xaxis_title="Net Return (%)", yaxis_title="Frequency", template="plotly_dark")
            st.plotly_chart(fig_mc, use_container_width=True)
            
            percentile = (np.array(random_returns) < actual_model_return).mean() * 100
            st.info(f"**Result:** Our model's return of {actual_model_return:.2f}% beat **{percentile:.1f}%** of random strategies. (p-value = {1 - (percentile/100):.3f})")

# ==========================================
# TAB 3: BREAK-EVEN COST
# ==========================================
with tab3:
    st.subheader("📉 Break-Even Cost Analysis")
    st.markdown("How much can you afford to pay in broker fees before this strategy becomes unprofitable? This chart calculates the net return across transaction costs from 0 to 50 bps.")
    
    cost_range = np.arange(0, 55, 5)
    returns_at_costs = []
    
    for c in cost_range:
        bt_c = run_backtest(test_data, model.predict(scaler.transform(test_data[FEATURES].values)), c, min_confidence)
        returns_at_costs.append((bt_c["cum_strategy"].iloc[-1] - 1) * 100)
        
    fig_cost = go.Figure()
    fig_cost.add_trace(go.Scatter(x=cost_range, y=returns_at_costs, mode='lines+markers', name="Net Return", line=dict(color="yellow", width=3)))
    fig_cost.add_hline(y=0, line_dash="dash", line_color="red")
    fig_cost.add_vline(x=cost_bps, line_dash="dash", line_color="green", annotation_text=f"Current: {cost_bps} bps")
    fig_cost.update_layout(title="Net Strategy Return vs. Transaction Costs", xaxis_title="Transaction Cost (bps)", yaxis_title="Net Return (%)", template="plotly_dark")
    st.plotly_chart(fig_cost, use_container_width=True)
    
    break_even = None
    for i in range(len(returns_at_costs)-1):
        if returns_at_costs[i] > 0 and returns_at_costs[i+1] <= 0:
            break_even = cost_range[i+1]
            break
            
    if break_even:
        st.success(f"**Break-Even Cost:** The strategy becomes unprofitable at **{break_even} bps**.")
    else:
        st.error("**Break-Even Cost:** The strategy is already unprofitable at 0 bps, or remains profitable up to 50 bps.")

# ==========================================
# TAB 4: MULTI-STOCK SCREENER
# ==========================================
with tab4:
    st.subheader("🔍 Market Screener")
    st.markdown("Scan the entire universe to find the strongest signals today.")
    
    if st.button("Run Market Scan"):
        results = []
        progress_bar = st.progress(0)
        
        for i, t in enumerate(TICKERS):
            try:
                df_t = fetch_live_data(t)
                m_t, s_t, _ = train_model(df_t)
                td_t = df_t[df_t["date"] > "2019-12-31"].copy()
                
                latest_t = df_t.iloc[-1]
                pred_t = m_t.predict(s_t.transform(latest_t[FEATURES].values.reshape(1, -1).astype(float)))[0]
                
                bt_t = run_backtest(td_t, m_t.predict(s_t.transform(td_t[FEATURES].values)), cost_bps, min_confidence)
                met_t = compute_metrics(bt_t)
                
                results.append({
                    "Ticker": t,
                    "Signal": "UP" if pred_t > min_confidence else ("DOWN" if pred_t < -min_confidence else "ABSTAIN"),
                    "Prediction": f"{pred_t:+.5f}",
                    "Backtest Return": f"{met_t['ann_strat']:.1f}%",
                    "Win Rate": f"{met_t['win_rate']*100:.1f}%",
                    "Trades": met_t["num_trades"]
                })
            except:
                pass
            progress_bar.progress((i + 1) / len(TICKERS))
            
        df_results = pd.DataFrame(results)
        st.dataframe(df_results, use_container_width=True, hide_index=True)
        
        st.info("**Note on Multiple Testing:** When screening 10+ stocks, 1 or 2 will likely look 'profitable' by pure chance. Always apply a Bonferroni correction (divide your p-value threshold by the number of stocks tested).")