import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import r2_score

from stats import wilson_ci, binomial_test, generate_verdict
from backtest import run_backtest, compute_metrics
from ui_components import plot_price_chart, plot_equity_drawdown, plot_feature_importance, plot_model_comparison

def render_dashboard_tab(stock_data, model, scaler, FEATURES, bt_data, metrics, verdict_title, verdict_desc, hit_rate, num_trades, cost_bps, min_confidence):
    col1, col2 = st.columns([3, 1])
    with col1:
        st.plotly_chart(plot_price_chart(stock_data), use_container_width=True)
    
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
    
    if "Real" in verdict_title: st.success(f"### {verdict_title}\n{verdict_desc}")
    elif "Inconclusive" in verdict_title: st.warning(f"### {verdict_title}\n{verdict_desc}")
    else: st.error(f"### {verdict_title}\n{verdict_desc}")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Net Strategy (Ann.)", f"{metrics['ann_strat']:.2f}%")
    c2.metric("Buy & Hold (Ann.)", f"{metrics['ann_mkt']:.2f}%")
    c3.metric("Max Drawdown", f"{metrics['max_dd']:.2f}%")
    ci_low, ci_high = wilson_ci(int(hit_rate * num_trades), num_trades)
    c4.metric(f"Win Rate (n={num_trades})", f"{hit_rate*100:.1f}%", f"95% CI: {ci_low*100:.1f}% - {ci_high*100:.1f}%", delta_color="off")

    st.plotly_chart(plot_equity_drawdown(bt_data, cost_bps), use_container_width=True)

def render_luck_test_tab(bt_data, cost_bps):
    st.subheader("🎲 Monte Carlo Random Luck Test")
    st.markdown("This simulation generates 1,000 random trading strategies to test if the model's returns are statistically distinguishable from random noise.")
    
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

def render_breakeven_tab(test_data, model, scaler, FEATURES, min_confidence, cost_bps):
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

def render_model_comparison_tab():
    st.subheader("🤖 Algorithm Comparison")
    st.markdown("We trained four fundamentally different algorithms on the exact same data (2010-2019) and tested them on the out-of-sample data (2020-Present).")
    
    # We use the hardcoded results from your Jupyter Notebook to keep the app fast
    df_results = pd.DataFrame({
        "Model": ["Zero Predictor", "Ridge Regression", "Neural Network", "Random Forest", "XGBoost"],
        "R² Score": [-0.00404, -0.00404, -0.00017, -0.00191, -0.00528],
        "Hit Rate": [0.5000, 0.5058, 0.4848, 0.5135, 0.5141],
        "Train Time": ["Instant", "Instant", "~10s", "26.61s", "0.74s"]
    })
    
    plot_model_comparison(df_results)
    
    st.markdown("""
    **Key Takeaways:**
    *   **XGBoost** achieved the highest hit rate (51.41%) and trained in just 0.74 seconds.
    *   **The Neural Network** actually performed *worse* than random guessing (48.48%), proving it overfit the noise.
    *   **Ridge Regression** (which we use for live predictions) achieved a 50.58% hit rate and is mathematically interpretable.
    """)

def render_honesty_report_tab(selected_ticker, cost_bps, hit_rate, p_val, num_trades, verdict_title, model, FEATURES):
    st.subheader("📄 System Honesty Report")
    st.markdown("This report summarizes the statistical reality of the model's performance on the selected ticker.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 📊 Performance Snapshot")
        st.write(f"**Ticker:** {selected_ticker}")
        st.write(f"**Model:** Ridge Regression (alpha=10.0)")
        st.write(f"**Train Window:** 2010-2019")
        st.write(f"**Test Window:** 2020-Present")
        st.write(f"**Transaction Costs:** {cost_bps} bps")
        
    with col2:
        st.markdown("#### ⚖️ Statistical Verdict")
        st.write(f"**Hit Rate:** {hit_rate*100:.1f}%")
        st.write(f"**P-Value:** {p_val:.4f}")
        st.write(f"**Total Trades:** {num_trades}")
        if "Real" in verdict_title: st.success(verdict_title)
        elif "Inconclusive" in verdict_title: st.warning(verdict_title)
        else: st.error(verdict_title)
        
    st.markdown("---")
    st.markdown("#### 📝 Plain English Summary")
    
    if "Real" in verdict_title:
        summary = f"The model demonstrated a statistically significant edge on {selected_ticker}. With a p-value of {p_val:.3f} and {num_trades} trades, the probability of this result being pure luck is less than 5%."
    elif "Inconclusive" in verdict_title:
        summary = f"The model showed a slight positive tendency on {selected_ticker}, but the result is statistically inconclusive (p-value = {p_val:.3f})."
    else:
        summary = f"The model failed to demonstrate any statistical edge on {selected_ticker}. Any profits generated are likely due to luck rather than predictive skill."
        
    st.info(summary)
    
    st.markdown("---")
    st.markdown("#### 🧠 Why did the model make this prediction?")
    st.plotly_chart(plot_feature_importance(model, FEATURES, selected_ticker), use_container_width=True)
    
    st.markdown("---")
    st.markdown("#### ⚠️ Limitations & Risks")
    st.markdown("- **Survivorship Bias:** The universe consists of current S&P 500 constituents.\n- **Slippage:** The backtest assumes a static transaction cost.\n- **Regime Shifts:** The model may not generalize to future regimes.")
    
    report_text = f"MARKET FORECAST TERMINAL - HONESTY REPORT\n\nTicker: {selected_ticker}\nVerdict: {verdict_title}\n\nSummary:\n{summary}"
    
    st.download_button(
        label="📥 Download Honesty Report (.txt)",
        data=report_text,
        file_name=f"honesty_report_{selected_ticker}.txt",
        mime="text/plain"
    )