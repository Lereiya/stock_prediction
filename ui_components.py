import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import streamlit as st

def plot_price_chart(stock_data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_data.tail(250)["date"], y=stock_data.tail(250)["close"], name="Close", line=dict(color="blue")))
    fig.update_layout(height=300, template="plotly_dark", margin=dict(l=0, r=0, t=0, b=0), yaxis_title="Price (USD)")
    return fig

def plot_equity_drawdown(bt_data, cost_bps):
    fig_bt = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
    fig_bt.add_trace(go.Scatter(x=bt_data["date"], y=(bt_data["cum_strategy"] - 1)*100, name=f"Model ({cost_bps} bps)", line=dict(color="green")), row=1, col=1)
    fig_bt.add_trace(go.Scatter(x=bt_data["date"], y=(bt_data["cum_market"] - 1)*100, name="Buy & Hold", line=dict(color="grey", dash="dash")), row=1, col=1)
    fig_bt.add_trace(go.Scatter(x=bt_data["date"], y=bt_data["drawdown"]*100, name="Drawdown", fill="tozeroy", line=dict(color="red")), row=2, col=1)
    fig_bt.update_layout(height=450, template="plotly_dark", hovermode="x unified", margin=dict(l=0, r=0, t=20, b=0))
    fig_bt.update_yaxes(title_text="Return (%)", row=1, col=1)
    fig_bt.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
    return fig_bt

def plot_feature_importance(model, features, ticker):
    weights = model.coef_
    importance_df = pd.DataFrame({"Feature": features, "Weight": weights}).sort_values(by="Weight", ascending=True)
    
    fig = go.Figure(go.Bar(
        x=importance_df["Weight"],
        y=importance_df["Feature"],
        orientation='h',
        marker_color=['red' if w < 0 else 'green' for w in importance_df["Weight"]]
    ))
    fig.update_layout(
        title=f"Feature Weights: What Drives {ticker} Predictions?",
        xaxis_title="Weight (Positive pushes price UP, Negative pushes DOWN)",
        yaxis_title="Technical Indicator",
        template="plotly_dark",
        height=400,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    return fig

def plot_model_comparison(results_df):
    st.markdown("### Final Model Comparison")
    
    # Display the dataframe as a beautiful table
    st.dataframe(results_df, use_container_width=True, hide_index=True)
    
    # Add a bar chart for Hit Rate comparison
    fig = go.Figure(data=[
        go.Bar(name='Hit Rate', x=results_df['Model'], y=results_df['Hit Rate'], marker_color='steelblue')
    ])
    fig.update_layout(title="Directional Hit Rate by Model", yaxis_title="Hit Rate", template="plotly_dark", height=400)
    fig.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="Random Guessing (50%)")
    st.plotly_chart(fig, use_container_width=True)