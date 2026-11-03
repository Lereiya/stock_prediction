import pandas as pd
import numpy as np

def run_backtest(df: pd.DataFrame, predictions: np.ndarray, cost_bps: float, min_confidence: float):
    df = df.copy()
    df["prediction"] = predictions
    
    df["signal"] = 0
    df.loc[df["prediction"] > min_confidence, "signal"] = 1
    df.loc[df["prediction"] < -min_confidence, "signal"] = -1
    
    df["actual_return"] = df["target"] 
    df["strategy_return"] = df["signal"] * df["actual_return"]
    
    df["trade"] = df["signal"].diff().abs()
    df["strategy_return"] -= (df["trade"] * (cost_bps / 10000))
    
    df["cum_strategy"] = (1 + df["strategy_return"]).cumprod()
    df["cum_market"] = (1 + df["actual_return"]).cumprod()
    
    df["peak"] = df["cum_strategy"].cummax()
    df["drawdown"] = (df["cum_strategy"] - df["peak"]) / df["peak"]
    
    return df

def compute_metrics(df: pd.DataFrame):
    strat_ret = df["strategy_return"].dropna()
    mkt_ret = df["actual_return"].dropna()
    
    ann_strat = (1 + strat_ret.mean())**252 - 1
    ann_mkt = (1 + mkt_ret.mean())**252 - 1
    vol_strat = strat_ret.std() * np.sqrt(252)
    
    sharpe = ann_strat / vol_strat if vol_strat > 0 else 0
    max_dd = df["drawdown"].min() * 100
    
    traded_days = df[df["signal"] != 0]
    num_trades = int(df["trade"].sum())
    win_rate = (traded_days["strategy_return"] > 0).mean() if len(traded_days) > 0 else 0
    
    return {
        "ann_strat": ann_strat * 100,
        "ann_mkt": ann_mkt * 100,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "win_rate": win_rate,
        "num_trades": num_trades,
    }