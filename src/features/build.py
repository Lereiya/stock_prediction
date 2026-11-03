"""Feature engineering: convert raw prices into ML features."""
import numpy as np
import pandas as pd

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Given a single ticker's price history (sorted by date), compute features."""
    out = df.copy()
    close = out["close"]

    # Target: next-day log return
    out["log_return"] = np.log(close / close.shift(1))
    out["target"] = out["log_return"].shift(-1)  # tomorrow's return

    # Momentum features
    for lag in [1, 2, 5, 10, 20]:
        out[f"ret_lag_{lag}"] = out["log_return"].rolling(lag).sum()

    # Volatility features
    for window in [5, 10, 20]:
        out[f"vol_{window}"] = out["log_return"].rolling(window).std()

    # RSI (14-day)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    out["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out["macd"] = ema12 - ema26

    # Bollinger band position
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    out["bb_position"] = (close - sma20) / (2 * std20 + 1e-9)

    # Volume features
    out["vol_ratio"] = out["volume"] / (out["volume"].rolling(20).mean() + 1e-9)

    return out

def build_feature_matrix(prices_path: str = "data/raw/prices.parquet"):
    raw = pd.read_parquet(prices_path)
    raw["date"] = pd.to_datetime(raw["date"])

    feature_frames = []
    for ticker, group in raw.groupby("ticker"):
        group = group.sort_values("date").reset_index(drop=True)
        feat = compute_features(group)
        feature_frames.append(feat)

    features = pd.concat(feature_frames, ignore_index=True)

    # Drop rows with NaN (from rolling windows)
    feature_cols = [c for c in features.columns if c.startswith(
        ("ret_lag", "vol_", "rsi", "macd", "bb_", "vol_ratio")
    )]
    features = features.dropna(subset=feature_cols + ["target"])

    features.to_parquet("data/processed/features.parquet")
    print(f"Saved {len(features):,} rows, {len(feature_cols)} features")

if __name__ == "__main__":
    build_feature_matrix()