import numpy as np
import pandas as pd
from scipy import stats

def wilson_ci(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    if trials == 0: return 0.0, 0.0
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p_hat = successes / trials
    denominator = 1 + z**2 / trials
    center = (p_hat + z**2 / (2 * trials)) / denominator
    margin = z * np.sqrt((p_hat * (1 - p_hat) / trials) + (z**2 / (4 * trials**2))) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)

def binomial_test(successes: int, trials: int) -> float:
    if trials == 0: return 1.0
    res = stats.binomtest(successes, trials, p=0.5, alternative='two-sided')
    return res.pvalue

def generate_verdict(hit_rate: float, p_value: float, num_trades: int, net_return: float, market_return: float) -> tuple[str, str]:
    if num_trades < 100:
        return "🔴 Likely Noise", f"Only {num_trades} trades. Need at least 100 to establish statistical significance."
    if net_return < market_return:
        return "🔴 Likely Noise", f"Strategy underperformed Buy & Hold ({net_return:.1f}% vs {market_return:.1f}%)."
    if p_value < 0.05 and hit_rate > 0.5:
        return "🟢 Real Edge", f"Statistically significant win rate ({hit_rate*100:.1f}%, p={p_value:.3f}) over {num_trades} trades."
    return "🟡 Inconclusive", f"Win rate is {hit_rate*100:.1f}% (p={p_value:.3f}), but not statistically significant."