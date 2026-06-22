"""
Cálculo de indicadores técnicos usados pelas estratégias.
Usa a biblioteca `ta` como base, complementada com implementações customizadas.
"""
import numpy as np
import pandas as pd
from scipy import stats


# ─────────────────────────────────────────────────────────
# Médias Móveis
# ─────────────────────────────────────────────────────────

def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=period).mean()


# ─────────────────────────────────────────────────────────
# Bollinger Bands
# ─────────────────────────────────────────────────────────

def bollinger_bands(
    series: pd.Series, period: int = 20, std_dev: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Retorna (banda_superior, média, banda_inferior).
    """
    mid = sma(series, period)
    std = series.rolling(window=period).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    return upper, mid, lower


# ─────────────────────────────────────────────────────────
# RSI
# ─────────────────────────────────────────────────────────

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


# ─────────────────────────────────────────────────────────
# MACD
# ─────────────────────────────────────────────────────────

def macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Retorna (macd_line, signal_line, histogram).
    """
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


# ─────────────────────────────────────────────────────────
# Z-Score (para Pair Trading)
# ─────────────────────────────────────────────────────────

def zscore(series: pd.Series, window: int = 60) -> pd.Series:
    """Z-score rolante de uma série."""
    roll_mean = series.rolling(window=window).mean()
    roll_std = series.rolling(window=window).std()
    return (series - roll_mean) / roll_std


def spread_zscore(
    series_a: pd.Series, series_b: pd.Series, window: int = 60
) -> pd.Series:
    """
    Z-score do spread logarítmico entre dois ativos (Pair Trading).
    Usa regressão OLS para calcular o hedge ratio.
    """
    log_a = np.log(series_a)
    log_b = np.log(series_b)

    # Calcular hedge ratio via OLS rolante
    hedge_ratios = []
    for i in range(window, len(log_a) + 1):
        x = log_b.iloc[i - window:i].values
        y = log_a.iloc[i - window:i].values
        slope, _, _, _, _ = stats.linregress(x, y)
        hedge_ratios.append(slope)

    hedge_series = pd.Series(
        [np.nan] * (window - 1) + hedge_ratios,
        index=series_a.index
    )

    spread = log_a - hedge_series * log_b
    return zscore(spread, window=window)


# ─────────────────────────────────────────────────────────
# Análise de Volume
# ─────────────────────────────────────────────────────────

def avg_volume(volume: pd.Series, window: int = 20) -> pd.Series:
    """Volume médio móvel."""
    return volume.rolling(window=window).mean()


def volume_ratio(volume: pd.Series, window: int = 20) -> pd.Series:
    """Razão volume atual / volume médio."""
    return volume / avg_volume(volume, window)


# ─────────────────────────────────────────────────────────
# ATR — Average True Range
# ─────────────────────────────────────────────────────────

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range para dimensionamento de stops dinâmicos."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()
