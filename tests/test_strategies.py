"""
Testes unitários das estratégias e utilitários.
Execute: pytest tests/test_strategies.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.utils.indicators import (
    ema, sma, bollinger_bands, rsi, macd, zscore, spread_zscore, atr
)
from src.risk.risk_manager import RiskManager, TradeSignal
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.momentum_macro import MomentumMacroStrategy
from src.strategies.pair_trading import PairTradingStrategy


# ─────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────

def make_ohlcv(n: int = 300, trend: float = 0.0, noise: float = 1.0) -> pd.DataFrame:
    """Gera OHLCV sintético."""
    np.random.seed(42)
    close = 50 * np.ones(n)
    for i in range(1, n):
        close[i] = close[i - 1] * (1 + trend + np.random.normal(0, noise / 100))
    high = close * 1.005
    low = close * 0.995
    open_ = close * (1 + np.random.normal(0, 0.002, n))
    volume = np.random.randint(100_000, 1_000_000, n).astype(float)
    index = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "open": open_, "high": high, "low": low,
        "close": close, "volume": volume
    }, index=index)


# ─────────────────────────────────────────────────────────
# Indicadores
# ─────────────────────────────────────────────────────────

class TestIndicators:

    def test_ema_length(self):
        s = pd.Series(range(100), dtype=float)
        result = ema(s, 9)
        assert len(result) == 100

    def test_sma_values(self):
        s = pd.Series([1, 2, 3, 4, 5], dtype=float)
        result = sma(s, 3)
        assert result.iloc[-1] == pytest.approx(4.0)

    def test_bollinger_bands_shape(self):
        df = make_ohlcv()
        upper, mid, lower = bollinger_bands(df["close"], 20, 2.0)
        assert all(upper.dropna() > mid.dropna())
        assert all(mid.dropna() > lower.dropna())

    def test_rsi_range(self):
        df = make_ohlcv()
        r = rsi(df["close"], 14)
        valid = r.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_macd_components(self):
        df = make_ohlcv()
        m, s, h = macd(df["close"])
        assert len(m) == len(df)
        assert (h == m - s).all()

    def test_zscore_mean_zero(self):
        s = pd.Series(np.random.normal(0, 1, 200))
        z = zscore(s, window=60)
        assert abs(z.dropna().mean()) < 0.5

    def test_atr_positive(self):
        df = make_ohlcv()
        a = atr(df["high"], df["low"], df["close"])
        assert (a.dropna() > 0).all()


# ─────────────────────────────────────────────────────────
# RiskManager
# ─────────────────────────────────────────────────────────

class TestRiskManager:

    def setup_method(self):
        self.rm = RiskManager(
            capital=100_000,
            max_pos_pct=0.20,
            stop_loss_pct=0.025,
            max_drawdown=0.10,
            kelly_fraction=0.25,
        )

    def test_initial_state(self):
        assert self.rm.current_capital == 100_000
        assert self.rm.current_drawdown == 0.0
        assert self.rm.is_trading_allowed()

    def test_drawdown_trigger(self):
        self.rm.update_capital(89_000)  # -11% drawdown
        assert not self.rm.is_trading_allowed()

    def test_position_size_output(self):
        signal = TradeSignal(
            ticker="SBSP3", direction="long", strategy="test",
            entry_price=50.0, stop_loss_price=48.75, take_profit_price=51.0,
        )
        pos = self.rm.calculate_position_size(signal)
        assert pos is not None
        assert pos.shares > 0
        assert pos.capital_allocated <= 100_000 * 0.20

    def test_max_exposure_respected(self):
        signal = TradeSignal(
            ticker="SBSP3", direction="long", strategy="test",
            entry_price=50.0, stop_loss_price=48.0, take_profit_price=52.0,
        )
        self.rm._open_positions["SBSP3"] = 20_000  # Já alocado
        pos = self.rm.calculate_position_size(signal)
        if pos:
            assert pos.capital_allocated + 20_000 <= 100_000 * 0.20 + 1

    def test_stop_price_calculation(self):
        stop = self.rm.calculate_stop_price(100.0, "long", 0.025)
        assert stop == pytest.approx(97.5)
        stop_short = self.rm.calculate_stop_price(100.0, "short", 0.025)
        assert stop_short == pytest.approx(102.5)


# ─────────────────────────────────────────────────────────
# Estratégia: Reversão à Média
# ─────────────────────────────────────────────────────────

class TestMeanReversion:

    def setup_method(self):
        self.strategy = MeanReversionStrategy(assets=["SBSP3", "EQTL3"])

    def test_no_signal_insufficient_data(self):
        df = make_ohlcv(n=10)
        result = self.strategy.analyze("SBSP3", df)
        assert result is None

    def test_no_signal_unknown_ticker(self):
        df = make_ohlcv(n=300)
        result = self.strategy.analyze("PETR4", df)
        assert result is None

    def test_long_signal_conditions(self):
        """Força condições de oversold para verificar sinal."""
        df = make_ohlcv(n=200)
        # Forçar close muito abaixo da banda inferior
        df.loc[df.index[-1], "close"] = df["close"].mean() * 0.85
        # Não garantimos sinal pois RSI pode não estar < 30, mas não deve crashar
        result = self.strategy.analyze("SBSP3", df)
        # Resultado pode ser None ou TradeSignal — ambos são válidos
        assert result is None or result.direction in ("long", "short")


# ─────────────────────────────────────────────────────────
# Estratégia: Momentum Macro
# ─────────────────────────────────────────────────────────

class TestMomentumMacro:

    def setup_method(self):
        self.strategy = MomentumMacroStrategy(assets=["SBSP3"])

    def test_no_signal_without_regime(self):
        df = make_ohlcv(n=200)
        # Regime padrão = unknown (não ativo)
        result = self.strategy.analyze("SBSP3", df)
        assert result is None

    def test_regime_activation(self):
        self.strategy.set_macro_regime("high_cut_expect")
        assert self.strategy._momentum_active is True

    def test_regime_deactivation(self):
        self.strategy.set_macro_regime("high_stable")
        assert self.strategy._momentum_active is False


# ─────────────────────────────────────────────────────────
# Estratégia: Pair Trading
# ─────────────────────────────────────────────────────────

class TestPairTrading:

    def setup_method(self):
        self.strategy = PairTradingStrategy(
            lookback_days=60,
            z_entry=2.0,
            z_exit=0.5,
            pairs=[("EQTL3", "TAEE11")],
        )

    def test_insufficient_data(self):
        df_a = make_ohlcv(n=30)
        df_b = make_ohlcv(n=30)
        result = self.strategy.analyze_pair("EQTL3", df_a, "TAEE11", df_b)
        assert result is None

    def test_valid_data_returns_none_or_signal(self):
        df_a = make_ohlcv(n=200, trend=0.001)
        df_b = make_ohlcv(n=200, trend=-0.001)
        result = self.strategy.analyze_pair("EQTL3", df_a, "TAEE11", df_b)
        # Pode ser None (z < 2) ou PairSignal — não deve lançar exceção
        assert result is None or hasattr(result, "zscore")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
