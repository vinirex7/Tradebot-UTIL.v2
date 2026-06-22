"""
Estratégia 1: Reversão à Média (Mean Reversion)
─────────────────────────────────────────────────
Racional: Utilities com beta baixo tendem a oscilar em torno de médias estáveis.
Quando o preço se afasta significativamente da média histórica, há alta
probabilidade de retorno à média.

Implementação:
  - Indicadores: Bollinger Bands (20, 2σ) + RSI (14 períodos)
  - Sinal LONG: preço abaixo da banda inferior + RSI < 30
  - Sinal SHORT: preço acima da banda superior + RSI > 70
  - Stop loss: 2.5% do valor de entrada
  - Take profit: retorno à média móvel de 20 períodos
  - Timeframe: H1 ou D1
  - Ativos: SBSP3, EQTL3, CPLE3, EGIE3, TAEE11
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
from loguru import logger

from src.utils.indicators import bollinger_bands, rsi, atr
from src.risk.risk_manager import TradeSignal


class MeanReversionStrategy:

    name = "mean_reversion"

    def __init__(
        self,
        bollinger_period: int = 20,
        bollinger_std: float = 2.0,
        rsi_period: int = 14,
        rsi_oversold: float = 30,
        rsi_overbought: float = 70,
        stop_loss_pct: float = 0.025,
        take_profit_pct: float = 0.015,
        assets: Optional[list[str]] = None,
        blackout_events: Optional[list] = None,
    ):
        self.bollinger_period = bollinger_period
        self.bollinger_std = bollinger_std
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.assets = assets or ["SBSP3", "EQTL3", "CPLE3", "EGIE3", "TAEE11"]
        self.blackout_events = blackout_events or []

    def analyze(self, ticker: str, ohlcv: pd.DataFrame) -> Optional[TradeSignal]:
        """
        Analisa OHLCV e retorna sinal de trade ou None.

        Args:
            ticker: Código do ativo
            ohlcv:  DataFrame com colunas open/high/low/close/volume

        Returns:
            TradeSignal ou None
        """
        if ticker not in self.assets:
            return None

        if len(ohlcv) < self.bollinger_period + self.rsi_period:
            logger.debug("Dados insuficientes para {} ({}  barras)", ticker, len(ohlcv))
            return None

        close = ohlcv["close"]
        upper, mid, lower = bollinger_bands(close, self.bollinger_period, self.bollinger_std)
        rsi_series = rsi(close, self.rsi_period)

        last_close = close.iloc[-1]
        last_upper = upper.iloc[-1]
        last_lower = lower.iloc[-1]
        last_mid = mid.iloc[-1]
        last_rsi = rsi_series.iloc[-1]

        # ── Sinal LONG ──
        if last_close < last_lower and last_rsi < self.rsi_oversold:
            entry = last_close
            stop = entry * (1 - self.stop_loss_pct)
            tp = last_mid  # Take profit na média de 20 períodos

            signal = TradeSignal(
                ticker=ticker,
                direction="long",
                strategy=self.name,
                entry_price=entry,
                stop_loss_price=stop,
                take_profit_price=tp,
                confidence=self._confidence(last_rsi, "long"),
                notes=f"BB lower={last_lower:.2f} | RSI={last_rsi:.1f} | mid={last_mid:.2f}",
            )
            logger.info(
                "[MeanRev] LONG {} | close={:.2f} < lower={:.2f} | RSI={:.1f}",
                ticker, last_close, last_lower, last_rsi
            )
            return signal

        # ── Sinal SHORT ──
        if last_close > last_upper and last_rsi > self.rsi_overbought:
            entry = last_close
            stop = entry * (1 + self.stop_loss_pct)
            tp = last_mid

            signal = TradeSignal(
                ticker=ticker,
                direction="short",
                strategy=self.name,
                entry_price=entry,
                stop_loss_price=stop,
                take_profit_price=tp,
                confidence=self._confidence(last_rsi, "short"),
                notes=f"BB upper={last_upper:.2f} | RSI={last_rsi:.1f} | mid={last_mid:.2f}",
            )
            logger.info(
                "[MeanRev] SHORT {} | close={:.2f} > upper={:.2f} | RSI={:.1f}",
                ticker, last_close, last_upper, last_rsi
            )
            return signal

        return None

    def check_exit(self, ticker: str, ohlcv: pd.DataFrame, position_dir: str) -> bool:
        """
        Verifica se posição deve ser fechada:
        - Long: fechamento acima da média de 20p (take profit)
        - Short: fechamento abaixo da média de 20p
        """
        close = ohlcv["close"]
        _, mid, _ = bollinger_bands(close, self.bollinger_period, self.bollinger_std)
        last_close = close.iloc[-1]
        last_mid = mid.iloc[-1]

        if position_dir == "long" and last_close >= last_mid:
            logger.info("[MeanRev] Sinal de saída LONG {} | close >= mid", ticker)
            return True
        if position_dir == "short" and last_close <= last_mid:
            logger.info("[MeanRev] Sinal de saída SHORT {} | close <= mid", ticker)
            return True
        return False

    @staticmethod
    def _confidence(rsi_value: float, direction: str) -> float:
        """Ajusta confiança do sinal pelo nível extremo do RSI."""
        if direction == "long":
            return min(1.0, max(0.5, (30 - rsi_value) / 20 + 0.7))
        return min(1.0, max(0.5, (rsi_value - 70) / 20 + 0.7))
