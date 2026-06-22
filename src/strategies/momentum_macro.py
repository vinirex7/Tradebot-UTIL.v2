"""
Estratégia 2: Momentum Macro-Driven (Selic/DI)
────────────────────────────────────────────────
Racional: O UTIL reage fortemente a expectativas de juros.
Quando há sinal claro de inflexão da Selic (corte iminente), o índice
tende a ter momentum forte e sustentado.

Implementação:
  - Filtro: curva DI curta abaixo de threshold → ativa modo long
  - Técnica: EMA 9 cruzando acima da EMA 21
  - Stop: fechamento abaixo da EMA 50
  - Horizonte: 2 semanas a 3 meses
  - Ativos: SBSP3, EQTL3, ENEV3, CPLE3
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
from loguru import logger

from src.utils.indicators import ema, macd
from src.risk.risk_manager import TradeSignal


class MomentumMacroStrategy:

    name = "momentum_macro"

    def __init__(
        self,
        ema_fast: int = 9,
        ema_mid: int = 21,
        ema_slow: int = 50,
        di1_threshold: float = 0.12,
        stop_loss_pct: float = 0.03,
        assets: Optional[list[str]] = None,
    ):
        self.ema_fast = ema_fast
        self.ema_mid = ema_mid
        self.ema_slow = ema_slow
        self.di1_threshold = di1_threshold   # DI1 1 ano abaixo = momentum long
        self.stop_loss_pct = stop_loss_pct
        self.assets = assets or ["SBSP3", "EQTL3", "ENEV3", "CPLE3"]
        self._macro_regime: str = "unknown"  # "high_cut_expect" | "easing" | etc.
        self._momentum_active: bool = False

    def set_macro_regime(self, regime: str) -> None:
        """
        Recebe o regime macroeconômico calculado pelo MacroFeed.
        Ativa/desativa o modo momentum conforme o regime.
        """
        self._macro_regime = regime
        self._momentum_active = regime in ("high_cut_expect", "easing")
        logger.info(
            "[Momentum] Regime macroeconômico: {} | Momentum ativo: {}",
            regime, self._momentum_active
        )

    def analyze(self, ticker: str, ohlcv: pd.DataFrame) -> Optional[TradeSignal]:
        """
        Gera sinal de momentum se:
        1. O regime macro permite (Selic com expectativa de corte), E
        2. EMA9 cruzou acima de EMA21 (confirmação técnica)
        """
        if ticker not in self.assets:
            return None

        if not self._momentum_active:
            logger.debug("[Momentum] Modo não ativo | regime={}", self._macro_regime)
            return None

        min_bars = max(self.ema_slow, 60)
        if len(ohlcv) < min_bars:
            return None

        close = ohlcv["close"]
        ema9 = ema(close, self.ema_fast)
        ema21 = ema(close, self.ema_mid)
        ema50 = ema(close, self.ema_slow)
        macd_line, signal_line, _ = macd(close)

        # EMA cruzamento: EMA9 acima de EMA21 e ambas acima de EMA50
        cross_up = (
            ema9.iloc[-1] > ema21.iloc[-1]
            and ema9.iloc[-2] <= ema21.iloc[-2]  # Cruzamento confirmado
            and ema21.iloc[-1] > ema50.iloc[-1]  # Tendência de alta
        )

        # Confirmação MACD
        macd_bullish = macd_line.iloc[-1] > signal_line.iloc[-1]

        if cross_up and macd_bullish:
            entry = close.iloc[-1]
            stop = ema50.iloc[-1]  # Stop na EMA50
            risk = entry - stop
            tp = entry + risk * 2  # Risk/Reward 1:2

            logger.info(
                "[Momentum] LONG {} | EMA9={:.2f} > EMA21={:.2f} | regime={}",
                ticker, ema9.iloc[-1], ema21.iloc[-1], self._macro_regime
            )
            return TradeSignal(
                ticker=ticker,
                direction="long",
                strategy=self.name,
                entry_price=entry,
                stop_loss_price=max(stop, entry * (1 - self.stop_loss_pct)),
                take_profit_price=tp,
                confidence=0.85,
                notes=(
                    f"Regime={self._macro_regime} | "
                    f"EMA9={ema9.iloc[-1]:.2f} | EMA21={ema21.iloc[-1]:.2f} | "
                    f"EMA50={ema50.iloc[-1]:.2f}"
                ),
            )

        return None

    def check_exit(self, ticker: str, ohlcv: pd.DataFrame, position_dir: str) -> bool:
        """
        Saída: fechamento abaixo da EMA 50 (invalidação da tendência).
        """
        if position_dir != "long":
            return False

        close = ohlcv["close"]
        ema50 = ema(close, self.ema_slow)

        if close.iloc[-1] < ema50.iloc[-1]:
            logger.info(
                "[Momentum] Saída {} | close={:.2f} < EMA50={:.2f}",
                ticker, close.iloc[-1], ema50.iloc[-1]
            )
            return True

        # Saída se regime virou
        if self._macro_regime in ("high_stable", "low_stable"):
            logger.info("[Momentum] Saída {} | regime voltou a {}", ticker, self._macro_regime)
            return True

        return False
