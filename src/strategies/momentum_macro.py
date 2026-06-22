"""
Estratégia: Momentum Macro — alinhada ao backtest atualizado
────────────────────────────────────────────────────────────
Versão live/paper equivalente à lógica restaurada em backtest/backtest_engine.py:

  - Entrada: EMA9 cruza acima da EMA21
  - Confirmação: EMA21 acima da EMA50 e MACD bullish
  - Stop: EMA50 dinâmica, com proteção máxima de 3% abaixo da entrada
  - Sem take profit fixo: deixa o trade correr enquanto a tendência segue válida
  - Ativos padrão: SBSP3, EQTL3, ENEV3, CPLE3

O filtro macro pode ser ligado por configuração, mas fica desligado por padrão
para reproduzir o backtest atualizado.
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
        macro_filter_enabled: bool = False,
        assets: Optional[list[str]] = None,
    ):
        self.ema_fast = ema_fast
        self.ema_mid = ema_mid
        self.ema_slow = ema_slow
        self.di1_threshold = di1_threshold
        self.stop_loss_pct = stop_loss_pct
        self.macro_filter_enabled = macro_filter_enabled
        self.assets = assets or ["SBSP3", "EQTL3", "ENEV3", "CPLE3"]
        self._macro_regime: str = "unknown"
        self._momentum_active: bool = True

    def set_macro_regime(self, regime: str) -> None:
        """
        Recebe o regime macroeconômico calculado pelo MacroFeed.

        Por padrão, o filtro macro fica desligado para manter o live/paper
        alinhado com o backtest atualizado. Se macro_filter_enabled=True,
        o momentum só opera em high_cut_expect ou easing.
        """
        self._macro_regime = regime
        self._momentum_active = (
            regime in ("high_cut_expect", "easing")
            if self.macro_filter_enabled
            else True
        )
        logger.info(
            "[Momentum] Regime macroeconômico: {} | filtro_macro={} | Momentum ativo: {}",
            regime, self.macro_filter_enabled, self._momentum_active
        )

    def analyze(self, ticker: str, ohlcv: pd.DataFrame) -> Optional[TradeSignal]:
        """
        Gera sinal de compra quando a lógica do backtest atualizado aparece:
        EMA9 cruza acima da EMA21, EMA21 está acima da EMA50 e MACD está bullish.
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

        if pd.isna(ema9.iloc[-1]) or pd.isna(ema21.iloc[-1]) or pd.isna(ema50.iloc[-1]):
            return None

        cross_up = (
            ema9.iloc[-1] > ema21.iloc[-1]
            and ema9.iloc[-2] <= ema21.iloc[-2]
            and ema21.iloc[-1] > ema50.iloc[-1]
        )
        macd_bullish = macd_line.iloc[-1] > signal_line.iloc[-1]

        if cross_up and macd_bullish:
            entry = float(close.iloc[-1])
            ema50_stop = float(ema50.iloc[-1])
            max_loss_stop = entry * (1 - self.stop_loss_pct)

            # Igual ao backtest: stop inicial pela EMA50, com proteção de perda máxima.
            stop = max(ema50_stop, max_loss_stop)

            logger.info(
                "[Momentum] LONG {} | EMA9={:.2f} > EMA21={:.2f} | EMA21>EMA50 | MACD bullish",
                ticker, ema9.iloc[-1], ema21.iloc[-1]
            )
            return TradeSignal(
                ticker=ticker,
                direction="long",
                strategy=self.name,
                entry_price=entry,
                stop_loss_price=stop,
                take_profit_price=0.0,  # Sem take profit fixo, igual ao backtest atualizado.
                confidence=1.0,
                notes=(
                    f"EMA9={ema9.iloc[-1]:.2f} | EMA21={ema21.iloc[-1]:.2f} | "
                    f"EMA50={ema50.iloc[-1]:.2f} | TP=none | macro_filter={self.macro_filter_enabled}"
                ),
            )

        return None

    def check_exit(
        self,
        ticker: str,
        ohlcv: pd.DataFrame,
        position_dir: str,
        entry_price: Optional[float] = None,
    ) -> bool:
        """
        Saída alinhada ao backtest atualizado:
        - Fecha se o preço cair abaixo da EMA50 dinâmica com folga de 1%.
        - Fecha se cair mais que stop_loss_pct abaixo da entrada.
        - Não fecha por take profit fixo.
        """
        if position_dir != "long":
            return False

        min_bars = max(self.ema_slow, 60)
        if len(ohlcv) < min_bars:
            return False

        close = ohlcv["close"]
        ema50 = ema(close, self.ema_slow)

        current_close = float(close.iloc[-1])
        trailing_stop = float(ema50.iloc[-1]) * 0.99

        if current_close < trailing_stop:
            logger.info(
                "[Momentum] Saída {} | close={:.2f} < trailing EMA50={:.2f}",
                ticker, current_close, trailing_stop
            )
            return True

        if entry_price is not None:
            hard_stop = float(entry_price) * (1 - self.stop_loss_pct)
            if current_close < hard_stop:
                logger.info(
                    "[Momentum] Saída {} | close={:.2f} < stop máximo={:.2f}",
                    ticker, current_close, hard_stop
                )
                return True

        return False
