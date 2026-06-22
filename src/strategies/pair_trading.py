"""
Estratégia 3: Pair Trading (Long/Short Intrasetorial)
──────────────────────────────────────────────────────
Racional: Dentro do UTIL existem sub-grupos com comportamentos
correlacionados. Desvios momentâneos de correlação podem ser explorados.

Pares implementados:
  Par 1: EQTL3 vs TAEE11 — Distribuidora/crescimento vs Transmissora/dividendo
  Par 2: SBSP3 vs ENEV3  — Saneamento vs Geração termoelétrica

Método: Z-score do spread logarítmico (lookback 60 dias).
  - Entrada: z-score > 2   → long underperformer, short outperformer
  - Saída:   z-score ≈ 0   → fechar ambas as pernas
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.utils.indicators import spread_zscore
from src.risk.risk_manager import TradeSignal


@dataclass
class PairSignal:
    """Sinal de pair trading com duas pernas."""
    pair_id: str
    long_ticker: str
    short_ticker: str
    zscore: float
    direction: str          # "open" | "close"
    long_signal: TradeSignal
    short_signal: TradeSignal
    notes: str = ""


class PairTradingStrategy:

    name = "pair_trading"

    PAIRS = [
        ("EQTL3", "TAEE11"),   # Distribuidora vs Transmissora
        ("SBSP3", "ENEV3"),    # Saneamento vs Termoelétrica
    ]

    def __init__(
        self,
        lookback_days: int = 60,
        z_entry: float = 2.0,
        z_exit: float = 0.5,
        stop_loss_pct: float = 0.03,
        pairs: Optional[list[tuple[str, str]]] = None,
    ):
        self.lookback_days = lookback_days
        self.z_entry = z_entry
        self.z_exit = z_exit
        self.stop_loss_pct = stop_loss_pct
        self.pairs = pairs or self.PAIRS
        self._active_pairs: dict[str, str] = {}  # pair_id -> direction

    def analyze_pair(
        self,
        ticker_a: str,
        ohlcv_a: pd.DataFrame,
        ticker_b: str,
        ohlcv_b: pd.DataFrame,
    ) -> Optional[PairSignal]:
        """
        Analisa um par e retorna sinal se z-score ultrapassar o threshold.
        """
        pair_id = f"{ticker_a}_{ticker_b}"

        # Alinhar índices
        close_a = ohlcv_a["close"].rename(ticker_a)
        close_b = ohlcv_b["close"].rename(ticker_b)
        aligned = pd.concat([close_a, close_b], axis=1).dropna()

        if len(aligned) < self.lookback_days + 10:
            logger.debug("[PairTrade] Dados insuficientes para par {}/{}", ticker_a, ticker_b)
            return None

        z = spread_zscore(
            aligned[ticker_a], aligned[ticker_b], window=self.lookback_days
        )
        current_z = z.iloc[-1]

        if pd.isna(current_z):
            return None

        last_a = aligned[ticker_a].iloc[-1]
        last_b = aligned[ticker_b].iloc[-1]

        pair_active = pair_id in self._active_pairs

        # ── Abertura de posição ──
        if not pair_active and abs(current_z) >= self.z_entry:
            if current_z > 0:
                # A caro, B barato → short A, long B
                long_tk, short_tk = ticker_b, ticker_a
                long_price, short_price = last_b, last_a
            else:
                # B caro, A barato → short B, long A
                long_tk, short_tk = ticker_a, ticker_b
                long_price, short_price = last_a, last_b

            logger.info(
                "[PairTrade] ABRINDO par {} | z={:.2f} | long={} short={}",
                pair_id, current_z, long_tk, short_tk
            )

            self._active_pairs[pair_id] = "open"

            return PairSignal(
                pair_id=pair_id,
                long_ticker=long_tk,
                short_ticker=short_tk,
                zscore=current_z,
                direction="open",
                long_signal=TradeSignal(
                    ticker=long_tk,
                    direction="long",
                    strategy=self.name,
                    entry_price=long_price,
                    stop_loss_price=long_price * (1 - self.stop_loss_pct),
                    take_profit_price=long_price * 1.04,
                    confidence=min(1.0, abs(current_z) / 3),
                    notes=f"Par={pair_id} | z={current_z:.2f} | leg=long",
                ),
                short_signal=TradeSignal(
                    ticker=short_tk,
                    direction="short",
                    strategy=self.name,
                    entry_price=short_price,
                    stop_loss_price=short_price * (1 + self.stop_loss_pct),
                    take_profit_price=short_price * 0.96,
                    confidence=min(1.0, abs(current_z) / 3),
                    notes=f"Par={pair_id} | z={current_z:.2f} | leg=short",
                ),
                notes=f"z={current_z:.2f}",
            )

        # ── Fechamento de posição ──
        if pair_active and abs(current_z) <= self.z_exit:
            logger.info(
                "[PairTrade] FECHANDO par {} | z={:.2f} retornou a 0",
                pair_id, current_z
            )
            del self._active_pairs[pair_id]

            return PairSignal(
                pair_id=pair_id,
                long_ticker=ticker_a,
                short_ticker=ticker_b,
                zscore=current_z,
                direction="close",
                long_signal=TradeSignal(
                    ticker=ticker_a, direction="close", strategy=self.name,
                    entry_price=last_a, stop_loss_price=0, take_profit_price=0,
                ),
                short_signal=TradeSignal(
                    ticker=ticker_b, direction="close", strategy=self.name,
                    entry_price=last_b, stop_loss_price=0, take_profit_price=0,
                ),
                notes=f"z={current_z:.2f} | reversão concluída",
            )

        return None

    def run_all_pairs(
        self, ohlcv_dict: dict[str, pd.DataFrame]
    ) -> list[PairSignal]:
        """Analisa todos os pares configurados."""
        signals = []
        for ticker_a, ticker_b in self.pairs:
            if ticker_a not in ohlcv_dict or ticker_b not in ohlcv_dict:
                logger.debug("[PairTrade] Dados ausentes para {}/{}", ticker_a, ticker_b)
                continue
            sig = self.analyze_pair(
                ticker_a, ohlcv_dict[ticker_a],
                ticker_b, ohlcv_dict[ticker_b],
            )
            if sig:
                signals.append(sig)
        return signals
