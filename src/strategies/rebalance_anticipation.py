"""
Estratégia 5: Antecipação de Rebalanceamento
──────────────────────────────────────────────
Racional: O UTIL é revisado quadrimestralmente (Jan, Mai, Set).
O rebalanceamento da carteira gera fluxo previsível que pode ser antecipado.

Implementação:
  - Monitorar prévias da B3 (publicadas antes do rebalanceamento)
  - Comprar ações prestes a entrar no índice (forte demanda passiva esperada)
  - Vender ações prestes a sair (saída de fundos passivos)
  - Horizonte: 1–10 dias úteis ao redor do rebalanceamento
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import pandas as pd
import requests
from loguru import logger

from src.risk.risk_manager import TradeSignal


class RebalanceAnticipationStrategy:

    name = "rebalance_anticipation"

    # Meses de rebalanceamento do UTIL (Janeiro, Maio, Setembro)
    REBALANCE_MONTHS = [1, 5, 9]
    # Dia aproximado de vigência da nova carteira (primeiro dia útil do mês)
    EFFECTIVE_DAY = 2

    def __init__(
        self,
        days_before: int = 5,
        stop_loss_pct: float = 0.025,
        entries_candidates: Optional[list[str]] = None,
        exits_candidates: Optional[list[str]] = None,
    ):
        self.days_before = days_before
        self.stop_loss_pct = stop_loss_pct
        # Candidatos a ENTRAR no próximo rebalanceamento (comprar antes)
        self.entries_candidates = entries_candidates or []
        # Candidatos a SAIR no próximo rebalanceamento (vender antes)
        self.exits_candidates = exits_candidates or []

    def update_candidates(
        self,
        entries: list[str],
        exits: list[str],
    ) -> None:
        """
        Atualiza listas de candidatos com base nas prévias da B3.
        Deve ser chamado após publicação da prévia oficial.
        """
        self.entries_candidates = entries
        self.exits_candidates = exits
        logger.info(
            "[Rebalance] Candidatos atualizados | entradas={} | saídas={}",
            entries, exits
        )

    def is_active_window(self) -> tuple[bool, Optional[date]]:
        """
        Verifica se estamos na janela de antecipação do próximo rebalanceamento.
        Retorna (ativo, data_rebalance).
        """
        today = date.today()

        for month in self.REBALANCE_MONTHS:
            year = today.year
            # Se o mês já passou este ano, olhar para o próximo ano
            rebal_date = date(year, month, self.EFFECTIVE_DAY)
            if rebal_date < today:
                if month == max(self.REBALANCE_MONTHS):
                    rebal_date = date(year + 1, self.REBALANCE_MONTHS[0], self.EFFECTIVE_DAY)
                else:
                    next_month = self.REBALANCE_MONTHS[self.REBALANCE_MONTHS.index(month) + 1]
                    rebal_date = date(year, next_month, self.EFFECTIVE_DAY)

            days_to = (rebal_date - today).days
            if 0 < days_to <= (self.days_before + 5):
                logger.info(
                    "[Rebalance] Janela ativa | rebal={} | dias={}", rebal_date, days_to
                )
                return True, rebal_date

        return False, None

    def scan(
        self, ohlcv_dict: dict[str, pd.DataFrame]
    ) -> list[TradeSignal]:
        """
        Gera sinais de antecipação se estiver na janela de rebalanceamento.
        """
        active, rebal_date = self.is_active_window()
        if not active:
            return []

        signals = []
        today = date.today()
        days_to = (rebal_date - today).days if rebal_date else 99

        # ── Candidatos a entrar → comprar (long) ──
        for ticker in self.entries_candidates:
            if ticker not in ohlcv_dict:
                continue
            entry_price = ohlcv_dict[ticker]["close"].iloc[-1]

            # Confiança decresce conforme se aproxima do rebalanceamento
            confidence = max(0.4, 1.0 - (self.days_before - days_to) * 0.1)

            signals.append(TradeSignal(
                ticker=ticker,
                direction="long",
                strategy=self.name,
                entry_price=entry_price,
                stop_loss_price=entry_price * (1 - self.stop_loss_pct),
                take_profit_price=entry_price * 1.03,
                confidence=confidence,
                notes=f"Antecipação de entrada no UTIL | rebal={rebal_date} | dias={days_to}",
            ))
            logger.info(
                "[Rebalance] LONG {} (candidato entrada) | rebal={} | dias={}",
                ticker, rebal_date, days_to
            )

        # ── Candidatos a sair → vender (short) ──
        for ticker in self.exits_candidates:
            if ticker not in ohlcv_dict:
                continue
            entry_price = ohlcv_dict[ticker]["close"].iloc[-1]
            confidence = max(0.4, 1.0 - (self.days_before - days_to) * 0.1)

            signals.append(TradeSignal(
                ticker=ticker,
                direction="short",
                strategy=self.name,
                entry_price=entry_price,
                stop_loss_price=entry_price * (1 + self.stop_loss_pct),
                take_profit_price=entry_price * 0.97,
                confidence=confidence,
                notes=f"Antecipação de saída do UTIL | rebal={rebal_date} | dias={days_to}",
            ))
            logger.info(
                "[Rebalance] SHORT {} (candidato saída) | rebal={} | dias={}",
                ticker, rebal_date, days_to
            )

        return signals
