"""
Estratégia 4: Captura de Dividendos (Dividend Capture)
────────────────────────────────────────────────────────
Racional: O UTIL é composto por grandes pagadoras de dividendos.
Entrar antes do ex-date e capturar o provento.

Filtros de qualidade:
  - DY > 5% na distribuição
  - Volume médio diário > R$50M
  - Ativos preferidos: TAEE11, EGIE3, CPFE3, ENGI11

Timing:
  - Entrada: 3–5 dias úteis antes do ex-date
  - Saída: no ex-date ou dia seguinte (depende do comportamento histórico)
"""
from __future__ import annotations

from datetime import date
from typing import Optional

import pandas as pd
from loguru import logger

from src.data.dividend_calendar import DividendCalendar
from src.risk.risk_manager import TradeSignal


class DividendCaptureStrategy:

    name = "dividend_capture"

    def __init__(
        self,
        days_before_exdate: int = 4,
        min_dy_pct: float = 0.05,
        min_volume_brl: float = 50_000_000,
        stop_loss_pct: float = 0.02,
        assets: Optional[list[str]] = None,
        calendar: Optional[DividendCalendar] = None,
    ):
        self.days_before_exdate = days_before_exdate
        self.min_dy_pct = min_dy_pct
        self.min_volume_brl = min_volume_brl
        self.stop_loss_pct = stop_loss_pct
        self.assets = assets or ["TAEE11", "EGIE3", "CPFE3", "ENGI11", "CPLE3"]
        self.calendar = calendar or DividendCalendar()
        self._active_captures: dict[str, date] = {}  # ticker -> ex_date alvo

    def scan(
        self, ohlcv_dict: dict[str, pd.DataFrame]
    ) -> list[TradeSignal]:
        """
        Varre calendário de dividendos e retorna sinais de entrada para
        oportunidades que estão a `days_before_exdate` do ex-date.
        """
        signals = []
        upcoming = self.calendar.get_upcoming_exdates(self.assets, days_ahead=30)

        if upcoming.empty:
            logger.debug("[DivCapture] Nenhum ex-date próximo.")
            return []

        # Volume médio dos últimos 20 pregões
        volume_data = {}
        for ticker, ohlcv in ohlcv_dict.items():
            if "volume" in ohlcv.columns and len(ohlcv) >= 20:
                avg_vol = ohlcv["volume"].iloc[-20:].mean()
                price = ohlcv["close"].iloc[-1]
                volume_data[ticker] = avg_vol * price  # volume financeiro

        for _, row in upcoming.iterrows():
            ticker = row["ticker"]
            ex_date = row["ex_date"]
            dy_pct = row["dy_pct"]
            days_to = row["days_to_exdate"]

            # Filtro de qualidade
            if dy_pct < self.min_dy_pct:
                continue
            if volume_data.get(ticker, 0) < self.min_volume_brl:
                continue

            # Janela de entrada
            entry_day = self.calendar.entry_date(ex_date, self.days_before_exdate)
            today = date.today()
            is_entry_window = (entry_day <= today <= ex_date)

            if not is_entry_window:
                continue

            # Evitar duplicata
            if ticker in self._active_captures:
                continue

            if ticker not in ohlcv_dict:
                continue

            entry_price = ohlcv_dict[ticker]["close"].iloc[-1]
            stop = entry_price * (1 - self.stop_loss_pct)
            # TP = preço + dividendo por ação (recuperação pós ex-date)
            dividend_value = row.get("dividend_value", 0.0)
            tp = entry_price + dividend_value * 0.8  # 80% do dividendo como TP

            logger.info(
                "[DivCapture] SINAL {} | ex-date={} | DY={:.2f}% | dias={} | entry={:.2f}",
                ticker, ex_date, dy_pct * 100, days_to, entry_price
            )

            self._active_captures[ticker] = ex_date

            signals.append(TradeSignal(
                ticker=ticker,
                direction="long",
                strategy=self.name,
                entry_price=entry_price,
                stop_loss_price=stop,
                take_profit_price=tp if tp > entry_price else entry_price * 1.01,
                confidence=min(1.0, dy_pct / 0.08),  # Maior DY = maior confiança
                notes=(
                    f"ex_date={ex_date} | DY={dy_pct*100:.2f}% | "
                    f"div_value=R${dividend_value:.4f} | days_to={days_to}"
                ),
            ))

        return signals

    def check_exit(self, ticker: str, ohlcv: pd.DataFrame) -> bool:
        """
        Saída: no dia do ex-date ou no dia seguinte.
        """
        if ticker not in self._active_captures:
            return False

        ex_date = self._active_captures[ticker]
        today = date.today()

        # Sair no ex-date ou após
        if today >= ex_date:
            logger.info("[DivCapture] Saída {} | ex-date={} atingido", ticker, ex_date)
            del self._active_captures[ticker]
            return True
        return False
