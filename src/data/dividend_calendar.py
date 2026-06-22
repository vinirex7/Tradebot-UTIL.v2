"""
Calendário de dividendos das ações do UTIL.
Busca dados de ex-dates e dividend yields via Status Invest e RI das empresas.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Optional

import requests
import pandas as pd
from bs4 import BeautifulSoup
from loguru import logger


class DividendCalendar:
    """Monitora o calendário de dividendos das 18 ações do UTIL."""

    STATUS_INVEST_BASE = "https://statusinvest.com.br"

    # Ações prioritárias para dividend capture (alto DY histórico)
    PRIORITY_TICKERS = ["TAEE11", "EGIE3", "CPFE3", "ENGI11", "CPLE3", "SAPR11"]

    def __init__(self):
        self._cache: dict[str, list] = {}
        self._headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/124.0.0.0 Safari/537.36"
        }

    def get_upcoming_exdates(
        self, tickers: list[str], days_ahead: int = 30
    ) -> pd.DataFrame:
        """
        Retorna ex-dates dos próximos N dias para os tickers informados.
        Resultado ordenado por proximidade do ex-date.
        """
        records = []
        for ticker in tickers:
            events = self._fetch_dividends(ticker)
            for event in events:
                ex_date = event.get("ex_date")
                if ex_date and isinstance(ex_date, date):
                    days_to = (ex_date - date.today()).days
                    if 0 <= days_to <= days_ahead:
                        records.append({
                            "ticker": ticker,
                            "ex_date": ex_date,
                            "days_to_exdate": days_to,
                            "dividend_value": event.get("value", 0.0),
                            "dy_pct": event.get("dy_pct", 0.0),
                            "type": event.get("type", "JCP"),
                        })

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df.sort_values("days_to_exdate", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def _fetch_dividends(self, ticker: str) -> list[dict]:
        """
        Busca dividendos de um ativo via Status Invest API.
        """
        if ticker in self._cache:
            return self._cache[ticker]

        try:
            url = f"{self.STATUS_INVEST_BASE}/acao/companytickerprovents"
            params = {
                "ticker": ticker,
                "chartProventsType": 1,  # Dividendos + JCP
            }
            resp = requests.get(url, params=params, headers=self._headers, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            events = []
            for item in data.get("assetEarningsModels", []):
                try:
                    ex_date = datetime.strptime(item["pd"], "%d/%m/%Y").date()
                    events.append({
                        "ex_date": ex_date,
                        "value": float(item.get("v", 0)),
                        "dy_pct": float(item.get("dy", 0)) / 100,
                        "type": item.get("et", "DIV"),
                    })
                except (KeyError, ValueError):
                    continue

            self._cache[ticker] = events
            logger.debug("Dividendos {} carregados: {} eventos", ticker, len(events))
            return events

        except Exception as e:
            logger.error("Erro ao buscar dividendos de {}: {}", ticker, e)
            return []

    def filter_by_quality(
        self,
        df: pd.DataFrame,
        min_dy: float = 0.05,
        min_volume_brl: float = 50_000_000,
        volume_data: Optional[dict] = None,
    ) -> pd.DataFrame:
        """
        Filtra oportunidades de dividend capture pelo DY mínimo
        e volume médio mínimo.
        """
        filtered = df[df["dy_pct"] >= min_dy].copy()

        if volume_data:
            filtered = filtered[
                filtered["ticker"].map(lambda t: volume_data.get(t, 0)) >= min_volume_brl
            ]

        return filtered.reset_index(drop=True)

    def entry_date(self, ex_date: date, days_before: int = 4) -> date:
        """
        Calcula a data de entrada (dias úteis antes do ex-date).
        Considera fins de semana.
        """
        d = ex_date
        business_days = 0
        while business_days < days_before:
            d -= timedelta(days=1)
            if d.weekday() < 5:  # Segunda a Sexta
                business_days += 1
        return d
