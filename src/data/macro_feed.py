"""
Feed de dados macroeconômicos: Selic, curva DI, Focus/Copom.
Usa python-bcb (Banco Central do Brasil API) como fonte primária.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import requests
import pandas as pd
from loguru import logger


class MacroFeed:
    """Dados macroeconômicos do BCB e B3."""

    BCB_BASE = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{serie}/dados"
    # Série 432 = Taxa Selic Over (% a.a.)
    SELIC_SERIE = 432
    # Série 13522 = Meta Selic definida pelo Copom (% a.a.)
    SELIC_META_SERIE = 13522

    def __init__(self):
        self._cache: dict = {}

    # ──────────────────────────────────────────────
    # Selic
    # ──────────────────────────────────────────────

    def get_selic_rate(self) -> Optional[float]:
        """
        Retorna a taxa Selic Over mais recente (% a.a.).
        """
        try:
            url = self.BCB_BASE.format(serie=self.SELIC_META_SERIE)
            params = {"formato": "json", "dataInicial": (date.today() - timedelta(days=30)).strftime("%d/%m/%Y")}
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if data:
                rate = float(data[-1]["valor"])
                logger.info("Selic Meta atual: {}% a.a.", rate)
                return rate / 100  # Retorna como decimal
        except Exception as e:
            logger.error("Erro ao buscar Selic: {}", e)
        return None

    def get_selic_history(self, days: int = 252) -> pd.DataFrame:
        """Histórico da Meta Selic."""
        try:
            start = (date.today() - timedelta(days=days)).strftime("%d/%m/%Y")
            url = self.BCB_BASE.format(serie=self.SELIC_META_SERIE)
            resp = requests.get(url, params={"formato": "json", "dataInicial": start}, timeout=15)
            resp.raise_for_status()
            df = pd.DataFrame(resp.json())
            df["data"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
            df["valor"] = df["valor"].astype(float) / 100
            df.set_index("data", inplace=True)
            df.rename(columns={"valor": "selic"}, inplace=True)
            return df
        except Exception as e:
            logger.error("Erro ao buscar histórico Selic: {}", e)
            return pd.DataFrame()

    # ──────────────────────────────────────────────
    # Curva DI Futuro (B3)
    # ──────────────────────────────────────────────

    def get_di_futures(self) -> Optional[dict]:
        """
        Busca cotações dos contratos DI Futuro da B3.
        Usa o endpoint público de dados da B3.
        Retorna dict com vencimento -> taxa.
        """
        try:
            url = "https://www.b3.com.br/pt_br/market-data-e-indices/servicos-de-dados/market-data/cotacoes/taxas-referenciais-brl/taxa-di/"
            # Fallback: usar dados do BCB Focus
            return self._get_focus_di()
        except Exception as e:
            logger.error("Erro ao buscar curva DI: {}", e)
            return None

    def _get_focus_di(self) -> dict:
        """
        Busca expectativas de juros do Relatório Focus (BCB).
        Retorna taxa Selic esperada para próximas reuniões.
        """
        try:
            url = "https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata/ExpectativasMercadoAnuais"
            params = {
                "$filter": "Indicador eq 'Selic' and baseCalculo eq '0'",
                "$orderby": "Data desc",
                "$top": "20",
                "$format": "json",
                "$$skip": "0",
            }
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json().get("value", [])
            result = {}
            for item in data:
                year = item.get("DataReferencia")
                median = item.get("Mediana")
                if year and median:
                    result[year] = float(median) / 100
            logger.info("Expectativas Selic Focus carregadas: {} registros", len(result))
            return result
        except Exception as e:
            logger.error("Erro ao buscar Focus: {}", e)
            return {}

    # ──────────────────────────────────────────────
    # Classificação do Regime de Juros
    # ──────────────────────────────────────────────

    def get_rate_regime(
        self, current_selic: float, focus_1y: Optional[float] = None
    ) -> str:
        """
        Classifica o regime de juros para ativar estratégias macro.

        Returns:
            "high_stable"     → Selic alta e sem perspectiva de corte iminente
            "high_cut_expect" → Selic alta mas com expectativa de corte
            "easing"          → Ciclo de cortes em andamento
            "low_stable"      → Selic baixa e estável
        """
        if focus_1y is None:
            focus_1y = current_selic

        diff = current_selic - focus_1y

        if current_selic >= 0.12:
            if diff > 0.015:  # Corte esperado de mais de 1.5pp
                return "high_cut_expect"
            return "high_stable"
        elif diff > 0.005:
            return "easing"
        return "low_stable"

    # ──────────────────────────────────────────────
    # IPCA
    # ──────────────────────────────────────────────

    def get_ipca_12m(self) -> Optional[float]:
        """IPCA acumulado 12 meses (série 13522 BCB)."""
        try:
            # Série 433 = IPCA mensal
            start = (date.today() - timedelta(days=400)).strftime("%d/%m/%Y")
            url = self.BCB_BASE.format(serie=433)
            resp = requests.get(url, params={"formato": "json", "dataInicial": start}, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if len(data) >= 12:
                last_12 = [float(d["valor"]) for d in data[-12:]]
                cumulative = 1.0
                for m in last_12:
                    cumulative *= (1 + m / 100)
                result = cumulative - 1
                logger.info("IPCA 12m: {:.2f}%", result * 100)
                return result
        except Exception as e:
            logger.error("Erro ao buscar IPCA: {}", e)
        return None
