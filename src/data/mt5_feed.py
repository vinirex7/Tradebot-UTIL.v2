"""
Feed de dados via MetaTrader 5.
Conecta ao MT5 instalado na máquina e busca dados de preços (OHLCV)
e informações de conta/portfólio.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Optional

import MetaTrader5 as mt5
import pandas as pd
from loguru import logger


TIMEFRAME_MAP = {
    "M1":  mt5.TIMEFRAME_M1,
    "M5":  mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1":  mt5.TIMEFRAME_H1,
    "H4":  mt5.TIMEFRAME_H4,
    "D1":  mt5.TIMEFRAME_D1,
    "W1":  mt5.TIMEFRAME_W1,
    "MN1": mt5.TIMEFRAME_MN1,
}


class MT5Feed:
    """Interface com o MetaTrader 5 para dados e execução."""

    def __init__(self, login: int, password: str, server: str, timeout: int = 60000):
        self.login = login
        self.password = password
        self.server = server
        self.timeout = timeout
        self._connected = False

    # ──────────────────────────────────────────────
    # Conexão
    # ──────────────────────────────────────────────

    def connect(self) -> bool:
        """Inicializa e autentica no MT5."""
        if not mt5.initialize(
            login=self.login,
            password=self.password,
            server=self.server,
            timeout=self.timeout,
        ):
            error = mt5.last_error()
            logger.error("Falha ao conectar ao MT5: {}", error)
            return False

        info = mt5.account_info()
        if info is None:
            logger.error("Não foi possível obter informações da conta.")
            mt5.shutdown()
            return False

        self._connected = True
        logger.info(
            "Conectado ao MT5 | Conta: {} | Servidor: {} | Saldo: R$ {:.2f}",
            info.login, info.server, info.balance
        )
        return True

    def disconnect(self) -> None:
        mt5.shutdown()
        self._connected = False
        logger.info("Desconectado do MT5.")

    def is_connected(self) -> bool:
        if not self._connected:
            return False
        return mt5.terminal_info() is not None

    def reconnect(self, max_attempts: int = 3, delay: int = 10) -> bool:
        """Tenta reconectar ao MT5."""
        for attempt in range(1, max_attempts + 1):
            logger.warning("Tentativa de reconexão {}/{}", attempt, max_attempts)
            if self.connect():
                return True
            time.sleep(delay)
        logger.error("Não foi possível reconectar após {} tentativas.", max_attempts)
        return False

    # ──────────────────────────────────────────────
    # Dados de Preço
    # ──────────────────────────────────────────────

    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        n_bars: int = 500,
        adjusted: bool = True,
    ) -> Optional[pd.DataFrame]:
        """
        Retorna OHLCV do símbolo no timeframe especificado.
        No MT5 da B3, o símbolo deve ter o sufixo correto (ex.: 'SBSP3').
        """
        if not self.is_connected():
            if not self.reconnect():
                return None

        tf = TIMEFRAME_MAP.get(timeframe.upper())
        if tf is None:
            logger.error("Timeframe inválido: {}", timeframe)
            return None

        # Ativa o símbolo se necessário
        if not mt5.symbol_select(symbol, True):
            logger.error("Símbolo não encontrado no MT5: {}", symbol)
            return None

        rates = mt5.copy_rates_from_pos(symbol, tf, 0, n_bars)
        if rates is None or len(rates) == 0:
            logger.error("Sem dados para {} [{}]", symbol, timeframe)
            return None

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        df.rename(columns={
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "tick_volume": "volume",
        }, inplace=True)
        df = df[["open", "high", "low", "close", "volume"]].copy()

        logger.debug("OHLCV {} [{}]: {} barras carregadas", symbol, timeframe, len(df))
        return df

    def get_last_price(self, symbol: str) -> Optional[float]:
        """Retorna o último preço de fechamento."""
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.error("Sem tick para: {}", symbol)
            return None
        return tick.last

    def get_bid_ask(self, symbol: str) -> tuple[float, float] | None:
        """Retorna (bid, ask) para calcular spread."""
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return None
        return tick.bid, tick.ask

    # ──────────────────────────────────────────────
    # Informações de Conta e Portfólio
    # ──────────────────────────────────────────────

    def get_account_info(self) -> dict:
        """Retorna informações da conta."""
        info = mt5.account_info()
        if info is None:
            return {}
        return {
            "balance": info.balance,
            "equity": info.equity,
            "margin": info.margin,
            "free_margin": info.margin_free,
            "profit": info.profit,
            "leverage": info.leverage,
        }

    def get_positions(self) -> pd.DataFrame:
        """Retorna posições abertas."""
        positions = mt5.positions_get()
        if not positions:
            return pd.DataFrame()
        df = pd.DataFrame([p._asdict() for p in positions])
        df["time"] = pd.to_datetime(df["time"], unit="s")
        return df

    def get_history_deals(self, from_date: datetime, to_date: datetime) -> pd.DataFrame:
        """Retorna histórico de negócios no período."""
        deals = mt5.history_deals_get(from_date, to_date)
        if not deals:
            return pd.DataFrame()
        df = pd.DataFrame([d._asdict() for d in deals])
        df["time"] = pd.to_datetime(df["time"], unit="s")
        return df
