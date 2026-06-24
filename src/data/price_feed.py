"""
Price Feed — Tradebot-UTIL.v2
══════════════════════════════

Feed de preços unificado com duas fontes:
  1. yfinance (primária — histórico ajustado por dividendos)
  2. MetaTrader 5 (fallback / dados intraday)

Responsabilidades:
  - Baixar e cachear histórico de preços dos 18 ativos UTIL
  - Construir o benchmark sintético UTIL
  - Prover preços de fechamento para a estratégia Active Momentum Tilt
  - Verificar liquidez mínima antes do rebalanceamento
"""
from __future__ import annotations

import time
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd
import numpy as np
from loguru import logger

try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    _YF_AVAILABLE = False
    logger.warning("yfinance não instalado — apenas MT5 disponível")

try:
    import MetaTrader5 as mt5
    _MT5_AVAILABLE = True
except ImportError:
    _MT5_AVAILABLE = False

from backtest.backtest_engine import UTIL_UNIVERSE, synthetic_util_benchmark


class PriceFeed:
    """
    Feed de preços para o universo UTIL.

    Prioridade de fonte:
        yfinance (histórico) → MT5 (preço atual do dia)

    Cache local em memória (TTL configurável).
    """

    CACHE_TTL_SECONDS = 3600  # 1 hora

    def __init__(self, config: dict):
        cfg = config.get("data", {})
        self.mt5_cfg: dict       = config.get("mt5", {})
        self.history_days: int   = int(cfg.get("price_history_days", 300))
        self.adjusted: bool      = bool(cfg.get("adjusted_prices", True))
        self.suffix: str         = str(cfg.get("yfinance_suffix", ".SA"))
        self.universe: list[str] = list(UTIL_UNIVERSE.keys())

        # Cache em memória: ticker → (timestamp, DataFrame)
        self._cache: dict[str, tuple[float, pd.DataFrame]] = {}
        self._benchmark_cache: Optional[tuple[float, pd.Series]] = None
        self._mt5_initialized: bool = False

        logger.info("PriceFeed inicializado | {} ativos | {} dias de histórico",
                    len(self.universe), self.history_days)

    # ─── Sessão MT5 ──────────────────────────────────────────────────────────

    def _ensure_mt5_session(self) -> bool:
        """
        Garante que o módulo MetaTrader5 esteja ligado ao terminal.

        Quando mt5.use_existing_session=true, usa initialize() sem credenciais para
        reaproveitar o MT5 já aberto/logado. Se mt5.path estiver preenchido, aponta
        para o executável correto caso o terminal ainda não esteja aberto.
        """
        if not _MT5_AVAILABLE:
            return False

        try:
            import MetaTrader5 as mt5  # noqa

            if self._mt5_initialized and mt5.terminal_info() is not None:
                return True

            # Se outra parte do bot já inicializou o MT5, apenas reaproveita.
            if mt5.terminal_info() is not None:
                self._mt5_initialized = True
                return True

            timeout = int(self.mt5_cfg.get("timeout", 60000))
            path = str(self.mt5_cfg.get("path", "") or "").strip()
            use_existing = bool(self.mt5_cfg.get("use_existing_session", True))

            if use_existing:
                kwargs = {"timeout": timeout}
                if path:
                    kwargs["path"] = path
                ok = mt5.initialize(**kwargs)
            else:
                kwargs = {
                    "login": int(self.mt5_cfg.get("login", 0)),
                    "password": str(self.mt5_cfg.get("password", "")),
                    "server": str(self.mt5_cfg.get("server", "")),
                    "timeout": timeout,
                }
                if path:
                    kwargs["path"] = path
                ok = mt5.initialize(**kwargs)

            if not ok:
                logger.debug("MT5 não inicializado no PriceFeed: {}", mt5.last_error())
                return False

            self._mt5_initialized = True
            return True
        except Exception as exc:
            logger.debug("Erro ao garantir sessão MT5 no PriceFeed: {}", exc)
            return False

    # ─── Download principal ────────────────────────────────────────────────

    def fetch_closes(
        self,
        tickers: Optional[list[str]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        use_cache: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """
        Baixa histórico OHLCV para os ativos do universo UTIL.

        Args:
            tickers:   Lista de tickers (padrão: universo completo)
            start:     Data inicial ISO (padrão: hoje - history_days)
            end:       Data final ISO (padrão: hoje)
            use_cache: Usar cache em memória

        Returns:
            dict[ticker] → DataFrame com colunas: open, high, low, close, volume
        """
        tickers = tickers or self.universe
        end = end or date.today().isoformat()
        start = start or (date.today() - timedelta(days=self.history_days)).isoformat()

        data: dict[str, pd.DataFrame] = {}

        for ticker in tickers:
            # Verifica cache
            if use_cache and ticker in self._cache:
                ts, df = self._cache[ticker]
                if time.time() - ts < self.CACHE_TTL_SECONDS:
                    data[ticker] = df
                    continue

            df = self._download_yfinance(ticker, start, end)
            if df is None or len(df) < 30:
                logger.warning("yfinance falhou para {} — tentando MT5", ticker)
                df = self._download_mt5(ticker, start, end)

            if df is not None and len(df) >= 30:
                self._cache[ticker] = (time.time(), df)
                data[ticker] = df
            else:
                logger.error("Sem dados para {} em nenhuma fonte", ticker)

        logger.info("PriceFeed: {} de {} ativos carregados", len(data), len(tickers))
        return data

    def _download_yfinance(
        self, ticker: str, start: str, end: str
    ) -> Optional[pd.DataFrame]:
        if not _YF_AVAILABLE:
            return None
        try:
            yfticker = ticker + self.suffix
            df = yf.Ticker(yfticker).history(
                start=start, end=end,
                auto_adjust=self.adjusted,
                actions=False,
            )
            if df.empty:
                return None
            # Normaliza
            df = df.copy()
            if hasattr(df.index, "tz") and df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            df.index = pd.to_datetime(df.index)
            df.columns = [c.lower() for c in df.columns]
            keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
            df = df[keep].dropna()
            return df.sort_index()
        except Exception as exc:
            logger.debug("yfinance erro {} | {}", ticker, exc)
            return None

    def _download_mt5(
        self, ticker: str, start: str, end: str
    ) -> Optional[pd.DataFrame]:
        """Baixa dados via MT5 usando a sessão existente quando disponível."""
        if not self._ensure_mt5_session():
            return None
        try:
            import MetaTrader5 as mt5  # noqa
            symbol = ticker  # MT5 usa ticker sem sufixo para B3
            mt5.symbol_select(symbol, True)
            rates = mt5.copy_rates_range(
                symbol,
                mt5.TIMEFRAME_D1,
                datetime.fromisoformat(start),
                datetime.fromisoformat(end),
            )
            if rates is None or len(rates) == 0:
                return None
            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df = df.set_index("time").rename(columns={
                "open": "open", "high": "high", "low": "low",
                "close": "close", "tick_volume": "volume",
            })
            keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
            return df[keep].dropna().sort_index()
        except Exception as exc:
            logger.debug("MT5 erro {} | {}", ticker, exc)
            return None

    # ─── Preço atual ────────────────────────────────────────────────────────

    def get_current_prices(self, tickers: Optional[list[str]] = None) -> dict[str, float]:
        """
        Retorna o último preço de fechamento disponível para cada ativo.

        Tenta MT5 primeiro (mais atualizado durante pregão), depois yfinance.
        """
        tickers = tickers or self.universe
        prices: dict[str, float] = {}

        for ticker in tickers:
            price = self._mt5_last_price(ticker)
            if price is None:
                price = self._yf_last_price(ticker)
            if price is not None:
                prices[ticker] = price
            else:
                logger.warning("Sem preço atual para {}", ticker)

        return prices

    def _mt5_last_price(self, ticker: str) -> Optional[float]:
        """Retorna último preço via MT5 reaproveitando a sessão existente."""
        if not self._ensure_mt5_session():
            return None
        try:
            import MetaTrader5 as mt5  # noqa
            mt5.symbol_select(ticker, True)
            tick = mt5.symbol_info_tick(ticker)
            if tick is None:
                return None
            candidates = (tick.last, tick.ask, tick.bid)
            for value in candidates:
                if value and value > 0:
                    return float(value)
            return None
        except Exception:
            return None

    def _yf_last_price(self, ticker: str) -> Optional[float]:
        if not _YF_AVAILABLE:
            return None
        try:
            info = yf.Ticker(ticker + self.suffix).fast_info
            return float(info.last_price) if info.last_price else None
        except Exception:
            return None

    # ─── Benchmark ──────────────────────────────────────────────────────────

    def get_benchmark(
        self,
        data: Optional[dict[str, pd.DataFrame]] = None,
        use_cache: bool = True,
    ) -> pd.Series:
        """
        Retorna o benchmark sintético UTIL.

        Se data for passado, usa diretamente. Caso contrário, baixa os dados.
        """
        if use_cache and self._benchmark_cache:
            ts, bm = self._benchmark_cache
            if time.time() - ts < self.CACHE_TTL_SECONDS:
                return bm

        if data is None:
            data = self.fetch_closes()

        bm = synthetic_util_benchmark(data)
        self._benchmark_cache = (time.time(), bm)
        return bm

    # ─── Closes alinhados ────────────────────────────────────────────────────

    def get_closes_df(
        self,
        data: Optional[dict[str, pd.DataFrame]] = None,
    ) -> pd.DataFrame:
        """
        Retorna DataFrame de fechamentos alinhados (date × ticker).
        Forward-fill de dados faltantes (máximo 5 dias).
        """
        if data is None:
            data = self.fetch_closes()

        if not data:
            return pd.DataFrame()

        closes = pd.concat(
            {t: df["close"] for t, df in data.items() if "close" in df.columns},
            axis=1,
        ).sort_index()

        # Forward fill limitado a 5 pregões
        closes = closes.ffill(limit=5).dropna(how="all")
        return closes

    # ─── Verificação de liquidez ──────────────────────────────────────────────

    def check_liquidity(
        self,
        tickers: list[str],
        data: Optional[dict[str, pd.DataFrame]] = None,
        min_avg_volume_k: float = 500.0,  # R$ mil de volume médio mínimo
        lookback_days: int = 20,
    ) -> dict[str, bool]:
        """
        Verifica se cada ativo tem liquidez mínima para operar.

        Critério: volume financeiro médio de 20 dias > min_avg_volume_k (R$ mil).
        """
        if data is None:
            data = self.fetch_closes()

        results: dict[str, bool] = {}
        for ticker in tickers:
            if ticker not in data:
                results[ticker] = False
                continue
            df = data[ticker].tail(lookback_days)
            if "close" not in df or "volume" not in df or len(df) < 5:
                results[ticker] = True  # assume OK se sem dados de volume
                continue
            avg_vol_fin = float((df["close"] * df["volume"]).mean()) / 1000  # em R$ mil
            ok = avg_vol_fin >= min_avg_volume_k
            if not ok:
                logger.warning(
                    "Liquidez baixa para {}: R$ {:.0f}k/dia (mínimo: R$ {:.0f}k)",
                    ticker, avg_vol_fin, min_avg_volume_k
                )
            results[ticker] = ok

        return results

    def clear_cache(self) -> None:
        """Limpa o cache de preços."""
        self._cache.clear()
        self._benchmark_cache = None
        logger.debug("Cache de preços limpo")
