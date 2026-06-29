"""
Backtest Engine — Tradebot-UTIL.v2
──────────────────────────────────
Motor compartilhado de dados, benchmark e backtests.

Correção de benchmark:
    O UTIL_UNIVERSE abaixo representa a carteira teórica atual usada pelo projeto
    com 18 ativos, incluindo AXIA3 e AXIA6. Antes o benchmark sintético usava
    apenas 16 ativos, o que deixava o resultado divergente da composição real
    documentada em config/universe.yaml.
"""
from __future__ import annotations

import sys
import types
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock do MetaTrader5 para permitir backtest em Linux sem MT5 instalado.
mt5_mock = types.ModuleType("MetaTrader5")
for _attr in [
    "TIMEFRAME_M1", "TIMEFRAME_M5", "TIMEFRAME_M15", "TIMEFRAME_M30",
    "TIMEFRAME_H1", "TIMEFRAME_H4", "TIMEFRAME_D1", "TIMEFRAME_W1", "TIMEFRAME_MN1",
    "ORDER_TYPE_BUY", "ORDER_TYPE_SELL", "TRADE_ACTION_DEAL",
    "ORDER_TIME_GTC", "ORDER_FILLING_IOC",
]:
    setattr(mt5_mock, _attr, 0)
mt5_mock.TRADE_RETCODE_DONE = 10009
mt5_mock.initialize = lambda **kw: False
mt5_mock.shutdown = lambda: None
mt5_mock.account_info = lambda: None
mt5_mock.terminal_info = lambda: None
mt5_mock.symbol_info = lambda s: None
mt5_mock.symbol_select = lambda s, b: False
mt5_mock.copy_rates_from_pos = lambda *a, **kw: None
mt5_mock.symbol_info_tick = lambda s: None
mt5_mock.positions_get = lambda **kw: []
mt5_mock.history_deals_get = lambda *a: []
mt5_mock.order_send = lambda r: None
mt5_mock.last_error = lambda: (0, "OK")
sys.modules["MetaTrader5"] = mt5_mock

import numpy as np
import pandas as pd
import yfinance as yf

from src.utils.indicators import ema, macd


# Composição do Índice UTIL — Vigência Maio-Agosto/2026.
# Pesos em percentual. Usado como benchmark sintético comum entre main e infra-1.
UTIL_UNIVERSE = {
    "SBSP3": 19.999,
    "AXIA3": 17.291,
    "EQTL3": 11.431,
    "ENEV3": 10.708,
    "CPLE3": 10.318,
    "CMIG4": 5.422,
    "ENGI11": 3.786,
    "AXIA6": 2.708,
    "EGIE3": 2.637,
    "ISAE4": 2.587,
    "CSMG3": 2.440,
    "TAEE11": 2.104,
    "SAPR11": 2.003,
    "CPFE3": 2.070,
    "NEOE3": 1.509,
    "ALUP11": 1.262,
    "ORVR3": 0.862,
    "AURE3": 0.855,
}


@dataclass
class Trade:
    ticker: str
    strategy: str
    direction: str
    entry_date: pd.Timestamp
    exit_date: Optional[pd.Timestamp]
    entry_price: float
    exit_price: float = 0.0
    shares: int = 0
    capital_used: float = 0.0
    pnl: float = 0.0
    return_pct: float = 0.0
    exit_reason: str = ""


@dataclass
class BacktestResult:
    strategy: str
    trades: list[Trade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)
    benchmark_curve: pd.Series = field(default_factory=pd.Series)

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def wins(self) -> list[Trade]:
        return [t for t in self.trades if t.pnl > 0]

    @property
    def losses(self) -> list[Trade]:
        return [t for t in self.trades if t.pnl <= 0]

    @property
    def win_rate(self) -> float:
        return len(self.wins) / self.total_trades * 100 if self.total_trades else 0.0

    @property
    def total_return_pct(self) -> float:
        # Mantido por compatibilidade com o runner atual.
        return sum(t.return_pct for t in self.trades)

    @property
    def avg_win(self) -> float:
        return float(np.mean([t.return_pct for t in self.wins])) if self.wins else 0.0

    @property
    def avg_loss(self) -> float:
        return float(np.mean([t.return_pct for t in self.losses])) if self.losses else 0.0

    @property
    def profit_factor(self) -> float:
        gross_win = sum(t.pnl for t in self.wins)
        gross_loss = abs(sum(t.pnl for t in self.losses))
        return gross_win / gross_loss if gross_loss > 0 else float("inf")

    @property
    def max_drawdown(self) -> float:
        if self.equity_curve.empty:
            return 0.0
        peak = self.equity_curve.cummax()
        dd = (self.equity_curve - peak) / peak
        return float(dd.min() * 100)

    @property
    def sharpe_ratio(self) -> float:
        if self.equity_curve.empty or len(self.equity_curve) < 2:
            return 0.0
        daily_ret = self.equity_curve.pct_change().dropna()
        if daily_ret.std() == 0:
            return 0.0
        return float((daily_ret.mean() / daily_ret.std()) * np.sqrt(252))

    @property
    def benchmark_return_pct(self) -> float:
        if self.benchmark_curve.empty:
            return 0.0
        return float((self.benchmark_curve.iloc[-1] / self.benchmark_curve.iloc[0] - 1) * 100)

    @property
    def alpha(self) -> float:
        return self.total_return_pct - self.benchmark_return_pct


def download_data(tickers: list[str], start: str, end: str, verbose: bool = True) -> dict[str, pd.DataFrame]:
    if verbose:
        print(f"\n  Baixando {len(tickers)} ativos de {start} até {end}...")

    data: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        try:
            df = yf.Ticker(ticker + ".SA").history(start=start, end=end, auto_adjust=True, actions=False)
            if df.empty:
                if verbose:
                    print(f"    ✗ {ticker}: sem dados retornados")
                continue
            df.columns = [c.lower() for c in df.columns]
            if hasattr(df.index, "tz") and df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
            df = df[cols].dropna()
            if len(df) >= 50:
                data[ticker] = df
                if verbose:
                    print(f"    ✓ {ticker}: {len(df)} pregões")
            elif verbose:
                print(f"    ✗ {ticker}: dados insuficientes ({len(df)} barras)")
        except Exception as e:
            if verbose:
                print(f"    ✗ {ticker}: {e}")
    return data


def synthetic_util_benchmark(data: dict[str, pd.DataFrame], target_index: Optional[pd.Index] = None) -> pd.Series:
    """Constrói benchmark sintético fiel à carteira atual do UTIL.

    Usa os pesos oficiais em UTIL_UNIVERSE, normalizados apenas entre os ativos
    disponíveis no backtest. As séries de preço devem estar ajustadas por
    proventos (auto_adjust=True no yfinance), aproximando um índice de retorno total.
    """
    available = {t: df for t, df in data.items() if t in UTIL_UNIVERSE and "close" in df.columns}
    if not available:
        return pd.Series(dtype=float)

    closes = pd.concat({ticker: df["close"] for ticker, df in available.items()}, axis=1).sort_index()
    if target_index is not None and len(target_index) > 0:
        closes = closes.reindex(target_index).ffill().dropna(how="all")
    else:
        closes = closes.dropna(how="all")
    if closes.empty:
        return pd.Series(dtype=float)

    # Normaliza pesos usando somente ativos com dados disponíveis naquela execução.
    weights = pd.Series({ticker: UTIL_UNIVERSE[ticker] for ticker in closes.columns}, dtype=float)
    weights = weights / weights.sum()
    return (closes / closes.iloc[0]).mul(weights, axis=1).sum(axis=1).dropna()


def download_benchmark(start: str, end: str) -> pd.Series:
    """Mantido por compatibilidade. Prefira synthetic_util_benchmark(data)."""
    data = download_data(list(UTIL_UNIVERSE.keys()), start, end, verbose=False)
    return synthetic_util_benchmark(data)


def run_momentum_macro(
    data: dict[str, pd.DataFrame],
    capital: float = 100_000.0,
    ema_fast: int = 9,
    ema_mid: int = 21,
    ema_slow: int = 50,
    stop_pct: float = 0.03,
    assets: Optional[list[str]] = None,
) -> list[Trade]:
    assets = assets or ["SBSP3", "EQTL3", "ENEV3", "CPLE3"]
    trades: list[Trade] = []
    alloc = capital / len(assets)

    for ticker in assets:
        if ticker not in data:
            continue

        ohlcv = data[ticker]
        close = ohlcv["close"]
        e9 = ema(close, ema_fast)
        e21 = ema(close, ema_mid)
        e50 = ema(close, ema_slow)
        macd_line, sig_line, _ = macd(close)
        position = None

        for i in range(ema_slow + 5, len(ohlcv)):
            c = close.iloc[i]
            dt = ohlcv.index[i]
            if pd.isna(e9.iloc[i]) or pd.isna(e21.iloc[i]) or pd.isna(e50.iloc[i]):
                continue

            cross_up = e9.iloc[i] > e21.iloc[i] and e9.iloc[i - 1] <= e21.iloc[i - 1] and e21.iloc[i] > e50.iloc[i]
            macd_bull = macd_line.iloc[i] > sig_line.iloc[i]

            if position is None and cross_up and macd_bull:
                shares = int(alloc / c)
                if shares > 0:
                    position = {"entry_date": dt, "entry_price": c, "stop": e50.iloc[i], "shares": shares}
            elif position is not None:
                position["stop"] = max(position["stop"], e50.iloc[i] * 0.99)
                exit_price = None
                exit_reason = ""
                if c < position["stop"]:
                    exit_price, exit_reason = c, "trailing_stop"
                elif c < position["entry_price"] * (1 - stop_pct):
                    exit_price, exit_reason = c, "stop_loss"

                if exit_price is not None:
                    ep = position["entry_price"]
                    sh = position["shares"]
                    ret = (exit_price - ep) / ep
                    trades.append(Trade(
                        ticker=ticker,
                        strategy="momentum_macro",
                        direction="long",
                        entry_date=position["entry_date"],
                        exit_date=dt,
                        entry_price=ep,
                        exit_price=exit_price,
                        shares=sh,
                        capital_used=ep * sh,
                        pnl=ret * ep * sh,
                        return_pct=ret * 100,
                        exit_reason=exit_reason,
                    ))
                    position = None

        if position:
            c = close.iloc[-1]
            ep = position["entry_price"]
            sh = position["shares"]
            ret = (c - ep) / ep
            trades.append(Trade(
                ticker=ticker,
                strategy="momentum_macro",
                direction="long",
                entry_date=position["entry_date"],
                exit_date=ohlcv.index[-1],
                entry_price=ep,
                exit_price=c,
                shares=sh,
                capital_used=ep * sh,
                pnl=ret * ep * sh,
                return_pct=ret * 100,
                exit_reason="end_of_period",
            ))

    return trades


class BacktestEngine:
    def __init__(
        self,
        start: str = "2019-01-01",
        end: str = "2026-01-01",
        capital: float = 100_000.0,
        tickers: Optional[list[str]] = None,
    ):
        self.start = start
        self.end = end
        self.capital = capital
        self.tickers = tickers or list(UTIL_UNIVERSE.keys())
        self.data: dict[str, pd.DataFrame] = {}
        self.benchmark: pd.Series = pd.Series(dtype=float)

    def load_data(self, verbose: bool = True) -> None:
        self.data = download_data(self.tickers, self.start, self.end, verbose)
        self.benchmark = synthetic_util_benchmark(self.data)

    def run(self, strategies: Optional[list[str]] = None) -> dict[str, BacktestResult]:
        available = {
            "momentum_macro": lambda: run_momentum_macro(self.data, self.capital),
        }
        to_run = strategies or list(available.keys())
        results: dict[str, BacktestResult] = {}

        for name in to_run:
            if name not in available:
                print(f"  Estratégia '{name}' não encontrada.")
                continue
            print(f"\n  Executando: {name}...")
            trades = available[name]()
            result = BacktestResult(strategy=name, trades=trades)

            all_exits = [(t.exit_date, t.pnl) for t in trades if t.exit_date is not None]
            if all_exits:
                ordered = sorted(all_exits)
                equity = pd.Series(
                    [self.capital] + [pnl for _, pnl in ordered],
                    index=pd.Index([pd.Timestamp(self.start)] + [dt for dt, _ in ordered]),
                ).cumsum()
                equity.iloc[0] = self.capital
                result.equity_curve = equity

            result.benchmark_curve = self.benchmark
            results[name] = result
            print(f"    → {len(trades)} operações | Win rate: {result.win_rate:.1f}% | Retorno: {result.total_return_pct:+.2f}%")

        return results

    def summary_table(self, results: dict[str, BacktestResult]) -> pd.DataFrame:
        rows = []
        for name, r in results.items():
            bench_ret = r.benchmark_return_pct
            rows.append({
                "Estratégia": name,
                "Operações": r.total_trades,
                "Win Rate (%)": round(r.win_rate, 1),
                "Retorno (%)": round(r.total_return_pct, 2),
                "UTIL Benchmark": round(bench_ret, 2),
                "Alpha (%)": round(r.total_return_pct - bench_ret, 2),
                "Profit Factor": round(r.profit_factor, 2),
                "Max Drawdown (%)": round(r.max_drawdown, 2),
                "Sharpe": round(r.sharpe_ratio, 2),
                "Ganho Médio (%)": round(r.avg_win, 2),
                "Perda Média (%)": round(r.avg_loss, 2),
            })
        return pd.DataFrame(rows).set_index("Estratégia")
