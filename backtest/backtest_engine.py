"""
Backtest Engine — Tradebot-UTIL.v2
────────────────────────────────────
Motor de backtest completo para todas as 5 estratégias.
Usa yfinance para dados históricos ajustados por dividendos.
100% compatível com Linux (sem MetaTrader5).

Universo: 18 ações do UTIL B3 (sufixo .SA no Yahoo Finance)
Benchmark: UTLL11 (ETF que replica o UTIL)
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock do MetaTrader5 para rodar sem o MT5 instalado
import types
mt5_mock = types.ModuleType("MetaTrader5")
for _attr in [
    "TIMEFRAME_M1","TIMEFRAME_M5","TIMEFRAME_M15","TIMEFRAME_M30",
    "TIMEFRAME_H1","TIMEFRAME_H4","TIMEFRAME_D1","TIMEFRAME_W1","TIMEFRAME_MN1",
    "ORDER_TYPE_BUY","ORDER_TYPE_SELL","TRADE_ACTION_DEAL",
    "ORDER_TIME_GTC","ORDER_FILLING_IOC",
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
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Optional

from src.utils.indicators import bollinger_bands, rsi, ema, macd, spread_zscore
from src.risk.risk_manager import RiskManager, TradeSignal


# ─────────────────────────────────────────────────────────
# Universo UTIL (composição Mai–Ago 2026)
# ─────────────────────────────────────────────────────────

UTIL_UNIVERSE = {
    "SBSP3":  19.999,
    "EQTL3":  11.431,
    "ENEV3":  10.708,
    "CPLE3":  10.318,
    "CMIG4":   5.422,
    "ENGI11":  3.786,
    "EGIE3":   2.637,
    "ISAE4":   2.587,
    "CSMG3":   2.440,
    "TAEE11":  2.104,
    "SAPR11":  2.003,
    "CPFE3":   2.070,
    "NEOE3":   1.509,
    "ALUP11":  1.262,
    "ORVR3":   0.862,
    "AURE3":   0.855,
}

# AXIA3/AXIA6 excluídos — listagem recente, sem histórico longo no Yahoo


# ─────────────────────────────────────────────────────────
# Estruturas de dados
# ─────────────────────────────────────────────────────────

@dataclass
class Trade:
    ticker: str
    strategy: str
    direction: str          # "long" | "short"
    entry_date: pd.Timestamp
    exit_date: Optional[pd.Timestamp]
    entry_price: float
    exit_price: float = 0.0
    shares: int = 0
    capital_used: float = 0.0
    pnl: float = 0.0
    return_pct: float = 0.0
    exit_reason: str = ""   # "take_profit" | "stop_loss" | "signal" | "end_of_period"


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
        return len(self.wins) / self.total_trades * 100 if self.total_trades else 0

    @property
    def total_return_pct(self) -> float:
        return sum(t.return_pct for t in self.trades)

    @property
    def avg_win(self) -> float:
        return np.mean([t.return_pct for t in self.wins]) if self.wins else 0

    @property
    def avg_loss(self) -> float:
        return np.mean([t.return_pct for t in self.losses]) if self.losses else 0

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
        return dd.min() * 100

    @property
    def sharpe_ratio(self) -> float:
        if self.equity_curve.empty or len(self.equity_curve) < 2:
            return 0.0
        daily_ret = self.equity_curve.pct_change().dropna()
        if daily_ret.std() == 0:
            return 0.0
        return (daily_ret.mean() / daily_ret.std()) * np.sqrt(252)

    @property
    def benchmark_return_pct(self) -> float:
        if self.benchmark_curve.empty:
            return 0.0
        return (self.benchmark_curve.iloc[-1] / self.benchmark_curve.iloc[0] - 1) * 100

    @property
    def alpha(self) -> float:
        return self.total_return_pct - self.benchmark_return_pct


# ─────────────────────────────────────────────────────────
# Download de dados
# ─────────────────────────────────────────────────────────

def download_data(
    tickers: list[str],
    start: str,
    end: str,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Baixa OHLCV do Yahoo Finance (ajustado por dividendos e splits).
    Baixa um ticker por vez para evitar erro de timezone com .SA.
    """
    if verbose:
        print(f"\n  Baixando {len(tickers)} ativos de {start} até {end}...")

    data = {}
    for ticker in tickers:
        yf_t = ticker + ".SA"
        try:
            obj = yf.Ticker(yf_t)
            df = obj.history(
                start=start,
                end=end,
                auto_adjust=True,
                actions=False,
            )

            if df.empty:
                if verbose:
                    print(f"    ✗ {ticker}: sem dados retornados")
                continue

            df.columns = [c.lower() for c in df.columns]

            # Remover timezone do índice (compatibilidade)
            if hasattr(df.index, 'tz') and df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            # Garantir colunas corretas
            cols_needed = ["open", "high", "low", "close", "volume"]
            cols_available = [c for c in cols_needed if c in df.columns]
            df = df[cols_available].dropna()

            if len(df) >= 50:
                data[ticker] = df
                if verbose:
                    print(f"    ✓ {ticker}: {len(df)} pregões")
            else:
                if verbose:
                    print(f"    ✗ {ticker}: dados insuficientes ({len(df)} barras)")

        except Exception as e:
            if verbose:
                print(f"    ✗ {ticker}: {e}")

    return data


def download_benchmark(start: str, end: str) -> pd.Series:
    """Baixa UTLL11 como benchmark do UTIL."""
    try:
        obj = yf.Ticker("UTLL11.SA")
        df = obj.history(start=start, end=end, auto_adjust=True, actions=False)
        if df.empty:
            return pd.Series()
        if hasattr(df.index, 'tz') and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.columns = [c.lower() for c in df.columns]
        return df["close"].dropna()
    except Exception:
        return pd.Series()



# ─────────────────────────────────────────────────────────
# Estratégia 2: Momentum Macro — v1 restaurado
#
# A v2 com trailing stop + TP 15% cortou cedo demais (+114% vs +277%).
# Restaurando a lógica original da v1: stop na EMA50 da entrada,
# sem take profit fixo (deixa o trade correr enquanto tendência válida).
# ─────────────────────────────────────────────────────────

def run_momentum_macro(
    data: dict[str, pd.DataFrame],
    capital: float = 100_000.0,
    ema_fast: int = 9,
    ema_mid: int = 21,
    ema_slow: int = 50,
    stop_pct: float = 0.03,          # Stop máximo 3% abaixo da entrada
    assets: Optional[list[str]] = None,
) -> list[Trade]:
    """
    Backtest de Momentum Macro — v1 restaurado.

    Lógica: EMA9 cruza acima de EMA21, com EMA21 > EMA50 e MACD bullish.
    Stop: EMA50 dinâmica (sobe junto com o mercado, nunca desce).
    Sem take profit fixo: deixa o trade correr enquanto a tendência for válida.
    """
    assets = assets or ["SBSP3", "EQTL3", "ENEV3", "CPLE3"]
    trades = []
    alloc  = capital / len(assets)

    for ticker in assets:
        if ticker not in data:
            continue

        ohlcv     = data[ticker]
        close     = ohlcv["close"]
        e9        = ema(close, ema_fast)
        e21       = ema(close, ema_mid)
        e50       = ema(close, ema_slow)
        macd_line, sig_line, _ = macd(close)

        position = None

        for i in range(ema_slow + 5, len(ohlcv)):
            c  = close.iloc[i]
            dt = ohlcv.index[i]

            if pd.isna(e9.iloc[i]) or pd.isna(e50.iloc[i]):
                continue

            cross_up  = (e9.iloc[i]  > e21.iloc[i]
                         and e9.iloc[i-1] <= e21.iloc[i-1]
                         and e21.iloc[i] > e50.iloc[i])
            macd_bull = macd_line.iloc[i] > sig_line.iloc[i]

            if position is None and cross_up and macd_bull:
                shares = int(alloc / c)
                if shares > 0:
                    position = {
                        "entry_date":  dt,
                        "entry_price": c,
                        "stop":        e50.iloc[i],   # stop na EMA50
                        "shares":      shares,
                    }

            elif position is not None:
                # Stop sobe com a EMA50, nunca desce
                new_stop = max(position["stop"], e50.iloc[i] * 0.99)
                position["stop"] = new_stop

                exit_price  = None
                exit_reason = ""

                if c < position["stop"]:
                    exit_price, exit_reason = c, "trailing_stop"
                elif c < position["entry_price"] * (1 - stop_pct):
                    exit_price, exit_reason = c, "stop_loss"

                if exit_price:
                    ep  = position["entry_price"]
                    sh  = position["shares"]
                    ret = (exit_price - ep) / ep
                    trades.append(Trade(
                        ticker=ticker, strategy="momentum_macro",
                        direction="long",
                        entry_date=position["entry_date"], exit_date=dt,
                        entry_price=ep, exit_price=exit_price,
                        shares=sh, capital_used=ep * sh,
                        pnl=ret * ep * sh, return_pct=ret * 100,
                        exit_reason=exit_reason,
                    ))
                    position = None

        if position:
            c   = close.iloc[-1]
            ep  = position["entry_price"]
            sh  = position["shares"]
            ret = (c - ep) / ep
            trades.append(Trade(
                ticker=ticker, strategy="momentum_macro",
                direction="long",
                entry_date=position["entry_date"], exit_date=ohlcv.index[-1],
                entry_price=ep, exit_price=c,
                shares=sh, capital_used=ep * sh,
                pnl=ret * ep * sh, return_pct=ret * 100,
                exit_reason="end_of_period",
            ))

    return trades

def run_pair_trading(
    data: dict[str, pd.DataFrame],
    capital: float = 100_000.0,
    lookback: int = 90,              # v2: janela maior (era 60)
    z_entry: float = 2.5,            # v2: threshold mais seletivo (era 2.0)
    z_exit: float = 0.3,             # v2: saída mais cedo (era 0.5)
    stop_pct: float = 0.06,          # v2: stop mais largo (era 4%)
    cooldown_days: int = 5,          # v2: pausa após stop loss
    pairs: Optional[list[tuple]] = None,
) -> list[Trade]:
    """
    Backtest de Pair Trading com z-score do spread OLS — v2.

    Melhorias vs v1:
    - z_entry 2.5: apenas desvios mais extremos (reduz overtrading)
    - Lookback 90 dias: relação histórica mais robusta
    - Cooldown após stop: evita reentrada imediata em par problemático
    - z_exit 0.3: sai antes do zero para garantir lucro
    """
    pairs = pairs or [("EQTL3", "TAEE11"), ("SBSP3", "ENEV3")]
    trades = []
    alloc = capital / len(pairs) / 2

    for tk_a, tk_b in pairs:
        if tk_a not in data or tk_b not in data:
            continue

        close_a = data[tk_a]["close"]
        close_b = data[tk_b]["close"]
        aligned = pd.concat([close_a.rename(tk_a), close_b.rename(tk_b)], axis=1).dropna()

        if len(aligned) < lookback + 10:
            continue

        z = spread_zscore(aligned[tk_a], aligned[tk_b], window=lookback)
        position = None
        cooldown_remaining = 0

        for i in range(lookback + 5, len(aligned)):
            dt     = aligned.index[i]
            curr_z = z.iloc[i]
            if pd.isna(curr_z):
                continue

            if cooldown_remaining > 0:
                cooldown_remaining -= 1
                continue

            price_a = aligned[tk_a].iloc[i]
            price_b = aligned[tk_b].iloc[i]

            if position is None and abs(curr_z) >= z_entry:
                if curr_z > 0:
                    long_tk, short_tk = tk_b, tk_a
                    long_p,  short_p  = price_b, price_a
                else:
                    long_tk, short_tk = tk_a, tk_b
                    long_p,  short_p  = price_a, price_b

                sh_long  = int(alloc / long_p)  if long_p  > 0 else 0
                sh_short = int(alloc / short_p) if short_p > 0 else 0

                if sh_long > 0 and sh_short > 0:
                    position = {
                        "entry_date": dt, "z_entry": curr_z,
                        "long_tk": long_tk, "short_tk": short_tk,
                        "long_entry": long_p, "short_entry": short_p,
                        "sh_long": sh_long, "sh_short": sh_short,
                    }

            elif position is not None:
                curr_long_p  = aligned[position["long_tk"]].iloc[i]
                curr_short_p = aligned[position["short_tk"]].iloc[i]

                long_ret     = (curr_long_p  - position["long_entry"])  / position["long_entry"]
                short_ret    = (position["short_entry"] - curr_short_p) / position["short_entry"]
                combined_ret = (long_ret + short_ret) / 2

                exit_flag = abs(curr_z) <= z_exit
                stop_flag = combined_ret < -stop_pct

                if exit_flag or stop_flag:
                    pnl_long  = long_ret  * position["long_entry"]  * position["sh_long"]
                    pnl_short = short_ret * position["short_entry"] * position["sh_short"]

                    trades.append(Trade(
                        ticker=f"{tk_a}/{tk_b}",
                        strategy="pair_trading",
                        direction="long/short",
                        entry_date=position["entry_date"], exit_date=dt,
                        entry_price=position["long_entry"],
                        exit_price=curr_long_p,
                        shares=position["sh_long"],
                        capital_used=alloc * 2,
                        pnl=pnl_long + pnl_short,
                        return_pct=combined_ret * 100,
                        exit_reason="take_profit" if exit_flag else "stop_loss",
                    ))
                    position = None
                    if stop_flag:
                        cooldown_remaining = cooldown_days

    return trades


# ─────────────────────────────────────────────────────────
# Estratégia 4: Captura de Dividendos — v2
# Mudanças: threshold de queda 2.5% (era 0.8%), máx 1 op/ativo/trimestre,
#           janela mínima de 20 dias entre operações no mesmo ativo
# ─────────────────────────────────────────────────────────
class BacktestEngine:
    """Orquestra o backtest de todas as estratégias."""

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
        self.benchmark: pd.Series = pd.Series()

    def load_data(self, verbose: bool = True) -> None:
        self.data = download_data(self.tickers, self.start, self.end, verbose)
        self.benchmark = download_benchmark(self.start, self.end)

    def run(self, strategies: Optional[list[str]] = None) -> dict[str, BacktestResult]:
        """
        Executa os backtests.
        strategies: lista de nomes ou None para todas.
        """
        available = {
            "momentum_macro":      lambda: run_momentum_macro(self.data, self.capital * 0.60),
            "pair_trading":        lambda: run_pair_trading(self.data, self.capital * 0.40),

        }

        to_run = strategies or list(available.keys())
        results = {}

        for name in to_run:
            if name not in available:
                print(f"  Estratégia '{name}' não encontrada.")
                continue
            print(f"\n  Executando: {name}...")
            trades = available[name]()
            result = BacktestResult(strategy=name, trades=trades)

            # Construir equity curve simples
            if trades:
                all_exits = [(t.exit_date, t.pnl) for t in trades if t.exit_date is not None]
                if all_exits:
                    equity = pd.Series(
                        [self.capital] + [pnl for _, pnl in sorted(all_exits)],
                        index=pd.Index(
                            [pd.Timestamp(self.start)]
                            + [dt for dt, _ in sorted(all_exits)]
                        ),
                    ).cumsum()
                    equity.iloc[0] = self.capital
                    result.equity_curve = equity

            result.benchmark_curve = self.benchmark
            results[name] = result
            print(f"    → {len(trades)} operações | Win rate: {result.win_rate:.1f}% | "
                  f"Retorno: {result.total_return_pct:+.2f}%")

        return results

    def summary_table(self, results: dict[str, BacktestResult]) -> pd.DataFrame:
        """Gera tabela resumo comparativa."""
        rows = []
        for name, r in results.items():
            bench_ret = r.benchmark_return_pct
            rows.append({
                "Estratégia":      name,
                "Operações":       r.total_trades,
                "Win Rate (%)":    round(r.win_rate, 1),
                "Retorno (%)":     round(r.total_return_pct, 2),
                "UTIL Benchmark":  round(bench_ret, 2),
                "Alpha (%)":       round(r.total_return_pct - bench_ret, 2),
                "Profit Factor":   round(r.profit_factor, 2),
                "Max Drawdown (%)":round(r.max_drawdown, 2),
                "Sharpe":          round(r.sharpe_ratio, 2),
                "Ganho Médio (%)": round(r.avg_win, 2),
                "Perda Média (%)": round(r.avg_loss, 2),
            })
        return pd.DataFrame(rows).set_index("Estratégia")
