"""
Backtest standalone — UTIL Best Assets Rotation
───────────────────────────────────────────────
Roda a estratégia dinâmica:
  1. Pega todos os ativos do UTIL.
  2. Calcula score por força relativa, momentum, tendência e risco.
  3. Opera os melhores ativos elegíveis, sem ficar preso a quatro.
  4. Vende quem sai da seleção ou perde tendência.
  5. Compra/substitui os novos selecionados.

Uso:
  python backtest/run_top4_rotation.py --start 2022-01-01 --end 2026-06-23 --capital 100000 --top-n 0 --max-positions 8 --csv
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest.backtest_engine import UTIL_UNIVERSE, download_benchmark, download_data
from src.strategies.top4_rotation import Top4UTILRotationStrategy


@dataclass
class ClosedTrade:
    ticker: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    shares: int
    pnl: float
    return_pct: float
    reason: str


def equity_value(cash: float, positions: dict, prices: pd.Series) -> float:
    value = cash
    for ticker, pos in positions.items():
        px = prices.get(ticker)
        if pd.notna(px):
            value += pos["shares"] * float(px)
    return float(value)


def run_backtest(
    start: str,
    end: str,
    capital: float,
    cadence: str = "weekly",
    top_n: int = 0,
    max_positions: int = 8,
    fee_pct: float = 0.0005,
    verbose: bool = True,
) -> tuple[pd.Series, list[ClosedTrade], pd.Series]:
    tickers = list(UTIL_UNIVERSE.keys())
    data = download_data(tickers, start, end, verbose=verbose)
    benchmark = download_benchmark(start, end)

    closes = pd.concat(
        {ticker: df["close"] for ticker, df in data.items() if "close" in df.columns},
        axis=1,
    ).dropna(how="all")

    active_slots = top_n if top_n > 0 else max_positions
    active_slots = max(1, active_slots)

    strategy = Top4UTILRotationStrategy(
        universe=[t for t in tickers if t in data],
        top_n=top_n,
        max_positions=max_positions,
        rebalance_frequency=cadence,
        weekly_rebalance_day="monday",
        lookback_short=63,
        lookback_mid=126,
        lookback_long=252,
        trend_ema=50,
        vol_lookback=63,
        min_score=-0.10,
        exit_score=-0.75,
        hard_stop_pct=0.05,
        max_position_pct=1 / active_slots,
    )

    cash = float(capital)
    positions: dict[str, dict] = {}
    trades: list[ClosedTrade] = []
    equity_points: list[tuple[pd.Timestamp, float]] = []

    min_i = strategy.min_bars
    for i in range(min_i, len(closes)):
        dt = closes.index[i]
        price_row = closes.iloc[i]
        hist = {ticker: df.loc[:dt].copy() for ticker, df in data.items() if df.index[0] <= dt}
        current_positions = {
            ticker: {"entry_price": pos["entry_price"], "shares": pos["shares"], "strategy": "top4_rotation"}
            for ticker, pos in positions.items()
        }

        rebalance_due = cadence == "daily" or (cadence == "weekly" and dt.strftime("%A").lower() == "monday")
        plan = strategy.analyze_universe(hist, current_positions=current_positions, force_rebalance=False)

        for ticker in list(plan.sell_tickers):
            if ticker not in positions or pd.isna(price_row.get(ticker)):
                continue
            px = float(price_row[ticker])
            pos = positions.pop(ticker)
            gross = px * pos["shares"]
            fee = gross * fee_pct
            cash += gross - fee
            pnl = (px - pos["entry_price"]) * pos["shares"] - fee
            ret = (px / pos["entry_price"] - 1) * 100
            trades.append(ClosedTrade(ticker, pos["entry_date"], dt, pos["entry_price"], px, pos["shares"], pnl, ret, "rotation_exit"))

        if rebalance_due:
            selected_count = max(1, len(plan.top_tickers))
            eq = equity_value(cash, positions, price_row)
            target_capital = eq / selected_count
            for signal in plan.buy_signals:
                ticker = signal.ticker
                if ticker in positions or pd.isna(price_row.get(ticker)):
                    continue
                px = float(price_row[ticker])
                if px <= 0:
                    continue
                shares = int(target_capital / px)
                if not ticker.endswith("11"):
                    shares = (shares // 100) * 100
                if shares <= 0:
                    continue
                cost = shares * px
                fee = cost * fee_pct
                if cost + fee > cash:
                    continue
                cash -= cost + fee
                positions[ticker] = {
                    "shares": shares,
                    "entry_price": px,
                    "entry_date": dt,
                }

        equity_points.append((dt, equity_value(cash, positions, price_row)))

    if not closes.empty:
        dt = closes.index[-1]
        price_row = closes.iloc[-1]
        for ticker, pos in list(positions.items()):
            if pd.isna(price_row.get(ticker)):
                continue
            px = float(price_row[ticker])
            gross = px * pos["shares"]
            fee = gross * fee_pct
            cash += gross - fee
            pnl = (px - pos["entry_price"]) * pos["shares"] - fee
            ret = (px / pos["entry_price"] - 1) * 100
            trades.append(ClosedTrade(ticker, pos["entry_date"], dt, pos["entry_price"], px, pos["shares"], pnl, ret, "end_of_period"))
        positions.clear()
        equity_points.append((dt, cash))

    equity = pd.Series([v for _, v in equity_points], index=[d for d, _ in equity_points]).sort_index()
    return equity, trades, benchmark


def print_summary(equity: pd.Series, trades: list[ClosedTrade], benchmark: pd.Series, capital: float) -> None:
    total_return = (equity.iloc[-1] / capital - 1) * 100 if not equity.empty else 0.0
    daily = equity.pct_change().dropna()
    sharpe = (daily.mean() / daily.std()) * np.sqrt(252) if len(daily) > 5 and daily.std() > 0 else 0.0
    dd = ((equity / equity.cummax()) - 1).min() * 100 if not equity.empty else 0.0
    bench_ret = (benchmark.iloc[-1] / benchmark.iloc[0] - 1) * 100 if benchmark is not None and not benchmark.empty else 0.0
    wins = [t for t in trades if t.pnl > 0]
    win_rate = len(wins) / len(trades) * 100 if trades else 0.0

    print("\n════════════════════════════════════════════════════════════")
    print("  UTIL Best Assets Rotation — Resultado")
    print("════════════════════════════════════════════════════════════")
    print(f"  Capital inicial : R$ {capital:,.2f}")
    print(f"  Capital final   : R$ {equity.iloc[-1]:,.2f}" if not equity.empty else "  Capital final   : n/d")
    print(f"  Retorno bot     : {total_return:+.2f}%")
    print(f"  Benchmark UTIL  : {bench_ret:+.2f}%")
    print(f"  Alpha           : {total_return - bench_ret:+.2f}%")
    print(f"  Max Drawdown    : {dd:.2f}%")
    print(f"  Sharpe          : {sharpe:.2f}")
    print(f"  Operações       : {len(trades)}")
    print(f"  Win rate        : {win_rate:.1f}%")


def save_csv(equity: pd.Series, trades: list[ClosedTrade]) -> None:
    out_dir = Path("logs")
    out_dir.mkdir(exist_ok=True)
    equity.to_csv(out_dir / "top4_rotation_equity.csv", header=["equity"])
    pd.DataFrame([vars(t) for t in trades]).to_csv(out_dir / "top4_rotation_trades.csv", index=False)
    print("\n  Arquivos salvos:")
    print("  - logs/top4_rotation_equity.csv")
    print("  - logs/top4_rotation_trades.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest UTIL Best Assets Rotation via yfinance")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2026-06-23")
    parser.add_argument("--capital", type=float, default=100000.0)
    parser.add_argument("--cadence", choices=["daily", "weekly"], default="weekly")
    parser.add_argument("--top-n", type=int, default=0, help="0 = todos os elegíveis até max-positions")
    parser.add_argument("--max-positions", type=int, default=8)
    parser.add_argument("--csv", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    equity, trades, benchmark = run_backtest(
        start=args.start,
        end=args.end,
        capital=args.capital,
        cadence=args.cadence,
        top_n=args.top_n,
        max_positions=args.max_positions,
        verbose=not args.quiet,
    )
    print_summary(equity, trades, benchmark, args.capital)
    if args.csv:
        save_csv(equity, trades)


if __name__ == "__main__":
    main()
