"""
Backtest Runner — Tradebot-UTIL.v2
────────────────────────────────────
Executa backtests das estratégias usando dados históricos do Yahoo Finance
(ajustados por dividendos) e calcula métricas de desempenho vs UTIL.

Uso:
    python scripts/run_backtest.py --strategy mean_reversion --start 2019-01-01 --end 2026-01-01
    python scripts/run_backtest.py --strategy all --start 2019-01-01 --end 2026-01-01
"""
import argparse
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False
    print("[AVISO] yfinance não instalado. Execute: pip install yfinance")

from src.strategies import (
    MeanReversionStrategy,
    MomentumMacroStrategy,
    PairTradingStrategy,
)
from src.risk.risk_manager import RiskManager, TradeSignal


# ─────────────────────────────────────────────────────────
# Funções de dados históricos
# ─────────────────────────────────────────────────────────

def download_data(tickers: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    """Baixa dados históricos do Yahoo Finance (sufixo .SA para B3)."""
    if not YF_AVAILABLE:
        return {}

    data = {}
    for ticker in tickers:
        yf_ticker = ticker + ".SA"
        try:
            df = yf.download(
                yf_ticker,
                start=start,
                end=end,
                auto_adjust=True,   # Ajuste por dividendos e splits
                progress=False,
            )
            if not df.empty:
                df.columns = [c.lower() for c in df.columns]
                df.rename(columns={"adj close": "close"}, inplace=True)
                data[ticker] = df[["open", "high", "low", "close", "volume"]]
                print(f"  ✓ {ticker}: {len(df)} barras")
            else:
                print(f"  ✗ {ticker}: sem dados")
        except Exception as e:
            print(f"  ✗ {ticker}: erro — {e}")
    return data


# ─────────────────────────────────────────────────────────
# Backtest de Reversão à Média
# ─────────────────────────────────────────────────────────

def backtest_mean_reversion(
    data: dict[str, pd.DataFrame],
    initial_capital: float = 100_000.0,
    stop_loss: float = 0.025,
) -> pd.DataFrame:
    """Backtest simples da estratégia de reversão à média."""
    strategy = MeanReversionStrategy(stop_loss_pct=stop_loss)
    trades = []
    capital = initial_capital

    for ticker in strategy.assets:
        if ticker not in data:
            continue

        ohlcv = data[ticker]
        position = None

        for i in range(50, len(ohlcv)):
            window = ohlcv.iloc[:i]
            signal = strategy.analyze(ticker, window)

            if position is None and signal is not None:
                position = {
                    "ticker": ticker,
                    "entry_date": ohlcv.index[i],
                    "entry_price": signal.entry_price,
                    "direction": signal.direction,
                    "stop": signal.stop_loss_price,
                    "target": signal.take_profit_price,
                }

            elif position is not None:
                current_price = ohlcv["close"].iloc[i]
                exit_trade = False
                exit_reason = ""

                # Stop loss
                if position["direction"] == "long" and current_price <= position["stop"]:
                    exit_trade, exit_reason = True, "stop_loss"
                elif position["direction"] == "short" and current_price >= position["stop"]:
                    exit_trade, exit_reason = True, "stop_loss"

                # Take profit
                if position["direction"] == "long" and current_price >= position["target"]:
                    exit_trade, exit_reason = True, "take_profit"
                elif position["direction"] == "short" and current_price <= position["target"]:
                    exit_trade, exit_reason = True, "take_profit"

                if exit_trade:
                    if position["direction"] == "long":
                        ret = (current_price - position["entry_price"]) / position["entry_price"]
                    else:
                        ret = (position["entry_price"] - current_price) / position["entry_price"]

                    trades.append({
                        "ticker": ticker,
                        "entry_date": position["entry_date"],
                        "exit_date": ohlcv.index[i],
                        "direction": position["direction"],
                        "entry_price": position["entry_price"],
                        "exit_price": current_price,
                        "return_pct": ret * 100,
                        "exit_reason": exit_reason,
                    })
                    position = None

    return pd.DataFrame(trades)


# ─────────────────────────────────────────────────────────
# Métricas de desempenho
# ─────────────────────────────────────────────────────────

def calculate_metrics(trades_df: pd.DataFrame, initial_capital: float = 100_000.0) -> dict:
    """Calcula métricas de desempenho do backtest."""
    if trades_df.empty:
        return {"erro": "Nenhuma operação encontrada"}

    wins = trades_df[trades_df["return_pct"] > 0]
    losses = trades_df[trades_df["return_pct"] <= 0]

    total_return = trades_df["return_pct"].sum()
    win_rate = len(wins) / len(trades_df) * 100

    avg_win = wins["return_pct"].mean() if not wins.empty else 0
    avg_loss = losses["return_pct"].mean() if not losses.empty else 0
    profit_factor = abs(wins["return_pct"].sum() / losses["return_pct"].sum()) if not losses.empty else float("inf")

    return {
        "total_operacoes": len(trades_df),
        "win_rate_pct": round(win_rate, 2),
        "retorno_total_pct": round(total_return, 2),
        "ganho_medio_pct": round(avg_win, 2),
        "perda_media_pct": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 2),
        "stop_losses": len(trades_df[trades_df["exit_reason"] == "stop_loss"]),
        "take_profits": len(trades_df[trades_df["exit_reason"] == "take_profit"]),
        "melhor_trade_pct": round(trades_df["return_pct"].max(), 2),
        "pior_trade_pct": round(trades_df["return_pct"].min(), 2),
    }


# ─────────────────────────────────────────────────────────
# Comparativo com UTIL (benchmark)
# ─────────────────────────────────────────────────────────

def download_util_benchmark(start: str, end: str) -> pd.Series:
    """
    Baixa o ETF UTLL11 como proxy do índice UTIL.
    """
    if not YF_AVAILABLE:
        return pd.Series()
    try:
        df = yf.download("UTLL11.SA", start=start, end=end, auto_adjust=True, progress=False)
        return df["Close"].dropna()
    except Exception:
        return pd.Series()


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Backtest — Tradebot-UTIL.v2")
    parser.add_argument("--strategy", default="mean_reversion",
                        choices=["mean_reversion", "all"])
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2026-01-01")
    parser.add_argument("--capital", type=float, default=100_000.0)
    args = parser.parse_args()

    print("\n" + "═" * 55)
    print("  Backtest — Tradebot-UTIL.v2")
    print(f"  Estratégia : {args.strategy}")
    print(f"  Período    : {args.start} → {args.end}")
    print(f"  Capital    : R$ {args.capital:,.2f}")
    print("═" * 55)

    # Universo de ativos
    UNIVERSE = [
        "SBSP3", "EQTL3", "CPLE3", "ENEV3", "CMIG4",
        "EGIE3", "TAEE11", "CPFE3", "ENGI11", "SAPR11",
        "NEOE3", "CSMG3", "ALUP11", "AURE3",
    ]

    print("\nBaixando dados históricos...")
    data = download_data(UNIVERSE, args.start, args.end)

    if not data:
        print("[ERRO] Nenhum dado disponível. Verifique a conexão e o yfinance.")
        sys.exit(1)

    if args.strategy in ("mean_reversion", "all"):
        print("\n── Backtest: Reversão à Média ──")
        trades = backtest_mean_reversion(data, initial_capital=args.capital)
        metrics = calculate_metrics(trades, args.capital)

        print("\nResultados:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

        # Comparar com benchmark
        print("\nBenchmark UTIL (UTLL11):")
        bench = download_util_benchmark(args.start, args.end)
        if not bench.empty:
            bench_return = (bench.iloc[-1] / bench.iloc[0] - 1) * 100
            print(f"  Retorno UTIL no período: {bench_return:.2f}%")
            print(f"  Retorno bot: {metrics.get('retorno_total_pct', 0):.2f}%")
            diff = metrics.get("retorno_total_pct", 0) - bench_return
            print(f"  Alpha gerado: {diff:+.2f}%")

        # Salvar resultados
        if not trades.empty:
            output = Path("logs/backtest_mean_reversion.csv")
            output.parent.mkdir(exist_ok=True)
            trades.to_csv(output, index=False)
            print(f"\n  Trades salvos em: {output}")

    print("\n" + "═" * 55)
    print("  Backtest concluído.")
    print("═" * 55 + "\n")


if __name__ == "__main__":
    main()
