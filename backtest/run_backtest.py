"""
Backtest Runner — Tradebot-UTIL.v2
────────────────────────────────────
Script principal para rodar o backtest completo via yfinance.
100% compatível com Linux (sem MetaTrader5).

Uso:
    python backtest/run_backtest.py
    python backtest/run_backtest.py --strategy momentum_macro
    python backtest/run_backtest.py --start 2022-01-01 --end 2026-01-01
    python backtest/run_backtest.py --capital 50000
    python backtest/run_backtest.py --plot
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from backtest.backtest_engine import BacktestEngine, UTIL_UNIVERSE

try:
    from tabulate import tabulate
    TABULATE = True
except ImportError:
    TABULATE = False


def print_header(text: str) -> None:
    print("\n" + "═" * 60)
    print(f"  {text}")
    print("═" * 60)


def print_trades(trades, max_rows: int = 20) -> None:
    """Imprime as últimas operações."""
    if not trades:
        print("  Nenhuma operação registrada.")
        return

    rows = []
    for t in sorted(trades, key=lambda x: x.entry_date)[-max_rows:]:
        rows.append({
            "Ticker":   t.ticker,
            "Dir":      t.direction[:1].upper(),
            "Entrada":  str(t.entry_date)[:10],
            "Saída":    str(t.exit_date)[:10] if t.exit_date else "—",
            "P. Entr.": f"{t.entry_price:.2f}",
            "P. Saída": f"{t.exit_price:.2f}",
            "Ret (%)":  f"{t.return_pct:+.2f}",
            "P&L (R$)": f"{t.pnl:+.2f}",
            "Motivo":   t.exit_reason,
        })

    df = pd.DataFrame(rows)
    if TABULATE:
        print(tabulate(df, headers="keys", tablefmt="rounded_outline", showindex=False))
    else:
        print(df.to_string(index=False))


def plot_results(results: dict, benchmark, start: str, end: str) -> None:
    """Gera gráfico das equity curves vs benchmark."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        fig.suptitle(
            f"Tradebot-UTIL.v2 — Backtest {start} → {end}",
            fontsize=14, fontweight="bold"
        )

        # --- Painel 1: Equity curves ---
        ax1 = axes[0]
        colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]

        for (name, result), color in zip(results.items(), colors):
            if not result.equity_curve.empty:
                eq = result.equity_curve
                ax1.plot(eq.index, eq.values, label=name.replace("_", " ").title(),
                         color=color, linewidth=1.8)

        # Benchmark
        if not benchmark.empty:
            # Normalizar benchmark para a mesma base de capital
            cap = list(results.values())[0].equity_curve.iloc[0] if results else 100_000
            bench_norm = benchmark / benchmark.iloc[0] * cap
            ax1.plot(bench_norm.index, bench_norm.values, label="UTIL Benchmark (UTLL11)",
                     color="#F44336", linewidth=2, linestyle="--")

        ax1.set_ylabel("Patrimônio (R$)", fontsize=11)
        ax1.set_title("Equity Curves vs Benchmark UTIL", fontsize=12)
        ax1.legend(loc="upper left", fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax1.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, _: f"R$ {x:,.0f}")
        )

        # --- Painel 2: Barras de retorno por estratégia ---
        ax2 = axes[1]
        names, bot_rets, bench_rets = [], [], []
        for name, result in results.items():
            names.append(name.replace("_", "\n").title())
            bot_rets.append(result.total_return_pct)
            bench_rets.append(result.benchmark_return_pct)

        x = range(len(names))
        width = 0.35
        bars1 = ax2.bar([i - width/2 for i in x], bot_rets, width,
                        label="Bot", color="#2196F3", alpha=0.85)
        bars2 = ax2.bar([i + width/2 for i in x], bench_rets, width,
                        label="UTIL Benchmark", color="#F44336", alpha=0.85)

        ax2.axhline(0, color="black", linewidth=0.8)
        ax2.set_ylabel("Retorno Total (%)", fontsize=11)
        ax2.set_title("Retorno Total: Bot vs Benchmark UTIL", fontsize=12)
        ax2.set_xticks(list(x))
        ax2.set_xticklabels(names, fontsize=9)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3, axis="y")

        # Rótulos nas barras
        for bar in bars1:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                     f"{h:+.1f}%", ha="center", va="bottom", fontsize=8)
        for bar in bars2:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                     f"{h:+.1f}%", ha="center", va="bottom", fontsize=8)

        plt.tight_layout()
        output_path = Path("logs/backtest_results.png")
        output_path.parent.mkdir(exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"\n  Gráfico salvo em: {output_path}")

    except Exception as e:
        print(f"\n  [Aviso] Não foi possível gerar o gráfico: {e}")


def save_trades_csv(results: dict) -> None:
    """Salva todas as operações em CSV."""
    all_trades = []
    for name, result in results.items():
        for t in result.trades:
            all_trades.append({
                "estrategia":  t.strategy,
                "ticker":      t.ticker,
                "direcao":     t.direction,
                "entrada_dt":  t.entry_date,
                "saida_dt":    t.exit_date,
                "preco_entr":  round(t.entry_price, 4),
                "preco_saida": round(t.exit_price, 4),
                "qtd_acoes":   t.shares,
                "capital_usado": round(t.capital_used, 2),
                "pnl_brl":     round(t.pnl, 2),
                "retorno_pct": round(t.return_pct, 4),
                "motivo_saida": t.exit_reason,
            })

    if all_trades:
        df = pd.DataFrame(all_trades).sort_values("entrada_dt")
        output = Path("logs/backtest_trades.csv")
        output.parent.mkdir(exist_ok=True)
        df.to_csv(output, index=False)
        print(f"\n  Operações salvas em: {output}")


def main():
    parser = argparse.ArgumentParser(
        description="Backtest Tradebot-UTIL.v2 (via yfinance, sem MetaTrader5)"
    )
    parser.add_argument(
        "--strategy",
        choices=["momentum_macro", "all"],
        default="all",
        help="Estratégia a backtestear (padrão: momentum_macro)",
    )
    parser.add_argument("--start",   default="2019-01-01", help="Data inicial (YYYY-MM-DD)")
    parser.add_argument("--end",     default="2026-01-01", help="Data final (YYYY-MM-DD)")
    parser.add_argument("--capital", type=float, default=100_000.0, help="Capital inicial em BRL")
    parser.add_argument("--plot",    action="store_true", help="Gerar gráfico PNG")
    parser.add_argument("--csv",     action="store_true", help="Salvar operações em CSV")
    parser.add_argument("--quiet",   action="store_true", help="Modo silencioso (menos output)")
    args = parser.parse_args()

    strategies = None if args.strategy == "all" else [args.strategy]

    print_header("Tradebot-UTIL.v2 — Backtest via yfinance")
    print(f"  Período  : {args.start}  →  {args.end}")
    print(f"  Capital  : R$ {args.capital:,.2f}")
    print(f"  Estratégia: {args.strategy}")

    # Inicializar engine
    engine = BacktestEngine(
        start=args.start,
        end=args.end,
        capital=args.capital,
    )

    # Carregar dados
    print_header("Baixando dados históricos (yfinance)")
    engine.load_data(verbose=not args.quiet)

    if not engine.data:
        print("\n  [ERRO] Nenhum dado disponível. Verifique sua conexão.")
        sys.exit(1)

    # Executar backtests
    print_header("Executando backtests")
    results = engine.run(strategies=strategies)

    if not results:
        print("  Nenhuma estratégia foi executada.")
        sys.exit(1)

    # Tabela de resultados
    print_header("Resultados — Resumo Comparativo")
    summary = engine.summary_table(results)
    if TABULATE:
        print(tabulate(summary, headers="keys", tablefmt="rounded_outline", floatfmt=".2f"))
    else:
        print(summary.to_string())

    # Detalhe por estratégia
    if not args.quiet:
        for name, result in results.items():
            print_header(f"Detalhes: {name.replace('_', ' ').title()}")
            print(f"  Total de operações : {result.total_trades}")
            print(f"  Win Rate           : {result.win_rate:.1f}%")
            print(f"  Retorno total bot  : {result.total_return_pct:+.2f}%")
            print(f"  UTIL benchmark     : {result.benchmark_return_pct:+.2f}%")
            print(f"  Alpha gerado       : {result.alpha:+.2f}%")
            print(f"  Profit Factor      : {result.profit_factor:.2f}")
            print(f"  Max Drawdown       : {result.max_drawdown:.2f}%")
            print(f"  Sharpe Ratio       : {result.sharpe_ratio:.2f}")
            print(f"\n  Últimas operações:")
            print_trades(result.trades, max_rows=10)

    # Gráfico
    if args.plot:
        print_header("Gerando gráfico")
        plot_results(results, engine.benchmark, args.start, args.end)

    # CSV
    if args.csv:
        save_trades_csv(results)

    print_header("Backtest concluído")
    # Alpha combinado (média ponderada)
    total_alpha = sum(r.alpha for r in results.values()) / len(results)
    print(f"  Alpha médio vs UTIL: {total_alpha:+.2f}%")
    if total_alpha > 0:
        print("  ✓ Bot supera o benchmark UTIL no período testado.")
    else:
        print("  ✗ Bot não superou o benchmark UTIL no período testado.")
        print("    Ajuste os parâmetros e refaça o backtest.")
    print()


if __name__ == "__main__":
    main()
