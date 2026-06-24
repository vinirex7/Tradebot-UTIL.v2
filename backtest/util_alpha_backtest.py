"""
Backtest UTIL Alpha Allocator — Tradebot-UTIL.v2 / branch infra-1
────────────────────────────────────────────────────────────────

Objetivo:
    Testar uma estratégia long-only que usa TODO o universo do Índice UTIL B3
    como universo elegível, mas escolhe dinamicamente apenas os ativos com
    melhor combinação de momentum, reversão à média, tendência, liquidez e risco.

Racional:
    - Benchmark: carteira sintética do UTIL ponderada pelos pesos oficiais.
    - Estratégia: alocação ativa top-N, com limite por ativo, regime de mercado
      do próprio UTIL, filtro de tendência e redução automática de risco.
    - Sem short, sem alavancagem, sem MetaTrader5. Roda em Linux via yfinance.

Uso:
    python backtest/util_alpha_backtest.py --start 2019-01-01 --end 2026-01-01
    python backtest/util_alpha_backtest.py --capital 100000 --rebalance W-FRI --plot
    python backtest/util_alpha_backtest.py --top-n 6 --max-weight 0.20 --csv

Observação:
    Este arquivo é uma base de pesquisa. Antes de migrar para paper/live,
    validar dados ajustados, custos, slippage, liquidez e robustez fora da amostra.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Instale yfinance: pip install yfinance") from exc


# Universo completo do UTIL conforme carteira teórica Maio-Agosto/2026.
# O bot pode NÃO usar todos em cada rebalanceamento, mas todos são elegíveis.
UTIL_UNIVERSE: dict[str, dict[str, object]] = {
    "SBSP3":  {"weight": 19.999, "subsector": "saneamento",              "liquidity_tier": 1},
    "AXIA3":  {"weight": 17.291, "subsector": "transmissao",             "liquidity_tier": 1},
    "EQTL3":  {"weight": 11.431, "subsector": "distribuicao",            "liquidity_tier": 1},
    "ENEV3":  {"weight": 10.708, "subsector": "geracao_termeletrica",    "liquidity_tier": 1},
    "CPLE3":  {"weight": 10.318, "subsector": "distribuicao_transmissao", "liquidity_tier": 1},
    "CMIG4":  {"weight": 5.422,  "subsector": "distribuicao",            "liquidity_tier": 1},
    "ENGI11": {"weight": 3.786,  "subsector": "distribuicao",            "liquidity_tier": 2},
    "AXIA6":  {"weight": 2.708,  "subsector": "transmissao",             "liquidity_tier": 2},
    "EGIE3":  {"weight": 2.637,  "subsector": "geracao",                 "liquidity_tier": 2},
    "ISAE4":  {"weight": 2.587,  "subsector": "transmissao",             "liquidity_tier": 2},
    "CSMG3":  {"weight": 2.440,  "subsector": "saneamento",              "liquidity_tier": 2},
    "TAEE11": {"weight": 2.104,  "subsector": "transmissao",             "liquidity_tier": 2},
    "SAPR11": {"weight": 2.003,  "subsector": "saneamento",              "liquidity_tier": 2},
    "CPFE3":  {"weight": 2.070,  "subsector": "distribuicao",            "liquidity_tier": 2},
    "NEOE3":  {"weight": 1.509,  "subsector": "distribuicao",            "liquidity_tier": 2},
    "ALUP11": {"weight": 1.262,  "subsector": "transmissao",             "liquidity_tier": 3},
    "ORVR3":  {"weight": 0.862,  "subsector": "saneamento_residuos",      "liquidity_tier": 3},
    "AURE3":  {"weight": 0.855,  "subsector": "geracao",                 "liquidity_tier": 3},
}


@dataclass
class BacktestStats:
    total_return_pct: float
    benchmark_return_pct: float
    alpha_pct: float
    cagr_pct: float
    max_drawdown_pct: float
    sharpe: float
    volatility_pct: float
    turnover: float
    exposure_pct: float


def _clean_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if hasattr(df.index, "tz") and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def download_ohlcv(tickers: list[str], start: str, end: str, verbose: bool = True) -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        yf_symbol = f"{ticker}.SA"
        try:
            df = yf.Ticker(yf_symbol).history(start=start, end=end, auto_adjust=True, actions=False)
            if df.empty:
                if verbose:
                    print(f"  ✗ {ticker}: sem dados no yfinance")
                continue
            df = _clean_index(df)
            df.columns = [str(c).lower() for c in df.columns]
            keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
            df = df[keep].dropna()
            if len(df) >= 260 and "close" in df:
                data[ticker] = df
                if verbose:
                    print(f"  ✓ {ticker}: {len(df)} pregões")
            elif verbose:
                print(f"  ✗ {ticker}: histórico insuficiente ({len(df)} pregões)")
        except Exception as exc:
            if verbose:
                print(f"  ✗ {ticker}: {exc}")
    return data


def aligned_closes(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    closes = pd.concat({t: df["close"] for t, df in data.items()}, axis=1).sort_index()
    return closes.ffill().dropna(how="all")


def build_benchmark(closes: pd.DataFrame) -> pd.Series:
    weights = pd.Series({t: float(UTIL_UNIVERSE[t]["weight"]) for t in closes.columns}, dtype=float)
    weights = weights / weights.sum()
    normalized = closes / closes.iloc[0]
    return normalized.mul(weights, axis=1).sum(axis=1).dropna()


def zscore_frame(df: pd.DataFrame, window: int) -> pd.DataFrame:
    mean = df.rolling(window).mean()
    std = df.rolling(window).std().replace(0, np.nan)
    return (df - mean) / std


def compute_scores(closes: pd.DataFrame, volumes: pd.DataFrame, benchmark: pd.Series) -> pd.DataFrame:
    ret_3m = closes.pct_change(63)
    ret_6m = closes.pct_change(126)
    ret_12m = closes.pct_change(252)

    ema50 = closes.ewm(span=50, adjust=False).mean()
    ema200 = closes.ewm(span=200, adjust=False).mean()
    trend = (closes / ema200 - 1.0).clip(-0.25, 0.25)

    vol_63 = closes.pct_change().rolling(63).std() * math.sqrt(252)
    vol_penalty = zscore_frame(vol_63, 252)

    # Reversão controlada: compra quedas moderadas dentro de tendência positiva;
    # evita "catching falling knives" em queda estrutural.
    ma20 = closes.rolling(20).mean()
    sd20 = closes.rolling(20).std().replace(0, np.nan)
    boll_z = (closes - ma20) / sd20
    mean_reversion = (-boll_z).clip(-2, 2)
    mean_reversion = mean_reversion.where(closes > ema200, mean_reversion * 0.25)

    mom_score = (
        0.45 * zscore_frame(ret_6m, 252)
        + 0.30 * zscore_frame(ret_12m, 252)
        + 0.15 * zscore_frame(ret_3m, 126)
    )
    trend_score = zscore_frame(trend, 252)

    liq_raw = (closes * volumes).rolling(20).mean()
    liquidity_score = zscore_frame(np.log(liq_raw.replace(0, np.nan)), 252)

    tier_bonus = pd.Series(
        {t: {1: 0.15, 2: 0.00, 3: -0.15}.get(int(UTIL_UNIVERSE[t]["liquidity_tier"]), 0.0) for t in closes.columns}
    )

    # Regime do próprio UTIL: se benchmark está abaixo da média longa, o bot fica mais seletivo.
    bench_ma100 = benchmark.rolling(100).mean()
    bench_ma200 = benchmark.rolling(200).mean()
    bench_trend = pd.Series(0.0, index=benchmark.index)
    bench_trend = bench_trend.mask(benchmark > bench_ma100, 0.50)
    bench_trend = bench_trend.mask((benchmark > bench_ma100) & (bench_ma100 > bench_ma200), 1.00)
    bench_trend = bench_trend.reindex(closes.index).ffill().fillna(0.0)

    score = (
        0.48 * mom_score
        + 0.18 * trend_score
        + 0.16 * mean_reversion
        + 0.10 * liquidity_score
        - 0.08 * vol_penalty
    )
    score = score.add(tier_bonus, axis=1)
    score = score.mul(0.70 + 0.30 * bench_trend, axis=0)

    # Penaliza ativos em tendência ruim independentemente do ranking.
    score = score.where(closes > ema200, score - 0.75)
    return score.replace([np.inf, -np.inf], np.nan)


def target_weights_for_date(
    date: pd.Timestamp,
    scores: pd.DataFrame,
    closes: pd.DataFrame,
    benchmark: pd.Series,
    top_n: int,
    max_weight: float,
    min_score: float,
    cash_floor: float,
) -> pd.Series:
    row = scores.loc[date].dropna()
    if row.empty:
        return pd.Series(0.0, index=closes.columns)

    bench_hist = benchmark.loc[:date]
    if len(bench_hist) < 200:
        return pd.Series(0.0, index=closes.columns)

    bench_now = bench_hist.iloc[-1]
    bench_ma100 = bench_hist.rolling(100).mean().iloc[-1]
    bench_ma200 = bench_hist.rolling(200).mean().iloc[-1]
    bench_dd = bench_now / bench_hist.cummax().iloc[-1] - 1.0

    if bench_now > bench_ma100 > bench_ma200:
        gross_exposure = 1.00
    elif bench_now > bench_ma200:
        gross_exposure = 0.75
    else:
        gross_exposure = 0.45

    if bench_dd < -0.12:
        gross_exposure *= 0.65
    if bench_dd < -0.20:
        gross_exposure *= 0.50

    gross_exposure = min(1.0 - cash_floor, max(0.0, gross_exposure))

    selected = row[row >= min_score].sort_values(ascending=False).head(top_n)
    if selected.empty:
        return pd.Series(0.0, index=closes.columns)

    positive = selected - selected.min() + 1.0
    raw = positive / positive.sum() * gross_exposure
    capped = raw.clip(upper=max_weight)

    # Redistribui sobra respeitando teto por ativo.
    for _ in range(5):
        spare = gross_exposure - capped.sum()
        if spare <= 1e-8:
            break
        room = max_weight - capped
        room = room[room > 1e-8]
        if room.empty:
            break
        add = (room / room.sum() * spare).clip(upper=room)
        capped.loc[add.index] += add

    weights = pd.Series(0.0, index=closes.columns)
    weights.loc[capped.index] = capped
    return weights


def run_backtest(
    closes: pd.DataFrame,
    volumes: pd.DataFrame,
    capital: float,
    rebalance: str,
    top_n: int,
    max_weight: float,
    min_score: float,
    cash_floor: float,
    fee_bps: float,
    slippage_bps: float,
) -> tuple[pd.Series, pd.Series, pd.DataFrame, float]:
    benchmark = build_benchmark(closes)
    scores = compute_scores(closes, volumes, benchmark)

    daily_returns = closes.pct_change().fillna(0.0)
    rebalance_dates = closes.resample(rebalance).last().index
    rebalance_dates = closes.index.intersection(rebalance_dates)

    weights = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
    current = pd.Series(0.0, index=closes.columns)
    turnover = 0.0
    cost_rate = (fee_bps + slippage_bps) / 10_000

    for dt in closes.index:
        if dt in rebalance_dates:
            target = target_weights_for_date(dt, scores, closes, benchmark, top_n, max_weight, min_score, cash_floor)
            turnover += float((target - current).abs().sum())
            current = target
        weights.loc[dt] = current

    # Usa pesos de ontem aplicados ao retorno de hoje para reduzir lookahead.
    shifted = weights.shift(1).fillna(0.0)
    strategy_returns = (shifted * daily_returns).sum(axis=1)

    # Custo de troca cobrado nos dias após mudança de peso.
    daily_turnover = shifted.diff().abs().sum(axis=1).fillna(0.0)
    strategy_returns = strategy_returns - daily_turnover * cost_rate

    equity = (1 + strategy_returns).cumprod() * capital
    bench_equity = (benchmark / benchmark.iloc[0]) * capital

    return equity.dropna(), bench_equity.reindex(equity.index).ffill().dropna(), weights, turnover


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    return float((equity / equity.cummax() - 1.0).min())


def annualized_sharpe(equity: pd.Series) -> float:
    rets = equity.pct_change().dropna()
    if rets.empty or rets.std() == 0:
        return 0.0
    return float(rets.mean() / rets.std() * math.sqrt(252))


def annualized_vol(equity: pd.Series) -> float:
    rets = equity.pct_change().dropna()
    return float(rets.std() * math.sqrt(252)) if not rets.empty else 0.0


def cagr(equity: pd.Series) -> float:
    if len(equity) < 2:
        return 0.0
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    if years <= 0:
        return 0.0
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1.0)


def summarize(equity: pd.Series, benchmark_equity: pd.Series, weights: pd.DataFrame, turnover: float) -> BacktestStats:
    total_ret = equity.iloc[-1] / equity.iloc[0] - 1.0
    bench_ret = benchmark_equity.iloc[-1] / benchmark_equity.iloc[0] - 1.0
    exposure = weights.sum(axis=1).mean()
    return BacktestStats(
        total_return_pct=total_ret * 100,
        benchmark_return_pct=bench_ret * 100,
        alpha_pct=(total_ret - bench_ret) * 100,
        cagr_pct=cagr(equity) * 100,
        max_drawdown_pct=max_drawdown(equity) * 100,
        sharpe=annualized_sharpe(equity),
        volatility_pct=annualized_vol(equity) * 100,
        turnover=turnover,
        exposure_pct=exposure * 100,
    )


def save_outputs(equity: pd.Series, benchmark: pd.Series, weights: pd.DataFrame, output_dir: Path, make_csv: bool, make_plot: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if make_csv:
        pd.DataFrame({"strategy": equity, "benchmark_util": benchmark}).to_csv(output_dir / "util_alpha_equity.csv")
        weights.to_csv(output_dir / "util_alpha_weights.csv")
        print(f"CSV salvo em: {output_dir}")

    if make_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(13, 7))
            ax.plot(equity.index, equity.values, label="UTIL Alpha Allocator")
            ax.plot(benchmark.index, benchmark.values, label="Benchmark UTIL sintético", linestyle="--")
            ax.set_title("Tradebot-UTIL.v2 — UTIL Alpha Allocator vs Benchmark")
            ax.set_ylabel("Patrimônio (R$)")
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            path = output_dir / "util_alpha_backtest.png"
            fig.savefig(path, dpi=150)
            print(f"Gráfico salvo em: {path}")
        except Exception as exc:
            print(f"[Aviso] Não foi possível gerar gráfico: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest UTIL Alpha Allocator — universo completo UTIL")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2026-01-01")
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument("--rebalance", default="W-FRI", help="Frequência pandas: W-FRI, W, M, 2W-FRI")
    parser.add_argument("--top-n", type=int, default=6)
    parser.add_argument("--max-weight", type=float, default=0.20)
    parser.add_argument("--min-score", type=float, default=-0.10)
    parser.add_argument("--cash-floor", type=float, default=0.03)
    parser.add_argument("--fee-bps", type=float, default=3.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--csv", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    tickers = list(UTIL_UNIVERSE.keys())
    if not args.quiet:
        print("Tradebot-UTIL.v2 — Backtest UTIL Alpha Allocator")
        print(f"Universo UTIL elegível: {len(tickers)} ativos")
        print("Ativos:", ", ".join(tickers))
        print(f"Período: {args.start} → {args.end} | Capital: R$ {args.capital:,.2f}")

    data = download_ohlcv(tickers, args.start, args.end, verbose=not args.quiet)
    if len(data) < 5:
        raise SystemExit("Poucos ativos com dados suficientes. Verifique conexão/datas/yfinance.")

    closes = aligned_closes(data)
    volumes = pd.concat({t: df["volume"] for t, df in data.items()}, axis=1).reindex(closes.index).ffill().fillna(0.0)

    equity, bench_equity, weights, turnover = run_backtest(
        closes=closes,
        volumes=volumes,
        capital=args.capital,
        rebalance=args.rebalance,
        top_n=args.top_n,
        max_weight=args.max_weight,
        min_score=args.min_score,
        cash_floor=args.cash_floor,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
    )
    bench_equity = bench_equity.reindex(equity.index).ffill()
    stats = summarize(equity, bench_equity, weights.reindex(equity.index).ffill(), turnover)

    print("\nResumo — UTIL Alpha Allocator")
    print("─" * 55)
    print(f"Retorno estratégia : {stats.total_return_pct:+.2f}%")
    print(f"Retorno benchmark  : {stats.benchmark_return_pct:+.2f}%")
    print(f"Alpha vs UTIL      : {stats.alpha_pct:+.2f}%")
    print(f"CAGR estratégia    : {stats.cagr_pct:+.2f}%")
    print(f"Max drawdown       : {stats.max_drawdown_pct:.2f}%")
    print(f"Sharpe             : {stats.sharpe:.2f}")
    print(f"Vol anualizada     : {stats.volatility_pct:.2f}%")
    print(f"Exposição média    : {stats.exposure_pct:.1f}%")
    print(f"Turnover bruto     : {stats.turnover:.2f}x")

    last_weights = weights.iloc[-1]
    last_weights = last_weights[last_weights > 0].sort_values(ascending=False)
    print("\nCarteira final sugerida pelo modelo:")
    if last_weights.empty:
        print("  100% caixa")
    else:
        for ticker, weight in last_weights.items():
            print(f"  {ticker:7s} {weight * 100:6.2f}%")

    if stats.alpha_pct > 0:
        print("\n✓ No período testado, a estratégia superou o benchmark UTIL sintético.")
    else:
        print("\n✗ No período testado, a estratégia não superou o benchmark. Ajustar parâmetros e validar fora da amostra.")

    save_outputs(
        equity=equity,
        benchmark=bench_equity,
        weights=weights,
        output_dir=Path("logs"),
        make_csv=args.csv,
        make_plot=args.plot,
    )


if __name__ == "__main__":
    main()
