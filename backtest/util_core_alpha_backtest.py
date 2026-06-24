"""
Backtest UTIL Core + Alpha — Tradebot-UTIL.v2 / branch infra-1
───────────────────────────────────────────────────────────────

Estratégia:
    Em vez de tentar vencer o UTIL escolhendo só poucos ativos, esta versão
    mantém um núcleo comprado no próprio UTIL sintético e usa uma parcela menor
    para gerar alpha por sobrepeso nos melhores ativos do universo.

Racional:
    - Quando o setor inteiro sobe, o bot não fica muito para trás do índice.
    - Quando há dispersão entre papéis, o sleeve alpha tenta ganhar excesso.
    - Em regime ruim, reduz exposição e aumenta caixa.

Uso:
    python backtest/util_core_alpha_backtest.py --start 2025-06-23 --end 2026-06-23
    python backtest/util_core_alpha_backtest.py --start 2024-06-23 --end 2026-06-23 --plot --csv
    python backtest/util_core_alpha_backtest.py --core-bull 0.85 --alpha-bull 0.15 --rebalance M
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import yfinance as yf
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Instale yfinance: pip install yfinance") from exc

from backtest.backtest_engine import UTIL_UNIVERSE, synthetic_util_benchmark


LIQUIDITY_TIER = {
    "SBSP3": 1,
    "AXIA3": 1,
    "EQTL3": 1,
    "ENEV3": 1,
    "CPLE3": 1,
    "CMIG4": 1,
    "ENGI11": 2,
    "AXIA6": 2,
    "EGIE3": 2,
    "ISAE4": 2,
    "CSMG3": 2,
    "TAEE11": 2,
    "SAPR11": 2,
    "CPFE3": 2,
    "NEOE3": 2,
    "ALUP11": 3,
    "ORVR3": 3,
    "AURE3": 3,
}


@dataclass
class Stats:
    strategy_return_pct: float
    benchmark_return_pct: float
    alpha_pct: float
    cagr_pct: float
    max_drawdown_pct: float
    sharpe: float
    volatility_pct: float
    exposure_pct: float
    turnover: float


def clean_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if hasattr(df.index, "tz") and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def download_ohlcv(tickers: list[str], start: str, end: str, verbose: bool = True) -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        try:
            df = yf.Ticker(f"{ticker}.SA").history(start=start, end=end, auto_adjust=True, actions=False)
            if df.empty:
                if verbose:
                    print(f"  ✗ {ticker}: sem dados")
                continue
            df = clean_index(df)
            df.columns = [str(c).lower() for c in df.columns]
            keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
            df = df[keep].dropna()
            if len(df) >= 80 and "close" in df.columns:
                data[ticker] = df
                if verbose:
                    print(f"  ✓ {ticker}: {len(df)} pregões")
            elif verbose:
                print(f"  ✗ {ticker}: histórico insuficiente ({len(df)} pregões)")
        except Exception as exc:
            if verbose:
                print(f"  ✗ {ticker}: {exc}")
    return data


def aligned_field(data: dict[str, pd.DataFrame], field: str) -> pd.DataFrame:
    frame = pd.concat({t: df[field] for t, df in data.items() if field in df.columns}, axis=1).sort_index()
    return frame.ffill().dropna(how="all")


def util_core_weights(columns: pd.Index) -> pd.Series:
    weights = pd.Series({ticker: UTIL_UNIVERSE[ticker] for ticker in columns if ticker in UTIL_UNIVERSE}, dtype=float)
    weights = weights / weights.sum()
    return weights.reindex(columns).fillna(0.0)


def cs_rank(frame: pd.DataFrame, higher_is_better: bool = True) -> pd.DataFrame:
    ranked = frame.rank(axis=1, pct=True)
    if not higher_is_better:
        ranked = 1.0 - ranked
    return ((ranked - 0.5) * 2.0).fillna(0.0)


def compute_alpha_scores(closes: pd.DataFrame, volumes: pd.DataFrame, benchmark: pd.Series) -> pd.DataFrame:
    bench = benchmark.reindex(closes.index).ffill()

    rel_1m = closes.pct_change(21).sub(bench.pct_change(21), axis=0)
    rel_3m = closes.pct_change(63).sub(bench.pct_change(63), axis=0)
    rel_6m = closes.pct_change(126).sub(bench.pct_change(126), axis=0)
    rel_12m = closes.pct_change(252).sub(bench.pct_change(252), axis=0)

    ema50 = closes.ewm(span=50, adjust=False, min_periods=30).mean()
    ema200 = closes.ewm(span=200, adjust=False, min_periods=60).mean()
    trend = (closes / ema200 - 1.0).clip(-0.30, 0.30)
    trend_short = (closes / ema50 - 1.0).clip(-0.20, 0.20)

    vol = closes.pct_change().rolling(63, min_periods=35).std() * math.sqrt(252)
    traded_value = np.log((closes * volumes).rolling(20, min_periods=10).mean().replace(0, np.nan))

    tier_bonus = pd.Series({t: {1: 0.06, 2: 0.00, 3: -0.06}.get(LIQUIDITY_TIER.get(t, 2), 0.0) for t in closes.columns})

    score = (
        0.12 * cs_rank(rel_1m)
        + 0.26 * cs_rank(rel_3m)
        + 0.23 * cs_rank(rel_6m)
        + 0.07 * cs_rank(rel_12m)
        + 0.13 * cs_rank(trend)
        + 0.07 * cs_rank(trend_short)
        + 0.05 * cs_rank(traded_value)
        + 0.07 * cs_rank(vol, higher_is_better=False)
    )
    score = score.add(tier_bonus, axis=1)
    score = score.where((closes >= ema200) | ema200.isna(), score - 0.10)
    return score.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def market_regime(date: pd.Timestamp, benchmark: pd.Series) -> str:
    hist = benchmark.loc[:date].dropna()
    if len(hist) < 40:
        return "warmup"

    now = hist.iloc[-1]
    ma80 = hist.rolling(80, min_periods=40).mean().iloc[-1]
    ma160 = hist.rolling(160, min_periods=80).mean().iloc[-1]
    if pd.isna(ma160):
        ma160 = ma80
    dd = now / hist.cummax().iloc[-1] - 1.0

    if now > ma80 and ma80 >= ma160:
        return "bull"
    if now > ma160 and dd > -0.10:
        return "neutral"
    return "defensive"


def sleeve_exposure(regime: str, args: argparse.Namespace) -> tuple[float, float]:
    if regime == "bull":
        return args.core_bull, args.alpha_bull
    if regime == "neutral":
        return args.core_neutral, args.alpha_neutral
    if regime == "defensive":
        return args.core_defensive, args.alpha_defensive
    return 0.0, 0.0


def alpha_sleeve_weights(row: pd.Series, columns: pd.Index, top_n: int, max_alpha_asset: float) -> pd.Series:
    weights = pd.Series(0.0, index=columns)
    selected = row.dropna().sort_values(ascending=False).head(max(1, min(top_n, len(row))))
    if selected.empty:
        return weights
    shifted = selected - selected.min() + 0.25
    raw = shifted / shifted.sum()
    capped = raw.clip(upper=max_alpha_asset)
    for _ in range(10):
        spare = 1.0 - capped.sum()
        if spare <= 1e-9:
            break
        room = (max_alpha_asset - capped).clip(lower=0)
        if room.sum() <= 1e-9:
            break
        add = (room / room.sum() * spare).clip(upper=room)
        capped = capped.add(add, fill_value=0.0)
    if capped.sum() > 0:
        capped = capped / capped.sum()
    weights.loc[capped.index] = capped
    return weights


def target_weights_for_date(
    date: pd.Timestamp,
    closes: pd.DataFrame,
    benchmark: pd.Series,
    scores: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.Series:
    core_exp, alpha_exp = sleeve_exposure(market_regime(date, benchmark), args)
    core = util_core_weights(closes.columns)
    alpha = alpha_sleeve_weights(scores.loc[date], closes.columns, args.top_n, args.max_alpha_asset)

    target = core_exp * core + alpha_exp * alpha
    target = target.clip(upper=args.max_total_asset)

    # Se o teto por ativo cortou demais, redistribui sobra para o core respeitando o teto.
    intended = core_exp + alpha_exp
    for _ in range(10):
        spare = intended - target.sum()
        if spare <= 1e-9:
            break
        room = (args.max_total_asset - target).clip(lower=0)
        if room.sum() <= 1e-9:
            break
        add = (core / core.sum() * spare).clip(upper=room)
        target = target.add(add, fill_value=0.0)

    return target.reindex(closes.columns).fillna(0.0)


def run_backtest(
    data: dict[str, pd.DataFrame],
    closes: pd.DataFrame,
    volumes: pd.DataFrame,
    capital: float,
    args: argparse.Namespace,
) -> tuple[pd.Series, pd.Series, pd.DataFrame, float]:
    benchmark = synthetic_util_benchmark(data, target_index=closes.index)
    closes = closes.reindex(benchmark.index).ffill().dropna(how="all")
    volumes = volumes.reindex(closes.index).ffill().fillna(0.0)
    benchmark = benchmark.reindex(closes.index).ffill().dropna()

    scores = compute_alpha_scores(closes, volumes, benchmark)
    daily_returns = closes.pct_change().fillna(0.0)
    rebalance_dates = closes.resample(args.rebalance).last().index
    rebalance_dates = closes.index.intersection(rebalance_dates)

    weights = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
    current = pd.Series(0.0, index=closes.columns)
    turnover = 0.0
    cost_rate = (args.fee_bps + args.slippage_bps) / 10_000

    for dt in closes.index:
        if dt in rebalance_dates:
            target = target_weights_for_date(dt, closes, benchmark, scores, args)
            turnover += float((target - current).abs().sum())
            current = target
        weights.loc[dt] = current

    shifted = weights.shift(1).fillna(0.0)
    strategy_returns = (shifted * daily_returns).sum(axis=1)
    daily_turnover = shifted.diff().abs().sum(axis=1).fillna(0.0)
    strategy_returns = strategy_returns - daily_turnover * cost_rate

    equity = (1.0 + strategy_returns).cumprod() * capital
    bench_equity = (benchmark / benchmark.iloc[0]) * capital
    bench_equity = bench_equity.reindex(equity.index).ffill()
    return equity.dropna(), bench_equity.dropna(), weights, turnover


def max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1.0).min()) if not equity.empty else 0.0


def cagr(equity: pd.Series) -> float:
    if len(equity) < 2:
        return 0.0
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1.0) if years > 0 else 0.0


def sharpe(equity: pd.Series) -> float:
    ret = equity.pct_change().dropna()
    if ret.empty or ret.std() == 0:
        return 0.0
    return float(ret.mean() / ret.std() * math.sqrt(252))


def volatility(equity: pd.Series) -> float:
    ret = equity.pct_change().dropna()
    return float(ret.std() * math.sqrt(252)) if not ret.empty else 0.0


def summarize(equity: pd.Series, benchmark: pd.Series, weights: pd.DataFrame, turnover: float) -> Stats:
    common = equity.index.intersection(benchmark.index)
    equity = equity.loc[common]
    benchmark = benchmark.loc[common]
    strategy_return = equity.iloc[-1] / equity.iloc[0] - 1.0
    benchmark_return = benchmark.iloc[-1] / benchmark.iloc[0] - 1.0
    return Stats(
        strategy_return_pct=strategy_return * 100,
        benchmark_return_pct=benchmark_return * 100,
        alpha_pct=(strategy_return - benchmark_return) * 100,
        cagr_pct=cagr(equity) * 100,
        max_drawdown_pct=max_drawdown(equity) * 100,
        sharpe=sharpe(equity),
        volatility_pct=volatility(equity) * 100,
        exposure_pct=weights.reindex(common).ffill().sum(axis=1).mean() * 100,
        turnover=turnover,
    )


def save_outputs(equity: pd.Series, benchmark: pd.Series, weights: pd.DataFrame, output_dir: Path, csv: bool, plot: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if csv:
        pd.DataFrame({"strategy": equity, "benchmark_util": benchmark}).to_csv(output_dir / "util_core_alpha_equity.csv")
        weights.to_csv(output_dir / "util_core_alpha_weights.csv")
        print(f"CSV salvo em: {output_dir}")
    if plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(13, 7))
            ax.plot(equity.index, equity.values, label="UTIL Core + Alpha")
            ax.plot(benchmark.index, benchmark.values, label="Benchmark UTIL", linestyle="--")
            ax.set_title("Tradebot-UTIL.v2 — UTIL Core + Alpha vs Benchmark")
            ax.set_ylabel("Patrimônio (R$)")
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            path = output_dir / "util_core_alpha_backtest.png"
            fig.savefig(path, dpi=150)
            print(f"Gráfico salvo em: {path}")
        except Exception as exc:
            print(f"[Aviso] Não foi possível gerar gráfico: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest UTIL Core + Alpha")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2026-01-01")
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument("--rebalance", default="M", help="Frequência pandas: M, W-FRI, 2W-FRI")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--core-bull", type=float, default=0.85)
    parser.add_argument("--alpha-bull", type=float, default=0.20)
    parser.add_argument("--core-neutral", type=float, default=0.70)
    parser.add_argument("--alpha-neutral", type=float, default=0.20)
    parser.add_argument("--core-defensive", type=float, default=0.40)
    parser.add_argument("--alpha-defensive", type=float, default=0.15)
    parser.add_argument("--max-alpha-asset", type=float, default=0.35)
    parser.add_argument("--max-total-asset", type=float, default=0.25)
    parser.add_argument("--fee-bps", type=float, default=3.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--csv", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    tickers = list(UTIL_UNIVERSE.keys())
    if not args.quiet:
        print("Tradebot-UTIL.v2 — Backtest UTIL Core + Alpha")
        print(f"Universo UTIL elegível: {len(tickers)} ativos")
        print("Ativos:", ", ".join(tickers))
        print(f"Período: {args.start} → {args.end} | Capital: R$ {args.capital:,.2f}")
        print(
            f"Regimes: bull core/alpha={args.core_bull:.0%}/{args.alpha_bull:.0%}, "
            f"neutral={args.core_neutral:.0%}/{args.alpha_neutral:.0%}, "
            f"defensive={args.core_defensive:.0%}/{args.alpha_defensive:.0%}"
        )

    data = download_ohlcv(tickers, args.start, args.end, verbose=not args.quiet)
    if len(data) < 5:
        raise SystemExit("Poucos ativos com dados suficientes. Verifique conexão/datas/yfinance.")

    closes = aligned_field(data, "close")
    volumes = aligned_field(data, "volume").reindex(closes.index).ffill().fillna(0.0)

    equity, benchmark, weights, turnover = run_backtest(data, closes, volumes, args.capital, args)
    stats = summarize(equity, benchmark, weights, turnover)

    print("\nResumo — UTIL Core + Alpha")
    print("─" * 55)
    print(f"Retorno estratégia : {stats.strategy_return_pct:+.2f}%")
    print(f"Retorno benchmark  : {stats.benchmark_return_pct:+.2f}%")
    print(f"Alpha vs UTIL      : {stats.alpha_pct:+.2f}%")
    print(f"CAGR estratégia    : {stats.cagr_pct:+.2f}%")
    print(f"Max drawdown       : {stats.max_drawdown_pct:.2f}%")
    print(f"Sharpe             : {stats.sharpe:.2f}")
    print(f"Vol anualizada     : {stats.volatility_pct:.2f}%")
    print(f"Exposição média    : {stats.exposure_pct:.1f}%")
    print(f"Turnover bruto     : {stats.turnover:.2f}x")

    last = weights.iloc[-1]
    last = last[last > 0].sort_values(ascending=False)
    print("\nCarteira final sugerida pelo modelo:")
    if last.empty:
        print("  100% caixa")
    else:
        for ticker, weight in last.items():
            print(f"  {ticker:7s} {weight * 100:6.2f}%")

    if stats.alpha_pct > 0:
        print("\n✓ No período testado, a estratégia superou o benchmark UTIL sintético.")
    else:
        print("\n✗ No período testado, a estratégia não superou o benchmark. Ajustar parâmetros e validar fora da amostra.")

    save_outputs(equity, benchmark, weights, Path("logs"), args.csv, args.plot)


if __name__ == "__main__":
    main()
