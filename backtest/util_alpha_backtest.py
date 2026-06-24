"""
Backtest UTIL Alpha Allocator — Tradebot-UTIL.v2 / branch infra-1
────────────────────────────────────────────────────────────────

Estratégia de pesquisa para a branch infra-1.

Objetivo:
    Usar TODOS os ativos do Índice UTIL como universo elegível e montar uma
    carteira ativa long-only que tenta superar o benchmark UTIL sintético.

Correção desta versão:
    O benchmark deste arquivo agora usa EXATAMENTE o mesmo cálculo/base do
    backtest principal: backtest.backtest_engine. Assim o resultado de
    "Retorno benchmark" fica comparável entre main e infra-1.

Uso:
    python backtest/util_alpha_backtest.py --start 2023-06-23 --end 2026-06-23
    python backtest/util_alpha_backtest.py --start 2023-06-23 --end 2026-06-23 --plot --csv
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

# Importa a MESMA definição de benchmark usada pelo backtest principal.
# Não duplique o cálculo aqui, para não haver divergência entre branches/arquivos.
from backtest.backtest_engine import UTIL_UNIVERSE as BENCHMARK_UTIL_UNIVERSE
from backtest.backtest_engine import synthetic_util_benchmark


# Universo completo elegível da estratégia infra-1.
# Atenção: isso é o universo de seleção do bot, não o benchmark.
STRATEGY_UNIVERSE: dict[str, dict[str, object]] = {
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
        try:
            df = yf.Ticker(f"{ticker}.SA").history(start=start, end=end, auto_adjust=True, actions=False)
            if df.empty:
                if verbose:
                    print(f"  ✗ {ticker}: sem dados no yfinance")
                continue
            df = _clean_index(df)
            df.columns = [str(c).lower() for c in df.columns]
            keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
            df = df[keep].dropna()
            if len(df) >= 120 and "close" in df:
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


def build_benchmark_from_shared_engine(data: dict[str, pd.DataFrame], target_index: pd.Index) -> pd.Series:
    """Benchmark idêntico ao backtest principal.

    O backtest principal calcula o UTIL sintético com synthetic_util_benchmark(data),
    usando BENCHMARK_UTIL_UNIVERSE de backtest_engine. Este wrapper só reindexa para
    a curva diária da estratégia, sem mudar pesos nem incluir/remover ativos.
    """
    benchmark = synthetic_util_benchmark(data, target_index=target_index)
    return benchmark.reindex(target_index).ffill().dropna()


def cs_rank(frame: pd.DataFrame, higher_is_better: bool = True) -> pd.DataFrame:
    ranked = frame.rank(axis=1, pct=True)
    if not higher_is_better:
        ranked = 1.0 - ranked
    return (ranked - 0.5) * 2.0


def compute_scores(closes: pd.DataFrame, volumes: pd.DataFrame, benchmark: pd.Series) -> pd.DataFrame:
    bench = benchmark.reindex(closes.index).ffill()

    ret_1m = closes.pct_change(21).sub(bench.pct_change(21), axis=0)
    ret_3m = closes.pct_change(63).sub(bench.pct_change(63), axis=0)
    ret_6m = closes.pct_change(126).sub(bench.pct_change(126), axis=0)
    ret_12m = closes.pct_change(252).sub(bench.pct_change(252), axis=0)

    ema200 = closes.ewm(span=200, adjust=False).mean()
    trend_strength = (closes / ema200 - 1.0).clip(-0.30, 0.30)

    ma20 = closes.rolling(20, min_periods=15).mean()
    sd20 = closes.rolling(20, min_periods=15).std().replace(0, np.nan)
    boll_z = (closes - ma20) / sd20
    mean_reversion = (-boll_z).clip(-2.0, 2.0)

    vol_63 = closes.pct_change().rolling(63, min_periods=40).std() * math.sqrt(252)
    traded_value = (closes * volumes).rolling(20, min_periods=10).mean()
    liquidity = np.log(traded_value.replace(0, np.nan))

    tier_bonus = pd.Series(
        {t: {1: 0.08, 2: 0.00, 3: -0.08}.get(int(STRATEGY_UNIVERSE[t]["liquidity_tier"]), 0.0) for t in closes.columns}
    )

    score = (
        0.10 * cs_rank(ret_1m)
        + 0.25 * cs_rank(ret_3m)
        + 0.27 * cs_rank(ret_6m)
        + 0.13 * cs_rank(ret_12m)
        + 0.12 * cs_rank(trend_strength)
        + 0.08 * cs_rank(mean_reversion)
        + 0.03 * cs_rank(liquidity)
        + 0.02 * cs_rank(vol_63, higher_is_better=False)
    )
    score = score.add(tier_bonus, axis=1)
    score = score.where(closes >= ema200, score - 0.18)
    return score.replace([np.inf, -np.inf], np.nan)


def regime_exposure(date: pd.Timestamp, benchmark: pd.Series, cash_floor: float) -> float:
    hist = benchmark.loc[:date].dropna()
    if len(hist) < 80:
        return 0.0

    now = hist.iloc[-1]
    ma80 = hist.rolling(80, min_periods=40).mean().iloc[-1]
    ma160 = hist.rolling(160, min_periods=80).mean().iloc[-1]
    peak = hist.cummax().iloc[-1]
    dd = now / peak - 1.0

    if now > ma80 and ma80 >= ma160:
        gross = 1.00
    elif now > ma160:
        gross = 0.95
    elif dd > -0.10:
        gross = 0.85
    else:
        gross = 0.70

    if dd < -0.18:
        gross *= 0.85
    if dd < -0.28:
        gross *= 0.70

    return min(1.0 - cash_floor, max(0.0, gross))


def target_weights_for_date(
    date: pd.Timestamp,
    scores: pd.DataFrame,
    closes: pd.DataFrame,
    benchmark: pd.Series,
    top_n: int,
    max_weight: float,
    cash_floor: float,
) -> pd.Series:
    row = scores.loc[date].dropna().sort_values(ascending=False)
    weights = pd.Series(0.0, index=closes.columns)
    if row.empty:
        return weights

    gross_exposure = regime_exposure(date, benchmark, cash_floor)
    if gross_exposure <= 0:
        return weights

    selected = row.head(max(1, min(top_n, len(row))))
    shifted = selected - selected.min() + 0.25
    raw = shifted / shifted.sum() * gross_exposure
    capped = raw.clip(upper=max_weight)

    for _ in range(10):
        spare = gross_exposure - capped.sum()
        if spare <= 1e-9:
            break
        room = (max_weight - capped).clip(lower=0)
        if room.sum() <= 1e-9:
            break
        add = (room / room.sum() * spare).clip(upper=room)
        capped = capped.add(add, fill_value=0.0)

    weights.loc[capped.index] = capped
    return weights


def run_backtest(
    data: dict[str, pd.DataFrame],
    closes: pd.DataFrame,
    volumes: pd.DataFrame,
    capital: float,
    rebalance: str,
    top_n: int,
    max_weight: float,
    cash_floor: float,
    fee_bps: float,
    slippage_bps: float,
) -> tuple[pd.Series, pd.Series, pd.DataFrame, float]:
    benchmark = build_benchmark_from_shared_engine(data, closes.index)
    closes = closes.reindex(benchmark.index).ffill().dropna(how="all")
    volumes = volumes.reindex(closes.index).ffill().fillna(0.0)
    benchmark = benchmark.reindex(closes.index).ffill().dropna()

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
            target = target_weights_for_date(dt, scores, closes, benchmark, top_n, max_weight, cash_floor)
            turnover += float((target - current).abs().sum())
            current = target
        weights.loc[dt] = current

    shifted_weights = weights.shift(1).fillna(0.0)
    strategy_returns = (shifted_weights * daily_returns).sum(axis=1)
    daily_turnover = shifted_weights.diff().abs().sum(axis=1).fillna(0.0)
    strategy_returns = strategy_returns - daily_turnover * cost_rate

    equity = (1.0 + strategy_returns).cumprod() * capital
    bench_equity = (benchmark / benchmark.iloc[0]) * capital
    bench_equity = bench_equity.reindex(equity.index).ffill()
    return equity.dropna(), bench_equity.dropna(), weights, turnover


def max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1.0).min()) if not equity.empty else 0.0


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
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1.0) if years > 0 else 0.0


def summarize(equity: pd.Series, benchmark_equity: pd.Series, weights: pd.DataFrame, turnover: float) -> BacktestStats:
    common = equity.index.intersection(benchmark_equity.index)
    equity = equity.loc[common]
    benchmark_equity = benchmark_equity.loc[common]
    total_ret = equity.iloc[-1] / equity.iloc[0] - 1.0
    bench_ret = benchmark_equity.iloc[-1] / benchmark_equity.iloc[0] - 1.0
    exposure = weights.reindex(common).ffill().sum(axis=1).mean()
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
    parser.add_argument("--cash-floor", type=float, default=0.00)
    parser.add_argument("--fee-bps", type=float, default=3.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--csv", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    tickers = list(STRATEGY_UNIVERSE.keys())
    if not args.quiet:
        print("Tradebot-UTIL.v2 — Backtest UTIL Alpha Allocator")
        print(f"Universo UTIL elegível: {len(tickers)} ativos")
        print("Ativos:", ", ".join(tickers))
        print(f"Período: {args.start} → {args.end} | Capital: R$ {args.capital:,.2f}")
        print("Benchmark compartilhado:", ", ".join(BENCHMARK_UTIL_UNIVERSE.keys()))

    data = download_ohlcv(tickers, args.start, args.end, verbose=not args.quiet)
    if len(data) < 5:
        raise SystemExit("Poucos ativos com dados suficientes. Verifique conexão/datas/yfinance.")

    closes = aligned_field(data, "close")
    volumes = aligned_field(data, "volume").reindex(closes.index).ffill().fillna(0.0)

    equity, bench_equity, weights, turnover = run_backtest(
        data=data,
        closes=closes,
        volumes=volumes,
        capital=args.capital,
        rebalance=args.rebalance,
        top_n=args.top_n,
        max_weight=args.max_weight,
        cash_floor=args.cash_floor,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
    )
    stats = summarize(equity, bench_equity, weights, turnover)

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
