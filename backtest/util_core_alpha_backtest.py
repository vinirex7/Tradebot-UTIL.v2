"""
Backtest UTIL Hybrid System — Tradebot-UTIL.v2 / branch infra-1
───────────────────────────────────────────────────────────────

Sistema híbrido com diagnóstico por camada:
- Core book
- Mean Reversion
- Momentum Macro
- Pair/Eventos long-only
- Caixa

Além do retorno total, o script imprime a atribuição de resultado por camada
para identificar qual alocação está gerando prejuízo e qual está agregando valor.
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


CORE_TICKERS = ["SBSP3", "AXIA3", "EQTL3", "ENEV3", "CPLE3"]
MEAN_REVERSION_TICKERS = ["SBSP3", "EQTL3", "CPLE3", "EGIE3", "TAEE11", "CMIG4", "CSMG3", "SAPR11"]
MOMENTUM_TICKERS = ["SBSP3", "AXIA3", "EQTL3", "ENEV3", "CPLE3", "CMIG4", "ENGI11", "EGIE3"]
BOOKS = ["core", "mean_reversion", "momentum", "pair_event"]

LIQUIDITY_TIER = {
    "SBSP3": 1, "AXIA3": 1, "EQTL3": 1, "ENEV3": 1, "CPLE3": 1, "CMIG4": 1,
    "ENGI11": 2, "AXIA6": 2, "EGIE3": 2, "ISAE4": 2, "CSMG3": 2, "TAEE11": 2,
    "SAPR11": 2, "CPFE3": 2, "NEOE3": 2, "ALUP11": 3, "ORVR3": 3, "AURE3": 3,
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


def rsi(close: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window, min_periods=window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def cs_rank(frame: pd.DataFrame, higher_is_better: bool = True) -> pd.DataFrame:
    ranked = frame.rank(axis=1, pct=True)
    if not higher_is_better:
        ranked = 1.0 - ranked
    return ((ranked - 0.5) * 2.0).fillna(0.0)


def util_weights(columns: pd.Index, subset: list[str] | None = None) -> pd.Series:
    names = [t for t in columns if t in UTIL_UNIVERSE and (subset is None or t in subset)]
    w = pd.Series({t: UTIL_UNIVERSE[t] for t in names}, dtype=float)
    if w.empty:
        return pd.Series(0.0, index=columns)
    w = w / w.sum()
    return w.reindex(columns).fillna(0.0)


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
    return "adverse"


def compute_indicators(closes: pd.DataFrame, volumes: pd.DataFrame, benchmark: pd.Series) -> dict[str, pd.DataFrame]:
    ma20 = closes.rolling(20, min_periods=15).mean()
    sd20 = closes.rolling(20, min_periods=15).std().replace(0, np.nan)
    z20 = (closes - ma20) / sd20
    rsi14 = rsi(closes, 14)
    ema50 = closes.ewm(span=50, adjust=False, min_periods=30).mean()
    ema200 = closes.ewm(span=200, adjust=False, min_periods=60).mean()
    bench = benchmark.reindex(closes.index).ffill()
    rel_1m = closes.pct_change(21).sub(bench.pct_change(21), axis=0)
    rel_3m = closes.pct_change(63).sub(bench.pct_change(63), axis=0)
    rel_6m = closes.pct_change(126).sub(bench.pct_change(126), axis=0)
    vol = closes.pct_change().rolling(63, min_periods=35).std() * math.sqrt(252)
    traded_value = np.log((closes * volumes).rolling(20, min_periods=10).mean().replace(0, np.nan))
    tier_bonus = pd.Series({t: {1: 0.04, 2: 0.00, 3: -0.04}.get(LIQUIDITY_TIER.get(t, 2), 0.0) for t in closes.columns})
    momentum_score = (
        0.28 * cs_rank(rel_1m)
        + 0.34 * cs_rank(rel_3m)
        + 0.18 * cs_rank(rel_6m)
        + 0.10 * cs_rank(closes / ema50 - 1.0)
        + 0.05 * cs_rank(traded_value)
        + 0.05 * cs_rank(vol, higher_is_better=False)
    ).add(tier_bonus, axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    mr_edge = (
        0.48 * cs_rank(-z20)
        + 0.32 * cs_rank(30 - rsi14)
        + 0.12 * cs_rank(traded_value)
        + 0.08 * cs_rank(vol, higher_is_better=False)
    ).add(tier_bonus, axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return {"z20": z20, "rsi14": rsi14, "ema50": ema50, "ema200": ema200, "momentum_score": momentum_score, "mr_edge": mr_edge}


def cap_and_redistribute(weights: pd.Series, max_asset: float, target_sum: float | None = None) -> pd.Series:
    w = weights.clip(lower=0.0, upper=max_asset)
    target = float(w.sum() if target_sum is None else target_sum)
    for _ in range(10):
        spare = target - float(w.sum())
        if spare <= 1e-10:
            break
        room = (max_asset - w).clip(lower=0.0)
        if room.sum() <= 1e-10:
            break
        w += (room / room.sum() * spare).clip(upper=room)
    return w.clip(lower=0.0, upper=max_asset)


def core_book(columns: pd.Index, exposure: float, max_asset: float) -> pd.Series:
    base = util_weights(columns, CORE_TICKERS) * exposure
    return cap_and_redistribute(base, max_asset, exposure)


def mean_reversion_book(date: pd.Timestamp, closes: pd.DataFrame, indicators: dict[str, pd.DataFrame], regime: str, exposure: float, max_asset: float) -> pd.Series:
    cols = closes.columns
    out = pd.Series(0.0, index=cols)
    if exposure <= 0 or regime == "adverse":
        return out
    z = indicators["z20"].loc[date]
    r = indicators["rsi14"].loc[date]
    ema200 = indicators["ema200"].loc[date]
    edge = indicators["mr_edge"].loc[date]
    price = closes.loc[date]
    eligible = []
    for ticker in MEAN_REVERSION_TICKERS:
        if ticker not in cols:
            continue
        trend_ok = pd.isna(ema200.get(ticker, np.nan)) or price[ticker] >= ema200[ticker] * 0.97
        signal_ok = (z[ticker] <= -1.15 and r[ticker] <= 40) or (z[ticker] <= -1.65 and r[ticker] <= 48)
        if trend_ok and signal_ok:
            eligible.append(ticker)
    if not eligible:
        return out
    selected = edge.loc[eligible].sort_values(ascending=False).head(4)
    shifted = selected - selected.min() + 0.25
    w = shifted / shifted.sum() * exposure
    out.loc[w.index] = w
    return cap_and_redistribute(out, max_asset, min(exposure, out.sum()))


def momentum_book(date: pd.Timestamp, closes: pd.DataFrame, indicators: dict[str, pd.DataFrame], regime: str, exposure: float, max_asset: float, top_n: int) -> pd.Series:
    cols = closes.columns
    out = pd.Series(0.0, index=cols)
    if exposure <= 0 or regime != "bull":
        return out
    score = indicators["momentum_score"].loc[date]
    ema50 = indicators["ema50"].loc[date]
    price = closes.loc[date]
    universe = [t for t in MOMENTUM_TICKERS if t in cols and (pd.isna(ema50[t]) or price[t] >= ema50[t])]
    if not universe:
        return out
    selected = score.loc[universe].sort_values(ascending=False).head(top_n)
    shifted = selected - selected.min() + 0.25
    w = shifted / shifted.sum() * exposure
    out.loc[w.index] = w
    return cap_and_redistribute(out, max_asset, min(exposure, out.sum()))


def pair_tilt_book(date: pd.Timestamp, closes: pd.DataFrame, exposure: float) -> pd.Series:
    out = pd.Series(0.0, index=closes.columns)
    if exposure <= 0 or len(closes.loc[:date]) < 80:
        return out
    pairs = [("EQTL3", "TAEE11"), ("SBSP3", "ENEV3"), ("CPLE3", "CMIG4")]
    chosen = []
    for a, b in pairs:
        if a not in closes.columns or b not in closes.columns:
            continue
        hist = np.log(closes[[a, b]].loc[:date]).dropna().tail(120)
        if len(hist) < 60:
            continue
        spread = hist[a] - hist[b]
        z = (spread.iloc[-1] - spread.mean()) / (spread.std() if spread.std() else np.nan)
        if pd.isna(z):
            continue
        if z > 1.5:
            chosen.append(b)
        elif z < -1.5:
            chosen.append(a)
    if chosen:
        out.loc[chosen] = exposure / len(chosen)
    return out


def layer_exposures(regime: str, args: argparse.Namespace) -> tuple[float, float, float, float]:
    if regime == "bull":
        return args.core_bull, args.mr_bull, args.momentum_bull, args.pair_bull
    if regime == "neutral":
        return args.core_neutral, args.mr_neutral, args.momentum_neutral, args.pair_neutral
    if regime == "adverse":
        return args.core_adverse, args.mr_adverse, args.momentum_adverse, args.pair_adverse
    return 0.0, 0.0, 0.0, 0.0


def target_books_for_date(date: pd.Timestamp, closes: pd.DataFrame, benchmark: pd.Series, indicators: dict[str, pd.DataFrame], args: argparse.Namespace) -> dict[str, pd.Series]:
    regime = market_regime(date, benchmark)
    core_exp, mr_exp, mom_exp, pair_exp = layer_exposures(regime, args)
    books = {
        "core": core_book(closes.columns, core_exp, args.max_asset),
        "mean_reversion": mean_reversion_book(date, closes, indicators, regime, mr_exp, args.max_asset),
        "momentum": momentum_book(date, closes, indicators, regime, mom_exp, args.max_asset, args.top_n),
        "pair_event": pair_tilt_book(date, closes, pair_exp),
    }
    total = sum(books.values(), pd.Series(0.0, index=closes.columns))
    if total.sum() > args.max_exposure:
        scale = args.max_exposure / total.sum()
        books = {name: w * scale for name, w in books.items()}
    return books


def combine_books(books: dict[str, pd.Series], max_asset: float, max_exposure: float) -> pd.Series:
    total = sum(books.values(), pd.Series(0.0, index=next(iter(books.values())).index))
    if total.sum() > max_exposure:
        total *= max_exposure / total.sum()
    return cap_and_redistribute(total, max_asset, min(max_exposure, total.sum()))


def run_backtest(data: dict[str, pd.DataFrame], closes: pd.DataFrame, volumes: pd.DataFrame, capital: float, args: argparse.Namespace) -> tuple[pd.Series, pd.Series, pd.DataFrame, float, dict[str, pd.Series], pd.DataFrame]:
    benchmark = synthetic_util_benchmark(data, target_index=closes.index)
    closes = closes.reindex(benchmark.index).ffill().dropna(how="all")
    volumes = volumes.reindex(closes.index).ffill().fillna(0.0)
    benchmark = benchmark.reindex(closes.index).ffill().dropna()
    indicators = compute_indicators(closes, volumes, benchmark)
    daily_returns = closes.pct_change().fillna(0.0)

    rebalance_dates = closes.index.intersection(closes.resample(args.rebalance).last().index)
    signal_dates = closes.resample(args.signal_frequency).last().index
    rebalance_dates = closes.index.intersection(rebalance_dates.union(signal_dates))

    weights = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
    book_weights = {name: pd.DataFrame(0.0, index=closes.index, columns=closes.columns) for name in BOOKS}
    current_books = {name: pd.Series(0.0, index=closes.columns) for name in BOOKS}
    current_total = pd.Series(0.0, index=closes.columns)
    turnover = 0.0
    book_turnover = {name: 0.0 for name in BOOKS}

    for dt in closes.index:
        if dt in rebalance_dates:
            target_books = target_books_for_date(dt, closes, benchmark, indicators, args)
            target_total = combine_books(target_books, args.max_asset, args.max_exposure)
            raw_total = sum(target_books.values(), pd.Series(0.0, index=closes.columns))
            scale = float(target_total.sum() / raw_total.sum()) if raw_total.sum() > 1e-12 else 0.0
            target_books = {name: w * scale for name, w in target_books.items()}

            turnover += float((target_total - current_total).abs().sum())
            for name in BOOKS:
                book_turnover[name] += float((target_books[name] - current_books[name]).abs().sum())
                current_books[name] = target_books[name]
            current_total = target_total
        weights.loc[dt] = current_total
        for name in BOOKS:
            book_weights[name].loc[dt] = current_books[name]

    cost_rate = (args.fee_bps + args.slippage_bps) / 10_000
    shifted = weights.shift(1).fillna(0.0)
    strategy_returns = (shifted * daily_returns).sum(axis=1)
    daily_turnover = shifted.diff().abs().sum(axis=1).fillna(0.0)
    strategy_returns = strategy_returns - daily_turnover * cost_rate

    book_returns = {}
    for name in BOOKS:
        shifted_book = book_weights[name].shift(1).fillna(0.0)
        gross = (shifted_book * daily_returns).sum(axis=1)
        cost = shifted_book.diff().abs().sum(axis=1).fillna(0.0) * cost_rate
        book_returns[name] = gross - cost

    equity = (1.0 + strategy_returns).cumprod() * capital
    bench_equity = (benchmark / benchmark.iloc[0]) * capital
    bench_equity = bench_equity.reindex(equity.index).ffill()
    return equity.dropna(), bench_equity.dropna(), weights, turnover, book_returns, pd.DataFrame(book_turnover, index=["turnover"]).T


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


def layer_attribution(book_returns: dict[str, pd.Series], weights: pd.DataFrame, book_turnover: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total_return = sum(book_returns.values(), pd.Series(0.0, index=next(iter(book_returns.values())).index))
    for name, rets in book_returns.items():
        contrib = (1.0 + rets).prod() - 1.0
        avg_daily = rets.mean()
        vol = rets.std() * math.sqrt(252) if rets.std() > 0 else 0.0
        shp = (rets.mean() / rets.std() * math.sqrt(252)) if rets.std() > 0 else 0.0
        rows.append({
            "camada": name,
            "contrib_pct": contrib * 100,
            "ret_medio_diario_bp": avg_daily * 10_000,
            "vol_anual_pct": vol * 100,
            "sharpe_aprox": shp,
            "turnover": float(book_turnover.loc[name, "turnover"]) if name in book_turnover.index else 0.0,
        })
    out = pd.DataFrame(rows).set_index("camada").sort_values("contrib_pct", ascending=False)
    out.loc["total_aprox"] = {
        "contrib_pct": ((1 + total_return).prod() - 1) * 100,
        "ret_medio_diario_bp": total_return.mean() * 10_000,
        "vol_anual_pct": total_return.std() * math.sqrt(252) * 100 if total_return.std() > 0 else 0.0,
        "sharpe_aprox": total_return.mean() / total_return.std() * math.sqrt(252) if total_return.std() > 0 else 0.0,
        "turnover": float(book_turnover["turnover"].sum()),
    }
    return out


def save_outputs(equity: pd.Series, benchmark: pd.Series, weights: pd.DataFrame, attribution: pd.DataFrame, output_dir: Path, csv: bool, plot: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if csv:
        pd.DataFrame({"strategy": equity, "benchmark_util": benchmark}).to_csv(output_dir / "util_hybrid_equity.csv")
        weights.to_csv(output_dir / "util_hybrid_weights.csv")
        attribution.to_csv(output_dir / "util_hybrid_attribution.csv")
        print(f"CSV salvo em: {output_dir}")
    if plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(13, 7))
            ax.plot(equity.index, equity.values, label="UTIL Hybrid System")
            ax.plot(benchmark.index, benchmark.values, label="Benchmark UTIL", linestyle="--")
            ax.set_title("Tradebot-UTIL.v2 — UTIL Hybrid System vs Benchmark")
            ax.set_ylabel("Patrimônio (R$)")
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            path = output_dir / "util_hybrid_backtest.png"
            fig.savefig(path, dpi=150)
            print(f"Gráfico salvo em: {path}")
        except Exception as exc:
            print(f"[Aviso] Não foi possível gerar gráfico: {exc}")


def print_attribution(attribution: pd.DataFrame) -> None:
    print("\nAtribuição por camada")
    print("─" * 78)
    print(f"{'Camada':18s} {'Contrib (%)':>12s} {'Média/dia (bp)':>15s} {'Vol (%)':>10s} {'Sharpe':>8s} {'Turnover':>10s}")
    for name, row in attribution.iterrows():
        print(f"{name:18s} {row['contrib_pct']:12.2f} {row['ret_medio_diario_bp']:15.2f} {row['vol_anual_pct']:10.2f} {row['sharpe_aprox']:8.2f} {row['turnover']:10.2f}")
    losers = attribution.drop(index="total_aprox", errors="ignore")
    if not losers.empty:
        worst = losers.sort_values("contrib_pct").iloc[0]
        best = losers.sort_values("contrib_pct", ascending=False).iloc[0]
        print(f"\nPior camada : {worst.name} ({worst['contrib_pct']:+.2f}%)")
        print(f"Melhor camada: {best.name} ({best['contrib_pct']:+.2f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest UTIL Hybrid System")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2026-01-01")
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument("--rebalance", default="M", help="Rebalanceamento do core/momentum: M, W-FRI, 2W-FRI")
    parser.add_argument("--signal-frequency", default="W-FRI", help="Frequência de checagem dos sinais táticos")
    parser.add_argument("--top-n", type=int, default=4)
    parser.add_argument("--max-asset", type=float, default=0.19)
    parser.add_argument("--max-exposure", type=float, default=1.00)
    parser.add_argument("--core-bull", type=float, default=0.60)
    parser.add_argument("--mr-bull", type=float, default=0.25)
    parser.add_argument("--momentum-bull", type=float, default=0.10)
    parser.add_argument("--pair-bull", type=float, default=0.05)
    parser.add_argument("--core-neutral", type=float, default=0.55)
    parser.add_argument("--mr-neutral", type=float, default=0.25)
    parser.add_argument("--momentum-neutral", type=float, default=0.05)
    parser.add_argument("--pair-neutral", type=float, default=0.05)
    parser.add_argument("--core-adverse", type=float, default=0.35)
    parser.add_argument("--mr-adverse", type=float, default=0.00)
    parser.add_argument("--momentum-adverse", type=float, default=0.00)
    parser.add_argument("--pair-adverse", type=float, default=0.05)
    parser.add_argument("--fee-bps", type=float, default=3.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--csv", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    tickers = list(UTIL_UNIVERSE.keys())
    if not args.quiet:
        print("Tradebot-UTIL.v2 — Backtest UTIL Hybrid System")
        print(f"Universo UTIL elegível: {len(tickers)} ativos")
        print("Ativos:", ", ".join(tickers))
        print(f"Período: {args.start} → {args.end} | Capital: R$ {args.capital:,.2f}")
        print("Alocação bull: core {:.0%}, mean reversion {:.0%}, momentum {:.0%}, pairs/eventos {:.0%}".format(args.core_bull, args.mr_bull, args.momentum_bull, args.pair_bull))

    data = download_ohlcv(tickers, args.start, args.end, verbose=not args.quiet)
    if len(data) < 5:
        raise SystemExit("Poucos ativos com dados suficientes. Verifique conexão/datas/yfinance.")
    closes = aligned_field(data, "close")
    volumes = aligned_field(data, "volume").reindex(closes.index).ffill().fillna(0.0)
    equity, benchmark, weights, turnover, book_returns, book_turnover = run_backtest(data, closes, volumes, args.capital, args)
    stats = summarize(equity, benchmark, weights, turnover)
    attribution = layer_attribution(book_returns, weights, book_turnover)

    print("\nResumo — UTIL Hybrid System")
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
    print_attribution(attribution)

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
    save_outputs(equity, benchmark, weights, attribution, Path("logs"), args.csv, args.plot)


if __name__ == "__main__":
    main()
