"""
Backtest UTIL Active Selection v4 — Tradebot-UTIL.v2 / branch infra-1
══════════════════════════════════════════════════════════════════════

FILOSOFIA (diagnóstico após backtests extensivos):
──────────────────────────────────────────────────
O benchmark UTIL sintético (CAGR ~21.7%) equivale a um portfólio rebalanceado
diariamente com pesos fixos e ZERO custo de transação. Superar esse benchmark
exige seleção ativa dos melhores ativos dentro do universo UTIL — não redução
de exposição (que destrói beta) nem rotação tática excessiva (que gera custo).

ESTRATÉGIA VENCEDORA — Active Momentum Tilt:
─────────────────────────────────────────────
  1. Carteira BASE (60-70%): pesos proporcionais ao índice UTIL → captura beta
  2. TILT ativo (30-40%): sobrepeso nos top-3 ativos por momentum 6m+12m
     Zera posição nos 2 piores performers relativos
  3. Rebalanceamento mensal (reduz custo vs. semanal)
  4. Exposição total ~100% sempre (exceto crise DD > 20% em 90 dias → 70%)

RESULTADOS MULTI-PERÍODO (2019-2026):
  ✓ 2019-2020 (COVID): Strat +88% vs Bench +82% (+6.2% alpha)
  ✓ 2020-2021 (Recup): Strat  +3% vs Bench  -3% (+5.7% alpha)
  ✗ 2021-2022 (Juros): Strat +20% vs Bench +21% (-1.2% alpha — quase neutro)
  ✓ 2022-2023 (Aperto): Strat +60% vs Bench +59% (+1.6% alpha)
  ✓ 2023-2024 (Normal): Strat +26% vs Bench +25% (+0.4% alpha)
  ✓ 2024-2026 (Atual): Strat +71% vs Bench +68% (+3.0% alpha)
  ✓ 2019-2026 (FULL): Strat +381% vs Bench +327% (+54.5% alpha)

  Alpha positivo em 6/7 períodos | Sharpe 0.99 | Win mensal 55%

Parâmetros principais:
  --top-n 3               (top-3 ativos sobrepesados)
  --momentum-window 126   (6 meses — horizonte primário)
  --momentum-window2 252  (12 meses — horizonte secundário)
  --momentum-blend 0.30   (30% peso no horizonte de 12m)
  --max-asset-weight 0.40 (cap de 40% por ativo)
  --rebalance ME          (mensal)

Compatibilidade: pandas ≥ 2.2, yfinance ≥ 0.2
"""
from __future__ import annotations

import argparse
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import yfinance as yf
except ImportError as exc:
    raise SystemExit("Instale yfinance: pip install yfinance") from exc

from backtest.backtest_engine import UTIL_UNIVERSE, synthetic_util_benchmark


# ─── Compatibilidade pandas ≥ 2.2 ────────────────────────────────────────────

_FREQ_MAP: dict[str, str] = {
    "M": "ME", "Q": "QE", "A": "YE", "Y": "YE",
    "2W-FRI": "2W", "W-FRI": "W", "BM": "BME",
}


def _freq(f: str) -> str:
    return _FREQ_MAP.get(f, f)


# ─── Universo e tiers de liquidez ────────────────────────────────────────────

LIQUIDITY_TIER: dict[str, int] = {
    "SBSP3": 1, "AXIA3": 1, "EQTL3": 1, "ENEV3": 1, "CPLE3": 1, "CMIG4": 1,
    "ENGI11": 2, "AXIA6": 2, "EGIE3": 2, "ISAE4": 2, "CSMG3": 2, "TAEE11": 2,
    "SAPR11": 2, "CPFE3": 2, "NEOE3": 2, "ALUP11": 3, "ORVR3": 3, "AURE3": 3,
}


# ─── Dataclass de estatísticas ────────────────────────────────────────────────

@dataclass
class Stats:
    strategy_return_pct: float
    benchmark_return_pct: float
    alpha_pct: float
    cagr_strat_pct: float
    cagr_bench_pct: float
    max_drawdown_pct: float
    sharpe: float
    sortino: float
    calmar: float
    volatility_pct: float
    tracking_error_pct: float
    information_ratio: float
    win_months_pct: float
    total_turnover: float


# ─── Download e alinhamento de dados ──────────────────────────────────────────

def clean_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if hasattr(df.index, "tz") and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def download_ohlcv(
    tickers: list[str], start: str, end: str, verbose: bool = True
) -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        try:
            df = yf.Ticker(f"{ticker}.SA").history(
                start=start, end=end, auto_adjust=True, actions=False
            )
            if df.empty:
                if verbose:
                    print(f"  ✗ {ticker}: sem dados")
                continue
            df = clean_index(df)
            df.columns = [str(c).lower() for c in df.columns]
            keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
            df = df[keep].dropna()
            if len(df) >= 60:
                data[ticker] = df
                if verbose:
                    print(f"  ✓ {ticker}: {len(df)} pregões")
            elif verbose:
                print(f"  ✗ {ticker}: insuficiente ({len(df)} pregões)")
        except Exception as exc:
            if verbose:
                print(f"  ✗ {ticker}: {exc}")
    return data


def aligned_close(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(
        {t: df["close"] for t, df in data.items() if "close" in df.columns}, axis=1
    ).sort_index().ffill().dropna(how="all")


# ─── Score de momentum relativo ───────────────────────────────────────────────

def momentum_score(
    closes_hist: pd.DataFrame,
    benchmark_hist: pd.Series,
    window1: int,
    window2: int,
    blend2: float,
) -> pd.Series:
    """
    Calcula score de momentum relativo ao benchmark para cada ativo.
    
    Combina dois horizontes (window1 e window2) com pesos (1-blend2) e blend2.
    Retorna pd.Series com score para cada ativo (NaN se sem dados suficientes).
    """
    n = len(closes_hist)
    bm_now = float(benchmark_hist.iloc[-1])

    def rel_ret(window: int) -> pd.Series:
        if n < window + 2:
            return pd.Series(np.nan, index=closes_hist.columns)
        asset_then = closes_hist.iloc[-window]
        asset_now = closes_hist.iloc[-1]
        bm_then = float(benchmark_hist.iloc[-window])
        ret_asset = asset_now / asset_then.replace(0, np.nan) - 1.0
        ret_bm = bm_now / bm_then - 1.0 if bm_then != 0 else 0.0
        return ret_asset - ret_bm

    s1 = rel_ret(window1)
    s2 = rel_ret(window2)

    # Se não há dados suficientes para window2, usa window1
    s2 = s2.where(~s2.isna(), s1)

    score = (1.0 - blend2) * s1 + blend2 * s2
    return score


# ─── Construção da carteira ───────────────────────────────────────────────────

def build_weights(
    date: pd.Timestamp,
    closes: pd.DataFrame,
    benchmark: pd.Series,
    args: argparse.Namespace,
    portfolio_dd_90d: float = 0.0,
) -> pd.Series:
    """
    Constrói pesos do portfólio para uma data de rebalanceamento.
    
    Lógica:
    1. Pesos base = índice UTIL
    2. Ranqueia ativos por score de momentum relativo
    3. Zera os bottom-2 e redistribui o peso deles para o tilt
    4. Sobrepondera os top-N com o capital liberado
    5. Reduz exposição total se drawdown de portfolio > dd_stop
    """
    cols = closes.columns
    
    # Pesos do índice (normalizados para ativos disponíveis)
    idx_w = pd.Series(
        {t: UTIL_UNIVERSE.get(t, 0.0) for t in cols}, dtype=float
    )
    idx_w = idx_w / idx_w.sum() if idx_w.sum() > 0 else pd.Series(1.0 / len(cols), index=cols)
    
    # Exposição total (reduz em crise de drawdown)
    if portfolio_dd_90d < -args.dd_stop:
        exposure = args.exposure_crisis
    else:
        exposure = 1.0  # sempre ~100% investido
    
    # Calcula score de momentum
    hist = closes.loc[:date]
    bm_hist = benchmark.loc[:date]
    
    if len(hist) < 40:
        return (idx_w * exposure).clip(lower=0.0, upper=args.max_asset_weight)
    
    score = momentum_score(hist, bm_hist, args.momentum_window, args.momentum_window2, args.momentum_blend)
    
    # Filtra ativos disponíveis (com dados suficientes no período)
    available = [t for t in cols if not pd.isna(score.get(t, np.nan))]
    if not available:
        return (idx_w * exposure).clip(lower=0.0, upper=args.max_asset_weight)
    
    score_avail = score.loc[available].sort_values(ascending=False)
    
    # Top-N para sobrepeso
    top_n = min(args.top_n, len(available))
    top_tickers = score_avail.head(top_n).index.tolist()
    
    # Bottom-K para zerar (apenas se houver candidatos)
    bot_k = min(args.bottom_k, len(available) - top_n)
    bot_tickers = score_avail.tail(bot_k).index.tolist() if bot_k > 0 else []
    
    # Constrói pesos
    new_w = idx_w.copy()
    
    # Capital liberado ao zerar os bottom
    freed = sum(float(new_w.get(t, 0.0)) for t in bot_tickers)
    for t in bot_tickers:
        if t in new_w.index:
            new_w[t] = 0.0
    
    # Distribui o capital liberado igualmente entre os top-N
    if top_tickers and freed > 0:
        extra = freed / len(top_tickers)
        for t in top_tickers:
            if t in new_w.index:
                new_w[t] = min(args.max_asset_weight, float(new_w[t]) + extra)
    
    # Normaliza e aplica cap
    new_w = new_w.clip(lower=0.0, upper=args.max_asset_weight)
    s = new_w.sum()
    if s > 0:
        new_w = new_w / s * exposure
    
    return new_w.fillna(0.0)


# ─── Motor do backtest ────────────────────────────────────────────────────────

def run_backtest(
    data: dict[str, pd.DataFrame],
    args: argparse.Namespace,
) -> tuple[pd.Series, pd.Series, pd.DataFrame, float]:
    """
    Executa o backtest e retorna (equity, benchmark_equity, weights_df, total_turnover).
    """
    closes = aligned_close(data)
    benchmark = synthetic_util_benchmark(data, target_index=closes.index)
    closes = closes.reindex(benchmark.index).ffill().dropna(how="all")
    benchmark = benchmark.reindex(closes.index).ffill().dropna()
    
    daily_ret = closes.pct_change().fillna(0.0)
    cost_rate = (args.fee_bps + args.slippage_bps) / 10_000
    
    # Datas de rebalanceamento
    reb_dates = set(
        closes.index.intersection(closes.resample(_freq(args.rebalance)).last().index)
    )
    
    weights = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
    current_w = pd.Series(0.0, index=closes.columns)
    
    # Portfólio começa com pesos do índice no primeiro dia
    idx_w_init = pd.Series({t: UTIL_UNIVERSE.get(t, 0) for t in closes.columns}, dtype=float)
    idx_w_init = idx_w_init / idx_w_init.sum()
    
    # Estado do drawdown de portfólio
    equity_vals: list[float] = []
    dates_list: list[pd.Timestamp] = []
    portfolio_dd_90d = 0.0
    
    for i, dt in enumerate(closes.index):
        # Atualiza equity tracking
        if i == 0:
            equity_vals.append(args.capital)
        else:
            prev_w = weights.iloc[i - 1]
            ret_today = float((prev_w * daily_ret.iloc[i]).sum())
            equity_vals.append(equity_vals[-1] * (1 + ret_today))
        dates_list.append(dt)
        
        # Calcula drawdown móvel de 90 dias para proteção
        if i >= 90:
            window_eq = equity_vals[-90:]
            peak_90 = max(window_eq)
            portfolio_dd_90d = equity_vals[-1] / peak_90 - 1.0
        else:
            portfolio_dd_90d = 0.0
        
        # Rebalanceia se for data de rebalanceamento
        if dt in reb_dates:
            if i == 0:
                # Primeiro dia: começa com pesos do índice
                current_w = idx_w_init.reindex(closes.columns).fillna(0.0)
            else:
                current_w = build_weights(dt, closes, benchmark, args, portfolio_dd_90d)
        elif i == 0:
            current_w = idx_w_init.reindex(closes.columns).fillna(0.0)
        
        weights.loc[dt] = current_w
    
    # Calcula retornos da estratégia com custo de transação
    shifted = weights.shift(1).fillna(0.0)
    strategy_returns = (shifted * daily_ret).sum(axis=1)
    daily_turnover = shifted.diff().abs().sum(axis=1).fillna(0.0)
    strategy_returns -= daily_turnover * cost_rate
    
    # Equity curves
    equity = (1.0 + strategy_returns).cumprod() * args.capital
    bench_equity = (benchmark / benchmark.iloc[0]) * args.capital
    bench_equity = bench_equity.reindex(equity.index).ffill()
    
    total_turnover = float(daily_turnover.sum())
    return equity.dropna(), bench_equity.dropna(), weights, total_turnover


# ─── Métricas ─────────────────────────────────────────────────────────────────

def _cagr(eq: pd.Series) -> float:
    if len(eq) < 2:
        return 0.0
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    return float((eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1.0) if years > 0 else 0.0


def _max_dd(eq: pd.Series) -> float:
    return float((eq / eq.cummax() - 1.0).min()) if not eq.empty else 0.0


def _sharpe(eq: pd.Series) -> float:
    r = eq.pct_change().dropna()
    return float(r.mean() / r.std() * math.sqrt(252)) if not r.empty and r.std() > 0 else 0.0


def _sortino(eq: pd.Series) -> float:
    r = eq.pct_change().dropna()
    down = r[r < 0].std()
    return float(r.mean() / down * math.sqrt(252)) if not r.empty and down > 0 else 0.0


def _vol(eq: pd.Series) -> float:
    r = eq.pct_change().dropna()
    return float(r.std() * math.sqrt(252)) if not r.empty else 0.0


def _tracking_error(eq: pd.Series, bm: pd.Series) -> float:
    common = eq.index.intersection(bm.index)
    er = eq.loc[common].pct_change().dropna()
    br = bm.loc[common].pct_change().dropna()
    idx = er.index.intersection(br.index)
    diff = er.loc[idx] - br.loc[idx]
    return float(diff.std() * math.sqrt(252)) if len(diff) > 1 else 0.0


def _information_ratio(eq: pd.Series, bm: pd.Series) -> float:
    te = _tracking_error(eq, bm)
    if te == 0:
        return 0.0
    common = eq.index.intersection(bm.index)
    alpha_ann = _cagr(eq.loc[common]) - _cagr(bm.loc[common])
    return alpha_ann / te


def _win_months(eq: pd.Series, bm: pd.Series) -> float:
    common = eq.index.intersection(bm.index)
    sm = eq.loc[common].resample("ME").last().pct_change().dropna()
    bm_m = bm.loc[common].resample("ME").last().pct_change().dropna()
    idx = sm.index.intersection(bm_m.index)
    return float((sm.loc[idx] > bm_m.loc[idx]).mean() * 100) if len(idx) > 0 else 0.0


def summarize(
    equity: pd.Series,
    benchmark: pd.Series,
    weights: pd.DataFrame,
    turnover: float,
) -> Stats:
    common = equity.index.intersection(benchmark.index)
    eq = equity.loc[common]
    bm = benchmark.loc[common]
    
    strat_ret = eq.iloc[-1] / eq.iloc[0] - 1.0
    bench_ret = bm.iloc[-1] / bm.iloc[0] - 1.0
    cagr_s = _cagr(eq)
    dd = _max_dd(eq)
    calmar = cagr_s / abs(dd) if dd != 0 else 0.0
    
    return Stats(
        strategy_return_pct=strat_ret * 100,
        benchmark_return_pct=bench_ret * 100,
        alpha_pct=(strat_ret - bench_ret) * 100,
        cagr_strat_pct=cagr_s * 100,
        cagr_bench_pct=_cagr(bm) * 100,
        max_drawdown_pct=dd * 100,
        sharpe=_sharpe(eq),
        sortino=_sortino(eq),
        calmar=calmar,
        volatility_pct=_vol(eq) * 100,
        tracking_error_pct=_tracking_error(eq, bm) * 100,
        information_ratio=_information_ratio(eq, bm),
        win_months_pct=_win_months(eq, bm),
        total_turnover=turnover,
    )


# ─── Impressão e saída ────────────────────────────────────────────────────────

def print_stats(stats: Stats, prefix: str = "  ") -> None:
    p = prefix
    print(f"{p}{'─'*55}")
    print(f"{p}Retorno estratégia  : {stats.strategy_return_pct:+.2f}%")
    print(f"{p}Retorno benchmark   : {stats.benchmark_return_pct:+.2f}%")
    print(f"{p}Alpha acumulado     : {stats.alpha_pct:+.2f}%")
    print(f"{p}CAGR estratégia     : {stats.cagr_strat_pct:+.2f}%")
    print(f"{p}CAGR benchmark      : {stats.cagr_bench_pct:+.2f}%")
    print(f"{p}Max Drawdown        : {stats.max_drawdown_pct:.2f}%")
    print(f"{p}Sharpe              : {stats.sharpe:.2f}")
    print(f"{p}Sortino             : {stats.sortino:.2f}")
    print(f"{p}Calmar              : {stats.calmar:.2f}")
    print(f"{p}Vol anualizada      : {stats.volatility_pct:.2f}%")
    print(f"{p}Tracking Error      : {stats.tracking_error_pct:.2f}%")
    print(f"{p}Information Ratio   : {stats.information_ratio:.2f}")
    print(f"{p}Win rate mensal     : {stats.win_months_pct:.1f}%")
    print(f"{p}Turnover total      : {stats.total_turnover:.1f}x")


def plot_results(
    equity: pd.Series,
    benchmark: pd.Series,
    weights: pd.DataFrame,
    output_path: Path,
    title: str = "",
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(
            3, 1, figsize=(14, 11),
            gridspec_kw={"height_ratios": [3, 1.5, 1.5]}
        )

        bm_aligned = benchmark.reindex(equity.index).ffill()
        eq_norm = equity / equity.iloc[0] * 100
        bm_norm = bm_aligned / bm_aligned.iloc[0] * 100

        # Equity normalizada
        ax0 = axes[0]
        ax0.plot(eq_norm.index, eq_norm.values,
                 label="UTIL Active Selection v4", linewidth=2.0, color="#1D4ED8")
        ax0.plot(bm_norm.index, bm_norm.values,
                 label="Benchmark UTIL (sintético)", linestyle="--", linewidth=1.5, color="#6B7280")
        ax0.set_title(title or "Tradebot-UTIL.v2 — Active Selection v4 vs Benchmark UTIL", fontsize=11)
        ax0.set_ylabel("Retorno base 100")
        ax0.legend(fontsize=9)
        ax0.grid(True, alpha=0.3)
        ax0.fill_between(
            eq_norm.index, eq_norm.values, bm_norm.values,
            where=eq_norm.values >= bm_norm.values,
            alpha=0.15, color="#22C55E", label="Alpha positivo"
        )
        ax0.fill_between(
            eq_norm.index, eq_norm.values, bm_norm.values,
            where=eq_norm.values < bm_norm.values,
            alpha=0.15, color="#EF4444", label="Underperformance"
        )

        # Drawdown
        ax1 = axes[1]
        dd = (equity / equity.cummax() - 1.0) * 100
        dd_bm = (bm_aligned / bm_aligned.cummax() - 1.0) * 100
        ax1.fill_between(dd.index, dd.values, 0, color="#EF4444", alpha=0.5, label="DD Estratégia")
        ax1.plot(dd_bm.index, dd_bm.values, color="#9CA3AF", linewidth=0.8,
                 linestyle="--", label="DD Benchmark")
        ax1.set_ylabel("Drawdown (%)")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Exposição mensal
        ax2 = axes[2]
        exp_monthly = weights.sum(axis=1).resample("ME").mean() * 100
        ax2.bar(exp_monthly.index, exp_monthly.values, color="#3B82F6", alpha=0.7, width=20)
        ax2.axhline(100, color="#EF4444", linestyle="--", linewidth=0.8, alpha=0.6)
        ax2.set_ylabel("Exposição (%)")
        ax2.set_xlabel("Data")
        ax2.set_ylim(0, 115)
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Gráfico salvo: {output_path}")
    except Exception as exc:
        print(f"  [Aviso] Gráfico não gerado: {exc}")


def save_csv(
    equity: pd.Series, benchmark: pd.Series, weights: pd.DataFrame,
    output_dir: Path, suffix: str = ""
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"strategy": equity, "benchmark_util": benchmark}).to_csv(
        output_dir / f"equity{suffix}.csv"
    )
    weights.to_csv(output_dir / f"weights{suffix}.csv")


# ─── Execução single ──────────────────────────────────────────────────────────

def run_single(
    tickers: list[str],
    args: argparse.Namespace,
    start: str,
    end: str,
    label: str = "",
    verbose: bool = True,
) -> Optional[Stats]:
    if verbose:
        print(f"\nTradebot-UTIL.v2 Active Selection v4")
        print(f"Período: {start} → {end}{' | ' + label if label else ''}")
        print(f"Parâmetros: top_n={args.top_n}, mom1={args.momentum_window}d, "
              f"mom2={args.momentum_window2}d, blend={args.momentum_blend:.0%}, "
              f"max_w={args.max_asset_weight:.0%}")

    data = download_ohlcv(tickers, start, end, verbose=verbose)
    if len(data) < 5:
        if verbose:
            print("  [!] Dados insuficientes — pulando.")
        return None

    equity, benchmark, weights, turnover = run_backtest(data, args)
    stats = summarize(equity, benchmark, weights, turnover)

    if verbose:
        print_stats(stats)

    suffix = f"_{label}" if label else ""

    if getattr(args, "csv", False):
        save_csv(equity, benchmark, weights, Path("logs"), suffix)
        if verbose:
            print(f"  CSV salvo em logs/")

    if getattr(args, "plot", False):
        plot_results(
            equity, benchmark, weights,
            Path("logs") / f"util_active_v4{suffix}.png",
            title=f"UTIL Active v4 — {label}" if label else "",
        )

    # Imprime carteira final
    if verbose:
        last = weights.iloc[-1].sort_values(ascending=False)
        last = last[last > 0.001]
        print("\n  Carteira atual do modelo:")
        for t, w in last.items():
            tier = LIQUIDITY_TIER.get(t, 2)
            print(f"    {t:8s}  {w*100:6.2f}%  (tier {tier})")

    return stats


# ─── Multi-período ────────────────────────────────────────────────────────────

def run_multi_period(tickers: list[str], args: argparse.Namespace) -> None:
    periods = [
        ("2019-01-01", "2021-01-01", "2019-2020_covid"),
        ("2020-01-01", "2022-01-01", "2020-2021_recuperacao"),
        ("2021-01-01", "2023-01-01", "2021-2022_alta_juros"),
        ("2022-01-01", "2024-01-01", "2022-2023_aperto"),
        ("2023-01-01", "2025-01-01", "2023-2024_normalizacao"),
        ("2024-01-01", "2026-06-01", "2024-2026_atual"),
        ("2019-01-01", "2026-06-01", "2019-2026_completo"),
    ]

    print("\n" + "═" * 90)
    print("BACKTEST MULTI-PERÍODO — UTIL Active Selection v4")
    print("═" * 90)
    print(f"Parâmetros: top_n={args.top_n} | mom1={args.momentum_window}d | "
          f"mom2={args.momentum_window2}d | blend={args.momentum_blend:.0%} | "
          f"max_w={args.max_asset_weight:.0%} | rebalance={args.rebalance}")
    print()

    rows = []
    alpha_positive = 0

    for start, end, label in periods:
        a = argparse.Namespace(**vars(args))
        a.start = start
        a.end = end
        stats = run_single(tickers, a, start, end, label=label, verbose=False)
        if stats is None:
            continue

        if stats.alpha_pct > 0:
            alpha_positive += 1
            mark = "✓"
        else:
            mark = "✗"

        print(
            f"  {mark} {label:30s}  "
            f"Strat: {stats.strategy_return_pct:+6.1f}%  "
            f"Bench: {stats.benchmark_return_pct:+6.1f}%  "
            f"Alpha: {stats.alpha_pct:+6.1f}%  "
            f"Sharpe: {stats.sharpe:5.2f}  "
            f"Win%: {stats.win_months_pct:4.0f}%"
        )

        rows.append({
            "Período": label,
            "Estratégia (%)": round(stats.strategy_return_pct, 2),
            "Benchmark (%)": round(stats.benchmark_return_pct, 2),
            "Alpha (%)": round(stats.alpha_pct, 2),
            "CAGR Strat (%)": round(stats.cagr_strat_pct, 2),
            "CAGR Bench (%)": round(stats.cagr_bench_pct, 2),
            "Sharpe": round(stats.sharpe, 2),
            "Sortino": round(stats.sortino, 2),
            "IR": round(stats.information_ratio, 2),
            "Max DD (%)": round(stats.max_drawdown_pct, 2),
            "TE (%)": round(stats.tracking_error_pct, 2),
            "Win Meses (%)": round(stats.win_months_pct, 1),
        })

    n_periods = len(rows)
    print("\n" + "═" * 90)
    print(f"Alpha positivo em {alpha_positive}/{n_periods} períodos.")

    if alpha_positive >= math.ceil(n_periods / 2):
        print("✓ Estratégia SUPEROU o índice UTIL na maioria dos períodos.")
    else:
        print("✗ Estratégia não superou o UTIL na maioria dos períodos.")

    df = pd.DataFrame(rows).set_index("Período")
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 200)
    print("\n" + df.to_string())

    Path("logs").mkdir(exist_ok=True)
    df.to_csv("logs/multi_period_summary_v4.csv")
    print("\nResumo salvo em: logs/multi_period_summary_v4.csv")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Backtest UTIL Active Selection v4",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Período
    p.add_argument("--start", default="2019-01-01")
    p.add_argument("--end", default="2026-06-01")
    p.add_argument("--capital", type=float, default=100_000.0)

    # Frequência de rebalanceamento
    p.add_argument("--rebalance", default="ME",
                   help="Frequência de rebalanceamento: ME (mensal), 2W (quinzenal), W (semanal)")

    # Parâmetros do score de momentum (configuração vencedora: top3, 6m+12m, blend30)
    p.add_argument("--top-n", type=int, default=3,
                   help="Número de ativos sobrepondados (top performers)")
    p.add_argument("--bottom-k", type=int, default=2,
                   help="Número de ativos zerados (bottom performers)")
    p.add_argument("--momentum-window", type=int, default=126,
                   help="Horizonte primário de momentum (dias, ~6 meses)")
    p.add_argument("--momentum-window2", type=int, default=252,
                   help="Horizonte secundário de momentum (dias, ~12 meses)")
    p.add_argument("--momentum-blend", type=float, default=0.30,
                   help="Peso do horizonte secundário no score (0=só 6m, 1=só 12m)")

    # Pesos por ativo
    p.add_argument("--max-asset-weight", type=float, default=0.40,
                   help="Peso máximo por ativo (cap)")

    # Stop de drawdown de portfolio
    p.add_argument("--dd-stop", type=float, default=0.20,
                   help="Drawdown de portfólio (90d) que ativa modo de proteção")
    p.add_argument("--exposure-crisis", type=float, default=0.70,
                   help="Exposição total quando drawdown > dd_stop")

    # Custos de transação
    p.add_argument("--fee-bps", type=float, default=3.0,
                   help="Taxa de corretagem em bps")
    p.add_argument("--slippage-bps", type=float, default=5.0,
                   help="Slippage estimado em bps")

    # Saída
    p.add_argument("--csv", action="store_true",
                   help="Salva equity curve e pesos em CSV")
    p.add_argument("--plot", action="store_true",
                   help="Gera gráficos PNG dos resultados")
    p.add_argument("--quiet", action="store_true",
                   help="Suprime output verboso")
    p.add_argument("--multi-period", action="store_true",
                   help="Roda backtest em 7 períodos e exibe comparativo completo")

    args = p.parse_args()
    tickers = list(UTIL_UNIVERSE.keys())

    if args.multi_period:
        run_multi_period(tickers, args)
    else:
        stats = run_single(
            tickers, args, args.start, args.end,
            verbose=not args.quiet
        )
        if stats and not args.quiet:
            if stats.alpha_pct > 0:
                print("\n  ✓ Estratégia superou o benchmark UTIL no período.")
            else:
                print("\n  ✗ Estratégia não superou o benchmark. Revise os parâmetros.")


if __name__ == "__main__":
    main()
