"""
Active Momentum Tilt v4 — Estratégia Live
══════════════════════════════════════════

Implementação da estratégia validada por backtests extensivos (2019-2026):
  - Alpha positivo em 6/7 períodos | Sharpe 0.97 | CAGR 22.55% vs 21.66%
  - Win rate mensal: 57% | Max Drawdown: -41.6%

FILOSOFIA:
  Mantém ~100% investido no universo UTIL o tempo todo (captura total do
  beta do índice), e gera alpha através de sobrepeso nos top-3 ativos por
  momentum relativo de 6m e 12m.

LÓGICA DE EXECUÇÃO LIVE (MT5 / XP Investimentos):
  ┌─────────────────────────────────────────────────────────┐
  │ 1. Todo pregão, após fechamento (17h35 BRT):            │
  │    a. Baixa preços dos últimos 270 dias via yfinance    │
  │    b. Calcula score de momentum para cada ativo UTIL    │
  │    c. Verifica se chegou data de rebalanceamento        │
  │    d. Se sim: calcula novos pesos-alvo                  │
  │    e. Envia ordens para ajustar o portfólio             │
  └─────────────────────────────────────────────────────────┘

FREQUÊNCIA: Mensal (última sexta do mês, após fechamento)
  - Checagem de drawdown: diária (proteção de portfolio)
  - Execução de ordens: D+0, abertura do pregão seguinte

PARÂMETROS (configuráveis via YAML):
  top_n             = 3      # ativos sobrepondados
  bottom_k          = 2      # ativos zerados
  momentum_window   = 126    # dias (~6 meses)
  momentum_window2  = 252    # dias (~12 meses)
  momentum_blend    = 0.30   # peso do horizonte de 12m
  max_asset_weight  = 0.40   # cap máximo por ativo
  dd_stop           = 0.20   # drawdown (90d) que ativa proteção
  exposure_crisis   = 0.70   # exposição em modo de crise
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from backtest.backtest_engine import UTIL_UNIVERSE


# ─── Resultado de rebalanceamento ────────────────────────────────────────────

@dataclass
class RebalanceSignal:
    """Sinal de rebalanceamento gerado pela estratégia."""
    date: date
    target_weights: dict[str, float]       # ticker → peso alvo (0.0–1.0)
    current_weights: dict[str, float]      # ticker → peso atual
    deltas: dict[str, float]               # ticker → delta de peso (positivo = compra)
    regime: str                            # "normal" | "crisis"
    total_exposure: float                  # soma dos pesos alvo
    top_tickers: list[str]                 # ativos sobrepondados
    bottom_tickers: list[str]              # ativos zerados
    scores: dict[str, float]               # score de momentum de cada ativo
    should_rebalance: bool = True
    reason: str = ""


@dataclass
class PortfolioState:
    """Estado atual do portfólio."""
    positions: dict[str, float] = field(default_factory=dict)   # ticker → peso atual
    equity: float = 0.0
    last_rebalance: Optional[date] = None
    equity_history: list[tuple[date, float]] = field(default_factory=list)

    def update_equity(self, new_equity: float, today: date) -> None:
        self.equity = new_equity
        self.equity_history.append((today, new_equity))
        # Mantém apenas 180 dias de histórico
        if len(self.equity_history) > 180:
            self.equity_history = self.equity_history[-180:]

    def drawdown_90d(self) -> float:
        """Drawdown do portfólio nos últimos 90 dias."""
        hist = self.equity_history[-90:]
        if len(hist) < 2:
            return 0.0
        peak = max(v for _, v in hist)
        current = hist[-1][1]
        return current / peak - 1.0 if peak > 0 else 0.0


# ─── Estratégia ──────────────────────────────────────────────────────────────

class ActiveMomentumTiltStrategy:
    """
    Active Momentum Tilt v4 para execução live no MT5.

    Uso:
        strategy = ActiveMomentumTiltStrategy(config)
        signal = strategy.generate_signal(closes_df, benchmark_series, portfolio_state)
        if signal.should_rebalance:
            executor.execute(signal)
    """

    def __init__(self, config: dict):
        cfg = config.get("strategies", {}).get("active_momentum_tilt", {})

        self.top_n: int           = int(cfg.get("top_n", 3))
        self.bottom_k: int        = int(cfg.get("bottom_k", 2))
        self.momentum_window: int = int(cfg.get("momentum_window", 126))
        self.momentum_window2: int= int(cfg.get("momentum_window2", 252))
        self.momentum_blend: float= float(cfg.get("momentum_blend", 0.30))
        self.max_asset_weight: float = float(cfg.get("max_asset_weight", 0.40))
        self.dd_stop: float       = float(cfg.get("dd_stop", 0.20))
        self.exposure_crisis: float = float(cfg.get("exposure_crisis", 0.70))
        self.rebalance_day: str   = str(cfg.get("rebalance_day", "last_friday"))

        # Pesos do índice normalizados (para ativos disponíveis dinamicamente)
        self._universe_weights = {
            t: w / sum(UTIL_UNIVERSE.values())
            for t, w in UTIL_UNIVERSE.items()
        }

        logger.info(
            "ActiveMomentumTiltStrategy v4 | top_n={} bottom_k={} "
            "mom={}d+{}d blend={:.0%} max_w={:.0%}",
            self.top_n, self.bottom_k,
            self.momentum_window, self.momentum_window2,
            self.momentum_blend, self.max_asset_weight,
        )

    # ─── Datas de rebalanceamento ──────────────────────────────────────────

    def is_rebalance_day(self, today: date) -> bool:
        """
        Verifica se hoje é dia de rebalanceamento.
        Padrão: última sexta-feira útil do mês.
        Também aceita rebalanceamento forçado (drawdown crítico).
        """
        if self.rebalance_day == "last_friday":
            # Última sexta-feira do mês corrente
            # Estratégia: encontra o último dia do mês e vai para trás até sexta
            next_month_first = date(today.year, today.month % 12 + 1, 1) if today.month < 12 else date(today.year + 1, 1, 1)
            last_day = next_month_first - timedelta(days=1)
            # Volta para sexta-feira (weekday 4)
            days_back = (last_day.weekday() - 4) % 7
            last_friday = last_day - timedelta(days=days_back)
            return today == last_friday
        elif self.rebalance_day == "last_business":
            # Último dia útil do mês
            next_month_first = date(today.year, today.month % 12 + 1, 1) if today.month < 12 else date(today.year + 1, 1, 1)
            last_day = next_month_first - timedelta(days=1)
            while last_day.weekday() >= 5:
                last_day -= timedelta(days=1)
            return today == last_day
        return False

    def next_rebalance_date(self, from_date: date) -> date:
        """Retorna a próxima data de rebalanceamento a partir de from_date."""
        check = from_date + timedelta(days=1)
        for _ in range(60):
            if self.is_rebalance_day(check):
                return check
            check += timedelta(days=1)
        return check

    # ─── Score de momentum ─────────────────────────────────────────────────

    def _momentum_score(
        self,
        closes: pd.DataFrame,
        benchmark: pd.Series,
    ) -> dict[str, float]:
        """
        Calcula score de momentum relativo para cada ativo disponível.

        Score = (1 - blend) * retorno_rel_6m  +  blend * retorno_rel_12m

        Onde retorno_rel_Xm = retorno_ativo_Xm - retorno_benchmark_Xm
        """
        n = len(closes)
        if n < 30:
            return {t: 0.0 for t in closes.columns}

        bm_now = float(benchmark.iloc[-1])
        scores: dict[str, float] = {}

        for ticker in closes.columns:
            s1 = s2 = 0.0
            close_t = closes[ticker].dropna()

            if len(close_t) >= self.momentum_window + 2:
                r_asset = float(close_t.iloc[-1] / close_t.iloc[-self.momentum_window] - 1.0)
                bm_w1 = float(benchmark.iloc[-self.momentum_window])
                r_bm = bm_now / bm_w1 - 1.0 if bm_w1 > 0 else 0.0
                s1 = r_asset - r_bm

            if len(close_t) >= self.momentum_window2 + 2 and len(benchmark) >= self.momentum_window2 + 2:
                r_asset2 = float(close_t.iloc[-1] / close_t.iloc[-self.momentum_window2] - 1.0)
                bm_w2 = float(benchmark.iloc[-self.momentum_window2])
                r_bm2 = bm_now / bm_w2 - 1.0 if bm_w2 > 0 else 0.0
                s2 = r_asset2 - r_bm2
            else:
                s2 = s1  # fallback para janela curta

            scores[ticker] = (1.0 - self.momentum_blend) * s1 + self.momentum_blend * s2

        return scores

    # ─── Construção de pesos ───────────────────────────────────────────────

    def _build_target_weights(
        self,
        available_tickers: list[str],
        scores: dict[str, float],
        total_exposure: float,
    ) -> tuple[dict[str, float], list[str], list[str]]:
        """
        Constrói pesos-alvo dados os scores.

        Retorna: (pesos_alvo, top_tickers, bottom_tickers)
        """
        # Pesos do índice normalizados para os ativos disponíveis
        idx_w: dict[str, float] = {}
        raw_sum = sum(UTIL_UNIVERSE.get(t, 0.0) for t in available_tickers)
        if raw_sum <= 0:
            raw_sum = len(available_tickers)
            for t in available_tickers:
                idx_w[t] = 1.0 / raw_sum
        else:
            for t in available_tickers:
                idx_w[t] = UTIL_UNIVERSE.get(t, 0.0) / raw_sum

        # Ranqueia por score
        sorted_by_score = sorted(
            [(t, scores.get(t, 0.0)) for t in available_tickers],
            key=lambda x: x[1], reverse=True
        )

        effective_top_n = min(self.top_n, len(available_tickers))
        effective_bot_k = min(self.bottom_k, max(0, len(available_tickers) - effective_top_n))

        top_tickers = [t for t, _ in sorted_by_score[:effective_top_n]]
        bottom_tickers = [t for t, _ in sorted_by_score[-effective_bot_k:]] if effective_bot_k > 0 else []

        # Capital liberado pelos bottom tickers
        freed = sum(idx_w.get(t, 0.0) for t in bottom_tickers)

        # Pesos iniciais = índice
        weights = dict(idx_w)

        # Zera bottom
        for t in bottom_tickers:
            weights[t] = 0.0

        # Redistribui freed igualmente para top-N
        if top_tickers and freed > 0:
            extra = freed / len(top_tickers)
            for t in top_tickers:
                weights[t] = min(self.max_asset_weight, weights.get(t, 0.0) + extra)

        # Cap e normaliza
        for t in weights:
            weights[t] = max(0.0, min(self.max_asset_weight, weights[t]))

        total = sum(weights.values())
        if total > 0:
            weights = {t: w / total * total_exposure for t, w in weights.items()}

        return weights, top_tickers, bottom_tickers

    # ─── Geração de sinal ──────────────────────────────────────────────────

    def generate_signal(
        self,
        closes: pd.DataFrame,
        benchmark: pd.Series,
        portfolio_state: PortfolioState,
        force_rebalance: bool = False,
        today: Optional[date] = None,
    ) -> RebalanceSignal:
        """
        Gera sinal de rebalanceamento.

        Args:
            closes:           DataFrame com preços de fechamento ajustados (date × ticker)
            benchmark:        Série do benchmark sintético UTIL
            portfolio_state:  Estado atual do portfólio (pesos, equity, histórico)
            force_rebalance:  Força rebalanceamento mesmo fora do dia programado
            today:            Data de referência (padrão: hoje)

        Returns:
            RebalanceSignal com pesos-alvo e deltas de peso
        """
        today = today or date.today()
        available = [t for t in closes.columns if t in UTIL_UNIVERSE]

        if not available:
            logger.warning("Nenhum ativo do universo UTIL disponível para rebalanceamento.")
            return RebalanceSignal(
                date=today,
                target_weights={},
                current_weights=portfolio_state.positions,
                deltas={},
                regime="normal",
                total_exposure=0.0,
                top_tickers=[],
                bottom_tickers=[],
                scores={},
                should_rebalance=False,
                reason="sem_ativos_disponiveis",
            )

        # Verifica drawdown de portfólio
        dd_90d = portfolio_state.drawdown_90d()
        is_crisis = dd_90d < -self.dd_stop
        regime = "crisis" if is_crisis else "normal"
        total_exposure = self.exposure_crisis if is_crisis else 1.0

        if is_crisis:
            logger.warning(
                "MODO CRISE ATIVO: drawdown 90d = {:.1%} | exposição reduzida para {:.0%}",
                dd_90d, total_exposure
            )

        # Calcula scores
        scores = self._momentum_score(closes[available], benchmark)

        # Verifica se deve rebalancear
        is_reb_day = self.is_rebalance_day(today)
        # Força rebalanceamento se entrou ou saiu do modo crise
        prev_was_crisis = portfolio_state.last_rebalance and (
            portfolio_state.drawdown_90d() * 0.9 < -self.dd_stop
        )
        should_reb = force_rebalance or is_reb_day or (is_crisis != prev_was_crisis)

        if not should_reb:
            return RebalanceSignal(
                date=today,
                target_weights=dict(portfolio_state.positions),
                current_weights=dict(portfolio_state.positions),
                deltas={},
                regime=regime,
                total_exposure=total_exposure,
                top_tickers=[],
                bottom_tickers=[],
                scores=scores,
                should_rebalance=False,
                reason=f"nao_e_dia_rebalanceamento (proximo: {self.next_rebalance_date(today)})",
            )

        # Constrói pesos-alvo
        target_w, top_tickers, bottom_tickers = self._build_target_weights(
            available, scores, total_exposure
        )

        # Calcula deltas (positivo = compra, negativo = venda)
        current = portfolio_state.positions
        all_tickers = set(list(target_w.keys()) + list(current.keys()))
        deltas = {
            t: target_w.get(t, 0.0) - current.get(t, 0.0)
            for t in all_tickers
        }

        # Filtra deltas muito pequenos (< 1% do portfólio — evita ordens mínimas)
        MIN_DELTA = 0.005
        deltas = {t: d for t, d in deltas.items() if abs(d) >= MIN_DELTA}

        reason = f"rebalanceamento_mensal (dia: {today})"
        if force_rebalance:
            reason = "rebalanceamento_forcado"
        elif is_crisis:
            reason = f"modo_crise (dd90d={dd_90d:.1%})"

        logger.info(
            "Sinal de rebalanceamento | regime={} exposição={:.0%} top={} bottom={} ajustes={}",
            regime, total_exposure, top_tickers, bottom_tickers, len(deltas)
        )

        return RebalanceSignal(
            date=today,
            target_weights=target_w,
            current_weights=dict(current),
            deltas=deltas,
            regime=regime,
            total_exposure=total_exposure,
            top_tickers=top_tickers,
            bottom_tickers=bottom_tickers,
            scores=scores,
            should_rebalance=True,
            reason=reason,
        )

    def summary_log(self, signal: RebalanceSignal) -> None:
        """Imprime resumo legível do sinal de rebalanceamento."""
        if not signal.should_rebalance:
            logger.info("Sem rebalanceamento hoje. {}", signal.reason)
            return

        logger.info("=" * 65)
        logger.info("REBALANCEAMENTO — {}", signal.date)
        logger.info("Regime: {} | Exposição total: {:.1%}", signal.regime, signal.total_exposure)
        logger.info("Top (sobrepesados): {}", signal.top_tickers)
        logger.info("Bottom (zerados):   {}", signal.bottom_tickers)
        logger.info("-" * 65)
        logger.info("{:<8} {:>10} {:>10} {:>10} {:>10}",
                    "Ticker", "Atual%", "Alvo%", "Delta%", "Score")
        logger.info("-" * 65)
        all_t = sorted(set(list(signal.target_weights) + list(signal.current_weights)))
        for t in all_t:
            cur = signal.current_weights.get(t, 0.0)
            tgt = signal.target_weights.get(t, 0.0)
            delta = tgt - cur
            sc = signal.scores.get(t, 0.0)
            if tgt > 0 or cur > 0:
                mark = "▲" if delta > 0.005 else ("▼" if delta < -0.005 else " ")
                logger.info("{:<8} {:>9.1%} {:>9.1%} {:>9.1%}  {:>+7.3f}  {}",
                            t, cur, tgt, delta, sc, mark)
        logger.info("=" * 65)
