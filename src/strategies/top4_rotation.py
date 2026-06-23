"""
Estratégia: UTIL Best Assets Rotation
─────────────────────────────────────
Seleciona dinamicamente os melhores ativos do universo UTIL por score
cross-sectional de força relativa, momentum, tendência e risco.

Mudança da infra-1:
- Não fica mais presa a Top 4.
- top_n <= 0 significa: operar todos os ativos elegíveis.
- max_positions limita o número máximo de ativos, se configurado.
- A posição só é vendida se sair do grupo elegível, perder tendência,
  bater stop ou ficar com score muito ruim.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.risk.risk_manager import TradeSignal
from src.utils.indicators import ema


@dataclass
class RotationScore:
    ticker: str
    score: float
    momentum_3m: float
    momentum_6m: float
    momentum_12m: float
    trend_strength: float
    volatility_3m: float
    eligible: bool
    reason: str = ""


@dataclass
class RotationPlan:
    top_tickers: list[str]
    scores: list[RotationScore]
    buy_signals: list[TradeSignal]
    sell_tickers: list[str]


class Top4UTILRotationStrategy:
    name = "top4_rotation"

    def __init__(
        self,
        universe: Optional[list[str]] = None,
        top_n: int = 0,
        max_positions: int = 8,
        rebalance_frequency: str = "weekly",
        weekly_rebalance_day: str = "monday",
        lookback_short: int = 63,
        lookback_mid: int = 126,
        lookback_long: int = 252,
        trend_ema: int = 50,
        vol_lookback: int = 63,
        min_score: float = -0.10,
        exit_score: float = -0.75,
        hard_stop_pct: float = 0.05,
        max_position_pct: float = 0.20,
        w_mom_short: float = 0.25,
        w_mom_mid: float = 0.35,
        w_mom_long: float = 0.20,
        w_trend: float = 0.20,
        w_low_vol: float = 0.10,
    ):
        self.universe = universe or []
        self.top_n = int(top_n)
        self.max_positions = int(max_positions or 0)
        self.rebalance_frequency = rebalance_frequency.lower()
        self.weekly_rebalance_day = weekly_rebalance_day.lower()
        self.lookback_short = int(lookback_short)
        self.lookback_mid = int(lookback_mid)
        self.lookback_long = int(lookback_long)
        self.trend_ema = int(trend_ema)
        self.vol_lookback = int(vol_lookback)
        self.min_score = float(min_score)
        self.exit_score = float(exit_score)
        self.hard_stop_pct = float(hard_stop_pct)
        self.max_position_pct = float(max_position_pct)
        self.weights = {
            "momentum_3m": float(w_mom_short),
            "momentum_6m": float(w_mom_mid),
            "momentum_12m": float(w_mom_long),
            "trend_strength": float(w_trend),
            "low_volatility": float(w_low_vol),
        }

    @property
    def min_bars(self) -> int:
        return max(self.lookback_long, self.trend_ema, self.vol_lookback) + 5

    def should_rebalance(self, dt: Optional[datetime] = None) -> bool:
        if self.rebalance_frequency == "daily":
            return True
        if self.rebalance_frequency == "weekly":
            dt = dt or datetime.now()
            return dt.strftime("%A").lower() == self.weekly_rebalance_day
        return True

    def score_universe(self, ohlcv_by_ticker: dict[str, pd.DataFrame]) -> list[RotationScore]:
        raw_rows: list[dict] = []

        for ticker in self.universe:
            df = ohlcv_by_ticker.get(ticker)
            if df is None or df.empty or "close" not in df.columns or len(df) < self.min_bars:
                raw_rows.append(self._empty_row(ticker, "dados_insuficientes"))
                continue

            close = df["close"].dropna()
            if len(close) < self.min_bars or close.iloc[-1] <= 0:
                raw_rows.append(self._empty_row(ticker, "historico_invalido"))
                continue

            last = float(close.iloc[-1])
            ema_trend = ema(close, self.trend_ema)
            returns = close.pct_change().dropna()

            mom_3m = last / float(close.iloc[-self.lookback_short]) - 1
            mom_6m = last / float(close.iloc[-self.lookback_mid]) - 1
            mom_12m = last / float(close.iloc[-self.lookback_long]) - 1
            trend_strength = last / float(ema_trend.iloc[-1]) - 1
            vol_3m = float(returns.tail(self.vol_lookback).std() * np.sqrt(252))

            trend_ok = last > float(ema_trend.iloc[-1]) and mom_6m > 0
            raw_rows.append({
                "ticker": ticker,
                "eligible": bool(trend_ok),
                "reason": "ok" if trend_ok else "sem_tendencia",
                "momentum_3m": float(mom_3m),
                "momentum_6m": float(mom_6m),
                "momentum_12m": float(mom_12m),
                "trend_strength": float(trend_strength),
                "volatility_3m": float(vol_3m),
            })

        raw = pd.DataFrame(raw_rows)
        if raw.empty:
            return []

        eligible = raw[raw["eligible"]].copy()
        raw["score"] = -999.0

        if not eligible.empty:
            for col in ["momentum_3m", "momentum_6m", "momentum_12m", "trend_strength", "volatility_3m"]:
                std = eligible[col].std(ddof=0)
                if pd.isna(std) or std == 0:
                    eligible[f"z_{col}"] = 0.0
                else:
                    eligible[f"z_{col}"] = (eligible[col] - eligible[col].mean()) / std

            eligible["score"] = (
                self.weights["momentum_3m"] * eligible["z_momentum_3m"]
                + self.weights["momentum_6m"] * eligible["z_momentum_6m"]
                + self.weights["momentum_12m"] * eligible["z_momentum_12m"]
                + self.weights["trend_strength"] * eligible["z_trend_strength"]
                - self.weights["low_volatility"] * eligible["z_volatility_3m"]
            )
            raw.loc[eligible.index, "score"] = eligible["score"]

        scores = []
        for row in raw.sort_values("score", ascending=False).itertuples(index=False):
            scores.append(RotationScore(
                ticker=str(row.ticker),
                score=float(row.score),
                momentum_3m=float(row.momentum_3m) if pd.notna(row.momentum_3m) else 0.0,
                momentum_6m=float(row.momentum_6m) if pd.notna(row.momentum_6m) else 0.0,
                momentum_12m=float(row.momentum_12m) if pd.notna(row.momentum_12m) else 0.0,
                trend_strength=float(row.trend_strength) if pd.notna(row.trend_strength) else 0.0,
                volatility_3m=float(row.volatility_3m) if pd.notna(row.volatility_3m) else 0.0,
                eligible=bool(row.eligible),
                reason=str(row.reason),
            ))
        return scores

    def select_tickers(self, scores: list[RotationScore]) -> list[str]:
        """Seleciona os melhores ativos, sem obrigar limite fixo de quatro."""
        ranked = [s for s in scores if s.eligible and s.score >= self.min_score]
        if self.top_n > 0:
            ranked = ranked[: self.top_n]
        elif self.max_positions > 0:
            ranked = ranked[: self.max_positions]
        return [s.ticker for s in ranked]

    def analyze_universe(
        self,
        ohlcv_by_ticker: dict[str, pd.DataFrame],
        current_positions: Optional[dict[str, dict]] = None,
        force_rebalance: bool = False,
    ) -> RotationPlan:
        current_positions = current_positions or {}
        scores = self.score_universe(ohlcv_by_ticker)
        selected = self.select_tickers(scores)

        sell_tickers: list[str] = []
        for ticker, pos in current_positions.items():
            if ticker not in self.universe:
                continue
            df = ohlcv_by_ticker.get(ticker)
            if df is None or df.empty:
                continue
            close = float(df["close"].iloc[-1])
            entry = float(pos.get("entry_price", 0.0) or 0.0)
            score_obj = next((s for s in scores if s.ticker == ticker), None)
            below_trend = self._below_trend(df)
            hard_stop = entry > 0 and close < entry * (1 - self.hard_stop_pct)
            not_selected = ticker not in selected
            score_exit = score_obj is not None and score_obj.score < self.exit_score
            if below_trend or hard_stop or not_selected or score_exit:
                sell_tickers.append(ticker)

        buy_signals: list[TradeSignal] = []
        for ticker in selected:
            if ticker in current_positions and ticker not in sell_tickers:
                continue
            df = ohlcv_by_ticker.get(ticker)
            if df is None or df.empty:
                continue
            close = float(df["close"].iloc[-1])
            ema_stop = float(ema(df["close"], self.trend_ema).iloc[-1]) * 0.99
            hard_stop = close * (1 - self.hard_stop_pct)
            stop = max(ema_stop, hard_stop)
            score_obj = next((s for s in scores if s.ticker == ticker), None)
            buy_signals.append(TradeSignal(
                ticker=ticker,
                direction="long",
                strategy=self.name,
                entry_price=close,
                stop_loss_price=stop,
                take_profit_price=0.0,
                confidence=1.0,
                notes=(
                    f"UTIL best assets rotation | selected={len(selected)} | score={score_obj.score:.3f} | "
                    f"mom3={score_obj.momentum_3m:.2%} | mom6={score_obj.momentum_6m:.2%} | "
                    f"mom12={score_obj.momentum_12m:.2%} | vol={score_obj.volatility_3m:.2%}"
                    if score_obj else "UTIL best assets rotation"
                ),
            ))

        logger.info(
            "[BestAssetsRotation] selecionados={} {} | compras={} | vendas={}",
            len(selected), selected, [s.ticker for s in buy_signals], sell_tickers,
        )
        return RotationPlan(selected, scores, buy_signals, sell_tickers)

    def _below_trend(self, df: pd.DataFrame) -> bool:
        if df is None or df.empty or len(df) < self.trend_ema + 2:
            return False
        close = df["close"].dropna()
        trend = ema(close, self.trend_ema)
        return float(close.iloc[-1]) < float(trend.iloc[-1]) * 0.99

    @staticmethod
    def _empty_row(ticker: str, reason: str) -> dict:
        return {
            "ticker": ticker,
            "eligible": False,
            "reason": reason,
            "momentum_3m": np.nan,
            "momentum_6m": np.nan,
            "momentum_12m": np.nan,
            "trend_strength": np.nan,
            "volatility_3m": np.nan,
        }
