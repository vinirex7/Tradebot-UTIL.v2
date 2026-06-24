"""
Risk Manager — Tradebot-UTIL.v2
════════════════════════════════

Gerenciamento de risco para a estratégia Active Momentum Tilt v4.

Responsabilidades:
  1. Monitorar drawdown do portfólio (90 dias)
  2. Verificar limites globais de exposição
  3. Validar ordens antes do envio
  4. Emitir alertas de risco
  5. Decidir pausa emergencial do bot
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional

from loguru import logger


# ─── Compatibilidade com estratégias legadas ──────────────────────────────────

@dataclass
class TradeSignal:
    """
    Sinal de trade para as estratégias legadas (MomentumMacro, Top4Rotation,
    RebalanceAnticipation). Mantido para compatibilidade retroativa.
    """
    ticker: str
    action: str          # "buy" | "sell" | "hold"
    strength: float = 0.0
    entry_price: float = 0.0
    stop_price: float = 0.0
    reason: str = ""


@dataclass
class RiskCheckResult:
    passed: bool
    reason: str
    action: str = "none"  # "none" | "reduce_exposure" | "pause_bot" | "emergency_stop"


class RiskManager:
    """
    Gerenciador de risco para o Tradebot-UTIL v4.

    Parâmetros configuráveis via YAML (trading:):
      max_drawdown_pct    — DD global máximo antes de pausar o bot
      max_position_pct    — Cap por ativo (linha de defesa extra)
      capital             — Capital total alocado
    """

    def __init__(
        self,
        capital: float,
        max_pos_pct: float = 0.40,
        stop_loss_pct: float = 0.00,
        max_drawdown: float = 0.25,
        kelly_fraction: float = 0.0,
    ):
        self.capital = capital
        self.max_pos_pct = max_pos_pct
        self.stop_loss_pct = stop_loss_pct
        self.max_drawdown = max_drawdown
        self.kelly_fraction = kelly_fraction

        # Histórico de equity para cálculo de drawdown
        self._equity_history: list[tuple[datetime, float]] = []
        self._peak_equity: float = capital
        self._is_paused: bool = False
        self._pause_reason: str = ""

        logger.info(
            "RiskManager | capital=R${:,.0f} max_dd={:.0%} max_pos={:.0%}",
            capital, max_drawdown, max_pos_pct
        )

    # ─── Atualização de estado ─────────────────────────────────────────────

    def update_equity(self, equity: float, timestamp: Optional[datetime] = None) -> None:
        """Registra o patrimônio atual do portfólio."""
        ts = timestamp or datetime.now()
        self._equity_history.append((ts, equity))
        if equity > self._peak_equity:
            self._peak_equity = equity
        # Mantém 180 dias de histórico
        if len(self._equity_history) > 180:
            self._equity_history = self._equity_history[-180:]

    # ─── Cálculo de drawdown ───────────────────────────────────────────────

    def current_drawdown(self) -> float:
        """Drawdown atual desde o pico histórico."""
        if not self._equity_history:
            return 0.0
        current = self._equity_history[-1][1]
        return current / self._peak_equity - 1.0 if self._peak_equity > 0 else 0.0

    def drawdown_90d(self) -> float:
        """Drawdown nos últimos 90 registros de equity."""
        hist = self._equity_history[-90:]
        if len(hist) < 2:
            return 0.0
        peak = max(v for _, v in hist)
        current = hist[-1][1]
        return current / peak - 1.0 if peak > 0 else 0.0

    def drawdown_since(self, days: int) -> float:
        """Drawdown nos últimos N registros."""
        hist = self._equity_history[-days:]
        if len(hist) < 2:
            return 0.0
        peak = max(v for _, v in hist)
        current = hist[-1][1]
        return current / peak - 1.0 if peak > 0 else 0.0

    # ─── Verificações de risco ─────────────────────────────────────────────

    def check_global_risk(self) -> RiskCheckResult:
        """
        Verifica limites globais de risco:
          - Drawdown total > max_drawdown → pausa o bot
        """
        if self._is_paused:
            return RiskCheckResult(
                passed=False,
                reason=f"Bot pausado: {self._pause_reason}",
                action="pause_bot",
            )

        dd = self.current_drawdown()
        if dd < -self.max_drawdown:
            self._is_paused = True
            self._pause_reason = f"Drawdown total {dd:.1%} excedeu limite {-self.max_drawdown:.1%}"
            logger.critical("RISCO CRÍTICO: {} → BOT PAUSADO", self._pause_reason)
            return RiskCheckResult(
                passed=False,
                reason=self._pause_reason,
                action="emergency_stop",
            )

        dd_90 = self.drawdown_90d()
        if dd_90 < -self.max_drawdown * 0.80:
            logger.warning(
                "Alerta de risco: drawdown 90d = {:.1%} (limite: {:.1%})",
                dd_90, -self.max_drawdown
            )
            return RiskCheckResult(
                passed=True,
                reason=f"Alerta: drawdown 90d {dd_90:.1%} próximo do limite",
                action="reduce_exposure",
            )

        return RiskCheckResult(passed=True, reason="OK")

    def check_order(
        self,
        ticker: str,
        target_weight: float,
        order_value: float,
        current_equity: float,
    ) -> RiskCheckResult:
        """
        Valida uma ordem individual antes de envio.

        Verifica:
          1. Peso por ativo não ultrapassa cap
          2. Valor mínimo de ordem
          3. Liquidez (placeholder — verificado no PriceFeed)
        """
        # Cap por ativo
        if target_weight > self.max_pos_pct + 0.01:
            return RiskCheckResult(
                passed=False,
                reason=f"{ticker}: peso alvo {target_weight:.1%} excede cap {self.max_pos_pct:.1%}",
                action="none",
            )

        # Valor mínimo
        if 0 < abs(order_value) < 100.0:
            return RiskCheckResult(
                passed=False,
                reason=f"{ticker}: valor de ordem R$ {order_value:.2f} abaixo do mínimo (R$ 100)",
                action="none",
            )

        return RiskCheckResult(passed=True, reason="OK")

    def validate_rebalance(
        self,
        target_weights: dict[str, float],
        current_equity: float,
    ) -> RiskCheckResult:
        """
        Valida o rebalanceamento completo antes da execução.

        Verifica:
          1. Risco global
          2. Exposição total ≤ 100%
          3. Cap por ativo
        """
        global_check = self.check_global_risk()
        if not global_check.passed and global_check.action == "emergency_stop":
            return global_check

        total = sum(target_weights.values())
        if total > 1.02:
            return RiskCheckResult(
                passed=False,
                reason=f"Exposição total {total:.1%} excede 100%",
                action="none",
            )

        for ticker, w in target_weights.items():
            if w > self.max_pos_pct + 0.01:
                return RiskCheckResult(
                    passed=False,
                    reason=f"{ticker}: peso {w:.1%} excede cap {self.max_pos_pct:.1%}",
                    action="none",
                )

        logger.info(
            "Rebalanceamento validado | {} ativos | exposição total {:.1%} | equity R${:,.0f}",
            len(target_weights), total, current_equity
        )
        return RiskCheckResult(passed=True, reason="OK")

    # ─── Controle de pausa ─────────────────────────────────────────────────

    def resume(self, reason: str = "manual") -> None:
        """Retoma o bot após pausa."""
        self._is_paused = False
        self._pause_reason = ""
        logger.info("Bot retomado: {}", reason)

    @property
    def is_paused(self) -> bool:
        return self._is_paused

    def status_summary(self) -> dict:
        return {
            "is_paused": self._is_paused,
            "pause_reason": self._pause_reason,
            "drawdown_total": self.current_drawdown(),
            "drawdown_90d": self.drawdown_90d(),
            "peak_equity": self._peak_equity,
            "current_equity": self._equity_history[-1][1] if self._equity_history else 0.0,
            "max_drawdown_limit": -self.max_drawdown,
        }
