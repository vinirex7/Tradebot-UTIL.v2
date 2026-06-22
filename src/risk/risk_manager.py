"""
Gestão de risco e position sizing.
Implementa Kelly Fracionário, stop loss, drawdown máximo e limite por ativo.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
from loguru import logger


@dataclass
class TradeSignal:
    """Sinal de entrada/saída gerado pelas estratégias."""
    ticker: str
    direction: str         # "long" | "short" | "close"
    strategy: str
    entry_price: float
    stop_loss_price: float
    take_profit_price: float
    confidence: float = 1.0   # 0..1 multiplicador do size
    notes: str = ""


@dataclass
class PositionSize:
    """Resultado do cálculo de tamanho de posição."""
    ticker: str
    shares: int
    capital_allocated: float
    pct_of_portfolio: float
    stop_loss_pct: float


class RiskManager:
    """
    Controla exposição, drawdown e dimensiona posições.

    Parâmetros-chave:
        capital         – Capital total disponível
        max_pos_pct     – Exposição máxima por ativo (0.20 = 20%)
        stop_loss_pct   – Stop loss padrão por operação (0.025 = 2.5%)
        max_drawdown    – Drawdown máximo antes de pausar (0.10 = 10%)
        kelly_fraction  – Fração conservadora do Kelly (0.25)
    """

    def __init__(
        self,
        capital: float,
        max_pos_pct: float = 0.20,
        stop_loss_pct: float = 0.025,
        max_drawdown: float = 0.10,
        kelly_fraction: float = 0.25,
    ):
        self.initial_capital = capital
        self.current_capital = capital
        self.peak_capital = capital
        self.max_pos_pct = max_pos_pct
        self.stop_loss_pct = stop_loss_pct
        self.max_drawdown = max_drawdown
        self.kelly_fraction = kelly_fraction
        self._open_positions: dict[str, float] = {}  # ticker -> capital alocado

    # ──────────────────────────────────────────────
    # Drawdown
    # ──────────────────────────────────────────────

    def update_capital(self, new_equity: float) -> None:
        """Atualiza capital corrente e calcula drawdown."""
        self.current_capital = new_equity
        if new_equity > self.peak_capital:
            self.peak_capital = new_equity

    @property
    def current_drawdown(self) -> float:
        """Drawdown atual em relação ao pico."""
        if self.peak_capital == 0:
            return 0.0
        return (self.peak_capital - self.current_capital) / self.peak_capital

    def is_trading_allowed(self) -> bool:
        """Verifica se drawdown máximo foi atingido."""
        dd = self.current_drawdown
        if dd >= self.max_drawdown:
            logger.warning(
                "Drawdown máximo atingido: {:.2f}% >= {:.2f}%. Trading pausado.",
                dd * 100, self.max_drawdown * 100
            )
            return False
        return True

    # ──────────────────────────────────────────────
    # Position Sizing
    # ──────────────────────────────────────────────

    def calculate_position_size(
        self,
        signal: TradeSignal,
        win_rate: float = 0.55,
        avg_win: float = 0.015,
        avg_loss: float = 0.025,
    ) -> Optional[PositionSize]:
        """
        Calcula o tamanho da posição usando Kelly Fracionário.

        Kelly% = (W*B - L) / B  onde:
          W = win_rate
          B = avg_win / avg_loss (payoff ratio)
          L = 1 - win_rate
        """
        if not self.is_trading_allowed():
            return None

        # Verificar exposição já alocada para este ativo
        existing = self._open_positions.get(signal.ticker, 0.0)
        max_allowed = self.current_capital * self.max_pos_pct

        if existing >= max_allowed:
            logger.warning(
                "Limite de exposição atingido para {}: R$ {:.2f}",
                signal.ticker, existing
            )
            return None

        # Kelly Criterion
        if avg_loss <= 0:
            kelly_pct = 0.0
        else:
            payoff = avg_win / avg_loss
            kelly_full = (win_rate * payoff - (1 - win_rate)) / payoff
            kelly_pct = max(0.0, kelly_full) * self.kelly_fraction

        # Limitar pelo máximo por ativo e capital disponível
        kelly_pct = min(kelly_pct, self.max_pos_pct)
        capital_to_allocate = self.current_capital * kelly_pct * signal.confidence

        # Não ultrapassar o limite do ativo
        available = max_allowed - existing
        capital_to_allocate = min(capital_to_allocate, available)

        if capital_to_allocate <= 0 or signal.entry_price <= 0:
            return None

        # Número de ações (arredondado para múltiplo de lote mínimo)
        lot_size = self._get_lot_size(signal.ticker)
        shares = int(capital_to_allocate / signal.entry_price)
        shares = max(lot_size, (shares // lot_size) * lot_size)

        actual_capital = shares * signal.entry_price

        result = PositionSize(
            ticker=signal.ticker,
            shares=shares,
            capital_allocated=actual_capital,
            pct_of_portfolio=actual_capital / self.current_capital,
            stop_loss_pct=self.stop_loss_pct,
        )

        logger.info(
            "Position size {} | {} ações | R$ {:.2f} ({:.1f}% do capital) | Kelly: {:.2f}%",
            signal.ticker, shares, actual_capital,
            result.pct_of_portfolio * 100, kelly_pct * 100
        )
        return result

    def register_open_position(self, ticker: str, capital: float) -> None:
        """Registra uma posição aberta."""
        self._open_positions[ticker] = self._open_positions.get(ticker, 0.0) + capital
        logger.debug("Posição aberta registrada: {} = R$ {:.2f}", ticker, capital)

    def release_position(self, ticker: str) -> None:
        """Remove posição fechada do controle."""
        if ticker in self._open_positions:
            del self._open_positions[ticker]

    @property
    def total_exposure(self) -> float:
        """Exposição total em capital absoluto."""
        return sum(self._open_positions.values())

    @property
    def exposure_pct(self) -> float:
        """Exposição como % do capital atual."""
        return self.total_exposure / self.current_capital if self.current_capital > 0 else 0.0

    # ──────────────────────────────────────────────
    # Stop Loss dinâmico
    # ──────────────────────────────────────────────

    def calculate_stop_price(
        self, entry_price: float, direction: str, stop_pct: Optional[float] = None
    ) -> float:
        """Calcula preço de stop loss."""
        pct = stop_pct if stop_pct is not None else self.stop_loss_pct
        if direction == "long":
            return entry_price * (1 - pct)
        return entry_price * (1 + pct)

    def calculate_take_profit(
        self, entry_price: float, direction: str, tp_pct: float = 0.015
    ) -> float:
        """Calcula preço de take profit."""
        if direction == "long":
            return entry_price * (1 + tp_pct)
        return entry_price * (1 - tp_pct)

    # ──────────────────────────────────────────────
    # Utilitários
    # ──────────────────────────────────────────────

    @staticmethod
    def _get_lot_size(ticker: str) -> int:
        """
        Retorna o lote mínimo para o ativo.
        Na B3 ações fracionárias têm sufixo F; lote padrão = 100.
        UNTs (TAEE11, ENGI11, SAPR11, ALUP11) usam UNT — lote padrão = 1.
        """
        if ticker.endswith("11"):
            return 1
        return 100

    def portfolio_summary(self) -> dict:
        """Resumo do estado do portfólio."""
        return {
            "capital_inicial": self.initial_capital,
            "capital_atual": self.current_capital,
            "capital_pico": self.peak_capital,
            "drawdown_atual_pct": round(self.current_drawdown * 100, 2),
            "exposicao_total_brl": round(self.total_exposure, 2),
            "exposicao_pct": round(self.exposure_pct * 100, 2),
            "posicoes_abertas": list(self._open_positions.keys()),
            "trading_permitido": self.is_trading_allowed(),
        }
