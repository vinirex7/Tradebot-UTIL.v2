"""
Execução de ordens via MetaTrader 5.
Suporta modos: live, paper (simulação local) e backtest.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import MetaTrader5 as mt5
import pandas as pd
from loguru import logger

from src.risk.risk_manager import PositionSize, TradeSignal


@dataclass
class OrderResult:
    """Resultado de uma ordem executada."""
    order_id: str
    ticker: str
    direction: str
    shares: int
    fill_price: float
    commission: float
    timestamp: datetime
    status: str          # "filled" | "partial" | "rejected" | "simulated"
    strategy: str
    notes: str = ""


class OrderExecutor:
    """
    Executa ordens no MetaTrader 5.

    Suporta dois modos:
        live   → envia ordens reais ao MT5 / XP Investimentos
        paper  → simula ordens sem enviar ao broker (paper trading)
    """

    DEVIATION_POINTS = 20   # Slippage máximo permitido em pontos
    MAGIC_NUMBER = 20260621  # Número mágico para identificar ordens do bot

    def __init__(self, mode: str = "paper"):
        self.mode = mode.lower()
        self._paper_positions: dict[str, dict] = {}
        self._paper_history: list[OrderResult] = []
        self._paper_capital = 0.0

    def set_paper_capital(self, capital: float) -> None:
        self._paper_capital = capital

    # ──────────────────────────────────────────────
    # Interface principal
    # ──────────────────────────────────────────────

    def send_order(
        self,
        signal: TradeSignal,
        position_size: PositionSize,
    ) -> Optional[OrderResult]:
        """
        Envia uma ordem de compra ou venda.
        Roteia para live ou paper conforme modo configurado.
        """
        if self.mode == "live":
            return self._send_live_order(signal, position_size)
        elif self.mode == "paper":
            return self._send_paper_order(signal, position_size)
        else:
            logger.error("Modo de execução inválido: {}", self.mode)
            return None

    def close_position(self, ticker: str, current_price: float) -> Optional[OrderResult]:
        """Fecha posição aberta."""
        if self.mode == "paper":
            return self._close_paper_position(ticker, current_price)
        return self._close_live_position(ticker, current_price)

    # ──────────────────────────────────────────────
    # Live Trading (MT5 real)
    # ──────────────────────────────────────────────

    def _send_live_order(
        self,
        signal: TradeSignal,
        position_size: PositionSize,
    ) -> Optional[OrderResult]:
        """Envia ordem real ao MT5."""
        info = mt5.symbol_info(signal.ticker)
        if info is None:
            logger.error("Símbolo não encontrado: {}", signal.ticker)
            return None

        if not info.visible:
            mt5.symbol_select(signal.ticker, True)

        price = mt5.symbol_info_tick(signal.ticker)
        if price is None:
            logger.error("Sem tick para {}", signal.ticker)
            return None

        order_type = mt5.ORDER_TYPE_BUY if signal.direction == "long" else mt5.ORDER_TYPE_SELL
        fill_price = price.ask if signal.direction == "long" else price.bid

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": signal.ticker,
            "volume": float(position_size.shares),
            "type": order_type,
            "price": fill_price,
            "sl": signal.stop_loss_price,
            "tp": signal.take_profit_price,
            "deviation": self.DEVIATION_POINTS,
            "magic": self.MAGIC_NUMBER,
            "comment": f"UTIL.v2|{signal.strategy}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)

        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            error = mt5.last_error()
            logger.error(
                "Ordem rejeitada: {} | retcode={} | erro={}",
                signal.ticker, result.retcode if result else "N/A", error
            )
            return None

        logger.info(
            "Ordem LIVE executada: {} {} {} ações @ R$ {:.2f} | ID={}",
            signal.direction.upper(), signal.ticker,
            position_size.shares, result.price, result.order
        )

        return OrderResult(
            order_id=str(result.order),
            ticker=signal.ticker,
            direction=signal.direction,
            shares=position_size.shares,
            fill_price=result.price,
            commission=result.comment if isinstance(result.comment, float) else 0.0,
            timestamp=datetime.now(),
            status="filled",
            strategy=signal.strategy,
        )

    def _close_live_position(self, ticker: str, current_price: float) -> Optional[OrderResult]:
        """Fecha posição live via MT5."""
        positions = mt5.positions_get(symbol=ticker)
        if not positions:
            logger.warning("Nenhuma posição aberta para {}", ticker)
            return None

        for pos in positions:
            if pos.magic != self.MAGIC_NUMBER:
                continue

            direction = "sell" if pos.type == mt5.ORDER_TYPE_BUY else "buy"
            price = mt5.symbol_info_tick(ticker)
            close_price = price.bid if direction == "sell" else price.ask

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": ticker,
                "volume": pos.volume,
                "type": mt5.ORDER_TYPE_SELL if direction == "sell" else mt5.ORDER_TYPE_BUY,
                "price": close_price,
                "position": pos.ticket,
                "deviation": self.DEVIATION_POINTS,
                "magic": self.MAGIC_NUMBER,
                "comment": "UTIL.v2|close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info("Posição {} fechada @ R$ {:.2f}", ticker, result.price)
                return OrderResult(
                    order_id=str(result.order),
                    ticker=ticker,
                    direction="close",
                    shares=int(pos.volume),
                    fill_price=result.price,
                    commission=0.0,
                    timestamp=datetime.now(),
                    status="filled",
                    strategy="close",
                )
        return None

    # ──────────────────────────────────────────────
    # Paper Trading (simulação)
    # ──────────────────────────────────────────────

    def _send_paper_order(
        self,
        signal: TradeSignal,
        position_size: PositionSize,
    ) -> OrderResult:
        """Simula uma ordem sem enviar ao broker."""
        order_id = str(uuid.uuid4())[:8]
        commission = position_size.capital_allocated * 0.0005  # 0.05% estimado XP

        self._paper_positions[signal.ticker] = {
            "direction": signal.direction,
            "shares": position_size.shares,
            "entry_price": signal.entry_price,
            "stop_loss": signal.stop_loss_price,
            "take_profit": signal.take_profit_price,
            "strategy": signal.strategy,
            "order_id": order_id,
        }

        result = OrderResult(
            order_id=order_id,
            ticker=signal.ticker,
            direction=signal.direction,
            shares=position_size.shares,
            fill_price=signal.entry_price,
            commission=commission,
            timestamp=datetime.now(),
            status="simulated",
            strategy=signal.strategy,
        )
        self._paper_history.append(result)

        logger.info(
            "[PAPER] {} {} {} ações @ R$ {:.2f} | Stop: {:.2f} | TP: {:.2f} | Comissão: {:.2f}",
            signal.direction.upper(), signal.ticker,
            position_size.shares, signal.entry_price,
            signal.stop_loss_price, signal.take_profit_price, commission
        )
        return result

    def _close_paper_position(self, ticker: str, current_price: float) -> Optional[OrderResult]:
        """Fecha posição em modo paper."""
        if ticker not in self._paper_positions:
            logger.warning("[PAPER] Nenhuma posição paper para {}", ticker)
            return None

        pos = self._paper_positions.pop(ticker)
        entry = pos["entry_price"]
        shares = pos["shares"]

        if pos["direction"] == "long":
            pnl = (current_price - entry) * shares
        else:
            pnl = (entry - current_price) * shares

        commission = current_price * shares * 0.0005

        result = OrderResult(
            order_id=str(uuid.uuid4())[:8],
            ticker=ticker,
            direction="close",
            shares=shares,
            fill_price=current_price,
            commission=commission,
            timestamp=datetime.now(),
            status="simulated",
            strategy=pos["strategy"],
            notes=f"P&L: R$ {pnl:.2f}",
        )
        self._paper_history.append(result)

        logger.info(
            "[PAPER] Posição {} fechada @ R$ {:.2f} | P&L: R$ {:.2f} | Comissão: R$ {:.2f}",
            ticker, current_price, pnl, commission
        )
        return result

    def get_paper_summary(self) -> pd.DataFrame:
        """Retorna histórico de operações paper."""
        if not self._paper_history:
            return pd.DataFrame()
        records = [vars(r) for r in self._paper_history]
        return pd.DataFrame(records)

    def get_open_paper_positions(self) -> dict:
        """Retorna posições paper abertas."""
        return dict(self._paper_positions)
