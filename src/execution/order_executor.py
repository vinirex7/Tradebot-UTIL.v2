"""
Order Executor — Tradebot-UTIL.v2
══════════════════════════════════

Executa ordens de rebalanceamento no MT5 (XP Investimentos).

Responsabilidades:
  - Converter deltas de peso em ordens de compra/venda
  - Calcular quantidade de lotes (lote padrão B3 = 100 ações)
  - Enviar ordens via MT5 com retry e logging
  - Modo paper: simula execução sem enviar ordens reais
  - Reportar resultado de cada ordem

Ordem de execução para minimizar risco de exposição:
  1. VENDAS primeiro (libera capital)
  2. COMPRAS depois (usa capital liberado)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from loguru import logger

try:
    import MetaTrader5 as mt5
    _MT5_AVAILABLE = True
except ImportError:
    _MT5_AVAILABLE = False

from src.strategies.active_momentum_tilt import RebalanceSignal


# ─── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class OrderResult:
    ticker: str
    direction: str          # "buy" | "sell" | "skip"
    shares: int
    requested_weight_delta: float
    executed_price: float
    value: float
    status: str             # "filled" | "partial" | "failed" | "skipped" | "paper"
    error: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RebalanceResult:
    date: datetime
    mode: str               # "live" | "paper"
    orders: list[OrderResult] = field(default_factory=list)
    total_buys: float = 0.0
    total_sells: float = 0.0
    total_cost: float = 0.0
    success: bool = True
    error: str = ""

    @property
    def net_value(self) -> float:
        return self.total_buys - self.total_sells

    def summary(self) -> str:
        filled = [o for o in self.orders if o.status in ("filled", "paper")]
        failed = [o for o in self.orders if o.status == "failed"]
        return (
            f"Rebalanceamento {'LIVE' if self.mode=='live' else 'PAPER'} | "
            f"{len(filled)} ordens OK | {len(failed)} falhas | "
            f"Compras R${self.total_buys:,.0f} | Vendas R${self.total_sells:,.0f} | "
            f"Custo estimado R${self.total_cost:,.0f}"
        )


# ─── Executor ─────────────────────────────────────────────────────────────────

class OrderExecutor:
    """
    Executor de ordens para o Tradebot-UTIL v4.

    Em modo 'paper': simula execução com preços de fechamento + slippage estimado.
    Em modo 'live':  envia ordens de mercado via MT5.
    """

    LOT_SIZE = 100  # B3 standard lot

    def __init__(self, config: dict):
        trading_cfg = config.get("trading", {})
        exec_cfg = config.get("execution", {})

        self.mode: str         = trading_cfg.get("mode", "paper")
        self.capital: float    = float(trading_cfg.get("capital", 100_000.0))
        self.fee_bps: float    = float(exec_cfg.get("fee_bps", 3.0))
        self.slippage_bps: float = float(exec_cfg.get("slippage_bps", 5.0))
        self.min_order_value: float = float(exec_cfg.get("min_order_value", 100.0))
        self.lot_size: int     = int(exec_cfg.get("lot_size", self.LOT_SIZE))
        self.order_type: str   = exec_cfg.get("order_type", "market")
        self.limit_offset_bps: float = float(exec_cfg.get("limit_offset_bps", 5.0))

        self._mt5_ready: bool = False
        if self.mode == "live":
            self._init_mt5(config.get("mt5", {}))

        logger.info(
            "OrderExecutor | modo={} fee={}bps slippage={}bps lote={}",
            self.mode.upper(), self.fee_bps, self.slippage_bps, self.lot_size
        )

    def _init_mt5(self, mt5_cfg: dict) -> None:
        """
        Inicializa a conexão com o MetaTrader 5.

        ESTRATÉGIA DE CONEXÃO (ordem de tentativa):
          1. mt5.initialize() sem credenciais — reutiliza a sessão já aberta
             (obrigatório quando MT5 está logado na XP; passar credenciais
              causa Authorization failed pois a XP bloqueia re-autenticação)
          2. mt5.initialize(path=...) — abre o MT5 a partir do executável
             configurado em mt5.path, sem credenciais (sessão salva no terminal)
          3. mt5.initialize(login=...) — apenas como último recurso, e somente
             se use_credentials=true estiver explicitamente no config
        """
        if not _MT5_AVAILABLE:
            logger.error("MetaTrader5 não instalado — instale com: pip install MetaTrader5")
            return

        import MetaTrader5 as mt5  # noqa

        use_creds   = bool(mt5_cfg.get("use_credentials", False))
        mt5_path    = str(mt5_cfg.get("path", "")).strip() or None
        timeout     = int(mt5_cfg.get("timeout", 60000))

        # ── Tentativa 1: sessão já aberta (sem credenciais) ────────────────
        logger.info("MT5: tentando conectar à sessão já aberta...")
        ok = mt5.initialize()
        if ok:
            info = mt5.account_info()
            if info:
                self._mt5_ready = True
                self.capital = float(info.balance)
                logger.info(
                    "MT5 conectado (sessão existente) | conta={} | servidor={} | saldo=R${:,.2f}",
                    info.login, info.server, info.balance
                )
                return
            else:
                # initialize() retornou True mas conta não está acessível — terminal sem conta ativa
                mt5.shutdown()
                logger.warning("MT5 inicializou mas sem conta ativa — tentando com path")

        # ── Tentativa 2: abre MT5 via path sem credenciais ────────────────
        if mt5_path:
            logger.info("MT5: abrindo terminal via path: {}", mt5_path)
            ok = mt5.initialize(path=mt5_path)
            if ok:
                info = mt5.account_info()
                if info:
                    self._mt5_ready = True
                    self.capital = float(info.balance)
                    logger.info(
                        "MT5 conectado (via path) | conta={} | servidor={} | saldo=R${:,.2f}",
                        info.login, info.server, info.balance
                    )
                    return
                mt5.shutdown()
                logger.warning("MT5 abriu via path mas sem conta ativa")

        # ── Tentativa 3: credenciais explícitas (somente se configurado) ──
        if use_creds:
            login    = int(mt5_cfg.get("login", 0))
            password = str(mt5_cfg.get("password", ""))
            server   = str(mt5_cfg.get("server", ""))
            if not login or not password:
                logger.error("MT5: use_credentials=true mas login/password não preenchidos no config")
                return
            logger.warning(
                "MT5: usando credenciais explícitas (último recurso — pode causar "
                "Authorization failed na XP se outra sessão estiver ativa)"
            )
            init_kwargs = {"timeout": timeout}
            if mt5_path:
                init_kwargs["path"] = mt5_path
            ok = mt5.initialize(login=login, password=password, server=server, **init_kwargs)
            if ok:
                info = mt5.account_info()
                self._mt5_ready = True
                self.capital = float(info.balance) if info else self.capital
                logger.info(
                    "MT5 conectado (credenciais) | conta={} | saldo=R${:,.2f}",
                    login, self.capital
                )
                return

        err = mt5.last_error()
        logger.error(
            "Falha ao conectar ao MT5. Código de erro: {}\n"
            "  → Verifique se o MetaTrader 5 está aberto e logado na XP.\n"
            "  → Se não estiver aberto, configure mt5.path no config.yaml.",
            err
        )

    # ─── Execução principal ────────────────────────────────────────────────

    def execute_rebalance(
        self,
        signal: RebalanceSignal,
        current_prices: dict[str, float],
        current_equity: float,
    ) -> RebalanceResult:
        """
        Executa o rebalanceamento completo a partir de um RebalanceSignal.

        Estratégia de execução:
          1. Calcula quantidade de ações para cada ativo (em lotes)
          2. Ordena: vendas primeiro, depois compras
          3. Envia via MT5 (live) ou simula (paper)
        """
        result = RebalanceResult(
            date=datetime.now(),
            mode=self.mode,
        )

        if not signal.should_rebalance or not signal.deltas:
            logger.info("Sem ordens para executar")
            return result

        # Calcula ordens
        orders_to_send: list[tuple[str, str, int, float]] = []  # (ticker, side, shares, price)
        for ticker, delta_w in signal.deltas.items():
            price = current_prices.get(ticker)
            if price is None or price <= 0:
                logger.warning("Sem preço para {} — pulando", ticker)
                result.orders.append(OrderResult(
                    ticker=ticker, direction="skip", shares=0,
                    requested_weight_delta=delta_w, executed_price=0.0,
                    value=0.0, status="skipped", error="sem_preco",
                ))
                continue

            # Valor alvo da variação
            value_delta = delta_w * current_equity
            shares_raw = value_delta / price
            # Arredonda para múltiplos do lote
            lots = round(shares_raw / self.lot_size)
            shares = abs(lots) * self.lot_size
            actual_value = shares * price

            if actual_value < self.min_order_value or shares == 0:
                logger.debug("{}: valor R${:.0f} abaixo do mínimo — ignorando", ticker, actual_value)
                continue

            side = "buy" if delta_w > 0 else "sell"
            orders_to_send.append((ticker, side, shares, price))

        # Ordena: vendas primeiro
        orders_to_send.sort(key=lambda x: 0 if x[1] == "sell" else 1)

        # Executa
        for ticker, side, shares, price in orders_to_send:
            if self.mode == "live":
                order_result = self._send_mt5_order(ticker, side, shares, price)
            else:
                order_result = self._simulate_order(ticker, side, shares, price)

            result.orders.append(order_result)
            if order_result.status in ("filled", "paper"):
                cost = order_result.value * (self.fee_bps + self.slippage_bps) / 10_000
                result.total_cost += cost
                if side == "buy":
                    result.total_buys += order_result.value
                else:
                    result.total_sells += order_result.value

        logger.info(result.summary())
        return result

    # ─── Envio MT5 ─────────────────────────────────────────────────────────

    def _send_mt5_order(
        self, ticker: str, side: str, shares: int, ref_price: float, retries: int = 3
    ) -> OrderResult:
        if not _MT5_AVAILABLE or not self._mt5_ready:
            logger.error("MT5 não disponível para enviar ordem de {}", ticker)
            return OrderResult(
                ticker=ticker, direction=side, shares=shares,
                requested_weight_delta=0.0, executed_price=ref_price,
                value=0.0, status="failed", error="mt5_indisponivel",
            )

        import MetaTrader5 as mt5  # noqa

        action = mt5.ORDER_TYPE_BUY if side == "buy" else mt5.ORDER_TYPE_SELL

        for attempt in range(retries):
            try:
                # Obtém preço atual
                tick = mt5.symbol_info_tick(ticker)
                if tick is None:
                    raise ValueError(f"Sem tick para {ticker}")

                price = tick.ask if side == "buy" else tick.bid

                # Garante que o símbolo está disponível
                mt5.symbol_select(ticker, True)
                time.sleep(0.1)

                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": ticker,
                    "volume": float(shares / self.lot_size),  # MT5 usa lotes
                    "type": action,
                    "price": price,
                    "deviation": 20,  # slippage máximo em pontos
                    "magic": 20260624,  # ID do bot
                    "comment": "tradebot-util-v4",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }

                result = mt5.order_send(request)

                if result is None:
                    err = mt5.last_error()
                    raise ValueError(f"order_send retornou None: {err}")

                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    logger.info(
                        "Ordem executada | {} {} {} ações @ R${:.2f}",
                        side.upper(), ticker, shares, price
                    )
                    return OrderResult(
                        ticker=ticker, direction=side, shares=shares,
                        requested_weight_delta=0.0, executed_price=price,
                        value=shares * price, status="filled",
                    )
                else:
                    raise ValueError(f"retcode={result.retcode} comment={result.comment}")

            except Exception as exc:
                logger.warning(
                    "Tentativa {}/{} falhou para {} {}: {}",
                    attempt + 1, retries, side, ticker, exc
                )
                time.sleep(2 ** attempt)  # backoff exponencial

        logger.error("Falha definitiva na ordem {} {}", side, ticker)
        return OrderResult(
            ticker=ticker, direction=side, shares=shares,
            requested_weight_delta=0.0, executed_price=ref_price,
            value=0.0, status="failed", error="max_retries_excedido",
        )

    # ─── Simulação paper ────────────────────────────────────────────────────

    def _simulate_order(
        self, ticker: str, side: str, shares: int, ref_price: float
    ) -> OrderResult:
        """Simula execução com slippage de mercado."""
        slip = self.slippage_bps / 10_000
        if side == "buy":
            exec_price = ref_price * (1 + slip)
        else:
            exec_price = ref_price * (1 - slip)

        value = shares * exec_price
        logger.info(
            "[PAPER] {} {} {} ações @ R${:.2f} = R${:,.0f}",
            side.upper(), ticker, shares, exec_price, value
        )
        return OrderResult(
            ticker=ticker, direction=side, shares=shares,
            requested_weight_delta=0.0, executed_price=exec_price,
            value=value, status="paper",
        )

    # ─── Portfolio atual via MT5 ───────────────────────────────────────────

    def get_current_positions(self) -> dict[str, dict]:
        """
        Retorna posições abertas no MT5.

        Returns:
            dict[ticker] → {"shares": int, "avg_price": float, "current_price": float}
        """
        if not _MT5_AVAILABLE or not self._mt5_ready:
            return {}
        try:
            import MetaTrader5 as mt5  # noqa
            positions = mt5.positions_get()
            if positions is None:
                return {}
            result = {}
            for pos in positions:
                result[pos.symbol] = {
                    "shares": int(pos.volume * self.lot_size),
                    "avg_price": float(pos.price_open),
                    "current_price": float(pos.price_current),
                    "profit": float(pos.profit),
                }
            return result
        except Exception as exc:
            logger.error("Erro ao buscar posições MT5: {}", exc)
            return {}

    def get_account_equity(self) -> float:
        """Retorna patrimônio líquido atual da conta MT5."""
        if not _MT5_AVAILABLE or not self._mt5_ready:
            return self.capital
        try:
            import MetaTrader5 as mt5  # noqa
            info = mt5.account_info()
            return float(info.equity) if info else self.capital
        except Exception:
            return self.capital

    def shutdown(self) -> None:
        if _MT5_AVAILABLE and self._mt5_ready:
            try:
                import MetaTrader5 as mt5  # noqa
                mt5.shutdown()
                logger.info("MT5 desconectado")
            except Exception:
                pass
