"""
Tradebot-UTIL.v2 — Entry Point Principal
─────────────────────────────────────────
Orquestra estratégias, feeds de dados e execução de ordens.

Branch main: Momentum Macro é a estratégia ativa e deve espelhar o
backtest atualizado em backtest/backtest_engine.py.

Uso:
    python main.py --mode paper
    python main.py --mode live
    python backtest/run_backtest.py --strategy momentum_macro
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import schedule
import yaml
from loguru import logger

from src.data.macro_feed import MacroFeed
from src.data.mt5_feed import MT5Feed
from src.execution.order_executor import OrderExecutor
from src.risk.risk_manager import RiskManager
from src.strategies import MomentumMacroStrategy, RebalanceAnticipationStrategy
from src.utils.logger import setup_logger


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


class TradebotUTIL:
    """Orquestrador principal do Tradebot-UTIL.v2."""

    def __init__(self, config: dict):
        self.cfg = config
        self.mode = config["trading"]["mode"]

        setup_logger(
            log_file=config["logging"]["file"],
            level=config["logging"]["level"],
        )
        logger.info("=" * 60)
        logger.info("Tradebot-UTIL.v2 iniciando | Modo: {}", self.mode)
        logger.info("=" * 60)

        mt5_cfg = config["mt5"]
        self.mt5 = MT5Feed(
            login=mt5_cfg["login"],
            password=mt5_cfg["password"],
            server=mt5_cfg["server"],
            timeout=mt5_cfg.get("timeout", 60000),
        )
        self.macro = MacroFeed()

        t_cfg = config["trading"]
        self.risk = RiskManager(
            capital=t_cfg["capital"],
            max_pos_pct=t_cfg["max_position_pct"],
            stop_loss_pct=t_cfg["stop_loss_pct"],
            max_drawdown=t_cfg["max_drawdown_pct"],
            kelly_fraction=t_cfg["kelly_fraction"],
        )

        self.executor = OrderExecutor(mode=self.mode)
        self.executor.set_paper_capital(t_cfg["capital"])

        with open("config/universe.yaml", "r") as f:
            universe_cfg = yaml.safe_load(f)
        self.universe = [a["ticker"] for a in universe_cfg["util_composition"]]

        s_cfg = config["strategies"]
        mom_cfg = s_cfg["momentum_macro"]
        self.strategies = {
            "momentum_macro": MomentumMacroStrategy(
                ema_fast=mom_cfg["ema_fast"],
                ema_mid=mom_cfg["ema_mid"],
                ema_slow=mom_cfg["ema_slow"],
                di1_threshold=mom_cfg["di1_threshold"],
                stop_loss_pct=t_cfg["stop_loss_pct"],
                macro_filter_enabled=mom_cfg.get("macro_filter_enabled", False),
                assets=mom_cfg["assets"],
            ),
            "rebalance_anticipation": RebalanceAnticipationStrategy(
                days_before=s_cfg["rebalance_anticipation"]["days_before_rebalance"],
            ),
        }

        self.strategy_weights = {
            "momentum_macro": mom_cfg.get("weight", 0.0),
            "rebalance_anticipation": s_cfg["rebalance_anticipation"].get("weight", 0.0),
        }
        self._running = False

    def start(self) -> None:
        """Inicia o bot."""
        if self.mode in ("live", "paper"):
            if not self.mt5.connect():
                logger.error("Falha ao conectar ao MT5. Abortando.")
                sys.exit(1)
            logger.info("Conexão MT5 estabelecida.")

        self._running = True
        self._update_macro()
        self._schedule_jobs()

        logger.info("Bot ativo. Pressione Ctrl+C para encerrar.")
        try:
            while self._running:
                schedule.run_pending()
                time.sleep(10)
        except KeyboardInterrupt:
            logger.info("Encerrando bot por interrupção do usuário.")
            self.stop()

    def stop(self) -> None:
        self._running = False
        self.mt5.disconnect()
        self._print_summary()

    def _schedule_jobs(self) -> None:
        # Mesmo candle diário usado no backtest: decisões uma vez ao dia.
        schedule.every().day.at("10:05").do(self._run_daily_cycle)
        schedule.every(4).hours.do(self._update_macro)
        schedule.every().day.at("17:15").do(self._print_summary)
        logger.info("Agendamento configurado: ciclo diário 10:05, resumo 17:15.")

    def _load_ohlcv_all(self, timeframe: str = "D1", n_bars: int = 300) -> dict:
        ohlcv_dict = {}
        for ticker in self.universe:
            df = self.mt5.get_ohlcv(ticker, timeframe, n_bars=n_bars)
            if df is not None and not df.empty:
                ohlcv_dict[ticker] = df
        logger.info("OHLCV carregado para {}/{} ativos.", len(ohlcv_dict), len(self.universe))
        return ohlcv_dict

    def _get_open_positions(self) -> dict[str, dict]:
        """Lê posições abertas do paper/live para aplicar a mesma saída do backtest."""
        if self.mode == "paper":
            return self.executor.get_open_paper_positions()

        positions = self.mt5.get_positions()
        if positions is None or positions.empty:
            return {}

        if "magic" in positions.columns:
            positions = positions[positions["magic"] == self.executor.MAGIC_NUMBER]

        out: dict[str, dict] = {}
        for _, row in positions.iterrows():
            ticker = str(row.get("symbol", ""))
            if not ticker:
                continue
            out[ticker] = {
                "direction": "long",
                "shares": int(row.get("volume", 0)),
                "entry_price": float(row.get("price_open", 0.0)),
                "strategy": "momentum_macro",
                "order_id": str(row.get("ticket", "")),
            }
        return out

    def _update_macro(self) -> None:
        """Atualiza macro; por padrão o filtro fica desligado para espelhar o backtest."""
        selic = self.macro.get_selic_rate()
        focus = self.macro.get_di_futures()

        if selic:
            focus_1y = list(focus.values())[0] if focus else None
            regime = self.macro.get_rate_regime(selic, focus_1y)
            self.strategies["momentum_macro"].set_macro_regime(regime)
            logger.info("Macro atualizado | Selic={:.2f}% | Regime={}", selic * 100, regime)

    def _run_daily_cycle(self) -> None:
        if not self.risk.is_trading_allowed():
            return

        logger.info("── Ciclo diário iniciado ──")
        n_bars = self.cfg.get("data", {}).get("price_history_days", 300)
        ohlcv = self._load_ohlcv_all(timeframe="D1", n_bars=n_bars)

        # 1) Primeiro fecha posições que violaram a saída do Momentum Macro.
        self._run_momentum_exit_cycle(ohlcv)

        # 2) Depois avalia novas entradas, igual ao backtest: EMA9/EMA21 + EMA21>EMA50 + MACD bullish.
        if self.cfg["strategies"].get("momentum_macro", {}).get("enabled", False):
            open_positions = self._get_open_positions()
            for ticker in self.strategies["momentum_macro"].assets:
                if ticker in open_positions:
                    continue
                if ticker in ohlcv:
                    signal = self.strategies["momentum_macro"].analyze(ticker, ohlcv[ticker])
                    if signal:
                        self._process_signal(signal, "momentum_macro")

        if self.cfg["strategies"].get("rebalance_anticipation", {}).get("enabled", False):
            reb_signals = self.strategies["rebalance_anticipation"].scan(ohlcv)
            for signal in reb_signals:
                self._process_signal(signal, "rebalance_anticipation")

    def _run_momentum_exit_cycle(self, ohlcv: dict) -> None:
        strategy = self.strategies["momentum_macro"]
        current_positions = self._get_open_positions()

        for ticker, pos in list(current_positions.items()):
            if pos.get("strategy") not in ("momentum_macro", strategy.name):
                continue
            if ticker not in ohlcv:
                continue
            should_exit = strategy.check_exit(
                ticker=ticker,
                ohlcv=ohlcv[ticker],
                position_dir=pos.get("direction", "long"),
                entry_price=pos.get("entry_price"),
            )
            if should_exit:
                price = float(ohlcv[ticker]["close"].iloc[-1])
                result = self.executor.close_position(ticker, price)
                if result:
                    self.risk.release_position(ticker)

    def _process_signal(self, signal, strategy_name: str) -> None:
        weight = self.strategy_weights.get(strategy_name, 0.0)
        if weight <= 0:
            logger.warning("Estratégia {} com peso zero. Sinal ignorado.", strategy_name)
            return

        strategy_capital = self.risk.current_capital * weight
        effective_risk = RiskManager(
            capital=strategy_capital,
            max_pos_pct=self.risk.max_pos_pct,
            stop_loss_pct=self.risk.stop_loss_pct,
            max_drawdown=self.risk.max_drawdown,
            # Config padrão tem kelly_fraction=0.0; aqui usamos 1.0 para que o limite
            # max_position_pct determine o sizing, igual à alocação do backtest.
            kelly_fraction=max(self.risk.kelly_fraction, 1.0),
        )

        pos_size = effective_risk.calculate_position_size(
            signal,
            win_rate=0.60,
            avg_win=0.05,
            avg_loss=0.025,
        )
        if pos_size is None:
            return

        result = self.executor.send_order(signal, pos_size)
        if result:
            self.risk.register_open_position(signal.ticker, pos_size.capital_allocated)

    def _print_summary(self) -> None:
        summary = self.risk.portfolio_summary()
        account = self.mt5.get_account_info() if self.mode == "live" else {}

        logger.info("═" * 50)
        logger.info("RESUMO DO PORTFÓLIO — Tradebot-UTIL.v2")
        for k, v in summary.items():
            logger.info("  {}: {}", k, v)
        if account:
            logger.info("  Saldo MT5: R$ {:.2f}", account.get("balance", 0))
            logger.info("  Equity MT5: R$ {:.2f}", account.get("equity", 0))
        logger.info("═" * 50)

        if self.mode == "paper":
            paper_df = self.executor.get_paper_summary()
            if not paper_df.empty:
                logger.info("Operações paper:\n{}", paper_df.to_string())


def parse_args():
    parser = argparse.ArgumentParser(description="Tradebot-UTIL.v2")
    parser.add_argument(
        "--mode", choices=["live", "paper", "backtest"],
        default="paper",
        help="Modo de operação"
    )
    parser.add_argument(
        "--config", default="config/config.yaml",
        help="Caminho para o arquivo de configuração"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    config_path = args.config
    if not Path(config_path).exists():
        print(f"[ERRO] Arquivo de configuração não encontrado: {config_path}")
        print("Execute: cp config/config.example.yaml config/config.yaml")
        print("E preencha com suas credenciais MT5/XP Investimentos.")
        sys.exit(1)

    config = load_config(config_path)
    config["trading"]["mode"] = args.mode

    bot = TradebotUTIL(config)
    bot.start()
