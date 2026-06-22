"""
Tradebot-UTIL.v2 — Entry Point Principal
─────────────────────────────────────────
Orquestra todas as estratégias, feeds de dados e execução de ordens.
Uso:
    python main.py --mode paper       # Paper trading (padrão)
    python main.py --mode live        # Live trading (use com cuidado!)
    python main.py --mode backtest    # Backtest (use run_backtest.py)
"""
import argparse
import sys
import time
import schedule
from pathlib import Path
import yaml
from loguru import logger

from src.utils.logger import setup_logger
from src.data.mt5_feed import MT5Feed
from src.data.macro_feed import MacroFeed
from src.risk.risk_manager import RiskManager
from src.execution.order_executor import OrderExecutor
from src.strategies import (
    MomentumMacroStrategy,
    RebalanceAnticipationStrategy,
)


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


class TradebotUTIL:
    """Orquestrador principal do Tradebot-UTIL.v2."""

    def __init__(self, config: dict):
        self.cfg = config
        self.mode = config["trading"]["mode"]

        # Setup logger
        setup_logger(
            log_file=config["logging"]["file"],
            level=config["logging"]["level"],
        )
        logger.info("=" * 60)
        logger.info("Tradebot-UTIL.v2 iniciando | Modo: {}", self.mode)
        logger.info("=" * 60)

        # Feeds de dados
        mt5_cfg = config["mt5"]
        self.mt5 = MT5Feed(
            login=mt5_cfg["login"],
            password=mt5_cfg["password"],
            server=mt5_cfg["server"],
            timeout=mt5_cfg.get("timeout", 60000),
        )
        self.macro = MacroFeed()

        # Risk Manager
        t_cfg = config["trading"]
        self.risk = RiskManager(
            capital=t_cfg["capital"],
            max_pos_pct=t_cfg["max_position_pct"],
            stop_loss_pct=t_cfg["stop_loss_pct"],
            max_drawdown=t_cfg["max_drawdown_pct"],
            kelly_fraction=t_cfg["kelly_fraction"],
        )

        # Executor de ordens
        self.executor = OrderExecutor(mode=self.mode)
        self.executor.set_paper_capital(t_cfg["capital"])

        # Estratégias
        s_cfg = config["strategies"]
        self.strategies = {
            "momentum_macro": MomentumMacroStrategy(
                ema_fast=s_cfg["momentum_macro"]["ema_fast"],
                ema_mid=s_cfg["momentum_macro"]["ema_mid"],
                ema_slow=s_cfg["momentum_macro"]["ema_slow"],
                di1_threshold=s_cfg["momentum_macro"]["di1_threshold"],
                assets=s_cfg["momentum_macro"]["assets"],
            ),
            "rebalance_anticipation": RebalanceAnticipationStrategy(
                days_before=s_cfg["rebalance_anticipation"]["days_before_rebalance"],
            ),
        }

        # Universo de ativos
        with open("config/universe.yaml", "r") as f:
            universe_cfg = yaml.safe_load(f)
        self.universe = [a["ticker"] for a in universe_cfg["util_composition"]]

        # Pesos das estratégias para alocação de capital
        self.strategy_weights = {
            "momentum_macro": s_cfg["momentum_macro"]["weight"],
            "rebalance_anticipation": s_cfg["rebalance_anticipation"]["weight"],
        }

        self._running = False

    # ──────────────────────────────────────────────
    # Inicialização
    # ──────────────────────────────────────────────

    def start(self) -> None:
        """Inicia o bot."""
        if self.mode in ("live", "paper"):
            if not self.mt5.connect():
                logger.error("Falha ao conectar ao MT5. Abortando.")
                sys.exit(1)
            logger.info("Conexão MT5 estabelecida.")

        self._running = True

        # Atualizar macro na inicialização
        self._update_macro()

        # Agendar tarefas (ciclo intraday removido — mean_reversion excluído)
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

    # ──────────────────────────────────────────────
    # Agendamento de tarefas
    # ──────────────────────────────────────────────

    def _schedule_jobs(self) -> None:
        """Define horários de execução das rotinas."""
        # Análise diária (abertura do pregão)
        schedule.every().day.at("10:05").do(self._run_daily_cycle)

        # Atualização de dados macro (a cada 4h)
        schedule.every(4).hours.do(self._update_macro)

        # Relatório de portfólio (fim do dia)
        schedule.every().day.at("17:15").do(self._print_summary)

        logger.info("Agendamento configurado.")

    # ──────────────────────────────────────────────
    # Ciclos de análise
    # ──────────────────────────────────────────────

    def _load_ohlcv_all(self, timeframe: str = "D1", n_bars: int = 300) -> dict:
        """Carrega OHLCV de todos os ativos do universo."""
        ohlcv_dict = {}
        for ticker in self.universe:
            df = self.mt5.get_ohlcv(ticker, timeframe, n_bars=n_bars)
            if df is not None and not df.empty:
                ohlcv_dict[ticker] = df
        logger.info("OHLCV carregado para {}/{} ativos.", len(ohlcv_dict), len(self.universe))
        return ohlcv_dict

    def _update_macro(self) -> None:
        """Atualiza dados macroeconômicos e ajusta estratégia de momentum."""
        selic = self.macro.get_selic_rate()
        focus = self.macro.get_di_futures()

        if selic:
            focus_1y = None
            if focus:
                focus_1y = list(focus.values())[0] if focus else None
            regime = self.macro.get_rate_regime(selic, focus_1y)
            self.strategies["momentum_macro"].set_macro_regime(regime)
            logger.info("Macro atualizado | Selic={:.2f}% | Regime={}", selic * 100, regime)

    def _run_daily_cycle(self) -> None:
        """Ciclo diário: Momentum Macro + antecipação de rebalanceamento."""
        if not self.risk.is_trading_allowed():
            return

        logger.info("── Ciclo diário iniciado ──")
        ohlcv = self._load_ohlcv_all(timeframe="D1", n_bars=300)

        # Momentum Macro — única estratégia ativa (100% do capital)
        if self.cfg["strategies"]["momentum_macro"]["enabled"]:
            for ticker in self.strategies["momentum_macro"].assets:
                if ticker in ohlcv:
                    signal = self.strategies["momentum_macro"].analyze(ticker, ohlcv[ticker])
                    if signal:
                        self._process_signal(signal, "momentum_macro")

        # Antecipação de Rebalanceamento
        if self.cfg["strategies"]["rebalance_anticipation"]["enabled"]:
            reb_signals = self.strategies["rebalance_anticipation"].scan(ohlcv)
            for signal in reb_signals:
                self._process_signal(signal, "rebalance_anticipation")

    # ──────────────────────────────────────────────
    # Processamento de sinais
    # ──────────────────────────────────────────────

    def _process_signal(self, signal, strategy_name: str) -> None:
        """Calcula position size e envia ordem."""
        # Ajustar capital disponível por peso da estratégia
        weight = self.strategy_weights.get(strategy_name, 0.1)
        strategy_capital = self.risk.current_capital * weight
        effective_risk = RiskManager(
            capital=strategy_capital,
            max_pos_pct=self.risk.max_pos_pct,
            stop_loss_pct=self.risk.stop_loss_pct,
            max_drawdown=self.risk.max_drawdown,
            kelly_fraction=self.risk.kelly_fraction,
        )

        pos_size = effective_risk.calculate_position_size(signal)
        if pos_size is None:
            return

        result = self.executor.send_order(signal, pos_size)
        if result:
            self.risk.register_open_position(signal.ticker, pos_size.capital_allocated)

    # ──────────────────────────────────────────────
    # Relatório
    # ──────────────────────────────────────────────

    def _print_summary(self) -> None:
        """Imprime resumo do portfólio."""
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


# ─────────────────────────────────────────────────────────
# Execução
# ─────────────────────────────────────────────────────────

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
    config["trading"]["mode"] = args.mode  # Override pelo argumento CLI

    bot = TradebotUTIL(config)
    bot.start()
