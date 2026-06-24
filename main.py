"""
Tradebot-UTIL.v2 — Entry Point Principal
══════════════════════════════════════════

Orquestra a estratégia Active Momentum Tilt v4 em produção.

USO:
    # Modo paper (simulação, sem ordens reais):
    python main.py --config config/config.active_momentum_tilt.yaml

    # Modo live (ordens reais via MT5):
    python main.py --config config/config.active_momentum_tilt.yaml --live

    # Forçar rebalanceamento imediato (ignora data programada):
    python main.py --config config/config.active_momentum_tilt.yaml --force-rebalance

    # Executar uma vez e sair (sem loop agendado):
    python main.py --config config/config.active_momentum_tilt.yaml --run-once

    # Apenas imprimir o sinal atual sem executar ordens:
    python main.py --config config/config.active_momentum_tilt.yaml --dry-run

FLUXO DIÁRIO:
    O bot roda em loop e, todo dia após o fechamento do pregão (17h35 BRT):
      1. Baixa preços históricos (yfinance)
      2. Calcula score de momentum para os 18 ativos UTIL
      3. Verifica se é dia de rebalanceamento (última sexta do mês)
      4. Se sim: calcula novos pesos-alvo e envia ordens ao MT5
      5. Monitora drawdown de portfólio (proteção automática)
      6. Envia notificações Telegram (se configurado)
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import yaml
from loguru import logger

from src.data.price_feed import PriceFeed
from src.execution.order_executor import OrderExecutor
from src.risk.risk_manager import RiskManager
from src.strategies.active_momentum_tilt import (
    ActiveMomentumTiltStrategy,
    PortfolioState,
)
from src.utils.logger import setup_logger


# ─── Carregamento de configuração ────────────────────────────────────────────

def load_config(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Config não encontrado: {path}\n"
            f"Copie config/config.active_momentum_tilt.yaml para {path} e preencha os valores."
        )
    with open(p) as f:
        cfg = yaml.safe_load(f)
    logger.info("Configuração carregada: {}", path)
    return cfg


# ─── Notificações (Telegram) ──────────────────────────────────────────────────

def send_telegram(token: str, chat_id: str, message: str) -> bool:
    try:
        import requests
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        resp = requests.post(url, json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}, timeout=10)
        return resp.status_code == 200
    except Exception as exc:
        logger.warning("Falha ao enviar Telegram: {}", exc)
        return False


def notify(config: dict, message: str) -> None:
    notif_cfg = config.get("notifications", {})
    if not notif_cfg.get("enabled", False):
        return
    token = notif_cfg.get("telegram_token", "")
    chat_id = notif_cfg.get("telegram_chat_id", "")
    if token and chat_id:
        send_telegram(token, chat_id, message)


# ─── Orquestrador principal ───────────────────────────────────────────────────

class TradebotUTILv4:
    """
    Orquestrador do Tradebot-UTIL v4 (Active Momentum Tilt).

    Coordena: PriceFeed → Strategy → RiskManager → OrderExecutor
    """

    def __init__(self, config: dict, mode_override: str | None = None):
        self.cfg = config
        self.mode = mode_override or config["trading"].get("mode", "paper")

        # Atualiza o modo no config para que o executor use corretamente
        self.cfg["trading"]["mode"] = self.mode

        setup_logger(
            log_file=config["logging"]["file"],
            level=config["logging"]["level"],
        )

        logger.info("=" * 65)
        logger.info("Tradebot-UTIL.v2 — Active Momentum Tilt v4")
        logger.info("Modo: {} | Iniciado: {}", self.mode.upper(), datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        logger.info("=" * 65)

        self.feed     = PriceFeed(config)
        self.strategy = ActiveMomentumTiltStrategy(config)
        self.risk     = RiskManager(
            capital=config["trading"]["capital"],
            max_pos_pct=config["trading"].get("max_position_pct", 0.40),
            stop_loss_pct=config["trading"].get("stop_loss_pct", 0.00),
            max_drawdown=config["trading"].get("max_drawdown_pct", 0.25),
        )
        self.executor = OrderExecutor(config)

        # Estado do portfólio (persistido entre ciclos)
        self.portfolio = PortfolioState(
            equity=config["trading"]["capital"],
        )

        # Inicializa posições a partir do MT5 (se live)
        if self.mode == "live":
            self._sync_positions_from_mt5()

    def _sync_positions_from_mt5(self) -> None:
        """Sincroniza posições atuais do MT5 com o estado interno."""
        try:
            positions = self.executor.get_current_positions()
            equity = float(self.executor.get_account_equity() or 0.0)

            if equity > 0:
                self.portfolio.equity = equity
                self.risk.update_equity(equity)
            else:
                equity = float(self.portfolio.equity or self.cfg["trading"].get("capital", 0.0))

            synced_positions: dict[str, float] = {}
            if positions and equity > 0:
                for ticker, pos in positions.items():
                    shares = float(pos.get("shares", 0) or 0)
                    current_price = float(pos.get("current_price", 0) or pos.get("avg_price", 0) or 0)
                    if shares <= 0 or current_price <= 0:
                        continue
                    weight = (shares * current_price) / equity
                    if weight > 0:
                        synced_positions[ticker] = weight

            self.portfolio.positions = synced_positions

            logger.info(
                "Sincronizado com MT5 | {} posições | equity R${:,.0f}",
                len(self.portfolio.positions), equity
            )
        except Exception as exc:
            logger.warning("Falha ao sincronizar posições MT5: {}", exc)

    # ─── Ciclo principal ────────────────────────────────────────────────────

    def run_cycle(self, force_rebalance: bool = False, dry_run: bool = False) -> bool:
        """
        Executa um ciclo completo de verificação/rebalanceamento.

        Returns:
            True se o ciclo foi concluído sem erros críticos.
        """
        today = date.today()
        logger.info("─ Ciclo {} ─────────────────────────────────────────────", today)

        # 1. Atualiza equity (live)
        if self.mode == "live":
            try:
                equity = self.executor.get_account_equity()
                if equity > 0:
                    self.portfolio.equity = equity
                    self.portfolio.update_equity(equity, today)
                    self.risk.update_equity(equity)
            except Exception as exc:
                logger.warning("Erro ao atualizar equity: {}", exc)

        # 2. Verifica risco global
        risk_check = self.risk.check_global_risk()
        if risk_check.action == "emergency_stop":
            msg = f"⛔ TRADEBOT PAUSADO: {risk_check.reason}"
            logger.critical(msg)
            notify(self.cfg, msg)
            return False

        # 3. Baixa dados de preço
        logger.info("Baixando dados de preço...")
        try:
            data = self.feed.fetch_closes(use_cache=False)
            if len(data) < 5:
                logger.error("Dados insuficientes: apenas {} ativos disponíveis", len(data))
                return False
            closes = self.feed.get_closes_df(data)
            benchmark = self.feed.get_benchmark(data)
        except Exception as exc:
            logger.error("Erro ao baixar dados: {}", exc)
            return False

        # 4. Gera sinal de rebalanceamento
        logger.info("Calculando sinal de rebalanceamento...")
        try:
            signal = self.strategy.generate_signal(
                closes=closes,
                benchmark=benchmark,
                portfolio_state=self.portfolio,
                force_rebalance=force_rebalance,
                today=today,
            )
            self.strategy.summary_log(signal)
        except Exception as exc:
            logger.error("Erro ao gerar sinal: {}", exc)
            return False

        # 5. Dry run: só mostra o sinal, não executa
        if dry_run:
            logger.info("[DRY RUN] Sinal calculado mas NÃO executado.")
            return True

        if not signal.should_rebalance:
            next_reb = self.strategy.next_rebalance_date(today)
            logger.info("Sem rebalanceamento hoje. Próximo: {}", next_reb)
            return True

        # 6. Valida o rebalanceamento
        risk_val = self.risk.validate_rebalance(
            signal.target_weights, self.portfolio.equity
        )
        if not risk_val.passed:
            logger.error("Rebalanceamento bloqueado pelo RiskManager: {}", risk_val.reason)
            notify(self.cfg, f"⚠️ Rebalanceamento bloqueado: {risk_val.reason}")
            return False

        # 7. Obtém preços atuais para cálculo de quantidade de ações
        current_prices = self.feed.get_current_prices()

        # 8. Executa ordens
        logger.info("Executando rebalanceamento | modo={}", self.mode.upper())
        try:
            result = self.executor.execute_rebalance(
                signal=signal,
                current_prices=current_prices,
                current_equity=self.portfolio.equity,
            )
        except Exception as exc:
            logger.error("Erro ao executar ordens: {}", exc)
            return False

        # 9. Atualiza estado do portfólio
        self.portfolio.positions = dict(signal.target_weights)
        self.portfolio.last_rebalance = today

        # 10. Log e notificação
        summary = result.summary()
        logger.info(summary)

        if self.cfg.get("notifications", {}).get("notify_on_rebalance", True):
            top = ", ".join(signal.top_tickers)
            bot_t = ", ".join(signal.bottom_tickers)
            msg = (
                f"✅ *Rebalanceamento {today}*\n"
                f"Modo: {'LIVE' if self.mode=='live' else 'PAPER'} | "
                f"Exposição: {signal.total_exposure:.0%}\n"
                f"▲ Sobrepondados: {top}\n"
                f"▼ Zerados: {bot_t}\n"
                f"Compras: R${result.total_buys:,.0f} | "
                f"Vendas: R${result.total_sells:,.0f}"
            )
            notify(self.cfg, msg)

        return True

    # ─── Loop agendado ──────────────────────────────────────────────────────

    def run_loop(self, force_rebalance: bool = False) -> None:
        """
        Loop principal: roda diariamente após o fechamento do pregão.

        Horário padrão: 17h35 BRT (após leilão de fechamento da B3).
        Também roda em dias úteis apenas.
        """
        import schedule

        exec_time = self.cfg["trading"].get("execution_time", "17:35")
        logger.info("Bot em loop agendado | execução diária às {} BRT", exec_time)
        logger.info("Pressione Ctrl+C para parar.")

        notify(
            self.cfg,
            f"🤖 *Tradebot-UTIL v4 iniciado*\n"
            f"Modo: {self.mode.upper()} | Capital: R${self.portfolio.equity:,.0f}\n"
            f"Próximo rebalanceamento: {self.strategy.next_rebalance_date(date.today())}"
        )

        def daily_job():
            # Apenas dias úteis
            if datetime.now().weekday() >= 5:
                logger.debug("Final de semana — sem ação")
                return
            self.run_cycle(force_rebalance=force_rebalance)

        schedule.every().day.at(exec_time).do(daily_job)

        # Roda também ao iniciar (para não perder o dia de hoje se iniciado após o horário)
        logger.info("Executando ciclo inicial...")
        self.run_cycle(force_rebalance=force_rebalance)

        try:
            while True:
                schedule.run_pending()
                time.sleep(60)
        except KeyboardInterrupt:
            logger.info("Bot interrompido pelo usuário.")
        finally:
            self.executor.shutdown()
            notify(self.cfg, "🛑 *Tradebot-UTIL v4 parado*")
            logger.info("Bot finalizado.")

    def shutdown(self) -> None:
        self.executor.shutdown()


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tradebot-UTIL.v2 — Active Momentum Tilt v4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", "-c",
        default="config/config.active_momentum_tilt.yaml",
        help="Caminho para o arquivo de configuração YAML",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Ativa modo live (ordens reais via MT5). Sobrepõe o YAML.",
    )
    parser.add_argument(
        "--force-rebalance",
        action="store_true",
        help="Força rebalanceamento imediato, ignorando a data programada",
    )
    parser.add_argument(
        "--run-once",
        action="store_true",
        help="Executa um único ciclo e sai (sem loop)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Calcula e mostra o sinal mas NÃO executa ordens",
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Roda o backtest multi-período da estratégia v4 e sai",
    )
    args = parser.parse_args()

    # Atalho para backtest
    if args.backtest:
        import subprocess
        ret = subprocess.run(
            [sys.executable, "-m", "backtest.util_core_alpha_backtest",
             "--multi-period", "--csv", "--plot"],
            cwd=Path(__file__).parent,
        )
        sys.exit(ret.returncode)

    # Carrega configuração
    config = load_config(args.config)

    # Override de modo
    mode = "live" if args.live else config["trading"].get("mode", "paper")

    # Confirma modo live com o usuário
    if mode == "live" and not args.force_rebalance:
        print("\n⚠️  MODO LIVE ATIVADO — Ordens reais serão enviadas ao MT5!")
        print(f"   Servidor: {config['mt5']['server']}")
        print(f"   Capital:  R${config['trading']['capital']:,.0f}")
        resp = input("   Confirma? (sim/não): ").strip().lower()
        if resp not in ("sim", "s", "yes", "y"):
            print("Cancelado.")
            sys.exit(0)

    bot = TradebotUTILv4(config, mode_override=mode)

    try:
        if args.run_once or args.dry_run:
            ok = bot.run_cycle(
                force_rebalance=args.force_rebalance,
                dry_run=args.dry_run,
            )
            sys.exit(0 if ok else 1)
        else:
            bot.run_loop(force_rebalance=args.force_rebalance)
    finally:
        bot.shutdown()


if __name__ == "__main__":
    main()
