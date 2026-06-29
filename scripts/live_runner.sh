#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════
# Tradebot-UTIL.v2 — Script de inicialização para VPS (Linux/Windows WSL)
# ══════════════════════════════════════════════════════════════
#
# Uso:
#   chmod +x scripts/live_runner.sh
#
#   # Modo paper (simulação):
#   ./scripts/live_runner.sh
#
#   # Modo live (ordens reais):
#   ./scripts/live_runner.sh --live
#
#   # Forçar rebalanceamento imediato:
#   ./scripts/live_runner.sh --force-rebalance
#
#   # Backtest antes de subir:
#   ./scripts/live_runner.sh --backtest
#
# Pré-requisitos:
#   - Python 3.11+
#   - pip install -r requirements.txt  (ou requirements_backtest.txt para backtest)
#   - config/config.yaml preenchida com credenciais MT5
#
# Variáveis de ambiente (opcional, sobrepõem o config.yaml):
#   TRADEBOT_MT5_LOGIN     — Login MT5
#   TRADEBOT_MT5_PASSWORD  — Senha MT5
#   TRADEBOT_CAPITAL       — Capital em R$
#   TRADEBOT_MODE          — "paper" ou "live"
#   TRADEBOT_CONFIG        — Caminho para config YAML
# ══════════════════════════════════════════════════════════════

set -euo pipefail

# ── Configuração ───────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_ROOT/.venv"
CONFIG_FILE="${TRADEBOT_CONFIG:-$PROJECT_ROOT/config/config.active_momentum_tilt.yaml}"
LOG_DIR="$PROJECT_ROOT/logs"
PID_FILE="$PROJECT_ROOT/tradebot.pid"

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔═══════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   Tradebot-UTIL.v2 — Active Momentum v4   ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════╝${NC}"
echo ""

# ── Verifica Python ────────────────────────────────────────────
PYTHON=$(which python3 || which python)
PY_VERSION=$("$PYTHON" --version 2>&1 | cut -d' ' -f2)
echo -e "${GREEN}✓ Python: $PY_VERSION${NC}"

# ── Ativa venv (cria se necessário) ───────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}Criando ambiente virtual...${NC}"
    "$PYTHON" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# ── Instala dependências ───────────────────────────────────────
echo -e "${YELLOW}Verificando dependências...${NC}"
pip install -q --upgrade pip
pip install -q -r "$PROJECT_ROOT/requirements.txt"
echo -e "${GREEN}✓ Dependências OK${NC}"

# ── Cria diretório de logs ────────────────────────────────────
mkdir -p "$LOG_DIR"

# ── Verifica config ────────────────────────────────────────────
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}✗ Config não encontrada: $CONFIG_FILE${NC}"
    echo -e "  Copie config/config.active_momentum_tilt.yaml e preencha as credenciais:"
    echo -e "  cp config/config.active_momentum_tilt.yaml config/config.yaml"
    exit 1
fi
echo -e "${GREEN}✓ Config: $CONFIG_FILE${NC}"

# ── Mata instância anterior (se houver) ──────────────────────
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo -e "${YELLOW}Parando instância anterior (PID: $OLD_PID)...${NC}"
        kill "$OLD_PID"
        sleep 2
    fi
    rm -f "$PID_FILE"
fi

# ── Parse de argumentos ────────────────────────────────────────
EXTRA_ARGS=()
for arg in "$@"; do
    EXTRA_ARGS+=("$arg")
done

# ── Lembrete: MT5 precisa estar aberto (modo live) ────────────
if [[ " ${EXTRA_ARGS[*]} " == *"--live"* ]]; then
    echo -e "${YELLOW}"
    echo "  ⚠️  ATENÇÃO: Modo LIVE ativado"
    echo "  Certifique-se de que o MetaTrader 5 está ABERTO e LOGADO na XP"
    echo "  antes de continuar. O bot conecta à sessão existente (sem re-login)."
    echo -e "${NC}"
    read -r -p "  MT5 está aberto e logado? (sim/não): " CONFIRM
    if [[ "$CONFIRM" != "sim" && "$CONFIRM" != "s" ]]; then
        echo "Cancelado. Abra o MT5, faça login e tente novamente."
        exit 0
    fi
fi

# ── Inicia o bot ───────────────────────────────────────────────
echo ""
echo -e "${GREEN}Iniciando Tradebot-UTIL.v2...${NC}"
echo -e "  Projeto:  $PROJECT_ROOT"
echo -e "  Config:   $CONFIG_FILE"
echo -e "  Logs:     $LOG_DIR"
echo ""

cd "$PROJECT_ROOT"

# Modo --backtest: instala extras e roda backtest
if [[ " ${EXTRA_ARGS[*]} " == *"--backtest"* ]]; then
    pip install -q -r requirements_backtest.txt
    "$PYTHON" main.py --config "$CONFIG_FILE" --backtest
    exit 0
fi

# Modo foreground (padrão para debug)
if [[ " ${EXTRA_ARGS[*]} " == *"--foreground"* ]]; then
    EXTRA_ARGS=("${EXTRA_ARGS[@]/--foreground/}")
    "$PYTHON" main.py --config "$CONFIG_FILE" "${EXTRA_ARGS[@]}"
    exit 0
fi

# Modo background (padrão para VPS)
nohup "$PYTHON" main.py --config "$CONFIG_FILE" "${EXTRA_ARGS[@]}" \
    >> "$LOG_DIR/tradebot_stdout.log" 2>&1 &

BOT_PID=$!
echo "$BOT_PID" > "$PID_FILE"

sleep 2
if kill -0 "$BOT_PID" 2>/dev/null; then
    echo -e "${GREEN}✓ Bot iniciado em background (PID: $BOT_PID)${NC}"
    echo -e "  Logs em tempo real:  tail -f $LOG_DIR/tradebot_live.log"
    echo -e "  Para parar:          kill $BOT_PID  ou  ./scripts/stop.sh"
else
    echo -e "${RED}✗ Bot falhou ao iniciar. Verifique os logs:${NC}"
    echo -e "  tail -50 $LOG_DIR/tradebot_stdout.log"
    exit 1
fi
