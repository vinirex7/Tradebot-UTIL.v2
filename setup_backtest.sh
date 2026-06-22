#!/bin/bash
# ============================================================
# setup_backtest.sh — Tradebot-UTIL.v2
# Instala o ambiente de backtest no Linux (sem MetaTrader5)
# ============================================================

set -e

echo ""
echo "════════════════════════════════════════════════"
echo "  Tradebot-UTIL.v2 — Setup de Backtest (Linux)"
echo "════════════════════════════════════════════════"

# Verificar Python 3.10+
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo ""
echo "[1] Python detectado: $PYTHON_VERSION"

# Criar virtualenv
echo ""
echo "[2] Criando ambiente virtual (.venv)..."
python3 -m venv .venv
echo "    ✓ Virtualenv criado em .venv/"

# Ativar virtualenv
source .venv/bin/activate
echo "    ✓ Virtualenv ativado"

# Atualizar pip
echo ""
echo "[3] Atualizando pip..."
pip install --upgrade pip --quiet

# Instalar dependências de backtest (sem MetaTrader5)
echo ""
echo "[4] Instalando dependências de backtest..."
pip install -r requirements_backtest.txt

echo ""
echo "[5] Criando diretório de logs..."
mkdir -p logs

echo ""
echo "════════════════════════════════════════════════"
echo "  Setup concluído!"
echo ""
echo "  Para rodar o backtest:"
echo ""
echo "    source .venv/bin/activate"
echo "    python backtest/run_backtest.py"
echo ""
echo "  Com gráfico PNG:"
echo "    python backtest/run_backtest.py --plot"
echo ""
echo "  Estratégia específica:"
echo "    python backtest/run_backtest.py --strategy mean_reversion"
echo ""
echo "  Período personalizado:"
echo "    python backtest/run_backtest.py --start 2022-01-01 --end 2026-01-01"
echo "════════════════════════════════════════════════"
echo ""
