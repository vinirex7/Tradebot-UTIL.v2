#!/usr/bin/env bash
# Mostra status do bot e últimas linhas do log
PROJECT_ROOT="$(dirname "$(dirname "${BASH_SOURCE[0]}")")"
PID_FILE="$PROJECT_ROOT/tradebot.pid"
LOG_FILE="$PROJECT_ROOT/logs/tradebot_live.log"

echo "=== Tradebot-UTIL.v2 Status ==="
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "✓ RODANDO (PID: $PID)"
    else
        echo "✗ PARADO (PID file existe mas processo não)"
    fi
else
    echo "✗ PARADO"
fi
echo ""
echo "=== Últimas 30 linhas do log ==="
if [ -f "$LOG_FILE" ]; then
    tail -30 "$LOG_FILE"
else
    echo "(log ainda não criado)"
fi
