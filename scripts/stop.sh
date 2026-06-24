#!/usr/bin/env bash
# Para o bot em background
PID_FILE="$(dirname "$(dirname "${BASH_SOURCE[0]}")")/tradebot.pid"
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        kill "$PID"
        echo "Bot parado (PID: $PID)"
    else
        echo "Bot não estava rodando"
    fi
    rm -f "$PID_FILE"
else
    echo "PID file não encontrado"
fi
