"""
Compatibilidade: scripts/run_backtest.py
──────────────────────────────────────
Este arquivo antigo agora apenas redireciona para o backtest atualizado da
branch main, evitando rodar a estratégia obsoleta de Reversão à Média.

Uso recomendado:
    python backtest/run_backtest.py --strategy momentum_macro --start 2019-01-01 --end 2026-01-01

Uso compatível:
    python scripts/run_backtest.py --strategy momentum_macro --start 2019-01-01 --end 2026-01-01
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest.run_backtest import main


if __name__ == "__main__":
    main()
