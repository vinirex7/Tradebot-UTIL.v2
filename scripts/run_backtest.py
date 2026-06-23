"""
Compatibilidade: scripts/run_backtest.py
──────────────────────────────────────
Na branch infra-1, o backtest correto é o da rotação dos melhores ativos UTIL.
Este wrapper evita rodar a Reversão à Média antiga por engano.

Uso recomendado:
    python backtest/run_top4_rotation.py --start 2019-01-01 --end 2026-06-23 --top-n 0 --max-positions 8

Uso compatível:
    python scripts/run_backtest.py --start 2019-01-01 --end 2026-06-23 --top-n 0 --max-positions 8
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest.run_top4_rotation import main


if __name__ == "__main__":
    main()
