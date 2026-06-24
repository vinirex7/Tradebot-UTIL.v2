"""
Logger — Tradebot-UTIL.v2
══════════════════════════

Configuração do loguru para uso em produção.
"""
from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


def setup_logger(
    log_file: str = "logs/tradebot.log",
    level: str = "INFO",
    rotation: str = "1 day",
    retention: str = "90 days",
    console: bool = True,
) -> None:
    """
    Configura o loguru com:
      - Saída no console (colorida)
      - Arquivo rotativo com retenção configurável
    """
    logger.remove()

    fmt_console = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> — <level>{message}</level>"
    )
    fmt_file = (
        "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
        "{name}:{line} — {message}"
    )

    if console:
        logger.add(sys.stderr, format=fmt_console, level=level, colorize=True)

    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_file,
        format=fmt_file,
        level=level,
        rotation=rotation,
        retention=retention,
        encoding="utf-8",
    )
