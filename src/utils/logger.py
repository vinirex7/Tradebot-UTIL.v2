"""
Logger centralizado usando loguru.
"""
import sys
from loguru import logger
from pathlib import Path


def setup_logger(log_file: str = "logs/tradebot.log", level: str = "INFO") -> None:
    """Configura o logger global."""
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    logger.remove()

    # Console
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level,
        colorize=True,
    )

    # Arquivo com rotação
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} - {message}",
        level=level,
        rotation="1 day",
        retention="30 days",
        compression="zip",
    )

    logger.info("Logger iniciado. Arquivo: {}", log_file)


__all__ = ["logger", "setup_logger"]
