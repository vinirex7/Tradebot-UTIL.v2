"""
Runner de backtest completo.
Importado pelo scripts/run_backtest.py.
"""
# Re-export para uso externo
from scripts.run_backtest import (
    download_data,
    backtest_mean_reversion,
    calculate_metrics,
    download_util_benchmark,
)
