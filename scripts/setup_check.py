"""
Script de verificação de setup.
Verifica: dependências Python, conexão MT5, acesso aos símbolos do UTIL.
"""
import sys
import importlib
from pathlib import Path

# Adiciona o root ao path
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_dependencies() -> bool:
    """Verifica se todas as dependências estão instaladas."""
    deps = [
        "MetaTrader5", "pandas", "numpy", "scipy", "ta",
        "requests", "yaml", "loguru", "schedule", "bs4",
    ]
    missing = []
    for dep in deps:
        try:
            importlib.import_module(dep)
            print(f"  ✓ {dep}")
        except ImportError:
            print(f"  ✗ {dep} — FALTANDO")
            missing.append(dep)
    return len(missing) == 0


def check_config() -> bool:
    """Verifica se config.yaml existe e tem as chaves obrigatórias."""
    config_path = Path("config/config.yaml")
    if not config_path.exists():
        print("  ✗ config/config.yaml não encontrado")
        print("    Execute: cp config/config.example.yaml config/config.yaml")
        return False

    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    required = ["mt5", "trading", "strategies", "logging"]
    missing = [k for k in required if k not in cfg]
    if missing:
        print(f"  ✗ Chaves faltando no config: {missing}")
        return False

    if cfg["mt5"]["login"] in (123456789, None, ""):
        print("  ⚠ mt5.login ainda é o valor padrão — atualize com sua conta XP")
        return False

    print("  ✓ config/config.yaml válido")
    return True


def check_mt5_connection() -> bool:
    """Testa conexão com o MetaTrader 5."""
    try:
        import yaml
        import MetaTrader5 as mt5

        with open("config/config.yaml") as f:
            cfg = yaml.safe_load(f)

        mt5_cfg = cfg["mt5"]
        ok = mt5.initialize(
            login=mt5_cfg["login"],
            password=mt5_cfg["password"],
            server=mt5_cfg["server"],
            timeout=mt5_cfg.get("timeout", 60000),
        )

        if not ok:
            err = mt5.last_error()
            print(f"  ✗ Falha ao conectar ao MT5: {err}")
            return False

        info = mt5.account_info()
        if info is None:
            print("  ✗ Conta não encontrada.")
            mt5.shutdown()
            return False

        print(f"  ✓ MT5 conectado | Conta: {info.login} | Servidor: {info.server}")
        print(f"  ✓ Saldo: R$ {info.balance:,.2f} | Equity: R$ {info.equity:,.2f}")
        mt5.shutdown()
        return True

    except Exception as e:
        print(f"  ✗ Erro na conexão MT5: {e}")
        return False


def check_symbols() -> bool:
    """Verifica se os símbolos do UTIL estão disponíveis no MT5."""
    try:
        import yaml
        import MetaTrader5 as mt5

        with open("config/config.yaml") as f:
            cfg = yaml.safe_load(f)

        with open("config/universe.yaml") as f:
            universe = yaml.safe_load(f)

        tickers = [a["ticker"] for a in universe["util_composition"]]

        mt5.initialize(
            login=cfg["mt5"]["login"],
            password=cfg["mt5"]["password"],
            server=cfg["mt5"]["server"],
        )

        ok_count = 0
        for ticker in tickers:
            info = mt5.symbol_info(ticker)
            if info:
                print(f"  ✓ {ticker} — bid: {mt5.symbol_info_tick(ticker).bid if mt5.symbol_info_tick(ticker) else 'N/A'}")
                ok_count += 1
            else:
                print(f"  ✗ {ticker} — não encontrado no MT5")

        mt5.shutdown()
        print(f"\n  Símbolos disponíveis: {ok_count}/{len(tickers)}")
        return ok_count > 0

    except Exception as e:
        print(f"  ✗ Erro ao verificar símbolos: {e}")
        return False


def check_macro_feed() -> bool:
    """Testa conexão com BCB para dados macro."""
    try:
        from src.data.macro_feed import MacroFeed
        feed = MacroFeed()
        selic = feed.get_selic_rate()
        if selic:
            print(f"  ✓ BCB API acessível | Selic: {selic*100:.2f}%")
            return True
        print("  ✗ Não foi possível obter a Selic do BCB")
        return False
    except Exception as e:
        print(f"  ✗ Erro no MacroFeed: {e}")
        return False


if __name__ == "__main__":
    print("\n" + "═" * 50)
    print("  Tradebot-UTIL.v2 — Verificação de Setup")
    print("═" * 50)

    results = []

    print("\n[1] Dependências Python:")
    results.append(check_dependencies())

    print("\n[2] Arquivo de configuração:")
    results.append(check_config())

    # Só testa MT5/símbolos se config estiver ok
    if results[-1]:
        print("\n[3] Conexão MetaTrader 5:")
        mt5_ok = check_mt5_connection()
        results.append(mt5_ok)

        if mt5_ok:
            print("\n[4] Símbolos UTIL no MT5:")
            results.append(check_symbols())

    print("\n[5] Feed Macroeconômico (BCB):")
    results.append(check_macro_feed())

    print("\n" + "═" * 50)
    if all(results):
        print("  ✓ SETUP COMPLETO — Bot pronto para operar!")
        print("  Execute: python main.py --mode paper")
    else:
        print("  ✗ Setup incompleto — corrija os erros acima.")
    print("═" * 50 + "\n")
