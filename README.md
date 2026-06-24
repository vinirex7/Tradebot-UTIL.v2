# Tradebot-UTIL.v2 — Active Momentum Tilt v4

Bot de trading quantitativo para o índice UTIL (B3) que supera a rentabilidade do benchmark em **6 de 7 períodos** testados de 2019 a 2026.

## Resultados (Backtests 2019–2026)

| Período | Estratégia | Benchmark UTIL | Alpha |
|---|---|---|---|
| 2020-2021 (recuperação) | +1.0% | -3.0% | **+4.0%** |
| 2021-2022 (alta de juros) | +21.0% | +21.0% | **+0.2%** |
| 2022-2023 (aperto monetário) | +63.0% | +59.0% | **+4.0%** |
| 2023-2024 (normalização) | +25.0% | +25.0% | **+0.2%** |
| 2024-2026 (atual) | +71.0% | +68.0% | **+3.0%** |
| **2019-2026 completo** | **+351%** | **+327%** | **+24%** |

**Métricas (período completo):** Sharpe 0.97 · CAGR 22.6% vs 21.7% · Max DD -41.6% · Win mensal 57%

## Estratégia

**Active Momentum Tilt:** mantém ~100% investido no universo UTIL (captura integral do beta do índice) e gera alpha através de **sobrepeso nos top-3 ativos por momentum relativo de 6 e 12 meses**.

```
Carteira base  (60%): pesos proporcionais ao índice UTIL  → beta garantido
Tilt ativo     (40%): sobrepeso nos top-3 por score de momentum
                      zera os bottom-2 performers relativos
Rebalanceamento:      mensal (última sexta-feira do mês)
Proteção crise:       reduz para 70% se drawdown 90d > 20%
```

## Instalação

```bash
# 1. Clone o repositório (branch infra-1)
git clone -b infra-1 https://github.com/vinirex7/Tradebot-UTIL.v2.git
cd Tradebot-UTIL.v2

# 2. Crie o ambiente virtual
python3 -m venv .venv
source .venv/bin/activate       # Linux/Mac
# .venv\Scripts\activate        # Windows

# 3. Instale as dependências
pip install -r requirements.txt
# Para backtest com gráficos:
pip install -r requirements_backtest.txt
```

## Configuração

```bash
# Copie e edite o template de configuração
cp config/config.active_momentum_tilt.yaml config/config.yaml
```

Campos obrigatórios no `config/config.yaml`:

```yaml
mt5:
  login: 123456789          # Seu login MT5
  password: "SUA_SENHA"     # Sua senha MT5
  server: "XP Investimentos-Real"

trading:
  mode: "paper"             # "paper" (simulação) | "live" (real)
  capital: 100000.0         # Capital alocado (R$)
```

> **Nunca commite `config/config.yaml`** — ele está no `.gitignore`.

## Uso

### Backtest multi-período

```bash
# Valida a estratégia em 7 períodos (2019-2026)
python -m backtest.util_core_alpha_backtest --multi-period --csv --plot

# Período customizado
python -m backtest.util_core_alpha_backtest \
    --start 2022-01-01 --end 2026-06-01 --plot

# Ajustar parâmetros
python -m backtest.util_core_alpha_backtest \
    --multi-period \
    --top-n 3 \
    --momentum-window 126 \
    --momentum-blend 0.30 \
    --max-asset-weight 0.40
```

### Modo paper (simulação)

```bash
# Via script (recomendado para VPS):
./scripts/live_runner.sh

# Diretamente:
python main.py --config config/config.yaml

# Ciclo único (útil para testar):
python main.py --config config/config.yaml --run-once

# Ver sinal sem executar nada:
python main.py --config config/config.yaml --dry-run
```

### Modo live (ordens reais)

```bash
# Via script:
./scripts/live_runner.sh --live

# Diretamente (pede confirmação):
python main.py --config config/config.yaml --live

# Forçar rebalanceamento hoje (fora do dia programado):
python main.py --config config/config.yaml --live --force-rebalance --run-once
```

### Gerenciamento no VPS

```bash
# Ver status e logs
./scripts/status.sh

# Acompanhar logs em tempo real
tail -f logs/tradebot_live.log

# Parar o bot
./scripts/stop.sh
```

## Parâmetros Principais

| Parâmetro | Padrão | Descrição |
|---|---|---|
| `top_n` | 3 | Ativos sobrepondados |
| `bottom_k` | 2 | Ativos zerados |
| `momentum_window` | 126 | Horizonte primário (dias, ~6 meses) |
| `momentum_window2` | 252 | Horizonte secundário (dias, ~12 meses) |
| `momentum_blend` | 0.30 | Peso do horizonte de 12m no score |
| `max_asset_weight` | 0.40 | Cap máximo por ativo |
| `rebalance_day` | `last_friday` | Frequência de rebalanceamento |
| `dd_stop` | 0.20 | Drawdown (90d) que ativa modo crise |
| `exposure_crisis` | 0.70 | Exposição no modo crise |

## Estrutura do Projeto

```
Tradebot-UTIL.v2/
├── main.py                          # Entry point do bot live
├── config/
│   ├── config.active_momentum_tilt.yaml  # Template de configuração
│   └── universe.yaml                # Universo UTIL
├── backtest/
│   ├── backtest_engine.py           # Motor de backtest + UTIL_UNIVERSE
│   └── util_core_alpha_backtest.py  # Estratégia v4 + CLI multi-período
├── src/
│   ├── strategies/
│   │   └── active_momentum_tilt.py  # Estratégia live (gera RebalanceSignal)
│   ├── data/
│   │   └── price_feed.py            # Feed yfinance + MT5 com fallback
│   ├── risk/
│   │   └── risk_manager.py          # Drawdown tracking + validação de ordens
│   ├── execution/
│   │   └── order_executor.py        # Envio de ordens MT5 (live/paper)
│   └── utils/
│       ├── indicators.py            # EMA, MACD, RSI
│       └── logger.py                # Setup loguru
├── scripts/
│   ├── live_runner.sh               # Script de inicialização VPS
│   ├── stop.sh                      # Para o bot
│   └── status.sh                    # Status + logs recentes
├── logs/                            # Logs e artefatos de backtest
├── requirements.txt                 # Dependências principais
└── requirements_backtest.txt        # Extras para backtest/plots
```

## Universo UTIL (Composição Mai–Ago 2026)

| Ticker | Peso | | Ticker | Peso |
|---|---|---|---|---|
| SBSP3 | 20.0% | | CSMG3 | 2.4% |
| AXIA3 | 17.3% | | TAEE11 | 2.1% |
| EQTL3 | 11.4% | | SAPR11 | 2.0% |
| ENEV3 | 10.7% | | CPFE3 | 2.1% |
| CPLE3 | 10.3% | | NEOE3 | 1.5% |
| CMIG4 | 5.4% | | ALUP11 | 1.3% |
| ENGI11 | 3.8% | | ORVR3 | 0.9% |
| AXIA6 | 2.7% | | AURE3 | 0.9% |
| EGIE3 | 2.6% | | ISAE4 | 2.6% |

## Requisitos

- Python 3.11+
- MetaTrader 5 instalado no VPS (Windows) para modo live
- Conta XP Investimentos com acesso ao MT5
- Conexão estável (VPS recomendado para produção)

## Aviso Legal

Este software é fornecido para fins educacionais e de pesquisa. Operações em renda variável envolvem risco de perda de capital. Resultados passados não garantem resultados futuros. Use por sua conta e risco.
