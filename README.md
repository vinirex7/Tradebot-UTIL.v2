# Tradebot-UTIL.v2

Bot de trading algorítmico focado no Índice UTIL (Utilidade Pública) da B3, operando via MetaTrader 5 conectado à XP Investimentos.

## Objetivo

Superar o desempenho do índice UTIL B3 com estratégias algorítmicas combinadas, aproveitando as características do setor de utilities: baixo beta, sensibilidade à Selic e distribuição consistente de dividendos.

## Estratégias Implementadas

| # | Estratégia | Racional | Timeframe |
|---|-----------|----------|-----------|
| 1 | **Reversão à Média** | Beta baixo + Bollinger + RSI | 60 min / Diário |
| 2 | **Momentum Macro-Driven** | Filtro Selic/DI + EMA cruzamento | 2 sem – 3 meses |
| 3 | **Pair Trading** | Z-score spread intrasetorial | Diário |
| 4 | **Captura de Dividendos** | Ex-date das top pagadoras | 3–5 dias |
| 5 | **Antecipação de Rebalanceamento** | Prévias quadrimestrais B3 | 1–10 dias |

## Ativos do Universo (UTIL — Mai–Ago 2026)

```
SBSP3  (19,999%)  AXIA3  (17,291%)  EQTL3  (11,431%)  ENEV3  (10,708%)
CPLE3  (10,318%)  CMIG4   (5,422%)  ENGI11  (3,786%)  AXIA6   (2,708%)
EGIE3   (2,637%)  ISAE4   (2,587%)  CSMG3   (2,440%)  SAPR11  (2,003%)
TAEE11  (2,104%)  CPFE3   (2,070%)  NEOE3   (1,509%)  ALUP11  (1,262%)
ORVR3   (0,862%)  AURE3   (0,855%)
```

## Requisitos

- Python 3.10+
- MetaTrader 5 (instalado e configurado com conta XP Investimentos)
- Conta XP Investimentos com acesso ao MetaTrader 5
- Conexão com internet estável

## Instalação Rápida

```bash
git clone https://github.com/SEU_USUARIO/Tradebot-UTIL.v2.git
cd Tradebot-UTIL.v2
pip install -r requirements.txt
cp config/config.example.yaml config/config.yaml
# Edite config/config.yaml com suas credenciais
python scripts/setup_check.py
```

## Configuração MT5 + XP Investimentos

1. Instale o MetaTrader 5 pela XP: [https://www.xpi.com.br/plataformas/metatrader/](https://www.xpi.com.br/plataformas/metatrader/)
2. Faça login com sua conta XP no MT5
3. Em `config/config.yaml`, preencha `mt5.login`, `mt5.password` e `mt5.server`
4. O servidor da XP no MT5 é normalmente: `XP Investimentos-Real` ou `XPI-Demo`

## Estrutura do Projeto

```
Tradebot-UTIL.v2/
├── src/
│   ├── strategies/          # Implementação das 5 estratégias
│   │   ├── mean_reversion.py
│   │   ├── momentum_macro.py
│   │   ├── pair_trading.py
│   │   ├── dividend_capture.py
│   │   └── rebalance_anticipation.py
│   ├── risk/                # Gestão de risco e position sizing
│   │   └── risk_manager.py
│   ├── data/                # Feeds de dados (MT5, Selic, dividendos)
│   │   ├── mt5_feed.py
│   │   ├── macro_feed.py
│   │   └── dividend_calendar.py
│   ├── execution/           # Execução de ordens via MT5
│   │   └── order_executor.py
│   └── utils/               # Utilitários gerais
│       ├── logger.py
│       └── indicators.py
├── config/
│   ├── config.example.yaml  # Template de configuração
│   └── universe.yaml        # Composição atual do UTIL
├── tests/                   # Testes unitários e backtests
│   ├── test_strategies.py
│   └── backtest_runner.py
├── scripts/
│   ├── setup_check.py       # Verifica conexão MT5 e dependências
│   └── run_backtest.py      # Executa backtest completo
├── logs/                    # Logs gerados em runtime
├── docs/
│   └── UTIL-B3-Estudo.md    # Estudo base da estratégia
├── main.py                  # Entry point principal
└── requirements.txt
```

## Parâmetros de Risco

| Parâmetro | Valor Padrão |
|-----------|-------------|
| Stop loss por operação | 2–3% do capital alocado |
| Exposição máxima por ativo | 20% do capital |
| Drawdown máximo (pausa) | 10% do capital total |
| Position sizing | Kelly Fracionário (fração = 0.25) |

## Uso

```bash
# Modo produção (live trading)
python main.py --mode live

# Modo paper trading (simulação sem dinheiro real)
python main.py --mode paper

# Backtest de uma estratégia específica
python scripts/run_backtest.py --strategy mean_reversion --start 2019-01-01 --end 2026-01-01

# Verificar status da conexão MT5
python scripts/setup_check.py
```

## Aviso Legal

Este software é fornecido apenas para fins educacionais e de pesquisa. Trading algorítmico envolve riscos significativos de perda de capital. Sempre realize paper trading extensivo antes de operar com dinheiro real. O autor não se responsabiliza por perdas financeiras.

---

Baseado no estudo: **Índice UTIL B3 – Estudo Completo para Estratégia de Tradingbot**
