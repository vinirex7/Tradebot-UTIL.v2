# Índice UTIL B3 – Estudo Completo para Estratégia de Tradingbot

> Documento de referência estratégica do Tradebot-UTIL.v2.
> Veja o arquivo original completo em `/uploaded_attachments/UTIL-B3-Estudo-Tradingbot.md`.

## Resumo Executivo

O Índice UTIL é revisado quadrimestralmente pela B3 e é composto por 18 ações do setor de utilidade pública.
A carteira vigente (Mai–Ago 2026) tem peso de ~70% concentrado nas 5 maiores posições.

## Estratégias Implementadas no Bot

1. **Reversão à Média** — Bollinger (20,2σ) + RSI(14) | Ativos: SBSP3, EQTL3, CPLE3, EGIE3, TAEE11
2. **Momentum Macro-Driven** — Filtro Selic/DI + EMA 9/21 | Ativos: SBSP3, EQTL3, ENEV3, CPLE3
3. **Pair Trading** — Z-score spread | Pares: EQTL3/TAEE11, SBSP3/ENEV3
4. **Captura de Dividendos** — Ex-date das top pagadoras | DY > 5%, Vol > R$50M
5. **Antecipação de Rebalanceamento** — Prévias B3 quadrimestrais

## Parâmetros de Risco

- Stop loss: 2–3% por operação
- Exposição máxima por ativo: 20% (espelha limite do índice)
- Drawdown máximo: 10% (pausa automática)
- Position sizing: Kelly Fracionário (fração 0.25)
