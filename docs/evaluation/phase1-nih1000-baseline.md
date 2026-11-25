# Avaliação Baseline NIH-1000 — Fase 1 (Apenas Pesquisa / Research-Only)

> **Aviso:** Este projeto é estritamente voltado para pesquisa (*research-only*). **Não é um dispositivo médico** e não deve ser utilizado para diagnóstico ou decisão clínica.

## 1. Baseline NIH-1000 (25/11/2026)

- **Ambiente / Modelo:** Tesla T4, `xrv-densenet121-pc`
- **Amostragem:** 1.000 pacientes distintos (250 calibração / 750 teste, 0 sobreposição entre splits)
- **Macro-AUROC:** 0,7089 (IC 95%: 0,6865–0,7302)
- **Critério pré-especificado (>= 0,70 no limite inferior do IC):** **NÃO atendido**

## 2. Estudo Pareado de Calibração (12 achados comuns)

Análise pareada nos 12 achados comuns (excluindo **Edema**, que apresentou 0 positivos no conjunto de calibração):

- **ECE médio:** 0,2540 (bruto) -> 0,2484 (Temperature, -2,19%) -> 0,0151 (Platt, -94,06%)
- **Delta pareado de ECE (Platt vs. Temperature):** -0,2333 (IC 95%: -0,2408 a -0,2199)
- **Brier score médio:** 0,1687 (bruto) -> 0,0697 (Platt)

## 3. Ressalvas Explícitas e Próximos Passos

- **Sobreposição com piloto:** 14/750 pacientes do conjunto de teste sobrepõem-se ao piloto anterior de 64 exames.
- **Baixa contagem de positivos na calibração:** Pneumotórax (2), Pneumonia (5) e Consolidação (6).
- **Edema:** sem ajuste Platt devido à ausência de exemplos positivos na calibração.
- **Precisão numérica em `platt.py`:** refinamento do piso de saída de `1e-7` para `1e-12` para preservar ordenação/ranking em probabilidades baixas.
- **Status de regressão:** permanece `NOT_EVALUATED` até a execução de benchmark independente em pacientes inéditos.
