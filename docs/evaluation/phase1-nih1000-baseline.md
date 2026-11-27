# Avaliação Baseline NIH-1000 — Fase 1 (Apenas Pesquisa / Research-Only)

> **Aviso:** Este projeto é estritamente voltado para pesquisa (*research-only*). **Não é um dispositivo médico** e não deve ser utilizado para diagnóstico ou decisão clínica.

> **Situação da proveniência (conferida em 2026-11-27): NÃO VERIFICÁVEL.** Os números abaixo foram
> transcritos de uma execução em Google Colab cujo artefato (`metrics.json`, predições, manifest
> da amostra) **não está commitado** em `artifacts/eval/` nem em `evidence/`. Nenhum deles pode
> ser reproduzido ou conferido a partir deste repositório, e nenhum sustenta nível de evidência,
> critério de aceitação ou alegação em outro documento. Ver [`HONEST_STATUS.md`](../../HONEST_STATUS.md)
> §"Relatos sem artefato". Os únicos números de desempenho com artefato são os de
> `artifacts/eval/xrv-densenet121-pc__20260912T202549Z/metrics.json` e
> `artifacts/eval/xrv-densenet121-pc__20260912T215742Z/metrics.json`.

> **Observações da conferência:** (1) o "critério pré-especificado (>= 0,70 no limite inferior do
> IC)" não tem registro de pré-especificação no repositório e **não** é o critério do protocolo
> vigente (`protocol_id: "2026-11-v1"`, [`docs/protocolo-evidencia.md`](../protocolo-evidencia.md));
> (2) o "piloto anterior de 64 exames" também não tem artefato no repositório; (3) a alteração do
> piso de saída de `calibration/platt.py` para `1e-12` foi conferida no código
> (`_OUTPUT_EPS = 1e-12`); (4) a consistência aritmética interna foi conferida, com uma
> diferença de arredondamento na redução relativa do ECE por temperatura (provavelmente calculada
> sobre valores não arredondados).

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
