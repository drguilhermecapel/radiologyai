doc_id: REG-10
title: Plano de Avaliação Clínica (Fase 3 — Pré-especificado)
status: RASCUNHO — v0.1
date: 2026-11
author: Dr. Guilherme Capel Pasqua (CRM-SP 175873)
supersedes: —
# Plano de Avaliação Clínica

> **Estado atual:** software em estágio de pesquisa — não é dispositivo médico e não possui avaliação clínica concluída. Este plano fixa desenho, desfechos, subgrupos e critérios de aceitação **antes** do treinamento do modelo próprio da Fase 3 e da validação externa em VinDr-CXR, para garantir caráter confirmatório. Ver [`HONEST_STATUS.md`](../../HONEST_STATUS.md) e [`ROADMAP.md`](../../ROADMAP.md) §5 (Fase 3).

## 1. Escopo e pergunta clínica

- **Dispositivo em avaliação:** RadiologyAI v1 (radiografia frontal de tórax em adultos, modalidade `CR`/`DX`, uso assistivo concorrente; ver [`01-intended-use.md`](01-intended-use.md)).
- **Pergunta primária:** um modelo próprio treinado em NIH ChestX-ray14 (split disjunto por paciente) supera a linha de base zero-shot registrada em `artifacts/eval/xrv-densenet121-pc__20260912T202549Z/` quando avaliado externamente em coorte independente com rótulos adjudicados por radiologistas (VinDr-CXR)?
- **O que este plano NÃO avalia:** uso autônomo, triagem/priorização de fila, exclusão (*rule-out*), população pediátrica ou incidências laterais.

## 2. Coortes e isolamento de dados (sem vazamento)

| Etapa | Conjunto de dados | Tipo de rótulo | Regra de isolamento |
|---|---|---|---|
| **Desenvolvimento e calibração** | NIH ChestX-ray14 (`train`/`val` disjuntos por paciente) | Minerado por NLP (limitação declarada) | Disjunção estrita de `patient_id` verificada por `REQ-051`; conjunto de calibração (*temperature scaling*) separado do treino |
| **Avaliação externa primária** | VinDr-CXR (18.000 exames; conjunto de teste com 3.000 exames lidos por 5 radiologistas) | Consenso de radiologistas | Conjunto mantido completamente isolado até o congelamento do artefato (pesos + SHA-256 + limiares de abstenção) |
| **Avaliação externa complementar** | CheXpert (conjunto de validação adjudicado) | Adjudicado | Cobertura parcial de achados (não avalia fibrose; ver `ROADMAP.md` §5) |

## 3. Desfechos e métricas obrigatórias

Toda execução oficial deve ser produzida por máquina via `radiologyai evaluate`, gerando `artifacts/eval/<run_id>/metrics.json` e `reports/EVALUATION_*.md` com proveniência completa (`git_sha`, `weights_sha256`, `manifest_sha256`, `seed`, versões de bibliotecas):
1. **Discriminação (primário):** AUROC por achado e AUROC macro com intervalo de confiança de 95% (DeLong / bootstrap com semente fixa, `REQ-050`).
2. **Calibração:** *Expected Calibration Error* (ECE) pré e pós-*temperature scaling* + diagrama de confiabilidade (`REQ-061`).
3. **Política de abstenção:** distribuição nas três bandas (*achado provável* / *não avaliável* / *achado improvável*) derivada exclusivamente de artefato medido (`REQ-062`), verificando ausência de afirmação de normalidade (`REQ-063`).
4. **Análise de subgrupos obrigatória:** incidência (PA vs AP), sexo, faixa etária adulta e equipamento/fonte quando disponível no metadado.
5. **Paridade de inferência:** diferença absoluta máxima inferior a `1e-4` entre PyTorch e ONNX em amostra de 100 exames.

## 4. Critérios de aceitação pré-especificados (METAS da Fase 3)

O modelo próprio só avança para a etapa de endurecimento regulatório se cumprir simultaneamente:
- **Superação da linha de base:** AUROC macro em avaliação externa superior ao resultado registrado em `artifacts/eval/xrv-densenet121-pc__20260912T202549Z/`, sem degradação clinicamente inaceitável em achados de alta severidade (pneumotórax, derrame pleural, cardiomegalia).
- **Calibração verificada:** redução ou manutenção de ECE após *temperature scaling* sem alteração da ordenação dos escores.
- **Reprodutibilidade integral:** execução reproduzível bit a bit (exceto timestamp) a partir de checkout limpo em CPU e GPU.
- **Fechamento de riscos abertos:** implementação e verificação de `REQ-043` / `RISK-006` (lateralidade) em [`04-risk-file.md`](04-risk-file.md).

## 5. Governança de desvios e transparência

Qualquer alteração de arquitetura, hiperparâmetros, limiar de abstenção ou exclusão de exames após exposição ao conjunto de teste externo (VinDr-CXR) invalida o caráter confirmatório da avaliação e deve ser registrada como desvio exploratório em `reports/`.

## Histórico de revisões

| Versão | Data | Alteração |
|---|---|---|
| v0.1 | 2026-11 | Criação do Plano de Avaliação Clínica pré-especificado (Fase 3) |
