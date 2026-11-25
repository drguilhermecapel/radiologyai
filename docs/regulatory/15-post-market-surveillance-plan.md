doc_id: REG-15
title: Plano de Vigilância Pós-Mercado (PMS) — Rascunho Preparatório
status: RASCUNHO — v0.1
date: 2026-11
author: Dr. Guilherme Capel Pasqua (CRM-SP 175873)
supersedes: —

# Plano de Vigilância Pós-Mercado (PMS) — RadiologyAI

> **AVISO REGULATÓRIO OBRIGATÓRIO:** Software em fase de pesquisa (**research-only**). **NÃO há implantação clínica ativa, NÃO é dispositivo médico aprovado pela ANVISA** (RDC 657/2022) e **NÃO está autorizado para uso em pacientes**. Este plano é um **rascunho preparatório** para governança futura na Fase 5.

## 1. Objetivo e Escopo

Estabelecer procedimentos preparatórios de coleta contínua de evidências de desempenho, segurança e qualidade para o RadiologyAI após eventual aprovação regulatória e entrada em operação clínica (inexistente no momento), em alinhamento com a **RDC 657/2022** (SaMD) e normas de tecnovigilância aplicáveis (RDC 67/2009 / NOTIVISA).

## 2. Monitoramento Proativo: Drift, Calibração e ECE

Quando houver implantação autorizada, o monitoramento contínuo utilizará a trilha de auditoria *append-only* (`persistence/audit.py`) e a CLI `check-regression` para acompanhar:

1. **Deriva de dados (*data drift*):** distribuição de metadados DICOM (fabricante, kVp, projeção, lateralidade conforme REQ-043) e taxa de entradas rejeitadas ou fora de especificação.
2. **Deriva de calibração e ECE:** acompanhamento periódico do *Expected Calibration Error* (ECE) e diagramas de confiabilidade (`calibration/temperature.py`) sobre amostras auditadas com padrão-ouro médico.
3. **Taxa de abstenção:** variação da banda *não avaliável* (`calibration/abstention.py`); aumentos abruptos disparam investigação de *domain shift* ou degradação de aquisição.
4. **Regressão de versão:** bloqueio de atualização que viole limiares pré-especificados em `10-clinical-evaluation-plan.md`.

## 3. Monitoramento Reativo: Queixas Técnicas e Tecnovigilância

1. **Canal de queixas técnicas e incidentes:** registro estruturado vinculado ao *git SHA*, SHA-256 dos pesos e ID da execução auditada (`persistence/audit.py`), articulado com `14-change-control-and-problem-resolution.md`.
2. **Classificação e notificação (ANVISA / NOTIVISA):** eventos adversos graves ou falhas sistemáticas com potencial de dano (H-01/H-02 em `04-risk-file.md`) seguirão fluxos e prazos legais de tecnovigilância após eventual registro sanitário.
3. **Ações corretivas e preventivas (CAPA):** suspensão preventiva de pesos afetados, revisão de limiares de abstenção e emissão de aviso de segurança aos usuários habilitados.

## 4. Periodicidade e Revisão

- Em estágio atual (pesquisa): apenas verificação automatizada em CI (`check_honesty.py`, `trace.py --check`, `soup.py --check` e `check-regression`).
- Em eventual operação futura: relatórios periódicos de PMS e atualização do arquivo de gerenciamento de risco (`04-risk-file.md`).
