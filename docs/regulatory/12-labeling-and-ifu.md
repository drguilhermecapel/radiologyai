doc_id: REG-12
title: Rotulagem e Instruções de Uso (IFU) — Rascunho Preparatório
status: RASCUNHO — v0.1
date: 2026-11
author: Dr. Guilherme Capel Pasqua (CRM-SP 175873)
supersedes: —

# Rotulagem e Instruções de Uso (IFU) — RadiologyAI

> **AVISO REGULATÓRIO OBRIGATÓRIO:** Software em fase de pesquisa (**research-only**). **NÃO é dispositivo médico aprovado pela ANVISA** (RDC 657/2022) e **NÃO possui autorização para uso clínico, diagnóstico, triagem ou decisão terapêutica** em pacientes reais. Este documento é um **rascunho preparatório** para alinhamento futuro na Fase 5 e não constitui rotulagem comercial válida.

## 1. Identificação do Software e Uso Pretendido (em pesquisa)

- **Nome do software:** RadiologyAI (v2 — esqueleto experimental).
- **Modalidade alvo inicial:** Radiografia de tórax (CXR) em ambiente experimental/acadêmico.
- **Finalidade pretendida (futura, sujeita a validação clínica e registro):** Apoio computacional à leitura médica de exames radiológicos, com exibição de probabilidades calibradas, banda de abstenção (*achado provável / não avaliável / achado improvável*) e mapas de explicabilidade (Grad-CAM).
- **Contraindicações e restrições atuais:**
  - Proibido o uso em assistência médica real, laudos clínicos, triagem de emergência ou substituição do médico radiologista.
  - Proibido o uso com pesos não calibrados ou sem verificação de integridade (SHA-256).

## 2. Requisitos de Rotulagem Eletrônica (RDC 657/2022 — Preparatório)

Quando aplicável em etapa futura pós-validação e submissão regulatória, a interface ("Sobre" / rodapé / resposta da API) deverá exibir de forma clara:

1. **Identificação única do software e versão:** versão semântica, *git SHA* e *hash* SHA-256 dos pesos do modelo carregado.
2. **Fabricante / Responsável Técnico:** a definir após formalização jurídica e Sistema de Gestão da Qualidade (SGQ).
3. **Número de registro/notificação ANVISA:** **não aplicável no momento (software não registrado)**.
4. **Aviso de uso restrito:** indicação explícita de que o resultado é auxiliar e exige revisão independente de profissional médico habilitado.

## 3. Instruções de Uso (IFU) — Resumo Operacional

1. **Entrada aceita:** imagens radiográficas em formato compatível conforme especificação da API (`/api/v1/*` em `docs/OPENAPI_SPEC.yaml`), com verificação de metadados DICOM quando aplicável (incluindo lateralidade, conforme REQ-043).
2. **Interpretação das saídas:**
   - **Achado provável / Achado improvável:** escores calibrados por *temperature scaling* (`calibration/temperature.py`), sujeitos a incerteza estatística.
   - **Não avaliável (abstenção):** acionado por `calibration/abstention.py` quando a confiança não atinge o limiar definido; exige avaliação humana integral sem apoio do escore.
   - **Mapa Grad-CAM:** visualização auxiliar gerada por `explain/gradcam.py`; não delimita lesões com precisão anatômica diagnóstica.
3. **Trilha de auditoria:** toda execução registra evento *append-only* encadeado por hash em `persistence/audit.py` (hash do exame, card, SHA-256 dos pesos, git SHA, saída e timestamp).

## 4. Limitações Conhecidas e Riscos Residuais

- Desempenho não validado em coortes clínicas prospectivas ou multicêntricas brasileiras.
- Sensibilidade a mudança de domínio (*domain shift*), equipamentos não representados no treinamento e artefatos de aquisição.
- Consulte `04-risk-file.md` e `10-clinical-evaluation-plan.md` para detalhes de mitigação e critérios de aceitação pendentes.
