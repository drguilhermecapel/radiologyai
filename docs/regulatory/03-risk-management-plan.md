doc_id: REG-03
title: Plano de Gestão de Risco (ISO 14971)
status: RASCUNHO — v0.1
date: 2026-11
author: Dr. Guilherme Capel Pasqua (CRM-SP 175873)
supersedes: —
# Plano de Gestão de Risco

> **Estado atual:** software em estágio de pesquisa — não é dispositivo médico e não possui aprovação regulatória ou autorização para uso clínico. Ver [`HONEST_STATUS.md`](../../HONEST_STATUS.md).

Este plano estabelece o processo de gestão de risco do **RadiologyAI** ao longo de todo o ciclo de vida do software, conforme **ISO 14971:2019**, **ISO/TR 24971:2020**, **IEC 62304 §7** e **RDC 657/2022 (ANVISA)**.

## 1. Escopo e enquadramento

- **Produto:** RadiologyAI (SaMD assistivo de leitura concorrente; ver [`01-intended-use.md`](01-intended-use.md)).
- **Classe de segurança de software:** IEC 62304 **Classe C** (ver [`02-software-safety-classification.md`](02-software-safety-classification.md)).
- **Enquadramento regulatório de trabalho:** ANVISA **Classe III** (RDC 751/2022, Regra 11), sujeito a confirmação formal em consultoria regulatória (Fase 5).
- **Abrangência do ciclo de vida:** concepção, engenharia de dados, treinamento, avaliação externa, verificação, implantação controlada, vigilância pós-mercado e descontinuação.

## 2. Responsabilidades e governança

| Papel | Responsabilidade na fase atual (solo / pesquisa) | Transição pré-submissão (Fase 6) |
|---|---|---|
| **Responsável clínico e técnico** | Dr. Guilherme Capel Pasqua (CRM-SP 175873): definição de perigos clínicos, critérios de aceitabilidade e aprovação de controles | Transferência para Comitê de Gestão de Risco sob QMS ISO 13485 |
| **Verificação automatizada** | Portões de CI (`scripts/trace.py`, `scripts/soup.py`, `scripts/check_honesty.py`, `pytest`, `bandit`, `gitleaks`) | Mantidos como evidência objetiva do Design History File (DHF) |

## 3. Escalas de severidade e probabilidade

### 3.1 Severidade do dano ao paciente (S)

| Nível | Categoria | Definição clínica |
|---|---|---|
| **S5** | Catastrófica | Morte ou lesão irreversível grave (ex.: atraso crítico no tratamento de pneumotórax hipertensivo ou hemorragia intracraniana) |
| **S4** | Crítica | Lesão permanente ou comprometimento clínico relevante que exige intervenção médica imediata |
| **S3** | Séria | Lesão reversível, atraso diagnóstico clinicamente significativo ou intervenção desnecessária invasiva |
| **S2** | Menor | Desconforto temporário, repetição de exame sem dano permanente ou atraso leve sem impacto prognóstico |
| **S1** | Desprezível | Sem dano ao paciente (ex.: indisponibilidade transitória de ferramenta assistiva com falha explícita) |

### 3.2 Probabilidade de ocorrência de dano (P)

Para software, conforme IEC 62304 §7.1.4 e ISO/TR 24971, a probabilidade de falha de software é conservadoramente assumida como **1** quando não há evidência empírica de controle; a estimativa de P considera a probabilidade de a situação perigosa ocorrer e resultar no dano sob leitura médica concorrente:

| Nível | Categoria | Critério qualitativo |
|---|---|---|
| **P5** | Frequente | Esperado ocorrer repetidamente no uso rotineiro |
| **P4** | Provável | Ocorre diversas vezes ao longo da vida útil do sistema |
| **P3** | Ocasional | Pode ocorrer algumas vezes no uso clínico |
| **P2** | Remota | Improvável, mas possível em cenários específicos |
| **P1** | Improvável | Tão improvável que se assume não ocorrer sob controles ativos |

## 4. Critérios de aceitabilidade de risco

| Zona | Combinações (S × P) | Ação obrigatória |
|---|---|---|
| **Inaceitável** | S5×(P2–P5), S4×(P3–P5), S3×(P4–P5) | Proibido liberar. Exige controle de risco por projeto antes de qualquer avanço |
| **ALARP (Reduzir tanto quanto praticável)** | S5×P1, S4×(P1–P2), S3×(P2–P3), S2×(P3–P5) | Implementar controles técnicos verificáveis e demonstrar que risco residual global é superado pelo benefício clínico |
| **Aceitável** | Demais combinações (S1×qualquer, S2×P1–P2, S3×P1) | Monitorar em vigilância pós-mercado; sem ação adicional imediata |

## 5. Hierarquia de controles de risco (ISO 14971 §7.1)

Os controles devem ser selecionados estritamente nesta ordem de prioridade:
1. **Segurança inerente por projeto:** falha fechada (`BackendUnavailableError`, verificação SHA-256), gate de escopo em `ModalityPlugin.validate()`, proibição de afirmar normalidade e abstenção em três bandas.
2. **Medidas de proteção no software ou processo:** portões automatizados no CI (`check_honesty.py`, `trace.py`, `soup.py`, `bandit`, `gitleaks`), trilha de auditoria encadeada por hash e des-identificação DICOM PS3.15.
3. **Informação de segurança (IFU / rotulagem):** contraindicações explícitas contra *rule-out*, leitura autônoma e uso fora de escopo (nunca usada isoladamente para justificar ausência de controle técnico quando o controle técnico é viável).

## 6. Rastreabilidade e verificação

Todo perigo registrado em [`04-risk-file.md`](04-risk-file.md) (`H-01`…`H-10` / `RISK-001`…`RISK-010`) deve rastrear para:
- um ou mais requisitos de segurança em `docs/regulatory/06-requirements/REQ-*.md`;
- um ou mais testes automatizados marcados com `@pytest.mark.requirement("REQ-xxx")`;
- verificação contínua via `scripts/trace.py --check` na matriz [`13-traceability.md`](13-traceability.md).

## 7. Revisão e manutenção do Arquivo de Risco

O Arquivo de Risco (`04-risk-file.md`) deve ser revisado obrigatoriamente:
- a cada mudança de uso pretendido ou adição de modalidade (`xr`, `us`, `ct`, `mr`);
- antes de qualquer avaliação externa formal (Fase 3) e antes de submissão regulatória (Fase 6);
- diante de anomalia crítica em SOUP ou evento adverso identificado em vigilância pós-mercado.

## Histórico de revisões

| Versão | Data | Alteração |
|---|---|---|
| v0.1 | 2026-11 | Criação do Plano de Gestão de Risco (Fase 0, ISO 14971:2019) |
