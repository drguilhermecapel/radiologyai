# Dossiê regulatório

Conjunto documental em construção para a trajetória SaMD/ANVISA descrita em
[`ROADMAP.md`](../../ROADMAP.md) §6.

> Nenhum documento aqui alega conformidade. São os artefatos de planejamento e
> desenvolvimento exigidos por IEC 62304, ISO 14971 e RDC 657/2022, sendo construídos
> em paralelo à engenharia — não depois dela.

| # | Documento | Estado |
|---|---|---|
| 01 | [Uso Pretendido / Indicações de Uso](01-intended-use.md) | rascunho v0.1 |
| 02 | [Classificação de Segurança do Software](02-software-safety-classification.md) | rascunho v0.1 |
| 03 | [Plano de Gestão de Risco (ISO 14971)](03-risk-management-plan.md) | rascunho v0.1 (Fase 0) |
| 04 | [Arquivo de Risco / Análise de Perigos](04-risk-file.md) | rascunho v0.1 — FMEA H-01…H-10 (Fase 0) |
| 05 | [Plano de Desenvolvimento de Software (62304 §5.1)](05-software-development-plan.md) | rascunho v0.1 (Fase 0) |
| 06 | [Especificação de Requisitos](06-requirements/) | **21 requisitos**, todos verificados |
| 07 | [Descrição da Arquitetura (62304 §5.3)](07-software-architecture.md) | rascunho v0.1 (Fase 2) |
| 08 | [Lista SOUP](08-soup-list.md) | **gerada** por `scripts/soup.py`, verificada no CI |
| 09 | [Plano e Protocolos de Verificação](09-verification-plan.md) | rascunho v0.1 (Fase 2) |
| 10 | [Plano de Avaliação Clínica](10-clinical-evaluation-plan.md) | rascunho v0.1 — pré-especificado (Fase 3) |
| 11 | [Cibersegurança + LGPD/DPIA](11-cybersecurity-and-lgpd-dpia.md) | rascunho v0.1 (Fase 3) |
| 12 | Rotulagem / IFU (pt-BR) | pendente — Fase 5 |
| 13 | [Matriz de Rastreabilidade](13-traceability.md) | **gerada** por `scripts/trace.py`, verificada no CI |
| 14 | [Controle de Mudanças e Resolução de Problemas](14-change-control-and-problem-resolution.md) | rascunho v0.1 (Fase 2) |
| 15 | Plano de Vigilância Pós-Mercado | pendente — Fase 5 |
| 16 | Model Cards e Dataset Datasheets | pendente — Fase 1, por artefato |

**Ordem de escrita:** 01 e 02 primeiro, antes de qualquer código do núcleo v2. Eles fixam
a classe de risco e a classe de segurança, das quais tudo o mais deriva.

## Portões automatizados

Três verificações de CI transformam estes documentos em evidência viva, em vez
de papel que diverge do código:

| Verificação | O que impõe |
|---|---|
| `scripts/trace.py --check` | Todo requisito tem teste verificador; nenhum teste cita requisito inexistente; todo requisito rastreia até um controle de risco |
| `scripts/soup.py --check` | A lista SOUP reflete as dependências declaradas; toda dependência tem propósito documentado |
| `scripts/check_honesty.py` | Nenhuma alegação de desempenho sem artefato de avaliação; nenhum sha256 inválido; nenhum import silencioso; nenhum marcador de simulação |

Uma matriz de rastreabilidade que pode divergir não vale nada. Uma que bloqueia
merge é evidência.
