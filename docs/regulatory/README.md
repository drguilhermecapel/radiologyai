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
| 03 | Plano de Gestão de Risco (ISO 14971) | pendente — Fase 0 |
| 04 | Arquivo de Risco / Análise de Perigos | pendente — Fase 0 (sementes H-01…H-10 em ROADMAP §6.3) |
| 05 | Plano de Desenvolvimento de Software (62304 §5.1) | pendente — Fase 0 |
| 06 | Especificação de Requisitos (`06-requirements/REQ-*.md`) | pendente — Fase 1 |
| 07 | Descrição da Arquitetura (62304 §5.3) | pendente — Fase 2 (rascunho em ROADMAP §4) |
| 08 | Lista SOUP (62304 §5.3.3, §8.1.2) | pendente — Fase 1, gerada de `uv.lock` |
| 09 | Plano e Protocolos de Verificação | pendente — Fase 2 |
| 10 | Plano de Avaliação Clínica | pendente — Fase 3, **antes** da avaliação definitiva |
| 11 | Cibersegurança + LGPD/DPIA | pendente — Fase 3 |
| 12 | Rotulagem / IFU (pt-BR) | pendente — Fase 5 |
| 13 | Matriz de Rastreabilidade | pendente — Fase 2, **gerada** por `scripts/trace.py` |
| 14 | Controle de Mudanças e Resolução de Problemas | pendente — Fase 2 |
| 15 | Plano de Vigilância Pós-Mercado | pendente — Fase 5 |
| 16 | Model Cards e Dataset Datasheets | pendente — Fase 1, por artefato |

**Ordem de escrita:** 01 e 02 primeiro, antes de qualquer código do núcleo v2. Eles fixam
a classe de risco e a classe de segurança, das quais tudo o mais deriva.
