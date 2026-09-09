---
doc_id: REG-13
title: Matriz de Rastreabilidade
status: GERADO AUTOMATICAMENTE — não editar à mão
generated_by: scripts/trace.py
---

# Matriz de rastreabilidade

IEC 62304 §5.7 — cada requisito é verificado por ao menos um teste, e cada
requisito rastreia até um controle de risco (ISO 14971).

Gerada por `scripts/trace.py --write` e **verificada no CI** com `--check`.
O build falha se um requisito ficar sem teste ou se um teste citar um
requisito inexistente.

| Requisito | Tipo | Controle de risco | Estado | Verificado por |
|---|---|---|---|---|
| **REQ-001** Falhar alto na ausência de dependência | safety | RISK-003 | implemented | `test_no_silent_import_fallbacks` |
| **REQ-002** Nenhum número gerado aleatoriamente | safety | RISK-007 | implemented | `test_no_random_in_non_evaluation_code` |
| **REQ-005** Aplicar Modality LUT sem perda de faixa | functional | RISK-005 | implemented | `TestModalityLUT` |
| **REQ-006** Inverter MONOCHROME1 | safety | RISK-005 | implemented | `TestMonochrome1` |
| **REQ-010** Pseudo-identificadores determinísticos | safety | RISK-009 | implemented | `TestDeterminism` |
| **REQ-011** Cobertura de des-identificação | safety | RISK-009 | implemented | `TestTagRemoval` |
| **REQ-020** Integridade verificável dos pesos | safety | RISK-004 | implemented | `TestSha256Validation` |
| **REQ-030** Inferência falha fechada | safety | RISK-004 | implemented | `TestFailClosed` |
| **REQ-031** Gate de escopo antes do modelo | safety | RISK-003 | implemented | `TestScopeGateRunsBeforeModel` |
| **REQ-042** Limites do uso pretendido em radiografia | safety | RISK-003 | implemented | `TestXRScopeGate` |
| **REQ-050** Avaliação reprodutível | functional | RISK-007 | implemented | `test_same_seed_is_bit_reproducible` |
| **REQ-051** Ausência de vazamento por paciente | safety | RISK-007 | implemented | `TestPatientLeakage` |

**12 requisitos · 12 verificações.**

> Os controles de risco (RISK-xxx) serão detalhados em
> `04-risk-file.md`. As sementes H-01…H-10 estão em `ROADMAP.md` §6.3.
