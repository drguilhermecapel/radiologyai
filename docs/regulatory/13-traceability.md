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
| **REQ-021** Saídas de modelo não treinadas não são reportadas | safety | RISK-004 | implemented | `test_untrained_heads_are_marked_not_silently_named` |
| **REQ-030** Inferência falha fechada | safety | RISK-004 | implemented | `TestFailClosed` |
| **REQ-031** Gate de escopo antes do modelo | safety | RISK-003 | implemented | `TestScopeGateOverHTTP`<br>`TestScopeGateRunsBeforeModel` |
| **REQ-042** Limites do uso pretendido em radiografia | safety | RISK-003 | implemented | `TestXRScopeGate` |
| **REQ-050** Avaliação reprodutível | functional | RISK-007 | implemented | `test_same_seed_is_bit_reproducible`<br>`test_records_full_provenance`<br>`test_same_seed_is_reproducible` |
| **REQ-051** Ausência de vazamento por paciente | safety | RISK-007 | implemented | `TestPatientLeakage` |
| **REQ-060** Explicabilidade derivada do modelo real | safety | RISK-008 | implemented | `test_null_gradients_yield_null_map` |
| **REQ-061** Calibração preserva a ordenação | functional | RISK-010 | implemented | `test_preserves_ranking_and_therefore_auroc` |
| **REQ-062** Política de abstenção a partir de medição | safety | RISK-001 | implemented | `TestAbstentionPolicy` |
| **REQ-063** O sistema nunca afirma normalidade | safety | RISK-002 | implemented | `TestNeverAssertsNormality` |
| **REQ-070** Trilha de auditoria à prova de adulteração | safety | RISK-007 | implemented | `TestTamperDetection` |
| **REQ-071** Reconstrução completa da decisão | safety | RISK-007 | implemented | `TestDecisionReconstruction` |
| **REQ-080** API não fabrica métricas | safety | RISK-004 | implemented | `TestMetricsNeverFabricated` |

**20 requisitos · 23 verificações.**

> Os controles de risco (RISK-xxx) serão detalhados em
> `04-risk-file.md`. As sementes H-01…H-10 estão em `ROADMAP.md` §6.3.
