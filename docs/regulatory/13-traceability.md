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
| **REQ-003** Toda verificação declarada é executada | safety | RISK-007 | implemented | `test_nenhum_diretorio_de_teste_casa_com_norecursedirs`<br>`test_todo_arquivo_de_teste_foi_importado` |
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
| **REQ-062** Política de abstenção a partir de medição | safety | RISK-001 | implemented | `TestThresholdsComeFromMeasuredArtifact`<br>`TestAbstentionPolicy` |
| **REQ-063** O sistema nunca afirma normalidade | safety | RISK-002 | implemented | `TestNeverAssertsNormality` |
| **REQ-070** Trilha de auditoria à prova de adulteração | safety | RISK-007 | implemented | `TestTamperDetection` |
| **REQ-071** Reconstrução completa da decisão | safety | RISK-007 | implemented | `TestDecisionReconstruction` |
| **REQ-080** API não fabrica métricas | safety | RISK-004 | implemented | `TestMetricsNeverFabricated` |
| **REQ-090** Nível de evidência é calculado, nunca digitado | safety | RISK-007 | implemented | `test_condition_cannot_declare_a_level`<br>`test_clean_passing_external_measurement_is_level_a`<br>`test_contaminated_external_measurement_is_only_level_c`<br>`test_evidence_record_has_no_level_field`<br>`test_failing_or_undecided_records_do_not_count`<br>`test_internal_measurement_is_level_c`<br>`test_no_evidence_is_level_d`<br>`test_one_passing_clean_external_measurement_is_enough`<br>`test_precedence_a_over_b_over_c`<br>`test_records_of_different_triples_are_rejected`<br>`test_unverified_published_evidence_does_not_count`<br>`test_verified_published_evidence_is_level_b` |
| **REQ-091** Achado exploratório não aparece na visão principal | safety | RISK-001 | implemented | `test_condition_without_generated_level_is_research_only`<br>`test_hand_edited_levels_cannot_promote_level_d`<br>`test_packaged_catalog_loads_empty_and_unknown_triple_is_level_d`<br>`test_level_entry_rejects_level_d_in_main_view`<br>`test_visibility_follows_level` |
| **REQ-092** Decisão exige ponto de operação com origem declarada | safety | RISK-001 | implemented | `test_every_verdict_names_the_points_it_used`<br>`test_no_operating_point_means_score_only`<br>`test_operating_point_requires_declared_source`<br>`test_points_on_unavailable_scale_are_ignored` |
| **REQ-093** Medição própria reprovada prevalece sobre alegação publicada | safety | RISK-007 | implemented | `test_failed_clean_external_measurement_refutes_internal_measurement`<br>`test_failed_clean_external_measurement_refutes_published_claims`<br>`test_failed_contaminated_external_measurement_does_not_refute`<br>`test_undecided_clean_external_measurement_does_not_refute` |

**25 requisitos · 51 verificações.**

> Os controles de risco (RISK-xxx) serão detalhados em
> `04-risk-file.md`. As sementes H-01…H-10 estão em `ROADMAP.md` §6.3.
