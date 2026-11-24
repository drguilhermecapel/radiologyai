doc_id: REG-04
title: Arquivo de Risco e Análise de Perigos (ISO 14971)
status: RASCUNHO — v0.1
date: 2026-11
author: Dr. Guilherme Capel Pasqua (CRM-SP 175873)
supersedes: —
# Arquivo de Risco / Análise de Perigos

> **Estado atual:** software em estágio de pesquisa — não é dispositivo médico. Os riscos residuais abaixo refletem controles de engenharia em desenvolvimento; a aceitação formal do risco residual global exige avaliação clínica concluída (Fase 3) e QMS ISO 13485 (Fase 6). Ver [`HONEST_STATUS.md`](../../HONEST_STATUS.md).

Este documento materializa a análise de perigos e o controle de riscos do **RadiologyAI** conforme [`03-risk-management-plan.md`](03-risk-management-plan.md) e **ISO 14971:2019**. Cada controle `RISK-001`…`RISK-010` corresponde diretamente aos perigos semente `H-01`…`H-10` de [`ROADMAP.md`](../../ROADMAP.md) §6.3 e rastreia aos requisitos verificados em [`13-traceability.md`](13-traceability.md).

## 1. Matriz de Análise de Perigos e Controles (FMEA)

| Controle | Perigo | Sequência de eventos e dano potencial | Risco inicial (S×P) | Controles de risco implementados / planejados | Requisitos vinculados | Verificação objetiva | Risco residual (S×P) | Estado |
|---|---|---|---|---|---|---|---|---|
| **RISK-001** | **H-01** Falso negativo em achado crítico | Modelo emite escore baixo em achado crítico (ex.: pneumotórax) → médico subestima risco → atraso terapêutico → dano grave ou morte | S5 × P3 (Inaceitável) | Banda de abstenção calibrada a partir de artefato medido (`calibration/abstention.py`); proibição de uso para *rule-out* em [`01-intended-use.md`](01-intended-use.md) e futura IFU | `REQ-062` | `TestThresholdsComeFromMeasuredArtifact`, `TestAbstentionPolicy` | S5 × P1 (ALARP) | Implementado (código); IFU pendente (Fase 5) |
| **RISK-002** | **H-02** Viés de automação | Saída excessivamente confiante ou rótulo "Normal" ancora o leitor humano e reduz escrutínio clínico | S4 × P3 (Inaceitável) | Proibição estrita de afirmar normalidade; exibição de incerteza e bandas em vez de *top-1* isolado; uso restrito a leitura concorrente | `REQ-063` | `TestNeverAssertsNormality` | S4 × P1 (ALARP) | Implementado (código); revisão de usabilidade pendente (Fase 5) |
| **RISK-003** | **H-03** Entrada fora de distribuição processada em silêncio | Exame lateral, pediátrico, modalidade incorreta ou dependência ausente gera predição espúria ou fallback silencioso | S4 × P4 (Inaceitável) | Gate de escopo `ModalityPlugin.validate()` executado antes do modelo; falha fechada na ausência de dependência; proibição de `except ImportError` em `src/` | `REQ-001`, `REQ-031`, `REQ-042` | `test_no_silent_import_fallbacks`, `TestScopeGateOverHTTP`, `TestScopeGateRunsBeforeModel`, `TestXRScopeGate` | S4 × P1 (ALARP) | Implementado e verificado no CI |
| **RISK-004** | **H-04** Descasamento de pesos/versão ou métricas fabricadas | Pesos corrompidos, modelo incorreto, cabeças não treinadas ou métricas simuladas na API induzem decisão sobre saída inválida | S4 × P4 (Inaceitável) | Verificação obrigatória de SHA-256 dos pesos na carga; falha fechada na inferência; cabeças não treinadas marcadas explicitamente; API nunca fabrica métricas | `REQ-020`, `REQ-021`, `REQ-030`, `REQ-080` | `TestSha256Validation`, `test_untrained_heads_are_marked_not_silently_named`, `TestFailClosed`, `TestMetricsNeverFabricated` | S4 × P1 (ALARP) | Implementado e verificado no CI |
| **RISK-005** | **H-05** Má interpretação de DICOM | Inversão fotométrica `MONOCHROME1` omitida ou `RescaleSlope`/`Intercept`/VOI LUT ignorados distorcem a imagem antes da inferência | S4 × P3 (Inaceitável) | Pipeline `io/` preserva faixa dinâmica com Modality LUT e inverte `MONOCHROME1` explicitamente | `REQ-005`, `REQ-006` | `TestModalityLUT`, `TestMonochrome1` | S4 × P1 (ALARP) | Implementado e verificado no CI |
| **RISK-006** | **H-06** Erro de lateralidade | Achado reportado com lado anatômico invertido (direito/esquerdo) leva a procedimento no lado errado | S5 × P2 (Inaceitável) | Lateralidade deve ser propagada de metadados DICOM confiáveis e asserida explicitamente, nunca inferida apenas do pixel; proibição de flip horizontal silencioso na inferência | Pendente (`REQ-043` planejado na Fase 3) | Teste unitário dedicado de preservação de lateralidade (Fase 3) | S5 × P2 (Aberto — bloqueia uso clínico) | **Pendente** (Fase 3) |
| **RISK-007** | **H-07** Deriva silenciosa, não reprodutibilidade ou perda de auditoria | Alteração de modelo/dados ou aleatoriedade oculta altera comportamento sem detecção; impossibilidade de reconstruir decisão | S3 × P4 (Inaceitável) | Proibição de `np.random` fora de avaliação; execução garantida de toda suíte de testes; avaliação reproduzível com proveniência completa; ausência de vazamento por paciente; log append-only encadeado por hash | `REQ-002`, `REQ-003`, `REQ-050`, `REQ-051`, `REQ-070`, `REQ-071` | `test_no_random_in_non_evaluation_code`, `TestPatientLeakage`, `TestTamperDetection`, `TestDecisionReconstruction` | S3 × P1 (Aceitável) | Implementado (núcleo); PMS pendente (Fase 5) |
| **RISK-008** | **H-08** Explicabilidade enganosa | Mapa de saliência gerado por heurística de borda (ex.: Canny) não reflete o modelo real e induz falsa confiança anatômica | S3 × P4 (Inaceitável) | Grad-CAM real sobre gradientes da última camada convolucional; teste de regressão exige mapa nulo quando gradientes são nulos | `REQ-060` | `test_null_gradients_yield_null_map` | S3 × P1 (Aceitável) | Implementado e verificado no CI |
| **RISK-009** | **H-09** Vazamento de PHI / violação LGPD | Identificadores de paciente permanecem em cabeçalhos DICOM, logs ou mensagens de erro | S3 × P3 (ALARP) | Perfil de des-identificação DICOM PS3.15 com pseudo-identificadores determinísticos e remoção verificada de tags sensíveis | `REQ-010`, `REQ-011` | `TestDeterminism`, `TestTagRemoval` | S3 × P1 (Aceitável) | Implementado (DICOM); DPIA/cibersegurança pendente (Fase 3) |
| **RISK-010** | **H-10** Confiança mal calibrada | Usuário interpreta escore bruto do modelo como probabilidade real de doença | S3 × P4 (Inaceitável) | Calibração por *temperature scaling* preservando ordenação (AUROC inalterado), reporte obrigatório de ECE e aviso explícito de que escores não equivalem a probabilidade diagnóstica | `REQ-061` | `test_preserves_ranking_and_therefore_auroc` | S3 × P1 (Aceitável) | Implementado (calibração); IFU pendente (Fase 5) |

## 2. Riscos introduzidos pelos controles de risco

- **Rejeição legítima de exames fora de escopo (`RISK-003`) e abstenção (`RISK-001`):** podem aumentar a proporção de exames sem escore assistivo, mas como o uso pretendido é leitura concorrente por médico qualificado (sem triagem nem automação), a falha fechada ou abstenção explícita é estritamente mais segura do que uma predição espúria.

## 3. Avaliação de completude e pendências abertas

- **9 de 10 controles de risco (`RISK-001`–`RISK-005`, `RISK-007`–`RISK-010`)** já possuem requisitos formais e verificação automatizada ativa em [`13-traceability.md`](13-traceability.md).
- **`RISK-006` (`H-06`, lateralidade):** permanece aberto até a implementação de `REQ-043` e validação em pipeline próprio de RX (Fase 3).
- **Decisão atual de aceitabilidade global:** **NÃO LIBERADO PARA USO CLÍNICO** até fechamento de `RISK-006`, conclusão da avaliação clínica externa (Fase 3), IFU/usabilidade (Fase 5) e auditoria sob QMS ISO 13485 (Fase 6).

## Histórico de revisões

| Versão | Data | Alteração |
|---|---|---|
| v0.1 | 2026-11 | Criação do Arquivo de Risco inicial com FMEA completo de H-01 a H-10 (Fase 0) |
