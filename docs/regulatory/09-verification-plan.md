---
doc_id: REG-09
title: Plano de Verificação de Software (IEC 62304 §5.5–5.7)
status: RASCUNHO — v0.3
date: 2026-11
author: Dr. Guilherme Capel Pasqua (CRM-SP 175873)
depends_on: [REG-05, REG-07]
---

# Plano de Verificação

> Verificação responde "construímos o software certo do jeito especificado?". A pergunta
> "o software serve ao uso clínico?" é validação e está no
> [plano de avaliação clínica](10-clinical-evaluation-plan.md). Software de pesquisa; não é
> dispositivo médico.

Objetivos: demonstrar, a cada mudança, que (1) todo requisito de
[`06-requirements/`](06-requirements/) está implementado e verificado por teste reprodutível; (2)
nenhum defeito da classe histórica do v1 (métrica fabricada, *fallback* silencioso, saliência
simulada, hash inválido) pode ser reintroduzido; (3) os controles de risco do
[arquivo de risco](04-risk-file.md) continuam eficazes.

## 1. Níveis de verificação

| Nível | Onde | Critério de aceitação |
|---|---|---|
| Unidade | `tests/unit/<área>/test_*.py` | asserções explícitas por unidade; cobertura de linhas ≥ 85% do pacote (`--cov-fail-under=85`) |
| Integração | `tests/integration/` | API, pipeline e persistência exercitados juntos com entradas sintéticas ou públicas |
| Sistema | executores de avaliação e testes ponta a ponta | resultados reproduzíveis com a mesma semente; nenhum resultado sem modelo real |
| Requisito | testes com `@pytest.mark.requirement("REQ-NNN")` | todo requisito tem ao menos um teste verificador (`scripts/trace.py --check`) |
| Estático | ruff, mypy `--strict`, `scripts/check_honesty.py`, bandit | zero violações |
| Pesos reais | jobs `ml` (Python 3.11) e `colab` (Python 3.13, torch mais recente) de `.github/workflows/ci.yml` | testes do backend, do Grad-CAM e da linha de base aprovados; sha256 do card conferido |
| Trava de regressão | `tests/evaluation/test_baseline_regression.py`; `radiologyai check-regression` | queda de AUROC macro acima da tolerância, menos rótulos avaliados ou dataset não externo ao treino ⇒ reprovação |
| SOUP e segurança | `scripts/soup.py --check`; `.github/workflows/security.yml` (gitleaks, bandit, pip-audit/OSV, SBOM CycloneDX) | lista SOUP igual ao `uv.lock`; zero segredos; pip-audit ainda não bloqueia ([REG-11](11-cybersecurity-lgpd.md) §5) |

Todo teste de requisito usa `@pytest.mark.requirement("REQ-NNN")` sob `--strict-markers`; as
execuções do GitHub Actions ficam vinculadas ao SHA do commit e à PR.

## 2. Verificações que são controles de risco

| Verificação | Comando ou teste | Perigo |
|---|---|---|
| Falha fechada com pesos ausentes ou hash divergente | testes de REQ-020, REQ-030 | RISK-004 |
| Gate de escopo roda antes do modelo | testes de REQ-031, REQ-042 | RISK-003 |
| MONOCHROME1 e Modality LUT | testes de REQ-005, REQ-006 com DICOM sintético | RISK-005 |
| Grad-CAM nulo para gradiente nulo | teste de REQ-060 | RISK-008 |
| Auditoria detecta linha adulterada | testes de REQ-070, REQ-071 | RISK-007 |
| Disjunção por paciente entre manifests | teste de REQ-051 | RISK-007, RISK-015 |
| Nível calculado, sem campo digitável | testes de REQ-090, REQ-093 | RISK-007, RISK-019 |
| Nível D fora da visão principal | teste de REQ-091 | RISK-012 |
| Sem aleatoriedade fora da avaliação | teste de REQ-002 e `check_honesty.py` R2 | RISK-007 |
| Nenhum `except ImportError` em `src/` | teste de REQ-001 e `check_honesty.py` R4 | RISK-003 |
| Nenhum número de desempenho sem proveniência em Markdown | `check_honesty.py` R1 | RISK-007 |
| sha256 válido em YAML/JSON | `check_honesty.py` R3 | RISK-004 |
| `import radiologyai` leve | passo do CI | — |

## 3. Testes com pesos reais

Testes que precisam de pesos ou dados baixados pulam de forma limpa no CI sem eles
(`pytest.mark.skipif` por `importlib.util.find_spec` ou existência do arquivo). Eles **são
executados** no ambiente de desenvolvimento onde os pesos existem, e o job `ml` do CI executa os
testes do backend TorchXRayVision com pesos reais. Um teste que só pula nunca conta como
verificação de um requisito: o relatório de verificação registra onde cada teste de pesos rodou.

## 4. Reprodutibilidade das avaliações

- Toda avaliação grava `artifacts/eval/<run_id>/metrics.json` com git sha, sha256 dos pesos,
  hash do manifesto do dataset, semente, versões de biblioteca, `not_evaluated` e limitações.
- Duas execuções com a mesma semente e o mesmo manifesto devem produzir as mesmas métricas
  (exceto carimbo de tempo).
- Relatórios `reports/EVALUATION_*.md` são gerados por `scripts/report.py` a partir do artefato,
  nunca escritos à mão.
- `.github/workflows/eval.yml` executa `radiologyai evaluate` sob demanda (`workflow_dispatch`)
  com card, manifest, número de reamostragens e semente explícitos, e publica `artifacts/eval/`
  como artefato do Actions. Um número só entra em documento depois que o `metrics.json`
  correspondente estiver commitado em `artifacts/eval/` (ou como registro em `evidence/`).
- Relatos de avaliação em `docs/evaluation/` sem artefato commitado **não** são verificação:
  ver [`HONEST_STATUS.md`](../../HONEST_STATUS.md).

## 5. Protocolos e relatórios de verificação

- **Protocolo antes, relatório depois.** O protocolo de verificação de uma liberação é o conjunto
  de testes e portões do commit etiquetado; o relatório é a saída do CI daquele commit.
- Pendente: gravar o relatório de verificação por commit em `artifacts/verification/<git_sha>/`
  (saída do pytest em JUnit XML, cobertura, resultado de cada portão) e anexá-lo à liberação.

## 6. Verificações planejadas por portão

| Portão | Verificação adicional |
|---|---|
| G0 | `git log --all -- medai_security.db` vazio; gitleaks limpo no histórico completo |
| G1 | todo registro A reproduz de checkout limpo; `levels.json` regenerado é idêntico |
| G1 | lateralidade propagada do DICOM (RISK-006) — coberta em RX (REQ-043, `TestXRLaterality`) e mamografia (REQ-141); requisito e teste a criar para TC, US e eco; verificação de REQ-043 em pipeline de RX com dado real |
| G1 | biomarcador abstém sem pré-condição (RISK-011) — REQ-133, REQ-135, REQ-136, REQ-138 escritos; a verificação deles é exigida antes do fechamento de G1 para TC |
| G2 | paridade torch ↔ ONNX para o modelo próprio (diferença absoluta máxima < `1e-4`, amostra de 100 exames); conjunto de teste sequestrado com hash registrado antes da avaliação; trava de regressão contra a linha de base `artifacts/eval/xrv-densenet121-pc__20260912T202549Z/metrics.json` |
| Plataforma | assinatura invalidada por mudança de conteúdo (RISK-018; REQ-156); RBAC por papel (RISK-020; REQ-151); testes ponta a ponta no navegador (pendentes) |

---

## Histórico de revisões

| Versão | Data | Alteração |
|---|---|---|
| v0.1 | 2026-11 | Criação |
| v0.2 | 2026-11-25 | Verificações de G1 e da plataforma ligadas aos requisitos REQ-1xx escritos |
| v0.3 | 2026-11-27 | Consolidação com a v0.1 de `main` (`f45beff`): objetivos, jobs `ml`/`colab`, trava de regressão, segurança/SBOM, `eval.yml`, paridade ONNX, REQ-043 |
