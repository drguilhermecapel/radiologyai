doc_id: REG-09
title: Plano e Protocolos de Verificação de Software (IEC 62304 §5.5–5.7)
status: RASCUNHO — v0.1
date: 2026-11
author: Dr. Guilherme Capel Pasqua (CRM-SP 175873)
supersedes: —
# Plano e Protocolos de Verificação de Software

> **Estado atual:** software em estágio de pesquisa — não é dispositivo médico. Este plano define a estratégia de verificação automatizada do núcleo v2 conforme **IEC 62304 §5.5–5.7**. Ver [`HONEST_STATUS.md`](../../HONEST_STATUS.md).

## 1. Objetivos da verificação

Demonstrar objetivamente, a cada alteração de código, que:
1. Todos os requisitos de software em `docs/regulatory/06-requirements/` estão implementados e verificados por testes reproduzíveis.
2. Nenhum defeito da classe histórica do protótipo v1 (métricas fabricadas, fallback silencioso, saliência simulada, hashes inválidos) pode ser reintroduzido.
3. Os controles de risco implementados em código ([`04-risk-file.md`](04-risk-file.md)) mantêm eficácia verificável após cada mudança.

## 2. Níveis de verificação e portões de CI

| Nível | Escopo | Ferramenta / Comando | Critério de aprovação |
|---|---|---|---|
| **Análise estática** | Estilo, lint e tipagem estrita | `ruff check`, `ruff format --check`, `mypy --strict src/radiologyai` | Zero erros |
| **Testes unitários** | Módulos em `src/radiologyai/` | `pytest tests/unit` | Todos os testes aprovados; nenhum diretório silenciado (`REQ-003`) |
| **Testes de integração** | API FastAPI, gate de escopo sobre HTTP e contrato de resposta | `pytest tests/integration` | Todos os testes aprovados |
| **Backend real e Colab** | Pesos reais `torchxrayvision`, SHA-256, cabeças não treinadas e Grad-CAM (py3.11 e py3.13) | Jobs `ml` e `colab` em `.github/workflows/ci.yml` | Todos os testes aprovados e SHA-256 conferido |
| **Rastreabilidade** | Vínculo bidirecional `REQ <-> TEST <-> RISK` | `python scripts/trace.py --check` | Exit 0; [`13-traceability.md`](13-traceability.md) atualizada |
| **Integridade SOUP** | Dependências declaradas vs inventário | `python scripts/soup.py --check` | Exit 0; [`08-soup-list.md`](08-soup-list.md) atualizada |
| **Guardião de honestidade** | Proibição de alegações sem proveniência, `np.random` indevido, `except ImportError` e simulação | `python scripts/check_honesty.py` | Zero violações |
| **Segurança e SBOM** | Segredos no histórico, SAST, CVEs em SOUP e SBOM CycloneDX | `.github/workflows/security.yml` | Zero segredos vazados e zero achados críticos |

## 3. Critérios de evidência e registros

- Todo teste de requisito utiliza `@pytest.mark.requirement("REQ-xxx")` sob `--strict-markers`.
- Execuções em GitHub Actions produzem logs vinculados ao SHA do commit e ao Pull Request.
- Avaliações quantitativas geram artefatos versionados em `artifacts/eval/<run_id>/` e relatórios `reports/EVALUATION_*.md` com `git_sha`, `weights_sha256`, `manifest_sha256` e `seed`.

## 4. Pendências planejadas de verificação (Fases 2–3)

- Teste de regressão dourada em CI (`tests/evaluation/test_baseline_regression.py`).
- Verificação de paridade numérica PyTorch ↔ ONNX (diferença máxima inferior a `1e-4`) após exportação na Fase 3.
- Verificação formal de preservação de lateralidade (`REQ-043` / `RISK-006`).

## Histórico de revisões

| Versão | Data | Alteração |
|---|---|---|
| v0.1 | 2026-11 | Criação do Plano e Protocolos de Verificação (Fase 2, IEC 62304 §5.5–5.7) |
