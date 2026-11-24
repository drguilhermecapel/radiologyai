doc_id: REG-05
title: Plano de Desenvolvimento de Software (IEC 62304 §5.1)
status: RASCUNHO — v0.1
date: 2026-11
author: Dr. Guilherme Capel Pasqua (CRM-SP 175873)
supersedes: —
# Plano de Desenvolvimento de Software

> **Estado atual:** software em estágio de pesquisa — não é dispositivo médico. Este plano documenta as práticas de engenharia e controle de configuração adotadas no núcleo v2 do **RadiologyAI** conforme **IEC 62304 §5.1**. Ver [`HONEST_STATUS.md`](../../HONEST_STATUS.md).

## 1. Modelo de ciclo de vida e fases

O desenvolvimento segue um modelo iterativo incremental em fases com portões de saída objetivos definidos em [`ROADMAP.md`](../../ROADMAP.md) §5:
- **Fase 0:** quarentena do legado (`legacy/`), retratação de alegações fabricadas, fundação regulatória e CI.
- **Fase 1:** núcleo v2 (`io/`, `modalities/`, `models/`, `inference/`, `evaluation/`) e linha de base honesta reproduzível.
- **Fase 2:** calibração, abstenção, explicabilidade real (Grad-CAM), auditoria encadeada, API FastAPI e rastreabilidade automatizada.
- **Fase 3:** treinamento próprio em radiografia de tórax, validação externa em dados com rótulo adjudicado (VinDr-CXR) e exportação ONNX.
- **Fases 4–6:** expansão multimodal (`us`, `ct`, `mr`), endurecimento regulatório/IFU e QMS ISO 13485.

## 2. Ambiente de desenvolvimento e pilha técnica

| Componente | Padrão adotado | Justificativa |
|---|---|---|
| **Linguagem** | Python `>=3.11,<3.14` | Compatibilidade entre CI reproduzível (3.11) e ambiente Colab (3.13) |
| **Empacotamento e lockfile** | `pyproject.toml` (Hatchling) + `uv.lock` | Instalação determinística (`uv sync --frozen`) e controle de SOUP |
| **Núcleo leve** | Sem dependência obrigatória de `torch` | Importação rápida e CI de CPU sem degradação silenciosa |
| **Stack de ML (extra `[ml]`)** | PyTorch, torchvision, MONAI, `timm`, `torchxrayvision`, `onnxruntime` | Ecossistema auditável com exportação determinística para ONNX |
| **Qualidade estática** | `ruff` (lint + format), `mypy --strict`, `bandit` | Detecção precoce de defeitos de tipagem, estilo e segurança |

## 3. Gestão de configuração e controle de versão (IEC 62304 §8)

- **Repositório:** GitHub (`drguilhermecapel/radiologyai`).
- **Estratégia de branches:** desenvolvimento em branches dedicadas integradas via Pull Request com aprovação de todos os portões de CI; código legado isolado em `legacy/` e excluído do pacote distribuído.
- **Rastreabilidade de artefatos:** cada execução de avaliação registra `git_sha`, `weights_sha256`, `manifest_sha256`, `seed` e versões de biblioteca em `artifacts/eval/`.

## 4. Gestão de SOUP (Software of Unknown Provenance — IEC 62304 §5.3.3 e §7.4)

- Todas as dependências declaradas são inventariadas automaticamente em [`08-soup-list.md`](08-soup-list.md) por `scripts/soup.py`.
- O CI bloqueia divergências entre `pyproject.toml`/`uv.lock` e a lista SOUP (`scripts/soup.py --check`).
- O workflow semanal `.github/workflows/security.yml` executa varredura de segredos (`gitleaks`), análise estática (`bandit`), auditoria de vulnerabilidades conhecidas (`pip-audit` / OSV) e geração de SBOM CycloneDX.

## 5. Verificação, rastreabilidade e portões de integração

Nenhuma alteração no núcleo v2 é considerada pronta sem aprovação simultânea de:
1. `ruff check` e `ruff format --check`;
2. `mypy --strict src/radiologyai`;
3. `pytest` (testes unitários, integração e backend real com pesos verificados por SHA-256);
4. `scripts/trace.py --check` (cobertura completa `REQ <-> TEST <-> RISK`);
5. `scripts/soup.py --check` (integridade do inventário SOUP);
6. `scripts/check_honesty.py` (bloqueio automático de alegações sem proveniência, aleatoriedade oculta, imports silenciosos e marcadores de simulação).

## 6. Documentação do ciclo de vida

Os artefatos regulatórios são versionados no próprio repositório em `docs/regulatory/` e evoluem em paralelo ao código, seguindo o índice em [`docs/regulatory/README.md`](README.md).

## Histórico de revisões

| Versão | Data | Alteração |
|---|---|---|
| v0.1 | 2026-11 | Criação do Plano de Desenvolvimento de Software (Fase 0, IEC 62304 §5.1) |
