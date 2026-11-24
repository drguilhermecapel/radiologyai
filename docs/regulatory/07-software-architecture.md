doc_id: REG-07
title: Descrição da Arquitetura de Software (IEC 62304 §5.3)
status: RASCUNHO — v0.1
date: 2026-11
author: Dr. Guilherme Capel Pasqua (CRM-SP 175873)
supersedes: —
# Descrição da Arquitetura de Software

> **Estado atual:** software em estágio de pesquisa — não é dispositivo médico. Esta especificação descreve a arquitetura implementada no núcleo v2 (`src/radiologyai/`) conforme **IEC 62304 §5.3**. Ver [`HONEST_STATUS.md`](../../HONEST_STATUS.md).

## 1. Visão geral e princípios arquiteturais

O pacote `radiologyai` (layout `src/radiologyai/`) substitui o protótipo legado (`legacy/`) sob quatro invariantes estruturais:
1. **Núcleo leve sem PyTorch obrigatório:** `import radiologyai` não importa `torch`, mantendo o CI de CPU rápido e falhando alto (`BackendUnavailableError`) quando o extra `[ml]` não está instalado (`REQ-001`).
2. **Falha fechada em toda fronteira crítica:** ausência de pesos, divergência de SHA-256 ou entrada fora de escopo interrompem a execução sem fallback heurístico (`REQ-020`, `REQ-030`, `REQ-031`).
3. **Extensibilidade multimodal por plugins:** cada modalidade (`xr`, `ct`, `mr`, `us`) é registrada via *entry points* em `pyproject.toml` (`radiologyai.modalities`), isolando pré-processamento, rótulos e validação de escopo.
4. **Rastreabilidade ponta a ponta:** toda inferência e avaliação propaga `git_sha`, identificador do model card, `weights_sha256` e registro em trilha de auditoria encadeada (`REQ-050`, `REQ-070`, `REQ-071`).

## 2. Decomposição em itens de software (IEC 62304 §5.3.1)

| Item (`src/radiologyai/`) | Responsabilidade | Classe | Requisitos |
|---|---|---|---|
| `io/` | Leitura DICOM (`pydicom`), Modality LUT sem perda de faixa, inversão `MONOCHROME1`, des-identificação PS3.15 | Classe C | `REQ-005`, `REQ-006`, `REQ-010`, `REQ-011` |
| `modalities/` | Contrato `ModalityPlugin`, registro por *entry points*, plugin `xr` (validação frontal/adulto/CR-DX e rótulos) e *stubs* `ct`, `mr`, `us` | Classe C | `REQ-031`, `REQ-042` |
| `models/` | `ModelCard` com validação SHA-256, backends de inferência (`backends/xrv.py`), marcação explícita de cabeças não treinadas | Classe C | `REQ-020`, `REQ-021` |
| `inference/` | Motor de inferência (`engine.py`) com gate de escopo antes do modelo e falha fechada | Classe C | `REQ-030`, `REQ-031` |
| `calibration/` | *Temperature scaling* preservando ordenação e abstenção em três bandas derivada de artefatos medidos, sem afirmar normalidade | Classe C | `REQ-061`, `REQ-062`, `REQ-063` |
| `explain/` | Grad-CAM real sobre gradientes da última camada convolucional (`gradcam.py`), sem heurísticas de borda | Classe C | `REQ-060` |
| `evaluation/` | Avaliação externa reproduzível com IC de DeLong, subgrupos, proveniência completa e guarda contra vazamento por paciente | Classe B/C | `REQ-050`, `REQ-051` |
| `audit/` | Log append-only encadeado por hash com detecção de adulteração e reconstrução da decisão | Classe C | `REQ-070`, `REQ-071` |
| `api/` | Factory FastAPI (`/api/v1/*`) conforme `docs/OPENAPI_SPEC.yaml`, sem métricas fabricadas e com gate de escopo sobre HTTP | Classe C | `REQ-031`, `REQ-080` |
| `config/` e `cli/` | Configuração tipada (`pydantic-settings`) e CLI (`typer`) | Classe B | `REQ-001` |
| `data/` | Adaptadores de datasets e manifests com verificação de disjunção por paciente (NIH ChestX-ray14, CheXpert) | Classe B | `REQ-051` |

## 3. Fluxo de dados na inferência

`Entrada DICOM -> io/ -> modalities/<mod>.validate() [gate] -> models/ [SHA-256] -> inference/engine.py -> calibration/ -> explain/gradcam.py -> audit/ -> api/ ou cli/`

## 4. Interfaces externas e segregação

- **API HTTP:** especificada em [`docs/OPENAPI_SPEC.yaml`](../OPENAPI_SPEC.yaml); segredos injetados apenas por variáveis de ambiente.
- **SOUP:** dependências travadas em `uv.lock` e inventariadas em [`08-soup-list.md`](08-soup-list.md).
- **Quarentena legada:** código v1 isolado em `legacy/` e excluído da distribuição, do `ruff`, do `mypy` e do `pytest`.

## Histórico de revisões

| Versão | Data | Alteração |
|---|---|---|
| v0.1 | 2026-11 | Criação da Descrição da Arquitetura de Software (Fase 2, IEC 62304 §5.3) |
