---
doc_id: REG-07
title: Descrição da Arquitetura de Software (IEC 62304 §5.3)
status: RASCUNHO — v0.2
date: 2026-11
author: Dr. Guilherme Capel Pasqua (CRM-SP 175873)
depends_on: [REG-01, REG-02, REG-05]
---

# Arquitetura de Software

> Software de pesquisa, não é dispositivo médico. Esta revisão (v0.2) foi conferida contra a
> árvore em 2026-11-25. A camada de análise e a plataforma existem no código, mas sua
> verificação ainda está em andamento: o estado de cada requisito é o da
> [matriz de rastreabilidade](13-traceability.md), não o deste texto.

## 1. Visão geral

Três camadas, com dependências sempre de cima para baixo:

```
┌──────────────────────────────────────────────────────────────────────────┐
│ PLATAFORMA  api/ (FastAPI) · persistence/ · worker/ · reporting/ (PDF,   │
│             DICOM SR/SEG, FHIR) · interop/ (DICOMweb/Orthanc) · frontend/│  em verificação
├──────────────────────────────────────────────────────────────────────────┤
│ ANÁLISE     analysis/: manifestos · registro · adaptadores · roteamento ·│
│             pipeline · artefatos · imagem · política · biomarcadores ·   │  em verificação
│             sondas · código vendorizado                                  │
│ EVIDÊNCIA   evidence/: predições → medição → registro com proveniência → │
│             levels.json (lê catalog/ e evaluation/)                      │  em verificação
├──────────────────────────────────────────────────────────────────────────┤
│ NÚCLEO      io/ · modalities/ · models/ · inference/ · calibration/ ·    │
│ (sem torch) catalog/ · evaluation/ · audit/ · explain/ · config/ ·       │  implementado
│             data/ · cli/ · errors.py                                     │
└──────────────────────────────────────────────────────────────────────────┘
          SOUP pesado (torch, MONAI, TotalSegmentator, timm, transformers…)
          só por extras opcionais, importado sob demanda
```

## 2. Itens de software do núcleo (implementados)

| Item | Responsabilidade | Requisitos |
|---|---|---|
| `io/reader.py`, `io/windowing.py`, `io/metadata.py` | leitura DICOM sem quantização: Modality LUT, VOI LUT, MONOCHROME1; metadados tipados | REQ-005, REQ-006 |
| `io/deident.py` | des-identificação (subconjunto PS3.15) com pseudo-ID HMAC determinístico | REQ-010, REQ-011 |
| `modalities/` (`base.py`, `registry.py`, `xr/`, `ct/`, `mr/`, `us/`) | plugin por modalidade descoberto por *entry point*; `validate()` rejeita entrada fora do escopo antes do modelo | REQ-031, REQ-042 |
| `models/card.py`, `models/registry.py`, `models/backends/` | model card com sha256 validado; backend TorchXRayVision | REQ-020, REQ-021 |
| `inference/engine.py`, `inference/types.py` | inferência falha-fechada; tipos de resultado congelados; nunca afirma normalidade | REQ-030, REQ-063 |
| `calibration/temperature.py`, `abstention.py`, `decision.py` | calibração que preserva ordenação; abstenção em três bandas; decisão só com ponto de operação de origem declarada | REQ-061, REQ-062, REQ-092 |
| `catalog/` (`schema.py`, `catalog.py`, `conditions.yaml`, `protocol.yaml`) | catálogo de condições; registros de evidência; `compute_level`; protocolo pré-registrado | REQ-090, REQ-091, REQ-093 |
| `evaluation/` | AUROC com IC bootstrap, ponto de operação, ECE, subgrupos, vazamento, executor e artefatos | REQ-050, REQ-051 |
| `audit/trail.py` | trilha append-only encadeada por hash | REQ-070, REQ-071 |
| `explain/gradcam.py` | Grad-CAM por *hooks* de gradiente no modelo real | REQ-060 |
| `api/` | FastAPI: gate de escopo por HTTP, sem métrica fabricada | REQ-080 |
| `data/` | manifests NIH ChestX-ray14 e carregador CheXpert | REQ-051 |
| `config/`, `errors.py`, `cli/` | configuração por ambiente; hierarquia de exceções `RadiologyAIError`; CLI | REQ-001 |

## 3. Camada de análise (`src/radiologyai/analysis/`)

| Item | Responsabilidade | Requisitos |
|---|---|---|
| `types.py`, `manifest.py`, `base.py` | tipos de resultado (ImageData, Finding, AnalysisResult); esquema dos manifestos YAML de modelo (licença, `commercial_use`, sem métricas); protocolo `ModelAdapter` | REQ-105 |
| `registry.py`, `manifests/*.yaml` | um manifesto por modelo (`available` ou `planned` com motivo); pesos só após sha256 | REQ-100, REQ-139 |
| `routing.py`, `routing.yaml` | tipo de exame por metadado, conservador; desconhecido ⇒ decisão humana | REQ-104, REQ-161 |
| `imaging.py` | DICOM, série, NIfTI, raster e cine → ImageData em escala física | REQ-106 |
| `pipeline.py`, `policy.py` | `analyze`: qualidade → adaptador → calibração → ponto de operação com origem → decisão → nível do catálogo | REQ-101, REQ-102, REQ-103 |
| `artifacts.py` | repositório de artefatos endereçado por conteúdo | REQ-107 |
| `adapters/xr_*.py` | TorchXRayVision (classificadores, gates de incidência e OOD, PSPNet/ICT), RAD-DINO, CheXzero | REQ-120…REQ-127 |
| `adapters/ct_*.py`, `biomarkers/` | TotalSegmentator em subprocesso offline; biomarcadores em HU com pré-condições de aquisição | REQ-130…REQ-138 |
| `adapters/mg_*.py`, `vendor/mammoscreen/` | mammoscreen vendorizado sem código remoto; densidade MONAI | REQ-140…REQ-144 |
| `adapters/echo_*.py`, `adapters/us_*.py`, `vendor/sam2/` | MedSAM2 (traçado assistido do VE) + Simpson; MedSAM em US; sondas de US | REQ-145…REQ-149 |
| `probes/` | ajuste determinístico de sondas lineares sobre *embeddings* congelados; artefato `safetensors` + JSON | REQ-125, REQ-148 |

## 3a. Fábrica de evidência (`src/radiologyai/evidence/`)

| Item | Responsabilidade | Requisitos |
|---|---|---|
| `predictions.py` | lê o parquet de predições produzido pela inferência (interface inferência → medição) | REQ-117 |
| `measure.py` | métricas, IC por bootstrap de paciente, metade de calibração/teste, subgrupos → `EvidenceRecord` | REQ-110…REQ-113 |
| `leakage.py` | tabela explícita de linhagem treino → dataset de avaliação | REQ-114 |
| `provenance.py` | git sha, sha256 dos pesos e comando, lidos — nunca digitados | REQ-115 |
| `levels.py`, `store.py` | geração determinística de `catalog/levels.json`; leitura/escrita de `evidence/` | REQ-116 |

Fluxo de uma análise:

```
entrada → imaging (decodifica, sem adivinhar) → routing (tipo de exame por metadado;
desconhecido ⇒ ExamTypeUnknownError, um humano escolhe) → gate de escopo do plugin →
para cada modelo roteado: registry (manifesto + pesos com sha256; planejado ou ausente ⇒
ModelUnavailableError) → adapter (escores, máscaras, medidas BRUTOS) → policy (calibração) →
ponto de operação com origem → decisão → nível de evidência do catálogo → Finding
```

Regras de separação:

1. **O adaptador não decide.** Devolve escores, mapas de rótulo e medidas; nunca limiar,
   calibração ou veredito. Decisão é exclusiva do pipeline.
2. **Todo rótulo emitido está declarado no manifesto** e mapeado para uma condição do catálogo.
3. **Pesos só entram pelo registro**, verificados contra o sha256 fixado no manifesto; sem acesso
   à rede durante a inferência.
4. **O manifesto declara licença** de código e de pesos e `commercial_use`, com a fonte.
5. **Medida quantitativa de TC/RM só é emitida com as pré-condições de aquisição satisfeitas**;
   senão, abstenção com motivo (RISK-011).
6. Um modelo que não pode rodar de verdade tem `status: planned` no manifesto. Não há stub.

## 4. Plataforma

| Componente | Código | Responsabilidade | Controle de risco | Requisitos |
|---|---|---|---|---|
| API | `api/` (`app.py`, `routers/`, `schemas.py`) | exames, análises, resultados, laudos; respostas tipadas; contrato OpenAPI gerado e versionado; prontidão real | RISK-004 | REQ-160, REQ-164, REQ-165 |
| Autenticação e RBAC | `persistence/security.py`, `persistence/users.py`, `api/deps.py` | argon2id, sessão opaca revogável, papel imposto por rota | RISK-020 | REQ-150, REQ-151 |
| Persistência | `persistence/db.py`, `models.py`, `migrations/` | banco relacional; evolução só por migração Alembic | RISK-007 | REQ-166 |
| Auditoria | `persistence/audit.py` | trilha no banco, append-only, encadeada por hash | RISK-007, RISK-020 | REQ-155 |
| Fila e worker | `worker/` | fila no banco, concorrência 1, resultado imutável, falha com motivo | RISK-004, RISK-007 | REQ-154 |
| Ingestão | `interop/ingest.py` | DICOM/zip, NIfTI, PNG/JPEG com limites; des-identificação na entrada | RISK-003, RISK-009 | REQ-152, REQ-153 |
| DICOMweb | `interop/dicomweb.py` | QIDO/WADO/STOW (Orthanc), importação des-identificada | RISK-009 | REQ-162 |
| Visualização | `interop/images.py`, `frontend/src/viewer/` | quadros decodificados no servidor sem perda de faixa; sobreposições só de máscaras/mapas reais | RISK-005, RISK-008 | REQ-163 |
| Laudo | `reporting/` | conteúdo canônico com hash; PDF pt-BR, DICOM SR TID 1500 + SEG, FHIR `DiagnosticReport` **preliminares** | RISK-002, RISK-018 | REQ-157, REQ-158, REQ-159 |
| Revisão e assinatura | `reporting/service.py` | assinatura vinculada ao hash do conteúdo analisado e exibido | RISK-018 | REQ-156 |
| Frontend | `frontend/` (React + TypeScript) | lista de trabalho, visualizador, revisão, modo pesquisa separado | RISK-008, RISK-012 | REQ-157 |

## 5. Segregação de SOUP (Classe C, §5.3.3–5.3.6)

- O núcleo depende só de numpy, pydicom, pydantic, pydantic-settings, PyYAML e typer.
- Dependências pesadas estão em extras opcionais (`imaging`, `eval`, `ml`, `cpu`, `volumetric`,
  `foundation`, `mammography`, `api`, `platform`) e são importadas dentro das funções que as usam. Ausência ⇒ erro que
  nomeia o extra, nunca degradação.
- Código GPL/AGPL não é ligado ao pacote (ex.: `pylibjpeg-libjpeg`, GPL-3, foi deixado de fora
  do extra `imaging`).
- A lista SOUP completa, com versão instalada e relevância para segurança, é gerada de
  `uv.lock`: [`08-soup-list.md`](08-soup-list.md).
- Modelos de terceiros (pesos) são SOUP também: inventário em
  [`MODEL_LICENSES.md`](../../MODEL_LICENSES.md), integridade por sha256.

## 6. Interfaces externas

| Interface | Direção | Formato |
|---|---|---|
| Arquivos DICOM, NIfTI, imagens raster, cine | entrada | ver `analysis/imaging` |
| DICOMweb (Orthanc) | entrada/saída | PS3.18 |
| API REST | entrada/saída | contrato gerado pela aplicação (`api/openapi.py`) e versionado em `docs/OPENAPI_SPEC.yaml` (REQ-164) |
| Laudo | saída | PDF; DICOM SR TID 1500; FHIR R4 `DiagnosticReport` + `Observation` |
| Artefatos de avaliação | saída | `artifacts/eval/<run_id>/metrics.json` |

## 7. Propriedades transversais

- **Falha fechada:** dependência ausente, pesos ausentes ou hash divergente ⇒ exceção explícita.
- **Determinismo de avaliação:** semente fixa; mesma semente ⇒ mesmos números (REQ-050).
- **Auditabilidade:** todo resultado registra entrada (hash), modelo, pesos, git sha, saída e
  decisão do médico (REQ-070, REQ-071).
- **Privacidade:** des-identificação na ingestão; nenhum identificador, nome de arquivo ou texto
  livre DICOM em log ou erro.

---

## Histórico de revisões

| Versão | Data | Alteração |
|---|---|---|
| v0.1 | 2026-11 | Criação: núcleo implementado; camadas de análise e plataforma em construção |
| v0.2 | 2026-11-25 | Conferido contra a árvore: itens da análise, fábrica de evidência e plataforma com requisitos |
