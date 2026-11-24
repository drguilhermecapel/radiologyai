---
doc_id: REG-07
title: Descrição da Arquitetura de Software (IEC 62304 §5.3)
status: RASCUNHO — v0.1
date: 2026-11
author: Dr. Guilherme Capel Pasqua (CRM-SP 175873)
depends_on: [REG-01, REG-02, REG-05]
---

# Arquitetura de Software

> Software de pesquisa, não é dispositivo médico. Partes marcadas "em construção" estão sendo
> implementadas em paralelo e serão confirmadas contra o código na próxima revisão deste
> documento.

## 1. Visão geral

Três camadas, com dependências sempre de cima para baixo:

```
┌──────────────────────────────────────────────────────────────────────────┐
│ PLATAFORMA  api/ (FastAPI) · persistência · fila de trabalho ·           │
│             laudo (PDF, DICOM SR, FHIR) · DICOMweb/Orthanc · frontend    │  em construção
├──────────────────────────────────────────────────────────────────────────┤
│ ANÁLISE     analysis/: manifestos · registro de modelos · adaptadores ·  │
│             roteamento · pipeline · artefatos · imagem · política        │  em construção
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

## 3. Camada de análise (`src/radiologyai/analysis/`, em construção)

Declarada no pacote: `types` (ImageData, Finding, AnalysisResult), `manifest` (esquema dos
manifestos YAML de modelo), `base` (protocolo `ModelAdapter`), `registry` (manifestos, pesos,
integridade, cache), `pipeline` (`analyze`), `routing` (tipo de exame e roteamento), `artifacts`
(repositório endereçado por conteúdo), `imaging` (DICOM, série, NIfTI, raster, cine → ImageData),
`policy` (calibração e pontos ajustados).

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

## 4. Plataforma (em construção)

| Componente | Responsabilidade | Controle de risco |
|---|---|---|
| API | criação de exames, disparo de análise, consulta de resultado; modelo indisponível ⇒ HTTP 503 | RISK-004 |
| Persistência e fila | estado dos exames, execução serializada de inferência pesada | RISK-007 |
| Laudo | PDF, DICOM SR (TID 1500), FHIR `DiagnosticReport` **preliminar** | RISK-001, RISK-002 |
| Revisão e assinatura | aceitar/editar/rejeitar por achado; assinatura vinculada ao hash do conteúdo | RISK-018 |
| Controle de acesso | papéis pesquisador, revisor, administrador | RISK-020 |
| Interoperabilidade | DICOMweb (QIDO/WADO/STOW) via Orthanc; saída alinhada ao IHE AI Results | — |
| Frontend | lista de trabalho, visualizador com sobreposições reais, modo pesquisa separado | RISK-008, RISK-012 |

## 5. Segregação de SOUP (Classe C, §5.3.3–5.3.6)

- O núcleo depende só de numpy, pydicom, pydantic, pydantic-settings, PyYAML e typer.
- Dependências pesadas estão em extras opcionais (`imaging`, `eval`, `ml`, `cpu`, `volumetric`,
  `foundation`, `api`) e são importadas dentro das funções que as usam. Ausência ⇒ erro que
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
| API REST | entrada/saída | contrato em `docs/OPENAPI_SPEC.yaml` (alvo) |
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
