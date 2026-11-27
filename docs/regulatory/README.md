# Dossiê regulatório

Conjunto documental em construção para a trajetória SaMD/ANVISA descrita em
[`ROADMAP.md`](../../ROADMAP.md) §8. O software é de pesquisa e não é dispositivo médico.

> Nenhum documento aqui alega conformidade. São os artefatos de planejamento e
> desenvolvimento exigidos por IEC 62304, ISO 14971 e RDC 657/2022, sendo construídos
> em paralelo à engenharia — não depois dela.

| # | Documento | Estado |
|---|---|---|
| 01 | [Uso Pretendido / Indicações de Uso](01-intended-use.md) | rascunho v0.2 — pesquisa multimodal, sem alegação clínica |
| 02 | [Classificação de Segurança do Software](02-software-safety-classification.md) | rascunho v0.1 — Classe C |
| 03 | [Plano de Gestão de Risco (ISO 14971)](03-risk-management-plan.md) | rascunho v0.2 — matriz estrita; divergência com a v0.1 de `main` registrada |
| 04 | [Arquivo de Risco / Análise de Perigos](04-risk-file.md) | rascunho vivo v0.3 — RISK-001…RISK-020, FMEA H-01…H-10 com testes verificadores, RISK-006 com REQ-043; nenhum residual aceito |
| 05 | [Plano de Desenvolvimento de Software (62304 §5.1)](05-development-plan.md) | rascunho v0.2 |
| 06 | [Especificação de Requisitos](06-requirements/) | requisitos `REQ-NNN.md`; contagem e verificação na matriz 13 |
| 07 | [Descrição da Arquitetura (62304 §5.3)](07-architecture.md) | rascunho v0.3 — núcleo (incl. ONNX, Platt, trava de regressão), análise, fábrica de evidência e plataforma conferidos contra a árvore |
| 08 | [Lista SOUP](08-soup-list.md) | **gerada** por `scripts/soup.py`, verificada no CI |
| 09 | [Plano de Verificação](09-verification-plan.md) | rascunho v0.3 |
| 10 | [Plano de Avaliação Clínica](10-clinical-evaluation-plan.md) | rascunho v0.2 — critérios = [protocolo pré-registrado](../protocolo-evidencia.md); modelo próprio de RX pré-especificado |
| 11 | [Cibersegurança + LGPD/DPIA](11-cybersecurity-lgpd.md) | rascunho v0.3 — expurgo de `medai_security.db` do histórico pendente |
| 12 | [Rotulagem / IFU (pt-BR)](12-labeling-ifu.md) | rascunho v0.2 — versão de pesquisa; rota legada e trilha de auditoria descritas |
| 13 | [Matriz de Rastreabilidade](13-traceability.md) | **gerada** por `scripts/trace.py`, verificada no CI |
| 14 | [Controle de Mudanças e Resolução de Problemas](14-change-control.md) | rascunho v0.2 |
| 15 | [Plano de Vigilância Pós-Mercado](15-post-market-surveillance.md) | rascunho v0.2 — plano para o portão G6 |
| 16 | Model Cards e Dataset Datasheets | manifestos de modelo em `src/radiologyai/analysis/manifests/` ([índice](../model-cards/README.md)); [`docs/datasheets/`](../datasheets/); inventário de licenças em [`MODEL_LICENSES.md`](../../MODEL_LICENSES.md) |

Documento transversal: **[Protocolo de evidência pré-registrado](../protocolo-evidencia.md)**
(`protocol_id: "2026-11-v1"`), versão legível de `src/radiologyai/catalog/protocol.yaml`.

Relatos de avaliação exploratória em `docs/evaluation/`
([NIH-1000](../evaluation/phase1-nih1000-baseline.md),
[coorte inédita do NIH](../evaluation/phase1-nih-unseen-benchmark.md)): **sem artefato
commitado**, não são evidência — ver [`HONEST_STATUS.md`](../../HONEST_STATUS.md).

**Consolidação de 2026-11-27:** os documentos paralelos criados em `main`
(`05-software-development-plan.md`, `07-software-architecture.md`,
`11-cybersecurity-and-lgpd-dpia.md`, `12-labeling-and-ifu.md`,
`14-change-control-and-problem-resolution.md`, `15-post-market-surveillance-plan.md`) e as versões
v0.1 de 03, 04, 09 e 10 de `main` foram incorporados aos documentos acima e os duplicados
removidos; o conteúdo original continua no histórico git (`origin/main`).

**Ordem de escrita:** 01 e 02 primeiro — eles fixam a classe de risco e a classe de
segurança, das quais tudo o mais deriva. A contagem de requisitos não é escrita aqui à mão:
ela está na matriz gerada (13), que o CI mantém em dia.

## Portões automatizados

Três verificações de CI transformam estes documentos em evidência viva, em vez
de papel que diverge do código:

| Verificação | O que impõe |
|---|---|
| `scripts/trace.py --check` | Todo requisito tem teste verificador; nenhum teste cita requisito inexistente; todo requisito rastreia até um controle de risco |
| `scripts/soup.py --check` | A lista SOUP reflete as dependências declaradas; toda dependência tem propósito documentado |
| `scripts/check_honesty.py` | Nenhuma alegação de desempenho sem artefato de avaliação; nenhum sha256 inválido; nenhum import silencioso; nenhum marcador de simulação |

Uma matriz de rastreabilidade que pode divergir não vale nada. Uma que bloqueia
merge é evidência.
