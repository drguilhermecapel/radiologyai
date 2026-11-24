# RadiologyAI (MedAI Radiologia)

Plataforma de **pesquisa** em interpretação de imagens radiológicas por inteligência artificial,
construída em torno de um **catálogo de condições graduado por evidência**: cada achado sugerido
mostra a força da evidência que o sustenta.

> ## ⚠️ Software de pesquisa — não é dispositivo médico
>
> **Não existe modelo treinado por este projeto. Não houve validação clínica. Não é regularizado
> na ANVISA, nem tem clearance FDA ou marcação CE. Não use para nenhuma decisão sobre um paciente
> real.** Toda saída é um *achado sugerido para confirmação médica*, nunca um diagnóstico.
>
> Existem hoje **duas** medições de desempenho, ambas de um modelo de terceiros num dataset
> público (abaixo). Versões anteriores deste README alegavam acurácia de 88–95% e "validação
> clínica": **essas alegações eram fabricadas e estão retratadas** — ver
> **[HONEST_STATUS.md](HONEST_STATUS.md)**.
>
> O plano está em **[ROADMAP.md](ROADMAP.md)** (v2.0).

---

## A ideia central

O objetivo é detectar e quantificar **o maior número possível de condições, cada uma com a
confiabilidade científica que de fato tem**, em RX, TC, RM, mamografia, ultrassom e
ecocardiografia, usando modelos abertos reais.

| Nível | Significado | Onde aparece |
|---|---|---|
| **A** | validado externamente por medição nossa, sem vazamento, aprovado no protocolo | visão principal |
| **B** | validação publicada revisada por pares ou limiar de diretriz, citação conferida na fonte | visão principal |
| **C** | validado internamente por medição nossa | visão principal, com selo |
| **D** | exploratório (zero-shot ou sem validação por condição) | **só no modo pesquisa** |

O nível é **calculado** a partir de registros de evidência com proveniência (`compute_level` em
`src/radiologyai/catalog/schema.py`), nunca digitado. Os critérios foram pré-registrados antes de
qualquer medição: [`docs/protocolo-evidencia.md`](docs/protocolo-evidencia.md).

## O que existe hoje

O código legado (v1) foi removido da árvore e continua recuperável no histórico git (commit
`16fdd47`). O núcleo v2 fica em `src/radiologyai/`.

| Camada | Estado |
|---|---|
| Leitura DICOM — VOI LUT, MONOCHROME1, RescaleSlope/Intercept, sem quantização | funcional, testada |
| Des-identificação PS3.15 (subconjunto) com pseudo-IDs determinísticos HMAC | funcional, testada |
| Plugin de modalidade + gate de escopo antes do modelo (controle do perigo RISK-003) | funcional para RX; TC/RM/US declarados no sistema de tipos |
| Model card com sha256 validado; inferência falha-fechada | funcional, testada |
| Backend TorchXRayVision (pesos reais, integridade sha256) | funcional, testada |
| Avaliação: AUROC com IC bootstrap, ponto de operação, ECE, subgrupos, vazamento | funcional, testada |
| Calibração por temperatura; abstenção em 3 bandas; decisão por ponto de operação com origem | funcional, testada |
| Grad-CAM **real** por hooks de gradiente | funcional, testada |
| Trilha de auditoria encadeada por hash | funcional, testada |
| API FastAPI — gate de escopo por HTTP, sem métrica fabricada | funcional, testada |
| Catálogo de condições: esquema, regra de nível, protocolo pré-registrado | funcional, testado; **nenhuma condição catalogada ainda** |
| Camada de análise multimodal (`src/radiologyai/analysis/`) | em construção |
| Plataforma web (lista de trabalho, visualizador, laudo preliminar, assinatura) | em construção |
| **Modelo próprio treinado** | **nenhum** |
| **Validação clínica** | **nenhuma** |

Requisitos rastreados até risco e teste: [`docs/regulatory/13-traceability.md`](docs/regulatory/13-traceability.md)
(gerada e verificada no CI).

## As duas medições reais

Modelo de terceiros `torchxrayvision densenet121-res224-pc`, treinado **só** em PadChest
(Espanha), avaliado no **split oficial de teste do NIH ChestX-ray14** (25.596 imagens, 2.797
pacientes, disjunto por paciente) — validação externa ao treino do modelo, em Google Colab (T4),
em 2026-09-12.

| Execução | Normalização | AUROC macro (14 achados) |
|---|---|---|
| [`xrv-densenet121-pc__20260912T202549Z`](reports/EVALUATION_nih-chestx-ray14_xrv-densenet121-pc__20260912T202549Z.md) | min-max por imagem | 0,6644 — `artifacts/eval/xrv-densenet121-pc__20260912T202549Z/metrics.json` |
| [`xrv-densenet121-pc__20260912T215742Z`](reports/EVALUATION_nih-chestx-ray14_xrv-densenet121-pc__20260912T215742Z.md) | fundo de escala (canônica do torchxrayvision) | 0,6636 — `artifacts/eval/xrv-densenet121-pc__20260912T215742Z/metrics.json` |

Destaques do relatório (todos de `artifacts/eval/xrv-densenet121-pc__20260912T215742Z/metrics.json`):

- Melhores achados: hérnia 0,829 (86 positivos, IC largo), cardiomegalia 0,796, derrame 0,752 — fonte `artifacts/eval/xrv-densenet121-pc__20260912T215742Z/metrics.json`.
- Fibrose 0,448, IC95 [0,421, 0,473], abaixo do acaso com IC que exclui 0,5 — fonte `artifacts/eval/xrv-densenet121-pc__20260912T215742Z/metrics.json`. É um resultado sobre rótulos: "Fibrosis" no PadChest e no NIH não nomeiam o mesmo achado.
- Incidência: PA 0,692 vs AP 0,630 — fonte `artifacts/eval/xrv-densenet121-pc__20260912T215742Z/metrics.json`.
- Calibração: ECE entre 0,20 e 0,51; os escores **não** são probabilidade de doença — fonte `artifacts/eval/xrv-densenet121-pc__20260912T215742Z/metrics.json`.
- Especificidade a sensibilidade 0,90 entre 0,11 e 0,48: **como triagem esta linha de base não serve** — tabela em [`reports/EVALUATION_nih-chestx-ray14_xrv-densenet121-pc__20260912T215742Z.md`](reports/EVALUATION_nih-chestx-ray14_xrv-densenet121-pc__20260912T215742Z.md).

O delta entre as duas execuções (−0,0008) mostra que o modelo é robusto a essa diferença de
contraste. O ROADMAP registrava 0,72–0,82 antes de medir (expectativa errada, mantida ao lado do
número); o README do v1 alegava 0,94 sem ter medido nada.

**O que estes números não são:** validação clínica, evidência de utilidade, desempenho em
população brasileira, ou desempenho de um modelo do produto. É um modelo de referência de
terceiros, sem calibração, sobre rótulos minerados por NLP.

**Reprodução:** [`notebooks/01_baseline_nih_cxr14_colab.ipynb`](notebooks/01_baseline_nih_cxr14_colab.ipynb).
**Próxima medição contra rótulo de radiologistas:** CheXpert (validação, 3 radiologistas) —
[`notebooks/02_chexpert_e_credenciamento.ipynb`](notebooks/02_chexpert_e_credenciamento.ipynb) e
[`docs/access/README.md`](docs/access/README.md). Nenhum número do CheXpert foi medido até aqui.

## Documentos

- **[ROADMAP.md](ROADMAP.md)** — v2.0: catálogo graduado por evidência, portões G0–G6,
  amplitude multimodal com modelos abertos, plataforma, trilha regulatória, cronograma
- **[HONEST_STATUS.md](HONEST_STATUS.md)** — retratação das alegações do v1, defeitos do legado,
  pendências abertas
- **[docs/protocolo-evidencia.md](docs/protocolo-evidencia.md)** — protocolo pré-registrado
- **[docs/regulatory/](docs/regulatory/README.md)** — uso pretendido, classificação, risco,
  desenvolvimento, arquitetura, verificação, avaliação clínica, cibersegurança/LGPD, IFU,
  mudanças, vigilância
- **[MODEL_LICENSES.md](MODEL_LICENSES.md)** — licença de cada modelo de terceiros usado ou
  planejado
- **[docs/access/README.md](docs/access/README.md)** — acesso a datasets (CheXpert, PhysioNet)
- **[docs/datasheets/](docs/datasheets/)** — datasheets de datasets
- `docs/OPENAPI_SPEC.yaml` — contrato de API alvo

## Desenvolvimento

```bash
uv sync --extra dev --extra eval --extra api --extra imaging
uv run ruff check src/ tests/ scripts/ && uv run ruff format --check src/ tests/
uv run mypy src/radiologyai
uv run pytest
uv run python scripts/check_honesty.py
uv run python scripts/trace.py --check
```

## Licença e contato

- Repositório: https://github.com/drguilhermecapel/radiologyai
- Autor: Dr. Guilherme Capel Pasqua — CRM-SP 175873
