# RadiologyAI (MedAI Radiologia)

Plataforma de **pesquisa** em interpretação de imagens radiológicas por inteligência artificial,
construída em torno de um **catálogo de condições graduado por evidência**: cada achado sugerido
mostra a força da evidência que o sustenta.

> ## ⚠️ Software de pesquisa — não é dispositivo médico
>
> **Nenhum modelo de ponta a ponta foi treinado por este projeto** (existem apenas sondas lineares
> ajustadas sobre codificadores congelados de terceiros). **Não houve validação clínica.** Não é
> regularizado na ANVISA, nem tem clearance FDA ou marcação CE. **Não use para nenhuma decisão
> sobre um paciente real.** Toda saída é um *achado sugerido para confirmação médica*, nunca um
> diagnóstico.
>
> **Duas condições têm nível A** (cardiomegalia, no NIH ChestX-ray14 e no Open-i, e opacidade
> pulmonar, no subconjunto adjudicado do NIH e **reprovada** no Open-i; modelos TorchXRayVision de
> terceiros, validação externa retrospectiva); **nenhuma tem nível B** (o B de derrame pericárdico
> foi retratado: o artigo avalia outro modelo, não os pesos distribuídos). Nível A aqui não é validação clínica. As medições são retrospectivas, em dados
> públicos, e estão listadas com o caminho de cada artefato em
> **[HONEST_STATUS.md](HONEST_STATUS.md)**, inclusive as reprovações e o nível A retratado.
> Versões anteriores deste README alegavam acurácia e "validação clínica" que eram **fabricadas
> e estão retratadas** (ver HONEST_STATUS.md).
>
> O plano está em **[ROADMAP.md](ROADMAP.md)** (v2.0).

---

## A ideia central

O objetivo é detectar e quantificar **o maior número possível de condições, cada uma com a
confiabilidade científica que de fato tem**, em RX, TC, RM, mamografia, ultrassom e
ecocardiografia, usando modelos abertos reais.

| Nível | Significado | Onde aparece |
|---|---|---|
| **A** | medido por este projeto em dado externo, sem sobreposição conhecida ou plausível com o treino nem com a seleção do modelo, e aprovado no protocolo | visão principal |
| **B** | desempenho publicado em estudo revisado por pares, ou limiar de diretriz, com citação conferida na fonte | visão principal |
| **C** | medido por este projeto e aprovado no protocolo, mas em hold-out interno, em dado externo com sobreposição conhecida ou possível com o treino ou a seleção, ou com entrada derivada do padrão de referência | visão principal, com selo e ressalvas |
| **D** | exploratório: sem medição aprovada, refutado por medição própria, ou limitado (modelo não distribuído, termo de uso só de pesquisa) | **só no modo pesquisa**; nunca no laudo |

O nível é **calculado** a partir de registros de evidência com proveniência (`compute_level` em
`src/radiologyai/catalog/schema.py`), nunca digitado. Os critérios foram pré-registrados antes de
qualquer medição: [`docs/protocolo-evidencia.md`](docs/protocolo-evidencia.md). O texto oficial de
cada nível é `LEVEL_MEANING_PT` em `src/radiologyai/catalog/schema.py`.

### O KPI hoje

Transcrito de `src/radiologyai/catalog/levels.json`, contando só modelos distribuídos (o CI falha
se divergir):

<!-- kpi-levels:inicio (gerado de src/radiologyai/catalog/levels.json) -->
| Modalidade (prefixo) | A | B | C | D | Refutadas |
|---|---:|---:|---:|---:|---:|
| `ct` | 0 | 0 | 0 | 1 | 0 |
| `cxr` | 2 | 0 | 2 | 15 | 15 |
| `echo` | 0 | 0 | 0 | 1 | 0 |
| `mg` | 0 | 0 | 0 | 2 | 1 |
| `mr` | 0 | 0 | 0 | 2 | 0 |
| `msk_xr` | 0 | 0 | 0 | 4 | 1 |
| `us` | 0 | 0 | 0 | 3 | 0 |
| **Todas** | 2 | 0 | 2 | 28 | 17 |
<!-- kpi-levels:fim -->

Os registros medidos foram gerados a partir de um commit, sem código sujo. Resultado: **A** para
cardiomegalia (`evidence/measured/nih-cxr14/xrv-densenet121-res224-pc/cardiomegaly.json`, AUROC
0,813 [0,783; 0,839], rótulo NLP; limítrofe no NIH, 0,796 no teste oficial completo; confirmada no
Open-i, `evidence/measured/openi/xrv-densenet121-res224-pc/cardiomegaly.json`, 0,874 [0,840;
0,905] — ver HONEST_STATUS) e opacidade pulmonar
(`evidence/measured/nih-cxr14-google/xrv-densenet121-res224-chex/lung_opacity.json`, AUROC 0,872
[0,848; 0,894], rótulo adjudicado); **C** para tuberculose (`xr-probe-tb`, hold-out interno);
fratura, câncer de mama em mamografia e 13 condições de tórax **refutadas** em todos os modelos
medidos (a coluna "Refutadas" do KPI conta 14 em `cxr` porque inclui a cardiomegalia, A com `pc` e
refutada com `chex` e `mimic_*`)
(`evidence/measured/grazpedwri-dx-sample/xr-probe-fracture/fracture.json`,
`evidence/measured/cmmd/mammoscreen/malignancy.json`). Nenhuma condição em B (o B de derrame pericárdico foi retratado; ver HONEST_STATUS). Detalhe em [HONEST_STATUS.md](HONEST_STATUS.md).

## O que existe hoje

O código legado (v1) foi removido da árvore e continua recuperável no histórico git (commit
`16fdd47`). O núcleo v2 fica em `src/radiologyai/`.

| Camada | Estado |
|---|---|
| Leitura DICOM — VOI LUT, MONOCHROME1, RescaleSlope/Intercept, sem quantização | funcional, testada |
| Des-identificação PS3.15 (subconjunto) com pseudo-IDs determinísticos HMAC | funcional, testada |
| Plugin de modalidade + gate de escopo antes do modelo (controle do perigo RISK-003) | funcional e testado para RX; plugins de TC/RM, mamografia e US escritos (REQ-137, REQ-144, REQ-149), verificação em andamento |
| Model card com sha256 validado; inferência falha-fechada | funcional, testada |
| Backend TorchXRayVision (pesos reais, integridade sha256) | funcional, testada |
| Avaliação: AUROC com IC bootstrap, ponto de operação, ECE, subgrupos, vazamento | funcional, testada |
| Calibração por temperatura e por Platt; ECE e diagrama de confiabilidade; abstenção em 3 bandas; decisão por ponto de operação com origem | funcional, testada; nenhuma calibração sobre dado real commitada como artefato |
| Backend ONNX Runtime (sha256 verificado) e trava de regressão (`radiologyai check-regression`) | funcional, testado; ainda sem execução sobre dado real |
| Grad-CAM **real** por hooks de gradiente | funcional, testada |
| Trilha de auditoria encadeada por hash | funcional, testada |
| API FastAPI — gate de escopo por HTTP, sem métrica fabricada | funcional, testada |
| Catálogo de condições: esquema, regra de nível, protocolo pré-registrado | funcional, testado; 63 condições em `conditions.yaml`, níveis em `levels.json` (KPI acima) |
| Camada de análise multimodal (`src/radiologyai/analysis/`) — manifestos por modelo, adaptadores RX/TC/RM/MG/US/eco, biomarcadores de TC com pré-condições, sondas lineares | código presente, verificação em andamento; inventário de modelos em [`MODEL_LICENSES.md`](MODEL_LICENSES.md) |
| Fábrica de evidência (`src/radiologyai/evidence/`) — predições → medição → registro com proveniência → `levels.json` | 98 registros em `evidence/measured/` (aprovados, reprovados e não elegíveis; desfecho de cada item em `evidence/measured/ledger.yaml`), medidos a partir de commit limpo |
| Plataforma web (API, banco, fila, laudo preliminar PDF/DICOM SR/FHIR, assinatura vinculada ao hash, RBAC, DICOMweb, frontend) | código presente, verificação em andamento |
| **Modelo próprio treinado** | **nenhum de ponta a ponta**; sondas lineares sobre RAD-DINO e BiomedCLIP congelados (TB, fratura, achados de punho, lesão mamária e hepática em US, esteatose), medidas sob o protocolo — nenhuma em medição externa limpa aprovada; fratura reprovada no GRAZPEDWRI-DX (ver HONEST_STATUS.md) |
| **Validação clínica** | **nenhuma** |

Requisitos rastreados até risco e teste: [`docs/regulatory/13-traceability.md`](docs/regulatory/13-traceability.md)
(gerada e verificada no CI).

## Linha de base de terceiros (NIH ChestX-ray14)

Anterior à fábrica de evidência e fora do cálculo de nível. Modelo de terceiros `torchxrayvision densenet121-res224-pc`, treinado **só** em PadChest
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

Relatos de execuções em Colab de 2026-11-25 ([NIH-1000](docs/evaluation/phase1-nih1000-baseline.md),
[coorte inédita do NIH](docs/evaluation/phase1-nih-unseen-benchmark.md)) **não têm artefato
commitado** e não contam como medição — ver [HONEST_STATUS.md](HONEST_STATUS.md).

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
- **[docs/evaluation/](docs/evaluation/phase1-nih1000-baseline.md)** — relatos exploratórios de
  Colab, sem artefato commitado (não verificáveis)
- **[docs/datasheets/](docs/datasheets/README.md)** — datasheets de datasets
- **[docs/model-cards/](docs/model-cards/README.md)** — onde estão os cartões de modelo (manifestos)
- `docs/OPENAPI_SPEC.yaml` — contrato OpenAPI gerado pela aplicação e versionado

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
