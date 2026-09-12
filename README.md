# RadiologyAI (MedAI Radiologia)

Plataforma de pesquisa em interpretação de imagens radiológicas por inteligência artificial.

> ## ⚠️ Software de pesquisa — não é dispositivo médico
>
> **Não existe nenhum modelo treinado neste repositório. Nenhuma métrica de desempenho foi medida. Não houve validação clínica. Não use para nenhuma decisão sobre um paciente real.**
>
> Versões anteriores deste README alegavam acurácia de 88–95% e "validação clínica". **Essas alegações eram fabricadas e estão retratadas.** Ver **[HONEST_STATUS.md](HONEST_STATUS.md)** para a retratação completa e a lista de defeitos conhecidos do código legado.
>
> O plano para transformar isto em algo real está em **[ROADMAP.md](ROADMAP.md)**.

---

## O que existe hoje

O núcleo v2 (`src/radiologyai/`) está em construção. O código v1 foi arquivado
em `legacy/` — está preservado, mas nada o importa e ele é excluído de build,
lint, tipos e CI.

| Camada | Estado |
|---|---|
| Leitura DICOM — VOI LUT, MONOCHROME1, RescaleSlope/Intercept, sem quantização | **funcional, testada** |
| Des-identificação PS3.15 (subconjunto) com pseudo-IDs determinísticos HMAC | **funcional, testada** |
| Janelamento por modalidade (presets de TC em HU) | **funcional, testada** |
| Plugin de modalidade + gate de escopo (o controle de risco H-03) | **funcional, testada** |
| Modalidades TC / RM / US | **declaradas**, não implementadas — levantam `NotImplementedError` |
| Model card com sha256 validado e verificação de integridade | **funcional, testada** |
| Motor de inferência falha-fechada | **funcional** — sem backend de pesos ainda |
| Métricas (AUROC com IC bootstrap, ponto de operação, ECE, vazamento) | **funcional, testada** |
| CLI (`radiologyai version / selftest / modalities / inspect / cards`) | **funcional** |
| Backend torchxrayvision (pesos reais, integridade sha256) | **funcional, testada** |
| Manifest do NIH ChestX-ray14 + detecção de vazamento | **funcional, testada** |
| Carregador do CheXpert (validação, rótulos por 3 radiologistas) | **funcional, testada** — ver [`docs/access/`](docs/access/README.md) |
| Executor de avaliação (AUROC com IC, subgrupos, proveniência) | **funcional, testada** |
| Calibração por temperatura (preserva ordenação, logo AUROC) | **funcional, testada** |
| Política de abstenção em 3 bandas, mais larga para achado crítico | **funcional, testada** |
| Grad-CAM **real** por hooks de gradiente | **funcional, testada** |
| Trilha de auditoria encadeada por hash, adulteração detectável | **funcional, testada** |
| API FastAPI — gate de escopo por HTTP, sem métrica fabricada | **funcional, testada** |
| Laudo estruturado em PDF, PACS, frontend | Fase 4 |
| **Modelo próprio treinado** | **nenhum** |
| **Métricas de desempenho medidas** | **duas** — AUROC macro 0,664 em validação externa; a segunda isola o pré-processamento |
| **Validação clínica** | **nenhuma** |

## Linha de base honesta — o primeiro número real

Medido no Google Colab (T4) em 2026-09-12, sobre o **split oficial de teste do
NIH ChestX-ray14**: 25.596 imagens, 2.797 pacientes, disjunto por paciente.
Modelo de terceiros `torchxrayvision densenet121-res224-pc`, treinado **só** em
PadChest (Espanha) — validação externa genuína, sem vazamento.

| | |
|---|---|
| **AUROC macro** | **0,664** sobre 14 achados — `artifacts/eval/xrv-densenet121-pc__20260912T215742Z/metrics.json` |
| Melhores | Hérnia 0,829 (n+=86, IC largo) · Cardiomegalia 0,796 · Derrame 0,752 · Edema 0,744 |
| Piores | Fibrose **0,448**, IC95 [0,421, 0,473] — abaixo do acaso com IC que exclui 0,5 |
| Incidência | PA 0,692 vs AP 0,630 — a lacuna prevista apareceu |
| Calibração | ECE 0,18–0,51: os escores **não** são probabilidade de doença |
| Ponto de operação | Especificidade a 90% de sensibilidade entre 0,12 e 0,48 — tabela completa em [`reports/EVALUATION_nih-chestx-ray14_xrv-densenet121-pc__20260912T215742Z.md`](reports/EVALUATION_nih-chestx-ray14_xrv-densenet121-pc__20260912T215742Z.md) |

Relatório completo: [`reports/EVALUATION_nih-chestx-ray14_xrv-densenet121-pc__20260912T215742Z.md`](reports/EVALUATION_nih-chestx-ray14_xrv-densenet121-pc__20260912T215742Z.md).

**Fibrose abaixo do acaso** não diz que o modelo erra fibrose: diz que o rótulo
"Fibrosis" do PadChest e o do NIH não nomeiam o mesmo achado. É um resultado
sobre rótulos, não sobre o modelo.

### Duas medições isolam o efeito do pré-processamento

O mesmo modelo, o mesmo split (manifest `39f31d78…`), variando só o carregador
de imagem:

| Run | Normalização | AUROC macro |
|---|---|---|
| [`xrv-densenet121-pc__20260912T202549Z`](reports/EVALUATION_nih-chestx-ray14_xrv-densenet121-pc__20260912T202549Z.md) | min-max por imagem | 0,6644 |
| [`xrv-densenet121-pc__20260912T215742Z`](reports/EVALUATION_nih-chestx-ray14_xrv-densenet121-pc__20260912T215742Z.md) | fundo de escala (canônica do torchxrayvision) | 0,6636 |

**Delta: −0,0008.** Nenhum achado mudou mais que 0,01; todos os IC95 se
sobrepõem. Eu previa melhora ao alinhar com o pré-processamento de treino —
errei: o modelo é robusto a essa diferença de contraste. A hipótese mais
provável, **não testada**, é que radiografias do NIH já ocupam quase todo o
intervalo dinâmico, tornando as duas normalizações quase equivalentes.

Os dois artefatos ficam versionados. Medir a diferença custou uma execução e
substituiu uma suposição por um número.

**Sobre a expectativa.** O ROADMAP registrava 0,72–0,82 antes da medição. O
medido é 0,664, e a previsão errada fica ao lado do número. O README do v1
alegava 0,94 sem ter medido nada.

**O que estes números não são:** validação clínica, evidência de utilidade, ou
desempenho do modelo do produto. É um modelo de referência de terceiros, sem
calibração, sobre rótulos minerados por NLP. Pela especificidade a 90% de
sensibilidade medida em [`reports/EVALUATION_nih-chestx-ray14_xrv-densenet121-pc__20260912T215742Z.md`](reports/EVALUATION_nih-chestx-ray14_xrv-densenet121-pc__20260912T215742Z.md), **como triagem esta linha de base
não serve** — e uma linha de base honesta tem que poder dizer isso.

**Reprodução:** [`notebooks/01_baseline_nih_cxr14_colab.ipynb`](notebooks/01_baseline_nih_cxr14_colab.ipynb).
O apêndice do notebook roda só a inferência, sem rebaixar o dataset.

## Referência técnica

- **[ROADMAP.md](ROADMAP.md)** — diagnóstico do estado atual, arquitetura-alvo, decisão de framework, fases, trilha regulatória, orçamento e cronograma
- **[HONEST_STATUS.md](HONEST_STATUS.md)** — retratação das alegações anteriores e defeitos conhecidos
- `docs/OPENAPI_SPEC.yaml` — contrato de API alvo
- `docs/CLINICAL_VALIDATION.md` — relatório da v1; é o único documento historicamente honesto do repositório (registra 20% de acurácia em 5 imagens sintéticas)

## Licença e contato

- Repositório: https://github.com/drguilhermecapel/radiologyai
- Autor: Dr. Guilherme Capel Pasqua — CRM-SP 175873
