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
| Executor de avaliação (AUROC com IC, subgrupos, proveniência) | **funcional, testada** |
| Calibração por temperatura (preserva ordenação, logo AUROC) | **funcional, testada** |
| Política de abstenção em 3 bandas, mais larga para achado crítico | **funcional, testada** |
| Grad-CAM **real** por hooks de gradiente | **funcional, testada** |
| Trilha de auditoria encadeada por hash, adulteração detectável | **funcional, testada** |
| API FastAPI — gate de escopo por HTTP, sem métrica fabricada | **funcional, testada** |
| Laudo estruturado em PDF, PACS, frontend | Fase 4 |
| **Modelo próprio treinado** | **nenhum** |
| **Métricas de desempenho medidas** | **uma** — AUROC macro 0,664 em validação externa (`artifacts/eval/xrv-densenet121-pc__20260912T202549Z/`) |
| **Validação clínica** | **nenhuma** |

## Linha de base honesta — o primeiro número real

Medido em 2026-09-12 no Google Colab (T4), sobre o **split oficial de teste do
NIH ChestX-ray14**: 25.596 imagens, 2.797 pacientes, disjunto por paciente.
Modelo de terceiros `torchxrayvision densenet121-res224-pc`, treinado **só** em
PadChest (Espanha) — validação externa genuína, sem vazamento.

| | |
|---|---|
| **AUROC macro** | **0,664** sobre 14 achados — fonte: `artifacts/eval/xrv-densenet121-pc__20260912T202549Z/metrics.json` |
| Melhores | Hérnia 0,836 (n+=86, IC largo) · Cardiomegalia 0,796 · Derrame 0,752 · Edema 0,745 — `reports/EVALUATION_xrv-densenet121-pc_nih-chestx-ray14_2026-09-12.md` |
| Piores | Fibrose **0,455** (abaixo do acaso: os rótulos "Fibrosis" do PadChest e do NIH não descrevem a mesma coisa) · Espessamento pleural 0,579 · Enfisema 0,591 — `reports/EVALUATION_xrv-densenet121-pc_nih-chestx-ray14_2026-09-12.md` |
| Incidência | PA 0,696 vs AP 0,631 — a lacuna prevista apareceu (`reports/EVALUATION_xrv-densenet121-pc_nih-chestx-ray14_2026-09-12.md`) |
| Calibração | ECE 0,18–0,51 em todos os achados: os escores **não** são probabilidade de doença |
| Proveniência | git `9d240cc`, pesos `a9148ef6…`, manifest `39f31d78…`, seed 20260101, bootstrap 2000 |

Relatório completo, gerado do artefato: [`reports/EVALUATION_xrv-densenet121-pc_nih-chestx-ray14_2026-09-12.md`](reports/EVALUATION_xrv-densenet121-pc_nih-chestx-ray14_2026-09-12.md).

**Sobre o número.** A expectativa registrada no ROADMAP era 0,72–0,82. O medido
é 0,664. A expectativa estava errada; o número fica. É exatamente para isto que
o projeto foi reconstruído: publicar o que foi medido, com intervalo de
confiança, em vez do que se gostaria de ter medido. O README do v1 alegava 0,94.

**O que este número não é:** validação clínica, evidência de utilidade, ou
desempenho do modelo do produto — é um modelo de referência de terceiros, sem
calibração, sobre rótulos minerados por NLP. A especificidade a 90% de
sensibilidade fica entre 0,12 e 0,48: como triagem, esta linha de base **não
serve** — e é isso que uma linha de base honesta deve dizer.

**Reprodução:** [`notebooks/01_baseline_nih_cxr14_colab.ipynb`](notebooks/01_baseline_nih_cxr14_colab.ipynb)
no commit `9d240cc`. O carregador de PNG mudou depois (normalização canônica pelo
fundo de escala, em vez de min-max por imagem); uma nova execução na ponta da
branch produzirá um segundo artefato, e os dois ficam registrados.

**A armadilha que o código evita:** usar `densenet121-res224-all` no NIH seria
*in-distribution* — esses pesos foram treinados no NIH. O `check_leakage`
detecta e recusa chamar isso de validação externa.

## Referência técnica

- **[ROADMAP.md](ROADMAP.md)** — diagnóstico do estado atual, arquitetura-alvo, decisão de framework, fases, trilha regulatória, orçamento e cronograma
- **[HONEST_STATUS.md](HONEST_STATUS.md)** — retratação das alegações anteriores e defeitos conhecidos
- `docs/OPENAPI_SPEC.yaml` — contrato de API alvo
- `docs/CLINICAL_VALIDATION.md` — relatório da v1; é o único documento historicamente honesto do repositório (registra 20% de acurácia em 5 imagens sintéticas)

## Licença e contato

- Repositório: https://github.com/drguilhermecapel/radiologyai
- Autor: Dr. Guilherme Capel Pasqua — CRM-SP 175873
