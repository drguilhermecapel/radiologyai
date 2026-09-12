# Roadmap — RadiologyAI

**De protótipo com métricas fabricadas a plataforma multimodal com trajetória SaMD/ANVISA.**

Documento de planejamento. Versão 1.0 — setembro/2026.
Autor do projeto: Dr. Guilherme Capel Pasqua (CRM-SP 175873).

---

## 0. Por que este documento existe

Este repositório **aparenta** ser uma plataforma clínica completa: 54 mil linhas de Python, 67 módulos, PACS/HL7/FHIR, *federated learning*, *global deployment orchestration*, *regulatory compliance*. O `README.md` afirma acurácia de 88–95% e "arquiteturas validadas clinicamente".

Nada disso é verdade. O levantamento abaixo é factual e verificável comando a comando.

Este roadmap parte desse diagnóstico e desenha o caminho até o objetivo declarado: **SaMD registrável na ANVISA, escopo multimodal (RX, TC, RM, US)**, com núcleo v2 limpo e legado arquivado.

---

## 1. Diagnóstico do estado atual

### 1.1 Não existe nenhum modelo treinado

```bash
find . -name "*.h5" -o -name "*.keras" -o -name "*.pt" -o -name "*.pth" \
       -o -name "*.onnx" -o -name "*.pb" -o -name "*.tflite"
# → vazio
```

`models/model_registry.json` declara três modelos com `accuracy: 0.92`, `auc: 0.94`, URLs de release inexistentes (o repositório tem **zero releases publicados**) e hashes fabricados:

```
"sha256_hash": "a3b4c5d6e7f8g9h0i1j2k3l4m5n6o7p8q9r0s1t2u3v4w5x6y7z8a9b0c1d2e3f4"
```

Contém `g`, `h`, `i` — não é hexadecimal. É placeholder.

### 1.2 O caminho real de predição é uma heurística OpenCV

Sem pesos, `medai_inference_system.py:252` devolve `_create_dummy_model()` e a inferência cai em `_analyze_image_fallback()` (linha 1510):

```python
fracture_score = min(0.3, pneumonia_score * 0.5)          # Simple heuristic
tumor_score    = min(0.3, pleural_effusion_score * 0.4)   # Simple heuristic
```

Com `print(f"DEBUG Pathology CLAHE input shape: ...")` no caminho de produção.

### 1.3 A explicabilidade é falsa — e este é o defeito mais grave do repositório

`src/medai_explainability.py:56` — `GradCAMExplainer.generate_heatmap()` **nunca referencia `self.model`**. Chama `_simulate_attention_map()`:

```python
edges = cv2.Canny(gray, 50, 150)
heatmap = cv2.GaussianBlur(edges.astype(np.float32), (15, 15), 0)
```

Detecção de bordas Canny, borrada, com ruído gaussiano somado. `IntegratedGradientsExplainer` faz o equivalente em `_simulate_integrated_gradients()` (linha 194).

Isso é servido a quem chama `POST /api/v1/explain` e `POST /api/v1/analyze?include_explanation=true`.

**Um mapa de saliência fabricado sobreposto à radiografia real de um paciente é pior que um número de acurácia fabricado**, porque manufatura a *aparência* de um modelo raciocinando sobre anatomia. Um clínico que vê o "calor" sobre uma região vai atribuir significado a um contorno de Canny.

### 1.4 A API fabrica métricas de validação clínica a cada requisição

`src/medai_fastapi_server.py:340`:

```python
y_true = np.array([1])  # Mock ground truth
...
clinical_metrics = clinical_evaluator.evaluate_model_performance(y_true, y_pred)
```

O resultado vai no corpo da resposta como `clinical_metrics`. A API informa a **todo** cliente que o modelo foi clinicamente validado, calculando isso a partir de um array de um elemento inventado.

### 1.5 O mapeamento de patologias do único caminho com pesos reais é clinicamente perigoso

`src/torchxray_integration.py` é o único ponto do repositório que carrega pesos de verdade (`torchxrayvision densenet121-res224-all`). Ele colapsa as 18 saídas do modelo em 5 categorias:

| Saída do modelo | Reportado como |
|---|---|
| **Pneumothorax** | **pneumonia** |
| **Cardiomegaly** | **normal** |
| **Edema** | **normal** |
| Atelectasis | normal |
| Emphysema | normal |
| Fibrosis | normal |
| Pleural_Thickening | pleural_effusion |

Um pneumotórax reportado como pneumonia. Cardiomegalia com edema reportados como "Normal". O modelo detectou corretamente e a camada de tradução destruiu o achado.

### 1.6 "Validação clínica" gerada por `np.random`

164 chamadas a `np.random` em `src/` (236 no repositório), concentradas nos módulos de validação e monitoramento. O caso central, `src/medai_advanced_clinical_validation.py:357`:

```python
cv_scores = np.random.normal(base_accuracy, 0.02, n_folds)
```

Cross-validation **sintetizada**. `medai_clinical_monitoring_dashboard.py:270` faz default para `sensitivity=0.94, specificity=0.92` quando não há dado.

E `medai_clinical_evaluation.py` contém `ClinicalValidationFramework.validate_for_clinical_use()` / `_determine_clinical_approval()` — **o software se autocertifica para uso clínico**.

### 1.7 Todo artefato real de treino registra falha

| Arquivo | Conteúdo |
|---|---|
| `models/comprehensive_training_report.json` | `successful_models: 0` · `"CRÍTICO: Nenhum modelo atende critérios clínicos"` |
| `models/advanced_training_summary.json` | `failed_models: [EfficientNetV2, VisionTransformer, ConvNeXt]` |
| `models/training_summary.json` | os 3 modelos com `accuracy: 0.3333` em 3 classes = acaso |
| `models/pre_trained/simple_demo/validation_results.json` | `AUC_mean: 0.5000`, `AUC_Pneumonia: 0.0` |
| `models/advanced_clinical_validation_study.json` | pneumonia `sensitivity: 0.325`, `auc_roc: 0.4333` — **abaixo do acaso** |
| `NIH_CHEST_XRAY/models_trained/evaluation_report_*.json` | `mean_auc: NaN`, `test_samples: 1` |
| `models/*_history.json` (6 arquivos) | `{}` |

Os logs (`training_log_*.csv`) mostram `loss` fixo em 0.6931 (= ln 2, rede não treinada) por 50 épocas.

### 1.8 Os dados são desenhos de OpenCV com nomes de arquivos do NIH

`create_synthetic_dataset.py`:

```python
cv2.ellipse(img, (cx-80, cy), (60,120), 0, 0, 360, 80, -1)   # "Pulmão esquerdo"
for i in range(8):                                            # "Costelas"
    cv2.line(img, (cx-120, y), (cx+120, y), 100, 2)
```

As 120 imagens em `data/nih_chest_xray/images/` recebem nomes de arquivos reais do NIH (`00000001_000.png`). Isso não é apenas dado sintético — é **risco de proveniência**: qualquer pessoa que copie esse diretório passa a ter arquivos que se apresentam como NIH ChestX-ray14 e não são.

Os quatro *loaders* de dataset em `medai_dataset_loaders.py` são stubs (`"CheXpert loader not fully implemented"`, idem LIDC-IDRI, BraTS, NIH).

### 1.9 A anonimização não é anonimização

`medai_dicom_processor.py:105` cobre 7 tags. Não cobre AccessionNumber, StudyDate/Time, InstitutionName/Address, StudyInstanceUID/SeriesInstanceUID/SOPInstanceUID, DeviceSerialNumber, OtherPatientIDs, tags privadas, nem anotação gravada no pixel. Não é o perfil DICOM PS3.15 Annex E.

Pior, os pseudo-IDs usam o `hash()` builtin do Python (linhas 132–134), que é **salgado por processo** — o mesmo paciente recebe um `ANON_xxxx` diferente a cada execução. Isso impede até manter um split estável por paciente. E `self._fernet = Fernet(Fernet.generate_key())` cria uma chave que nada usa.

Sob a LGPD isso é defeito de conformidade, não *code smell*.

### 1.10 O caminho de TC descarta as Unidades Hounsfield que acabou de calcular

`medai_modality_normalizer.normalize_ct()` é código bom: RescaleSlope/Intercept reais → HU → janela por órgão. Mas `dicom_to_array()` em seguida faz `(normalized_array * 255).astype(np.uint8)` e `preprocess_for_ai()` divide por 255. Uma única janela fixa de partes moles, quantizada em 8 bits, antes de qualquer modelo ver o dado. Correto para exibição; errado para um modelo de TC. Além disso, não há `apply_voi_lut` nem inversão de MONOCHROME1 no caminho principal.

### 1.11 Overclaiming documental — o passivo pessoal

O repositório é público e assinado com seu nome e CRM.

`README.md`:
> "arquiteturas ensemble state-of-the-art **validadas clinicamente**"
> "**Alta Acurácia**: >95% para condições críticas, >90% para condições moderadas"
> "**Análise de Viés**: ✅ Sistema validado sem viés detectado"
> "**Pronto para Produção**: ✅ Sistema validado"

`MODELS_LICENSE.md`: "**Acurácia Validada**: 92.3% (Sensibilidade: 90%, Especificidade: 89%)" — para modelos que não existem.

`docs/USER_GUIDE.md`: "91% de acurácia" na linha 200 e "**Acurácia: 20% atual**" na linha 412. O mesmo documento se contradiz.

`docs/CLINICAL_VALIDATION.md` é o único documento honesto: **"Overall Accuracy: 20% (1/5 correct predictions)"** — em 5 imagens sintéticas.

`FINAL_DELIVERY_PACKAGE.md`: "✅ Todos os Testes Passaram (100%)".

**Consequência:** a RDC 751/2022 e a Lei 6.437/77 tratam alegação de desempenho de dispositivo médico não registrado como infração sanitária. É também o primeiro achado de qualquer *due diligence* de investidor, parceiro ou comitê de ética. **Corrigir isso é P0 — antes de qualquer linha de código novo.**

### 1.12 Dívida estrutural

- **Nenhum `__init__.py`** no repositório, mas 37 imports relativos. Cada um envolto em `except ImportError: self.X = None` — 13 blocos só em `medai_integration_manager.py`. O sistema **sobe com todos os subsistemas em `None`** e não avisa.
- Autenticação stubada para sempre aceitar: `type('MockSecurity', (), {'authenticate': lambda self,u,p: True})`.
- 4 servidores, 2 launchers, 7 scripts de treino, 31 scripts `test_*.py` soltos na raiz (só 1 arquivo importa pytest), 5 READMEs, 8 scripts `fix_*`, `fix_syntax,py.txt` (vírgula no nome), `pasted_content.txt`.
- **Zero CI** (não existe `.github/`), zero `pyproject.toml`, zero lockfile, zero `conftest.py`.
- `requirements.txt` só com `>=`, faltando 11 pacotes efetivamente importados: `scipy`, `torch`, `psutil`, `h5py`, `PyYAML`, `reportlab`, `click`, `torchxrayvision`, `pynetdicom`, `hl7`, `plotly`.
- Python inconsistente: README 3.12 · Dockerfile `python:3.9-slim` · instalador 3.7+.
- Segredos commitados: `medai_config.json` (jwt_secret/encryption_key), `docker-compose.yml` (senhas default), `medai_security.db` (SQLite de 20 KB — e o `.gitignore` só tinha `db.sqlite3`, então não houve regra que pudesse pegá-lo).
- Caminhos absolutos de outra máquina: `/home/ubuntu/repos/radiologyai/`, `/mnt/data/NIH_CHEST_XRAY`, `C:/Program Files/`.
- `Dockerfile` roda `--workers 4`: quatro processos uvicorn, cada um com sua cópia do modelo — 4× RAM sem ganho no alvo CPU, e ordenação não-determinística do log de auditoria entre processos.

### 1.13 Governança do repositório

- **Não existe branch `main`.** O default remoto é `devin/1749325214-windows-radiology-ai-program`.
- **35 branches remotos**, a maioria `codex/*` e `devin/*` abandonados.
- **6 PRs abertos desde junho/2025.** O PR #3 é literalmente *"replace heavy TensorFlow-based code with lightweight stubs... ensure tests run without external data or heavy libraries"* — propõe passar nos testes removendo a IA.
- **Zero releases**, confirmando que as `download_url` do registro apontam para o nada.

### 1.14 O que vale salvar

| Arquivo | Destino |
|---|---|
| `medai_dicom_processor.py` (501) | **Migrar e corrigir** — melhor código do repo, mas reescrever anonimização, remover cache MD5 ilimitado, remover a quantização uint8 |
| `medai_modality_normalizer.py` (319) | **Migrar** — HU windowing correto; dividir por modalidade |
| `medai_confidence_calibration.py` (384) | **Migrar** a parte numpy/sklearn (~250 LOC framework-agnósticas) |
| `medai_clinical_evaluation.py` (1061) | **Migrar a matemática**; **deletar** `validate_for_clinical_use` e `_determine_clinical_approval` |
| `medai_medical_augmentation.py` (235) | **Parcial** — `elastic_deformation`, `simulate_breathing_motion` |
| `torchxray_integration.py` (376) | **Migrar o loader**; **deletar** `pathology_mapping` e `clinical_thresholds` |
| `medai_report_generator.py` | **Migrar a estrutura** (reportlab) |
| `docs/OPENAPI_SPEC.yaml` | **Manter** como contrato-alvo |
| `docs/CLINICAL_VALIDATION.md` | **Manter e reetiquetar** — é o documento honesto |
| `medai_explainability.py` | **ARQUIVAR** — falso; salvar só `overlay_heatmap` |
| `medai_sota_models.py` / `_real.py` | **ARQUIVAR** — `timm` + MONAI substituem |
| `medai_inference_system.py` | **ARQUIVAR** — é o que está sendo eliminado |
| `medai_integration_manager.py` | **ARQUIVAR** — é o anti-padrão |
| `medai_regulatory_compliance.py` | **ARQUIVAR** — conformidade é Markdown + QMS, nunca um enum Python |
| demais ~45 `medai_*` + 59 arquivos da raiz | **ARQUIVAR** em `legacy/` |

---

## 2. Escopo: multimodal em série, não em paralelo

O objetivo é SaMD ANVISA com RX, TC, RM e US.

Cada modalidade é, na prática, um dispositivo separado — pré-processamento, dataset, arquitetura, *ground truth*, validação e indicação de uso próprios. TC e RM exigem tratamento volumétrico 3D. Quatro modalidades em paralelo, por uma pessoa, produzem quatro protótipos rasos — que é exatamente o estado atual deste repositório.

**A plataforma é multimodal desde a Fase 1** (o ponto de extensão existe e as quatro modalidades estão no sistema de tipos). **O escopo regulatório não é.** Registra-se uma indicação primeiro; cada modalidade seguinte é uma alteração/inclusão contra um QMS existente e uma plataforma já validada, e o custo marginal cai cerca de 5 a 10 vezes.

---

## 3. Decisão de framework: PyTorch + MONAI

**Abandonar TensorFlow/Keras. Reconstruir sobre PyTorch + MONAI + `timm`, exportando ONNX para inferência.**

1. **O custo de migração do modelo é literalmente zero** — não existem pesos. `medai_sota_models.py` constrói grafos aleatoriamente inicializados. Não há nada treinado para portar. Este é o momento mais barato em que essa decisão estará disponível.
2. **Os únicos pesos reais alcançáveis hoje já são PyTorch** (`torchxrayvision`). O marco de linha de base honesta é uma inferência torch de qualquer forma.
3. **TC e RM são o argumento decisivo.** Exigem carregamento volumétrico com *spacing* correto, `Spacingd` para isotrópico, `ScaleIntensityRanged` sobre HU, `CropForegroundd`, treino por *patches* 3D que não cabem na VRAM e inferência por janela deslizante com mistura gaussiana nas bordas. MONAI entrega tudo isso testado e citável. Keras não entrega nada — por isso `medai_sota_models.py` tem `_extract_3d_patches` escrito à mão e sem teste.
4. **Ecossistema**: nnU-Net, TotalSegmentator, MedSAM, `timm`, `captum`, MONAI Label, EchoNet — todos torch.
5. **A explicabilidade passa a funcionar.** Grad-CAM real precisa de hook em camada nomeada: `register_full_backward_hook`, 30 linhas. O caminho TF produziu uma falsificação justamente porque extrair gradientes de um `tf.keras.Model` que você não construiu é incômodo — e ninguém se deu ao trabalho.
6. **Reprodutibilidade para V&V**: `torch.use_deterministic_algorithms(True)` + seeds + `CUBLAS_WORKSPACE_CONFIG` dá execução bit-reprodutível, que é o que uma verificação exige demonstrar.
7. **ONNX Runtime** dá alvo CPU sem dependência de torch, grafo fixo e sha256 estável — o artefato certo para release regulado. Atende "GPU, mas compatível com CPU".

**Custo real da migração: ~1,5 dia, e você deleta mais do que escreve.** O único código que toca TF e vale a pena é um método de `medai_confidence_calibration.py` (~2 h) e `MedicalAugmentationTF` (~1 dia, substituído por `Compose` do MONAI).

**Pinos:** `torch==2.5.1`, `monai==1.4.0`, `torchxrayvision==1.3.4`, `timm`, `pydicom==2.4.4` (2.x — a 3.x mudou a API de pixel handlers), `SimpleITK==2.4.x`, `onnxruntime==1.20.x`, Python `>=3.11,<3.14` (3.11 primário; o Colab já está em 3.13). Lockfile com `uv`. Cada versão entra literalmente na lista SOUP.

---

## 4. Arquitetura-alvo

`src`-layout, um pacote `radiologyai`. Não é preferência de estilo: torna impossível importar o pacote acidentalmente da raiz do repositório — exatamente a falha que permitiu 67 módulos planos se importarem de 37 maneiras diferentes.

```
pyproject.toml · uv.lock            # o lockfile é artefato regulatório (fonte da lista SOUP)
src/radiologyai/
├── __init__.py                     # só __version__; zero efeito colateral, zero import pesado
├── errors.py                       # hierarquia própria; nenhum except nu no pacote
├── config/                         # pydantic-settings; segredos só por env
├── io/                             # camada DICOM, agnóstica de modalidade
│   ├── reader.py                   # dcmread, VOI LUT, MONOCHROME1, RescaleSlope/Intercept
│   ├── series.py                   # ordena por ImagePositionPatient → volume 3D
│   ├── windowing.py                # presets WC/WW  ← migrar de medai_dicom_processor
│   ├── deident.py                  # PS3.15 Annex E; pseudo-ID HMAC-SHA256 DETERMINÍSTICO
│   └── metadata.py                 # StudyMetadata tipado (pydantic), não dict[str,str]
├── modalities/                     # ★ PONTO DE EXTENSÃO
│   ├── base.py                     # Protocol ModalityPlugin (§4.1)
│   ├── registry.py                 # descoberta por entry-point
│   └── xr/ · ct/ · mr/ · us/
├── models/
│   ├── card.py                     # sha256 VALIDADO (64 hex) — pegaria model_registry.json
│   ├── registry.py · store.py      # resolve → baixa → verifica hash → cacheia
│   ├── cards/*.yaml                # um por versão de modelo, commitado
│   └── backends/{torch,onnx}.py
├── inference/
│   ├── types.py                    # Finding, Prediction, StudyResult (pydantic, frozen)
│   └── engine.py                   # FALHA FECHADA. Sem fallback. Nunca.
├── calibration/
│   ├── temperature.py · metrics.py # ECE, MCE, diagrama de confiabilidade
│   └── abstention.py               # banda "não avaliável" — controle de risco primário
├── explain/gradcam.py              # REAL: hooks forward/backward em módulo torch nomeado
├── evaluation/
│   ├── metrics.py                  # AUROC/AUPRC + IC DeLong + bootstrap estratificado
│   ├── operating_point.py          # threshold em sensibilidade-alvo, com IC
│   ├── subgroup.py                 # sexo / faixa etária / incidência / fabricante
│   └── runner.py · report.py       # escreve artifacts/eval/<run_id>/
├── reporting/                      # laudo + PDF + disclaimers da IFU, versionados
├── api/                            # FastAPI: create_app() factory, routers/, security por env
├── persistence/
│   ├── models.py                   # SQLAlchemy + alembic
│   └── audit.py                    # append-only, encadeado por hash (prev_hash)
└── cli/main.py                     # typer: ingest, predict, evaluate, calibrate, trace

tests/{unit,integration,evaluation,requirements}/
datasets/manifests/*.csv            # splits + sha256 por arquivo — COMMITADO. Pixel não.
artifacts/                          # gitignored, exceto artifacts/eval/*/metrics.json
docs/regulatory/ · docs/model-cards/ · docs/datasheets/
legacy/                             # tudo o que sai; excluído de build, lint, CI e cobertura
```

### 4.1 O ponto de extensão de modalidade

```python
class ModalityPlugin(Protocol):
    code: ClassVar[str]                     # DICOM (0008,0060): "DX","CR","CT","MR","US"
    sop_classes: ClassVar[frozenset[str]]
    dimensionality: ClassVar[Literal["2d", "3d", "cine"]]

    def validate(self, study) -> ValidationReport: ...
    def preprocess(self, study, card) -> torch.Tensor: ...
    def postprocess(self, output, card) -> list[Finding]: ...
    def explain(self, study, card, target) -> Explanation: ...
```

Registrado por entry-point em `pyproject.toml`. Três consequências:

1. **Uma modalidade pode virar distribuição separada** (`radiologyai-ct`), com versão e relatório de validação próprios. Registra-se RX enquanto TC ainda é `0.x`.
2. **`validate()` é controle de risco, não conveniência.** O maior modo de falha silenciosa de IA radiológica em produção é entrada fora de distribuição — incidência lateral num modelo PA, exame pediátrico, AP de leito, um *scout* de TC confundido com RX. Tornar a rejeição um método obrigatório do plugin dá ao arquivo de risco um gancho real. **É a correção direta de `_analyze_image_fallback`.**
3. `dimensionality` força o caminho 3D a existir no sistema de tipos desde o primeiro dia, para que a implementação de RX não assente premissas 2D no código comum.

### 4.2 Regra de ouro

*Nenhum caminho de código produz um número que não veio de um modelo real medido.*

Sem `_create_dummy_model`, sem `_analyze_image_fallback`, sem `np.random` em métrica, sem `except ImportError: self.X = None`, sem saliência simulada, sem *mock ground truth*. Dependência ausente ou hash divergente → erro alto e explícito.

---

## 5. Roadmap por fases

Premissa: **8–12 h/semana, solo.** Prazos em tempo decorrido real.

### Fase 0 — Verdade e quarentena · 2–3 semanas (~25 h)

1. Escrever `docs/regulatory/01-intended-use.md` e `02-software-safety-classification.md`. **Antes de qualquer código.**
2. `HONEST_STATUS.md` na raiz, linkado do topo do README: nenhum modelo treinado existe; todas as métricas publicadas foram fabricadas; não é para uso clínico.
3. Remover as alegações falsas de `README.md`, `MODELS_LICENSE.md`, `docs/USER_GUIDE.md`, `FINAL_DELIVERY_PACKAGE.md`, `testing_criteria.md` — **agora**, sem esperar o número da Fase 1.
4. Deletar `models/model_registry.json`; marcar todo JSON de métrica como não-medição.
5. Deletar `data/nih_chest_xray/` (PNGs sintéticos com nomes de arquivos NIH).
6. Branch `v2`; `git mv` de tudo para `legacy/`.
7. `git filter-repo` para expurgar `medai_security.db` do histórico; rotacionar o que ele tocava; remover segredos de `medai_config.json` e `docker-compose.yml`.
8. `pyproject.toml` + `uv.lock` + `.github/workflows/ci.yml` verdes em pacote vazio.
9. Governança: criar `main` como default, fechar os 6 PRs obsoletos, podar os ~30 branches mortos, proteção de branch.
10. Iniciar credenciamento PhysioNet (curso CITI, ~6 h). A revisão sai em 24–48 h
    depois do relatório enviado; o gargalo é o curso. Ver `docs/access/README.md`.
11. Comprar HD externo de 4 TB; iniciar download do NIH ChestX-ray14.
12. Rascunhar `03-risk-management-plan.md` e semear `04-risk-file.md` com H-01…H-10 (§6.3).

**Saída:** zero alegação falsa em qualquer documento fora de `legacy/`; sem segredos no histórico; CI verde; credenciamento submetido.

### Fase 1 — Esqueleto v2 + a linha de base honesta · 6–8 semanas (~70 h)

O marco que muda a natureza do projeto.

- `io/` completo com testes unitários contra DICOMs sintéticos construídos com pydicom em `conftest.py`: MONOCHROME1, RescaleSlope/Intercept, VOI LUT, WC/WW multivalorado, tags ausentes.
- `modalities/base.py` + `registry.py` + plugin `xr`. Plugins `ct`/`mr`/`us` como *stubs* que levantam `NotImplementedError` — a interface fica provada multimodal desde o início.
- `models/` com `ModelCard` de sha256 validado; `inference/engine.py` **falha fechada**.
- `evaluation/` com IC de DeLong e análise de subgrupos.

**A armadilha de vazamento — e como evitá-la.** O movimento óbvio é rodar `densenet121-res224-all` no NIH ChestX-ray14. **Não faça.** Os pesos `-all` foram treinados em NIH, PadChest, CheXpert, MIMIC-CXR, RSNA e OpenI. Avaliar no NIH seria *in-distribution* — uma nova fabricação, só que mais sutil.

**Use `densenet121-res224-pc`** (só PadChest, Hospital San Juan / Alicante) **avaliado no split oficial `test_list.txt` do NIH** (25.596 imagens, disjunto por paciente). Países, equipamentos, populações e pipelines de rotulagem diferentes. Validação externa genuína, sem credenciamento, custo zero.

Alinhamento de rótulos explícito em `modalities/xr/labels.py`, com teste unitário. As 4 saídas do xrv sem contrapartida no NIH (`Lung Lesion`, `Lung Opacity`, `Enlarged Cardiomediastinum`, `Fracture`) são reportadas como `evaluated: false`, nunca descartadas em silêncio.

Um comando:
```bash
radiologyai evaluate --card xrv-densenet121-pc \
  --manifest datasets/manifests/nih_cxr14_test.csv \
  --bootstrap 2000 --seed 20260101 --out artifacts/eval/
```

`metrics.json` obrigatoriamente carrega: `git_sha`, `weights_sha256`, `manifest_sha256`, `seed`, versões de biblioteca, AUROC por patologia com IC95%, ponto de operação em sensibilidade-alvo, ECE, subgrupos, `not_evaluated[]` e `limitations[]`.

**Expectativa registrada antes da medição: 0,72–0,82.** Medido: **0,664** (`artifacts/eval/xrv-densenet121-pc__20260912T202549Z/`). A expectativa estava errada — cardiomegalia (0,796) e derrame (0,752) ficaram abaixo do previsto, e fibrose (0,455) ficou abaixo do acaso, o que indica que "Fibrosis" no PadChest e no NIH não nomeiam o mesmo achado. A lacuna PA/AP prevista apareceu (0,696 vs 0,631). Registrar a previsão errada ao lado do número medido faz parte do método: o v1 alegava 0,94.

Na mesma PR: `models/model_registry.json` deletado, README citando só `reports/EVALUATION_*.md`.

**Trava de regressão no CI:** `tests/evaluation/test_baseline_regression.py` roda o mesmo caminho de código sobre um subconjunto de 50 imagens commitado e exige AUROC macro dentro de ±0,02 de um valor dourado. CPU, ~40 s, em toda PR. Qualquer mudança de pré-processamento que degrade o modelo em silêncio passa a quebrar o build.

**Saída:** AUROC por patologia, medido, com IC, reproduzível bit a bit por terceiro a partir de um checkout limpo. Cobertura ≥85% em `io/` e `evaluation/`.

**Estado em setembro/2026 — infraestrutura concluída, execução pendente:**

| Item | Estado |
|---|---|
| Pacote `radiologyai`, CI, 3 portões automatizados | concluído |
| Backend torchxrayvision com verificação de sha256 | concluído |
| Manifest do NIH (25.596 imagens, 2.797 pacientes, disjunção verificada) | **commitado** |
| Executor de avaliação com IC, subgrupos e proveniência completa | concluído |
| Notebook do Colab | `notebooks/01_baseline_nih_cxr14_colab.ipynb` |
| **Executar sobre as 25.596 imagens** | **concluído em 2026-09-12** — AUROC macro 0,664; duas execuções isolam o efeito do pré-processamento (delta −0,0008) |

**Achado durante a implementação:** os pesos `-pc` têm 3 das 18 cabeças de saída
**não treinadas** (o PadChest não continha `Lung Lesion`, `Lung Opacity` nem
`Enlarged Cardiomediastinum`). Usar a lista `default_pathologies` da biblioteca
como se fosse a do modelo faria reportar AUROC de uma cabeça nunca treinada —
mesma classe de defeito das métricas fabricadas do v1, por mecanismo mais sutil.
Tratado em REQ-021. Os 14 rótulos do NIH têm cabeça treinada, então a linha de
base é válida para todos eles.

### Fase 2 — Produtizar o caminho de RX · 8–10 semanas (~90 h)

- `calibration/` — *temperature scaling* em split retido; ECE e diagrama de confiabilidade em todo relatório.
- **`calibration/abstention.py`** — três bandas: *achado provável* / *não avaliável* / *achado improvável*. Controle de risco primário para H-01/H-02; precisa existir antes de qualquer uso clínico-adjacente.
- `explain/gradcam.py` real. Teste unitário: modelo com gradientes zerados na última conv produz mapa nulo — **o teste que o código v1 reprovaria**.
- `api/` — factory FastAPI, `/api/v1/*` conforme `docs/OPENAPI_SPEC.yaml`, segredos por env. **Nenhum *mock ground truth* em lugar nenhum.**
- `persistence/audit.py` — log append-only encadeado por hash (hash do exame, id do card, sha256 dos pesos, git sha, saída, timestamp, decisão do médico).
- Dockerfile `python:3.11-slim`, caminho ONNX, `--workers 1`, non-root, healthcheck.
- SRS em `docs/regulatory/06-requirements/`; `scripts/trace.py`; portão de rastreabilidade no CI.

### Fase 3 — Modelo próprio de RX + validação externa · 10–14 semanas (~120 h)

1. Escrever `10-clinical-evaluation-plan.md` **primeiro**, com critérios de aceitação fixados de antemão. Sem isso o resultado é exploratório, não confirmatório.
2. Treinar em NIH ChestX-ray14 (split disjunto por paciente), MONAI + `timm` (DenseNet121 ou ConvNeXt-Tiny, init ImageNet, BCE multi-label, AMP). Meta: superar a linha de base zero-shot.
3. **Validar externamente em VinDr-CXR** — 18.000 exames com **rótulo por consenso de radiologistas**, não minerado por NLP; conjunto de teste de 3.000 lido por 5 radiologistas. É o melhor conjunto de rótulos de RX de tórax gratuito que existe.
4. Exportar ONNX; verificar paridade torch↔onnx < 1e-4; publicar pesos no Hugging Face com sha256 real.

**Saída:** um modelo que você treinou, validado externamente contra rótulos adjudicados, com IC, subgrupos, calibração e política de abstenção, mais relatório de verificação assinado. **É o primeiro ponto em que o projeto é credível para alguém de fora — e o primeiro em que gastar dinheiro com empresa é racional.**

### Fase 4 — Segunda modalidade: ecocardiografia · 12–16 semanas (~130 h)

Plugin `us` (cine, amostragem de frames, classificação de janela); CAMUS + EchoNet-Dynamic; tarefa de FEVE ou qualidade de janela **adjudicada por você**.

**Por que ecocardiografia antes de TC:** CAMUS + EchoNet somam **9 GB** e custam zero. Ecocardiografia é a única modalidade em que você tem expertise de domínio que um engenheiro solo não pode comprar — você adjudica rótulos, identifica artefatos, julga se uma FEVE é plausível e define um ponto de operação clinicamente significativo sem contratar um painel de leitores. Em todas as outras modalidades você compete com equipes bem financiadas sobre os mesmos dados públicos. Em eco, sua formação médica está no caminho crítico. TC/RM exigem parceria com radiologista para adjudicação — planeje isso.

Prova do ponto de extensão: adicionar uma modalidade deve tocar apenas `modalities/us/` e `models/cards/`. **Um check de CI rejeita PR que mexa no núcleo junto.**

### Fase 5 — TC e RM (3D) + endurecimento regulatório · 6–9 meses

Plugin `ct` em LUNA16 (MONAI 3D: `Spacingd`, `ScaleIntensityRanged` sobre HU, `CropForegroundd`, inferência por janela deslizante) — é aqui que a GPU em nuvem é gasta. Plugin `mr` em BraTS. IFU pt-BR, cibersegurança, plano de vigilância pós-mercado. Opcionalmente PACS/DICOMweb. **Contratar consultoria regulatória brasileira para confirmar a classe de risco antes de comprometer o caminho.**

### Fase 6 — Empresa, QMS, dossiê ANVISA · 12–24 meses, travado por dinheiro

Empresa + AFE + Responsável Técnico; QMS ISO 13485 e certificação; BPF; dossiê; petição. **Travado por financiamento, não por engenharia.** O artefato da Fase 3 é o que você usa para levantá-lo.

---

## 6. Trilha regulatória (paralela, desde a Fase 0)

### 6.1 Classificação de segurança de software: adotar Classe C

IEC 62304 §4.3: **A** = sem lesão possível; **B** = lesão não-séria possível; **C** = morte ou lesão séria possível.

O argumento para **B** seria: o software é assistivo e não autônomo, um radiologista qualificado lê todo exame, e uma falha não alcança o paciente sem passar por julgamento humano independente.

**Recomendação: Classe C.** Razões:

1. **O uso pretendido inclui achados letais.** Pneumotórax em RX, hemorragia intracraniana em TC e nódulo pulmonar estão todos no escopo declarado. Um falso negativo que ancore o leitor é plausivelmente fatal. §4.3 pergunta se a morte é *possível*, não provável.
2. **§4.3 só permite rebaixar a classe se o controle de risco for externo ao software E sua eficácia for justificada.** "O radiologista pega" é exatamente a alegação que a literatura de viés de automação contesta — leitores que veem uma saída negativa da IA demonstravelmente perdem achados que veriam sozinhos. Você teria de defender essa justificativa com evidência que não tem.
3. **Custo assimétrico.** Ser questionado numa classificação B no meio da avaliação significa retrofitar documentação de projeto em nível de unidade sobre um código pronto. Construir para C a partir de um diretório vazio custa documentação que você deveria produzir de qualquer forma.
4. **O delta é pequeno aqui.** C acrescenta sobre B: arquitetura detalhada com segregação de SOUP (§5.3.3–5.3.6), documentação de projeto de unidade (§5.4.2) e verificação de unidade com critérios de aceitação definidos (§5.5.3). Seu CI já precisa de testes unitários com portão de cobertura.
5. **A saída existe.** Se depois você estreitar o uso pretendido — "segunda leitura apenas, achados não time-critical, sem alegação de triagem, exclui pneumotórax e hemorragia" — documenta-se o rebaixamento para B. Rebaixar depois é fácil; promover depois é reescrita.

### 6.2 Enquadramento brasileiro

- **RDC 751/2022** — classificação de risco (I–IV), notificação vs. registro. A classificação de SaMD se dá pela **Regra 11**, que transpõe a lógica do IMDRF.
- **RDC 657/2022** — regulamento específico de SaMD.
- Um CADx/CADt que informa diagnóstico de condições graves cai, na leitura mais provável, em **Classe III** → **registro** (não notificação), o que puxa **certificação BPF** e dossiê pesado. Classe IV se a alegação envolver condição com risco de morte sem revisão humana.
- Pré-requisitos societários: **AFE**, **Responsável Técnico**, QMS alinhado a **ISO 13485** / RDC 665/2022.
- Ver também **IN 190/2024** e **RDC 848/2024**, que alteraram partes do arcabouço.

**Ressalva explícita:** o arcabouço regulatório brasileiro se moveu repetidamente. Trate cada número de RDC acima como ponto de partida a verificar contra o texto consolidado vigente no site da ANVISA, e contrate consultoria regulatória para confirmar a classificação antes de gastar dinheiro no caminho. **Não deixe essa incerteza atrasar a engenharia** — todo documento de §6.4 é exigido sob qualquer versão do arcabouço, e também sob IMDRF e MDR.

**Um alívio relevante:** a RDC 657/2022 isenta de regularização SaMD desenvolvido internamente por serviço de saúde e de uso exclusivo desse serviço, nas classes I e II. Não cobre Classe III, mas abre caminho legítimo para uso interno restrito e coleta de evidência antes do registro. Confirmar com consultoria.

### 6.3 Perigos-semente para o arquivo de risco

Não os genéricos — estes saem do que existe neste código:

| ID | Perigo | Controle | Corrige |
|---|---|---|---|
| H-01 | Falso negativo em achado crítico (pneumotórax, HIC) → tratamento tardio → morte | banda de abstenção; nunca exibir "Normal" como afirmação positiva; IFU proíbe uso para *rule-out* | — |
| H-02 | Viés de automação — negativo confiante ancora o leitor | UI mostra incerteza; sem *top-1* único; leitura concorrente na IFU | — |
| H-03 | Entrada fora de distribuição processada em silêncio (lateral, pediátrico, AP de leito, modalidade errada) | `ModalityPlugin.validate()` rejeita | §1.2 `_analyze_image_fallback` |
| H-04 | Descasamento versão de pesos/código → modelo errado usado em silêncio | sha256 verificado na carga, falha fechada; versão em todo resultado e registro de auditoria | §1.1 hashes fabricados |
| H-05 | Má interpretação de DICOM — MONOCHROME1 não invertido, RescaleSlope ignorado, VOI LUT errado | teste unitário por interpretação fotométrica | §1.10 |
| H-06 | Erro de lateralidade em achado reportado | lateralidade propagada do DICOM e asserida, nunca inferida do pixel | — |
| H-07 | Deriva de desempenho silenciosa pós-implantação | log de auditoria + monitoramento de distribuição de entrada + PMS | §1.6 |
| H-08 | Explicabilidade enganosa — saliência que não reflete o modelo | explicação derivada de gradientes reais; teste de modelo nulo → mapa nulo | §1.3 |
| H-09 | Vazamento de PHI por des-identificação incompleta, logs ou mensagens de erro | perfil PS3.15; redação de log | §1.9 |
| H-10 | Confiança mal calibrada — 0,85 não significa 85% | *temperature scaling*; ECE reportado; IFU declara que escores não são probabilidade de doença | — |

Cada um vira um `REQ-` e um `TEST-`. Essa cadeia é o que um avaliador realmente inspeciona.

### 6.4 Documentos a escrever AGORA — baratos, alto retorno, sem empresa

| # | Documento | Quando | Por que tem alavancagem |
|---|---|---|---|
| 01 | **Uso Pretendido / Indicações de Uso** | **antes de qualquer código v2** | A página mais densa em alavancagem do projeto. Fixa modalidade, região anatômica, população, usuário, cenário de cuidado, alegação assistivo-vs-autônomo e contraindicações. Determina a classe de risco, a classe de segurança, os requisitos de dados e o que `validate()` deve rejeitar. Escrevê-la primeiro previne a deriva de escopo que já afundou este repositório uma vez. ~3 páginas. |
| 02 | Classificação de segurança do software | antes do código | §6.1 escrito. 2 páginas |
| 03 | Plano de Gestão de Risco (ISO 14971 + ISO/TR 24971) | Fase 0 | 5 páginas |
| 04 | Arquivo de Risco / Análise de Perigos | Fase 0, vivo | §6.3. Perigo → sequência → situação → dano → severidade/probabilidade → controle → verificação → risco residual |
| 05 | Plano de Desenvolvimento (62304 §5.1) | Fase 0 | Descreve política de branch, revisão, portões de CI e release — que você constrói de qualquer forma. Transforma `.github/workflows/` em evidência regulatória |
| 06 | SRS (`06-requirements/REQ-*.md`) | Fase 1→2, vivo | Cada requisito com id, tipo, `risk_controls:` e `verified_by:` |
| 07 | Arquitetura (62304 §5.3) | Fase 2 | §4 é o primeiro rascunho; Classe C exige mostrar segregação de SOUP |
| 08 | **Lista SOUP** | Fase 1, automatizada | Cada dependência: nome, versão, fabricante, propósito, lista de anomalias publicada (consulta CVE). **`scripts/soup.py` gera a partir de `uv.lock` + API do OSV.** É por isso que o lockfile é artefato regulatório e por isso que `requirements.txt` só com `>=` é defeito de conformidade, não desleixo |
| 09 | Plano + Protocolos/Relatórios de Verificação | Fase 2→3 | Protocolo antes da execução, relatório depois. CI escreve `artifacts/verification/<git_sha>/` |
| 10 | Plano de Avaliação Clínica | Fase 3 | Critérios de aceitação fixados **antes** da avaliação definitiva, senão ela não é confirmatória |
| 11 | Cibersegurança + LGPD/DPIA | Fase 3 | Modelo de ameaças, SBOM CycloneDX, PHI em repouso/trânsito. Registra também a remediação de `medai_security.db`/`jwt_secret` |
| 12 | Rotulagem / IFU (pt-BR) | Fase 5 | Advertências, limitações, tabela de desempenho com IC, "não substitui avaliação médica", significado da banda de abstenção |
| 13 | Matriz de rastreabilidade | Fase 2, **gerada** | §6.5 |
| 14 | Controle de Mudanças + Resolução de Problemas (§6, §9) | Fase 2 | Seu template de PR e de issue, descritos como processo |
| 15 | Plano de Vigilância Pós-Mercado | Fase 5 | Inclui monitoramento de deriva — que é também requisito de projeto sobre `persistence/audit.py` |
| 16 | Model Cards + Dataset Datasheets | Fase 1, por artefato | Alimentam diretamente a avaliação clínica |

**Escreva 01 e 02 nesta semana, antes de tocar em código.** São duas tardes e restringem tudo a jusante.

### 6.5 Rastreabilidade — barata e imposta pelo CI

Requisitos em Markdown com frontmatter YAML:

```markdown
---
id: REQ-042
type: safety
risk_controls: [RISK-003]
verified_by: [tests/requirements/test_req_042.py]
---
O sistema DEVE rejeitar exame cuja tag DICOM (0008,0060) Modality ou
(0018,5101) View Position não conste do conjunto aceito pelo ModalityPlugin
ativo, e NÃO DEVE produzir predição para ele.
```

Testes declaram cobertura: `@pytest.mark.requirement("REQ-042")`.

`scripts/trace.py` (~200 LOC) gera `13-traceability.md` e **quebra o CI** se: algum requisito não tem teste verificador, algum teste cita REQ inexistente, algum perigo não tem controle, ou algum controle não tem requisito.

**Esse portão é o valor inteiro.** Uma matriz de rastreabilidade que pode divergir não vale nada; uma que bloqueia merge é evidência.

### 6.6 O portão do dinheiro

Tudo em §6.4 custa tempo. O que segue custa dinheiro que não está no orçamento:

| Item | Ordem de grandeza |
|---|---|
| Implantação + certificação ISO 13485 | R$ 30.000 – 80.000+ |
| AFE + registro de empresa + Responsável Técnico | recorrente |
| Certificação BPF (exigida para registro Classe III/IV) | auditoria + prazo |
| Taxas de petição ANVISA | reduzidas para EPP, ainda materiais |
| Consultoria regulatória para dossiê Classe III | R$ 15.000 – 50.000 |
| Estudo clínico prospectivo | CEP/CONEP + custos de centro |

**Portanto: faça todo o §6.4 agora, a custo zero, e abra a empresa apenas quando um número de validação externa real justificar o gasto.** Um arquivo técnico registrável sem empresa é ativo financiável. Uma empresa sem modelo validado é conta mensal. Os documentos de §6.4 são também o que um parceiro, um investidor ou um comitê de ética pede para ver primeiro — então se pagam antes de a ANVISA entrar na conta.

---

## 7. Dados e orçamento

R$ 3.000 ≈ USD 550. **Armazenamento é o gargalo, não computação** — você tem GPU.

| Item | Custo |
|---|---|
| HD externo USB 3.0 de 4 TB | R$ 700–900 — **comprar primeiro** |
| Cloudflare R2, ~150 GB (artefatos, pesos ONNX) — egress zero | ~R$ 15/mês |
| GPU sob demanda para 3D (Vast.ai/RunPod, A100 ~USD 1,0–1,5/h) | R$ 600 (~100 h), só a partir da Fase 5 |
| VPS + domínio para a API de demonstração | R$ 40/mês, opcional até a Fase 3 |
| Hugging Face Hub (repos públicos) | R$ 0 |
| Reserva | R$ 800 |
| **Total** | **≈ R$ 2.800** |

### Datasets recomendados

**RX de tórax (Fases 1–3)**

| Dataset | Acesso | Tamanho | Uso |
|---|---|---|---|
| **NIH ChestX-ray14** | livre, sem registro | 42 GB | treino/val + split de teste da linha de base honesta |
| **VinDr-CXR** | PhysioNet credenciado + DUA | 190 GB DICOM (~20 GB em PNG) | **seu conjunto de teste externo real** — 18.000 exames com consenso de radiologistas |
| **CheXpert** | registro Stanford AIMI, grátis | `-small` 11 GB | segunda fonte externa; o `valid` de 234 exames é rotulado por especialista |
| PadChest | registro livre | 1 TB | pular |
| MIMIC-CXR-JPG | PhysioNet credenciado | 570 GB | pular — NIH+VinDr+CheXpert já cobrem três populações |

**TC (Fase 5):** **LUNA16** (~60 GB, livre, subconjunto curado do LIDC-IDRI, tarefa definida e leaderboard estabelecido) → LIDC-IDRI completo (125 GB, TCIA) se precisar da anotação dos 4 radiologistas. RSNA Intracranial Hemorrhage (~180 GB, Kaggle) tem alto valor clínico mas é caro em disco.

**RM (Fase 5):** **BraTS** (~15 GB, melhor porta de entrada — pequeno, pré-processado, co-registrado, com *skull-stripping*, 4 sequências e décadas de baselines publicados). IXI (~15 GB, cérebros saudáveis — ideal para testar normalização e rejeição OOD). fastMRI: pular, é dataset de reconstrução.

**US (Fase 4):** **CAMUS** (~2 GB, 500 pacientes, A2C/A4C, segmentação ED/ES + FEVE) e **EchoNet-Dynamic** (~7 GB, 10.030 vídeos apicais 4 câmaras com FEVE).

### Credenciamento PhysioNet — comece na semana 1

VinDr-CXR exige conta credenciada: curso CITI "Data or Specimens Only Research" (grátis, ~6 h) mais referência. Como médico com CRM ativo é diretamente obtenível, e **a revisão da PhysioNet costuma sair em 24–48 h** depois do relatório CITI enviado — o gargalo é o curso, não a fila. (Versões anteriores deste roadmap diziam 2–6 semanas; estava errado.)

**Há um caminho imediato para rótulos adjudicados, antes do VinDr:** o conjunto de validação do CheXpert — 234 imagens, voto majoritário de 3 radiologistas certificados — exige apenas registro na Stanford AIMI, sem credenciamento. É bem menor que o VinDr (234 imagens contra 3.000 exames), portanto com ICs largos, mas já responde à pergunta que a linha de base do NIH deixou aberta: fibrose em 0,448 é o modelo ou é o rótulo? Implementado em `radiologyai.data.chexpert`; instruções em `docs/access/README.md`.

### Governança de dados (imposta pelo CI)

1. Nenhum pixel no git, nunca. `.gitignore`: `data/`, `*.dcm`, `*.nii*`, `*.mhd`, `*.raw`. Exceção: fixtures sintéticas e o subconjunto de 50 imagens em `tests/data/`.
2. Manifests com sha256 por arquivo são commitados. São o registro de reprodutibilidade e custam ~3 MB.
3. Todo dataset ganha `docs/datasheets/<dataset>.md`: fonte, licença, termos de DUA, direito de redistribuição, população, **proveniência do rótulo (minerado por NLP vs. adjudicado)**, vieses conhecidos, citação.
4. **Splits por paciente, sempre.** Um check de CI assere interseção zero de `patient_id` entre manifests. Vazamento por paciente é a forma mais comum de um número publicado de IA radiológica estar errado.

---

## 8. Verificação

### CI

`ci.yml` — toda PR, CPU, ~5 min:
```bash
uv sync --frozen                       # lockfile precisa estar atualizado
uv run ruff check src/ tests/ && uv run ruff format --check
uv run mypy --strict src/radiologyai/
uv run pytest tests/unit tests/integration --cov=radiologyai --cov-fail-under=85
uv run pytest tests/evaluation/test_baseline_regression.py    # AUROC dourado ±0.02
uv run python scripts/trace.py --check                        # REQ↔TEST↔RISK completos
uv run python scripts/soup.py --check                         # SOUP bate com uv.lock
uv run python scripts/check_honesty.py                        # §8.2
```

`security.yml` — semanal + PR: `bandit -r src/`, `pip-audit`, `gitleaks detect`, SBOM CycloneDX.

`eval.yml` — manual/tag, GPU: avaliação completa, sobe `artifacts/eval/<run_id>/`, abre PR com o `reports/EVALUATION_*.md` regenerado. **Registros de verificação produzidos por máquina, não à mão.**

### Portões por fase

| Fase | Portão | Comando |
|---|---|---|
| 0 | segredos expurgados | `git log --all -- medai_security.db` → vazio; `gitleaks detect --log-opts="--all"` limpo |
| 0 | sem alegação falsa | `grep -rniE "9[0-5]%\|acurácia validada\|validado clinicamente" --include=*.md . \| grep -v legacy/` → vazio |
| 1 | inferência falha fechada | `pytest -k "missing_weights or hash_mismatch"` — deve levantar, nunca degradar |
| 1 | sem aleatoriedade oculta | `grep -rn "np\.random" src/radiologyai/ \| grep -v augment` → vazio |
| 1 | sem import silencioso | `grep -rn "except ImportError" src/` → vazio |
| 1 | avaliação reprodutível | duas execuções, mesma seed → `metrics.json` idêntico byte a byte (menos timestamp) |
| 1 | higiene de import | `python -X importtime -c "import radiologyai"` < 200 ms; torch não importado |
| 2 | integridade da auditoria | `pytest -k audit_chain` — linha mutada deve ser detectada |
| 2 | explicabilidade real | `pytest -k test_null_gradients_yield_null_map` |
| 2 | rastreabilidade completa | `scripts/trace.py --check` exit 0 |
| 3 | disjunção por paciente | `pytest -k patient_leakage` sobre todos os manifests |
| 3 | paridade ONNX | diff máx. torch vs onnx < 1e-4 em 100 imagens |
| 3 | validação externa | métricas VinDr reproduzem de checkout limpo, em CPU **e** GPU |
| 4 | fronteira do plugin | CI rejeita PR que adiciona modalidade e mexe em `radiologyai/{io,inference,models}/` |
| 5 | correção 3D | inferência por janela deslizante em volume sintético com lesão plantada recupera a lesão |

### 8.2 `scripts/check_honesty.py` — o guardião que mais importa

Check de CI (~80 LOC) que quebra o build em qualquer um destes:

1. Alegação numérica de desempenho (`\d{2,3}(\.\d+)?\s*%`, `AUC\s*[:=]\s*0\.\d+`) em Markdown fora de `legacy/` que não esteja em bloco gerado a partir de `artifacts/eval/`.
2. `np.random` / `random.` fora de `modalities/*/augment.py` e `tests/`.
3. Campo sha256 em model card que não seja 64 caracteres hex minúsculos. **Só isso já teria pegado `model_registry.json`.**
4. `except ImportError` em `src/`.
5. `# Mock`, `# Simulated`, `# TODO: real`, `_simulate_`, `_fallback_analysis` em `src/`.
6. Model card sem artefato de avaliação vinculado.

Este é o mecanismo que impede a recorrência do modo de falha que produziu o repositório atual. **O problema nunca foi falta de competência — foi que nada no sistema jamais objetou quando um número foi inventado. Torne a objeção automática.**

---

## 9. Cronograma honesto

| Marco | Decorrido, solo, 8–12 h/sem |
|---|---|
| Repositório honesto e governado | **1 mês** |
| Um número externamente válido e reproduzível no repositório | **2–3 meses** |
| API de RX implantável, com calibração, abstenção, explicabilidade real e auditoria | **5–7 meses** |
| Modelo próprio validado externamente contra rótulos adjudicados | **9–12 meses** |
| Duas modalidades sobre plataforma de plugin comprovada | **13–17 meses** |
| Quatro modalidades + arquivo técnico completo de RX | **20–28 meses** |
| Registro ANVISA concedido, uma modalidade | **3–4 anos**, travado por dinheiro |
| Registro ANVISA, quatro modalidades | **5–7 anos solo** |

**A aritmética desconfortável:** a 10 h/semana, a Fase 6 não é alcançável sozinho. A estratégia realista é **tornar a Fase 3 financiável** — um modelo genuinamente validado, com relatório de validação externa honesto, um conjunto documental em formato de QMS e uma base de código limpa é exatamente o que um hospital parceiro, um edital (FAPESP/FINEP) ou um cofundador precisa ver.

**Otimize o plano para chegar à Fase 3 em ~12 meses, não para chegar à ANVISA sozinho.**

Os aceleradores reais são: (a) parceria com radiologista para adjudicação, (b) vínculo institucional para dados e ética — o InCor é sua porta natural, (c) financiamento que compre horas de engenharia. Nenhum deles é código.

---

## Referências regulatórias

- ANVISA — [RDC 657/2022](https://anvisalegis.datalegis.net/action/ActionDatalegis.php?acao=abrirTextoAto&tipo=RDC&numeroAto=00000657&seqAto=000&valorAno=2022&orgao=RDC%2FDC%2FANVISA%2FMS), software como dispositivo médico
- ANVISA — [RDC 751/2022](https://anvisalegis.datalegis.net/action/ActionDatalegis.php?acao=abrirTextoAto&tipo=RDC&numeroAto=00000751&seqAto=000&valorAno=2022&orgao=RDC%2FDC%2FANVISA%2FMS), classificação de risco, notificação e registro (Regra 11 para SaMD)
- ANVISA — [Perguntas e Respostas RDC 657/2022](https://www.gov.br/anvisa/pt-br/assuntos/noticias-anvisa/2022/software-como-dispositivo-medico-perguntas-e-respostas/perguntas-respostas-rdc-657-de-2022-v1-01-09-2022.pdf)
- ANVISA — [Manual para regularização de equipamentos médicos](https://www.gov.br/anvisa/pt-br/assuntos/noticias-anvisa/2025/anvisa-publica-nova-versao-de-manual-para-regularizacao-de-equipamentos-medicos)
- IEC 62304 — ciclo de vida de software de dispositivo médico
- ISO 14971:2019 + ISO/TR 24971 — gestão de risco
- ISO 13485 — sistema de gestão da qualidade
- IMDRF — framework de categorização de SaMD
- [Project MONAI](https://github.com/Project-MONAI/MONAI) — framework de imagem médica em PyTorch
