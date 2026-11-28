# Roadmap — RadiologyAI v2.0 (otimizado)

**O maior número possível de condições, cada uma com a confiabilidade científica que de fato tem.**

Documento de planejamento. Versão 2.0 — novembro/2026 (estado conferido contra a árvore em 2026-11-27; marcas de conclusão da v1.0 registradas em `main` incorporadas ao §1). Substitui a versão 1.0 (setembro/2026),
que continua no histórico git.
Autor do projeto: Dr. Guilherme Capel Pasqua (CRM-SP 175873).

> **Software de pesquisa. Não é dispositivo médico.** Não é regularizado na ANVISA, não tem
> clearance FDA nem marcação CE. Toda saída é um **achado sugerido para confirmação médica**,
> nunca um diagnóstico. Estado real do projeto: [`HONEST_STATUS.md`](HONEST_STATUS.md).

---

## 0. O critério que governa este roadmap

O critério do dono do projeto é **diagnosticar o maior número possível de condições com
confiabilidade científica**. As duas metades puxam em sentidos opostos — amplitude sem
evidência é o que afundou o v1; evidência sem amplitude é um produto que não serve a ninguém.
A v2.0 resolve a tensão com uma regra única:

> **Toda condição pode entrar no sistema. Nenhuma condição sobe de nível sem evidência que o
> próprio software calcula.**

Em termos operacionais:

1. A unidade de progresso é a **condição catalogada com nível de evidência calculado** (§2),
   não a funcionalidade entregue.
2. O indicador principal (KPI) é o **número de condições em nível A e em nível B, por
   modalidade**, gerado a partir de `levels.json` — nunca digitado.
3. As fases por funcionalidade do v1.0 viram **portões de evidência G0–G6** (§4). Cada portão
   tem critério de saída verificável e um artefato que o comprova.
4. Amplitude multimodal **agora**, com modelos abertos reais (§5), porque o catálogo graduado
   permite mostrar uma condição de nível D no modo pesquisa sem fingir que ela é A.

O que **não** muda do v1.0: a decisão PyTorch (§7.1), a Classe C (§8.1), os perigos H-01…H-10
(§8.3), os documentos regulatórios baratos (§8.4), o portão do dinheiro (§8.6) e a regra de ouro.

---

## 1. Diagnóstico — resumo

A versão 1.0 deste roadmap abria com um diagnóstico de 14 subseções do repositório legado:
nenhum peso treinado, predição por heurística OpenCV, Grad-CAM falso (bordas Canny borradas),
métricas de "validação clínica" geradas por `np.random`, *mock ground truth* devolvido pela API,
pneumotórax reportado como pneumonia, dados sintéticos com nomes de arquivo do NIH,
anonimização incompleta e não determinística, segredos commitados. O diagnóstico completo e a
retratação pública estão em **[`HONEST_STATUS.md`](HONEST_STATUS.md)**; o texto integral da v1.0
está no histórico git deste arquivo.

**Onde estamos em novembro/2026:**

| Item | Estado |
|---|---|
| Código legado | removido da árvore (commit `c25f686`); recuperável no histórico (`16fdd47`) |
| Núcleo v2 `src/radiologyai/` | DICOM, des-identificação, plugins de modalidade, model cards com sha256, inferência falha-fechada, calibração, abstenção, Grad-CAM real, auditoria encadeada, API |
| Entregas da Fase 1–2 da v1.0 (marcadas como concluídas em `main`, conferidas na árvore) | `calibration/temperature.py`, `calibration/platt.py` e `calibration/reliability.py` (ECE e diagrama de confiabilidade) — calibração sobre dado real ainda não commitada como artefato; `calibration/abstention.py` (três bandas); `explain/gradcam.py` com teste de gradiente nulo; `api/` FastAPI; `persistence/audit.py`; SRS, `scripts/trace.py` e portão no CI; backend ONNX (`models/backends/onnx.py`, sha256 verificado, em `build_backend`) e trava de regressão (`evaluation/regression.py`, `tests/evaluation/test_baseline_regression.py`, CLI `radiologyai check-regression`) — **ambos sem execução sobre dado real do CheXpert/NIH**; `Dockerfile` (`python:3.11-slim`, `--workers 1`, não-root, *healthcheck*) e `.dockerignore`; workflow manual de avaliação reprodutível `.github/workflows/eval.yml` |
| Lateralidade | REQ-043 (RISK-006): `Laterality`/`ImageLaterality` propagadas em `StudyMetadata`, conflito ou código inválido rejeitado no RX, pré-processamento sem inversão esquerda-direita |
| Requisitos rastreados | `REQ-NNN.md` em [`docs/regulatory/06-requirements/`](docs/regulatory/06-requirements/), cada um com teste verificador; contagem na [matriz gerada](docs/regulatory/13-traceability.md); portão `scripts/trace.py --check` no CI |
| Medições de desempenho | linha de base de terceiros no NIH ChestX-ray14, AUROC macro 0,664 — `artifacts/eval/xrv-densenet121-pc__20260912T215742Z/metrics.json` (fora do cálculo de nível); 98 registros sob o protocolo em `evidence/measured/` (ledger em `evidence/measured/ledger.yaml`): cardiomegalia e opacidade pulmonar aprovadas externamente no NIH (nível A), fratura reprovada no GRAZPEDWRI-DX, câncer de mama reprovado no CMMD, lista completa em [`HONEST_STATUS.md`](HONEST_STATUS.md). Os relatos de Colab de 2026-11-25 em [`docs/evaluation/`](docs/evaluation/phase1-nih1000-baseline.md) **não têm artefato commitado** e não contam como medição (ver [`HONEST_STATUS.md`](HONEST_STATUS.md)) |
| Camada de análise `src/radiologyai/analysis/` | manifestos por modelo (`available` com pesos sha256, ou `planned` com motivo), adaptadores para RX, TC, RM, mamografia, US e eco, biomarcadores de TC com pré-condições, sondas lineares; verificação em andamento |
| Fábrica de evidência `src/radiologyai/evidence/` | predições → medição sob o protocolo → registro com proveniência → `levels.json`; níveis re-julgados no build contra `protocol.yaml` e o plano; registros medidos a partir de commit, sem código sujo |
| Plataforma | API, persistência, fila, laudo preliminar (PDF, DICOM SR, FHIR), DICOMweb e frontend no código; verificação em andamento |
| Catálogo de condições | esquema, regra de nível e protocolo pré-registrado commitados (`src/radiologyai/catalog/`); 63 condições em `conditions.yaml`; nenhuma em A ou B (KPI em §2.3); evidência publicada em `evidence/published/` |
| Validação clínica | **nenhuma** |
| Governança do repositório | `main` é o ramo padrão e está protegida; restam 6 ramos remotos (conferido pela API do GitHub em 2026-11-27); PRs obsoletos fechados e ramos mortos podados conforme registrado em `main` |
| Pendência de governança | `medai_security.db` continua no **histórico** git (presente em `3f5ae26`, `c80c22e`, `0b89c95`); expurgo com `git filter-repo` pendente |

---

## 2. O catálogo de condições graduado por evidência

O catálogo (`src/radiologyai/catalog/`) responde a uma pergunta para o produto inteiro: *para
esta condição, este modelo e este rótulo, qual é a força da evidência?* A resposta nunca é
digitada: `compute_level()` em `schema.py` a deriva dos registros de evidência, e `levels.json` é
um arquivo gerado (REQ-090).

### 2.1 Níveis

| Nível | Significado | Onde aparece |
|---|---|---|
| **A** | Medido **por este projeto** em dado externo, sem sobreposição conhecida ou plausível com o treino nem com a seleção do modelo, com entrada independente do padrão de referência, aprovado no protocolo | visão principal e laudo |
| **B** | Desempenho publicado em estudo revisado por pares, ou limiar de diretriz aplicado a medida com desempenho publicado; citação **conferida na fonte** e aprovada na barra de publicação | visão principal e laudo, com a citação |
| **C** | Medido por este projeto e aprovado no protocolo, mas em hold-out interno por paciente, em dado externo com sobreposição possível/conhecida com o treino ou a seleção, ou com entrada derivada do padrão de referência | visão principal, com selo e ressalvas |
| **D** | Exploratório: zero-shot, sem medição aprovada, refutado por medição própria, ou limitado (modelo não distribuído, manifesto só de pesquisa) | **somente no modo pesquisa**; nunca no laudo (REQ-091) |

Regras estruturais (pré-registradas em `compute_level`):

- **Medição nossa prevalece.** Se existem medições externas limpas e todas reprovam, a condição
  cai para D *mesmo com publicação favorável* (REQ-093). Um artigo não salva um modelo que
  falhou no nosso dado.
- **Sem campo de nível.** `Condition` e `EvidenceRecord` rejeitam chaves desconhecidas; não há
  onde escrever "nível A" à mão.
- **Medido exige proveniência.** Registro `measured_*` sem git sha, hash do manifesto do dataset
  e sha256 dos pesos é rejeitado pelo esquema.
- **Publicado exige citação conferida.** Registro `published` sem `verified_at_source` não
  promove nada. Alegação de fornecedor e clearance de *outro* produto são contexto, nunca
  evidência de nível B.

### 2.2 Decisão por ponto de operação com origem

Um escore só vira "positivo" ou "negativo" em relação a um `OperatingPoint` explícito
(`src/radiologyai/calibration/decision.py`) que carrega papel (`rule_out`, `rule_in`,
`balanced`, `guideline`) e **origem** (ajuste do protocolo, DOI de diretriz). Sem ponto de
operação utilizável, a decisão é `score_only`: o escore aparece sem veredito, nunca com limiar
inventado (REQ-092). Os pontos `rule_out`/`rule_in` são ajustados na metade de calibração e
reportados na metade de teste, conforme o protocolo ([`docs/protocolo-evidencia.md`](docs/protocolo-evidencia.md)).

### 2.3 O KPI

> **KPI = número de condições em nível A e em nível B, por modalidade.**
> Secundários: condições em C; condições em D (amplitude exploratória); condições **refutadas**
> por medição própria (um número que deve ser publicado, não escondido).

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

A tabela é transcrita de `src/radiologyai/catalog/levels.json` contando **só entradas
`distributed: true`** (modelo com manifesto que cobre os pesos medidos); o CI
(`tests/unit/test_status_docs_levels.py`) falha se este roadmap, o README ou o HONEST_STATUS
divergirem dela. Os registros medidos foram gerados a partir de um commit: C só para
tuberculose (`xr-probe-tb`, hold-out interno) e fratura refutada (AUROC 0,690 no GRAZPEDWRI-DX —
`evidence/measured/grazpedwri-dx-sample/xr-probe-fracture/fracture.json`). O nível A que o commit
`23af5ca` atribuiu à TB foi **retratado** (Shenzhen reutilizado na seleção do codificador; sonda
não distribuída) — ver [`HONEST_STATUS.md`](HONEST_STATUS.md).

---

## 3. O protocolo pré-registrado

`src/radiologyai/catalog/protocol.yaml` (`protocol_id: "2026-11-v1"`, congelado em
2026-11-24) fixa **antes de qualquer medição** os critérios de aceitação: elegibilidade mínima
de casos, barras de discriminação, detecção, concordância e segmentação, bootstrap por
paciente, separação calibração/teste, alvos de *rule-out*/*rule-in*, barra para evidência
publicada e subgrupos obrigatórios. Mudar um limiar depois de ver resultado exige **novo
`protocol_id`**, e cada registro de evidência nomeia o protocolo sob o qual foi julgado. A
versão legível, com a política de vazamento, está em
**[`docs/protocolo-evidencia.md`](docs/protocolo-evidencia.md)**.

---

## 4. Portões de evidência G0–G6

As fases do v1.0 eram de funcionalidade ("produtizar RX", "segunda modalidade"). Os portões
são de **evidência**: cada um responde "o que podemos afirmar a mais, e com que prova?".
Engenharia acontece em paralelo, em todos eles.

| Portão | Pergunta que responde | Critério de saída (verificável) | Estado |
|---|---|---|---|
| **G0 — Integridade de engenharia** | O software pode mentir? | CI verde com ruff, mypy `--strict`, pytest, `scripts/check_honesty.py`, `scripts/trace.py --check`, `scripts/soup.py --check`; regra de ouro imposta; histórico sem segredos; `main` protegida | **parcial** — portões automatizados ativos; `main` protegida; expurgo de `medai_security.db` do histórico pendente |
| **G1 — Catálogo de evidência pública** | Quantas condições têm evidência, e de que força, usando só dados e literatura públicos? | Todo modelo que roda de verdade tem entradas no catálogo; `levels.json` gerado; KPI por modalidade publicado; todo registro A reproduzível de checkout limpo; todo registro B com citação conferida | **em andamento** — 98 registros medidos de checkout limpo; 2 condições em A (NIH), A de TB retratado; câncer de mama refutado no CMMD; 13 condições de tórax refutadas em todos os modelos no NIH; CQ500 com inferência incompleta |
| **G2 — Modelo próprio + validação externa retrospectiva brasileira** | Funciona em dado brasileiro de vários centros, que nunca vimos? | Protocolo por condição aprovado em CEP (Lei 14.874/2024; dispensa de TCLE pela Res. CNS 466/2012 item IV.8 quando justificável); dados de ≥ 2 centros brasileiros des-identificados pelo controlador; relatório STARD-AI/CLAIM; conjunto de teste sequestrado | não iniciado — depende de parceria institucional |
| **G3 — Estudo de leitores MRMC** | O médico assistido erra menos? | Estudo multi-leitor multi-caso com padrão de referência e desenho pré-especificado (Obuchowski & Bullen 2022, DOI 10.1148/radiol.211593) | não iniciado — modelo de protocolo em [docs/clinical/protocolo-mrmc-G3.md](docs/clinical/protocolo-mrmc-G3.md) (não submetido a CEP); sem modo de leitura no software; depois de G2 |
| **G4 — Piloto prospectivo silencioso** | Funciona no fluxo real, sem influenciar o cuidado? | Execução em paralelo ao fluxo, saídas **ocultas** do médico assistente, comparação posterior; relato DECIDE-AI (DOI 10.1038/s41591-022-01772-9); protocolo CEP | não iniciado — modelo de protocolo em [docs/clinical/protocolo-piloto-silencioso-G4.md](docs/clinical/protocolo-piloto-silencioso-G4.md) (não submetido a CEP); **habilitador no software implementado (REQ-195)**: `MEDAI_SHADOW_MODE` — análises rodam, são gravadas e auditadas; saída de IA oculta de todo papel sem `shadow:view` (papel `researcher`); nenhum envio ao PACS; exportação CSV sem PHI e auditada (`GET /api/v1/shadow/export`); pendentes protocolo no CEP, integração no hospital e relato DECIDE-AI; depois de G3 |
| **G5 — Submissão regulatória** | Pode virar produto? | Empresa com AFE e Responsável Técnico; BPF (RDC 665/2022); dossiê conforme RDC 657/2022 Art. 12 e RDC 751/2022; requisitos essenciais RDC 848/2024 | travado por dinheiro (§8.6) |
| **G6 — Pós-mercado** | Continua funcionando? | Plano de vigilância pós-mercado em operação, monitoramento de deriva, retorno do médico estruturado (perfil IHE AIRA) | não iniciado |

**Por que G2 é o portão crítico:** G1 mede o que modelos de terceiros fazem em dados públicos,
quase todos norte-americanos, europeus ou asiáticos. Nenhum número de G1 diz algo sobre a
população brasileira. G2 é o primeiro ponto em que o projeto é credível para um hospital
parceiro, um edital ou um investidor — e é o portão a tornar financiável (§12.2).

**Regra de tamanho amostral para G2 (proposta, exige decisão do dono e novo `protocol_id`):**
mínimo de 100 positivos e 100 negativos com padrão de referência por condição (Collins et al.
2016, DOI 10.1002/sim.6787; Vergouwe et al. 2005, DOI 10.1016/j.jclinepi.2004.06.017), ou cálculo
de Riley et al. 2021 (DOI 10.1002/sim.9025) quando a prevalência for baixa. São regras derivadas
para modelos de predição; aplicá-las a classificadores de imagem é extrapolação declarada. O
protocolo atual (`2026-11-v1`) exige menos (30/30) porque foi desenhado para G1.

---

## 5. Amplitude multimodal agora — com modelos abertos reais

Princípio: **amplitude entra em D, sobe por medição**. Todo modelo abaixo roda de verdade
(pesos reais com sha256), ou está como `planned` com o motivo. Nenhum stub. A coluna "caminho de
nível" diz o que precisa acontecer para a condição sair de D; os dados de viabilidade vêm dos
*spikes* de engenharia de novembro/2026 (execução real em CPU, 4 núcleos, sem GPU), e nenhum
número de desempenho desses *spikes* é citado aqui porque eles não são avaliações (n pequeno,
sem protocolo).

### 5.1 Radiografia de tórax (RX)

| Componente | Modelo / licença | O que calcula | Caminho de nível | Estado |
|---|---|---|---|---|
| Classificador multirrótulo | TorchXRayVision DenseNet121 (código Apache-2.0), conjuntos de pesos `pc`, `chex`, `mimic_nb`, `mimic_ch`, `nih`, `rsna`, `all`, e ResNet50 `res512-all` | até 18 saídas por conjunto; cabeças não treinadas nunca reportadas (REQ-021) | **A** por avaliação externa com pares sem vazamento (§9.4); B só com validação revisada por pares do mesmo conjunto de pesos (a tabela `BENCHMARKS.md` do repositório não é revisada por pares e não conta) | pesos `pc` medidos no NIH (`artifacts/eval/xrv-densenet121-pc__20260912T215742Z/metrics.json`); `pc`, `chex`, `mimic_nb`, `mimic_ch`, `all` e `nih` medidos no teste oficial do NIH e no subconjunto adjudicado (82 registros; tabela em `HONEST_STATUS.md`): A para cardiomegalia (`pc`) e opacidade pulmonar (`chex`, `mimic_*`) |
| Gate de distribuição (OOD) | autoencoder ResNetAE do TorchXRayVision + classificador de incidência xinario | rejeita entrada que não é RX de tórax frontal antes do modelo (H-03; REQ-122) | requisito de segurança, verificado por teste — não é condição do catálogo | adaptador disponível |
| Índice cardiotorácico | segmentador PSPNet ChestX-Det (via TorchXRayVision) | razão coração/tórax a partir de máscaras | **D** até medirmos Dice e erro do ICT contra máscaras manuais; limiar 0,5 só em PA ortostático por metadado DICOM, senão "não interpretável contra 0,5" (REQ-124) | adaptador disponível |
| Tuberculose | sonda linear sobre *embeddings* do RAD-DINO congelado (cartão: só pesquisa ⇒ `commercial_use: no`); treino em Shenzhen, teste em Montgomery (NLM, termos próprios) | escore de TB | validação cruzada dentro do mesmo conjunto é **C**, nunca externa; treino num conjunto e teste no outro pode chegar a A se passar no protocolo | sonda ajustada (manifesto `xr-probe-tb`); hold-out interno Shenzhen aprovado (máximo C: Shenzhen também escolheu o codificador) — `evidence/measured/tb-shenzhen/xr-probe-tb/tuberculosis.json`; Montgomery não elegível (29 positivos) — `evidence/measured/tb-montgomery/xr-probe-tb/tuberculosis.json` |
| Fratura e achados de punho pediátrico (RX musculoesquelético) | sondas lineares sobre RAD-DINO; FracAtlas e GRAZPEDWRI-DX (CC BY 4.0) | escores de fratura, gesso, metal, osteopenia | mesmo raciocínio: externo só entre conjuntos | fratura **refutada**: reprovada no GRAZPEDWRI-DX — `evidence/measured/grazpedwri-dx-sample/xr-probe-fracture/fracture.json`; gesso, metal e osteopenia não elegíveis no hold-out (poucos positivos) — `evidence/measured/grazpedwri-dx-sample-test30/` |

### 5.2 Tomografia computadorizada (TC)

| Componente | Modelo / licença | O que calcula | Caminho de nível | Estado |
|---|---|---|---|---|
| Anatomia | TotalSegmentator `total` (Apache-2.0); "not a medical device" na própria documentação | 117 classes anatômicas | base para biomarcadores | viabilidade demonstrada em CPU |
| Derrame pleural/pericárdico | TotalSegmentator `pleural_pericard_effusion` (Apache-2.0) | volume segmentado (REQ-131, REQ-132) | **D**: os artigos de Sexauer et al. 2022 (DOI 10.1097/RLI.0000000000000869) e Wilder-Smith et al. 2022 (DOI 10.3390/diagnostics12051045) avaliam nnU-Nets de um rótulo; os pesos distribuídos (Dataset315_thoraxCT, 3 rótulos, 600 casos) são outro modelo, então a evidência não se transfere (B retratado em 2026-11-28); classificador simples-vs-complexo **não** exposto | adaptador disponível |
| Hemorragia intracerebral | TotalSegmentator `cerebral_bleed` (Apache-2.0) | volume segmentado | **D** — nenhuma acurácia de segmentação publicada para este modelo; caminho para A: medição externa no CQ500 (CC BY-NC-SA 4.0) | adaptador disponível; medição no CQ500 pendente (5 de 80 exames; `evidence/measured/ledger.yaml`) |
| Lesões hepáticas | TotalSegmentator `liver_lesions` (Apache-2.0) | lesões segmentadas | publicado com teste interno e externo (Nicoli et al., DOI 10.1007/s10278-025-01716-y), mas **não** é ferramenta de exclusão; nível a decidir pelo protocolo | adaptador disponível |
| Nódulo pulmonar | TotalSegmentator `lung_nodules` | nódulos segmentados | **D** — sem artigo; treinado em parte no LIDC-IDRI, logo **não há conjunto de teste limpo** disponível | adaptador disponível; sem caminho de nível no momento |
| Cistos renais | TotalSegmentator `kidney_cysts` | cistos segmentados | **D** — sem artigo | adaptador disponível |
| Biomarcadores em HU | máscaras do TotalSegmentator + HU | atenuação trabecular de L1 (osteoporose oportunista), atenuação hepática (esteatose), %LAA-950 (enfisema), diâmetro aórtico, volume esplênico, índice de Evans | **B** pelo limiar de diretriz ou estudo de validação do método, **somente** quando as pré-condições de aquisição são satisfeitas (fase de contraste, kVp, kernel, espessura, inspiração); senão **abstenção com motivo** (REQ-133…REQ-138) | adaptador `ct-biomarkers` disponível; limiares só de fontes conferidas (REQ-134) |
| Escore de cálcio coronário | TotalSegmentator `coronary_arteries` | — | exige chave de licença do TotalSegmentator (gratuita só para uso não comercial) | **planejado** |
| Sarcopenia / composição corporal | TotalSegmentator `tissue_types` | — | idem: tarefa com chave de licença | **planejado** |
| Fase de contraste | classificador `totalseg_get_phase` | pré-condição dos biomarcadores | exige licença; enquanto isso, fase desconhecida ⇒ biomarcador **abstém** | **planejado** |

### 5.3 Ressonância magnética (RM)

| Componente | Modelo / licença | O que calcula | Caminho de nível | Estado |
|---|---|---|---|---|
| Anatomia | TotalSegmentator `total_mr` (Apache-2.0) | classes anatômicas em RM | base | adaptador disponível |
| Glioma (subregiões) | bundle MONAI `brats_mri_segmentation` (Apache-2.0; dados BraTS 2018 com termos próprios) | TC/WT/ET em glioma **já conhecido** | **D** — Dice do desenvolvedor no split de validação usado para seleção; **nunca** usar saída não vazia como "tumor presente" | **planejado**: exige T1c, T1, T2 e FLAIR co-registrados, sem crânio, a 1 mm (pré-processamento BraTS não implementado) e não há caso de teste aberto no cache |
| Lesões de substância branca | LST-AI (Wiltgen et al. 2024, DOI 10.1016/j.nicl.2024.103611; MIT) | carga lesional em T1w + FLAIR | publicado; saída é "carga lesional", **nunca** "esclerose múltipla" | **planejado**: pacote fora do ambiente travado e nenhum par T1w + FLAIR aberto no cache para verificar execução real |
| Lesões hepáticas em RM | TotalSegmentator `liver_lesions_mr` (Apache-2.0) | lesões segmentadas | mesmo artigo de Nicoli et al.; desempenho em RM inferior ao de TC | planejado |

### 5.4 Mamografia

| Componente | Modelo / licença | O que calcula | Caminho de nível | Estado |
|---|---|---|---|---|
| Escore de câncer | `ianpan/mammoscreen` + recorte `ianpan/mammo-crop` (código e pesos Apache-2.0; `commercial_use: unclear` porque os termos dos dados de treino não foram lidos); código vendorizado sem execução remota (REQ-142); exige VOI LUT completa (REQ-140) | escore por mama (CC + MLO) | **D**: o cartão do modelo relata só hold-out interno, sem IC; CBIS-DDSM foi usado no pré-treino e **não** é externo; CMMD (CC BY 4.0) é externo, mas só tem doença biopsiada (benigno vs maligno), não rastreamento | adaptador disponível; **refutado** no CMMD: AUROC 0,790 [0,715; 0,859], n=156 mamas de teste (`evidence/measured/cmmd/mammoscreen/malignancy.json`) |
| Densidade mamária | cabeça de densidade do mammoscreen; bundle MONAI `breast_density_classification` (Apache-2.0, "not to be used for diagnostic purposes") | categoria A–D | **D** — nenhuma métrica publicada com proveniência | adaptadores disponíveis |
| Risco em 5 anos | Mirai (MIT; "not intended ... for any other clinical use") | risco futuro, **não** detecção | nenhum dado disponível tem seguimento para medir | **planejado**: *snapshots* em pickle legado do pacote `onconet`, sem dado para verificar |

### 5.5 Ultrassonografia

| Componente | Modelo / licença | O que calcula | Caminho de nível | Estado |
|---|---|---|---|---|
| Lesão mamária benigno/maligno | sonda linear sobre BiomedCLIP congelado (herda a restrição "só pesquisa" do cartão ⇒ `commercial_use: no`) | escore de malignidade | BUS-BRA (INCA, Rio de Janeiro; biópsia; CC BY 4.0) e BrEaST (CC BY 4.0) permitem treino num e teste noutro; BUSI tem imagens duplicadas (carta de Pawłowska et al. 2023, DOI 10.1016/j.dib.2023.109247) e **nunca** é conjunto externo | sonda ajustada (`us-probe-breast-malignancy`) no espelho de BUSI — licença do espelho não conferida na fonte original (pendência em [`MODEL_LICENSES.md`](MODEL_LICENSES.md)); reajuste em BUS-BRA/BrEaST recomendado; aprovada em BUSI (interno) e BUS-BRA (externo, sobreposição possível, caixa derivada da máscara de referência), reprovada no BrEaST — registros em `evidence/measured/bus-bra/`, `evidence/measured/busi/`, `evidence/measured/breast-lesions-usg/`; mantida em D (manifesto só de pesquisa) |
| Segmentação interativa | MedSAM ViT-B (Apache-2.0) | máscara a partir de caixa **desenhada pelo usuário** | ferramenta de medição assistida; nível só para medidas derivadas, com Dice contra máscara de especialista; mm só com espaçamento físico (REQ-147) | adaptador disponível |
| Esteatose hepática; lesão hepática benigno/maligno | sondas sobre BiomedCLIP; conjunto Byra (55 pacientes, biópsia; CC BY 4.0) e Zenodo 7272660 (CC BY 4.0) | escores | C/D: um centro cada; Byra é coorte bariátrica com n efetivo 55 | esteatose não elegível (Byra: 19 positivos e 9 negativos no teste) — `evidence/measured/byra-fatty-liver/us-probe-liver-steatosis/liver_steatosis.json`; lesão hepática reprovada (limite inferior do IC) — `evidence/measured/liver-lesion-us-7272660/us-probe-liver-lesion-malignancy/liver_lesion_malignant.json`; manifestos só de pesquisa |

### 5.6 Ecocardiografia

| Componente | Modelo / licença | O que calcula | Caminho de nível | Estado |
|---|---|---|---|---|
| Traçado assistido do VE + FE por Simpson biplanar | MedSAM2 `MedSAM2_US_Heart.pt` (código Apache-2.0; pesos com licença contraditória — metadado `cc-by-sa-4.0`, texto "research and education purposes" ⇒ tratado como **não comercial**) | contorno do VE em A2C/A4C, volumes e FE | **modo pesquisa**; o médico confirma ou corrige cada contorno; concordância com FE de referência julgada pela barra de ICC do protocolo; CAMUS (CC BY-NC-SA 4.0) como referência; caixas do usuário em ED e ES, sem propagação silenciosa (REQ-145, REQ-146) | adaptador disponível; com a caixa derivada da máscara de referência aprova a barra de ICC no CAMUS (`evidence/measured/camus/medsam2-echo-lv/lv_ef.json`), sem caixa reprova (`evidence/measured/camus/medsam2-echo-lv/lv_ef.prompt_free.json`); mantido em D (manifesto só de pesquisa; a caixa do teste não é a do usuário) |
| EchoPrime | licença acadêmica Cedars-Sinai: só pesquisador acadêmico, **proibida** transferência, uso comercial exige licença | — | é recuperação (kNN sobre 1,23 milhão de laudos de pacientes), não regressão; rodá-lo exige redistribuir esse corpus; FE ruim na amostra CAMUS do *spike* | **excluído** |
| PanEcho | nenhuma licença publicada nos caminhos verificados | — | sem licença vale o direito autoral padrão | **excluído** até publicarem licença |

### 5.7 Zero-shot — só modo pesquisa

CheXzero (código MIT; pesos sem licença própria, treinados em MIMIC-CXR ⇒ `commercial_use: no`;
adaptador disponível, REQ-126), BiomedCLIP (MIT; o cartão declara "**any** deployed use case ...
is currently out of scope") e Merlin (TC 3D, MIT; executado só em *spike*, sem adaptador) entram
em **D** e ficam no modo pesquisa. Servem para explorar amplitude e para priorizar o
que medir. Uma condição zero-shot só sai de D pela mesma regra de qualquer outra: medição externa
aprovada no protocolo. BiomedCLIP permanece em modo pesquisa mesmo se medido, pelo próprio termo
de uso.

### 5.8 Licenças — a regra

- Pesos não comerciais ou só-pesquisa são **permitidos**, marcados `commercial_use: "no"`.
- Código GPL/AGPL ligado ao pacote e espelhos de dataset que "lavam" licença (redistribuição sem
  o DUA original) são **excluídos**.
- Sem licença publicada ⇒ não carrega, não avalia, não distribui (PanEcho).
- Inventário completo: [`MODEL_LICENSES.md`](MODEL_LICENSES.md).

---

## 6. A plataforma web

O catálogo só tem valor se o médico vê, no fluxo, cada achado com seu nível, sua fonte e sua
incerteza. Componentes (todos sob a regra de ouro). Estado em 2026-11-25: código presente em
`api/`, `persistence/`, `worker/`, `reporting/`, `interop/` e `frontend/`, com requisitos
REQ-150…REQ-166; o estado de verificação de cada um está na
[matriz de rastreabilidade](docs/regulatory/13-traceability.md).

| Componente | O que faz | Controle de risco |
|---|---|---|
| Lista de trabalho | exames ingeridos, estado da análise, abstenções com motivo | H-03 (entrada fora do escopo visível, nunca silenciosa) |
| Visualizador com sobreposições | imagem + máscaras e mapas calculados pelo modelo real; mapa só existe se veio de gradiente/máscara real | H-08 |
| Laudo estruturado preliminar | PDF, DICOM SR (TID 1500), FHIR `DiagnosticReport` com status preliminar; cada achado com nível, fonte e ponto de operação | H-01, H-02, H-10 |
| Revisão e assinatura do médico | aceitar / editar / rejeitar cada achado; assinatura **vinculada ao hash do conteúdo** analisado e exibido | H-02, H-04 |
| Controle de acesso por papel (RBAC) | pesquisador, médico revisor, administrador | H-09 |
| Trilha de auditoria encadeada | hash da entrada, modelo, sha256 dos pesos, git sha, saída, decisão do médico; adulteração detectável (REQ-070, REQ-071) | H-04, H-07 |
| Interoperabilidade | DICOMweb (QIDO/WADO/STOW) com Orthanc; saída alinhada ao perfil IHE AI Results (AIR) | — |
| Modo pesquisa | interruptor explícito que revela nível D com marca d'água; nunca ativo por padrão | REQ-091 |

**O que a plataforma não faz:** triagem, priorização de fila, notificação de achado crítico,
laudo autônomo, comunicação ao paciente. Qualquer um deles reclassifica o software (§8.2) e
contraria a Resolução CFM nº 2.454/2026 (Art. 5 §2, vedação de delegar à IA a comunicação de
diagnóstico sem mediação humana).

---

## 7. Arquitetura

### 7.1 Decisão de framework: PyTorch + MONAI (mantida)

Abandonar TensorFlow/Keras foi decidido no v1.0 e continua certo, por razões que só ficaram mais
fortes: (1) o custo de migração era zero, não havia pesos; (2) **todos** os modelos abertos que o
§5 usa são PyTorch — TorchXRayVision, TotalSegmentator/nnU-Net, MONAI, MedSAM/MedSAM2,
mammoscreen, CheXzero, Merlin; (3) TC e RM exigem o carregamento volumétrico e a inferência por
janela deslizante que o MONAI entrega testados; (4) Grad-CAM real exige *hooks* em módulo nomeado;
(5) `torch.use_deterministic_algorithms(True)` dá a reprodutibilidade que a verificação exige;
(6) ONNX Runtime continua sendo o alvo de release em CPU.

### 7.2 Camadas

Detalhada em [`docs/regulatory/07-architecture.md`](docs/regulatory/07-architecture.md). Resumo:

```
núcleo (sem torch)   io/ · modalities/ · models/ (cards, registry, backends) · inference/ ·
                     calibration/ (temperature, abstention, decision) · catalog/ · evaluation/ ·
                     audit/ · explain/ · config/ · errors.py
análise              analysis/: manifestos de modelo, registro com verificação de pesos,
                     adaptadores por modelo, roteamento por tipo de exame, pipeline
                     (qualidade → predição → calibração → ponto de operação → decisão → nível),
                     biomarcadores com pré-condições, sondas lineares, medidas assistidas —
                     dependências pesadas atrás de extras opcionais, importadas sob demanda
plataforma           api/ · persistência · fila de trabalho · laudo/interoperabilidade · frontend
```

O ponto de extensão por modalidade continua sendo o `ModalityPlugin`
(`src/radiologyai/modalities/base.py`), registrado por *entry point*: `validate()` rejeita a
entrada fora do escopo **antes** do modelo — controle de risco, não conveniência — e
`dimensionality` (`2d`, `3d`, `cine`) mantém o caminho volumétrico no sistema de tipos.

### 7.3 Regra de ouro

*Nenhum caminho de código produz predição, métrica, mapa de calor ou alegação regulatória que
um modelo ou algoritmo real não tenha calculado sobre a entrada real.* Falha ⇒ exceção explícita
ou abstenção com motivo. Modelo que não roda de verdade fica `planned`, sem stub. Sem
`_create_dummy_model`, sem *fallback* heurístico, sem saliência simulada, sem *mock ground truth*.

Regras de arquitetura (impostas pelo CI): `import radiologyai` não importa torch, numpy,
SimpleITK nem cv2; dependência ausente ⇒ erro que nomeia o extra, nunca `except ImportError`;
nenhum `np.random`/`random` fora da estatística de avaliação; todo resultado é pydantic congelado
com `extra="forbid"`.

---

## 8. Trilha regulatória (paralela, desde já)

### 8.1 Classe de segurança de software: **C** (mantida)

IEC 62304 §4.3: A = nenhuma lesão possível; B = lesão não séria; C = morte ou lesão séria
possível. A versão 2.0 só reforça a escolha: o escopo multimodal inclui hemorragia intracraniana,
derrame pericárdico e câncer de mama. Um falso negativo que ancore o leitor é plausivelmente
fatal, e o controle externo "o radiologista pega" é exatamente o que a literatura de viés de
automação contesta. Rebaixar depois é fácil; promover depois é reescrita. Argumento completo em
[`docs/regulatory/02-software-safety-classification.md`](docs/regulatory/02-software-safety-classification.md).

### 8.2 Enquadramento brasileiro (identificadores conferidos na fonte)

- **RDC 657/2022** (DOU 30/03/2022; vigente desde 1º/07/2022) — software como dispositivo médico.
  Art. 5: isenção *in house* só para classes I e II, com registros completos de validação
  guardados por 10 anos. Art. 12: relatório técnico para classes III/IV. Art. 13: declarar
  conformidade com IEC 62304, IEC 62366-1 e ISO 14971 (versões mais atuais permitidas). Art. 16:
  modificação que afete desempenho exige petição — não há mecanismo tipo PCCP. A revisão da RDC
  657 consta da lista preliminar da Agenda Regulatória 2026-2027 (situação "Em andamento").
- **RDC 751/2022** (vigente desde 1º/03/2023) — classificação; **Regra 11**: software que informa
  decisão diagnóstica é classe II, III se a decisão pode causar deterioração grave ou cirurgia,
  IV se pode causar morte ou deterioração irreversível. Classes I/II: notificação; III/IV:
  registro, com certificado de BPF. Art. 2 §8: dispositivo destinado a investigação clínica é
  isento, proibidos comercialização e outros usos.
- **RDC 665/2022** — boas práticas de fabricação (controle de projeto, validação de software).
- **RDC 848/2024** (DOU 08/03/2024) — requisitos essenciais; Arts. 49–53 exigem ciclo de vida,
  gestão de risco, V&V e cibersegurança para qualquer SaMD; revogou a RDC 546/2021.
- **LGPD** (Lei 13.709/2018) — dado de saúde é sensível (Art. 5 II); a base de pesquisa do Art. 11
  II c é de **órgão de pesquisa** (Art. 5 XVIII), o que uma pessoa física não é: dados
  hospitalares entram pelo parceiro institucional.
- **Ética em pesquisa** — Lei 14.874/2024, Decreto 12.651/2025, Res. CNS 466/2012 (dispensa de
  TCLE, item IV.8) e Res. CNS 738/2024 (bancos de dados).
- **Resolução CFM nº 2.454/2026** — IA como ferramenta de apoio, médico responsável final, uso
  registrado no prontuário, vedada comunicação de diagnóstico pela IA sem mediação humana.

**Posição de trabalho:** o produto, quando e se for produto, é classe III pela Regra 11 (os
achados de maior risco puxam a classe) — **a confirmar com consultoria regulatória** antes de
gastar dinheiro. Hoje ele é software de pesquisa e não deve exibir linguagem de finalidade
clínica ("detecta", "diagnostica", "triagem").

### 8.3 Perigos — arquivo de risco

Os perigos-semente H-01…H-10 do v1.0 viraram `RISK-001`…`RISK-010` no arquivo de risco
([`docs/regulatory/04-risk-file.md`](docs/regulatory/04-risk-file.md)), com os mesmos números
(H-0N ↔ RISK-00N) e os requisitos que os controlam:

| ID | Perigo | Controles principais |
|---|---|---|
| RISK-001 (H-01) | Falso negativo em achado crítico | abstenção; nunca "Normal" (REQ-062, REQ-063); nível D fora da visão principal (REQ-091) |
| RISK-002 (H-02) | Viés de automação | incerteza visível; sem *top-1* isolado; revisão obrigatória |
| RISK-003 (H-03) | Entrada fora de distribuição processada em silêncio | gate de escopo antes do modelo (REQ-031, REQ-042); falha alta (REQ-001) |
| RISK-004 (H-04) | Descasamento pesos/código | sha256 verificado (REQ-020); falha fechada (REQ-030) |
| RISK-005 (H-05) | Má interpretação DICOM | Modality LUT, MONOCHROME1 (REQ-005, REQ-006) |
| RISK-006 (H-06) | Erro de lateralidade | lateralidade do DICOM, nunca do pixel; RX (REQ-043), mamografia (REQ-141), TC/US/eco (REQ-194: ausente = "lateralidade desconhecida", conflito ou código inválido sem lado; TC entrega ao modelo o lado da geometria LPS; eco não aplicável) |
| RISK-007 (H-07) | Desempenho real diverge do declarado | avaliação reprodutível, nível calculado, auditoria (REQ-050, REQ-090, REQ-093, REQ-070) |
| RISK-008 (H-08) | Explicabilidade enganosa | Grad-CAM de gradiente real (REQ-060) |
| RISK-009 (H-09) | Vazamento de dado pessoal | des-identificação PS3.15 (REQ-010, REQ-011) |
| RISK-010 (H-10) | Confiança mal calibrada | calibração preserva ordenação (REQ-061); escore não é probabilidade |

A multimodalidade acrescenta `RISK-011`…`RISK-020` (biomarcador sem pré-condição, zero-shot lido
como validado, medida assistida aceita sem revisão, licença não comercial violada, vazamento
treino/teste em evidência, roteamento errado, assinatura desvinculada, evidência publicada não
aplicável, acesso não autorizado) — ver o arquivo de risco, cuja coluna de requisitos é tirada do
`risk_controls` de cada `REQ-NNN.md`.

### 8.4 Documentos regulatórios

| # | Documento | Estado |
|---|---|---|
| 01 | [Uso pretendido](docs/regulatory/01-intended-use.md) | v0.2 — escopo multimodal de pesquisa, sem alegação clínica |
| 02 | [Classificação de segurança](docs/regulatory/02-software-safety-classification.md) | v0.1 — Classe C |
| 03 | [Plano de gestão de risco](docs/regulatory/03-risk-management-plan.md) | v0.2 |
| 04 | [Arquivo de risco](docs/regulatory/04-risk-file.md) | v0.3 — RISK-001…RISK-020, ligados aos requisitos; FMEA H-01…H-10 com testes |
| 05 | [Plano de desenvolvimento](docs/regulatory/05-development-plan.md) | v0.2 |
| 06 | [Requisitos](docs/regulatory/06-requirements/) | todos com teste verificador (portão de CI) |
| 07 | [Arquitetura](docs/regulatory/07-architecture.md) | v0.3 — conferida contra a árvore |
| 08 | [Lista SOUP](docs/regulatory/08-soup-list.md) | gerada por `scripts/soup.py` |
| 09 | [Plano de verificação](docs/regulatory/09-verification-plan.md) | v0.3 |
| 10 | [Plano de avaliação clínica](docs/regulatory/10-clinical-evaluation-plan.md) | v0.2 — critérios = protocolo; modelo próprio de RX pré-especificado |
| 11 | [Cibersegurança e LGPD](docs/regulatory/11-cybersecurity-lgpd.md) | v0.3 |
| 12 | [Rotulagem / IFU](docs/regulatory/12-labeling-ifu.md) | v0.2 — rascunho pt-BR, uso em pesquisa |
| 13 | [Rastreabilidade](docs/regulatory/13-traceability.md) | gerada por `scripts/trace.py` |
| 14 | [Controle de mudanças](docs/regulatory/14-change-control.md) | v0.2 |
| 15 | [Vigilância pós-mercado](docs/regulatory/15-post-market-surveillance.md) | v0.2 — plano para G6 |
| 16 | Model cards e datasheets | `docs/model-cards/`, `docs/datasheets/` — por artefato |

### 8.5 Rastreabilidade imposta pelo CI

Requisitos em Markdown com frontmatter (`id`, `type`, `risk_controls`, `verified_by`), testes
marcados com `@pytest.mark.requirement("REQ-NNN")`, e `scripts/trace.py --check` quebra o build
se um requisito ficar sem teste, um teste citar requisito inexistente ou um requisito não
declarar controle de risco. **Uma matriz que pode divergir não vale nada; uma que bloqueia merge
é evidência.**

### 8.6 O portão do dinheiro (mantido)

Tudo em §8.4 custa tempo. G5 custa dinheiro que não está no orçamento: empresa, AFE, Responsável
Técnico, QMS ISO 13485 e certificação BPF, consultoria regulatória para dossiê classe III, taxas
de petição, estudo clínico com custos de centro. As ordens de grandeza do v1.0 (dezenas de
milhares de reais para QMS e para consultoria) continuam como estimativa a refazer com
orçamentos reais.

**Portanto: faça G0–G1 e toda a documentação agora, a custo zero; abra a empresa só quando G2
produzir um número de validação externa brasileira que justifique o gasto.** Um arquivo técnico
registrável sem empresa é ativo financiável. Uma empresa sem modelo validado é conta mensal.

---

## 9. Dados e acesso

### 9.1 Plano de acesso

| Fonte | Acesso | Uso | Estado |
|---|---|---|---|
| NIH ChestX-ray14 (split oficial de teste) | livre | avaliação externa de pesos sem NIH | em uso; manifest commitado |
| Open-i (Indiana) | livre; laudos CC BY-NC-ND 4.0 | avaliação externa RX | baixado |
| Shenzhen / Montgomery (NLM) | livre, termos próprios | TB | baixado |
| FracAtlas, GRAZPEDWRI-DX | CC BY 4.0 | fratura | baixado |
| CQ500 | CC BY-NC-SA 4.0 | hemorragia intracraniana (externo ao TotalSegmentator) | baixado (subconjunto) |
| CMMD | CC BY 4.0 (TCIA) | mamografia, externo ao mammoscreen | amostra baixada |
| BUS-BRA, BrEaST | CC BY 4.0 | lesão mamária em US | baixado |
| CAMUS | CC BY-NC-SA 4.0, só pesquisa não comercial | eco, FE por Simpson | amostra baixada |
| **CheXpert** (validação, rótulo por 3 radiologistas) | registro Stanford AIMI + Research Use Agreement | primeira medição contra consenso | carregador pronto; dado não baixado — ver [`docs/access/README.md`](docs/access/README.md) |
| **VinDr-CXR**, MIMIC-CXR | credenciamento PhysioNet (curso CITI "Data or Specimens Only Research" + referência) | teste externo com rótulo de radiologistas; pares MIMIC do TorchXRayVision | credenciamento pendente — depende do dono, é pessoal e intransferível |
| **Parceiros brasileiros** (InCor como porta natural) | protocolo CEP, dados des-identificados pelo controlador, acordo de uso | **G2** | a negociar |

### 9.2 Governança de dados (imposta pelo CI)

1. Nenhum pixel no git. Manifests com sha256 por arquivo são commitados.
2. Todo dataset ganha `docs/datasheets/<dataset>.md`: fonte, licença/DUA, proveniência do rótulo
   (NLP vs. adjudicado vs. histologia), perfil de des-identificação, vieses conhecidos.
3. **Splits por paciente, sempre**; bootstrap agrupado por paciente.
4. Espelho não oficial de dataset que exige DUA não é usado.

### 9.3 Dado brasileiro sob LGPD e CEP

Rodar modelos sobre o PACS de um hospital, mesmo retrospectivamente e des-identificado, é
pesquisa com seres humanos (Lei 14.874/2024 Art. 2 XXXIV) e exige protocolo aprovado em CEP. A
des-identificação é feita pelo **controlador** (o hospital) antes da saída, com o perfil DICOM
PS3.15 Annex E e as opções registradas no datasheet. Imagem com chave de reidentificação é dado
pseudonimizado — ainda dado pessoal sensível — e não sai do ambiente do parceiro.

### 9.4 Vazamento entre treino e teste — fatos verificados

- Os desafios SIIM-ACR (pneumotórax) e RSNA (pneumonia) são **anotações de imagens do NIH**
  (Filice et al. 2020, DOI 10.1007/s10278-019-00299-9; Shih et al. 2019, DOI 10.1148/ryai.2019180041).
  Nenhum modelo treinado em NIH pode ser avaliado neles como externo.
- O TorchXRayVision separa treino e teste por `GroupShuffleSplit` 80/20 aleatório por paciente
  (semente 0), não pelo split oficial do NIH. A reprodução desse split mostrou que cerca de 20,7
  mil das 25.596 imagens do teste oficial do NIH caem na parte de **treino** dos pesos `nih` e
  `all`. Logo, para o teste oficial do NIH, os únicos pares externos são `pc`, `chex`,
  `mimic_nb` e `mimic_ch`.

---

## 10. Verificação

Toda PR roda, em CPU: `ruff check` + `ruff format --check`, `mypy --strict`, pytest (unitário e
integração; testes que precisam de pesos reais pulam de forma limpa quando os pesos não estão
presentes, mas são executados onde estão), `scripts/trace.py --check`, `scripts/soup.py --check`
e `scripts/check_honesty.py`. A trava de regressão (`radiologyai check-regression`) compara cada
`metrics.json` novo com a linha de base e reprova queda de AUROC macro, perda de rótulos avaliados
ou dataset não externo ao treino; `.github/workflows/eval.yml` executa `radiologyai evaluate` sob
demanda com semente e manifest explícitos.

`scripts/check_honesty.py` é o guardião que mais importa. Quebra o build em: alegação numérica de
desempenho em Markdown sem proveniência na mesma linha; `np.random`/`random` fora da avaliação;
campo sha256 que não seja 64 hex minúsculos; `except ImportError` em `src/`; marcadores de
simulação; model card com campo de desempenho. **O problema do v1 nunca foi falta de
competência — foi que nada no sistema objetava quando um número era inventado.**

Plano completo: [`docs/regulatory/09-verification-plan.md`](docs/regulatory/09-verification-plan.md).

---

## 11. O que foi cortado ou adiado, e por quê

O documento de arquitetura legado (`src/medai_architecture_doc.md`, histórico `16fdd47`)
prometia uma v1.2 e uma v2.0; o código legado tinha módulos de *global deployment*.

| Item do roadmap antigo | Decisão | Motivo |
|---|---|---|
| Aprendizado federado | **cortado** | resolve um problema (dados que não podem sair do hospital) que só existe depois de G2 com vários parceiros; antes disso é complexidade sem dado |
| AutoML | **cortado** | o gargalo é rótulo confiável e validação externa, não busca de hiperparâmetro |
| Integração com robótica médica | **cortado** | fora de qualquer uso pretendido defensável |
| Realidade aumentada | **cortado** | idem |
| *Global deployment orchestration* | **cortado** | não há o que implantar; implantação vem depois de G5 |
| Análise em tempo real (*streaming*) | **cortado** | radiologia é exame armazenado; latência não é o problema |
| SHAP/LIME | **cortado** | só explicação derivada do modelo real e verificada por teste (REQ-060) |
| Modelos 3D nativos | **mantido** | via TotalSegmentator/nnU-Net e MONAI (§5.2–5.3) |
| DICOM SR nativo | **mantido** | TID 1500 no laudo preliminar (§6) |
| Integração com EHR | **reduzido** | FHIR `DiagnosticReport` preliminar; HL7 v2 só se um parceiro exigir |
| Dashboard de monitoramento | **adiado para G6** | sem uso real não há deriva a monitorar |
| Análise temporal de progressão | **adiado** | exige exames prévios pareados e protocolo próprio |
| Multi-tenancy | **adiado** | um parceiro por vez até G4 |
| Triagem / priorização / notificação | **fora de escopo** | reclassifica o dispositivo (Regra 11) e exige outro programa de evidência |

---

## 12. Cronograma honesto e estratégia

### 12.1 O que mudou desde o v1.0

O v1.0 supunha um médico sozinho, 8–12 h/semana. Parte da engenharia agora é feita por agentes,
e por isso G0 e G1 andam muito mais rápido do que o v1.0 previa. **Isso não acelera nada do que
importa a partir de G2:** credenciamento pessoal, aprovação em CEP, acordo com hospital, tempo de
leitores para MRMC, dinheiro para G5. Os prazos abaixo são estimativas de tempo decorrido, não
compromissos, e serão revistos a cada portão.

| Marco | Dependência real | Estimativa |
|---|---|---|
| G0 completo (inclui expurgo do histórico) | uma operação de `git filter-repo` coordenada | semanas |
| G1 — KPI publicado para todas as modalidades do §5 | CPU e disco; credenciamento PhysioNet para os pares MIMIC/VinDr | 2–4 meses |
| G2 — protocolo aprovado em CEP | parceiro institucional, pesquisador responsável vinculado | 6–12 meses a partir do contato |
| G2 — relatório de validação externa brasileira | dados des-identificados de ≥ 2 centros | 12–18 meses |
| G3 — MRMC | leitores recrutados, financiamento | depois de G2 |
| G4 — piloto silencioso | CEP + integração no hospital | depois de G3 |
| G5 — registro ANVISA de uma indicação | empresa, QMS, BPF, consultoria | anos; travado por dinheiro |

### 12.2 Tornar G2 financiável

A aritmética do v1.0 continua valendo: sozinho, G5 não é alcançável. A estratégia é chegar a G2
com o pacote que um hospital parceiro, um edital (FAPESP/FINEP) ou um cofundador precisa ver:

1. **Catálogo G1 público e reproduzível** — o KPI por modalidade, com cada A reproduzível de
   checkout limpo e cada B com citação conferida. É a prova de método.
2. **Protocolo pré-registrado** ([`docs/protocolo-evidencia.md`](docs/protocolo-evidencia.md)) e
   plano de avaliação clínica ([`docs/regulatory/10-clinical-evaluation-plan.md`](docs/regulatory/10-clinical-evaluation-plan.md))
   prontos para virar projeto de CEP.
3. **Dossiê documental em formato de QMS** (§8.4) — o que um comitê de ética e um investidor
   pedem primeiro.
4. **Uma indicação estreita para G2**, escolhida pelo KPI de G1: a condição com melhor evidência
   pública *e* relevância clínica no parceiro. Candidatas naturais pela formação do dono:
   ecocardiografia (FE assistida) e RX de tórax (cardiomegalia/derrame).

Os aceleradores reais continuam não sendo código: (a) parceria com radiologista para
adjudicação, (b) vínculo institucional para dados e ética — o InCor é a porta natural, (c)
financiamento que compre tempo de leitores e de regulatório.

---

## Apêndice — correspondência de seções v1.0 → v2.0

Código, testes e documentos citam seções da v1.0. Até serem atualizados, use esta tabela:

| Citação v1.0 | Onde está agora |
|---|---|
| §1, §1.x (diagnóstico do legado) | [`HONEST_STATUS.md`](HONEST_STATUS.md); resumo em §1 |
| §2 (escopo em série) | §0 e §5 (amplitude em D, nível por medição) |
| §3 (PyTorch + MONAI) | §7.1 |
| §4, §4.1 (arquitetura, `ModalityPlugin`) | §7.2 e [`docs/regulatory/07-architecture.md`](docs/regulatory/07-architecture.md) |
| §4.2 (regra de ouro) | §7.3 |
| §5 (fases 0–6) | §4 (portões G0–G6) e §5 (modalidades) |
| §6.1 (Classe C) | §8.1 |
| §6.2 (enquadramento brasileiro) | §8.2 |
| §6.3 (perigos H-01…H-10) | §8.3 e [`docs/regulatory/04-risk-file.md`](docs/regulatory/04-risk-file.md) |
| §6.4 (documentos) | §8.4 |
| §6.5 (rastreabilidade) | §8.5 |
| §6.6 (portão do dinheiro) | §8.6 |
| §7 (dados e orçamento) | §9 |
| §8, §8.2 (verificação, `check_honesty.py`) | §10 e [`docs/regulatory/09-verification-plan.md`](docs/regulatory/09-verification-plan.md) |
| §9 (cronograma) | §12 |

---

## Referências

Regulatórias (identificadores e datas conferidos na fonte em 2026-11-24):

- ANVISA RDC 657/2022 (SaMD); RDC 751/2022 (classificação, Regra 11); RDC 665/2022 (BPF);
  RDC 848/2024 (requisitos essenciais)
- Lei 13.709/2018 (LGPD); Lei 14.874/2024 e Decreto 12.651/2025 (ética em pesquisa);
  Res. CNS 466/2012, 738/2024; Resolução CFM nº 2.454/2026
- IEC 62304:2006+AMD1:2015 (Ed. 1.1); ISO 14971:2019; IEC 62366-1 Ed. 1.1; IEC 81001-5-1:2021;
  AAMI CR34971:2022; DICOM PS3.15 Annex E

Diretrizes de relato e desenho:

- STARD-AI — Sounderajah et al., Nat Med 2025, DOI 10.1038/s41591-025-03953-8
- CLAIM 2024 — Tejani et al., Radiol Artif Intell 2024, DOI 10.1148/ryai.240300
- TRIPOD+AI — Collins et al., BMJ 2024, DOI 10.1136/bmj-2023-078378
- DECIDE-AI — Vasey et al., Nat Med 2022, DOI 10.1038/s41591-022-01772-9
- PROBAST+AI — Moons et al., BMJ 2025, DOI 10.1136/bmj-2024-082505
- MRMC — Obuchowski & Bullen, Radiology 2022, DOI 10.1148/radiol.211593
- Tamanho amostral — Collins et al. 2016, DOI 10.1002/sim.6787; Vergouwe et al. 2005,
  DOI 10.1016/j.jclinepi.2004.06.017; Riley et al. 2021, DOI 10.1002/sim.9025

Modelos e dados: ver [`MODEL_LICENSES.md`](MODEL_LICENSES.md) e `docs/datasheets/`.
