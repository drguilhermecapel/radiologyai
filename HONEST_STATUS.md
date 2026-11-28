# Estado real deste projeto

> **Leia isto antes de qualquer outro documento do repositório.**

## Resumo

**RadiologyAI / MedAI Radiologia é software de pesquisa. Não é um dispositivo médico.**

- **Nenhum modelo de ponta a ponta foi treinado por este projeto.** O que roda são modelos abertos
  de terceiros, com pesos verificados por sha256, e sondas lineares (regressão logística) que o
  projeto ajustou sobre codificadores congelados de terceiros (RAD-DINO, BiomedCLIP).
- **Todas as métricas de desempenho publicadas pela versão legada (v1) deste repositório foram
  fabricadas.** Não correspondiam a nenhuma medição.
- **Medições com artefato no repositório:** as duas execuções de linha de base de 2026-09-12 em
  `artifacts/eval/` e **98 registros medidos sob o protocolo pré-registrado** em
  `evidence/measured/` (sondas do projeto, MedSAM2, mammoscreen e 82 registros dos modelos
  TorchXRayVision no NIH ChestX-ray14; lista completa, inclusive as reprovações, em "Medições sob
  o protocolo" abaixo). Os relatos de Colab de 2026-11-25 **não
  têm artefato** e não contam. Nenhuma métrica deve ser lida neste repositório sem o caminho do
  artefato que a sustenta.
- **Duas condições têm nível A, ambas em RX de tórax e com modelos de terceiros:** cardiomegalia
  (`xrv-densenet121-res224-pc`, teste oficial do NIH, rótulo NLP, AUROC 0,813 [0,783; 0,839]) e
  opacidade pulmonar (`xrv-densenet121-res224-chex`, subconjunto do NIH adjudicado por
  radiologistas, AUROC 0,872 [0,848; 0,894]). Nível A aqui significa só *validação externa
  retrospectiva de algoritmo isolado* em um conjunto público; não é validação clínica. O nível A
  de tuberculose que `levels.json` chegou a mostrar foi **retratado** (ver "Nível A retratado").
  **Nenhuma condição tem nível B.** O B de derrame pericárdico em TC anunciado em 2026-11-27 foi
  **retratado** em 2026-11-28: o artigo (Wilder-Smith et al., Diagnostics 2022) avalia um nnU-Net
  de um rótulo só (derrame pericárdico, 1 mm, treino 316), e os pesos que distribuímos
  (`totalseg-pleural-pericard-effusion`) são o Dataset315_thoraxCT — 3 rótulos (pulmão, derrame
  pleural, derrame pericárdico), 1,5 mm, 600 casos de treino (`dataset.json` conferido). Evidência
  de outro modelo não se transfere; o mesmo vale para os registros de derrame pleural de Sexauer
  et al. 2022. Os registros foram para o contexto com o motivo `pesos_diferentes`.
  A maior parte das condições de tórax medidas no NIH foi
  **reprovada** (detalhe abaixo).
- **Não houve validação clínica.** Nenhuma. Nenhum dado clínico de parceiro brasileiro foi
  analisado; o único dado brasileiro medido é o conjunto público BUS-BRA (US de mama), em medição
  retrospectiva de algoritmo isolado.
- **Não é aprovado, registrado ou notificado em nenhuma agência regulatória** (ANVISA, FDA,
  CE/MDR).
- **Não deve ser usado para tomar, informar ou influenciar qualquer decisão diagnóstica ou
  terapêutica sobre um paciente real.** Toda saída é um *achado sugerido para confirmação médica*.

## O que foi retratado

As versões anteriores do `README.md`, `MODELS_LICENSE.md` e `docs/USER_GUIDE.md` afirmavam:

| Alegação anterior | Realidade |
|---|---|
| "Acurácia Validada: 92.3%" | nenhum modelo foi treinado; nenhuma acurácia foi medida |
| "Alta Acurácia: >95% para condições críticas" | fabricado |
| "arquiteturas ensemble validadas clinicamente" | nenhuma validação clínica ocorreu |
| "Sistema validado sem viés detectado" | nenhuma análise de viés foi executada |
| "Pronto para Produção: Sistema validado" | fabricado |
| `model_registry.json`: `accuracy: 0.92, auc: 0.94` | referia-se a arquivos que nunca existiram; o campo `sha256_hash` não era sequer hexadecimal |

**Todas essas alegações são retratadas por este documento.**

## O que o código realmente fazia

Estes são defeitos conhecidos do código legado. O código legado foi **removido da árvore**
(commit `c25f686`) e continua recuperável no histórico git (`16fdd47`). Os defeitos estão
documentados aqui porque quem usou ou leu este repositório precisa saber:

1. **Sem pesos, a "predição" era uma heurística de OpenCV.** `medai_inference_system.py:1510` calculava escores de patologia com expressões como `fracture_score = min(0.3, pneumonia_score * 0.5)`.

2. **A explicabilidade era falsa.** `medai_explainability.py` nunca consultava o modelo. O "Grad-CAM" servido por `POST /api/v1/explain` era detecção de bordas `cv2.Canny`, borrada e com ruído gaussiano somado. Um mapa de saliência fabricado sobre uma radiografia real é especialmente enganoso, porque manufatura a aparência de um modelo raciocinando sobre anatomia.

3. **A API fabricava métricas de validação a cada requisição.** `medai_fastapi_server.py:340` criava `y_true = np.array([1])  # Mock ground truth` e devolvia o resultado ao cliente como `clinical_metrics`.

4. **O mapeamento de patologias do único caminho com pesos reais era clinicamente perigoso.** `torchxray_integration.py` reportava **pneumotórax como pneumonia** e **cardiomegalia e edema como "normal"**.

5. **Métricas de "validação clínica" eram geradas por `np.random`.** Por exemplo, `medai_advanced_clinical_validation.py:357`: `cv_scores = np.random.normal(base_accuracy, 0.02, n_folds)`.

6. **A anonimização DICOM era incompleta e não-determinística.** Cobria 7 tags (não é o perfil PS3.15 Annex E) e gerava pseudo-IDs com o `hash()` builtin do Python, que é salgado por processo — o mesmo paciente recebia um identificador diferente a cada execução.

7. **Os dados de exemplo eram desenhos de OpenCV com nomes de arquivos do NIH ChestX-ray14.** Não eram radiografias.

8. **Todo artefato real de treino registrava falha**: `success_rate: 0.0`, `mean_auc: NaN`, `AUC_mean: 0.5`, acurácia 0,3333 em 3 classes.

## Por que este documento existe

O repositório é público e assinado com o nome e o CRM de um médico. Alegar desempenho clínico de software diagnóstico não registrado é, além de cientificamente indefensável, potencialmente uma infração sanitária sob a RDC 751/2022 e a Lei 6.437/77.

Corrigir isso publicamente é o pré-requisito de tudo o mais.

## Linha de base de terceiros (2026-09-12)

Duas execuções do modelo de terceiros `torchxrayvision densenet121-res224-pc` (treinado só em
PadChest) sobre o split oficial de teste do NIH ChestX-ray14 (25.596 imagens, 2.797 pacientes),
variando apenas o carregador de imagem:

- AUROC macro 0,6636 — `artifacts/eval/xrv-densenet121-pc__20260912T215742Z/metrics.json`
- AUROC macro 0,6644 — `artifacts/eval/xrv-densenet121-pc__20260912T202549Z/metrics.json`

É uma medição retrospectiva de algoritmo isolado, sobre rótulos minerados por NLP, sem
calibração, e anterior à fábrica de evidência: não é registro de `evidence/measured/` e não entra
no cálculo de nível. Relatório completo em
[`reports/EVALUATION_nih-chestx-ray14_xrv-densenet121-pc__20260912T215742Z.md`](reports/EVALUATION_nih-chestx-ray14_xrv-densenet121-pc__20260912T215742Z.md).

## Medições sob o protocolo (`evidence/measured/`, conferidas em 2026-11-27)

Fonte: [`evidence/measured/ledger.yaml`](evidence/measured/ledger.yaml) (desfecho de cada item de
[`evidence/measured/plan.yaml`](evidence/measured/plan.yaml)) e o registro JSON de cada linha.
Protocolo `2026-11-v1` ([`docs/protocolo-evidencia.md`](docs/protocolo-evidencia.md)): AUROC
≥ 0,80 com limite inferior do IC ≥ 0,75, ou ICC ≥ 0,75, com ≥ 30 pacientes positivos e ≥ 30
negativos. "Teste" é a metade de teste do split por paciente; IC95% bootstrap por paciente.
Tudo abaixo é medição retrospectiva de algoritmo isolado em dado público; **nada aqui é
validação clínica**.

**Aprovados no protocolo — nenhum é externo limpo:**

| Condição / modelo | Dataset (tipo, vazamento) | Resultado | Por que não sustenta A |
|---|---|---|---|
| TB / `xr-probe-tb` | Shenzhen, hold-out interno (CV), `known` | AUROC 0,982 [0,969; 0,992], n=331 — `evidence/measured/tb-shenzhen/xr-probe-tb/tuberculosis.json` | treino e teste no mesmo conjunto; Shenzhen também usado na seleção do codificador; sem id de paciente |
| TB / `xr-probe-tb.trained-montgomery` | Shenzhen, externo ao treino, `known` pela seleção | AUROC 0,944 [0,919; 0,966], n=331 — `evidence/measured/tb-shenzhen/xr-probe-tb.trained-montgomery/tuberculosis.json` | Shenzhen inteiro escolheu o codificador; sonda de comparação **não distribuída** (ver "Nível A retratado") |
| Fratura / `xr-probe-fracture` | FracAtlas, hold-out interno, `known` | AUROC 0,857 [0,832; 0,881], n=989 imagens — `evidence/measured/fracatlas/xr-probe-fracture/fracture.json` | interno; dobras por imagem (sem id de paciente); 87 JPEGs truncados só na classe negativa; também refutado externamente (abaixo) |
| Lesão mamária maligna / `us-probe-breast-malignancy` | BUSI, interno, `known` | AUROC 0,910 [0,866; 0,949], n=323 — `evidence/measured/busi/us-probe-breast-malignancy/breast_lesion_malignant.json` | interno; BUSI sem id de paciente; licença da cópia não conferida; região = caixa da máscara de referência + 25% |
| Lesão mamária maligna / `us-probe-breast-malignancy` | BUS-BRA (Brasil), externo, `possible` | AUROC 0,860 [0,830; 0,888], n=934 — `evidence/measured/bus-bra/us-probe-breast-malignancy/breast_lesion_malignant.json` | sobreposição possível com o pré-treino do BiomedCLIP; região derivada da máscara de referência; manifesto só de pesquisa |
| FE reduzida / `medsam2-echo-lv` (caixa) | CAMUS, externo, `possible` | ICC 0,861 [0,769; 0,913], n=100 — `evidence/measured/camus/medsam2-echo-lv/lv_ef.json` | caixa do VE derivada da máscara de referência + 5 px e quadros ED/ES do CAMUS; sobreposição possível com o treino do MedSAM2; manifesto só de pesquisa |

**Reprovados:**

| Condição / modelo | Dataset (tipo, vazamento) | Resultado | Consequência |
|---|---|---|---|
| Hemorragia intracraniana / `totalseg-cerebral-bleed` (`totalseg-cerebral-bleed@cq500`) | CQ500 (Índia), externo, treino sem sobreposição; referência por maioria de 3 radiologistas | detecção na regra pré-especificada (volume ≥ 0,1 mL): sensibilidade 0,700 [0,550; 0,825], especificidade 0,975 [0,925; 1,000], 40 com e 40 sem HIC — `evidence/measured/cq500/totalseg-cerebral-bleed/intracranial_hemorrhage.json` | **refuta** `ct.intracranial_hemorrhage` (limite inferior da sensibilidade < 0,80). O modelo segmenta hemorragia intracerebral e a referência é HIC de qualquer subtipo, mas a perda não vem só daí: contagem descritiva (não pré-especificada) detectou 22 de 29 intraparenquimatosas, 6 de 10 subdurais, 12 de 14 subaracnoides e 5 de 5 intraventriculares; 1 exame sem volume contou como não detectado |
| Fratura / `xr-probe-fracture` | GRAZPEDWRI-DX (punho pediátrico), externo, treino sem sobreposição, seleção `known` | AUROC 0,690 [0,633; 0,743], n=632 imagens — `evidence/measured/grazpedwri-dx-sample/xr-probe-fracture/fracture.json` | **refuta** a condição pela regra do protocolo (reprovação externa sem vazamento pelo treino; o reuso na seleção não anula uma reprovação) |
| Lesão mamária maligna / `us-probe-breast-malignancy` | BrEaST-Lesions-USG (Polônia), externo, `possible` | AUROC 0,820 [0,740; 0,893], n=126 — `evidence/measured/breast-lesions-usg/us-probe-breast-malignancy/breast_lesion_malignant.json` | limite inferior do IC abaixo de 0,75 |
| Lesão hepática maligna / `us-probe-liver-lesion-malignancy` | Zenodo 7272660, interno, `known` | AUROC 0,811 [0,609; 0,848], n=362 — `evidence/measured/liver-lesion-us-7272660/us-probe-liver-lesion-malignancy/liver_lesion_malignant.json` | limite inferior do IC abaixo de 0,75 |
| FE reduzida / `medsam2-echo-lv` (sem prompt) | CAMUS, externo, `possible` | ICC 0,000 [−0,031; 0,017], n=100 — `evidence/measured/camus/medsam2-echo-lv/lv_ef.prompt_free.json` | sem a caixa do usuário a FE não concorda com a referência |
| Câncer de mama / `mammoscreen` (`mammoscreen@cmmd`) | CMMD (TCIA, biópsia), externo, treino sem sobreposição | AUROC 0,790 [0,715; 0,859], metade de teste n=156 mamas (75 malignas) — `evidence/measured/cmmd/mammoscreen/malignancy.json` | **refuta** `mg.breast_cancer` pela regra do protocolo (AUROC < 0,80 e limite inferior < 0,75); só mede benigno x maligno em lesões biopsiadas, não rastreamento |

**Não elegíveis (abaixo dos mínimos; não aprovam nem refutam):**

| Condição / modelo | Dataset | Motivo — registro |
|---|---|---|
| TB / `xr-probe-tb` | Montgomery (externo ao treino, seleção `known`) | 29 positivos < 30 — `evidence/measured/tb-montgomery/xr-probe-tb/tuberculosis.json` |
| Osteopenia / `xr-probe-grazpedwri-osteopenia` | GRAZPEDWRI-DX, hold-out 30% | 11 pacientes positivos — `evidence/measured/grazpedwri-dx-sample-test30/xr-probe-grazpedwri-osteopenia/osteopenia.json` |
| Metal / `xr-probe-grazpedwri-metal` | GRAZPEDWRI-DX, hold-out 30% | 1 paciente positivo — `evidence/measured/grazpedwri-dx-sample-test30/xr-probe-grazpedwri-metal/metal.json` |
| Gesso / `xr-probe-grazpedwri-cast` | GRAZPEDWRI-DX, hold-out 30% | 12 pacientes positivos — `evidence/measured/grazpedwri-dx-sample-test30/xr-probe-grazpedwri-cast/cast.json` |
| Esteatose / `us-probe-liver-steatosis` | Byra (55 pacientes, um centro) | 19 positivos e 9 negativos no teste — `evidence/measured/byra-fatty-liver/us-probe-liver-steatosis/liver_steatosis.json` |

### RX de tórax: TorchXRayVision no NIH ChestX-ray14 (82 registros)

Seis modelos TorchXRayVision (`pc`, `chex`, `mimic_nb`, `mimic_ch`, `all`, `nih`) no **teste oficial
do NIH** (25.596 imagens; rótulos minerados de laudos por NLP) e no **subconjunto adjudicado por
radiologistas** (Google, Majkowska et al. 2020; 1.962 imagens, 4 classes). Itens pré-especificados
por `scripts/xrv_nih_evidence.py plan` e commitados antes de qualquer métrica; desenho do protocolo
(metade de calibração e metade de teste por paciente, IC por bootstrap agrupado por paciente).

Como ler:

- **Aprovados sem vazamento pelo treino (sustentam A):** cardiomegalia com `pc`; opacidade
  pulmonar com `chex`, `mimic_ch` e `mimic_nb`.
- **A cardiomegalia é um A limítrofe, que depende do sorteio da metade de teste.** O registro
  (metade de teste, semente pré-registrada 20261124) dá AUROC 0,813. No **teste oficial completo**
  o mesmo modelo dá **0,796**, e na metade de calibração 0,773 — abaixo da barra de 0,80.
  Repetindo a divisão com 40 sementes, a AUROC da metade de teste varia de 0,763 a 0,820
  (mediana 0,795) e só um terço das sementes passaria. O mesmo 0,796 já constava da medição de
  setembro (`artifacts/eval/xrv-densenet121-pc__20260912T215742Z/metrics.json`). Quatro modelos sem
  vazamento foram testados para cardiomegalia e só `pc` passa. O subgrupo feminino tem limite
  inferior do IC de 0,749. O nível A é mantido porque é o que a regra pré-registrada dá (mudar a
  regra depois do resultado seria pior), mas ele **não** deve ser lido como evidência robusta.
  Por decisão do responsável, a decisão positiva/negativa deste trio ficou **suspensa** (REQ-193)
  e foi **reativada** após a segunda validação externa, no Open-i (0,874 [0,840; 0,905]; ver
  abaixo). Histórico em `evidence/decision_holds.yaml`. As decisões usam os pontos de operação do
  registro do NIH (maior `n_total` entre os registros A, regra do REQ-192).
  Próxima versão do protocolo (não retroativa): considerar exigir a barra também na amostra
  completa.
- **Opacidade pulmonar é estável:** AUROC 0,873 na amostra completa, 0,874 na calibração e 0,872
  no teste. O subconjunto do Google é enriquecido (~58% de opacidade): limiares, VPP e VPN não se
  transferem à prevalência clínica.
- **Escolha dos pesos `pc`:** feita em c80c22e (2026-09-09), antes de qualquer medição; não houve
  comparação entre modelos no NIH. A inferência do teste oficial foi feita em fragmentos ao longo
  de vários commits sem mudança funcional no código de inferência; o checkout limpo vale para o
  cálculo das métricas.
- **`all` e `nih` têm vazamento `known`:** foram treinados no NIH, e o split do TorchXRayVision não
  é o oficial, então parte das imagens de teste pode ter estado no treino. A regra pré-registrada
  põe aprovação externa com vazamento conhecido em **C**; para estes dois modelos esse C **não**
  equivale a hold-out limpo. Não muda o KPI por condição (as duas condições já estão em A por
  modelos sem vazamento).
- **Refutadas no KPI:** a coluna "Refutadas" conta a condição com ao menos um modelo refutado.
  Em `cxr` são 14: as 13 abaixo e a cardiomegalia (A com `pc`; refutada com `chex` e `mimic_*`).
- **Reprovados:** atelectasia, consolidação, derrame pleural, edema, enfisema, fibrose,
  espessamento pleural, infiltrado, massa, nódulo, pneumonia, pneumotórax e lesão pulmonar
  ficaram abaixo da barra (AUROC ≥ 0,80 com limite inferior do IC ≥ 0,75) em todos os modelos sem
  vazamento. Pneumotórax no conjunto adjudicado: 0,748 no melhor caso (`pc`).
- **Não elegíveis** (menos positivos que o mínimo do protocolo na metade de teste): hérnia (45
  positivos no total) e fratura no conjunto adjudicado.

| Condição | Modelo (`xrv-densenet121-res224-…`) | Conjunto | Vazamento | AUROC [IC 95%] | positivos/n (teste) | Protocolo | Registro |
|---|---|---|---|---|---|---|---|
| `cxr.atelectasis` | `all` | teste oficial (NLP) | known | 0,713 [0,692; 0,733] | 1672/12113 | reprovado | `evidence/measured/nih-cxr14/xrv-densenet121-res224-all/atelectasis.json` |
| `cxr.atelectasis` | `chex` | teste oficial (NLP) | none | 0,560 [0,534; 0,586] | 1672/12113 | reprovado | `evidence/measured/nih-cxr14/xrv-densenet121-res224-chex/atelectasis.json` |
| `cxr.atelectasis` | `mimic_ch` | teste oficial (NLP) | none | 0,604 [0,582; 0,627] | 1672/12113 | reprovado | `evidence/measured/nih-cxr14/xrv-densenet121-res224-mimic_ch/atelectasis.json` |
| `cxr.atelectasis` | `mimic_nb` | teste oficial (NLP) | none | 0,581 [0,555; 0,604] | 1672/12113 | reprovado | `evidence/measured/nih-cxr14/xrv-densenet121-res224-mimic_nb/atelectasis.json` |
| `cxr.atelectasis` | `nih` | teste oficial (NLP) | known | 0,664 [0,643; 0,687] | 1672/12113 | reprovado | `evidence/measured/nih-cxr14/xrv-densenet121-res224-nih/atelectasis.json` |
| `cxr.atelectasis` | `pc` | teste oficial (NLP) | none | 0,645 [0,624; 0,666] | 1672/12113 | reprovado | `evidence/measured/nih-cxr14/xrv-densenet121-res224-pc/atelectasis.json` |
| `cxr.cardiomegaly` | `all` | teste oficial (NLP) | known | 0,861 [0,835; 0,883] | 602/12176 | aprovado | `evidence/measured/nih-cxr14/xrv-densenet121-res224-all/cardiomegaly.json` |
| `cxr.cardiomegaly` | `chex` | teste oficial (NLP) | none | 0,664 [0,614; 0,707] | 602/12176 | reprovado | `evidence/measured/nih-cxr14/xrv-densenet121-res224-chex/cardiomegaly.json` |
| `cxr.cardiomegaly` | `mimic_ch` | teste oficial (NLP) | none | 0,636 [0,593; 0,674] | 602/12176 | reprovado | `evidence/measured/nih-cxr14/xrv-densenet121-res224-mimic_ch/cardiomegaly.json` |
| `cxr.cardiomegaly` | `mimic_nb` | teste oficial (NLP) | none | 0,669 [0,622; 0,709] | 602/12176 | reprovado | `evidence/measured/nih-cxr14/xrv-densenet121-res224-mimic_nb/cardiomegaly.json` |
| `cxr.cardiomegaly` | `nih` | teste oficial (NLP) | known | 0,819 [0,788; 0,845] | 602/12176 | aprovado | `evidence/measured/nih-cxr14/xrv-densenet121-res224-nih/cardiomegaly.json` |
| `cxr.cardiomegaly` | `pc` | teste oficial (NLP) | none | 0,813 [0,783; 0,839] | 602/12176 | aprovado | `evidence/measured/nih-cxr14/xrv-densenet121-res224-pc/cardiomegaly.json` |
| `cxr.consolidation` | `all` | teste oficial (NLP) | known | 0,703 [0,683; 0,723] | 891/12235 | reprovado | `evidence/measured/nih-cxr14/xrv-densenet121-res224-all/consolidation.json` |
| `cxr.consolidation` | `chex` | teste oficial (NLP) | none | 0,671 [0,650; 0,692] | 891/12235 | reprovado | `evidence/measured/nih-cxr14/xrv-densenet121-res224-chex/consolidation.json` |
| `cxr.consolidation` | `mimic_ch` | teste oficial (NLP) | none | 0,632 [0,608; 0,655] | 891/12235 | reprovado | `evidence/measured/nih-cxr14/xrv-densenet121-res224-mimic_ch/consolidation.json` |
| `cxr.consolidation` | `mimic_nb` | teste oficial (NLP) | none | 0,607 [0,576; 0,636] | 891/12235 | reprovado | `evidence/measured/nih-cxr14/xrv-densenet121-res224-mimic_nb/consolidation.json` |
| `cxr.consolidation` | `nih` | teste oficial (NLP) | known | 0,679 [0,657; 0,702] | 891/12235 | reprovado | `evidence/measured/nih-cxr14/xrv-densenet121-res224-nih/consolidation.json` |
| `cxr.consolidation` | `pc` | teste oficial (NLP) | none | 0,669 [0,648; 0,690] | 891/12235 | reprovado | `evidence/measured/nih-cxr14/xrv-densenet121-res224-pc/consolidation.json` |
| `cxr.emphysema` | `all` | teste oficial (NLP) | known | 0,608 [0,570; 0,646] | 511/12059 | reprovado | `evidence/measured/nih-cxr14/xrv-densenet121-res224-all/emphysema.json` |
| `cxr.emphysema` | `nih` | teste oficial (NLP) | known | 0,533 [0,486; 0,578] | 511/12059 | reprovado | `evidence/measured/nih-cxr14/xrv-densenet121-res224-nih/emphysema.json` |
| `cxr.emphysema` | `pc` | teste oficial (NLP) | none | 0,605 [0,561; 0,648] | 511/12059 | reprovado | `evidence/measured/nih-cxr14/xrv-densenet121-res224-pc/emphysema.json` |
| `cxr.fracture` | `all` | adjudicado (Google) | known | 0,568 [0,464; 0,660] | 36/946 | não elegível | `evidence/measured/nih-cxr14-google/xrv-densenet121-res224-all/fracture.json` |
| `cxr.fracture` | `chex` | adjudicado (Google) | none | 0,525 [0,409; 0,633] | 36/946 | não elegível | `evidence/measured/nih-cxr14-google/xrv-densenet121-res224-chex/fracture.json` |
| `cxr.fracture` | `mimic_ch` | adjudicado (Google) | none | 0,513 [0,428; 0,594] | 36/946 | não elegível | `evidence/measured/nih-cxr14-google/xrv-densenet121-res224-mimic_ch/fracture.json` |
| `cxr.fracture` | `mimic_nb` | adjudicado (Google) | none | 0,493 [0,395; 0,588] | 36/946 | não elegível | `evidence/measured/nih-cxr14-google/xrv-densenet121-res224-mimic_nb/fracture.json` |
| `cxr.fracture` | `pc` | adjudicado (Google) | none | 0,546 [0,433; 0,649] | 36/946 | não elegível | `evidence/measured/nih-cxr14-google/xrv-densenet121-res224-pc/fracture.json` |
| `cxr.hernia` | `all` | teste oficial (NLP) | known | 0,872 [0,777; 0,954] | 45/12043 | não elegível | `evidence/measured/nih-cxr14/xrv-densenet121-res224-all/hernia.json` |
| `cxr.hernia` | `nih` | teste oficial (NLP) | known | 0,727 [0,572; 0,868] | 45/12043 | não elegível | `evidence/measured/nih-cxr14/xrv-densenet121-res224-nih/hernia.json` |
| `cxr.hernia` | `pc` | teste oficial (NLP) | none | 0,829 [0,719; 0,915] | 45/12043 | não elegível | `evidence/measured/nih-cxr14/xrv-densenet121-res224-pc/hernia.json` |
| `cxr.infiltration` | `all` | teste oficial (NLP) | known | 0,668 [0,652; 0,683] | 2813/12222 | reprovado | `evidence/measured/nih-cxr14/xrv-densenet121-res224-all/infiltration.json` |
| `cxr.infiltration` | `nih` | teste oficial (NLP) | known | 0,640 [0,623; 0,656] | 2813/12222 | reprovado | `evidence/measured/nih-cxr14/xrv-densenet121-res224-nih/infiltration.json` |
| `cxr.infiltration` | `pc` | teste oficial (NLP) | none | 0,660 [0,644; 0,674] | 2813/12222 | reprovado | `evidence/measured/nih-cxr14/xrv-densenet121-res224-pc/infiltration.json` |
| `cxr.lung_lesion` | `all` | adjudicado (Google) | known | 0,687 [0,622; 0,744] | 149/954 | reprovado | `evidence/measured/nih-cxr14-google/xrv-densenet121-res224-all/lung_lesion.json` |
| `cxr.lung_lesion` | `chex` | adjudicado (Google) | none | 0,563 [0,507; 0,619] | 149/954 | reprovado | `evidence/measured/nih-cxr14-google/xrv-densenet121-res224-chex/lung_lesion.json` |
| `cxr.lung_lesion` | `mimic_ch` | adjudicado (Google) | none | 0,559 [0,505; 0,612] | 149/954 | reprovado | `evidence/measured/nih-cxr14-google/xrv-densenet121-res224-mimic_ch/lung_lesion.json` |
| `cxr.lung_lesion` | `mimic_nb` | adjudicado (Google) | none | 0,561 [0,506; 0,614] | 149/954 | reprovado | `evidence/measured/nih-cxr14-google/xrv-densenet121-res224-mimic_nb/lung_lesion.json` |
| `cxr.lung_mass` | `all` | teste oficial (NLP) | known | 0,714 [0,681; 0,744] | 864/12153 | reprovado | `evidence/measured/nih-cxr14/xrv-densenet121-res224-all/mass.json` |
| `cxr.lung_mass` | `nih` | teste oficial (NLP) | known | 0,679 [0,640; 0,713] | 864/12153 | reprovado | `evidence/measured/nih-cxr14/xrv-densenet121-res224-nih/mass.json` |
| `cxr.lung_mass` | `pc` | teste oficial (NLP) | none | 0,619 [0,579; 0,653] | 864/12153 | reprovado | `evidence/measured/nih-cxr14/xrv-densenet121-res224-pc/mass.json` |
| `cxr.lung_nodule` | `all` | teste oficial (NLP) | known | 0,630 [0,587; 0,668] | 777/12318 | reprovado | `evidence/measured/nih-cxr14/xrv-densenet121-res224-all/nodule.json` |
| `cxr.lung_nodule` | `nih` | teste oficial (NLP) | known | 0,603 [0,555; 0,647] | 777/12318 | reprovado | `evidence/measured/nih-cxr14/xrv-densenet121-res224-nih/nodule.json` |
| `cxr.lung_nodule` | `pc` | teste oficial (NLP) | none | 0,602 [0,559; 0,638] | 777/12318 | reprovado | `evidence/measured/nih-cxr14/xrv-densenet121-res224-pc/nodule.json` |
| `cxr.lung_opacity` | `all` | adjudicado (Google) | known | 0,925 [0,906; 0,941] | 535/949 | aprovado | `evidence/measured/nih-cxr14-google/xrv-densenet121-res224-all/lung_opacity.json` |
| `cxr.lung_opacity` | `chex` | adjudicado (Google) | none | 0,872 [0,848; 0,894] | 535/949 | aprovado | `evidence/measured/nih-cxr14-google/xrv-densenet121-res224-chex/lung_opacity.json` |
| `cxr.lung_opacity` | `mimic_ch` | adjudicado (Google) | none | 0,856 [0,830; 0,882] | 535/949 | aprovado | `evidence/measured/nih-cxr14-google/xrv-densenet121-res224-mimic_ch/lung_opacity.json` |
| `cxr.lung_opacity` | `mimic_nb` | adjudicado (Google) | none | 0,848 [0,819; 0,875] | 535/949 | aprovado | `evidence/measured/nih-cxr14-google/xrv-densenet121-res224-mimic_nb/lung_opacity.json` |
| `cxr.pleural_effusion` | `all` | teste oficial (NLP) | known | 0,789 [0,773; 0,804] | 2284/12204 | reprovado | `evidence/measured/nih-cxr14/xrv-densenet121-res224-all/pleural_effusion.json` |
| `cxr.pleural_effusion` | `chex` | teste oficial (NLP) | none | 0,667 [0,642; 0,691] | 2284/12204 | reprovado | `evidence/measured/nih-cxr14/xrv-densenet121-res224-chex/pleural_effusion.json` |
| `cxr.pleural_effusion` | `mimic_ch` | teste oficial (NLP) | none | 0,696 [0,675; 0,716] | 2284/12204 | reprovado | `evidence/measured/nih-cxr14/xrv-densenet121-res224-mimic_ch/pleural_effusion.json` |
| `cxr.pleural_effusion` | `mimic_nb` | teste oficial (NLP) | none | 0,690 [0,667; 0,712] | 2284/12204 | reprovado | `evidence/measured/nih-cxr14/xrv-densenet121-res224-mimic_nb/pleural_effusion.json` |
| `cxr.pleural_effusion` | `nih` | teste oficial (NLP) | known | 0,753 [0,737; 0,769] | 2284/12204 | reprovado | `evidence/measured/nih-cxr14/xrv-densenet121-res224-nih/pleural_effusion.json` |
| `cxr.pleural_effusion` | `pc` | teste oficial (NLP) | none | 0,750 [0,730; 0,768] | 2284/12204 | reprovado | `evidence/measured/nih-cxr14/xrv-densenet121-res224-pc/pleural_effusion.json` |
| `cxr.pleural_thickening` | `all` | teste oficial (NLP) | known | 0,680 [0,651; 0,710] | 479/11998 | reprovado | `evidence/measured/nih-cxr14/xrv-densenet121-res224-all/pleural_thickening.json` |
| `cxr.pleural_thickening` | `nih` | teste oficial (NLP) | known | 0,662 [0,630; 0,692] | 479/11998 | reprovado | `evidence/measured/nih-cxr14/xrv-densenet121-res224-nih/pleural_thickening.json` |
| `cxr.pleural_thickening` | `pc` | teste oficial (NLP) | none | 0,541 [0,503; 0,576] | 479/11998 | reprovado | `evidence/measured/nih-cxr14/xrv-densenet121-res224-pc/pleural_thickening.json` |
| `cxr.pneumonia` | `all` | teste oficial (NLP) | known | 0,644 [0,603; 0,679] | 262/12457 | reprovado | `evidence/measured/nih-cxr14/xrv-densenet121-res224-all/pneumonia.json` |
| `cxr.pneumonia` | `chex` | teste oficial (NLP) | none | 0,563 [0,524; 0,600] | 262/12457 | reprovado | `evidence/measured/nih-cxr14/xrv-densenet121-res224-chex/pneumonia.json` |
| `cxr.pneumonia` | `mimic_ch` | teste oficial (NLP) | none | 0,555 [0,518; 0,591] | 262/12457 | reprovado | `evidence/measured/nih-cxr14/xrv-densenet121-res224-mimic_ch/pneumonia.json` |
| `cxr.pneumonia` | `mimic_nb` | teste oficial (NLP) | none | 0,541 [0,504; 0,579] | 262/12457 | reprovado | `evidence/measured/nih-cxr14/xrv-densenet121-res224-mimic_nb/pneumonia.json` |
| `cxr.pneumonia` | `nih` | teste oficial (NLP) | known | 0,624 [0,589; 0,658] | 262/12457 | reprovado | `evidence/measured/nih-cxr14/xrv-densenet121-res224-nih/pneumonia.json` |
| `cxr.pneumonia` | `pc` | teste oficial (NLP) | none | 0,642 [0,607; 0,675] | 262/12457 | reprovado | `evidence/measured/nih-cxr14/xrv-densenet121-res224-pc/pneumonia.json` |
| `cxr.pneumothorax` | `all` | adjudicado (Google) | known | 0,738 [0,684; 0,789] | 99/946 | reprovado | `evidence/measured/nih-cxr14-google/xrv-densenet121-res224-all/pneumothorax.json` |
| `cxr.pneumothorax` | `all` | teste oficial (NLP) | known | 0,634 [0,595; 0,672] | 1197/12109 | reprovado | `evidence/measured/nih-cxr14/xrv-densenet121-res224-all/pneumothorax.json` |
| `cxr.pneumothorax` | `chex` | adjudicado (Google) | none | 0,694 [0,630; 0,753] | 99/946 | reprovado | `evidence/measured/nih-cxr14-google/xrv-densenet121-res224-chex/pneumothorax.json` |
| `cxr.pneumothorax` | `chex` | teste oficial (NLP) | none | 0,651 [0,617; 0,683] | 1197/12109 | reprovado | `evidence/measured/nih-cxr14/xrv-densenet121-res224-chex/pneumothorax.json` |
| `cxr.pneumothorax` | `mimic_ch` | adjudicado (Google) | none | 0,664 [0,600; 0,725] | 99/946 | reprovado | `evidence/measured/nih-cxr14-google/xrv-densenet121-res224-mimic_ch/pneumothorax.json` |
| `cxr.pneumothorax` | `mimic_ch` | teste oficial (NLP) | none | 0,588 [0,538; 0,634] | 1197/12109 | reprovado | `evidence/measured/nih-cxr14/xrv-densenet121-res224-mimic_ch/pneumothorax.json` |
| `cxr.pneumothorax` | `mimic_nb` | adjudicado (Google) | none | 0,713 [0,649; 0,769] | 99/946 | reprovado | `evidence/measured/nih-cxr14-google/xrv-densenet121-res224-mimic_nb/pneumothorax.json` |
| `cxr.pneumothorax` | `mimic_nb` | teste oficial (NLP) | none | 0,616 [0,568; 0,657] | 1197/12109 | reprovado | `evidence/measured/nih-cxr14/xrv-densenet121-res224-mimic_nb/pneumothorax.json` |
| `cxr.pneumothorax` | `nih` | adjudicado (Google) | known | 0,646 [0,581; 0,705] | 99/946 | reprovado | `evidence/measured/nih-cxr14-google/xrv-densenet121-res224-nih/pneumothorax.json` |
| `cxr.pneumothorax` | `nih` | teste oficial (NLP) | known | 0,579 [0,532; 0,625] | 1197/12109 | reprovado | `evidence/measured/nih-cxr14/xrv-densenet121-res224-nih/pneumothorax.json` |
| `cxr.pneumothorax` | `pc` | adjudicado (Google) | none | 0,748 [0,695; 0,797] | 99/946 | reprovado | `evidence/measured/nih-cxr14-google/xrv-densenet121-res224-pc/pneumothorax.json` |
| `cxr.pneumothorax` | `pc` | teste oficial (NLP) | none | 0,679 [0,645; 0,712] | 1197/12109 | reprovado | `evidence/measured/nih-cxr14/xrv-densenet121-res224-pc/pneumothorax.json` |
| `cxr.pulmonary_edema` | `all` | teste oficial (NLP) | known | 0,775 [0,746; 0,802] | 474/12255 | reprovado | `evidence/measured/nih-cxr14/xrv-densenet121-res224-all/pulmonary_edema.json` |
| `cxr.pulmonary_edema` | `chex` | teste oficial (NLP) | none | 0,705 [0,668; 0,740] | 474/12255 | reprovado | `evidence/measured/nih-cxr14/xrv-densenet121-res224-chex/pulmonary_edema.json` |
| `cxr.pulmonary_edema` | `mimic_ch` | teste oficial (NLP) | none | 0,696 [0,666; 0,726] | 474/12255 | reprovado | `evidence/measured/nih-cxr14/xrv-densenet121-res224-mimic_ch/pulmonary_edema.json` |
| `cxr.pulmonary_edema` | `mimic_nb` | teste oficial (NLP) | none | 0,738 [0,708; 0,768] | 474/12255 | reprovado | `evidence/measured/nih-cxr14/xrv-densenet121-res224-mimic_nb/pulmonary_edema.json` |
| `cxr.pulmonary_edema` | `nih` | teste oficial (NLP) | known | 0,739 [0,713; 0,764] | 474/12255 | reprovado | `evidence/measured/nih-cxr14/xrv-densenet121-res224-nih/pulmonary_edema.json` |
| `cxr.pulmonary_edema` | `pc` | teste oficial (NLP) | none | 0,751 [0,720; 0,781] | 474/12255 | reprovado | `evidence/measured/nih-cxr14/xrv-densenet121-res224-pc/pulmonary_edema.json` |
| `cxr.pulmonary_fibrosis` | `all` | teste oficial (NLP) | known | 0,660 [0,613; 0,702] | 207/12017 | reprovado | `evidence/measured/nih-cxr14/xrv-densenet121-res224-all/fibrosis.json` |
| `cxr.pulmonary_fibrosis` | `nih` | teste oficial (NLP) | known | 0,653 [0,599; 0,704] | 207/12017 | reprovado | `evidence/measured/nih-cxr14/xrv-densenet121-res224-nih/fibrosis.json` |
| `cxr.pulmonary_fibrosis` | `pc` | teste oficial (NLP) | none | 0,442 [0,379; 0,500] | 207/12017 | reprovado | `evidence/measured/nih-cxr14/xrv-densenet121-res224-pc/fibrosis.json` |

### RX de tórax: TorchXRayVision no Open-i (70 registros)

Seis modelos TorchXRayVision em **todas as frontais do Open-i** (Indiana University, NLMCXR):
4.014 imagens de 2.561 laudos. Rótulos: termos MeSH **automáticos** (extraídos por máquina do laudo) por
busca de substring, como no torchxrayvision — não o MeSH de curadores, nem leitura adjudicada da
imagem; 1957 frontais sem termo automático contam como negativas. Com o MeSH de curadores, a
AUROC de cardiomegalia na metade de teste muda pouco (`pc` 0,867; `nih` 0,826; conferido pela
revisão adversarial). A cardiomegalia é robusta à divisão no Open-i: `pc` 0,873 / 0,873 / 0,874 e
`nih` 0,825 / 0,825 / 0,826 (amostra completa / calibração / teste). O Open-i não publica id de paciente: divisão e bootstrap
são agrupados por laudo, e um paciente com mais de um laudo pode cair nas duas metades, o que
tende a otimizar a estimativa. Itens pré-especificados (`scripts/xrv_nih_evidence.py plan
--datasets openi-frontal`) e commitados antes da inferência.

Como ler:

- **Cardiomegalia confirmada externamente pela segunda vez:** `pc` 0,874 [0,840; 0,905] e `nih`
  0,825 [0,786; 0,862], ambos sem vazamento pelo treino. No NIH, `pc` tinha passado no limite
  (0,813 na metade de teste; 0,796 na amostra completa). Com isso o responsável retirou a
  suspensão de decisão do trio `pc` (REQ-193). O trio `nih` passou a A **só** pelo Open-i, com
  uma única validação externa, e decide com os pontos desse registro.
- **Opacidade pulmonar reprovada no Open-i, nos mesmos modelos que a sustentam em A:** `chex`
  0,601 [0,557; 0,643], `mimic_ch` 0,635, `mimic_nb` 0,639 (217 positivos na metade de teste).
  O A vem só do subconjunto do NIH adjudicado por radiologistas (0,848–0,872). Pela regra
  pré-registrada basta uma aprovação externa limpa, então o nível não muda, mas o sinal é
  **contraditório**: o rótulo do Open-i ("opacity" por substring no MeSH automático) é mais
  ruidoso e de outra definição que "Airspace opacity" adjudicado, e a diferença também pode ser
  de população. Não leia este A como robusto. Por decisão do responsável (2026-11-28), a decisão
  positiva/negativa da opacidade pulmonar está **suspensa** em `chex`, `mimic_ch`, `mimic_nb` e
  `all` (`evidence/decision_holds.yaml`, REQ-193): o produto mostra só o escore, com o motivo.
- **`all` treinou no Open-i** (vazamento `known`): as aprovações de cardiomegalia (0,867) e de
  derrame pleural (0,829) põem esses trios em **C** pela regra pré-registrada, mas esse C **não**
  é hold-out limpo — parte destas imagens esteve no treino.
- **Derrame pleural reprovado de novo nos modelos sem vazamento** (`pc` 0,798, `nih` 0,775,
  `mimic_nb` 0,739, `mimic_ch` 0,737, `chex` 0,723), como no NIH.
- **Fratura em RX de tórax refutada** (era não elegível no subconjunto do Google): AUROC de 0,50 a
  0,66 em todos os modelos, 44 imagens positivas na metade de teste; por isso a coluna "Refutadas"
  de `cxr` sobe para 15.
- **Não elegíveis** (positivos contados por laudo na metade de teste, mínimo 30): edema 27,
  pneumonia 25, nódulo 25, hérnia 19, espessamento pleural 10, pneumotórax 8, fibrose 7, massa 3
  (contagens do modelo `pc`); nada é concluído delas.

| Condição | Modelo (`xrv-densenet121-res224-…`) | Vazamento | AUROC [IC 95%] | positivos/n (teste) | Protocolo | Registro |
|---|---|---|---|---|---|---|
| `cxr.atelectasis` | `all` | known | 0,745 [0,708; 0,781] | 185/2026 | reprovado | `evidence/measured/openi/xrv-densenet121-res224-all/atelectasis.json` |
| `cxr.atelectasis` | `chex` | none | 0,647 [0,604; 0,685] | 185/2026 | reprovado | `evidence/measured/openi/xrv-densenet121-res224-chex/atelectasis.json` |
| `cxr.atelectasis` | `mimic_ch` | none | 0,671 [0,626; 0,713] | 185/2026 | reprovado | `evidence/measured/openi/xrv-densenet121-res224-mimic_ch/atelectasis.json` |
| `cxr.atelectasis` | `mimic_nb` | none | 0,672 [0,630; 0,711] | 185/2026 | reprovado | `evidence/measured/openi/xrv-densenet121-res224-mimic_nb/atelectasis.json` |
| `cxr.atelectasis` | `nih` | none | 0,725 [0,686; 0,764] | 185/2026 | reprovado | `evidence/measured/openi/xrv-densenet121-res224-nih/atelectasis.json` |
| `cxr.atelectasis` | `pc` | none | 0,721 [0,681; 0,760] | 185/2026 | reprovado | `evidence/measured/openi/xrv-densenet121-res224-pc/atelectasis.json` |
| `cxr.cardiomegaly` | `all` | known | 0,867 [0,840; 0,895] | 138/2027 | aprovado | `evidence/measured/openi/xrv-densenet121-res224-all/cardiomegaly.json` |
| `cxr.cardiomegaly` | `chex` | none | 0,779 [0,738; 0,817] | 138/2027 | reprovado | `evidence/measured/openi/xrv-densenet121-res224-chex/cardiomegaly.json` |
| `cxr.cardiomegaly` | `mimic_ch` | none | 0,715 [0,670; 0,757] | 138/2027 | reprovado | `evidence/measured/openi/xrv-densenet121-res224-mimic_ch/cardiomegaly.json` |
| `cxr.cardiomegaly` | `mimic_nb` | none | 0,726 [0,683; 0,767] | 138/2027 | reprovado | `evidence/measured/openi/xrv-densenet121-res224-mimic_nb/cardiomegaly.json` |
| `cxr.cardiomegaly` | `nih` | none | 0,825 [0,786; 0,862] | 138/2027 | aprovado | `evidence/measured/openi/xrv-densenet121-res224-nih/cardiomegaly.json` |
| `cxr.cardiomegaly` | `pc` | none | 0,874 [0,840; 0,905] | 138/2027 | aprovado | `evidence/measured/openi/xrv-densenet121-res224-pc/cardiomegaly.json` |
| `cxr.emphysema` | `all` | known | 0,687 [0,639; 0,735] | 52/2026 | reprovado | `evidence/measured/openi/xrv-densenet121-res224-all/emphysema.json` |
| `cxr.emphysema` | `nih` | none | 0,636 [0,573; 0,695] | 52/2026 | reprovado | `evidence/measured/openi/xrv-densenet121-res224-nih/emphysema.json` |
| `cxr.emphysema` | `pc` | none | 0,741 [0,665; 0,812] | 52/2026 | reprovado | `evidence/measured/openi/xrv-densenet121-res224-pc/emphysema.json` |
| `cxr.fracture` | `all` | known | 0,653 [0,579; 0,729] | 44/2026 | reprovado | `evidence/measured/openi/xrv-densenet121-res224-all/fracture.json` |
| `cxr.fracture` | `chex` | none | 0,533 [0,464; 0,603] | 44/2026 | reprovado | `evidence/measured/openi/xrv-densenet121-res224-chex/fracture.json` |
| `cxr.fracture` | `mimic_ch` | none | 0,503 [0,421; 0,595] | 44/2026 | reprovado | `evidence/measured/openi/xrv-densenet121-res224-mimic_ch/fracture.json` |
| `cxr.fracture` | `mimic_nb` | none | 0,583 [0,509; 0,666] | 44/2026 | reprovado | `evidence/measured/openi/xrv-densenet121-res224-mimic_nb/fracture.json` |
| `cxr.fracture` | `pc` | none | 0,655 [0,575; 0,730] | 44/2026 | reprovado | `evidence/measured/openi/xrv-densenet121-res224-pc/fracture.json` |
| `cxr.hernia` | `all` | known | 0,920 [0,874; 0,969] | 30/2026 | não elegível | `evidence/measured/openi/xrv-densenet121-res224-all/hernia.json` |
| `cxr.hernia` | `nih` | none | 0,675 [0,576; 0,799] | 30/2026 | não elegível | `evidence/measured/openi/xrv-densenet121-res224-nih/hernia.json` |
| `cxr.hernia` | `pc` | none | 0,827 [0,765; 0,905] | 30/2026 | não elegível | `evidence/measured/openi/xrv-densenet121-res224-pc/hernia.json` |
| `cxr.infiltration` | `all` | known | 0,690 [0,617; 0,760] | 60/2026 | reprovado | `evidence/measured/openi/xrv-densenet121-res224-all/infiltration.json` |
| `cxr.infiltration` | `nih` | none | 0,563 [0,495; 0,636] | 60/2026 | reprovado | `evidence/measured/openi/xrv-densenet121-res224-nih/infiltration.json` |
| `cxr.infiltration` | `pc` | none | 0,619 [0,560; 0,679] | 60/2026 | reprovado | `evidence/measured/openi/xrv-densenet121-res224-pc/infiltration.json` |
| `cxr.lung_lesion` | `all` | known | 0,502 [0,481; 0,522] | 2/2026 | não elegível | `evidence/measured/openi/xrv-densenet121-res224-all/lung_lesion.json` |
| `cxr.lung_lesion` | `chex` | none | 0,623 [0,603; 0,644] | 2/2026 | não elegível | `evidence/measured/openi/xrv-densenet121-res224-chex/lung_lesion.json` |
| `cxr.lung_lesion` | `mimic_ch` | none | 0,602 [0,589; 0,615] | 2/2026 | não elegível | `evidence/measured/openi/xrv-densenet121-res224-mimic_ch/lung_lesion.json` |
| `cxr.lung_lesion` | `mimic_nb` | none | 0,613 [0,597; 0,628] | 2/2026 | não elegível | `evidence/measured/openi/xrv-densenet121-res224-mimic_nb/lung_lesion.json` |
| `cxr.lung_mass` | `all` | known | 0,843 [0,666; 0,998] | 5/2025 | não elegível | `evidence/measured/openi/xrv-densenet121-res224-all/mass.json` |
| `cxr.lung_mass` | `nih` | none | 0,673 [0,521; 0,829] | 5/2025 | não elegível | `evidence/measured/openi/xrv-densenet121-res224-nih/mass.json` |
| `cxr.lung_mass` | `pc` | none | 0,612 [0,562; 0,725] | 5/2025 | não elegível | `evidence/measured/openi/xrv-densenet121-res224-pc/mass.json` |
| `cxr.lung_nodule` | `all` | known | 0,616 [0,515; 0,724] | 36/2028 | não elegível | `evidence/measured/openi/xrv-densenet121-res224-all/nodule.json` |
| `cxr.lung_nodule` | `nih` | none | 0,637 [0,542; 0,734] | 36/2028 | não elegível | `evidence/measured/openi/xrv-densenet121-res224-nih/nodule.json` |
| `cxr.lung_nodule` | `pc` | none | 0,591 [0,485; 0,691] | 36/2028 | não elegível | `evidence/measured/openi/xrv-densenet121-res224-pc/nodule.json` |
| `cxr.lung_opacity` | `all` | known | 0,622 [0,592; 0,651] | 217/2023 | reprovado | `evidence/measured/openi/xrv-densenet121-res224-all/lung_opacity.json` |
| `cxr.lung_opacity` | `chex` | none | 0,601 [0,557; 0,643] | 217/2023 | reprovado | `evidence/measured/openi/xrv-densenet121-res224-chex/lung_opacity.json` |
| `cxr.lung_opacity` | `mimic_ch` | none | 0,635 [0,594; 0,674] | 217/2023 | reprovado | `evidence/measured/openi/xrv-densenet121-res224-mimic_ch/lung_opacity.json` |
| `cxr.lung_opacity` | `mimic_nb` | none | 0,639 [0,600; 0,676] | 217/2023 | reprovado | `evidence/measured/openi/xrv-densenet121-res224-mimic_nb/lung_opacity.json` |
| `cxr.pleural_effusion` | `all` | known | 0,829 [0,782; 0,876] | 92/2027 | aprovado | `evidence/measured/openi/xrv-densenet121-res224-all/pleural_effusion.json` |
| `cxr.pleural_effusion` | `chex` | none | 0,723 [0,672; 0,771] | 92/2027 | reprovado | `evidence/measured/openi/xrv-densenet121-res224-chex/pleural_effusion.json` |
| `cxr.pleural_effusion` | `mimic_ch` | none | 0,737 [0,687; 0,789] | 92/2027 | reprovado | `evidence/measured/openi/xrv-densenet121-res224-mimic_ch/pleural_effusion.json` |
| `cxr.pleural_effusion` | `mimic_nb` | none | 0,739 [0,681; 0,796] | 92/2027 | reprovado | `evidence/measured/openi/xrv-densenet121-res224-mimic_nb/pleural_effusion.json` |
| `cxr.pleural_effusion` | `nih` | none | 0,775 [0,717; 0,831] | 92/2027 | reprovado | `evidence/measured/openi/xrv-densenet121-res224-nih/pleural_effusion.json` |
| `cxr.pleural_effusion` | `pc` | none | 0,798 [0,751; 0,843] | 92/2027 | reprovado | `evidence/measured/openi/xrv-densenet121-res224-pc/pleural_effusion.json` |
| `cxr.pleural_thickening` | `all` | known | 0,846 [0,744; 0,932] | 14/2026 | não elegível | `evidence/measured/openi/xrv-densenet121-res224-all/pleural_thickening.json` |
| `cxr.pleural_thickening` | `nih` | none | 0,613 [0,502; 0,754] | 14/2026 | não elegível | `evidence/measured/openi/xrv-densenet121-res224-nih/pleural_thickening.json` |
| `cxr.pleural_thickening` | `pc` | none | 0,703 [0,597; 0,835] | 14/2026 | não elegível | `evidence/measured/openi/xrv-densenet121-res224-pc/pleural_thickening.json` |
| `cxr.pneumonia` | `all` | known | 0,665 [0,594; 0,731] | 40/2028 | não elegível | `evidence/measured/openi/xrv-densenet121-res224-all/pneumonia.json` |
| `cxr.pneumonia` | `chex` | none | 0,538 [0,458; 0,617] | 40/2028 | não elegível | `evidence/measured/openi/xrv-densenet121-res224-chex/pneumonia.json` |
| `cxr.pneumonia` | `mimic_ch` | none | 0,604 [0,507; 0,698] | 40/2028 | não elegível | `evidence/measured/openi/xrv-densenet121-res224-mimic_ch/pneumonia.json` |
| `cxr.pneumonia` | `mimic_nb` | none | 0,585 [0,503; 0,668] | 40/2028 | não elegível | `evidence/measured/openi/xrv-densenet121-res224-mimic_nb/pneumonia.json` |
| `cxr.pneumonia` | `nih` | none | 0,532 [0,455; 0,619] | 40/2028 | não elegível | `evidence/measured/openi/xrv-densenet121-res224-nih/pneumonia.json` |
| `cxr.pneumonia` | `pc` | none | 0,659 [0,570; 0,745] | 40/2028 | não elegível | `evidence/measured/openi/xrv-densenet121-res224-pc/pneumonia.json` |
| `cxr.pneumothorax` | `all` | known | 0,648 [0,516; 0,745] | 12/2025 | não elegível | `evidence/measured/openi/xrv-densenet121-res224-all/pneumothorax.json` |
| `cxr.pneumothorax` | `chex` | none | 0,472 [0,269; 0,631] | 12/2025 | não elegível | `evidence/measured/openi/xrv-densenet121-res224-chex/pneumothorax.json` |
| `cxr.pneumothorax` | `mimic_ch` | none | 0,636 [0,449; 0,835] | 12/2025 | não elegível | `evidence/measured/openi/xrv-densenet121-res224-mimic_ch/pneumothorax.json` |
| `cxr.pneumothorax` | `mimic_nb` | none | 0,621 [0,423; 0,812] | 12/2025 | não elegível | `evidence/measured/openi/xrv-densenet121-res224-mimic_nb/pneumothorax.json` |
| `cxr.pneumothorax` | `nih` | none | 0,633 [0,507; 0,758] | 12/2025 | não elegível | `evidence/measured/openi/xrv-densenet121-res224-nih/pneumothorax.json` |
| `cxr.pneumothorax` | `pc` | none | 0,509 [0,309; 0,766] | 12/2025 | não elegível | `evidence/measured/openi/xrv-densenet121-res224-pc/pneumothorax.json` |
| `cxr.pulmonary_edema` | `all` | known | 0,838 [0,775; 0,891] | 45/2027 | não elegível | `evidence/measured/openi/xrv-densenet121-res224-all/pulmonary_edema.json` |
| `cxr.pulmonary_edema` | `chex` | none | 0,741 [0,686; 0,793] | 45/2027 | não elegível | `evidence/measured/openi/xrv-densenet121-res224-chex/pulmonary_edema.json` |
| `cxr.pulmonary_edema` | `mimic_ch` | none | 0,737 [0,663; 0,803] | 45/2027 | não elegível | `evidence/measured/openi/xrv-densenet121-res224-mimic_ch/pulmonary_edema.json` |
| `cxr.pulmonary_edema` | `mimic_nb` | none | 0,780 [0,701; 0,851] | 45/2027 | não elegível | `evidence/measured/openi/xrv-densenet121-res224-mimic_nb/pulmonary_edema.json` |
| `cxr.pulmonary_edema` | `nih` | none | 0,670 [0,600; 0,740] | 45/2027 | não elegível | `evidence/measured/openi/xrv-densenet121-res224-nih/pulmonary_edema.json` |
| `cxr.pulmonary_edema` | `pc` | none | 0,850 [0,805; 0,891] | 45/2027 | não elegível | `evidence/measured/openi/xrv-densenet121-res224-pc/pulmonary_edema.json` |
| `cxr.pulmonary_fibrosis` | `all` | known | 0,900 [0,836; 0,956] | 7/2026 | não elegível | `evidence/measured/openi/xrv-densenet121-res224-all/fibrosis.json` |
| `cxr.pulmonary_fibrosis` | `nih` | none | 0,906 [0,851; 0,961] | 7/2026 | não elegível | `evidence/measured/openi/xrv-densenet121-res224-nih/fibrosis.json` |
| `cxr.pulmonary_fibrosis` | `pc` | none | 0,744 [0,575; 0,893] | 7/2026 | não elegível | `evidence/measured/openi/xrv-densenet121-res224-pc/fibrosis.json` |

**Pendentes:** nenhum item de medição pendente. Itens
excluídos do plano (provas de caminho, calibração de portões, execuções de fumaça) e o motivo de
cada um estão no ledger.

### Níveis: o que `levels.json` diz hoje

O nível é **calculado** (`scripts/evidence_build_levels.py` → `src/radiologyai/catalog/levels.json`);
este documento só o transcreve. O KPI conta **apenas entradas `distributed: true`** (modelo com
manifesto que cobre os pesos medidos):

<!-- kpi-levels:inicio (gerado de src/radiologyai/catalog/levels.json) -->
| Modalidade (prefixo) | A | B | C | D | Refutadas |
|---|---:|---:|---:|---:|---:|
| `ct` | 0 | 0 | 0 | 1 | 1 |
| `cxr` | 2 | 0 | 2 | 15 | 15 |
| `echo` | 0 | 0 | 0 | 1 | 0 |
| `mg` | 0 | 0 | 0 | 2 | 1 |
| `mr` | 0 | 0 | 0 | 2 | 0 |
| `msk_xr` | 0 | 0 | 0 | 4 | 1 |
| `us` | 0 | 0 | 0 | 3 | 0 |
| **Todas** | 2 | 0 | 2 | 28 | 18 |
<!-- kpi-levels:fim -->

O catálogo tem 63 condições; 34 têm entrada de modelo distribuído em `levels.json`. Os ids de
modelo dos registros publicados em `evidence/published/` foram corrigidos para os ids dos
manifestos (antes 140 de 148 não casavam e ficavam fora da contagem); a ingestão agora falha se um
registro citar modelo sem manifesto.

**Níveis vigentes** (registros gerados a partir de um commit, sem código sujo): **A** para
cardiomegalia (`xrv-densenet121-res224-pc` no NIH e no Open-i; `xrv-densenet121-res224-nih` no
Open-i) e opacidade pulmonar (`xrv-densenet121-res224-chex`, `mimic_ch`, `mimic_nb`, no NIH); nenhum **B** (o de derrame pericárdico foi retratado); **C** para tuberculose / `xr-probe-tb` (hold-out interno) e
derrame pleural / `xrv-densenet121-res224-all` (aprovado no Open-i, onde o modelo treinou: não é
hold-out limpo);
fratura (osteoarticular e em RX de tórax), câncer de mama em mamografia e a maior parte das
condições de tórax medidas no NIH e no Open-i **refutados** (D);
`xr-probe-tb.trained-montgomery` fora da contagem (não distribuído); FE e lesão mamária mantidas
em **D** apesar de aprovarem (manifesto só de pesquisa e entrada derivada da referência). Esta
tabela muda quando `levels.json` mudar — o CI (`tests/unit/test_status_docs_levels.py`) falha se
ela divergir.

### Nível A retratado

O commit `23af5ca` gerou um `levels.json` com **nível A** para tuberculose
(`xr-probe-tb.trained-montgomery`, AUROC 0,944 em Shenzhen —
`evidence/measured/tb-shenzhen/xr-probe-tb.trained-montgomery/tuberculosis.json`) e C para FE
reduzida e lesão mamária. **O nível A está retratado**, por dois motivos independentes:

1. **Reuso na seleção.** O codificador RAD-DINO de todas as sondas de RX foi escolhido comparando
   AUROCs de transferência Montgomery ↔ Shenzhen calculadas no Shenzhen inteiro
   ([`evidence/selection/xr-encoder/selection.yaml`](evidence/selection/xr-encoder/selection.yaml),
   resultados em `evidence/selection/xr-encoder/compare.json`). O conjunto que escolheu o modelo
   não é teste externo: o vazamento é `known` e a medição sustenta no máximo C.
2. **Modelo não distribuído.** A sonda treinada em Montgomery é de comparação: não tem manifesto,
   seus tensores não foram persistidos e ela não roda no produto nem é reproduzível de checkout
   limpo. Ela fica fora de qualquer contagem.

Os níveis C de FE (`medsam2-echo-lv`) e de lesão mamária (`us-probe-breast-malignancy`) também
deixaram de valer: os dois manifestos declaram `visibility_default: research`, e as duas medições
usaram como entrada uma caixa derivada da máscara de referência (oráculo da caixa do usuário).

## Relatos sem artefato (conferidos em 2026-11-27)

Dois documentos incorporados de `main` relatam execuções em Google Colab de 2026-11-25:

- [`docs/evaluation/phase1-nih1000-baseline.md`](docs/evaluation/phase1-nih1000-baseline.md) —
  amostra de 1.000 pacientes do NIH ChestX-ray14, `xrv-densenet121-pc`, discriminação e
  comparação de calibração por temperatura e por Platt;
- [`docs/evaluation/phase1-nih-unseen-benchmark.md`](docs/evaluation/phase1-nih-unseen-benchmark.md) —
  outra amostra de 1.000 pacientes do NIH, `xrv-densenet121-pc` contra `densenet121-res224-all`,
  calibração de Platt e uma execução do portão de regressão.

**Nenhum número desses dois documentos tem artefato no repositório** (nem `metrics.json` em
`artifacts/eval/`, nem predições, nem o manifest da amostra; o próprio relato da coorte inédita
diz que o ZIP do Colab não foi recuperado). Por isso eles **não** são repetidos aqui, no README nem
no ROADMAP, e não sustentam nenhum nível de evidência. Além disso: o critério "≥ 0,70 no limite
inferior do IC" citado no relato NIH-1000 não tem registro de pré-especificação no repositório e
não é o do protocolo vigente; o "piloto de 64 exames" citado nos dois relatos também não tem
artefato; e o modelo `densenet121-res224-all` foi treinado com dados do NIH, de modo que a
comparação entre os dois modelos não é externa para `all` (o portão de regressão bloqueou isso,
corretamente). Para que esses resultados contem, a execução precisa ser refeita com
`radiologyai evaluate` (ou `.github/workflows/eval.yml`) e o artefato commitado.

## Pendências que ainda não foram resolvidas

- **`medai_security.db` continua no histórico git.** O arquivo SQLite do legado saiu da árvore,
  mas está nos commits `3f5ae26`, `c80c22e` e `0b89c95`. O expurgo com `git filter-repo` (e a
  rotação do que ele possa ter tocado) está pendente; até lá, trate seu conteúdo como exposto.
  O mesmo vale para segredos do legado em `medai_config.json` e `docker-compose.yml` no histórico.
- **Nenhum registro externo limpo aprovado.** Toda aprovação própria é interna, tem vazamento
  possível/conhecido ou usou entrada derivada da referência. *(Desatualizado: ver "Níveis
  vigentes" — há duas condições em A, medidas no NIH, e uma em B, por publicação.)*
- **Licença de um dos dados de treino não conferida.** A sonda de lesão mamária em US foi ajustada
  numa cópia não oficial do BUSI cuja licença foi inferida do artigo; até conferir na fonte (ou
  reajustar em BUS-BRA/BrEaST), ela não sustenta evidência; o manifesto é só de pesquisa e o
  catálogo a mantém em D — ver [`MODEL_LICENSES.md`](MODEL_LICENSES.md).
- **Uso comercial incerto na maioria dos pesos.** Vários pesos têm licença permissiva, mas foram
  treinados em dados com termos não comerciais ou não lidos (`commercial_use: unclear`); tratados
  como não comerciais até confirmação.
- **Nenhum dado clínico brasileiro de parceiro.** O BUS-BRA é público e retrospectivo; validar em
  população brasileira exige parceria institucional e aprovação em CEP (portão G2 do roadmap).

## O que vem a seguir

Veja [`ROADMAP.md`](ROADMAP.md) (v2.0): catálogo de condições graduado por evidência, portões de
evidência G0–G6 e amplitude multimodal com modelos abertos reais — cada condição entra como
exploratória e só sobe de nível por medição que o próprio software calcula, sob o protocolo
pré-registrado em [`docs/protocolo-evidencia.md`](docs/protocolo-evidencia.md).

---

*Última atualização: 27 de novembro de 2026.*
