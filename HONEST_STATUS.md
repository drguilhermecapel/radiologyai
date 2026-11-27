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
  `artifacts/eval/` e **15 registros medidos sob o protocolo pré-registrado** em
  `evidence/measured/` (sondas do projeto e o MedSAM2, de terceiros; lista completa, inclusive
  as reprovações, em "Medições sob o protocolo" abaixo). Os relatos de Colab de 2026-11-25 **não
  têm artefato** e não contam. Nenhuma métrica deve ser lida neste repositório sem o caminho do
  artefato que a sustenta.
- **Nenhuma condição tem nível A ou B.** O nível A que `levels.json` chegou a mostrar para
  tuberculose foi **retratado** (ver "Nível A retratado"). Pelas regras atuais, a melhor evidência
  própria sustenta no máximo **C**; o único C esperado é o da sonda de TB (hold-out interno). As
  outras aprovações ficam em D porque os manifestos são só de pesquisa.
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
| Fratura / `xr-probe-fracture` | GRAZPEDWRI-DX (punho pediátrico), externo, treino sem sobreposição, seleção `known` | AUROC 0,690 [0,633; 0,743], n=632 imagens — `evidence/measured/grazpedwri-dx-sample/xr-probe-fracture/fracture.json` | **refuta** a condição pela regra do protocolo (reprovação externa sem vazamento pelo treino; o reuso na seleção não anula uma reprovação) |
| Lesão mamária maligna / `us-probe-breast-malignancy` | BrEaST-Lesions-USG (Polônia), externo, `possible` | AUROC 0,820 [0,740; 0,893], n=126 — `evidence/measured/breast-lesions-usg/us-probe-breast-malignancy/breast_lesion_malignant.json` | limite inferior do IC abaixo de 0,75 |
| Lesão hepática maligna / `us-probe-liver-lesion-malignancy` | Zenodo 7272660, interno, `known` | AUROC 0,811 [0,609; 0,848], n=362 — `evidence/measured/liver-lesion-us-7272660/us-probe-liver-lesion-malignancy/liver_lesion_malignant.json` | limite inferior do IC abaixo de 0,75 |
| FE reduzida / `medsam2-echo-lv` (sem prompt) | CAMUS, externo, `possible` | ICC 0,000 [−0,031; 0,017], n=100 — `evidence/measured/camus/medsam2-echo-lv/lv_ef.prompt_free.json` | sem a caixa do usuário a FE não concorda com a referência |

**Não elegíveis (abaixo dos mínimos; não aprovam nem refutam):**

| Condição / modelo | Dataset | Motivo — registro |
|---|---|---|
| TB / `xr-probe-tb` | Montgomery (externo ao treino, seleção `known`) | 29 positivos < 30 — `evidence/measured/tb-montgomery/xr-probe-tb/tuberculosis.json` |
| Osteopenia / `xr-probe-grazpedwri-osteopenia` | GRAZPEDWRI-DX, hold-out 30% | 11 pacientes positivos — `evidence/measured/grazpedwri-dx-sample-test30/xr-probe-grazpedwri-osteopenia/osteopenia.json` |
| Metal / `xr-probe-grazpedwri-metal` | GRAZPEDWRI-DX, hold-out 30% | 1 paciente positivo — `evidence/measured/grazpedwri-dx-sample-test30/xr-probe-grazpedwri-metal/metal.json` |
| Gesso / `xr-probe-grazpedwri-cast` | GRAZPEDWRI-DX, hold-out 30% | 12 pacientes positivos — `evidence/measured/grazpedwri-dx-sample-test30/xr-probe-grazpedwri-cast/cast.json` |
| Esteatose / `us-probe-liver-steatosis` | Byra (55 pacientes, um centro) | 19 positivos e 9 negativos no teste — `evidence/measured/byra-fatty-liver/us-probe-liver-steatosis/liver_steatosis.json` |

**Pendentes (inferência incompleta; nenhuma métrica é extrapolada de conjunto incompleto):**
`xrv-densenet121-res224-pc@nih-cxr14-test`, `xrv-densenet121-res224-chex@nih-cxr14-test`,
`xrv-densenet121-res224-mimic_nb@nih-cxr14-test`, `xrv-densenet121-res224-mimic_ch@nih-cxr14-test`,
`xrv-densenet121-res224-all@nih-cxr14-test` e `xrv-densenet121-res224-nih@nih-cxr14-test` (7 de 10
fragmentos do teste oficial do NIH; `all` e `nih` serão `known` mesmo completos),
`mammoscreen@cmmd` (80 de 300 mamas) e `totalseg-cerebral-bleed@cq500` (5 de 80 exames). Itens
excluídos do plano (provas de caminho, calibração de portões, execuções de fumaça) e o motivo de
cada um estão no ledger.

### Níveis: o que `levels.json` diz hoje

O nível é **calculado** (`scripts/evidence_build_levels.py` → `src/radiologyai/catalog/levels.json`);
este documento só o transcreve. O KPI conta **apenas entradas `distributed: true`** (modelo com
manifesto que cobre os pesos medidos):

<!-- kpi-levels:inicio (gerado de src/radiologyai/catalog/levels.json) -->
| Modalidade (prefixo) | A | B | C | D | Refutadas |
|---|---:|---:|---:|---:|---:|
| `cxr` | 0 | 0 | 0 | 1 | 0 |
| `echo` | 0 | 0 | 0 | 1 | 0 |
| `mg` | 0 | 0 | 0 | 2 | 0 |
| `mr` | 0 | 0 | 0 | 2 | 0 |
| `msk_xr` | 0 | 0 | 0 | 4 | 0 |
| `us` | 0 | 0 | 0 | 3 | 0 |
| **Todas** | 0 | 0 | 0 | 13 | 0 |
<!-- kpi-levels:fim -->

O catálogo tem 63 condições; 34 têm entrada em `levels.json` e 13 têm entrada de modelo
distribuído. As demais entradas são de ids de modelo sem manifesto (inclusive os ids dos
registros publicados em `evidence/published/`, que não casam com os manifestos) e ficam fora da
contagem.

**Estado transitório desta árvore:** os 15 registros acima foram regenerados com código ainda não
commitado e carregam `git_dirty`; o build de níveis os exclui do cálculo, por isso a tabela está
toda em D e sem refutação. A re-medição a partir de um commit é **pendente**. Aplicando as regras
atuais aos mesmos registros, o resultado esperado é: **C** só para tuberculose / `xr-probe-tb`
(hold-out interno); fratura **refutada** (D); `xr-probe-tb.trained-montgomery` fora da contagem
(não distribuído); FE e lesão mamária mantidas em **D** apesar de aprovarem (manifesto só de
pesquisa e entrada derivada da referência). Esta tabela será atualizada quando `levels.json`
mudar — o CI (`tests/unit/test_status_docs_levels.py`) falha se ela divergir.

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
- **Re-medição a partir de commit.** Os registros de `evidence/measured/` desta árvore carregam
  `git_dirty` e estão fora do cálculo de nível até serem re-medidos de um commit (ver acima).
- **Nenhum registro externo limpo aprovado.** Toda aprovação própria é interna, tem vazamento
  possível/conhecido ou usou entrada derivada da referência. Não há condição em A nem em B.
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
