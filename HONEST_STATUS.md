# Estado real deste projeto

> **Leia isto antes de qualquer outro documento do repositório.**

## Resumo

**RadiologyAI / MedAI Radiologia é software de pesquisa. Não é um dispositivo médico.**

- **Nenhum modelo de ponta a ponta foi treinado por este projeto.** O que roda são modelos abertos
  de terceiros, com pesos verificados por sha256, e sondas lineares (regressão logística) que o
  projeto ajustou sobre codificadores congelados de terceiros (RAD-DINO, BiomedCLIP). Nenhuma
  dessas sondas foi avaliada sob o protocolo pré-registrado até esta data.
- **Todas as métricas de desempenho publicadas pela versão legada (v1) deste repositório foram
  fabricadas.** Não correspondiam a nenhuma medição.
- **Existem hoje duas medições reais com artefato no repositório**, ambas do mesmo modelo de
  terceiros no mesmo dataset público (ver abaixo). Há também dois relatos de execuções em Colab
  **sem artefato commitado** (ver "Relatos sem artefato"), que não contam como medição. Nenhuma
  outra métrica deve ser lida neste repositório sem o caminho do artefato que a sustenta.
- **Não houve validação clínica.** Nenhuma. Nenhum dado de paciente brasileiro foi analisado.
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

## As medições reais que existem (2026-09-12)

Duas execuções do modelo de terceiros `torchxrayvision densenet121-res224-pc` (treinado só em
PadChest) sobre o split oficial de teste do NIH ChestX-ray14 (25.596 imagens, 2.797 pacientes),
variando apenas o carregador de imagem:

- AUROC macro 0,6636 — `artifacts/eval/xrv-densenet121-pc__20260912T215742Z/metrics.json`
- AUROC macro 0,6644 — `artifacts/eval/xrv-densenet121-pc__20260912T202549Z/metrics.json`

É uma medição retrospectiva de algoritmo isolado, sobre rótulos minerados por NLP, sem
calibração. **Continua não havendo validação clínica.** A única métrica existente é pior do que o
README do v1 alegava e pior do que a expectativa registrada antes de medir; relatório completo em
[`reports/EVALUATION_nih-chestx-ray14_xrv-densenet121-pc__20260912T215742Z.md`](reports/EVALUATION_nih-chestx-ray14_xrv-densenet121-pc__20260912T215742Z.md).

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
- **O catálogo de condições está vazio.** O esquema, a regra de nível A–D, o protocolo
  pré-registrado (`src/radiologyai/catalog/`) e a fábrica de evidência (`src/radiologyai/evidence/`)
  existem, mas nenhuma condição tem nível calculado e nenhum registro medido foi commitado.
- **Licença de um dos dados de treino não conferida.** A sonda de lesão mamária em US foi ajustada
  numa cópia não oficial do BUSI cuja licença foi inferida do artigo; até conferir na fonte (ou
  reajustar em BUS-BRA/BrEaST), ela não sustenta evidência — ver
  [`MODEL_LICENSES.md`](MODEL_LICENSES.md).
- **Uso comercial incerto na maioria dos pesos.** Vários pesos têm licença permissiva, mas foram
  treinados em dados com termos não comerciais ou não lidos (`commercial_use: unclear`); tratados
  como não comerciais até confirmação.
- **Nenhum dado brasileiro.** Validar em população brasileira exige parceria institucional e
  aprovação em CEP (portão G2 do roadmap).

## O que vem a seguir

Veja [`ROADMAP.md`](ROADMAP.md) (v2.0): catálogo de condições graduado por evidência, portões de
evidência G0–G6 e amplitude multimodal com modelos abertos reais — cada condição entra como
exploratória e só sobe de nível por medição que o próprio software calcula, sob o protocolo
pré-registrado em [`docs/protocolo-evidencia.md`](docs/protocolo-evidencia.md).

---

*Última atualização: 27 de novembro de 2026.*
