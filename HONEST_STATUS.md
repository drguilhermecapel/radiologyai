# Estado real deste projeto

> **Leia isto antes de qualquer outro documento do repositório.**

## Resumo

**RadiologyAI / MedAI Radiologia é software de pesquisa. Não é um dispositivo médico.**

- **Não existe nenhum modelo treinado neste repositório.** Zero arquivos de pesos.
- **Todas as métricas de desempenho publicadas anteriormente neste repositório foram fabricadas.** Não correspondem a nenhuma medição.
- **Não houve validação clínica.** Nenhuma.
- **Não é aprovado, registrado ou notificado em nenhuma agência regulatória** (ANVISA, FDA, CE/MDR).
- **Não deve ser usado para tomar, informar ou influenciar qualquer decisão diagnóstica ou terapêutica sobre um paciente real.**

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

Estes são defeitos conhecidos do código legado (agora em processo de arquivamento). Estão documentados aqui porque quem usou ou leu este repositório precisa saber:

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

## Primeira medição real (2026-09-12)

Existe agora **um** número de desempenho neste repositório, e ele está lastreado:
AUROC macro **0,664** de um modelo de terceiros (PadChest) no split oficial de
teste do NIH ChestX-ray14 — `artifacts/eval/xrv-densenet121-pc__20260912T202549Z/metrics.json`.

É uma medição retrospectiva de algoritmo isolado, sobre rótulos minerados por
NLP, sem calibração. **Continua não havendo validação clínica.** O que mudou é
que a frase "nenhuma métrica foi medida" deixou de ser verdadeira — e a única
métrica existente é pior do que o README do v1 alegava (0,94) e pior do que a
expectativa registrada no ROADMAP (0,72–0,82).

## O que vem a seguir

Veja [`ROADMAP.md`](ROADMAP.md). O primeiro marco técnico é publicar **um** número de desempenho real, medido, reproduzível, com intervalo de confiança — e não publicar nenhum outro até lá.

---

*Última atualização: setembro de 2026.*
