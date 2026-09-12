# Avaliação — xrv-densenet121-pc em NIH ChestX-ray14

> **Gerado por `scripts/report.py` a partir de `artifacts/eval/xrv-densenet121-pc__20260912T202549Z/metrics.json`.** Não editar à mão.
> Todo número abaixo é reproduzível a partir desse artefato (artifacts/eval/).

## Proveniência

| Campo | Valor |
|---|---|
| run_id | `xrv-densenet121-pc__20260912T202549Z` |
| Executado em | 2026-09-12T20:25:49 UTC |
| Código (git_sha) | `9d240ccd54b24c245c79142a9a59d51e9ff3a181` |
| Modelo | xrv-densenet121-pc v1.0.0 — pesos `a9148ef62ae4e7a3…` |
| Treinado em | PadChest |
| Dataset | NIH ChestX-ray14, split oficial test_list.txt — manifest `39f31d789c3ccc1c…` |
| Imagens / pacientes | 25.596 / 2.797 |
| Externo ao treino | **sim** (externo) |
| Python / torch / xrv | 3.13.15 / 2.11.0+cu128 / 1.5.4 |
| Seed / bootstrap | 20260101 / 2000 |

## Resultado

**AUROC macro: 0.664** sobre 14 rótulos (fonte: `artifacts/eval/xrv-densenet121-pc__20260912T202549Z/metrics.json`).

| Achado | AUROC | IC 95% | n+ | Prevalência | Esp. @ sens. 90% | ECE |
|---|---:|---:|---:|---:|---:|---:|
| Hernia | 0.836 | [0.780, 0.887] | 86 | 0.003 | 0.427 | 0.183 |
| Cardiomegaly | 0.796 | [0.783, 0.808] | 1.069 | 0.042 | 0.483 | 0.319 |
| Effusion | 0.752 | [0.745, 0.760] | 4.658 | 0.182 | 0.424 | 0.361 |
| Edema | 0.745 | [0.731, 0.759] | 925 | 0.036 | 0.439 | 0.303 |
| Pneumothorax | 0.674 | [0.664, 0.685] | 2.665 | 0.104 | 0.281 | 0.328 |
| Pneumonia | 0.656 | [0.634, 0.677] | 555 | 0.022 | 0.280 | 0.510 |
| Consolidation | 0.654 | [0.643, 0.665] | 1.815 | 0.071 | 0.353 | 0.372 |
| Atelectasis | 0.654 | [0.644, 0.663] | 3.279 | 0.128 | 0.296 | 0.341 |
| Infiltration | 0.650 | [0.642, 0.657] | 6.112 | 0.239 | 0.276 | 0.378 |
| Mass | 0.650 | [0.636, 0.662] | 1.748 | 0.068 | 0.227 | 0.352 |
| Nodule | 0.609 | [0.595, 0.623] | 1.623 | 0.063 | 0.189 | 0.397 |
| Emphysema | 0.591 | [0.575, 0.606] | 1.093 | 0.043 | 0.223 | 0.252 |
| Pleural_Thickening | 0.579 | [0.563, 0.595] | 1.143 | 0.045 | 0.189 | 0.325 |
| Fibrosis | 0.455 | [0.430, 0.480] | 435 | 0.017 | 0.115 | 0.255 |

## Subgrupos (AUROC macro)

| Grupo | Valor | n | AUROC macro |
|---|---|---:|---:|
| sex | F | 10.714 | 0.666 |
| sex | M | 14.882 | 0.662 |
| age_band | 18-39 | 8.290 | 0.665 |
| age_band | 40-59 | 11.224 | 0.657 |
| age_band | 60-74 | 5.467 | 0.650 |
| age_band | 75+ | 611 | 0.635 |
| view_position | AP | 14.500 | 0.631 |
| view_position | PA | 11.096 | 0.696 |

## Não avaliados

- `__untrained_14` — cabeça de saída não treinada neste conjunto de pesos; o valor produzido não tem significado e não é reportado
- `Fracture` — sem rótulo correspondente em NIH ChestX-ray14
- `__untrained_16` — cabeça de saída não treinada neste conjunto de pesos; o valor produzido não tem significado e não é reportado
- `__untrained_17` — cabeça de saída não treinada neste conjunto de pesos; o valor produzido não tem significado e não é reportado

## Limitações declaradas

- Rótulos minerados por NLP dos laudos, não adjudicados por radiologista; a acurácia estimada da rotulagem é de aproximadamente 90%.
- Apenas incidências frontais (PA e AP). Perfil não está representado.
- Sem sujeitos pediátricos identificados separadamente.
- 'Infiltration' do NIH não é diretamente comparável ao rótulo homônimo de outros datasets; a concordância entre definições é fraca.
- Esta é uma medição retrospectiva de desempenho de algoritmo isolado, NÃO uma validação clínica.
- Nenhuma calibração foi aplicada: os escores NÃO são probabilidade de doença.
- Medição retrospectiva de desempenho de algoritmo isolado. NÃO constitui validação clínica nem evidência de utilidade clínica.

---

Este relatório mede o desempenho isolado de um algoritmo de terceiros, retrospectivamente, sobre rótulos minerados por NLP. **Não é validação clínica** e não sustenta nenhuma alegação de uso clínico. Ver `HONEST_STATUS.md`.
