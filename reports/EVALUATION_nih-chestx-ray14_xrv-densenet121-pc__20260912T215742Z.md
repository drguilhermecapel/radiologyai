# Avaliação — xrv-densenet121-pc em NIH ChestX-ray14

> **Gerado por `scripts/report.py` a partir de `artifacts/eval/xrv-densenet121-pc__20260912T215742Z/metrics.json`.** Não editar à mão.
> Todo número abaixo é reproduzível a partir desse artefato (artifacts/eval/).

## Proveniência

| Campo | Valor |
|---|---|
| run_id | `xrv-densenet121-pc__20260912T215742Z` |
| Executado em | 2026-09-12T21:57:42 UTC |
| Código (git_sha) | `990b2886e0e1321210c84e2fe6e1a4eba31b054c` |
| Modelo | xrv-densenet121-pc v1.0.0 — pesos `a9148ef62ae4e7a3…` |
| Treinado em | PadChest |
| Dataset | NIH ChestX-ray14, split oficial test_list.txt — manifest `39f31d789c3ccc1c…` |
| Imagens / pacientes | 25.596 / 2.797 |
| Externo ao treino | **sim** (externo) |
| Python / torch / xrv | 3.13.15 / 2.11.0+cu128 / 1.5.4 |
| Seed / bootstrap | 20260101 / 2000 |

## Resultado

**AUROC macro: 0.664** sobre 14 rótulos (fonte: `artifacts/eval/xrv-densenet121-pc__20260912T215742Z/metrics.json`).

| Achado | AUROC | IC 95% | n+ | Prevalência | Esp. @ sens. 90% | ECE |
|---|---:|---:|---:|---:|---:|---:|
| Hernia | 0.829 | [0.771, 0.882] | 86 | 0.003 | 0.419 | 0.201 |
| Cardiomegaly | 0.796 | [0.783, 0.808] | 1.069 | 0.042 | 0.484 | 0.327 |
| Effusion | 0.752 | [0.744, 0.759] | 4.658 | 0.182 | 0.426 | 0.372 |
| Edema | 0.744 | [0.730, 0.758] | 925 | 0.036 | 0.434 | 0.306 |
| Pneumothorax | 0.675 | [0.665, 0.686] | 2.665 | 0.104 | 0.276 | 0.331 |
| Pneumonia | 0.655 | [0.634, 0.677] | 555 | 0.022 | 0.279 | 0.507 |
| Atelectasis | 0.655 | [0.645, 0.664] | 3.279 | 0.128 | 0.295 | 0.342 |
| Consolidation | 0.654 | [0.643, 0.665] | 1.815 | 0.071 | 0.354 | 0.371 |
| Mass | 0.651 | [0.637, 0.663] | 1.748 | 0.068 | 0.224 | 0.356 |
| Infiltration | 0.651 | [0.643, 0.658] | 6.112 | 0.239 | 0.276 | 0.376 |
| Nodule | 0.611 | [0.597, 0.625] | 1.623 | 0.063 | 0.187 | 0.397 |
| Emphysema | 0.596 | [0.580, 0.611] | 1.093 | 0.043 | 0.231 | 0.258 |
| Pleural_Thickening | 0.574 | [0.558, 0.590] | 1.143 | 0.045 | 0.175 | 0.338 |
| Fibrosis | 0.448 | [0.421, 0.473] | 435 | 0.017 | 0.109 | 0.262 |

## Subgrupos (AUROC macro)

| Grupo | Valor | n | AUROC macro |
|---|---|---:|---:|
| sex | F | 10.714 | 0.667 |
| sex | M | 14.882 | 0.660 |
| age_band | 18-39 | 8.290 | 0.664 |
| age_band | 40-59 | 11.224 | 0.657 |
| age_band | 60-74 | 5.467 | 0.649 |
| age_band | 75+ | 611 | 0.627 |
| view_position | AP | 14.500 | 0.630 |
| view_position | PA | 11.096 | 0.692 |

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
