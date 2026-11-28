# Datasheet — CBIS-DDSM (Curated Breast Imaging Subset of DDSM, TCIA)

> Escrito à mão a partir dos arquivos do cache local (`/root/medai-cache/datasets/cbis-ddsm-meta/`,
> `cbis-ddsm-sample/` e `/root/medai-cache/spikes/mammo/meta/cbis_*`, fora do git), do código que os
> consome (`scripts/predict_xr_gates.py`, `radiologyai.evidence.leakage`, manifesto `mammoscreen`) e
> da página oficial da TCIA (consultada em 2026-11-27). Não há `DATASET.json`. Nenhum pixel é
> versionado. O que não pôde ser conferido está marcado **não conferido**.

## Uso no projeto

**Não é conjunto de avaliação.** Foi usado no pré-treino do `ianpan/mammoscreen` (manifesto
`mammoscreen`, `training_data: [cbis-ddsm, rsna-screening-mammography]`): medir o mammoscreen nele é
vazamento `known` (linhagem `cbis-ddsm` → `ddsm` em `leakage.py`). Aqui serve só como fonte
"não tórax" (mamografia) na calibração dos portões de RX (`predict_xr_gates.py`) e como contexto
publicado (concordância das máscaras de massa, `evidence/published/context.yaml`,
`cbis_ddsm_mask_dice`). Nenhum item de `evidence/measured/plan.yaml` usa `cbis-ddsm`.

## Ressalva principal

CBIS-DDSM está no (pré-)treino do mammoscreen: **nunca** é externo para ele.

## Fonte e licença

| Campo | Valor |
|---|---|
| Fonte | TCIA, coleção CBIS-DDSM — https://www.cancerimagingarchive.net/collection/cbis-ddsm/ ; DOI 10.7937/K9/TCIA.2016.7O02S9CY |
| Citação | Sawyer-Lee R, Gimenez F, Hoogi A, Rubin D. *Curated Breast Imaging Subset of Digital Database for Screening Mammography (CBIS-DDSM)*. TCIA, 2016 |
| Licença | arquivo `LICENSE` no zip baixado: "The CBIS-DDSM collection is distributed under the CC BY 3.0 at http://creativecommons.org/licenses/by/3.0/ By downloading the data, you agree to abide by terms of this license." + *Data Usage Policy* da TCIA. `cbis_series.json`: `LicenseName` "Creative Commons Attribution 3.0 Unported License" |
| Descrição oficial (página TCIA) | 6671 sujeitos/6775 estudos/10239 imagens no DICOM, mas "there are only 1,566 actual participants"; "254 images were identified in which a mass was not clearly seen" (removidas); refinamento de ROI por *level set* "only for masses and not calcifications"; divisão "20% of the cases for testing" separadamente para massas e calcificações |
| Artigo de dados (Lee et al., Sci Data 2017) | **não conferido** (não localizado no PubMed nesta sessão) |

## O que o projeto baixou

| Arquivo | Linhas | Pacientes | Patologia | sha256 |
|---|---:|---:|---|---|
| `mass_case_description_train_set.csv` | 1318 | 691 | MALIGNANT 637, BENIGN 577, BENIGN_WITHOUT_CALLBACK 104 | `a4bcc1c32bfd040212737e6e8c304819334abe717cfefff56431dc50ae6db723` |
| `mass_case_description_test_set.csv` | 378 | 201 | MALIGNANT 147, BENIGN 194, BENIGN_WITHOUT_CALLBACK 37 | `60f0b4504ec024106bbdd1555030d220163ea9da0ea4688473a4c9c3e8ab0457` |
| `calc_case_description_train_set.csv` | 1546 | 602 | MALIGNANT 544, BENIGN 528, BENIGN_WITHOUT_CALLBACK 474 | `3c7b07e8202f66709f446b9695e96c5d17484a65fda6a24d0170a0c1b81d56e3` |
| `calc_case_description_test_set.csv` | 326 | 151 | MALIGNANT 129, BENIGN 130, BENIGN_WITHOUT_CALLBACK 67 | `03a2f72efb6ed29f417e07f4598950e6a1e02e6b11250ff516ebe7924ace3689` |

União dos quatro CSVs: 1566 `patient_id` distintos (bate com a página). Uma linha = uma anormalidade
(colunas `left or right breast`, `image view`, `assessment` BI-RADS, `pathology`, `subtlety`).

Imagens: `cbis-ddsm-sample/` tem **uma** série (`Calc-Training_P_01227_RIGHT_CC`, 1 DICOM + LICENSE).
`cbis_series.json` cataloga 6775 séries (não baixadas); `cbis_sample252_*.txt` listam 252 séries
candidatas que não foram baixadas para este cache. sha256 por DICOM: **não registrado**.

## Padrão de referência

Patologia verificada herdada do DDSM ("verified pathology information"; "pathologic diagnosis for
training data are also included", página TCIA). ROIs originais do DDSM indicam "a general position
of lesions, but not a precise segmentation"; quem as desenhou: **não conferido**.

## Vieses e limitações

- Filmes digitalizados (anos 1990), não mamografia digital atual; codec PVRG-JPEG e conversão para
  DICOM 16 bits pelo curador.
- Um participante tem vários `PatientID` no DICOM (um por série): agrupar por `patient_id` do CSV.
- `BENIGN_WITHOUT_CALLBACK` não é biópsia benigna no mesmo sentido de BENIGN.
- Máscaras de calcificação sem refinamento.

## Modelos treinados nele

`mammoscreen` (pré-treino). O manifesto `mammo-crop` declara só `rsna-screening-mammography`.

## Des-identificação e governança

Dataset público já des-identificado pelo distribuidor (TCIA). CC BY 3.0; nenhuma imagem é versionada.
