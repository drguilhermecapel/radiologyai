# Datasheet — BUSI (Breast Ultrasound Images, Al-Dhabyani et al. 2020) - HF mirror MedOtter/BUSI

> Gerado a partir do `DATASET.json` registrado no momento do download (cache local do projeto,
> fora do git) e revisado à mão. Nenhum pixel é versionado. Os campos abaixo são **fatos
> registrados na obtenção**, citados como estavam; nenhum é métrica de desempenho.

## Uso no projeto

Sonda `us-probe-breast-malignancy` foi ajustada nesta cópia.

## Ressalva principal

**Espelho não oficial** (Hugging Face `MedOtter/BUSI`): licença inferida do artigo, não conferida na fonte dos dados — pendência em [`MODEL_LICENSES.md`](../../MODEL_LICENSES.md). Sem identificador de paciente e com duplicatas (carta de Pawłowska et al. 2023, DOI 10.1016/j.dib.2023.109247): **nunca** é conjunto externo.

## Registro de obtenção

| Campo | Valor |
|---|---|
| `name` | BUSI (Breast Ultrasound Images, Al-Dhabyani et al. 2020) - HF mirror MedOtter/BUSI |
| `source_urls` | ["https://huggingface.co/datasets/MedOtter/BUSI (revision a1c102bb4aba406bc685b6aa1152fe23f608f091)", "original paper doi:10.1016/j.dib.2019.104863"] |
| `licence` | {"as_fetched": "HF card YAML: \"license: cc-by-4.0\"; card text: \"**CC BY 4.0** Redistribution and commercial use are permitted under the terms below. Source of the terms: https://pmc.ncbi.nlm.nih.gov/articles/PMC6906728/ ... Primary source (Data in Brief 2020, Al-Dhabyani et al.) states \\\"open access article under the CC BY license (creativecommons.org/licenses/by/4.0/)\\\".\"", "caveat": "This is a third-party re-host. The licence statement is the mirror's inference from the licence of the Data in Brief ARTICLE; a licence file from the original data distributors was not fetched in this se … |
| `counts_per_label` | {"benign": 437, "malignant": 210, "normal": 133} |
| `n_images` | 780 |
| `multifocal_images` | 17 |
| `patient_id_note` | No patient/case identifier column; image_id is e.g. "benign (100)". Card says 600 patients for 780 images, so several images per patient with no way to group them. |
| `layout` | single parquet, columns: image{bytes,path} (PNG RGB), mask{bytes,path} (PNG L 0/255, multi-lesion masks OR-merged by the mirror), class_label, image_id, has_multifocal_lesion |
| `image_size_range_wh` | ["190x335 (min)", "1048x578 (max)"] |
| `known_issues` | Card cites Pawlowska et al. 2023: ~235 duplicated images, 8 lesions in both benign and malignant folders, ~70 axilla images, 295 images with overlaid annotations (calipers, doppler markers, text), >=7 with biopsy needle. Measured here: 1 exact byte-duplicate row (that pair spans two classes). |
| `retrieved` | 2026-11-24 |
| `notes_origin` | written by us-medsam spike; every value computed from the downloaded files or copied from the fetched source |
| `patient_ids` | não |
| arquivos registrados com sha256 | 2 |

## Des-identificação e governança

Dataset público já des-identificado pelo distribuidor. Splits sempre por paciente quando há
identificador; sem identificador, a medição declara o vazamento possível (protocolo §8).
Licença e termos acima prevalecem sobre qualquer resumo em outro documento.
