# Datasheet — BUS-BRA

> Gerado a partir do `DATASET.json` registrado no momento do download (cache local do projeto,
> fora do git) e revisado à mão. Nenhum pixel é versionado. Os campos abaixo são **fatos
> registrados na obtenção**, citados como estavam; nenhum é métrica de desempenho.

## Uso no projeto

Candidato a treino ou teste externo de sonda de lesão mamária em US; dado brasileiro (Rio de Janeiro).

## Ressalva principal

Tem identificador de caso (`Case`): splits por paciente possíveis.

## Registro de obtenção

| Campo | Valor |
|---|---|
| `name` | BUS-BRA |
| `source_urls` | ["https://zenodo.org/records/8231412", "doi:10.5281/zenodo.8231412", "paper doi:10.1002/mp.16812"] |
| `licence` | {"zenodo_record": "license id: cc-by-4.0", "in_archive_LICENSE_txt": "Copyright (c) 2023\n\nPermission is hereby granted, free of charge, to any person obtaining a copy of this dataset and associated documentation files (the \"Dataset\"), to deal in the Dataset without restriction, including without limitation the rights to use, copy, modify, merge, publish, and/or distribute copies of the Dataset, and to permit persons to whom the Dataset is furnished to do so, subject to the following conditions:\n\nAll documents and papers that uses the Dataset  are required to cite the following article:\n … |
| `n_images` | 1875 |
| `counts_per_label` | {"benign": 1268, "malignant": 607} |
| `birads_counts` | {"4": 693, "2": 562, "3": 463, "5": 157} |
| `histology_top` | {"fibroadenoma": 835, "invasive ductal carcinoma": 520, "cyst": 142, "fibrocystic changes": 106, "invasive lobular carcinoma": 42, "intraductal papilloma": 41, "sclerosing adenosis": 37, "hyperplasia": 31, "lipoma": 17, "phyllodes tumor": 13} |
| `device_counts` | {"GE Logiq 7 @10-14MHz": 902, "GE Logiq 5 @10-12MHz": 809, "Toshiba Aplio 300 @12-14 MHz": 139, "U-Systems @10-14MHz": 25} |
| `n_patients` | 1064 |
| `patient_id_note` | column Case; a case never mixes benign and malignant (checked). Official patient-level 5-fold and 10-fold splits in 5-fold-cv.csv / 10-fold-cv.csv. |
| `layout` | extracted/BUSBRA/{Images/bus_XXXX-{l,r,s}.png, Masks/mask_XXXX-*.png, bus_data.csv (ID,Case,Histology,Pathology,BIRADS,Device,Width,Height,Side,BBOX[x,y,w,h]), 5-fold-cv.csv, 10-fold-cv.csv, LICENSE.txt} |
| `retrieved` | 2026-11-24 |
| `notes_origin` | written by us-medsam spike; every value computed from the downloaded files or copied from the fetched source |
| `patient_ids` | sim |
| arquivos registrados com sha256 | 1 |

## Des-identificação e governança

Dataset público já des-identificado pelo distribuidor. Splits sempre por paciente quando há
identificador; sem identificador, a medição declara o vazamento possível (protocolo §8).
Licença e termos acima prevalecem sobre qualquer resumo em outro documento.
