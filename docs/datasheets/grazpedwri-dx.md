# Datasheet — GRAZPEDWRI-DX

> Gerado a partir do `DATASET.json` registrado no momento do download (cache local do projeto,
> fora do git) e revisado à mão. Nenhum pixel é versionado. Os campos abaixo são **fatos
> registrados na obtenção**, citados como estavam; nenhum é métrica de desempenho.

## Uso no projeto

Treino/teste das sondas `xr-probe-grazpedwri-{cast,metal,osteopenia}`.

## Ressalva principal

Punho pediátrico; não generaliza para adulto.

## Registro de obtenção

| Campo | Valor |
|---|---|
| `name` | GRAZPEDWRI-DX |
| `fetched` | 2026-11-24 |
| `figshare_article` | 14825193 |
| `doi` | 10.6084/m9.figshare.14825193.v2 |
| `licence` | CC BY 4.0 |
| `licence_evidence` | figshare API /v2/articles/14825193 license = {"name": "CC BY 4.0"} |
| `csv_rows` | 20327 |
| `n_patients` | 6091 |
| `has_patient_ids` | True |
| `csv_columns` | {"filestem": "str", "patient_id": "int64", "study_number": "int64", "timehash": "int64", "gender": "str", "age": "float64", "laterality": "str", "projection": "int64", "initial_exam": "float64", "ao_classification": "str", "cast": "float64", "diagnosis_uncertain": "float64", "osteopenia": "float64", "fracture_visible": "float64", "metal": "float64", "pixel_spacing": "float64", "device_manufacturer": "str"} |
| `encoding` | binary tags are 1.0 or empty(NaN): fracture_visible, osteopenia, metal, cast, initial_exam, diagnosis_uncertain; projection 1/2/3; gender M/F/O; age in years (float) |
| `full_counts` | {"fracture_visible": 13550, "osteopenia": 2473, "metal": 708, "cast": 5776, "projection": {"2": 10148, "1": 10086, "3": 93}, "gender": {"M": 12040, "F": 8285, "O": 2}, "age": {"min": 0.2, "median": 11.2, "max": 19.0}} |
| `sample` | {"dir": "images_sample", "index": "sample_index.tsv", "rule": "numpy default_rng(0).choice over the 1500 patients whose images are all inside images_part3.zip, 400 patients, all their images", "n_patients": 400, "n_images": 1274, "fracture_visible": {"1": 791, "0": 483}, "pixel_format": "16-bit grayscale PNG (PIL mode I;16)"} |

## Des-identificação e governança

Dataset público já des-identificado pelo distribuidor. Splits sempre por paciente quando há
identificador; sem identificador, a medição declara o vazamento possível (protocolo §8).
Licença e termos acima prevalecem sobre qualquer resumo em outro documento.
