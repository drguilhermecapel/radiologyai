# Datasheet — FracAtlas

> Gerado a partir do `DATASET.json` registrado no momento do download (cache local do projeto,
> fora do git) e revisado à mão. Nenhum pixel é versionado. Os campos abaixo são **fatos
> registrados na obtenção**, citados como estavam; nenhum é métrica de desempenho.

## Uso no projeto

Treino/teste da sonda `xr-probe-fracture` (RAD-DINO).

## Ressalva principal

Sem identificador de paciente: validação agrupada por paciente impossível; qualquer medição aqui é no máximo **C** e declara `leakage: possible`.

## Registro de obtenção

| Campo | Valor |
|---|---|
| `name` | FracAtlas |
| `fetched` | 2026-11-24 |
| `figshare_article` | 22363012 |
| `doi` | 10.6084/m9.figshare.22363012.v7 |
| `version` | 7 |
| `source_url` | https://ndownloader.figshare.com/files/65518038 |
| `archive` | {"file": "FracAtlas.zip", "bytes": 338436460, "sha256": "b67ec2d290a022b3dcf47f78e9a37f7edcc80592c0571f439355bf00bd9f0e23", "md5_figshare": "fe9da2c7c285915ebee69dfdab8fd396 (matches local md5sum)"} |
| `licence` | CC BY 4.0 |
| `licence_evidence` | figshare API /v2/articles/22363012 license = {"name": "CC BY 4.0"}; description: "The dataset is licensed under a CC-BY 4.0 license." |
| `n_images` | 4083 |
| `labels` | {"column": "fractured", "counts": {"0": 3364, "1": 719}} |
| `csv` | raw/FracAtlas/dataset.csv |
| `csv_columns` | ["image_id", "hand", "leg", "hip", "shoulder", "mixed", "hardware", "multiscan", "fractured", "fracture_count", "frontal", "lateral", "oblique"] |
| `column_sums` | {"hand": 1538, "leg": 2273, "hip": 338, "shoulder": 349, "mixed": 398, "hardware": 99, "multiscan": 396, "fractured": 719, "fracture_count": 924, "frontal": 2503, "lateral": 1492, "oblique": 418} |
| `has_patient_ids` | False |
| `caveats` | ["no patient/study identifier in dataset.csv: patient-grouped CV impossible", "4085 jpg files on disk for 4083 csv rows: IMG0003375.jpg and IMG0003376.jpg exist in BOTH images/Fractured and images/Non_fractured", "87 JPEGs lack the end-of-image marker (59 of them raise \"image file is truncated\" in PIL), ALL in Non_fractured (IMG0004028..): a label-correlated artefact", "figshare description says 4,073 images; csv v7 has 4083 rows"] |

## Des-identificação e governança

Dataset público já des-identificado pelo distribuidor. Splits sempre por paciente quando há
identificador; sem identificador, a medição declara o vazamento possível (protocolo §8).
Licença e termos acima prevalecem sobre qualquer resumo em outro documento.
