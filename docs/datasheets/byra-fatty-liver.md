# Datasheet — Dataset of B-mode fatty liver ultrasound images (Byra et al. 2018)

> Gerado a partir do `DATASET.json` registrado no momento do download (cache local do projeto,
> fora do git) e revisado à mão. Nenhum pixel é versionado. Os campos abaixo são **fatos
> registrados na obtenção**, citados como estavam; nenhum é métrica de desempenho.

## Uso no projeto

Treino/teste da sonda `us-probe-liver-steatosis`.

## Ressalva principal

Um centro, coorte bariátrica, 55 pacientes: no máximo **C**.

## Registro de obtenção

| Campo | Valor |
|---|---|
| `name` | Dataset of B-mode fatty liver ultrasound images (Byra et al. 2018) |
| `source_urls` | ["https://zenodo.org/records/1009146", "doi:10.5281/zenodo.1009146", "paper doi:10.1007/s11548-018-1843-2"] |
| `licence` | {"zenodo_record": "license id: cc-by-4.0", "zenodo_description": "\"Please refer to the above work if you use the dataset in your research.\""} |
| `n_patients` | 55 |
| `n_images` | 550 |
| `images_per_patient` | 10 |
| `image_shape_hw` | [434, 636] |
| `dtype` | uint8 |
| `counts_per_label` | {"patients_class0": 17, "patients_class1": 38, "images_class0": 170, "images_class1": 380} |
| `label_definition` | struct fields per patient: id, class, fat, images(10,434,636). Observed: class==1 exactly when fat>=5 (class 0: fat 0..4; class 1: fat 5..85). fat = biopsy steatosis percentage. |
| `patients_fat_ge30` | 19 |
| `patient_id_note` | field id (1..55); the 10 images per patient are consecutive frames of one cine sequence -> MUST group by patient |
| `layout` | single MATLAB v5 .mat readable with scipy.io.loadmat; data is a (1,55) struct array |
| `retrieved` | 2026-11-24 |
| `notes_origin` | written by us-medsam spike; every value computed from the downloaded files or copied from the fetched source |
| `patient_ids` | sim |
| arquivos registrados com sha256 | 1 |

## Des-identificação e governança

Dataset público já des-identificado pelo distribuidor. Splits sempre por paciente quando há
identificador; sem identificador, a medição declara o vazamento possível (protocolo §8).
Licença e termos acima prevalecem sobre qualquer resumo em outro documento.
