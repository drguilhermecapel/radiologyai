# Datasheet — Annotated Ultrasound Liver images (Xu Yiming et al., Zenodo 7272660)

> Gerado a partir do `DATASET.json` registrado no momento do download (cache local do projeto,
> fora do git) e revisado à mão. Nenhum pixel é versionado. Os campos abaixo são **fatos
> registrados na obtenção**, citados como estavam; nenhum é métrica de desempenho.

## Uso no projeto

Treino/teste da sonda `us-probe-liver-lesion-malignancy`.

## Ressalva principal

Um centro.

## Registro de obtenção

| Campo | Valor |
|---|---|
| `name` | Annotated Ultrasound Liver images (Xu Yiming et al., Zenodo 7272660) |
| `source_urls` | ["https://zenodo.org/records/7272660", "doi:10.5281/zenodo.7272660"] |
| `md5_note` | local md5sum of all three zips matches the Zenodo checksums |
| `licence` | {"zenodo_record": "license id: cc-by-4.0", "zenodo_description": "\"This dataset despite being open to public access, is subject to copyright. Any use of data contained within this dataset must receive appropriate acknowledgement and credit, meaning you should contact the author of the published data and cite the publicaion and the URL.\"", "caveat": "The description (\"you should contact the author\") is stricter/vaguer than the CC BY 4.0 tag; no associated publication is named on the record."} |
| `counts_per_label` | {"benign": 200, "malignant": 435, "normal": 100} |
| `file_counts` | {"Benign": {"image": 200, "segmentation/mass": 200, "segmentation/outline": 200, "segmentation/liver": 200}, "Malignant": {"image": 435, "segmentation/mass": 435, "segmentation/outline": 435, "segmentation/liver": 432}, "Normal": {"image": 100, "segmentation/outline": 100, "segmentation/liver": 100}} |
| `patient_id_note` | files are numbered 1..N per class; no patient/exam identifier, no histology, no scanner metadata, no per-image diagnosis beyond the folder name |
| `layout` | extracted/{Benign,Malignant,Normal}/image/N.jpg ; segmentation/{outline,liver,mass}/N.json = polygon as list of [x,y] float pairs (mass only for Benign/Malignant; 3 Malignant images lack a liver polygon) |
| `retrieved` | 2026-11-24 |
| `notes_origin` | written by us-medsam spike; every value computed from the downloaded files or copied from the fetched source |
| `patient_ids` | não |
| arquivos registrados com sha256 | 3 |

## Des-identificação e governança

Dataset público já des-identificado pelo distribuidor. Splits sempre por paciente quando há
identificador; sem identificador, a medição declara o vazamento possível (protocolo §8).
Licença e termos acima prevalecem sobre qualquer resumo em outro documento.
