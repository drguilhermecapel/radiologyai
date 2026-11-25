# Datasheet — NLM Shenzhen Hospital CXR Set (China set)

> Gerado a partir do `DATASET.json` registrado no momento do download (cache local do projeto,
> fora do git) e revisado à mão. Nenhum pixel é versionado. Os campos abaixo são **fatos
> registrados na obtenção**, citados como estavam; nenhum é métrica de desempenho.

## Uso no projeto

Treino da sonda `xr-probe-tb` (RAD-DINO). Medição nele é **interna** (C).

## Ressalva principal

Os termos pedem para **não compartilhar fora do grupo de pesquisa**: nenhuma imagem entra no repositório nem em imagem de contêiner distribuída.

## Registro de obtenção

| Campo | Valor |
|---|---|
| `name` | NLM Shenzhen Hospital CXR Set (China set) |
| `fetched` | 2026-11-24 |
| `source_url` | https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Shenzhen-Hospital-CXR-Set/ |
| `n_files` | 5334 |
| `bytes` | 3798828496 |
| `manifest_sha256` | b25c397b4eea4d8cf7fd2ec4a243e1ba7ba0a4422c122fca12cdaed86b6cc31c |
| `n_images` | 662 |
| `image_dir` | raw/CXR_png |
| `labels` | {"encoding": "file name suffix CHNCXR_<case>_<0\|1>.png ; 0 = normal, 1 = abnormal (manifestation of tuberculosis)", "counts": {"1": 336, "0": 326}} |
| `clinical_readings` | raw/ClinicalReadings/<stem>.txt (free text: sex, age, reading); parsed into index.tsv (sex, age_years, reading) |
| `sex_counts` | {"M": 449, "F": 213} |
| `age_years` | {"min": 0.175, "median": 33.0, "max": 89.0} |
| `has_patient_ids` | False |
| `patient_note` | One PNG per case number (all case numbers unique); readme counts "cases"; no patient identifier and no explicit statement that one case = one patient. |
| `licence` | not stated (custom terms) |
| `terms_evidence` | {"file": "raw/NLM-ChinaCXRSet-ReadMe.docx", "text": "No formal licence. Readme 'Citation and Use': 'Please do not share the dataset outside of your research group/ organization, but forward new dataset requests to us.' Attribution to NLM/NIH (and Shenzhen No.3 People's Hospital) and citation of Jaeger 2014 (PMID 24108713) and Candemir 2014 (PMID 24239990) requested."} |
| `pixel_format` | {"P": 635, "RGB": 27} |

## Des-identificação e governança

Dataset público já des-identificado pelo distribuidor. Splits sempre por paciente quando há
identificador; sem identificador, a medição declara o vazamento possível (protocolo §8).
Licença e termos acima prevalecem sobre qualquer resumo em outro documento.
