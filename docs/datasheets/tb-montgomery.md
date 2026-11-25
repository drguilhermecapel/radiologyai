# Datasheet — NLM Montgomery County CXR Set

> Gerado a partir do `DATASET.json` registrado no momento do download (cache local do projeto,
> fora do git) e revisado à mão. Nenhum pixel é versionado. Os campos abaixo são **fatos
> registrados na obtenção**, citados como estavam; nenhum é métrica de desempenho.

## Uso no projeto

Teste externo da sonda `xr-probe-tb` treinada em Shenzhen (centro e país diferentes).

## Ressalva principal

Mesmos termos da NLM: sem redistribuição.

## Registro de obtenção

| Campo | Valor |
|---|---|
| `name` | NLM Montgomery County CXR Set |
| `fetched` | 2026-11-24 |
| `source_url` | https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Montgomery-County-CXR-Set/MontgomerySet/ |
| `n_files` | 554 |
| `bytes` | 618580523 |
| `manifest_sha256` | afe66dd83a7d494cb95d1abc00045d98e96b9c2c8155b983f11f107b2352c655 |
| `n_images` | 138 |
| `image_dir` | raw/CXR_png |
| `labels` | {"encoding": "file name suffix MCUCXR_<case>_<0\|1>.png ; 0 = normal, 1 = abnormal (manifestation of tuberculosis)", "counts": {"0": 80, "1": 58}} |
| `clinical_readings` | raw/ClinicalReadings/<stem>.txt (free text: sex, age, reading); parsed into index.tsv (sex, age_years, reading) |
| `sex_counts` | {"F": 74, "M": 63, "O": 1} |
| `age_years` | {"min": 4.0, "median": 40.0, "max": 89.0} |
| `has_patient_ids` | False |
| `patient_note` | One PNG per case number (all case numbers unique); readme counts "cases"; no patient identifier and no explicit statement that one case = one patient. |
| `licence` | not stated (custom terms) |
| `terms_evidence` | {"file": "raw/NLM-MontgomeryCXRSet-ReadMe.pdf", "text": "No formal licence. Readme 'Citation and Use': 'Please do not share the dataset outside of your research group/ organization, but forward new dataset requests to us.' Attribution to NLM/NIH (and Shenzhen No.3 People's Hospital) and citation of Jaeger 2014 (PMID 24108713) and Candemir 2014 (PMID 24239990) requested."} |
| `pixel_format` | {"L": 138} |

## Des-identificação e governança

Dataset público já des-identificado pelo distribuidor. Splits sempre por paciente quando há
identificador; sem identificador, a medição declara o vazamento possível (protocolo §8).
Licença e termos acima prevalecem sobre qualquer resumo em outro documento.
