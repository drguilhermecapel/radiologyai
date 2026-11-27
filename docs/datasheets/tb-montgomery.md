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

## Padrão de referência e uso (revisão 2026-11-27)

- **Rótulo:** normal (0) x "manifestation of tuberculosis" (1), pelo sufixo do arquivo; o readme
  (`NLM-MontgomeryCXRSet-ReadMe.pdf`) diz que os dados vêm do "Montgomery County`s Tuberculosis
  screening program", matriz 4020 x 4892, espaçamento 0,0875 mm, 12 bits. Como o rótulo foi
  confirmado: **não conferido** — o readme não diz.
- `raw/ManualMask/{leftMask,rightMask}`: máscaras pulmonares "manually generated … under the
  supervision of a radiologist" (readme); não usadas pelo projeto. `montgomery_consensus_roi.csv`:
  origem **não conferida**.
- **Modelos treinados nele:** nenhum manifesto distribuído. A sonda de comparação
  `xr-probe-tb.trained-montgomery` (não distribuída) foi ajustada nele. É conjunto de **seleção** do
  codificador RAD-DINO.
- **Uso:** `xr-probe-tb@tb-montgomery` (teste externo da sonda de Shenzhen; vazamento `known` por
  seleção do codificador, logo nunca nível A). Fonte "tórax frontal" dos portões de RX.

## Des-identificação e governança

Dataset público já des-identificado pelo distribuidor. Splits sempre por paciente quando há
identificador; sem identificador, a medição declara o vazamento possível (protocolo §8).
Licença e termos acima prevalecem sobre qualquer resumo em outro documento.
