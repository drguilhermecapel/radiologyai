# Datasheet — BrEaST-Lesions-USG (TCIA)

> Gerado a partir do `DATASET.json` registrado no momento do download (cache local do projeto,
> fora do git) e revisado à mão. Nenhum pixel é versionado. Os campos abaixo são **fatos
> registrados na obtenção**, citados como estavam; nenhum é métrica de desempenho.

## Uso no projeto

Candidato a teste externo de sonda de lesão mamária em US (BrEaST, TCIA).

## Registro de obtenção

| Campo | Valor |
|---|---|
| `name` | BrEaST-Lesions-USG (TCIA) |
| `source_urls` | ["https://www.cancerimagingarchive.net/collection/breast-lesions-usg/", "paper doi:10.1038/s41597-024-02984-z"] |
| `licence` | {"as_fetched": "TCIA collection page, Data Access table (Version 1, updated 2024/01/08): \"Images and segmentations ... PNG and ZIP Download (66.63mb) 256 256 522 CC BY 4.0\"; \"Clinical data ... XLSX Download (39.23kb) 256 CC BY 4.0\". No licence file inside the zip."} |
| `n_images` | 256 |
| `counts_per_label` | {"benign": 154, "malignant": 98, "normal": 4} |
| `birads_counts` | {"4c": 49, "4b": 46, "5": 46, "4a": 44, "3": 37, "2": 30, "1": 4} |
| `verification_counts` | {"confirmed by biopsy": 197, "confirmed by follow-up care": 55, "not applicable": 4} |
| `diagnosis_top` | {"not applicable": 59, "Invasive carcinoma of no special type (NST)": 54, "Fibroadenoma": 29, "Benign mammary dysplasia": 26, "Invasive lobular carcinoma": 12, "Fibrosclerosis": 10, "Invasive carcinoma of no special type (NST)&Ductal carcinoma in situ (DCIS)": 7, "Intraductal papilloma": 7, "Pseudoangiomatous stromal hyperplasia (PASH)": 5, "Ductal carcinoma in situ (DCIS)": 4} |
| `patient_id_note` | CaseID 1..256, one image per patient (page: "256 breast ultrasound scans collected from 256 patients and 266 benign and malignant segmented lesions") |
| `file_counts` | {"images": 256, "tumor_masks": 252, "other_masks": 14} |
| `clinical_columns` | ["CaseID", "Image_filename", "Mask_tumor_filename", "Mask_other_filename", "Pixel_size", "Age", "Tissue_composition", "Signs", "Symptoms", "Shape", "Margin", "Echogenicity", "Posterior_features", "Halo", "Calcifications", "Skin_thickening", "Interpretation", "BIRADS", "Verification", "Diagnosis", "Classification"] |
| `layout` | extracted/BrEaST-Lesions_USG-images_and_masks/caseNNN.png, caseNNN_tumor.png, caseNNN_otherK.png ; clinical_BrEaST-Lesions-USG_clinical_dat.csv and clinical_Data_Dictionary.csv are CSV exports of the two xlsx sheets (made with openpyxl 3.1.5) |
| `retrieved` | 2026-11-24 |
| `notes_origin` | written by us-medsam spike; every value computed from the downloaded files or copied from the fetched source |
| `patient_ids` | sim |
| arquivos registrados com sha256 | 2 |

## Des-identificação e governança

Dataset público já des-identificado pelo distribuidor. Splits sempre por paciente quando há
identificador; sem identificador, a medição declara o vazamento possível (protocolo §8).
Licença e termos acima prevalecem sobre qualquer resumo em outro documento.
