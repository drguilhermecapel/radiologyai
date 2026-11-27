# Datasheet — openi (Indiana University chest X-ray collection, NLMCXR)

> Gerado a partir do `DATASET.json` registrado no momento do download (cache local do projeto,
> fora do git) e revisado à mão. Nenhum pixel é versionado. Os campos abaixo são **fatos
> registrados na obtenção**, citados como estavam; nenhum é métrica de desempenho.

## Uso no projeto

Candidato a avaliação externa de RX de tórax (Indiana).

## Ressalva principal

Laudos CC BY-NC-ND 4.0: uso não comercial, sem obra derivada dos laudos.

## Registro de obtenção

| Campo | Valor |
|---|---|
| `name` | openi (Indiana University chest X-ray collection, NLMCXR) |
| `aux_files_from_torchxrayvision` | [{"path": "/home/user/radiologyai/.venv/lib/python3.11/site-packages/torchxrayvision/data/nlmcxr_dicom_metadata.csv.gz", "bytes": 955593, "sha256": "996c7d7a23b6bef51b1308dfa9671fb59b00e67c8928f7ff45cc002ca16ebf3e"}, {"path": "/home/user/radiologyai/.venv/lib/python3.11/site-packages/torchxrayvision/data/nlmcxr_tsne_pa.csv.gz", "bytes": 16976, "sha256": "98d66188265a15f00038a5c894adeeaafcd78dbf82f687308f1443793805a253"}] |
| `licence` | {"per_report_xml": "<licenseType>open-access</licenseType> <licenseURL>http://creativecommons.org/licenses/by-nc-nd/4.0/</licenseURL> <ccLicense>byncnd</ccLicense>", "faq_text_as_fetched": "Images from the Indiana University hospital network are distributed under the terms of the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License (  http://creativecommons.org/licenses/by-nc-nd/4.0/  ).", "faq_download_terms_as_fetched": "Where can I get the Chest X-ray images in Open-i ?   The chest x-ray images from the Indiana University hospital network are available here:  P … |
| `schema` | {"report_xml": "ecgen-radiology/<n>.xml, root <eCitation>: uId@id (CXR<n>), pmcId@id, licenseType/licenseURL/ccLicense, MedlineCitation/Article/Abstract/AbstractText@Label in {COMPARISON, INDICATION, FINDINGS, IMPRESSION}, MeSH/major* (manual coding 'Term/qualifier/...'; 'normal' or 'No Indexing' possible), MeSH/automatic* (machine-extracted terms), parentImage@id* (= PNG file stem) with figureId, caption, panel/url", "png": "png/<parentImage id>.png, RGB uint8 (3 identical-looking channels; convert('L')), width 512, height 419-726 in a 202-image sample", "openi_index.csv": "one row per (repor … |
| `counts` | {"reports_tgz": "/root/medai-cache/datasets/openi/NLMCXR_reports.tgz", "xml_reports": 3955, "reports_with_0_images": 104, "images_per_report_hist": {"0": 104, "1": 446, "2": 3208, "3": 181, "4": 15, "5": 1}, "image_rows": 7470, "unique_imageids": 7470, "reports_empty_findings": 530, "reports_empty_impression": 34, "reports_major_normal": 1391, "reports_no_automatic_tags": 1773, "abstract_labels": ["text_COMPARISON", "text_FINDINGS", "text_IMPRESSION", "text_INDICATION"], "top_major": {"normal": 1391, "Lung/hypoinflation": 245, "Lung/hyperdistention": 164, "Cardiomegaly": 157, "Cardiomegaly/mil … |
| `caveats` | ["CC BY-NC-ND 4.0: non-commercial, no derivatives; NLM asks not to share the dataset outside the research group.", "Reports are de-identified with XXXX tokens; no patient age; no patient id other than the report uid (one report = one study).", "No authoritative frontal/lateral flag: DICOM View Position is missing for 2,589 of 7,470 images and disagrees with the t-SNE label for 198; the torchxrayvision t-SNE file covers only 3,851 images (3,267 PA, 584 L).", "Labels are substring matches on machine-extracted MeSH/automatic terms: 'granuloma' also fires on 'calcified granuloma', 'mass' on 'mass  … |
| arquivos registrados com sha256 | 2 |

## Padrão de referência, contagens e uso (revisão 2026-11-27)

Contado em `openi_index.csv` (7470 linhas, uma por imagem; todas com PNG): `View Position` PA 3591,
AP 423, LL 717, LATERAL 119, RL 29, LARGE LATERAL 2, vazio 2589. Frontais PA/AP = **4014 imagens de
2561 `uid`**. O `uid` é o do laudo (CXR<n>): um laudo = um estudo; **não há id de paciente** — os
"2561 pacientes" da população em `plan.yaml` são, na verdade, 2561 laudos (avisar ao autor do plano).
Arquivos com sha256 (`SHA256SUMS`): `NLMCXR_png.tgz` `baf3abfe…44bdf2` (apagado após extração),
`NLMCXR_reports.tgz` `8fb6de7e…68667a`. Termos citados de `TERMS_openi_faq_extract.txt`: "distributed
under the terms of the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International
License" e "We request not to share the datasets outside of your research group/organization".

- **Rótulos:** casamento de substring em termos MeSH (`major` atribuído por curadores a partir do
  laudo; `automatic` extraído por máquina), via torchxrayvision; não é leitura adjudicada da imagem
  (`reference_standard: other` no plano).
- **Modelos treinados nele:** `xrv-densenet121-res224-all` (`training_data` inclui `openi`):
  vazamento `known` para esse peso.
- **Uso:** 70 itens `*@openi-frontal#<rótulo>` em `plan.yaml`, todos `status: pending`
  ("inferência do Open-i em andamento"); 138 frontais entraram na calibração do portão OOD
  (`predict_xr_gates.py`); laterais (LL/LATERAL) são a fonte "tórax lateral" dos portões.
- **Não conferido:** idade e sexo (coluna `Patient's Sex` vazia em todas as linhas); equipamento.

## Des-identificação e governança

Dataset público já des-identificado pelo distribuidor. Splits sempre por paciente quando há
identificador; sem identificador, a medição declara o vazamento possível (protocolo §8).
Licença e termos acima prevalecem sobre qualquer resumo em outro documento.
