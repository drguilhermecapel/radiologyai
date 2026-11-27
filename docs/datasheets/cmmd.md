# Datasheet — CMMD (The Chinese Mammography Database, TCIA)

> Escrito à mão a partir dos arquivos do cache local (`/root/medai-cache/datasets/cmmd-sample/`,
> `cmmd-sample300/` e `/root/medai-cache/spikes/mammo/meta/`, fora do git), do código que os consome
> (`scripts/predict_mg_cmmd.py`, `scripts/predict_mg_sample.py`, `scripts/predict_xr_gates.py`), do
> plano de medição e da página oficial da TCIA (consultada em 2026-11-27). Não há `DATASET.json`.
> Nenhum pixel é versionado. O que não pôde ser conferido está marcado **não conferido**.

## Uso no projeto

Avaliação externa do `mammoscreen` (item `mammoscreen@cmmd`, `status: measure`, referência
`histopathology`): discriminação benigno x maligno por mama. Também: prova de caminho com 8
pacientes (`predict_mg_sample.py`, `cmmd-sample/`) e fonte "não tórax" na calibração dos portões
de RX (`predict_xr_gates.py`).

## Ressalva principal

CMMD só tem mamas **biopsiadas**: a medida é benigno x maligno entre lesões suspeitas, **não**
rastreamento. A amostra de 300 pacientes é balanceada por desenho; VPP/VPN não generalizam.

## Fonte e licença

| Campo | Valor |
|---|---|
| Fonte | TCIA, coleção CMMD — https://www.cancerimagingarchive.net/collection/cmmd/ ; DOI 10.7937/tcia.eqde-4b16 |
| Citação | Cui C, et al. (2021) *The Chinese Mammography Database (CMMD)* — TCIA (citação sugerida na página) |
| Licença | arquivo `LICENSE` dentro de cada zip baixado: "The CMMD collection is distributed under the CC BY 4.0 at https://creativecommons.org/licenses/by/4.0/ By downloading the data, you agree to abide by terms of this license." Mais a *Data Usage Policy* da TCIA (não reidentificar; citar o DOI). Metadados de série (`cmmd_series.json`): `LicenseName` "Creative Commons Attribution 4.0 International License" |
| Descrição oficial (página TCIA) | "3,728 mammographies from these 1,775 patients"; exames entre julho de 2012 e janeiro de 2016; GE Senographe DS; "biopsy confirmed type of benign or malignant tumors"; "For the D2-XXXX dataset, it is a dataset that only involves malignant tumors" |
| Hospital de origem | **não conferido** (a página não o nomeia) |

## O que o projeto baixou

| Item | Valor (contado no cache) |
|---|---|
| `CMMD_clinicaldata_revision.csv` | 1872 linhas (uma por mama rotulada), 1775 pacientes; sha256 `39bddf22a4d842619a0050d9c41e70efda71ec5ec67dc6e3b18635962751262e` |
| Rótulos no CSV clínico | Malignant 1316, Benign 556; LeftRight L 988, R 884; abnormality mass 1149, both 461, calcification 262; subtipo molecular preenchido em 749 linhas |
| Idade (CSV clínico) | min 17, mediana 46, máx 87 |
| `cmmd_series.json` | 1775 séries MG, 5202 imagens, 22.861.926.932 bytes (catálogo NBIA, não baixado por inteiro) |
| `cmmd_sample300.txt` | 300 pacientes (150 Benign, 150 Malignant; 219 D1, 81 D2); 790 imagens, 3.470.585.340 bytes |
| `cmmd-sample300/` | 300 pastas, 790 DICOMs (bate com a lista) |
| Mamas rotuladas na amostra | 315 (Benign 165, Malignant 150) |
| `cmmd-sample/` | 8 pacientes, 18 DICOMs (lista `cmmd_sample8.txt`) |
| sha256 por DICOM | **não registrado** |
| Procedimento de sorteio dos 300 | **não conferido** (a lista existe; o código que a gerou não está no repositório) |

## Padrão de referência

Biópsia (histopatologia), segundo a página da TCIA. No projeto: `y_true` = 1 se a biópsia daquela
mama é maligna; a mama contralateral sem rótulo não entra (em D2 ela não é biópsia benigna).
Unidade = mama (CC + MLO). A lateralidade da mama vem do DICOM (REQ-141) e o rótulo do CSV clínico
(`LeftRight`).

## Vieses e limitações

- População chinesa, um fabricante (GE); um tempo só, sem seguimento: não valida o Mirai.
- Só lesões biopsiadas (espectro enriquecido); D2 contém apenas malignos.
- Prevalência da amostra (~48% maligno por mama) fixada pelo desenho.
- Medido: n = 156 mamas de teste (75 positivas) no registro `evidence/measured/cmmd/mammoscreen/malignancy.json`
  (as demais da amostra foram para a calibração, conforme o protocolo).

## Modelos treinados nele

Nenhum manifesto declara `cmmd` em `training_data`. O `mammoscreen` declara `cbis-ddsm` e
`rsna-screening-mammography` (o item do plano usa a chave `rsna-screening-mammo`); vazamento
registrado: `none`.

## Des-identificação e governança

Dataset público já des-identificado pelo distribuidor (TCIA). CC BY 4.0 permite redistribuição com
atribuição, mas nenhuma imagem é versionada aqui.
