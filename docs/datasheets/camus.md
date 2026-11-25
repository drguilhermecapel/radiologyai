# Datasheet — CAMUS (Cardiac Acquisitions for Multi-structure Ultrasound Segmentation)

> Gerado a partir do `DATASET.json` registrado no momento do download (cache local do projeto,
> fora do git) e revisado à mão. Nenhum pixel é versionado. Os campos abaixo são **fatos
> registrados na obtenção**, citados como estavam; nenhum é métrica de desempenho.

## Uso no projeto

Referência de volumes e FE para o traçado assistido MedSAM2 + Simpson (barra de ICC do protocolo).

## Ressalva principal

CC BY-NC-SA 4.0, só pesquisa não comercial. Não se sabe se pacientes do CAMUS entraram no treino do MedSAM2 (ver manifesto `medsam2-echo-lv`): medição declara `leakage: possible`.

## Registro de obtenção

| Campo | Valor |
|---|---|
| `name` | CAMUS (Cardiac Acquisitions for Multi-structure Ultrasound Segmentation) |
| `source` | https://humanheart-project.creatis.insa-lyon.fr/database/api/v1 (coleção CAMUS_public; pasta database_nifti 63fde55f73e9f004868fb7ac) |
| `licence` | CC BY-NC-SA 4.0 (LICENSE_TERMS.md); somente pesquisa científica não comercial |
| `citation` | Leclerc et al., IEEE TMI 38(9):2198-2210, 2019, doi:10.1109/TMI.2019.2900516 |
| `subset` | Info_*.cfg + ED/ES (imagem e máscara de referência) por corte; sem half_sequence |
| `n_files` | 5505 |
| `total_bytes` | 363174281 |

## Des-identificação e governança

Dataset público já des-identificado pelo distribuidor. Splits sempre por paciente quando há
identificador; sem identificador, a medição declara o vazamento possível (protocolo §8).
Licença e termos acima prevalecem sobre qualquer resumo em outro documento.
