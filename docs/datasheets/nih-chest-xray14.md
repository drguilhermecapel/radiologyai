# Datasheet — NIH ChestX-ray14

| Campo | Valor |
|---|---|
| **Nome** | NIH ChestX-ray14 (ChestX-ray8 estendido) |
| **Fonte** | NIH Clinical Center, EUA |
| **Publicação** | Wang X, Peng Y, Lu L, et al. *ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases*. CVPR 2017 |
| **Acesso** | Livre, sem registro — https://nihcc.app.box.com/v/ChestXray-NIHCC |
| **Tamanho** | ~42 GB comprimido (12 tarballs) |
| **Licença** | Uso irrestrito para pesquisa, conforme termos do NIH |

## Composição

| | Imagens | Pacientes |
|---|---|---|
| Total | 112.120 | 30.805 |
| `train_val_list.txt` | 86.524 | 28.008 |
| `test_list.txt` | 25.596 | 2.797 |

Contagens **verificadas** por `scripts/fetch_nih_cxr14.py` e por
`radiologyai manifest`, e a disjunção por paciente entre os splits foi
confirmada com `assert_patient_disjoint` (REQ-051): zero interseção.

## Prevalência no split oficial de teste

Medida a partir de `datasets/manifests/nih_cxr14_test.csv`:

| Achado | Positivos | Prevalência |
|---|---:|---:|
| Infiltration | 6.112 | 23,88% |
| Effusion | 4.658 | 18,20% |
| Atelectasis | 3.279 | 12,81% |
| Pneumothorax | 2.665 | 10,41% |
| Consolidation | 1.815 | 7,09% |
| Mass | 1.748 | 6,83% |
| Nodule | 1.623 | 6,34% |
| Pleural_Thickening | 1.143 | 4,47% |
| Emphysema | 1.093 | 4,27% |
| Cardiomegaly | 1.069 | 4,18% |
| Edema | 925 | 3,61% |
| Pneumonia | 555 | 2,17% |
| Fibrosis | 435 | 1,70% |
| Hernia | 86 | 0,34% |

Hérnia tem 86 positivos: qualquer AUROC para ela virá com intervalo de confiança
largo, e isso é declarado no artefato em vez de ser escondido atrás de uma média.

## Proveniência do rótulo — a limitação central

Os rótulos foram **minerados por NLP dos laudos radiológicos**, não adjudicados
por radiologista. O próprio NIH estima acurácia de rotulagem em torno de 90%.

Consequência prática: o teto de desempenho mensurável neste dataset é limitado
pelo ruído do rótulo, e um modelo pode ser penalizado por acertar o que o rótulo
errou. **Um número medido aqui não é comparável a um número medido contra
consenso de radiologistas.**

A validação contra rótulos adjudicados vem na Fase 3, com o **VinDr-CXR**
(conjunto de teste lido por 5 radiologistas).

## Vieses e limitações conhecidos

- Apenas incidências frontais (PA e AP). Perfil não representado.
- Uma única instituição, um único país. Distribuição de equipamento e de
  população não generalizável.
- A coluna `Patient Age` contém valores implausíveis (acima de 400) por erro de
  digitação no registro original — descartados por `_parse_age`.
- `Infiltration` do NIH não é diretamente comparável ao rótulo homônimo de outros
  datasets; a concordância entre definições é fraca.
- Sem sujeitos pediátricos identificados separadamente, ainda que existam.
- O dataset é notoriamente sujeito a atalhos espúrios: modelos podem aprender
  marcadores de posicionamento, texto gravado ou artefato de equipamento em vez
  de patologia.

## Uso neste projeto

| Uso | Fase |
|---|---|
| Split de teste oficial, para a linha de base honesta | Fase 1 |
| Split train_val, para o modelo próprio | Fase 3 |

**Nunca** para validação externa de um modelo que o tenha visto em treino. O
executor de avaliação detecta esse caso automaticamente (`check_leakage`) e marca
o resultado como `in-distribution`, recusando-se a chamá-lo de validação externa.

## Citação

```bibtex
@inproceedings{wang2017chestxray8,
  title     = {ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks
               on Weakly-Supervised Classification and Localization of Common
               Thorax Diseases},
  author    = {Wang, Xiaosong and Peng, Yifan and Lu, Le and Lu, Zhiyong and
               Bagheri, Mohammadhadi and Summers, Ronald M},
  booktitle = {CVPR},
  year      = {2017}
}
```
