# Licenças de modelos de terceiros

Inventário de todo modelo (pesos e código) que o projeto usa, avalia ou planeja usar. Fontes:
arquivos de licença e cartões de modelo lidos na origem em novembro/2026 (repositórios oficiais,
cartões do Hugging Face, documentação do desenvolvedor). **"a confirmar"** = não verificado na
fonte; não presuma nada nesses casos.

> Software de pesquisa; não é dispositivo médico. Este documento não contém nenhuma métrica de
> desempenho: desempenho vive em `artifacts/eval/` e no catálogo de condições.

## Regras

1. Pesos não comerciais ou só de pesquisa são **permitidos** no projeto, marcados
   `commercial_use: "no"` no manifesto do modelo e nesta tabela.
2. Código GPL/AGPL ligado ao pacote é **excluído**.
3. Modelo sem licença publicada **não é carregado, avaliado nem distribuído**.
4. Licença acadêmica intransferível ⇒ o projeto **não redistribui** os pesos nem os empacota numa
   imagem entregue a terceiros.
5. Espelho de pesos ou de dados que não seja a fonte oficial não é usado para contornar termos.
6. Cada linha precisa bater com o manifesto do modelo em `src/radiologyai/`; divergência é defeito.

## Inventário

Legenda de uso comercial: **sim** = a licença permite; **não** = a licença ou o cartão proíbe ou
restringe a pesquisa; **a confirmar** = não verificado.

| Modelo | Uso no projeto | Código | Pesos | Uso comercial | Fonte | Estado |
|---|---|---|---|---|---|---|
| TorchXRayVision DenseNet121 (`pc`, `chex`, `mimic_nb`, `mimic_ch`, `nih`, `all`) | RX de tórax, até 18 achados | Apache-2.0 | distribuídos pela biblioteca; licença própria dos pesos a confirmar (treinados sobre datasets com termos próprios, inclusive MIMIC-CXR credenciado) | a confirmar | repositório `mlmed/torchxrayvision` | em uso (`pc` medido) |
| TorchXRayVision autoencoder ResNet | gate de distribuição em RX | Apache-2.0 | idem | a confirmar | idem | viabilidade demonstrada |
| PSPNet ChestX-Det (via TorchXRayVision) | máscaras para índice cardiotorácico | Apache-2.0 (biblioteca) | origem Deepwise / ChestX-Det; a confirmar | a confirmar | `xrv.baseline_models.chestx_det`; Lian et al., arXiv:2104.10326 | viabilidade demonstrada |
| Classificador de incidência `chestViewSplit` | gate de incidência em RX | a confirmar | a confirmar | a confirmar | — | viabilidade demonstrada; licença pendente antes de integrar |
| TotalSegmentator — tarefas abertas (`total`, `total_mr`, `cerebral_bleed`, `pleural_pericard_effusion`, `liver_lesions`, `liver_lesions_mr`, `lung_nodules`, `kidney_cysts`, `vertebrae_body`, `ventricle_parts` e demais da lista aberta) | TC/RM: anatomia, patologias, biomarcadores | Apache-2.0 | Apache-2.0 ("Openly available for any usage") | sim | README do repositório `wasserth/TotalSegmentator`; o projeto declara "not a medical device and is not intended for clinical usage" | em uso / viabilidade demonstrada |
| TotalSegmentator — tarefas com chave (`coronary_arteries`, `tissue_types`, `heartchambers_highres`, `aortic_dissection`, classificador de fase de contraste) | cálcio coronário, composição corporal, fase | Apache-2.0 | licença gratuita **só para uso não comercial**; comercial mediante contato | não | idem | **planejado** (depende de chave de licença) |
| MONAI `brats_mri_segmentation` | RM: subregiões de glioma conhecido | a confirmar | a confirmar | a confirmar | MONAI model zoo (Hugging Face `MONAI/brats_mri_segmentation`) | planejado |
| LST-AI | RM: lesões de substância branca | a confirmar | a confirmar | a confirmar | Wiltgen et al. 2024, DOI 10.1016/j.nicl.2024.103611 | planejado |
| `ianpan/mammoscreen` | mamografia: escore de câncer e densidade | Apache-2.0 | Apache-2.0 | sim (licença); dados de treino do desafio RSNA com termos próprios, a confirmar | cartão no Hugging Face (`license: apache-2.0`) | viabilidade demonstrada |
| `ianpan/mammo-crop` | recorte de mamografia | Apache-2.0 | Apache-2.0 | sim | cartão no Hugging Face | viabilidade demonstrada |
| MONAI `breast_density_classification` | densidade mamária | Apache-2.0 | Apache-2.0 | sim (licença); metadado do bundle: "This is an example, not to be used for diagnostic purposes" | `Project-MONAI/model-zoo` | viabilidade demonstrada |
| Mirai | risco de câncer de mama em 5 anos | MIT | MIT | sim (licença); README: não destinado a "any other clinical use" | README do repositório `yala/Mirai` | planejado (sem dado com seguimento para medir) |
| MedSAM ViT-B (`wanglab/medsam-vit-base`) | segmentação interativa por caixa | Apache-2.0 | Apache-2.0 | sim | `bowang-lab/MedSAM` LICENSE; cartão no Hugging Face | viabilidade demonstrada |
| MedSAM2 `MedSAM2_US_Heart.pt` | eco: traçado assistido do VE | Apache-2.0 | **contraditória**: metadado `cc-by-sa-4.0`; texto do cartão "can only be used for research and education purposes" | **não** (vale a restrição mais forte) | `bowang-lab/MedSAM2` LICENSE; cartão `wanglab/MedSAM2` | viabilidade demonstrada; só modo pesquisa |
| RAD-DINO (`microsoft/rad-dino`) | encoder de RX para sondas | MIT | MIT | **não** — cartão: "shared for research purposes only ... not meant to be used for clinical practice" | cartão no Hugging Face | avaliado em *spike* |
| BiomedCLIP (`microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`) | zero-shot e encoder para sondas | MIT | MIT | **não** — cartão: "Any deployed use case of the model --- commercial or otherwise --- is currently out of scope" | cartão no Hugging Face | só modo pesquisa |
| CheXzero | zero-shot em RX | MIT (código) | a confirmar; treinado em MIMIC-CXR (acesso credenciado PhysioNet) | não (até confirmar) | repositório `rajpurkarlab/CheXzero` | só modo pesquisa |
| Merlin | zero-shot em TC 3D | MIT | a confirmar (repositório MIT) | a confirmar | LICENSE de `StanfordMIMI/Merlin` | só modo pesquisa |
| EchoPrime | — | Cedars-Sinai Academic Software License: só pesquisador acadêmico ou sem fins lucrativos; transferência proibida | idem; o modelo exige um corpus de 1,23 milhão de laudos derivados de pacientes | **não** | LICENSE de `echonet/EchoPrime` | **excluído** (licença intransferível; recuperação sobre laudos de terceiros) |
| PanEcho | — | **nenhuma licença publicada** nos caminhos verificados | — | — | README de `CarDS-Yale/PanEcho` sem licença; arquivos LICENSE ausentes | **excluído** até haver licença |
| MedGemma, MedSigLIP, CXR Foundation (Google HAI-DEF) | — | termos HAI-DEF (acesso restrito) | idem | restrito; a definição de "Clinical Use" dos termos inclui uso em estudo de pesquisa | cartões no Hugging Face e termos HAI-DEF | não planejado (exige leitura jurídica) |
| CT-CLIP | — | CC BY-NC-SA 4.0 | idem | não | README de `ibrahimethemhamamci/CT-CLIP` | não planejado |

## Datasets

Licenças de datasets ficam nos datasheets ([`docs/datasheets/`](docs/datasheets/)) e no plano de
acesso do [`ROADMAP.md`](ROADMAP.md) §9.1. Pontos que afetam modelos: CAMUS e CQ500 são
CC BY-NC-SA 4.0 (avaliação só não comercial); BUSI foi obtido de um espelho no Hugging Face cuja
licença declarada deve ser conferida contra a publicação original antes de qualquer uso além de
verificação de viabilidade.
