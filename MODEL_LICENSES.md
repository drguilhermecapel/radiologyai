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
6. Cada linha precisa bater com o manifesto do modelo em
   [`src/radiologyai/analysis/manifests/`](src/radiologyai/analysis/manifests/); divergência é defeito.

## Inventário

Legenda de uso comercial (valor do campo `commercial_use` do manifesto): **sim** (`yes`) = a
licença permite; **não** (`no`) = a licença, o cartão ou os termos dos dados de treino proíbem ou
restringem a pesquisa; **incerto** (`unclear`) = a licença dos pesos é permissiva ou ausente, mas
os termos dos dados de treino não foram lidos ou não são claros — **trate como não comercial até
confirmar**.

### Modelos com manifesto `available` (rodam de verdade, pesos com sha256)

Manifestos em [`src/radiologyai/analysis/manifests/`](src/radiologyai/analysis/manifests/). A
coluna "Manifesto" é a fonte normativa; divergência entre esta tabela e o manifesto é defeito
desta tabela.

| Modelo | Manifesto(s) | Uso no projeto | Código | Pesos | Uso comercial | Fonte |
|---|---|---|---|---|---|---|
| TorchXRayVision DenseNet121 `chex`, `all` | `xrv-densenet121-res224-chex`, `-all` | RX de tórax, multirrótulo | Apache-2.0 | sem licença própria declarada (asset de release do repositório) | **não** — treino inclui CheXpert (Research Use Agreement, não comercial) e, no `all`, MIMIC-CXR | LICENSE de `mlmed/torchxrayvision` |
| TorchXRayVision DenseNet121 `pc`, `nih`, `mimic_nb`, `mimic_ch`, `rsna`; ResNet50 `res512-all` | `xrv-densenet121-res224-{pc,nih,mimic_nb,mimic_ch,rsna}`, `xrv-resnet50-res512-all` | RX de tórax, multirrótulo | Apache-2.0 | idem | incerto — termos de PadChest, NIH, MIMIC-CXR (PhysioNet credenciado), RSNA, VinDr | idem |
| TorchXRayVision ResNetAE 101-elastic | `xrv-ae-ood` | portão fora de distribuição em RX | Apache-2.0 | idem | incerto (treino inclui CheXpert) | idem |
| PSPNet ChestX-Det (via TorchXRayVision) | `xrv-pspnet-chestxdet` | máscaras → índice cardiotorácico | Apache-2.0 | idem | incerto (ChestX-Det, Deepwise, sobre imagens do NIH) | idem |
| Classificador de incidência xinario (via TorchXRayVision) | `xrv-view-xinario` | portão frontal × lateral | Apache-2.0 | idem | incerto (dados de `xinario/chestViewSplit`) | idem |
| RAD-DINO (`microsoft/rad-dino`) + sondas deste projeto | `xr-probe-tb`, `xr-probe-fracture`, `xr-probe-grazpedwri-{cast,metal,osteopenia}` | encoder congelado de RX; sondas lineares de TB, fratura e achados do GRAZPEDWRI-DX | MIT (RAD-DINO) + código deste projeto | RAD-DINO MIT; cartão: "shared for research purposes only ... not meant to be used for clinical practice" | **não** | cartão e LICENSE de `microsoft/rad-dino` no Hugging Face |
| CheXzero | `chexzero` | zero-shot em RX — **só modo pesquisa** | MIT | sem licença própria declarada; treinado em MIMIC-CXR | **não** | LICENSE e NOTICE de `rajpurkarlab/CheXzero` |
| TotalSegmentator — tarefas abertas (`total`, `total_mr`, `body`, `cerebral_bleed`, `pleural_pericard_effusion`, `liver_lesions`, `liver_vessels`, `lung_nodules`, `kidney_cysts`, `vertebrae_body`, `ventricle_parts`) | `totalseg-*` | TC/RM: anatomia, achados segmentados, base dos biomarcadores | Apache-2.0 | Apache-2.0 ("Openly available for any usage") | sim; o projeto declara "not a medical device" | README de `wasserth/TotalSegmentator` |
| Biomarcadores de TC | `ct-biomarkers` | HU e geometria sobre máscaras do TotalSegmentator | código deste projeto | sem pesos (algoritmo) | sim | — |
| `ianpan/mammoscreen` | `mammoscreen` | mamografia: escore de câncer e densidade por mama | Apache-2.0 (código vendorizado sem execução remota) | Apache-2.0 | incerto — termos dos dados de treino (CBIS-DDSM; competição RSNA no Kaggle) não lidos | cartão no Hugging Face (`license: apache-2.0`) |
| `ianpan/mammo-crop` | `mammo-crop` | recorte da mama | Apache-2.0 | Apache-2.0 | incerto (idem) | cartão no Hugging Face |
| MONAI `breast_density_classification` | `monai-breast-density` | densidade A–D | Apache-2.0 | Apache-2.0 | incerto — dados de treino Mayo Clinic não públicos; metadado: "This is an example, not to be used for diagnostic purposes" | `Project-MONAI/model-zoo` |
| MedSAM ViT-B (`wanglab/medsam-vit-base`) | `medsam-vit-base` | segmentação interativa por caixa em US | Apache-2.0 | Apache-2.0 | incerto — treino em bases públicas, várias não comerciais | LICENSE de `bowang-lab/MedSAM`; cartão no Hugging Face |
| MedSAM2 `MedSAM2_US_Heart.pt` | `medsam2-echo-lv` | eco: traçado assistido do VE + Simpson — **só modo pesquisa** | Apache-2.0 | **contraditória**: metadado `cc-by-sa-4.0`; texto do cartão "can only be used for research and education purposes" | **não** (vale a restrição mais forte) | LICENSE de `bowang-lab/MedSAM2`; cartão `wanglab/MedSAM2` |
| BiomedCLIP + sondas deste projeto | `us-probe-breast-malignancy`, `us-probe-liver-lesion-malignancy`, `us-probe-liver-steatosis` | encoder congelado de US; sondas lineares | MIT (BiomedCLIP/open_clip) + código deste projeto | MIT; cartão: "Any deployed use case of the model --- commercial or otherwise --- is currently out of scope" | **não** | LICENSE.md e README.md de `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` |

### Modelos com manifesto `planned` (não rodam no produto; motivo no manifesto)

| Modelo | Manifesto | Código | Pesos | Uso comercial | Por que está planejado |
|---|---|---|---|---|---|
| EchoPrime | `echoprime` | Cedars-Sinai Academic Software License (só pesquisador acadêmico/sem fins lucrativos; transferência proibida) | idem | **não** | rodou no *spike* de eco, mas a licença impede distribuir e a inferência exige redistribuir um corpus de 1,23 milhão de laudos derivados de pacientes — **excluído** |
| PanEcho | `panecho` | **nenhuma licença publicada** | nenhuma | — | sem licença vale o direito autoral padrão; não baixado nem executado — **excluído** até haver licença |
| LST-AI | `lst-ai` | MIT (greedy, HD-BET, FastSurfer Apache-2.0) | MIT (distribuídos com o pacote) | incerto | pacote fora do ambiente travado e nenhum par T1w + FLAIR aberto no cache para verificar execução real |
| MONAI `brats_mri_segmentation` | `monai-brats` | Apache-2.0 | Apache-2.0 | incerto (termos BraTS 2018) | exige as 4 sequências co-registradas sem crânio a 1 mm (pré-processamento BraTS não implementado) e não há caso de teste no cache |
| Mirai | `mirai` | MIT | MIT | incerto; README: não destinado a "any other clinical use" | *snapshots* em pickle legado do pacote `onconet`; nenhum dado disponível com seguimento para medir risco futuro |

### Modelos sem manifesto (avaliados em *spike* ou só considerados)

| Modelo | Código | Pesos | Uso comercial | Situação |
|---|---|---|---|---|
| TotalSegmentator — tarefas com chave (`coronary_arteries`, `tissue_types`, `heartchambers_highres`, `aortic_dissection`, classificador de fase de contraste) | Apache-2.0 | licença gratuita **só para uso não comercial**, mediante chave | não | **planejado** — cálcio coronário, sarcopenia e fase de contraste dependem da chave |
| Merlin (TC 3D, zero-shot) | MIT (`StanfordMIMI/Merlin` LICENSE) | licença dos pesos a confirmar | a confirmar | rodou em *spike* (CPU); sem adaptador — só modo pesquisa quando integrado |
| BiomedCLIP como roteador zero-shot de modalidade | MIT | MIT; cartão restringe a pesquisa | não | rodou em *spike*; não usado no roteamento (roteamento é por metadado) |
| MedGemma, MedSigLIP, CXR Foundation (Google HAI-DEF) | termos HAI-DEF (acesso restrito) | idem | restrito; a definição de "Clinical Use" dos termos inclui uso em estudo de pesquisa | não planejado (exige leitura jurídica) |
| CT-CLIP | CC BY-NC-SA 4.0 | idem | não | não planejado |

## Datasets

Licenças de datasets ficam nos datasheets ([`docs/datasheets/`](docs/datasheets/)) e no plano de
acesso do [`ROADMAP.md`](ROADMAP.md) §9.1. Pontos que afetam modelos: CAMUS e CQ500 são
CC BY-NC-SA 4.0 (avaliação só não comercial); BUSI foi obtido do espelho `MedOtter/BUSI` no Hugging
Face, cuja declaração "CC BY 4.0" é inferida da licença do **artigo** (Data in Brief 2020), não de
um arquivo de licença dos distribuidores dos dados. Pela regra 5, isso é **pendência aberta**: a
sonda `us-probe-breast-malignancy` foi ajustada nesse espelho e não pode sustentar evidência além
de viabilidade até a licença ser conferida na fonte original — ou até a sonda ser reajustada em
BUS-BRA ou BrEaST (CC BY 4.0, fontes oficiais).
