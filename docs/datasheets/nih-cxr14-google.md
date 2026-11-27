# Datasheet — NIH ChestX-ray14, rótulos adjudicados do Google (Majkowska et al., Radiology 2020)

> Escrito à mão a partir do `DATASET.json` do cache local (`/root/medai-cache/datasets/nih-cxr14/`,
> fora do git; chave `google_adjudicated_labels`), do arquivo `google_labels_joined.csv` (contado
> aqui), do plano de medição e dos metadados do artigo no PubMed. Imagens: as do NIH ChestX-ray14 —
> ver [nih-chest-xray14.md](nih-chest-xray14.md). Nenhum pixel é versionado. O que não pôde ser
> conferido está marcado **não conferido**.

## Uso no projeto

Padrão de referência **adjudicado por radiologistas** sobre imagens do teste oficial do NIH: 19
itens `*@nih-google-adjudicated#<rótulo>` em `evidence/measured/plan.yaml` (`status: measure`,
`reference_standard: radiologist_adjudicated`), para os pesos xrv `pc`, `chex`, `mimic_nb`,
`mimic_ch`, `all` (fratura, pneumotórax, `lung_lesion`, `lung_opacity`) e `nih` (pneumotórax).

## Ressalva principal

Derivado do NIH (`leakage.py`: `nih-cxr14-google` deriva de `nih-cxr14`): qualquer modelo com
NIH no treino tem vazamento `known` aqui. Subconjunto **enriquecido** (opacidade ~58% no teste):
limiares, VPP e VPN não se transferem.

## Fonte e licença

| Campo | Valor |
|---|---|
| Artigo | Majkowska A, Mittal S, Steiner DF, et al. *Chest Radiograph Interpretation with Deep Learning Models: Assessment with Radiologist-adjudicated Reference Standards and Population-adjusted Evaluation*. Radiology 2020;294(2):421-431, doi:10.1148/radiol.2019191293 (PMID 31793848; metadados conferidos no PubMed) |
| Arquivo usado | `google2019_nih-chest-xray-labels.csv.gz` empacotado no wheel `torchxrayvision` 1.5.7; 45.090 bytes; sha256 `1d1b846e463753ed0219f67e21b10c8aef381897dbd64865c8191aa5794e8048` |
| Distribuição original do Google | **não conferido** (não baixado da fonte do Google) |
| Licença dos rótulos | "not stated in the torchxrayvision package or README" (`DATASET.json`); licença na fonte do Google: **não conferido** |
| Licença das imagens | termos do NIH (ver datasheet do NIH) |

## O que o projeto tem

`google_labels_joined.csv` (sha256 `a78c65f7241a0b546fa0d5654f4d722161724899bcf949884ce521c6bc188cf6`),
contado aqui:

| | Imagens | Pacientes | Fratura | Pneumotórax | Opacidade do espaço aéreo | Nódulo ou massa |
|---|---:|---:|---:|---:|---:|---:|
| Total | 4376 | 1695 | 186 | 238 | 2166 | 605 |
| `Set Id = test` (= `in_nih_test`) | 1962 | 860 | 72 | 195 | 1135 | 295 |
| `Set Id = val` | 2414 | 835 | 114 | 43 | 1031 | 310 |

Incidências no teste: AP 1126, PA 836. O resumo do artigo cita "2412 ChestX-ray14 validation set
images and 1962 test set images"; o arquivo do torchxrayvision tem **2414** de validação — diferença
de 2 imagens **não explicada** aqui. As 1962 de teste batem.

Os pixels vêm do teste oficial do NIH já baixado (HF `timm/nih-chest-xray-14`, revisão
`c1bf5796…`, 10 fragmentos com sha256 no `DATASET.json`; recodificação JPEG q95 1024×1024).

## Padrão de referência

Resumo do artigo (PubMed): "Reference standards were defined by radiologist-adjudicated image
review"; quatro achados (pneumothorax, opacity, nodule or mass, fracture) em radiografias frontais.
Número de leitores por imagem e regra de adjudicação: **não conferido** (texto completo não
acessível nesta sessão).

## Vieses e limitações

- Mesmas imagens, mesma instituição e mesmos atalhos do NIH; muda só o rótulo.
- 4 classes cujas definições não coincidem com as do xrv: "Nodule or mass" ↔ `lung_lesion`,
  "Airspace opacity" ↔ `lung_opacity` (aproximação declarada nos itens do plano).
- Até várias imagens por paciente (1962 imagens / 860 pacientes): IC agrupado por paciente.
- Seleção enriquecida em positivos pelo desenho do estudo original.

## Modelos treinados nele

`xrv-densenet121-res224-all` declara `nih-cxr14-google` (e `nih-cxr14`) em `training_data`.
Por linhagem, todo modelo com `nih-cxr14` no treino (`xrv-…-nih`, `-rsna`, `-all`, `resnet50-res512-all`,
`pspnet-chestxdet`, `ae-ood`) também compartilha as imagens.

## Des-identificação e governança

Imagens já des-identificadas pelo NIH. Nenhum pixel ou rótulo é versionado no git.
