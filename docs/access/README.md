# Acesso a datasets — o que exige credencial e o que não exige

Dois caminhos para **rótulos adjudicados por radiologista**, que é o que separa
"modelo fraco" de "rótulo ruidoso". A primeira linha de base mediu AUROC 0,664
contra rótulos minerados por NLP; fibrose saiu em 0,448 com IC95 excluindo 0,5,
o que sugere fortemente discordância de definição entre datasets, não erro do
modelo. Só um conjunto adjudicado resolve essa ambiguidade.

| Dataset | Rótulos | Acesso | Prazo | Tamanho |
|---|---|---|---|---|
| **CheXpert** (validação) | 3 radiologistas, voto majoritário | registro Stanford AIMI + Research Use Agreement | imediato | 234 imagens |
| **VinDr-CXR** (teste) | 5 radiologistas | credenciamento PhysioNet | 24–48 h após CITI | 3.000 exames |
| NIH ChestX-ray14 | minerados por NLP dos laudos (estimativa do próprio NIH; ver [datasheet](../datasheets/nih-chest-xray14.md)) | livre | imediato | 25.596 (teste) |

**Comece pelo CheXpert**: é imediato e já permite a primeira medição contra
consenso. O VinDr é o alvo definitivo — 3.000 exames contra 234 —, e o
credenciamento corre em paralelo.

---

## 1. CheXpert — sem credenciamento

1. Abra https://aimi.stanford.edu/datasets/chexpert-chest-x-rays
2. Preencha o formulário e aceite o **Stanford University Research Use Agreement**
3. O link de download chega por e-mail
4. Baixe o `CheXpert-v1.0-small` (~11 GB) — resolução reduzida basta, já que o
   modelo redimensiona para 224×224 de qualquer forma

Rodar a avaliação:

```bash
radiologyai evaluate --card xrv-densenet121-pc \
    --dataset chexpert-valid \
    --data-root /caminho/CheXpert-v1.0-small \
    --out artifacts/eval
```

O carregador usa apenas o `valid.csv` (234 imagens, rotuladas por consenso),
filtra incidências laterais (fora do uso pretendido declarado em REG-01 §2) e
trata rótulos incertos (`-1.0`) como **ausentes**, nunca como negativos —
contá-los como negativos inflaria a especificidade artificialmente.

**Limitação a manter em vista:** 234 imagens produzem intervalos de confiança
largos, e achados de baixa prevalência não terão suporte para avaliação. O
executor declara isso explicitamente em `not_evaluated[]`.

---

## 2. PhysioNet — credenciamento (para VinDr-CXR e MIMIC-CXR)

Processo vinculado à identidade: exige conta própria, documento verificável,
certificado de treinamento em seu nome e um referenciador que atesta por você.
**Ninguém pode fazer por você** — uma credencial obtida por terceiro é revogada
quando descoberta, e leva junto o acesso a todos os datasets.

### Passo 1 — Curso CITI (o mais demorado, ~6 h)

1. https://about.citiprogram.org → *Register*
2. Em **"Select Your Organization Affiliation"**, procure e selecione
   **Massachusetts Institute of Technology Affiliates** — é a afiliação que a
   PhysioNet aceita para quem não tem vínculo com instituição já cadastrada.
   Instruções oficiais: https://physionet.org/about/citi-course/
3. Escolha o curso **"Data or Specimens Only Research"** (não o curso completo
   de sujeitos humanos — é mais longo e não é o exigido)
4. Ao concluir, baixe o **Completion Report**, não o *Completion Certificate*.
   São arquivos diferentes; a PhysioNet pede o **Report**, que lista os módulos
   e as notas

### Passo 2 — Conta e credenciamento

1. https://physionet.org/register/
2. Use um **e-mail institucional** se tiver; ele acelera a verificação
3. https://physionet.org/settings/credentialing/ → preencher

Dados para o formulário:

| Campo | Valor |
|---|---|
| Nome | Guilherme Capel Pasqua |
| Profissão | Médico (*Physician*) |
| Instituição | *(sua vinculação atual — InCor/USP se o doutorado servir de vínculo)* |
| Registro profissional | CRM-SP 175873 · RQE 137036 |
| País | Brasil |

**Referenciador:** alguém que possa atestar sua identidade e propósito de
pesquisa — orientador do doutorado, chefe de serviço ou colega docente.
Precisa de nome, cargo, instituição e e-mail institucional. A PhysioNet
contacta essa pessoa diretamente.

**Campo de descrição da pesquisa** — texto pronto para colar:

> Retrospective evaluation of deep learning models for chest radiograph
> interpretation. I am a cardiologist (CRM-SP 175873) and PhD candidate in
> Cardiology. The project benchmarks published chest X-ray models against
> radiologist-adjudicated labels to quantify the gap between NLP-mined and
> expert-adjudicated ground truth, with per-finding confidence intervals and
> subgroup analysis. No clinical deployment is intended; the work is
> research-only and results are published with full provenance. Data will not
> be redistributed and will remain on encrypted storage under my control.

Ajuste se a sua vinculação institucional mudar o enquadramento.

### Passo 3 — VinDr-CXR

Aprovada a credencial: https://physionet.org/content/vindr-cxr/ → aceitar o
**PhysioNet Credentialed Health Data License 1.5.0** → baixar.

**Prazo real:** a revisão costuma sair em 24–48 h depois do relatório CITI
enviado. O gargalo é o curso, não a fila.

---

## Regra que vale para os três

Nenhum pixel entra no repositório. O que é versionado é o **manifest** — quais
imagens, de qual paciente, com quais rótulos, mais o sha256 do conteúdo. Ver
`data/DATASET_CARD.md`.
