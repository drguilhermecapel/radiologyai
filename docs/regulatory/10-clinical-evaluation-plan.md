---
doc_id: REG-10
title: Plano de Avaliação Clínica
status: RASCUNHO — v0.2
date: 2026-11
author: Dr. Guilherme Capel Pasqua (CRM-SP 175873)
depends_on: [REG-01, REG-04]
---

# Plano de Avaliação Clínica

> Software de pesquisa; não é dispositivo médico. Este plano fixa **antes** de qualquer avaliação
> definitiva como cada condição será avaliada e o que conta como aprovação. Os critérios de
> aceitação são os do protocolo pré-registrado
> ([`../protocolo-evidencia.md`](../protocolo-evidencia.md), `protocol_id: "2026-11-v1"`);
> qualquer mudança exige novo `protocol_id`.

## 1. Estrutura da evidência

A RDC 657/2022 (Art. 2) define três componentes, que espelham o documento IMDRF N41 citado pelo
MDCG 2020-1: **associação clínica válida** (validade científica), **validação analítica** e
**validação clínica** (utilidade clínica). Este plano os mapeia aos portões do
[`ROADMAP.md`](../../ROADMAP.md):

| Componente | O que mostra | Portões | Fonte |
|---|---|---|---|
| Associação clínica válida | a saída (achado, medida, limiar) tem significado clínico estabelecido | G1 — nível B: literatura revisada por pares e diretrizes, citação conferida na fonte | catálogo |
| Validação analítica | o software calcula a saída corretamente e de modo reprodutível | G0 verificação; G1 nível A/C em dados públicos; G2 em dados brasileiros | `artifacts/eval/` |
| Validação clínica | a saída melhora a decisão do médico | G3 (MRMC), G4 (piloto silencioso) | estudos com CEP |

## 2. Critérios de aceitação comuns (= protocolo)

| Item | Critério |
|---|---|
| Elegibilidade | ≥ 30 positivos e ≥ 30 negativos (G1); proposta de ≥ 100 e ≥ 100 para G2, com novo `protocol_id` |
| Discriminação | AUROC ≥ 0,80 e limite inferior do IC95 ≥ 0,75 |
| Detecção em limiar fixo | limite inferior do IC95 de sensibilidade e de especificidade ≥ 0,80 |
| Concordância de medida contínua | ICC ≥ 0,75 |
| Segmentação | Dice ≥ 0,85 |
| IC | 95%, bootstrap de 2000 iterações **agrupado por paciente**, semente 20261124 |
| Pontos de operação | ajustados na metade de calibração (50%, por paciente), reportados na metade de teste; alvo de *rule-out* = sensibilidade 0,90; *rule-in* = especificidade 0,90 |
| Subgrupos | sexo, faixa etária, incidência (quando o metadado existe) |
| Vazamento | medição externa só sustenta A com `leakage: none` |
| Relato | STARD-AI; CLAIM 2024; falhas publicadas |

## 3. Plano por modalidade

Para cada linha: conjunto de avaliação, padrão de referência, desfecho do protocolo e o que se
pode alegar se aprovado. "Dados brasileiros" significa G2, sob CEP.

### 3.1 RX de tórax

| Condição / saída | Conjunto (G1) | Padrão de referência | Desfecho | G2 |
|---|---|---|---|---|
| Achados do TorchXRayVision (cardiomegalia, derrame, edema, pneumotórax…) | NIH ChestX-ray14 teste oficial com pesos sem NIH (`pc`, `chex`, `mimic_nb`, `mimic_ch`); CheXpert validação; VinDr-CXR após credenciamento | NLP (NIH); 3 radiologistas (CheXpert); radiologistas (VinDr) | discriminação | RX de ≥ 2 centros, leitura adjudicada |
| Índice cardiotorácico | conjunto com máscaras manuais de coração e pulmão (a identificar) | segmentação de especialista | segmentação + concordância | idem |
| Tuberculose | treino Shenzhen → teste Montgomery e vice-versa | conforme datasets NLM | discriminação | depende de parceiro com confirmação microbiológica |

### 3.1a Modelo próprio de RX de tórax (G2; ex-Fase 3 da v1.0)

Pré-especificação herdada da v0.1 deste plano em `main` (commit `6f4b7ec`), fixada **antes** do
treinamento de qualquer modelo próprio e antes de qualquer exposição ao conjunto de teste
externo, para manter o caráter confirmatório. Não substitui o §2: um modelo próprio também
precisa passar no protocolo vigente para chegar a nível A.

- **Dispositivo em avaliação:** radiografia frontal de tórax em adulto (`CR`/`DX`), uso assistivo
  concorrente ([REG-01](01-intended-use.md)).
- **Pergunta primária:** um modelo próprio treinado em NIH ChestX-ray14 (split disjunto por
  paciente) supera a linha de base registrada em
  `artifacts/eval/xrv-densenet121-pc__20260912T202549Z/` quando avaliado externamente em coorte
  independente com rótulo de radiologistas (VinDr-CXR)?
- **Fora do escopo desta avaliação:** uso autônomo, triagem/priorização, exclusão (*rule-out*),
  pediatria, incidência lateral.

| Etapa | Conjunto | Rótulo | Isolamento |
|---|---|---|---|
| Desenvolvimento e calibração | NIH ChestX-ray14, `train`/`val` disjuntos por paciente | NLP (limitação declarada) | disjunção de paciente verificada (REQ-051); conjunto de calibração separado do treino |
| Avaliação externa primária | VinDr-CXR, após credenciamento PhysioNet (18.000 exames; teste de 3.000 lidos por 5 radiologistas, segundo a v0.1 — tamanhos a conferir no datasheet quando o acesso existir) | consenso de radiologistas | isolado até o congelamento do artefato (pesos + sha256 + limiares de abstenção) |
| Avaliação externa complementar | CheXpert, conjunto de validação | adjudicado por 3 radiologistas | cobertura parcial de achados (não avalia fibrose) |

Desfechos obrigatórios, todos produzidos por `radiologyai evaluate` em
`artifacts/eval/<run_id>/metrics.json` com proveniência completa:

1. Discriminação (primário): AUROC por achado e macro, IC95 por bootstrap agrupado por paciente
   com semente fixa (REQ-050).
2. Calibração: ECE antes e depois da calibração (temperatura ou Platt, ambas monotônicas) e
   diagrama de confiabilidade (REQ-061).
3. Abstenção: distribuição nas três bandas derivada só de artefato medido (REQ-062), sem afirmar
   normalidade (REQ-063).
4. Subgrupos: incidência (PA vs. AP), sexo, faixa etária adulta, equipamento/fonte quando houver
   metadado.
5. Paridade de inferência: diferença absoluta máxima < `1e-4` entre PyTorch e ONNX numa amostra de
   100 exames.

Critérios de aceitação pré-especificados (METAS, não resultados):

- AUROC macro externo superior ao da linha de base em
  `artifacts/eval/xrv-densenet121-pc__20260912T202549Z/`, sem degradação clinicamente inaceitável
  em achados de alta severidade (pneumotórax, derrame pleural, cardiomegalia) — a margem
  "clinicamente inaceitável" precisa ser fixada em número no protocolo de G2 antes do treino;
- ECE reduzido ou mantido após calibração, sem alteração da ordenação;
- execução reprodutível bit a bit (exceto carimbo de tempo) de checkout limpo;
- RISK-006 fechado: REQ-043 verificado no pipeline de RX ([`04-risk-file.md`](04-risk-file.md)).

**Desvios:** qualquer mudança de arquitetura, hiperparâmetro, limiar de abstenção ou exclusão de
exames depois da exposição ao teste externo invalida o caráter confirmatório e é registrada como
desvio exploratório em `reports/`.

**Medições exploratórias anteriores:** os relatos em
[`docs/evaluation/phase1-nih1000-baseline.md`](../evaluation/phase1-nih1000-baseline.md) e
[`docs/evaluation/phase1-nih-unseen-benchmark.md`](../evaluation/phase1-nih-unseen-benchmark.md)
(Colab, 2026-11-25) não têm artefato commitado e não sustentam nenhum critério deste plano — ver
[`HONEST_STATUS.md`](../../HONEST_STATUS.md).

### 3.2 RX musculoesquelético

| Condição / saída | Conjunto (G1) | Padrão de referência | Desfecho | G2 |
|---|---|---|---|---|
| Fratura | treino FracAtlas → teste GRAZPEDWRI-DX (e vice-versa), respeitando faixa etária | rótulo do dataset | discriminação | parceiro com laudo adjudicado |

### 3.3 TC

| Condição / saída | Conjunto (G1) | Padrão de referência | Desfecho |
|---|---|---|---|
| Hemorragia intracerebral (`cerebral_bleed`) | CQ500 (externo ao modelo) | leituras de radiologistas do CQ500 | detecção em limiar fixo de volume, pré-especificado |
| Derrame pleural/pericárdico | publicação (nível B) + conjunto público com rótulo, se disponível | segmentação/rótulo | detecção / segmentação |
| Biomarcadores em HU (L1, fígado, %LAA-950, aorta, baço, Evans) | limiar de diretriz ou de estudo de método (nível B); verificação analítica com fantomas ou casos públicos | método de referência publicado | `threshold` + verificação das pré-condições |
| Nódulo pulmonar | **sem conjunto limpo** (LIDC-IDRI em parte no treino) | — | não avaliável em G1 |

### 3.4 RM

| Condição / saída | Conjunto (G1) | Padrão de referência | Desfecho |
|---|---|---|---|
| Subregiões de glioma (BraTS bundle) | conjunto BraTS de edição não usada no treino, se houver licença | segmentação de especialista | segmentação |
| Carga lesional de substância branca (LST-AI) | conjunto público com máscaras de especialista | segmentação de especialista | segmentação; nunca diagnóstico de EM |

### 3.5 Mamografia

| Condição / saída | Conjunto (G1) | Padrão de referência | Desfecho |
|---|---|---|---|
| Escore de malignidade (mammoscreen) | CMMD — população **diagnóstica** biopsiada, sem casos normais | histologia | discriminação benigno vs. maligno; **não** se alega rastreamento |
| Densidade | conjunto com densidade BI-RADS por radiologista (a identificar) | leitura de radiologista | concordância (kappa a pré-especificar em novo protocolo) |

### 3.6 Ultrassonografia

| Condição / saída | Conjunto (G1) | Padrão de referência | Desfecho |
|---|---|---|---|
| Lesão mamária | BUS-BRA ↔ BrEaST (treino num, teste no outro) | histologia ou seguimento | discriminação |
| Segmentação interativa (MedSAM) | BUS-BRA, BrEaST | máscara de especialista | segmentação (Dice), caixa vinda do usuário ou declarada |
| Esteatose | Byra (55 pacientes, biópsia) | histologia | discriminação; n pequeno, IC largo |

### 3.7 Ecocardiografia

| Condição / saída | Conjunto (G1) | Padrão de referência | Desfecho | G2 |
|---|---|---|---|---|
| Traçado do VE e FE por Simpson (MedSAM2) | CAMUS (licença não comercial) | FE de referência do CAMUS; contornos de especialista | segmentação (Dice) + concordância (ICC) | modo pesquisa até aprovar; G2 com ecos de parceiro e FE do laudo de ecocardiografista |

## 4. Estudos posteriores

- **G3 — MRMC:** desenho com populações de leitores e de casos, padrão de referência e análise
  pré-especificados (Obuchowski & Bullen, Radiology 2022, DOI 10.1148/radiol.211593; Gallas et
  al., Acad Radiol 2012, DOI 10.1016/j.acra.2011.12.016). Hipótese, desfecho e tamanho amostral a
  fixar em protocolo próprio antes de recrutar leitores.
- **G4 — piloto prospectivo silencioso:** saídas ocultas do médico assistente; relato DECIDE-AI
  (Vasey et al., Nat Med 2022, DOI 10.1038/s41591-022-01772-9).
- Ensaio randomizado, se algum dia, segue SPIRIT-AI e CONSORT-AI.

## 5. Ética

Todo estudo com dado de paciente brasileiro exige protocolo aprovado no sistema CEP/Conep
(Lei 14.874/2024; Decreto 12.651/2025); dispensa de TCLE solicitada com justificativa (Res. CNS
466/2012, item IV.8); bancos de dados conforme Res. CNS 738/2024. Datasets públicos são usados
sob os termos de cada um; a necessidade de apreciação por CEP para dado público anonimizado será
confirmada com o CEP da instituição parceira.

---

## Histórico de revisões

| Versão | Data | Alteração |
|---|---|---|
| v0.1 | 2026-11 | Criação; critérios iguais ao protocolo `2026-11-v1` |
| v0.2 | 2026-11-27 | Consolidação com a v0.1 de `main`: pré-especificação do modelo próprio de RX (§3.1a) e ligação aos relatos exploratórios sem artefato |
