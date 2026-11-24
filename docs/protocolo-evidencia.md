# Protocolo de evidência pré-registrado

| Campo | Valor |
|---|---|
| `protocol_id` | `2026-11-v1` |
| Congelado em | 2026-11-24 |
| Fonte normativa | [`src/radiologyai/catalog/protocol.yaml`](../src/radiologyai/catalog/protocol.yaml) |
| Regra de nível | `compute_level()` em [`src/radiologyai/catalog/schema.py`](../src/radiologyai/catalog/schema.py) |
| Autor | Dr. Guilherme Capel Pasqua (CRM-SP 175873) |

> Este documento é a versão legível de `protocol.yaml`. **Em caso de divergência, vale o YAML**,
> e a divergência é um defeito deste documento. Nenhum número abaixo é desempenho medido: são
> **critérios de aceitação fixados antes de qualquer medição**.

## 1. Por que pré-registrar

Um critério escolhido depois de ver o resultado não é critério, é descrição. O protocolo foi
commitado antes de qualquer medição do catálogo; o histórico git é a prova do pré-registro.
Mudar qualquer limiar depois de ver resultados exige **novo `protocol_id`**, e todo registro de
evidência nomeia o protocolo sob o qual foi julgado (`MeasurementProvenance.protocol_id`). A
exigência de testar contra "métricas e limiares probabilísticos previamente definidos" aparece
também no AI Act europeu (Regulamento (UE) 2024/1689, Art. 9(8)); aqui ela vale desde já.

Quem decide se um registro passa (`passes_protocol`) é o avaliador, a partir dos números e deste
protocolo — **nunca uma pessoa editando o campo**. Registro reprovado precisa declarar
`failure_reasons`.

## 2. Elegibilidade de um registro medido

| Parâmetro (`protocol.yaml`) | Valor | Significado |
|---|---|---|
| `min_positive` | 30 | casos positivos mínimos, pelo padrão de referência |
| `min_negative` | 30 | casos negativos mínimos |

Abaixo disso o registro não é elegível para A nem C, qualquer que seja a estimativa pontual.
Contagem por **paciente** quando o desfecho é por paciente; por imagem só quando cada paciente
contribui uma imagem.

Estes mínimos foram desenhados para G1 (catálogo sobre dados públicos). Para G2 (validação
externa brasileira) a proposta é 100 positivos e 100 negativos (ver
[`ROADMAP.md`](../ROADMAP.md) §4), o que exigirá um novo `protocol_id`.

## 3. Barras por desfecho

O esquema reconhece cinco desfechos (`Endpoint`): `discrimination`, `detection`, `agreement`,
`segmentation`, `threshold`.

| Desfecho | Parâmetros (`protocol.yaml`) | Barra |
|---|---|---|
| **Discriminação** (escore contínuo vs. rótulo binário) | `auroc_min`, `auroc_ci_low_min` | AUROC pontual ≥ 0,80 **e** limite inferior do IC ≥ 0,75 |
| **Detecção em limiar fixo** pré-especificado (ex.: "volume de hemorragia segmentado > 0") | `sens_spec_ci_low_min` | limite inferior do IC da sensibilidade **e** da especificidade ≥ 0,80 |
| **Concordância** de medida contínua com referência clínica (ex.: fração de ejeção) | `icc_min` | ICC ≥ 0,75 |
| **Segmentação** contra máscara de especialista | `segmentation_dice_min` | Dice ≥ 0,85 (no YAML o campo fica no bloco da barra de publicação; vale para registro publicado e medido) |
| **Limiar de diretriz** aplicado a medida validada (`threshold`) | barra de publicação (§6) | a medida de base precisa de validação publicada e as pré-condições de aquisição do limiar precisam ser satisfeitas no exame |

## 4. Estatística

| Parâmetro | Valor | Regra |
|---|---|---|
| `ci_level` | 0.95 | todo IC reportado é de 95% |
| `bootstrap_iterations` | 2000 | reamostragem percentílica |
| `bootstrap_cluster` | `patient` | **a unidade reamostrada é o paciente**, nunca a imagem: imagens do mesmo paciente são correlacionadas e reamostrá-las independentemente estreita o IC artificialmente |
| `random_seed` | 20261124 | semente fixa; duas execuções com a mesma semente produzem os mesmos números |

## 5. Pontos de operação: metade de calibração, metade de teste

| Parâmetro | Valor | Regra |
|---|---|---|
| `calibration_fraction` | 0.5 | o conjunto é dividido **por paciente** em duas metades |
| `rule_out_sensitivity_target` | 0.90 | o ponto de exclusão (`rule_out`) é o limiar que atinge essa sensibilidade na metade de calibração |
| `rule_in_specificity_target` | 0.90 | o ponto de inclusão (`rule_in`) é o limiar que atinge essa especificidade na metade de calibração |

Os limiares são **ajustados na metade de calibração e reportados na metade de teste**. Reportar
sensibilidade e especificidade no mesmo dado em que o limiar foi escolhido é otimista por
construção e não é aceito. O `OperatingPoint` resultante carrega `source` (este protocolo e a
execução) e `evidence_ref` (o artefato em `artifacts/eval/`), e nenhuma métrica
([`calibration/decision.py`](../src/radiologyai/calibration/decision.py)). Entre os dois pontos o
achado é `indeterminate`; sem ponto utilizável, `score_only`.

O alvo de 0,90 em *rule-out* **não** autoriza uso para exclusão de doença: o uso pretendido
proíbe *rule-out* clínico ([`01-intended-use.md`](regulatory/01-intended-use.md)). O ponto serve
para definir a banda de abstenção.

## 6. Barra para evidência publicada (nível B)

| Parâmetro | Valor |
|---|---|
| `published_auroc_min` | AUROC publicado ≥ 0,80 |
| `published_sens_spec_min` | sensibilidade e especificidade publicadas ≥ 0,80 |
| `published_min_positive` | ≥ 30 positivos no conjunto publicado |

Além dos números, um registro `published` ou `guideline_threshold` só promove a condição se:

1. a publicação é **revisada por pares** ou é **diretriz clínica** (preprint, cartão de modelo,
   README de repositório e página de fornecedor não contam);
2. a citação (`quote`) contém literalmente o número e foi **conferida na fonte**
   (`verified_at_source`, ligado só pela ferramenta de conferência);
3. a evidência é sobre **o mesmo modelo e versão** que rodamos — evidência sobre produto comercial
   que detecta o mesmo achado não se transfere para um modelo aberto;
4. a validação publicada é externa ao conjunto de treino daquele modelo, ou o registro declara o
   vazamento (§8).

Diretrizes de avaliação usadas para julgar artigos: PROBAST+AI (Moons et al., BMJ 2025,
DOI 10.1136/bmj-2024-082505).

## 7. Subgrupos

`subgroups: [sex, age_band, view]`. Toda avaliação reporta o desempenho por sexo, faixa etária
e incidência (ou o equivalente da modalidade) **quando o dataset tem o metadado**; quando não tem,
o relatório diz "subgrupo não avaliável: metadado ausente". Subgrupo com menos casos que os
mínimos do §2 é reportado com a contagem e sem julgamento.

## 8. Política de vazamento

Cada registro declara `leakage`: `none`, `possible` ou `known`, com `leakage_note`.

| Situação | Consequência |
|---|---|
| Medição externa com `leakage: none` aprovada | pode sustentar **A** |
| Medição externa com `leakage: possible` ou `known` aprovada | no máximo **C** |
| Medição externa limpa reprovada (e nenhuma limpa aprovada) | condição **refutada**: cai para **D** mesmo com publicação favorável |

Fatos verificados que este protocolo aplica:

1. **SIIM-ACR (pneumotórax) e RSNA (pneumonia) derivam do NIH ChestX-ray14.** São anotações de
   imagens do NIH (Filice et al. 2020, DOI 10.1007/s10278-019-00299-9; Shih et al. 2019,
   DOI 10.1148/ryai.2019180041). Nenhum modelo treinado com NIH pode ser avaliado nelas como
   externo; o mesmo vale para os rótulos do Google sobre imagens do NIH.
2. **O TorchXRayVision não usa o split oficial do NIH.** Seus scripts de treino separam 80/20 por
   `GroupShuffleSplit` aleatório agrupado por paciente (semente 0). Reproduzindo esse split, a
   maior parte das 25.596 imagens do teste oficial do NIH (cerca de 20,7 mil) cai na parte de
   **treino** dos pesos `nih` e `all`. Avaliar esses pesos no teste oficial do NIH é `leakage:
   known`.
3. **Pares externos válidos para o teste oficial do NIH:** os pesos `pc`, `chex`, `mimic_nb` e
   `mimic_ch` — nenhum inclui NIH no treino.
4. **Mamografia:** CBIS-DDSM foi usado no pré-treino do `ianpan/mammoscreen`; medição nele é
   `known`. CMMD não aparece no cartão do modelo e é externo.
5. **Ultrassom de mama:** BUSI tem imagens duplicadas, inclusive entre classes (carta de
   Pawłowska et al. 2023, DOI 10.1016/j.dib.2023.109247) e não tem identificador de paciente;
   nunca é conjunto externo, e medições nele usam o subconjunto limpo da carta.
6. **Validação cruzada dentro do mesmo dataset** (ex.: sonda de TB treinada e testada em
   Shenzhen) é medição **interna** (C), nunca externa.

Duplicatas conhecidas são removidas antes de medir, e a remoção é registrada no datasheet.

## 9. Relato completo, incluindo falhas

- Todo registro medido carrega proveniência: git sha, `protocol_id`, hash do manifesto do
  dataset, sha256 dos pesos, comando e data. Sem isso o esquema rejeita o registro.
- **Falhas são publicadas.** Registros reprovados e condições refutadas aparecem no catálogo e no
  KPI; não são apagados nem re-executados até passar.
- Rótulos sem cabeça treinada, achados sem rótulo correspondente e subgrupos sem metadado vão
  para `not_evaluated` com o motivo — nunca somem em silêncio.
- Relato segundo STARD-AI (Sounderajah et al., Nat Med 2025, DOI 10.1038/s41591-025-03953-8) e
  CLAIM 2024 (Tejani et al., Radiol Artif Intell 2024, DOI 10.1148/ryai.240300).
- O padrão de referência é declarado com o vocabulário do esquema (`radiologist_adjudicated`,
  `radiologist_single_read`, `report_nlp`, `histopathology`, `expert_segmentation`,
  `clinical_reference_measurement`, `other`). AUROC contra rótulo minerado por NLP é
  concordância com rótulo ruidoso, e o relatório diz isso.

## 10. O que este protocolo não cobre

- Utilidade clínica, efeito no leitor ou no desfecho do paciente (portões G3 e G4 do roadmap).
- Desempenho em população brasileira (G2).
- Equivalência a radiologista.

Ver o plano de avaliação clínica: [`regulatory/10-clinical-evaluation-plan.md`](regulatory/10-clinical-evaluation-plan.md).
