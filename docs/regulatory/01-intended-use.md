---
doc_id: REG-01
title: Declaração de Uso Pretendido / Indicações de Uso
status: RASCUNHO — v0.2
date: 2026-11
author: Dr. Guilherme Capel Pasqua (CRM-SP 175873)
supersedes: v0.1 (2026-09)
---

# Uso Pretendido / Indicações de Uso

> **Estado atual: software de pesquisa. Não é dispositivo médico.** Não é regularizado na
> ANVISA, não tem clearance FDA nem marcação CE. Nenhuma indicação clínica é alegada. Ver
> [`HONEST_STATUS.md`](../../HONEST_STATUS.md).

Este é o documento fundador do arquivo técnico. Ele fixa o que o software faz, para quem, sobre
quais imagens, e o que ele **não** faz. Dele derivam a classe de risco ANVISA (RDC 751/2022,
Regra 11), a classe de segurança de software (IEC 62304 §4.3, ver REG-02), os requisitos de
dados de validação ([`../protocolo-evidencia.md`](../protocolo-evidencia.md)) e o que os plugins
de modalidade devem rejeitar em tempo de execução.

**Nenhuma alteração neste documento é trivial.** Toda mudança reabre a classificação de risco, a
análise de perigos ([`04-risk-file.md`](04-risk-file.md)) e o plano de avaliação clínica
([`10-clinical-evaluation-plan.md`](10-clinical-evaluation-plan.md)).

---

## 1. Uso pretendido atual — pesquisa

O RadiologyAI é uma **plataforma de pesquisa** que executa modelos abertos de análise de imagem
sobre exames **de pesquisa** (datasets públicos ou dados des-identificados obtidos sob protocolo
aprovado em CEP) e apresenta, para cada condição de um catálogo, um **achado sugerido para
confirmação médica** acompanhado do seu **nível de evidência** (A, B, C ou D), da fonte dessa
evidência e da incerteza.

Finalidades permitidas:

1. medir e publicar a força da evidência de modelos abertos por condição (portão G1 do roadmap);
2. conduzir estudos retrospectivos aprovados em CEP (G2), estudos de leitores (G3) e pilotos
   silenciosos (G4) — cada um sob o seu protocolo;
3. desenvolver e verificar o software segundo IEC 62304 Classe C, como preparação de um eventual
   dossiê regulatório.

## 2. Escopo multimodal de pesquisa (indicações de pesquisa)

| Grupo | Modalidade DICOM | Região / exame | Exemplos de condições do catálogo | Restrições de entrada |
|---|---|---|---|---|
| RX de tórax | `CR`, `DX` | tórax, incidência frontal (PA/AP) | achados do TorchXRayVision, índice cardiotorácico, tuberculose | adulto; perfil e incidências especiais rejeitados; gate de distribuição |
| RX musculoesquelético | `CR`, `DX` | extremidades | fratura | adulto e pediátrico **somente** quando o conjunto de validação da condição incluir a faixa |
| TC | `CT` | crânio, tórax, abdome | hemorragia intracerebral, derrame pleural/pericárdico, lesões hepáticas, biomarcadores em HU | pré-condições de aquisição por biomarcador (fase, kVp, kernel, espessura); senão abstenção |
| RM | `MR` | encéfalo, abdome | subregiões de glioma **conhecido**, carga lesional de substância branca, lesões hepáticas | sequências exigidas presentes e co-registradas; nunca usado como detector de tumor |
| Mamografia | `MG` | mama, CC + MLO | escore de malignidade, densidade | VOI LUT aplicado; recorte; rastreamento **fora** do escopo |
| Ultrassonografia | `US` | mama, fígado | lesão mamária, esteatose, segmentação interativa | imagem B-mode; anotações gravadas no pixel tratadas como risco de vazamento |
| Ecocardiografia | `US` | apical 2 e 4 câmaras | traçado assistido do VE e FE por Simpson | **somente modo pesquisa**; contorno confirmado pelo médico |

Condições de nível **D** (zero-shot, sem medição aprovada por condição, refutadas ou limitadas)
aparecem **somente no modo pesquisa** e nunca entram no laudo, no PDF, no SR, na SEG, no FHIR nem
na exportação ao PACS (REQ-091).

### 2.1 Usuário pretendido

Médico (radiologista, cardiologista ecocardiografista ou médico com competência na modalidade)
atuando como **pesquisador** ou **leitor de estudo**, e engenheiros/pesquisadores no papel de
avaliação técnica. Não se destina a leigos, pacientes ou profissionais não médicos para
interpretação.

## 3. Contraindicações e o que o software NÃO faz

### 3.1 O que o software não é e não faz

- **Não produz diagnóstico.** Produz achados sugeridos que um médico interpreta, confirma, edita
  ou rejeita.
- **Não se destina ao cuidado de pacientes.** Não deve informar, tomar ou influenciar decisão
  diagnóstica ou terapêutica sobre paciente real fora de um protocolo de pesquisa aprovado — e,
  mesmo dentro de um, no piloto silencioso (G4) as saídas ficam **ocultas** do médico assistente.
- **Não faz triagem, priorização de fila, notificação de achado crítico, rastreamento, laudo
  autônomo nem exclusão de doença (*rule-out*).**
- **Não se comunica com o paciente** (Resolução CFM nº 2.454/2026, Art. 5 §2).
- **Não afirma normalidade.** Ausência de achado sugerido ≠ exame normal (REQ-063).

Linguagem vedada em interface, documentação e divulgação: "detecta", "diagnostica", "exclui",
"triagem", "rastreia", "prioriza", "auxilia o radiologista no cuidado", "validado",
"clinicamente validado", "aprovado", "certificado", "equivalente a radiologista".

### 3.2 Contraindicações

O software **não deve ser usado**:

1. em imagem identificável de paciente atendido na rotina, fora de protocolo CEP aprovado;
2. para excluir doença, ainda que o ponto de operação de exclusão exista no catálogo;
3. como única base de qualquer conclusão — toda saída exige revisão médica;
4. fora das restrições de entrada do §2; o software deve rejeitar, não degradar;
5. com pesos de licença não comercial em qualquer contexto comercial
   ([`MODEL_LICENSES.md`](../../MODEL_LICENSES.md));
6. em imagem de qualidade insuficiente, artefato relevante, ou modalidade/sequência fora da
   declarada para a condição.

## 4. Controles de risco embutidos no uso pretendido

| Controle | Descrição | Requisitos |
|---|---|---|
| Gate de escopo antes do modelo | modalidade, incidência, idade ou qualidade fora do escopo ⇒ rejeição com motivo, sem predição | REQ-031, REQ-042 |
| Nível de evidência calculado | A–D derivado de registros com proveniência; nunca digitado | REQ-090, REQ-093 |
| Visão principal sem nível D | exploratório só no modo pesquisa | REQ-091 |
| Decisão só com ponto de operação de origem declarada | sem ponto ⇒ escore sem veredito | REQ-092 |
| Banda de abstenção | *provável* / **não avaliável** / *improvável*; a banda intermediária é resposta legítima | REQ-062 |
| Proibição de afirmar normalidade | — | REQ-063 |
| Integridade do modelo | sha256 dos pesos verificado na carga; divergência ⇒ falha | REQ-020, REQ-030 |
| Explicabilidade real | mapa só de gradiente ou máscara do modelo efetivamente usado | REQ-060 |
| Trilha de auditoria | entrada, modelo, pesos, git sha, saída, decisão do médico, encadeados por hash | REQ-070, REQ-071 |

## 5. O que NÃO é alegado

- Nenhum desempenho clínico. Números de desempenho existem apenas como registros de evidência do
  catálogo, cada um com dataset, n, IC e proveniência.
- Nenhuma equivalência ou superioridade a médico.
- Nenhum desempenho em população, equipamento ou instituição não representados no conjunto de
  avaliação — em particular, nenhum desempenho em população brasileira antes do portão G2.

## 6. Fronteira que reclassifica o software

Qualquer uma das mudanças abaixo transforma o software em dispositivo com finalidade clínica e
reabre todo o arquivo técnico:

- uso sobre paciente real com retorno do resultado ao cuidado (sai da isenção de investigação
  clínica da RDC 751/2022 Art. 2 §8 e passa a exigir regularização);
- triagem, priorização ou notificação automática;
- laudo autônomo, sem revisão médica;
- alegação de exclusão de achado.

Ver [`02-software-safety-classification.md`](02-software-safety-classification.md) §4.
A numeração das seções 2 (escopo), 4 (controles) e 6 (fronteira) é citada no código e em REG-02; mantê-la estável.

## 7. Enquadramento regulatório pretendido, se um dia for produto

- **Brasil:** SaMD sob a RDC 657/2022, classificado pela Regra 11 da RDC 751/2022. Posição de
  trabalho: **classe III** (registro), porque os achados de maior risco do catálogo puxam a
  classe — **a confirmar com consultoria regulatória**. A indicação seria liberada por condição,
  sustentada por G2–G4.
- **Classe de segurança de software:** IEC 62304 **Classe C** (REG-02).
- **Gestão de risco:** ISO 14971:2019 com AAMI CR34971:2022 para os modos de falha de
  aprendizado de máquina ([`03-risk-management-plan.md`](03-risk-management-plan.md)).
- **QMS:** ISO 13485 / RDC 665/2022, quando houver pessoa jurídica.

---

## Histórico de revisões

| Versão | Data | Alteração |
|---|---|---|
| v0.1 | 2026-09 | Criação. Escopo-alvo: RX de tórax assistivo e concorrente; demais modalidades planejadas |
| v0.2 | 2026-11 | Escopo reescrito como **pesquisa multimodal** sem alegação clínica; catálogo A–D; linguagem vedada; contraindicação de pesos não comerciais; fronteira de reclassificação atualizada |
