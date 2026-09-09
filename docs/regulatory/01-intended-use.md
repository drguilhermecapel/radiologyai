---
doc_id: REG-01
title: Declaração de Uso Pretendido / Indicações de Uso
status: RASCUNHO — v0.1
date: 2026-09
author: Dr. Guilherme Capel Pasqua (CRM-SP 175873)
supersedes: —
---

# Uso Pretendido / Indicações de Uso

> **Estado atual: o dispositivo descrito neste documento NÃO EXISTE.** Este é o documento-alvo
> que define o que será construído e validado. Nenhum modelo foi treinado; nenhuma indicação
> pode ser alegada hoje. Ver [`HONEST_STATUS.md`](../../HONEST_STATUS.md).

Este é o documento fundador do arquivo técnico. Ele fixa modalidade, região anatômica, população,
usuário, cenário de cuidado, a natureza da alegação e as contraindicações. Dele derivam:
a classe de risco ANVISA (RDC 751/2022, Regra 11), a classe de segurança de software
(IEC 62304 §4.3), os requisitos de dados de validação, e o que o `ModalityPlugin.validate()`
deve rejeitar em tempo de execução.

**Nenhuma alteração neste documento é trivial.** Toda mudança de uso pretendido reabre a
classificação de risco, a análise de perigos e o plano de avaliação clínica.

---

## 1. Uso pretendido (intended use)

O RadiologyAI é um software autônomo (SaMD) destinado a **auxiliar profissionais médicos
qualificados na interpretação de imagens radiológicas**, fornecendo informação de suporte
à decisão sob a forma de escores de probabilidade por achado, mapas de saliência e um
rascunho de laudo estruturado.

O software **não produz diagnóstico**. Produz informação que um médico qualificado
interpreta, aceita, edita ou rejeita.

## 2. Indicações de uso — por versão

O escopo é liberado em série. Cada indicação abaixo só passa a valer quando sustentada por
um relatório de validação externa em `artifacts/eval/` e por um ciclo de verificação concluído.

### v1 — Radiografia de tórax (a primeira indicação a buscar)

| Campo | Definição |
|---|---|
| **Modalidade** | Radiografia computadorizada/digital de tórax (DICOM Modality `CR` ou `DX`) |
| **Região anatômica** | Tórax |
| **Incidência** | Frontal (PA e AP). **Perfil e incidências especiais estão FORA do escopo** |
| **População** | Adultos (≥ 18 anos). **População pediátrica está FORA do escopo** |
| **Usuário pretendido** | Médico radiologista ou médico com competência em interpretação de radiografia de tórax. Não se destina a leigos, pacientes ou profissionais não médicos |
| **Cenário de cuidado** | Ambiente ambulatorial e hospitalar, em fluxo de leitura **concorrente** (o médico lê o exame; o software oferece informação adicional) |
| **Achados no escopo** | A serem fixados a partir do conjunto de rótulos validado, dentro de: atelectasia, cardiomegalia, consolidação, derrame pleural, edema, enfisema, fibrose, hérnia, infiltrado, massa, nódulo, espessamento pleural, pneumonia, pneumotórax |
| **Natureza da alegação** | **Assistiva e concorrente.** Não é triagem, não é priorização de fila, não é notificação automática, não é leitura autônoma, não é *rule-out* |

### v2 — Ecocardiografia (planejada)
### v3 — Tomografia computadorizada (planejada)
### v4 — Ressonância magnética (planejada)

Cada uma exige seu próprio ciclo completo de dados, treino, validação externa, análise de
risco e verificação, e sua própria adição a este documento antes de qualquer alegação.

## 3. Contraindicações e limitações declaradas

O software **não deve ser usado**:

1. **Para excluir doença** (*rule-out*). Um escore baixo não é evidência de ausência de achado.
2. **Como única base de qualquer decisão clínica.** Toda saída exige revisão médica.
3. **Em população pediátrica.**
4. **Em incidências fora do escopo declarado** (perfil, decúbito lateral, incidências especiais).
5. **Em modalidade fora da declarada** para a versão em uso.
6. **Para priorização de fila ou notificação de achado crítico**, salvo se e quando uma
   indicação de triagem for separadamente validada e registrada — o que implica reclassificação.
7. **Em imagens de qualidade insuficiente**, rotação significativa, artefato relevante ou
   penetração inadequada. O software deve rejeitar esses casos, não degradar silenciosamente.

## 4. Controles de risco embutidos no uso pretendido

Estes não são recursos opcionais. São condições do uso pretendido e requisitos de projeto
rastreáveis (ver `04-risk-file.md` e `06-requirements/`):

| Controle | Descrição |
|---|---|
| **Rejeição de entrada fora de distribuição** | `ModalityPlugin.validate()` rejeita modalidade, incidência, idade ou qualidade fora do escopo. Falha fechada, sem predição |
| **Banda de abstenção** | Saída em três bandas: *achado provável* / **não avaliável** / *achado improvável*. A banda intermediária é uma resposta legítima, não uma falha |
| **Proibição de afirmar normalidade** | O software nunca exibe "Normal" como afirmação positiva. Ausência de achado provável ≠ exame normal |
| **Incerteza sempre visível** | Escores calibrados com intervalo; nunca um rótulo *top-1* isolado |
| **Verificação de integridade do modelo** | sha256 dos pesos verificado na carga; divergência → falha, nunca fallback |
| **Explicabilidade real** | Mapa de saliência derivado de gradientes do modelo efetivamente usado. Saliência simulada é proibida por verificação de CI |
| **Trilha de auditoria** | Todo resultado registra hash da entrada, versão do modelo, sha256 dos pesos, git sha, saída, e a decisão do médico |

## 5. O que NÃO é alegado

- Não se alega desempenho equivalente ou superior ao de radiologista.
- Não se alega redução de tempo de leitura.
- Não se alega desempenho em populações, equipamentos ou instituições não representados
  nos conjuntos de validação, e a composição desses conjuntos é declarada no relatório
  de avaliação.
- Não se alega funcionamento em modalidade, incidência ou faixa etária fora de §2.

## 6. Fronteira que reclassifica o dispositivo

Qualquer uma das mudanças abaixo **altera a classe de risco e a classe de segurança de
software** e exige reabrir todo o arquivo técnico:

- passar de leitura concorrente para **triagem, priorização de fila ou notificação automática**;
- passar de assistivo para **autônomo** (laudo sem revisão médica);
- alegar uso para **exclusão** de achado;
- incluir **população pediátrica**;
- incluir achados com risco imediato de morte sob alegação de detecção sem revisão humana.

Ver `02-software-safety-classification.md` §4.

## 7. Enquadramento regulatório pretendido

- **Brasil (ANVISA):** software como dispositivo médico sob a RDC 657/2022, classificado
  pela Regra 11 da RDC 751/2022. Posição de trabalho: **Classe III**, sujeita a **registro**
  (não notificação). A confirmar com consultoria regulatória antes de comprometer o caminho.
- **Classe de segurança de software:** **IEC 62304 Classe C** (ver `02-...`).
- **Gestão de risco:** ISO 14971:2019 + ISO/TR 24971.
- **QMS:** ISO 13485 / RDC 665/2022 — a ser implementado quando houver veículo societário.

---

## Histórico de revisões

| Versão | Data | Alteração |
|---|---|---|
| v0.1 | 2026-09 | Criação. Rascunho inicial, pré-desenvolvimento do núcleo v2 |
