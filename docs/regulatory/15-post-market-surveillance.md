---
doc_id: REG-15
title: Plano de Vigilância Pós-Mercado
status: RASCUNHO — v0.1 (plano para o portão G6)
date: 2026-11
author: Dr. Guilherme Capel Pasqua (CRM-SP 175873)
depends_on: [REG-03, REG-04, REG-14]
---

# Plano de Vigilância Pós-Mercado

> Não há mercado: o software é de pesquisa e não é dispositivo médico. Este plano existe porque
> os requisitos de projeto que ele impõe (registro por inferência, retorno estruturado do médico,
> monitoramento de distribuição de entrada) precisam estar no software **antes** de G4/G6 —
> retrofitá-los depois é caro. Durante os pilotos (G4) ele funciona como vigilância do estudo.

## 1. Objetivos

1. Detectar **deriva** de desempenho e de distribuição de entrada (RISK-007).
2. Detectar **desempenho inferior em subgrupo** que os conjuntos de avaliação não mostraram.
3. Coletar e tratar **relatos de problema** e incidentes.
4. Realimentar o arquivo de risco, o catálogo (rebaixamento de nível) e o controle de mudanças.

## 2. Fontes de dados

| Fonte | O que fornece | Requisito de projeto |
|---|---|---|
| Trilha de auditoria por inferência | hash da entrada, modelo, pesos, git sha, saída, ponto de operação, usuário, carimbo de tempo | REQ-070, REQ-071 |
| Decisão do médico por achado | aceitar / editar / rejeitar; motivo; contorno corrigido | revisão e assinatura na plataforma (RISK-018) |
| Distribuição de entrada | fabricante, modelo de equipamento, protocolo, incidência, idade, sexo, taxa de abstenção por motivo | metadados retidos na des-identificação (Retain Device Identity, Retain Patient Characteristics) |
| Relatos de problema | issues e contatos | [REG-14](14-change-control.md) §4 |
| Literatura e avisos | novos artigos, retratações, avisos de segurança de modelos de terceiros, CVEs de SOUP | lista SOUP e varredura semanal |

O retorno estruturado do médico deve seguir o formato do perfil **IHE AI Result Assessment
(AIRA)**, que existe para "post market surveillance, detection of data drift".

## 3. Indicadores

| Indicador | Gatilho de investigação (a fixar em protocolo antes de G4) |
|---|---|
| Taxa de rejeição pelo médico, por condição | mudança relevante em relação à linha de base do piloto |
| Concordância médico × sugestão, por condição e subgrupo | queda em relação à avaliação de G2 |
| Taxa de abstenção por motivo | aumento súbito (novo equipamento, novo protocolo) |
| Distribuição de escores | deslocamento em relação à janela de referência |
| Incidentes | qualquer incidente com potencial de dano ⇒ investigação imediata |

Os limiares de gatilho **não** são fixados agora: sem dado de piloto, qualquer número seria
inventado. Eles entram num protocolo de vigilância pré-registrado antes de G4.

## 4. Ações

- Rebaixamento de nível no catálogo quando nova medição externa reprovar (regra de
  `compute_level`: medição própria prevalece).
- Desativação de uma condição ou modelo (status `planned`) sem afetar o restante.
- Mudança pelo processo de [REG-14](14-change-control.md); se o software for regularizado,
  avaliação de necessidade de petição (RDC 657/2022 Art. 16) e de notificação de tecnovigilância
  (procedimentos ANVISA a confirmar).
- Atualização do arquivo de risco e da IFU.

## 5. Relatórios

Relatório periódico de vigilância (periodicidade a definir em G4) com indicadores, incidentes,
ações e mudanças de nível. Obrigações formais de relato ao regulador só se aplicam após
regularização e serão incorporadas então.

---

## Histórico de revisões

| Versão | Data | Alteração |
|---|---|---|
| v0.1 | 2026-11 | Criação |
