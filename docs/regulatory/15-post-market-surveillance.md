---
doc_id: REG-15
title: Plano de Vigilância Pós-Mercado
status: RASCUNHO — v0.2 (plano para o portão G6)
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

Instrumentos já existentes no código: a trilha de auditoria (`audit/trail.py`; no banco,
`persistence/audit.py`), a trava de regressão (`radiologyai check-regression`, que compara um
`metrics.json` novo com a linha de base e falha fechada), o ECE e o diagrama de confiabilidade
(`calibration/reliability.py`) e a abstenção por motivo (`calibration/abstention.py`).

O retorno estruturado do médico deve seguir o formato do perfil **IHE AI Result Assessment
(AIRA)**, que existe para "post market surveillance, detection of data drift".

## 3. Indicadores

| Indicador | Gatilho de investigação (a fixar em protocolo antes de G4) |
|---|---|
| Taxa de rejeição pelo médico, por condição | mudança relevante em relação à linha de base do piloto |
| Concordância médico × sugestão, por condição e subgrupo | queda em relação à avaliação de G2 |
| Taxa de abstenção por motivo | aumento súbito (novo equipamento, novo protocolo) |
| Distribuição de escores | deslocamento em relação à janela de referência |
| Distribuição de metadados de entrada | fabricante, kVp, incidência, lateralidade (REQ-043) e taxa de entradas rejeitadas fora do padrão de referência |
| Calibração (ECE, diagrama de confiabilidade) em amostra auditada com padrão de referência médico | piora em relação à avaliação vigente |
| Regressão de versão | nova versão reprovada por `check-regression` ou pelos critérios de [REG-10](10-clinical-evaluation-plan.md) ⇒ atualização bloqueada |
| Incidentes | qualquer incidente com potencial de dano ⇒ investigação imediata |

Os limiares de gatilho **não** são fixados agora: sem dado de piloto, qualquer número seria
inventado. Eles entram num protocolo de vigilância pré-registrado antes de G4.

## 4. Ações

Monitoramento reativo: canal de queixas técnicas e incidentes com registro estruturado vinculado
ao git sha, ao sha256 dos pesos e ao identificador da execução auditada, tratado pelo processo de
[REG-14](14-change-control.md). Eventos com potencial de dano (RISK-001, RISK-002) têm prioridade
máxima.

- Rebaixamento de nível no catálogo quando nova medição externa reprovar (regra de
  `compute_level`: medição própria prevalece).
- Desativação de uma condição ou modelo (status `planned`) sem afetar o restante.
- Mudança pelo processo de [REG-14](14-change-control.md); se o software for regularizado,
  avaliação de necessidade de petição (RDC 657/2022 Art. 16) e de notificação de tecnovigilância
  (a v0.1 de `main` citava a RDC 67/2009 e o NOTIVISA; norma vigente e prazos a confirmar na fonte
  antes de G5).
- Atualização do arquivo de risco e da IFU.
- CAPA: suspensão preventiva dos pesos afetados, revisão de limiares de abstenção e aviso de
  segurança aos usuários habilitados.

## 5. Relatórios

No estágio atual (pesquisa, sem implantação), a vigilância se limita à verificação automatizada
no CI (`check_honesty.py`, `trace.py --check`, `soup.py --check`, trava de regressão) e à
varredura semanal de SOUP.

Relatório periódico de vigilância (periodicidade a definir em G4) com indicadores, incidentes,
ações e mudanças de nível. Obrigações formais de relato ao regulador só se aplicam após
regularização e serão incorporadas então.

---

## Histórico de revisões

| Versão | Data | Alteração |
|---|---|---|
| v0.1 | 2026-11 | Criação |
| v0.2 | 2026-11-27 | Consolidação com `15-post-market-surveillance-plan.md` de `main` (removido): instrumentos existentes, metadados de entrada, calibração, bloqueio de regressão, canal reativo, CAPA, tecnovigilância a confirmar, estágio atual |
