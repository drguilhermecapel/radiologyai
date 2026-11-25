doc_id: REG-14
title: Controle de Mudanças e Resolução de Problemas (IEC 62304 §8 e §9)
status: RASCUNHO — v0.1
date: 2026-11
author: Dr. Guilherme Capel Pasqua (CRM-SP 175873)
supersedes: —
# Controle de Mudanças e Resolução de Problemas

> **Estado atual:** software em estágio de pesquisa — não é dispositivo médico. Este procedimento define como mudanças de código/documentação e anomalias são controladas no núcleo v2 conforme **IEC 62304 §8 (Gestão de Configuração)** e **§9 (Processo de Resolução de Problemas)**. Ver [`HONEST_STATUS.md`](../../HONEST_STATUS.md).

## 1. Fluxo de controle de mudanças (IEC 62304 §8.2)

Toda alteração no núcleo v2 (`src/radiologyai/`, `tests/`, `scripts/`, `docs/regulatory/`, `pyproject.toml`, `uv.lock`, `.github/workflows/`) segue obrigatoriamente:
1. **Branch dedicada e Pull Request:** nenhuma mudança é integrada sem Pull Request com descrição do motivo, arquivos afetados e impacto em requisitos/riscos.
2. **Análise de impacto:** antes do merge, verificar se a mudança afeta:
   - [`01-intended-use.md`](01-intended-use.md) (mudança de escopo reabre classificação e risco);
   - [`04-risk-file.md`](04-risk-file.md) (novo perigo ou alteração de controle de risco);
   - `docs/regulatory/06-requirements/` e [`13-traceability.md`](13-traceability.md);
   - [`08-soup-list.md`](08-soup-list.md) (adição ou atualização de dependência).
3. **Verificação automatizada obrigatória:** todos os portões de `.github/workflows/ci.yml` e `.github/workflows/security.yml` devem estar verdes antes do merge.

## 2. Classificação e resolução de problemas (IEC 62304 §9)

Anomalias detectadas em CI, testes, avaliação externa, monitoramento semanal de SOUP (`pip-audit`) ou uso em pesquisa são registradas como Issues no GitHub com classificação de severidade:

| Classe | Critério | Tratamento obrigatório |
|---|---|---|
| **Crítica (Segurança)** | Falha de controle de risco (`RISK-001`…`RISK-010`), vazamento de PHI, hash inválido aceito, métrica fabricada ou vulnerabilidade crítica em SOUP | Bloqueio imediato de liberação; investigação de causa raiz; teste de regressão obrigatório vinculado a `REQ-xxx`; atualização de [`04-risk-file.md`](04-risk-file.md) |
| **Alta (Funcional / CI)** | Quebra de inferência, falha em teste unitário/integração, regressão de linha de base ou quebra de portão de CI | Correção em PR dedicado com verificação completa antes de novos recursos |
| **Moderada / Baixa** | Melhoria de ergonomia, documentação não normativa ou refatoração sem impacto em risco | Planejamento em backlog com rastreabilidade em PR |

## 3. Regra de não recorrência

Todo defeito de segurança ou integridade de dados resolvido deve resultar em um teste automatizado ou regra em `scripts/check_honesty.py` / `scripts/trace.py` que impeça sua reintrodução silenciosa.

## 4. Transição para QMS (Fase 6)

Na constituição do QMS sob ISO 13485 / RDC 665/2022, os registros de Pull Requests, Issues, commits assinados e logs de CI deste repositório comporão a evidência histórica de controle de mudanças e CAPA.

## Histórico de revisões

| Versão | Data | Alteração |
|---|---|---|
| v0.1 | 2026-11 | Criação do procedimento de Controle de Mudanças e Resolução de Problemas (Fase 2, IEC 62304 §8–9) |
