---
doc_id: REG-14
title: Controle de Mudanças e Resolução de Problemas (IEC 62304 §6, §8, §9)
status: RASCUNHO — v0.2
date: 2026-11
author: Dr. Guilherme Capel Pasqua (CRM-SP 175873)
depends_on: [REG-05]
---

# Controle de Mudanças e Resolução de Problemas

> Software de pesquisa; não é dispositivo médico. Este processo já é executado pelo fluxo de PR
> e CI; aqui ele é descrito como processo.

## 1. Classes de mudança

| Classe | Exemplos | Exigências |
|---|---|---|
| **M0 — manutenção** | correção de bug sem efeito em saída, atualização de SOUP sem mudança de comportamento, documentação | PR com CI verde |
| **M1 — mudança de comportamento** | novo pré-processamento, nova regra de abstenção, mudança de interface | M0 + análise de impacto no arquivo de risco + requisito novo ou alterado com teste |
| **M2 — mudança de modelo ou evidência** | novo modelo, novos pesos, retreino, nova condição no catálogo, novo dataset de avaliação | M1 + manifesto com sha256 e licença + nova avaliação sob o protocolo vigente; nível recalculado, nunca herdado |
| **M3 — mudança de uso pretendido ou de protocolo** | nova indicação, nova população, mudança de limiar do protocolo | M2 + revisão de REG-01, REG-02, REG-04, REG-10; **novo `protocol_id`** para qualquer limiar |

Se o software um dia for regularizado: mudanças M2/M3 que "afetem significativamente as
funcionalidades clínicas, segurança e eficácia clínicos ou desempenho" ou criem novas indicações
exigem petição à ANVISA (RDC 657/2022 Art. 16); não há hoje mecanismo equivalente ao PCCP da FDA.
Por isso o registro de mudanças já segue o formato PCCP (descrição da modificação, protocolo da
modificação, avaliação de impacto) — para que um plano futuro possa ser escrito a partir de
histórico real.

## 2. Fluxo de uma mudança

Escopo: toda alteração em `src/radiologyai/`, `tests/`, `scripts/`, `docs/regulatory/`,
`pyproject.toml`, `uv.lock` e `.github/workflows/` segue este fluxo; nenhuma mudança entra sem PR
com motivo, arquivos afetados e impacto em requisitos e riscos.

1. **Proposta**: issue ou descrição de PR com a classe (M0–M3) e o motivo.
2. **Análise de impacto**: perigos afetados (`RISK-NNN`), requisitos afetados (`REQ-NNN`),
   condições do catálogo afetadas. Lista de verificação: muda o uso pretendido
   ([`01-intended-use.md`](01-intended-use.md), reabre classificação e risco)? cria perigo ou altera
   controle ([`04-risk-file.md`](04-risk-file.md))? altera requisito ou teste
   ([`06-requirements/`](06-requirements/), [`13-traceability.md`](13-traceability.md))? adiciona ou
   atualiza dependência ([`08-soup-list.md`](08-soup-list.md))?
3. **Implementação** em ramo próprio.
4. **Verificação**: portões do CI ([`09-verification-plan.md`](09-verification-plan.md)); para
   M2, artefato de avaliação novo em `artifacts/eval/`.
5. **Revisão e aprovação**: revisão de PR; para M2/M3, aprovação do responsável registrada na PR.
6. **Integração** em `main`; matriz de rastreabilidade e lista SOUP regeneradas pelos scripts.
7. **Liberação** (quando houver): etiqueta, notas de versão com as mudanças por classe.

## 3. Gestão de configuração

- Identificação: commit git; versão do pacote em `pyproject.toml`; `uv.lock`; sha256 de cada
  conjunto de pesos no manifesto ou model card; `protocol_id`; hash do manifesto de cada dataset.
- Todo resultado e toda evidência registram esses identificadores — é o que torna a mudança
  rastreável até a saída.
- Documentos gerados (`08-soup-list.md`, `13-traceability.md`, `reports/EVALUATION_*.md`,
  `levels.json`) nunca são editados à mão.

## 4. Resolução de problemas (§9)

| Etapa | Registro |
|---|---|
| Relato | issue com descrição, passos, versão (commit), modelo e pesos envolvidos — **sem dado de paciente** |
| Triagem | severidade pela escala do [plano de risco](03-risk-management-plan.md); perigo relacionado |
| Investigação | causa; outras versões ou condições afetadas |
| Correção | PR que referencia a issue; teste de regressão que falhava antes |
| Verificação | CI verde; para problema de desempenho, nova avaliação |
| Tendência | revisão periódica das issues fechadas para padrões (entrada de [REG-15](15-post-market-surveillance.md)) |

Fontes de anomalia: CI, testes, avaliação externa, varredura semanal de SOUP (pip-audit) e uso em
pesquisa. Classes de problema:

| Classe | Critério | Tratamento obrigatório |
|---|---|---|
| **Crítica (segurança)** | falha de controle de risco (`RISK-NNN`), vazamento de dado pessoal, hash inválido aceito, métrica fabricada, vulnerabilidade crítica em SOUP | bloqueio imediato de liberação; causa raiz; teste de regressão marcado com o `REQ-NNN`; atualização do arquivo de risco |
| **Alta (funcional / CI)** | quebra de inferência, teste reprovado, regressão contra a linha de base (`radiologyai check-regression`), portão de CI quebrado | correção em PR dedicada, com verificação completa, antes de novos recursos |
| **Moderada / baixa** | ergonomia, documentação não normativa, refatoração sem impacto em risco | backlog, rastreável por PR |

**Regra de não recorrência:** todo defeito de segurança ou de integridade de dado resolvido gera
um teste automatizado ou uma regra em `scripts/check_honesty.py` / `scripts/trace.py` que impeça
a reintrodução silenciosa.

Problema que afeta evidência já publicada (ex.: vazamento descoberto depois): o registro afetado
é marcado, o nível é recalculado e a mudança aparece no KPI — **não** se apaga o registro antigo.

## 5. Pendências

- Modelos de issue e de PR com os campos acima.
- Proteção de branch em `main`: ativa (conferido pela API do GitHub em 2026-11-27); falta
  documentar aqui as regras exatas (revisões e verificações obrigatórias).
- Transição para QMS (G5, ISO 13485 / RDC 665/2022): os registros de PRs, issues, commits e logs
  de CI deste repositório passam a compor a evidência histórica de controle de mudanças e CAPA;
  commits assinados a adotar.

---

## Histórico de revisões

| Versão | Data | Alteração |
|---|---|---|
| v0.1 | 2026-11 | Criação |
| v0.2 | 2026-11-27 | Consolidação com `14-change-control-and-problem-resolution.md` de `main` (removido): escopo, lista de impacto, classes de problema, não recorrência, transição para QMS |
