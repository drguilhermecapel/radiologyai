---
doc_id: REG-02
title: Classificação de Segurança do Software (IEC 62304 §4.3)
status: RASCUNHO — v0.1
date: 2026-09
author: Dr. Guilherme Capel Pasqua (CRM-SP 175873)
depends_on: [REG-01]
---

# Classificação de Segurança do Software

## 1. Critério normativo

IEC 62304 §4.3 define três classes segundo o dano possível decorrente de uma falha do software,
**assumindo que a falha ocorra**:

| Classe | Critério |
|---|---|
| **A** | Nenhuma lesão ou dano à saúde é possível |
| **B** | Lesão não-séria é possível |
| **C** | Morte ou lesão séria é possível |

A norma permite classificar em nível mais baixo quando um controle de risco **externo ao
software** reduz o dano — desde que a eficácia desse controle seja justificada e verificada.

## 2. O argumento para Classe B

Seria possível sustentar Classe B da seguinte forma:

> O software é assistivo e não autônomo. Um médico qualificado lê todo exame e toma toda
> decisão. Uma falha do software não alcança o paciente sem atravessar julgamento humano
> independente. Portanto o dano residual está limitado a não-sério.

## 3. Decisão: **Classe C**

Rejeitamos o argumento de §2 e adotamos **Classe C**. Razões, em ordem de peso:

### 3.1 O uso pretendido inclui achados letais

REG-01 §2 coloca **pneumotórax** no escopo da v1, e as versões planejadas incluem
**hemorragia intracraniana** em TC e **nódulo pulmonar**. Um falso negativo que ancore o
leitor é plausivelmente fatal. A norma pergunta se morte ou lesão séria é **possível**,
não se é provável.

### 3.2 O controle externo invocado é justamente o que a literatura contesta

"O radiologista pega o erro" é a alegação central que a literatura de **viés de automação**
disputa: leitores expostos a uma saída negativa de IA demonstravelmente perdem achados que
identificariam sem assistência. Para rebaixar a classe com base nesse controle, §4.3 exige
justificar sua eficácia. Não temos essa evidência, e a posição honesta é que a leitura
assistida por IA **muda o comportamento do leitor** — o que é precisamente o mecanismo do
perigo H-02 no arquivo de risco.

### 3.3 O custo é assimétrico

Ser questionado sobre uma classificação B no meio de uma avaliação regulatória significa
retrofitar documentação de projeto em nível de unidade e verificação de unidade sobre um
código já pronto. Construir para Classe C a partir de um diretório vazio custa documentação
que deveríamos produzir de qualquer forma.

### 3.4 O delta técnico é pequeno neste projeto

Classe C acrescenta sobre Classe B:

| Requisito adicional | Custo real aqui |
|---|---|
| §5.3.3–5.3.6 — arquitetura detalhada com segregação de SOUP | A lista SOUP já é gerada de `uv.lock`; a segregação já está no desenho de `radiologyai/` |
| §5.4.2 — documentação de projeto em nível de unidade | Docstrings + `07-architecture.md`, que existiriam de qualquer modo |
| §5.5.3 — verificação de unidade com critérios de aceitação definidos | O CI já exige testes unitários com portão de cobertura; o delta é escrever os critérios |

### 3.5 A saída existe e é barata; a entrada não

Rebaixar depois é fácil: se o uso pretendido for estreitado — *"segunda leitura apenas,
achados não time-critical, sem alegação de triagem, excluindo pneumotórax"* — documenta-se
o rebaixamento para B com justificativa. **Promover depois é reescrita.**

## 4. Fronteira de reclassificação

Esta classificação é revisada obrigatoriamente se qualquer item de REG-01 §6 ocorrer.
Em particular, passar a **operação autônoma** ou a **triagem/notificação automática**
elimina o argumento de revisão humana por completo e não admite discussão sobre Classe C.

## 5. Consequências operacionais imediatas

Como Classe C, valem desde o primeiro commit do núcleo v2:

1. Todo requisito em `06-requirements/` tem id, controle de risco vinculado e teste verificador.
2. `scripts/trace.py` **quebra o CI** se algum requisito, perigo ou controle ficar órfão.
3. Cobertura de teste unitário com portão mínimo, com critérios de aceitação declarados por unidade.
4. Lista SOUP (`08-soup-list.md`) gerada de `uv.lock` + consulta de anomalias publicadas (OSV/CVE),
   verificada no CI. É por isso que um `requirements.txt` apenas com `>=` é defeito de
   conformidade, e não desleixo.
5. Nenhum caminho de degradação silenciosa. Dependência ausente, pesos ausentes ou hash
   divergente → falha alta e explícita.

## 6. Referências

- IEC 62304 — Medical device software — Software life cycle processes, §4.3
- ISO 14971:2019 — Application of risk management to medical devices
- ISO/TR 24971 — Guidance on the application of ISO 14971
- ANVISA RDC 657/2022; RDC 751/2022 (Regra 11)
- IMDRF — *Software as a Medical Device: Possible Framework for Risk Categorization*

---

## Histórico de revisões

| Versão | Data | Alteração |
|---|---|---|
| v0.1 | 2026-09 | Criação. Classe C adotada com a justificativa de §3 |
