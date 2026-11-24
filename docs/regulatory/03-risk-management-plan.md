---
doc_id: REG-03
title: Plano de Gestão de Risco (ISO 14971:2019)
status: RASCUNHO — v0.1
date: 2026-11
author: Dr. Guilherme Capel Pasqua (CRM-SP 175873)
depends_on: [REG-01, REG-02]
---

# Plano de Gestão de Risco

> Software de pesquisa, não é dispositivo médico. Este plano existe porque a gestão de risco
> precisa começar antes do produto, e porque a RDC 848/2024 (Art. 50) e a RDC 665/2022 (Art. 18)
> exigem processo contínuo de gestão de risco de qualquer fabricante. Nenhuma conformidade é
> alegada.

## 1. Escopo

Aplica-se ao software RadiologyAI inteiro — núcleo, camada de análise por modalidade, catálogo de
evidência e plataforma web — em todas as fases do ciclo de vida descritas no
[plano de desenvolvimento](05-development-plan.md) e nos portões G0–G6 do
[`ROADMAP.md`](../../ROADMAP.md) §4. O uso pretendido considerado é o de
[REG-01](01-intended-use.md) v0.2 (pesquisa) **e** o uso clínico assistivo hipotético da posição
de trabalho de REG-01 §7, porque a análise feita só para o uso de pesquisa subestimaria os
perigos que o desenho precisa controlar desde já.

## 2. Normas e referências

- ISO 14971:2019 — aplicação da gestão de risco a dispositivos médicos.
- ISO/TR 24971 — orientação para a ISO 14971.
- AAMI CR34971:2022 (reconhecida pela FDA) — aplicação da ISO 14971 a IA/aprendizado de máquina;
  usada como lista de verificação de modos de falha de ML (qualidade de dado, viés, confiança
  excessiva, deriva).
- IEC 62304 Ed. 1.1 §4.3 e §7 — classificação de segurança e gestão de risco de software
  ([REG-02](02-software-safety-classification.md): Classe C).
- IEC 81001-5-1:2021 — segurança no ciclo de vida (riscos de cibersegurança, [REG-11](11-cybersecurity-lgpd.md)).

## 3. Responsabilidades

| Papel | Quem | Responsabilidade |
|---|---|---|
| Responsável pela gestão de risco | Dr. Guilherme Capel Pasqua | aprova o plano, o arquivo de risco e a aceitabilidade do risco residual |
| Engenharia (humana ou agentes) | contribuidores do repositório | identificam perigos ao propor mudança; implementam controles como requisitos `REQ-NNN` com teste |
| Revisão clínica | médico revisor (a designar por modalidade, a partir de G2) | revisa severidade e plausibilidade clínica das sequências |

Enquanto não houver equipe, o responsável acumula os papéis; o registro de cada decisão no git é
a evidência.

## 4. Critérios de aceitabilidade

### 4.1 Severidade

| Nível | Descrição |
|---|---|
| S1 — desprezível | inconveniência, sem efeito no cuidado |
| S2 — menor | atraso ou exame adicional desnecessário, sem lesão |
| S3 — séria | lesão ou deterioração que exige intervenção médica |
| S4 — crítica | deterioração irreversível ou intervenção cirúrgica desnecessária |
| S5 — catastrófica | morte |

### 4.2 Probabilidade

Não há dado de campo; a probabilidade é estimada **qualitativamente** até G2/G4 produzirem
frequências observadas. Escala P1 (improvável) a P5 (frequente). Para falhas de software, este
plano **não credita baixa probabilidade de falha do próprio software**: assume que a falha ocorre
e estima apenas a probabilidade de ela levar a dano (prática da IEC 62304 para classificação; o
texto exato da cláusula na Ed. 1.1 está a confirmar).

### 4.3 Matriz

| | P1 | P2 | P3 | P4 | P5 |
|---|---|---|---|---|---|
| S5 | inaceitável | inaceitável | inaceitável | inaceitável | inaceitável |
| S4 | ALARP | inaceitável | inaceitável | inaceitável | inaceitável |
| S3 | ALARP | ALARP | inaceitável | inaceitável | inaceitável |
| S2 | aceitável | ALARP | ALARP | inaceitável | inaceitável |
| S1 | aceitável | aceitável | aceitável | ALARP | ALARP |

ALARP = reduzir tanto quanto praticável e justificar o residual. Política adicional, mais estrita
que a matriz: **todo perigo de severidade S4 ou S5 exige ao menos um controle por projeto
inerente** (não apenas informação ao usuário), implementado como requisito com teste.

## 5. Atividades

| Atividade | Quando | Saída |
|---|---|---|
| Identificação de perigos | a cada nova modalidade, modelo, condição do catálogo ou mudança de uso pretendido | linha nova em [`04-risk-file.md`](04-risk-file.md) |
| Estimativa de risco | idem | S × P antes do controle |
| Controle de risco | idem | requisito `REQ-NNN` em `06-requirements/` com `risk_controls: [RISK-NNN]` |
| Verificação do controle | a cada PR | teste marcado `@pytest.mark.requirement`; `scripts/trace.py --check` no CI |
| Risco residual | ao fechar cada portão G | avaliação registrada no arquivo de risco |
| Risco-benefício | antes de G4 e de G5 | relatório de risco-benefício (ainda não existe: não há benefício clínico demonstrado) |
| Revisão pós-produção | G6 | entrada de dados de vigilância ([REG-15](15-post-market-surveillance.md)) |

## 6. Ordem de preferência dos controles (ISO 14971 §7.1)

1. **Segurança inerente por projeto** — ex.: gate de escopo antes do modelo, nível de evidência
   calculado, falha fechada.
2. **Medidas de proteção** no software ou no processo — ex.: abstenção, auditoria encadeada,
   modo pesquisa separado.
3. **Informação de segurança** — IFU ([REG-12](12-labeling-ifu.md)), avisos na interface.

Informação ao usuário nunca é o único controle de perigo S4/S5.

## 7. Rastreabilidade

`RISK-NNN` (arquivo de risco) → `REQ-NNN` (requisito, campo `risk_controls`) → teste
(`@pytest.mark.requirement`). A matriz é gerada por `scripts/trace.py` em
[`13-traceability.md`](13-traceability.md) e verificada no CI. Os identificadores `RISK-001` a
`RISK-010` correspondem um a um aos perigos-semente `H-01` a `H-10` do roadmap.

## 8. Itens específicos de IA (AAMI CR34971)

O arquivo de risco cobre explicitamente: saída errada confiante; viés de automação; mudança de
distribuição (equipamento, população, protocolo de aquisição); desempenho inferior em subgrupo;
falha silenciosa em entrada fora de distribuição; vazamento entre treino e teste que superestima
a evidência; e deriva pós-implantação.

---

## Histórico de revisões

| Versão | Data | Alteração |
|---|---|---|
| v0.1 | 2026-11 | Criação |
