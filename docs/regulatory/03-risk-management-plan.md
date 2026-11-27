---
doc_id: REG-03
title: Plano de Gestão de Risco (ISO 14971:2019)
status: RASCUNHO — v0.2
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

Abrangência do ciclo de vida: concepção, engenharia de dados, treinamento ou ajuste de sondas,
avaliação (G1–G2), verificação, implantação controlada (G4), vigilância pós-mercado (G6) e
**descontinuação** (retirada de modelo, condição ou versão, com preservação dos registros).

Enquadramento de trabalho: IEC 62304 Classe C ([REG-02](02-software-safety-classification.md));
ANVISA classe III pela RDC 751/2022, Regra 11, se e quando houver produto — **a confirmar** com
consultoria regulatória antes de G5 ([`ROADMAP.md`](../../ROADMAP.md) §8.2).

## 2. Normas e referências

- ISO 14971:2019 — aplicação da gestão de risco a dispositivos médicos.
- ISO/TR 24971 — orientação para a ISO 14971.
- AAMI CR34971:2022 (reconhecida pela FDA) — aplicação da ISO 14971 a IA/aprendizado de máquina;
  usada como lista de verificação de modos de falha de ML (qualidade de dado, viés, confiança
  excessiva, deriva).
- IEC 62304 Ed. 1.1 §4.3 e §7 — classificação de segurança e gestão de risco de software
  ([REG-02](02-software-safety-classification.md): Classe C).
- IEC 81001-5-1:2021 — segurança no ciclo de vida (riscos de cibersegurança, [REG-11](11-cybersecurity-lgpd.md)).
- RDC 657/2022 (ANVISA, SaMD) e RDC 751/2022 (classificação) — enquadramento brasileiro.

## 3. Responsabilidades

| Papel | Quem | Responsabilidade |
|---|---|---|
| Responsável pela gestão de risco | Dr. Guilherme Capel Pasqua | aprova o plano, o arquivo de risco e a aceitabilidade do risco residual |
| Engenharia (humana ou agentes) | contribuidores do repositório | identificam perigos ao propor mudança; implementam controles como requisitos `REQ-NNN` com teste |
| Revisão clínica | médico revisor (a designar por modalidade, a partir de G2) | revisa severidade e plausibilidade clínica das sequências |

| Verificação automatizada | portões de CI (`scripts/trace.py`, `scripts/soup.py`, `scripts/check_honesty.py`, `pytest`, bandit, gitleaks) | evidência objetiva de que os controles continuam eficazes a cada mudança; mantidos como parte do *Design History File* |

Enquanto não houver equipe, o responsável acumula os papéis; o registro de cada decisão no git é
a evidência. Na transição pré-submissão (G5), a responsabilidade passa a um comitê de gestão de
risco sob QMS ISO 13485 / RDC 665/2022.

## 4. Critérios de aceitabilidade

### 4.1 Severidade

| Nível | Descrição | Exemplo no contexto deste software |
|---|---|---|
| S1 — desprezível | inconveniência, sem efeito no cuidado | indisponibilidade transitória com falha explícita (o médico lê sem apoio) |
| S2 — menor | atraso ou exame adicional desnecessário, sem lesão | repetição de exame; atraso leve sem impacto prognóstico |
| S3 — séria | lesão ou deterioração que exige intervenção médica | atraso diagnóstico clinicamente significativo, reversível; procedimento invasivo desnecessário de baixo risco |
| S4 — crítica | deterioração irreversível ou intervenção cirúrgica desnecessária | lesão permanente; cirurgia ou biópsia no lado errado |
| S5 — catastrófica | morte | atraso crítico no tratamento de pneumotórax hipertensivo ou de hemorragia intracraniana |

### 4.2 Probabilidade

Não há dado de campo; a probabilidade é estimada **qualitativamente** até G2/G4 produzirem
frequências observadas.

| Nível | Categoria | Critério qualitativo |
|---|---|---|
| P5 | frequente | esperado repetidamente no uso rotineiro |
| P4 | provável | ocorre várias vezes ao longo da vida útil |
| P3 | ocasional | pode ocorrer algumas vezes no uso |
| P2 | remota | improvável, mas possível em cenários específicos |
| P1 | improvável | tão improvável que se assume não ocorrer com os controles ativos |

A probabilidade estimada é a de a situação perigosa ocorrer **e** resultar em dano sob leitura
médica concorrente. Para falhas de software, este
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

| Zona | Ação obrigatória |
|---|---|
| inaceitável | proibido liberar; exige controle por projeto antes de qualquer avanço de portão |
| ALARP | controles técnicos verificáveis e demonstração de que o risco residual global é superado pelo benefício (que ainda não existe: ver §5, risco-benefício) |
| aceitável | monitorar na vigilância ([REG-15](15-post-market-surveillance.md)); sem ação imediata |

**Reconciliação com a v0.1 de `main`.** A versão deste plano criada em `main` (commit `ec00a73`)
usava uma matriz mais permissiva: S5×P1 era ALARP (aqui é inaceitável) e S1 era sempre
aceitável. Esta versão canônica mantém a matriz acima, mais estrita, porque o arquivo de risco
ainda não tem frequência observada que sustente P1 para nenhum perigo S5. As estimativas de risco
residual que a v0.1 de `main` registrou constam em [`04-risk-file.md`](04-risk-file.md) §1a como
**propostas não aceitas**.

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

Revisão obrigatória do arquivo de risco, além das acima: a cada mudança de uso pretendido ou
adição de modalidade; antes de qualquer avaliação externa formal (G1 para cada condição, G2) e
antes de submissão regulatória (G5); diante de anomalia crítica em SOUP ou de evento adverso ou
incidente relatado ([REG-14](14-change-control.md) §4).

## 6. Ordem de preferência dos controles (ISO 14971 §7.1)

1. **Segurança inerente por projeto** — ex.: gate de escopo (`ModalityPlugin.validate()`) antes
   do modelo, nível de evidência calculado, falha fechada (`BackendUnavailableError`, sha256 dos
   pesos verificado na carga), proibição de afirmar normalidade.
2. **Medidas de proteção** no software ou no processo — ex.: abstenção em três bandas, auditoria
   encadeada por hash, des-identificação DICOM PS3.15, modo pesquisa separado, portões de CI
   (`check_honesty.py`, `trace.py`, `soup.py`, bandit, gitleaks).
3. **Informação de segurança** — IFU ([REG-12](12-labeling-ifu.md)), avisos na interface,
   contraindicações explícitas contra *rule-out*, leitura autônoma e uso fora de escopo.

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
| v0.2 | 2026-11-27 | Consolidação com a v0.1 criada em `main` (`ec00a73`): exemplos de severidade, critérios de probabilidade, zonas de ação, gatilhos de revisão, papel da verificação automatizada, ciclo de vida até a descontinuação; matriz estrita mantida e divergência registrada |
