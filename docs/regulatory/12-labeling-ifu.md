---
doc_id: REG-12
title: Rotulagem e Instruções de Uso (IFU) — rascunho pt-BR
status: RASCUNHO — v0.1 (uso em pesquisa)
date: 2026-11
author: Dr. Guilherme Capel Pasqua (CRM-SP 175873)
depends_on: [REG-01, REG-04]
---

# Instruções de Uso — RadiologyAI (versão de pesquisa)

> Rascunho. Para um dispositivo regularizado, o conteúdo mínimo da IFU de SaMD está na RDC
> 657/2022 Art. 7 (procedimentos de atualização, requisitos mínimos de hardware e software,
> princípio de funcionamento com descrição genérica dos algoritmos e suas associações clínicas
> válidas, alertas, especificações de interoperabilidade, informações de cibersegurança). Esta
> versão cobre o uso em pesquisa.

---

## AVISO

**USO EXCLUSIVO EM PESQUISA. NÃO É DISPOSITIVO MÉDICO.**
Não regularizado na ANVISA. Sem clearance FDA. Sem marcação CE.
**Não use para diagnóstico, triagem, rastreamento, exclusão de doença ou qualquer decisão sobre
o cuidado de um paciente.**
Toda saída é um **achado sugerido para confirmação médica**.

---

## 1. O que é

Plataforma de pesquisa que executa modelos abertos de análise de imagem sobre exames de pesquisa
e apresenta, para cada condição do catálogo, um achado sugerido com o seu **nível de evidência**,
a fonte dessa evidência e a incerteza. Uso pretendido completo:
[`01-intended-use.md`](01-intended-use.md).

## 2. Quem pode usar

Médicos e pesquisadores atuando em projeto de pesquisa. Exames de pacientes reais só podem ser
processados dentro de protocolo aprovado em CEP. Resolução CFM nº 2.454/2026: o médico permanece
responsável final; a IA é apenas ferramenta de apoio; é vedado delegar à IA a comunicação de
diagnóstico ao paciente.

## 3. Como ler um achado

Cada achado mostra:

| Elemento | Significado |
|---|---|
| Condição | o que o modelo procura, no vocabulário do catálogo |
| **Nível de evidência** | quão forte é a evidência de que este modelo, para esta condição, funciona (ver §4) |
| Fonte | o artefato de avaliação nosso (A/C) ou a publicação conferida (B) |
| Escore | saída do modelo; **não é probabilidade de doença** |
| Decisão | "sugerido", "não sugerido", "indeterminado" ou "somente escore" (ver §5) |
| Mapa / contorno | só aparece quando calculado pelo modelo real sobre esta imagem |

## 4. Os níveis de evidência

| Nível | O que significa | O que **não** significa |
|---|---|---|
| **A** | medimos este modelo, nesta condição, num conjunto externo rotulado que ele nunca viu, e ele passou nos critérios pré-registrados | não significa que funciona na sua população, no seu equipamento, nem que melhora o seu diagnóstico |
| **B** | um estudo revisado por pares ou uma diretriz sustenta esta saída; conferimos a citação na fonte | **não** reproduzimos o número; ele vale para a população do estudo |
| **C** | medimos em dado separado por paciente do mesmo conjunto de treino, ou em conjunto externo com possível vazamento | evidência fraca; pode cair em dado externo |
| **D** | exploratório: zero-shot ou sem validação para esta condição | **sem evidência**. Aparece só no **modo pesquisa** e nunca sozinho no laudo |

Uma condição pode ser **refutada**: se nossa medição externa limpa reprovou, ela cai para D mesmo
que exista artigo favorável.

## 5. Decisão, abstenção e "não avaliável"

- Um escore só vira "sugerido" ou "não sugerido" em relação a um ponto de operação com origem
  declarada. Sem ele, o sistema mostra **somente o escore**, sem veredito.
- Entre o ponto de exclusão e o de inclusão o achado é **indeterminado**. Isso é uma resposta
  legítima, não uma falha.
- **"Não sugerido" não é "normal".** O sistema nunca afirma que um exame é normal.
- **Abstenção:** o sistema se recusa a analisar, e diz o motivo, quando a entrada está fora do
  escopo (modalidade, incidência, idade, sequência), quando o tipo de exame não pode ser
  estabelecido, quando o modelo não está disponível (pesos ausentes ou com integridade
  reprovada) ou quando a medida exige pré-condição de aquisição que o exame não satisfaz (ex.:
  fase de contraste desconhecida para atenuação hepática). Abstenção nunca é substituída por um
  resultado aproximado.

## 6. Advertências

1. Escores **não** são probabilidade de doença; a calibração é imperfeita e varia por dataset.
2. Nenhum desempenho foi medido em população brasileira.
3. Viés de automação: saída errada da IA piora o desempenho de leitores, inclusive experientes.
   Leia o exame antes de olhar os achados sugeridos.
4. Medidas assistidas (contorno do ventrículo esquerdo, fração de ejeção, segmentação interativa)
   exigem que você confira e, se preciso, corrija o contorno antes de aceitar o número.
5. Modelos de segmentação de lesão conhecida (ex.: glioma) não são detectores: saída não vazia
   não significa lesão presente.
6. Alguns modelos têm pesos de licença **não comercial** ou só de pesquisa
   ([`MODEL_LICENSES.md`](../../MODEL_LICENSES.md)); seu uso fora de pesquisa é proibido pelos
   próprios termos.
7. Imagens com texto gravado no pixel (comum em ultrassom) podem conter dado pessoal.

## 7. Limitações por modalidade

Ver [`ROADMAP.md`](../../ROADMAP.md) §5 e o plano de avaliação clínica
([`10-clinical-evaluation-plan.md`](10-clinical-evaluation-plan.md)). Em resumo: RX de tórax só
frontal e adulto; biomarcadores de TC só com pré-condições satisfeitas; mamografia avaliada em
população diagnóstica, não de rastreamento; ecocardiografia somente em modo pesquisa.

## 8. Desempenho

Não há tabela de desempenho nesta IFU. Cada número existe apenas como registro de evidência do
catálogo, com dataset, contagens, intervalo de confiança e caminho do artefato
(`artifacts/eval/`), e é mostrado na própria interface junto ao achado.

## 9. Requisitos mínimos

A definir por liberação: sistema operacional, Python, CPU/GPU, memória e navegador suportados.

## 10. Contato e relato de problemas

Problemas, resultados suspeitos e incidentes de privacidade: abrir issue no repositório do
projeto ou contatar o responsável, Dr. Guilherme Capel Pasqua (CRM-SP 175873).

---

## Histórico de revisões

| Versão | Data | Alteração |
|---|---|---|
| v0.1 | 2026-11 | Criação, versão de pesquisa |
