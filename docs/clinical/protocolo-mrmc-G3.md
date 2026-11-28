---
doc_id: CLIN-G3
title: Protocolo de estudo de leitores multi-leitor multi-caso (MRMC) — portão G3
status: MODELO DE PROTOCOLO — não submetido a CEP; não iniciado
date: 2026-11-28
author: Dr. Guilherme Capel Pasqua (CRM-SP 175873)
depends_on: [REG-01, REG-10, REG-11, REG-12]
---

# Protocolo MRMC — portão G3 (modelo)

> **Modelo de protocolo — não submetido a CEP; não iniciado.** Nenhum leitor foi recrutado,
> nenhum caso foi selecionado, nenhum dado foi coletado. Os campos entre `⟨…⟩` são
> **parâmetros a fixar** antes da submissão; nenhum tamanho de efeito, variância ou tamanho
> amostral final é proposto aqui, porque qualquer número escrito agora seria inventado.
>
> **Software de pesquisa. Não é dispositivo médico.** Este estudo não é validação clínica de
> produto e não autoriza uso assistencial ([`01-intended-use.md`](../regulatory/01-intended-use.md)).

## 0. Pré-condições para sair do papel

| Pré-condição | Por quê | Estado em 2026-11-28 |
|---|---|---|
| Condição em **nível A** em `src/radiologyai/catalog/levels.json` no momento do congelamento do protocolo | só se testa o efeito no leitor de uma saída que já passou em validação externa de algoritmo isolado | 2 condições (abaixo) |
| Preferencialmente, evidência G2 (dado brasileiro, ≥ 2 centros) para a mesma condição e o mesmo modelo | o [`ROADMAP.md`](../../ROADMAP.md) §12.1 põe G3 **depois** de G2; os casos de G3 virão de parceiro brasileiro | G2 não iniciado |
| Modo de leitura do software implementado e verificado (§12) | sem ele não há cegamento nem registro confiável | **não existe** |
| Parceiro institucional, CEP, leitores e financiamento | ROADMAP §12.1 | não iniciado |

### 0.1 Condições elegíveis hoje (nível A)

Transcritas de `src/radiologyai/catalog/levels.json` (`protocol_id: 2026-11-v1`) e de
[`HONEST_STATUS.md`](../../HONEST_STATUS.md). São medições **retrospectivas de algoritmo
isolado em dados públicos**; não dizem nada sobre o efeito no leitor, que é justamente o que
G3 mede.

| Condição | Modelo(s) em A | Registro(s) que sustentam A | Ressalva que afeta o desenho de G3 |
|---|---|---|---|
| `cxr.cardiomegaly` | `xrv-densenet121-res224-pc`; `xrv-densenet121-res224-nih` | `evidence/measured/nih-cxr14/xrv-densenet121-res224-pc/cardiomegaly.json` (AUROC 0,813 [0,783; 0,839], rótulo NLP); `evidence/measured/openi/xrv-densenet121-res224-pc/cardiomegaly.json` (0,874 [0,840; 0,905]); `evidence/measured/openi/xrv-densenet121-res224-nih/cardiomegaly.json` (0,825 [0,786; 0,862]) | no NIH é um A **limítrofe** (0,796 na amostra completa; só um terço das 40 sementes de divisão passaria — `HONEST_STATUS.md`); os modelos `chex` e `mimic_*` foram reprovados para esta condição |
| `cxr.lung_opacity` | `xrv-densenet121-res224-chex`, `-mimic_ch`, `-mimic_nb` | `evidence/measured/nih-cxr14-google/xrv-densenet121-res224-chex/lung_opacity.json` (0,872 [0,848; 0,894], subconjunto adjudicado e enriquecido, ~58% positivos) | os **mesmos** modelos foram **reprovados** no Open-i (0,601–0,639 — `evidence/measured/openi/xrv-densenet121-res224-*/lung_opacity.json`); o A se mantém pela regra pré-registrada, mas a transferência entre conjuntos é duvidosa e limiares não se transferem para a prevalência clínica |

Consequência de desenho: o protocolo final escolhe **um** par condição–modelo por estudo
(`⟨condição⟩`, `⟨model_id⟩`, `⟨sha256 dos pesos⟩`), congelado antes do recrutamento, e o nível
A tem de continuar valendo na data do congelamento. Se o par cair de nível (por exemplo, por
medição G2 reprovada), o estudo não começa. Um resultado nulo ou negativo em G3 é plausível
com estas ressalvas e será publicado (§9).

## 1. Objetivo e hipótese

**Pergunta (ROADMAP §4, G3):** o médico assistido pela saída do modelo erra menos do que sem ela?

| Item | Conteúdo (modelo) |
|---|---|
| Objetivo primário | comparar o desempenho diagnóstico de leitores **com** e **sem** a saída do modelo para `⟨condição⟩` em RX de tórax frontal de adulto |
| Desfecho primário | diferença na AUROC média dos leitores (assistido − não assistido), com o leitor dando um escore de confiança `⟨escala, ex.: 0–100⟩` por caso; **ou**, se pré-especificado em vez disso, diferença na sensibilidade e na especificidade médias num limiar de decisão binária do leitor |
| Hipótese H0 / H1 | H0: ΔAUROC = 0 (não diferença); H1: ΔAUROC ≠ 0. Alternativa de não inferioridade (H0: Δ ≤ −`⟨margem⟩`) só se justificada e pré-especificada |
| Desfechos secundários | sensibilidade e especificidade dos leitores no limiar binário; tempo de leitura por caso; concordância entre leitores; desempenho por subgrupo (sexo, faixa etária, incidência PA/AP, equipamento — quando o metadado existe, como em [`protocolo-evidencia.md`](../protocolo-evidencia.md) §7); taxa de mudança de resposta induzida pela IA (certa → errada e errada → certa) |
| Fora do escopo | uso autônomo, triagem, priorização, exclusão de doença (*rule-out*), pediatria, incidência lateral ([`01-intended-use.md`](../regulatory/01-intended-use.md) §3) |

A escolha entre AUROC e sensibilidade/especificidade como desfecho primário é feita **antes**
do cálculo amostral, porque muda os parâmetros de entrada (§6). Testar vários desfechos exige
correção por multiplicidade; Obuchowski & Bullen (2022) tratam da correção para estudos com
múltiplos desfechos de acurácia (conferido no resumo; o método específico do texto integral
**não conferido** — §13).

## 2. Desenho

| Elemento | Especificação (modelo) |
|---|---|
| Tipo | estudo de leitores retrospectivo, multi-leitor multi-caso, **totalmente cruzado** (*fully crossed*): todo leitor lê todo caso nas duas condições (sem IA e com IA) |
| Modo de assistência | `⟨concorrente: a saída aparece junto com a imagem⟩` **ou** `⟨sequencial: o leitor registra a resposta sem IA, depois vê a IA e pode revisar⟩`. A escolha é pré-especificada; o desenho sequencial mede outra coisa e afeta o tamanho amostral (§6) |
| Sessões | ≥ 2 sessões por leitor; em cada sessão, metade dos casos sem IA e metade com IA (ou blocos), contrabalançado entre leitores |
| Período de *washout* | `⟨N semanas⟩` entre a leitura de um mesmo caso nas duas condições; o valor é fixado no protocolo e justificado (nenhuma fonte para um valor específico foi conferida por este documento) |
| Ordem de leitura | aleatorizada **por leitor e por sessão**, com semente registrada; o software gera e grava a sequência (§12) |
| Ambiente | estação de leitura `⟨monitor de diagnóstico / especificação⟩`, condições de luz `⟨…⟩`, sem acesso a laudo, prontuário ou exames anteriores (salvo se pré-especificado) |
| Informação clínica ao leitor | `⟨nenhuma / idade e sexo / indicação do exame⟩`, igual nas duas condições |
| Treinamento | sessão de familiarização com casos **fora** do conjunto de estudo, incluindo exemplos de erro da IA; a IFU ([`12-labeling-ifu.md`](../regulatory/12-labeling-ifu.md) §6.3) já adverte sobre viés de automação |

Desenhos alternativos (*split-plot*, leitores diferentes por bloco de casos) reduzem o tempo
de leitura mas exigem análise própria; adotar um deles exige revisar §6 e §7 com estatístico.

## 3. Casos: seleção e enriquecimento

| Item | Especificação (modelo) |
|---|---|
| Fonte | RX de tórax frontal de adulto de `⟨instituição(ões) parceira(s)⟩`, retrospectivo, sob CEP; **nunca** de conjunto usado no treino ou na seleção do modelo (`pc` treinou em PadChest; `chex` em CheXpert; `mimic_*` em MIMIC-CXR; `nih` em NIH ChestX-ray14 — conferir no manifesto do modelo) |
| Período e amostragem | exames consecutivos ou aleatórios de `⟨período⟩` em `⟨serviço⟩`, antes de qualquer enriquecimento; o fluxo de seleção é relatado (diagrama STARD) |
| Elegibilidade | adulto (`⟨≥ 18 anos⟩`), incidência frontal PA/AP declarada no cabeçalho DICOM, modalidade `CR`/`DX`; exclusões pré-especificadas (qualidade inadequada pelo critério `⟨…⟩`, exame já usado em G2 como teste sequestrado, etc.) |
| Enriquecimento | razão positivos:negativos `⟨p:n⟩` escolhida para eficiência estatística. Enriquecimento distorce prevalência: **VPP e VPN não serão relatados na prevalência da amostra**, só recalculados para uma prevalência de referência declarada (mesma regra proposta em `protocolo-evidencia.md` §11.3) |
| Espectro | casos positivos cobrindo gravidade `⟨leve/moderada/grave⟩`; negativos incluindo outros achados (não só normais), para não inflar a acurácia |
| Inferência da IA | rodada **antes** das sessões, com o modelo e os pesos congelados (sha256), e gravada com proveniência; casos em que o software se abstém (gate de escopo, OOD) são **mantidos** no estudo e o leitor assistido vê a abstenção como tal |
| Sequestro | os casos não são vistos por ninguém da equipe que ajusta o software depois do congelamento |

## 4. Padrão de referência

| Item | Especificação (modelo) |
|---|---|
| Painel | `⟨k⟩` radiologistas torácicos com `⟨≥ X anos⟩` de experiência, **que não são leitores do estudo** |
| Procedimento | leitura independente, cega à saída da IA e às leituras do estudo; discordâncias resolvidas por `⟨consenso / maioria / adjudicador sênior⟩` |
| Informação adicional | `⟨TC de tórax em até N dias / ecocardiograma (para cardiomegalia) / seguimento clínico⟩` quando disponível; definir antes, por condição |
| Definição operacional | critério escrito para `⟨condição⟩` (ex.: para cardiomegalia, se o painel usa ou não o índice cardiotorácico e em que incidência; para opacidade pulmonar, o que conta como opacidade) |
| Concordância do painel | relatar kappa ou equivalente entre membros antes da adjudicação |

Obuchowski & Bullen (2022) apontam como problema comum o padrão de referência imperfeito,
frequentemente correlacionado com o teste avaliado (conferido no resumo). Por isso o painel não
vê a IA e o rótulo minerado por NLP **não** é aceito como padrão de referência em G3.

## 5. Leitores

| Parâmetro | Valor |
|---|---|
| Número de leitores | `⟨r⟩` — vem do cálculo do §6, não de conveniência |
| Estratos de experiência | `⟨ex.: radiologistas gerais; residentes de radiologia; médicos não radiologistas⟩`, com número por estrato pré-especificado |
| Critérios | `⟨anos de prática, volume anual de RX de tórax, título de especialista⟩`; nenhum leitor participou do desenvolvimento do software nem do painel de referência |
| Instituições | `⟨1 ou mais⟩`; leitores de mais de um centro aumentam a generalização |
| Relação com a pesquisa | leitores são **participantes de pesquisa** (Lei 14.874/2024, Art. 2 XXXIV: a pesquisa envolve o ser humano de forma direta ou indireta); assinam TCLE próprio (§10) |

## 6. Tamanho amostral — método e entradas (sem números)

Referência de base exigida pelo ROADMAP: Obuchowski & Bullen, *Radiology* 2022 (revisão dos
fundamentos de desenho e análise MRMC, com referências às discussões detalhadas). O procedimento
operacional de poder e tamanho amostral para análise OR/DBM está em Hillis, Obuchowski &
Berbaum, *Acad Radiol* 2011, e no programa descrito por Hillis & Schartz, *J Med Imaging* 2018.

**Método.** Cálculo de poder para o teste OR (Obuchowski–Rockette) com leitores e casos
aleatórios, desenho totalmente cruzado, dois tratamentos (sem IA, com IA). Para esse desenho,
Hillis & Schartz (2018) dão o parâmetro de não centralidade

  λ = r·d² / [2(σ²_TR + σ²_ε − Cov₁ + (r − 1)(Cov₂ − Cov₃))]

com teste por distribuição F não central e graus de liberdade do denominador de Hillis (2007).
O mesmo programa aceita sensibilidade e especificidade como medida de desempenho, além de AUROC.

**Entradas obrigatórias** (todas a preencher; nenhuma é sugerida aqui):

| Entrada | Símbolo | De onde vem |
|---|---|---|
| Tamanho de efeito mínimo de interesse clínico | d = `⟨ΔAUROC ou Δsens/Δesp⟩` | decisão clínica justificada no protocolo; **não** derivada da medição de algoritmo isolado |
| Variância leitor×tratamento | σ²_TR = `⟨…⟩` | estudo piloto ou estudo prévio semelhante |
| Variância de erro | σ²_ε = `⟨…⟩` | idem; para AUROC pode ser conjecturada pela equação de Obuchowski (1994) citada por Hillis & Schartz |
| Covariâncias (ou correlações r₁, r₂, r₃) | Cov₁, Cov₂, Cov₃ = `⟨…⟩` | idem |
| Número de casos do piloto | c* = `⟨…⟩` | piloto |
| Razão normais:anormais do piloto e do estudo planejado | `⟨…⟩` | Hillis et al. (2011) incorporam a diferença de razão entre piloto e estudo |
| Nível de significância | α = `⟨…⟩` | protocolo |
| Poder desejado | 1 − β = `⟨…⟩` | protocolo |
| Situação de inferência | leitores e casos aleatórios | este desenho |
| Tipo de hipótese | `⟨diferença / não inferioridade com margem⟩` | §1 |
| Desenho de leitura | `⟨concorrente / sequencial⟩`; totalmente cruzado | §2 |

**Saída:** combinações (r, c) que atingem o poder; o protocolo escolhe uma e registra a razão
(tempo de leitor, disponibilidade de casos). Hillis et al. (2011) e Hillis & Schartz (2018)
recomendam que os parâmetros venham de piloto ou de estudo prévio comparável em modalidade,
experiência dos leitores e seleção de casos. **Recomendação deste modelo: um piloto MRMC pequeno
com os mesmos leitores-tipo, casos-tipo e interface, analisado por OR/DBM, antes de fixar N.**
O piloto não entra na análise confirmatória.

## 7. Análise estatística

| Item | Especificação |
|---|---|
| Método primário | Obuchowski–Rockette (OR) com leitores e casos aleatórios (Obuchowski & Rockette 1995), com graus de liberdade de Hillis (2007); o DBM (Dorfman, Berbaum & Metz 1992), na versão modificada recomendada por Hillis, Berbaum & Metz (2008), é equivalente ao OR quando ambos usam o mesmo estimador de AUROC (Hillis & Schartz 2018) e pode ser relatado como sensibilidade da análise |
| Estimador de AUROC por leitor | `⟨empírico (trapezoidal) / paramétrico⟩`, pré-especificado |
| Unidade de análise | o **paciente** (um exame por paciente); se houver mais de uma imagem por paciente, só a primeira elegível |
| Software | `⟨OR-DBM MRMC (Univ. Iowa) / RJafroc / iMRMC⟩`, versão registrada; o script de análise é commitado **antes** da abertura dos dados |
| Secundários | sensibilidade e especificidade médias no limiar binário do leitor, pelo mesmo arcabouço OR quando aplicável; tempo de leitura (modelo misto); subgrupos descritivos, sem teste formal se abaixo dos mínimos de `protocolo-evidencia.md` §2 |
| Multiplicidade | hierarquia pré-especificada: primário primeiro; secundários só se o primário for significativo, ou com correção declarada |
| Dados faltantes | caso não lido por um leitor ⇒ relatado; análise principal com casos completos, `⟨regra⟩` pré-especificada |
| Análises proibidas | escolher limiar, subconjunto de leitores ou de casos depois de ver resultados; trocar o desfecho primário |

## 8. Controles de viés

| Viés | Controle |
|---|---|
| Memória do caso entre condições | *washout* (§2), ordem aleatorizada, casos distratores `⟨opcional⟩` |
| Ordem/aprendizado | contrabalanceamento de sessões e de condição entre leitores |
| Viés de automação | treinamento com exemplos de erro da IA; registro da taxa de mudança errada induzida |
| Revisão do padrão de referência | painel cego à IA e às leituras (§4) |
| Espectro e enriquecimento | §3; prevalência nunca extrapolada |
| Vazamento | casos fora de treino e seleção do modelo (§3), conferido no manifesto e no datasheet |
| Viés do patrocinador/desenvolvedor | análise por estatístico sem papel no desenvolvimento `⟨nome/função⟩`; protocolo e script commitados antes; resultado publicado qualquer que seja |
| Viés de localização | se o desfecho exigir localizar o achado, pré-especificar a regra (Obuchowski & Bullen 2022 discutem ajuste para viés de localização — conferido no resumo) |
| Interface | a saída mostrada na condição "com IA" é a mesma que o laudo mostraria: nível, fonte, escore, decisão com ponto de operação de origem declarada (REQ-092, REQ-192) ou "somente escore"; nunca nível D (REQ-091) |

## 9. Relato

- **STARD-AI** (Sounderajah et al., *Nat Med* 2025) — diretriz para estudos de acurácia
  diagnóstica centrados em IA; 18 itens novos ou modificados sobre a lista STARD 2015
  (conferido no resumo, PubMed).
- **CLAIM 2024** (Tejani et al., *Radiol Artif Intell* 2024).
- Fluxograma de casos e de leitores; características dos leitores; todos os resultados,
  **inclusive nulos e negativos**, publicados, como em `protocolo-evidencia.md` §9.
- Registro prévio do estudo em `⟨registro público de estudos, a escolher⟩` antes da primeira
  leitura.

## 10. Ética

| Item | Conteúdo (modelo) |
|---|---|
| Marco legal | Lei 14.874/2024 (pesquisa com seres humanos), em vigor 90 dias após publicação no DOU de 29/05/2024 (Art. 65); Decreto 12.651/2025, que a regulamenta e mantém válidas as normas do CNS até que a Instância Nacional de Ética em Pesquisa publique normas próprias, se não conflitarem com a lei e o decreto (Art. 39) |
| Apreciação | protocolo submetido ao CEP da instituição proponente e das coparticipantes; em instituição do SUS, CEP vinculado à instituição onde estão os participantes e Termo de Anuência Institucional (Res. CNS 580/2018, Art. 10 e §1); uso de acervo da instituição só com aprovação ética (Art. 3) e com os procedimentos de sigilo descritos (Art. 10 §2) |
| Leitores | TCLE (Lei 14.874/2024, Art. 18); retirada a qualquer tempo sem justificativa (Art. 18 §7); dados de desempenho individual não divulgados de forma identificável |
| Pacientes cujas imagens são usadas | pedido de **dispensa de TCLE** justificado ao sistema CEP (Res. CNS 466/2012, item IV.8: quando a obtenção for inviável ou significar riscos substanciais à privacidade e confidencialidade), **a confirmar com o CEP**: a Lei 14.874/2024 não tem dispositivo expresso de dispensa (Art. 18 exige autorização expressa do participante; a única exceção encontrada é a de emergência, Art. 18 §6), e a vigência do item IV.8 decorre do Art. 39 do Decreto 12.651/2025 enquanto não houver conflito — ponto para leitura jurídica |
| Prazo de análise | 30 dias úteis a partir do aceite da documentação completa (Lei 14.874/2024, Art. 14) |
| Guarda | dados da pesquisa guardados por 5 anos após o término (Res. CNS 466/2012, XI.2.f), ou o prazo maior que o CEP ou o controlador exigirem |

## 11. Dados (des-identificação e LGPD)

- Des-identificação **pelo controlador** (instituição parceira), na origem, segundo DICOM PS3.15
  Annex E, com as opções de [`11-cybersecurity-lgpd.md`](../regulatory/11-cybersecurity-lgpd.md)
  §3 (Retain Patient Characteristics para subgrupos; Clean Pixel Data quando houver texto no pixel).
- Base legal: LGPD (Lei 13.709/2018) Art. 11 II c — estudos por órgão de pesquisa, garantida,
  sempre que possível, a anonimização; "órgão de pesquisa" é definido no Art. 5 XVIII, por isso o
  tratamento ocorre sob a instituição parceira (REG-11 §4).
- Dado pseudonimizado continua dado pessoal (LGPD Art. 13 §4; Art. 12 para anonimizado); a chave
  fica com o controlador; o compartilhamento é só de dado anônimo ou codificado, com a chave sob
  o gestor (Lei 14.874/2024, Art. 51 §1); a LGPD se aplica (Art. 61).
- Nenhuma imagem sai para nuvem ou API de terceiro; inferência local (REG-11 §4).
- Registro das operações de tratamento (LGPD Art. 37) e relatório de impacto quando exigido
  (Art. 38); medidas de segurança (Art. 46).

## 12. O que o software precisa fornecer (requisitos a especificar)

Nada disto existe hoje como "modo de leitura de estudo". Cada item vira um `REQ-NNN` com teste
verificador antes do piloto MRMC.

| Necessidade | Descrição | Base existente |
|---|---|---|
| Modo de leitura MRMC | sessão por leitor com lista de casos em ordem aleatorizada (semente gravada); condição "sem IA" oculta toda saída de IA (escore, nível, mapa, máscara); condição "com IA" mostra só o par condição–modelo do estudo | RBAC por rota (REQ-151); ocultação de saída do modo silencioso (REQ-195, implementado no software) é um precedente técnico, mas não é um modo de leitura |
| Captura da resposta | escore de confiança na escala pré-especificada e decisão binária; no desenho sequencial, resposta pré-IA travada antes de exibir a IA | — |
| Bloqueio | resposta gravada não pode ser editada após avançar; nenhum acesso ao padrão de referência nem a laudo | — |
| Cronometragem | tempo por caso, do carregamento à submissão | — |
| Registro auditável | cada exibição e cada resposta na trilha encadeada por hash (REQ-070, REQ-155), com leitor, caso, condição, sequência, modelo, sha256 dos pesos e git sha | trilha existente |
| Exportação | arquivo sem PHI no formato de entrada do software de análise (leitor × caso × condição × escore), com sha256 registrado | o CSV do modo silencioso (REQ-195) é precedente |
| Congelamento | modelo, pesos, pontos de operação e versão do software fixos durante todo o estudo; mudança = desvio registrado | sha256 de pesos (REQ-020, REQ-030) |

## 13. Referências e estado de conferência

| Referência | Como foi conferida |
|---|---|
| Obuchowski NA, Bullen J. Multireader diagnostic accuracy imaging studies: fundamentals of design and analysis. *Radiology* 2022;303(1):26-34. DOI 10.1148/radiol.211593 | metadados e resumo no PubMed (PMID 35166584). **Texto integral não conferido** (editor devolveu 403): as afirmações deste documento atribuídas ao artigo limitam-se ao resumo |
| Hillis SL, Obuchowski NA, Berbaum KS. Power estimation for multireader ROC methods: an updated and unified approach. *Acad Radiol* 2011;18(2):129-42. DOI 10.1016/j.acra.2010.09.007 | PubMed (PMID 21232681), resumo |
| Hillis SL, Schartz KM. Multireader sample size program for diagnostic studies: demonstration and methodology. *J Med Imaging* 2018;5(4):045503. DOI 10.1117/1.JMI.5.4.045503 | texto integral no PMC (PMC6265604): fórmula de λ, entradas, suporte a sensibilidade/especificidade |
| Obuchowski NA, Rockette HE. Hypothesis testing of diagnostic accuracy for multiple readers and multiple tests: an ANOVA approach with dependent observations. *Commun Stat Simul Comput* 1995;24(2):285-308. DOI 10.1080/03610919508813243 | metadados no Crossref; texto **não conferido** |
| Dorfman DD, Berbaum KS, Metz CE. Receiver operating characteristic rating analysis: generalization to the population of readers and patients with the jackknife method. *Invest Radiol* 1992;27(9):723-31. DOI 10.1097/00004424-199209000-00015 | PubMed (PMID 1399456) e Crossref; texto **não conferido** |
| Hillis SL, Berbaum KS, Metz CE. Recent developments in the Dorfman-Berbaum-Metz procedure for multireader ROC study analysis. *Acad Radiol* 2008;15(5):647-61. DOI 10.1016/j.acra.2007.12.015 | PubMed (PMID 18423323), resumo |
| Hillis SL. A comparison of denominator degrees of freedom methods for multiple observer ROC analysis. *Stat Med* 2007;26(3):596-619. DOI 10.1002/sim.2532 | PubMed (PMID 16538699), resumo |
| Obuchowski NA. Computing sample size for receiver operating characteristic studies. *Invest Radiol* 1994;29(2):238-43. DOI 10.1097/00004424-199402000-00020 | só pela lista de referências de Hillis & Schartz 2018; **não conferido** na fonte |
| Gallas BD et al. Evaluating imaging and computer-aided detection and diagnosis devices at the FDA. *Acad Radiol* 2012;19(4):463-77. DOI 10.1016/j.acra.2011.12.016 | PubMed (PMID 22306064), resumo (citado em REG-10 §4) |
| Sounderajah V et al. The STARD-AI reporting guideline for diagnostic accuracy studies using artificial intelligence. *Nat Med* 2025;31(10):3283-3289. DOI 10.1038/s41591-025-03953-8 | PubMed (PMID 40954311), resumo |
| Tejani AS et al. Checklist for Artificial Intelligence in Medical Imaging (CLAIM): 2024 update. *Radiol Artif Intell* 2024;6(4):e240300. DOI 10.1148/ryai.240300 | PubMed (PMID 38809149) |
| Lei 14.874, de 28/05/2024 (DOU 29/05/2024) — Arts. 2 XXXIV, 14, 18, 18 §6, 18 §7, 51 §1, 61, 65 | texto em planalto.gov.br/ccivil_03/_ato2023-2026/2024/lei/l14874.htm |
| Decreto 12.651, de 07/10/2025 (DOU 08/10/2025) — Arts. 39, 40 | texto em planalto.gov.br/ccivil_03/_ato2023-2026/2025/decreto/d12651.htm |
| Resolução CNS 466/2012 — itens IV.8 e XI.2.f | PDF oficial em gov.br/conselho-nacional-de-saude (atos-normativos/resolucoes/2012/resolucao-no-466.pdf) |
| Resolução CNS 580/2018 — Arts. 3, 10, 10 §1, 10 §2 | PDF oficial em gov.br/conselho-nacional-de-saude (atos-normativos/resolucoes/2018/resolucao-no-580.pdf) |
| LGPD, Lei 13.709/2018 — Arts. 5 XVIII, 11 II c, 12, 13 §4, 37, 38, 46 | texto em planalto.gov.br/ccivil_03/_ato2015-2018/2018/lei/l13709.htm |
| Duração do *washout*; uso de casos distratores | **não conferido** — sem fonte; parâmetro do protocolo |

---

## Histórico de revisões

| Versão | Data | Alteração |
|---|---|---|
| v0.1 | 2026-11-28 | Criação do modelo; condições elegíveis transcritas de `levels.json`; nenhum parâmetro numérico de efeito ou N |
