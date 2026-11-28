---
doc_id: CLIN-G4
title: Protocolo de piloto prospectivo silencioso — portão G4
status: MODELO DE PROTOCOLO — não submetido a CEP; não iniciado
date: 2026-11-28
author: Dr. Guilherme Capel Pasqua (CRM-SP 175873)
depends_on: [REG-01, REG-10, REG-11, REG-12, CLIN-G3]
---

# Piloto prospectivo silencioso — portão G4 (modelo)

> **Modelo de protocolo — não submetido a CEP; não iniciado.** Nenhuma instituição, exame ou
> participante está envolvido. Campos entre `⟨…⟩` são **parâmetros a fixar** antes da
> submissão. Nenhum limiar de desempenho, de deriva ou de parada é proposto aqui.
>
> **Software de pesquisa. Não é dispositivo médico.** No piloto silencioso as saídas ficam
> **ocultas** do médico assistente e nada volta ao cuidado
> ([`01-intended-use.md`](../regulatory/01-intended-use.md) §3.1). Se alguma saída chegar ao
> cuidado, o estudo deixa de ser silencioso e o software cruza a fronteira de reclassificação
> (REG-01 §6).

## 0. Pré-condições

| Pré-condição | Estado em 2026-11-28 |
|---|---|
| Condição(ões) em nível A em `src/radiologyai/catalog/levels.json` na data do congelamento | `cxr.cardiomegaly` e `cxr.lung_opacity`, com as ressalvas de [`protocolo-mrmc-G3.md`](protocolo-mrmc-G3.md) §0.1 |
| G3 concluído para a mesma condição e o mesmo modelo (ROADMAP §12.1: G4 depois de G3) | não iniciado |
| Modo silencioso no software — **REQ-195 (implementado no software; integração no hospital pendente)** | implementado no software; não validado em campo (§5) |
| Governança LGPD antes de operar com dado real ([`11-cybersecurity-lgpd.md`](../regulatory/11-cybersecurity-lgpd.md) §4a): controlador, operador e encarregado definidos; base legal documentada; TLS e criptografia em repouso; RBAC revisado; teste de intrusão | não iniciado |
| Instituição parceira, CEP aprovado, Termo de Anuência Institucional | não iniciado |

## 1. Enquadramento: DECIDE-AI num estudo silencioso

O ROADMAP (§4, G4) pede relato DECIDE-AI (Vasey et al., *Nat Med* 2022). DECIDE-AI foi escrito
para a avaliação clínica **inicial, em pequena escala e ao vivo**; o próprio glossário da diretriz
define avaliação ao vivo como aquela em que as decisões têm impacto direto no cuidado, "em
oposição à avaliação *offline* ou em *shadow mode*, em que as decisões não têm impacto direto no
cuidado" (versão BMJ, conferida no PMC). **Um piloto silencioso é, portanto, *shadow mode* no
vocabulário da própria diretriz.** Este protocolo usa a lista DECIDE-AI como arcabouço de relato
**adaptado** e declara, item a item, o que não se aplica (§10). Isso tem de constar no relatório
final.

## 2. Objetivos

| Tipo | Objetivo |
|---|---|
| Primário | estimar a concordância entre a saída do modelo para `⟨condição⟩` e o laudo do radiologista, em exames consecutivos do fluxo real, sem influenciar o cuidado |
| Secundários | taxa de falha técnica; taxa e motivos de abstenção; tempo de processamento; deriva de entrada e de escore ao longo do tempo; desempenho por subgrupo (sexo, faixa etária, incidência, equipamento, quando há metadado); concordância contra um padrão de referência adjudicado num subconjunto `⟨opcional⟩` |
| Não é objetivo | medir efeito no médico (isso é G3), utilidade clínica, desfecho do paciente, nem acurácia diagnóstica definitiva (o laudo de rotina não é padrão de referência) |

## 3. Local, população e elegibilidade

| Item | Especificação (modelo) |
|---|---|
| Local | serviço de radiologia de `⟨instituição⟩`, `⟨público SUS / privado⟩`, `⟨setores: PS, internação, ambulatório⟩` |
| Período | `⟨data de início⟩` a `⟨data de fim⟩` ou até `⟨N exames⟩`; N justificado no protocolo (precisão desejada da estimativa de concordância), não proposto aqui |
| Inclusão | RX de tórax frontal (PA/AP) de adulto (`⟨≥ 18 anos⟩`), `CR`/`DX`, realizado no período, com laudo assinado |
| Exclusão | incidência só lateral; exames de pesquisa de outros protocolos `⟨se houver conflito⟩`; pacientes que tenham manifestado oposição ao uso de dados em pesquisa, se a instituição mantiver esse registro |
| Amostragem | **consecutiva**, sem enriquecimento: a prevalência é a do serviço, e é isso que o piloto mede |
| Exames rejeitados pelo software | permanecem no denominador; o motivo de rejeição (gate de escopo, OOD, lateralidade, integridade) é desfecho, não exclusão |

## 4. Fluxo

1. O exame é adquirido e laudado **como sempre**; nada muda para o técnico, o radiologista ou
   o médico assistente (Res. CNS 580/2018, Arts. 5 e 6: a pesquisa não interfere na rotina nem
   na atividade dos trabalhadores, salvo justificativa e autorização do dirigente).
2. Uma cópia do exame chega à instância do software **dentro da rede da instituição**
   (`⟨roteamento DICOM / importação DICOMweb⟩`), é des-identificada na ingestão e analisada com
   modelo e pesos congelados (sha256).
3. Com `MEDAI_SHADOW_MODE` ligado, a análise roda, é gravada e auditada, mas a saída de IA só é
   servida a quem tem a permissão `shadow:view` (papel `researcher`, equipe do piloto) e nada é
   enviado ao PACS (REQ-195, implementado no software — §5).
4. Depois de `⟨intervalo mínimo após a assinatura do laudo, ex.: N dias⟩`, a equipe do piloto
   exporta o CSV pseudonimizado (dado pessoal pela LGPD, art. 13 §4; sem identificador direto, texto livre ou pixel) e o controlador liga cada linha ao laudo pelo pseudônimo
   (§8).
5. O laudo é convertido em rótulo por condição por `⟨dois anotadores independentes, cegos à
   saída da IA, com adjudicação⟩`. Extração por NLP só como análise secundária e declarada,
   porque rótulo minerado de laudo é ruidoso (`protocolo-evidencia.md` §9).
6. Análise conforme o plano congelado (§6).

## 5. Base técnica: REQ-195 (implementado no software; integração no hospital pendente)

Descrição conforme o código na árvore em 2026-11-28; **sujeita a mudança até o REQ-195 ser
verificado**. O protocolo final deve citar a versão (git sha) verificada.

| Função | Comportamento esperado |
|---|---|
| Configuração | `MEDAI_SHADOW_MODE` (padrão desligado) em `src/radiologyai/persistence/config.py` |
| Ocultação | sem `shadow:view`, rotas de estudos, análises e artefatos não servem saída de IA (mapa e máscara ⇒ 403) — `src/radiologyai/api/routers/` |
| Papel | `researcher`: lê a saída de IA para comparação posterior, **sem ato médico** (não revisa, não cria nem assina laudo) — `src/radiologyai/persistence/security.py` |
| Exportação | `src/radiologyai/api/routers/shadow.py`: CSV com uma linha por achado; estudo identificado só por id opaco e pelo Study Instance UID já pseudonimizado por HMAC; sem nome, identificador, data de nascimento, *accession*, texto livre ou pixel; cada exportação auditada com o sha256 do arquivo |
| Trilha | auditoria encadeada por hash (REQ-070, REQ-155) |

Verificações exigidas **antes** do primeiro exame real (cada uma com evidência de teste):
o médico com papel `physician` não vê nenhuma saída de IA em nenhuma rota nem no PACS; nenhuma
exportação DICOM/FHIR/SR sai com o modo ligado; o CSV não contém identificador direto, texto livre nem pixel (continua dado pessoal pseudonimizado; inspeção por amostragem e
teste automatizado); a cadeia de auditoria verifica de ponta a ponta.

## 6. Desfechos e análise

| Desfecho | Definição (modelo) | Análise |
|---|---|---|
| Concordância com o laudo (primário) | por exame, decisão do modelo (`decision` no CSV, com ponto de operação de origem declarada, REQ-192) × rótulo do laudo | proporção de concordância positiva e negativa, kappa de Cohen, sensibilidade e especificidade **relativas ao laudo**, IC95% `⟨método, ex.: bootstrap por paciente como em protocolo-evidencia.md §4⟩` |
| Escore × laudo | `raw_score` × rótulo | AUROC relativa ao laudo, com a mesma ressalva |
| Exames "somente escore" ou "indeterminado" | `decision` sem veredito | relatados em separado; nunca recodificados como negativos |
| Falha técnica | análise que não termina, erro, integridade reprovada | taxa por 100 exames, por causa |
| Abstenção | `abstention_reason` | taxa por motivo; abstenção não é erro do protocolo, mas é desfecho de segurança |
| Tempo | `requested_at` → `finished_at` | mediana e percentis |
| Deriva de entrada | distribuição de metadados (equipamento, incidência, idade, sexo) por janela de `⟨semana/mês⟩` | estatística pré-especificada `⟨…⟩` contra a janela inicial |
| Deriva de saída | distribuição de `raw_score` e taxa de positivos por janela | idem; alarme em `⟨limiar a fixar⟩` |
| Subgrupos | sexo, faixa etária, incidência, equipamento | descritivo; abaixo dos mínimos de `protocolo-evidencia.md` §2, só contagens |

**Interpretação obrigatória:** o laudo de rotina é comparador, não verdade. Discordância pode
ser erro do modelo, do laudo, ou de ambos. Se houver subconjunto adjudicado, o protocolo
pré-especifica como é sorteado (aleatório, não só discordantes), para não enviesar a estimativa.

Condições fora do nível A que o software também calcula entram no CSV (todas as visibilidades);
são **exploratórias**, relatadas como tal e nunca promovem nível.

## 7. Segurança e regras de parada

| Evento | Classificação | Ação |
|---|---|---|
| Qualquer saída de IA visível para quem cuida do paciente (tela, PACS, laudo, relatório, conversa da equipe) | **desvio grave** | suspensão imediata; notificação ao CEP; análise de causa; reinício só com aprovação |
| Incidente com dado pessoal (PHI no CSV, em log, fora da rede) | incidente de segurança | suspensão; procedimento do encarregado do controlador (REG-11 §4); notificação ao CEP |
| Integridade de pesos ou da cadeia de auditoria reprovada | falha de integridade | suspensão até correção e re-verificação |
| Taxa de falha técnica acima de `⟨limiar a fixar⟩` em `⟨janela⟩` | falha operacional | pausa e revisão |
| Deriva acima do alarme pré-especificado | sinal | revisão pela equipe; não muda o modelo durante o estudo (mudança = desvio, DECIDE-AI item 11) |
| Achado da IA discordante do laudo, com possível dano | **dilema ético a decidir com o CEP antes do início** | ver abaixo |

**Achados discordantes.** Como as saídas são revistas só depois do laudo, a equipe pode ver uma
saída positiva onde o laudo é negativo. O protocolo precisa fixar, com o CEP, uma das políticas:
`⟨(a) nenhuma comunicação — o estudo é observacional e o modelo não tem validação para isso; ou
(b) revisão por radiologista independente do estudo e comunicação pelo fluxo assistencial
normal apenas quando esse radiologista, por leitura própria da imagem, identificar achado
relevante⟩`. Em nenhuma hipótese a saída da IA é comunicada como tal ao paciente ou à equipe
assistente (REG-01 §3.1; IFU §2: Resolução CFM 2.454/2026 citada em REG-12, **não conferida**
por este documento). A política escolhida e sua frequência de acionamento entram no relatório
(DECIDE-AI 6b, 13b, 16).

## 8. Dados e LGPD (Lei 13.709/2018)

| Item | Conteúdo (modelo) |
|---|---|
| Controlador | `⟨instituição parceira⟩` (REG-11 §4); o projeto não trata dado identificado |
| Base legal | Art. 11 II c (estudos por órgão de pesquisa, garantida, sempre que possível, a anonimização); órgão de pesquisa conforme Art. 5 XVIII (órgão ou entidade da administração pública, ou pessoa jurídica de direito privado **sem fins lucrativos** com pesquisa na missão — um hospital privado com fins lucrativos não se enquadra por si só, e o tratamento teria de ocorrer sob uma instituição que se enquadre ou sob outra base legal); Art. 13 (estudos em saúde pública: tratamento dentro do órgão, dado anonimizado ou pseudonimizado sempre que possível, sem revelar dado pessoal na divulgação, §1; segurança sob responsabilidade do órgão de pesquisa, §2) — enquadramento **a confirmar** pelo encarregado e pela assessoria jurídica do controlador |
| Pseudonimização | HMAC na ingestão; a chave (`MEDAI_DEIDENT_KEY`) fica com o controlador; dado pseudonimizado continua pessoal (Art. 13 §4); só dado anônimo ou codificado é compartilhado, com a chave sob o gestor (Lei 14.874/2024, Art. 51 §1; LGPD subsidiária, Art. 61) |
| Minimização | o CSV de REQ-195 não tem identificador; ligação com o laudo feita pelo controlador, dentro da instituição |
| Local | instância dentro da rede da instituição; nenhum envio a nuvem ou API de terceiro |
| Registro e impacto | registro das operações (Art. 37); relatório de impacto (Art. 38) — rascunho em REG-11 §4 |
| Segurança | medidas técnicas e administrativas (Art. 46); REG-11 §4a |
| Retenção e descarte | prazo do protocolo; dados da pesquisa guardados 5 anos após o término (Res. CNS 466/2012, XI.2.f), ou prazo maior exigido; descarte seguro registrado |

## 9. Ética (Brasil)

| Item | Conteúdo (modelo) |
|---|---|
| Lei 14.874/2024 | pesquisa com seres humanos inclui o manejo de seus dados (Art. 2 XXXIV); anonimato, privacidade e confidencialidade (Art. 19); análise pelo CEP em até 30 dias úteis da documentação completa (Art. 14); pesquisa de interesse estratégico para o SUS tem prioridade (Art. 15), se assim enquadrada |
| Decreto 12.651/2025 | regulamenta a lei e institui a Instância Nacional de Ética em Pesquisa (Art. 10); normas do CNS permanecem válidas até normas próprias da Instância, se não conflitarem (Art. 39); CEPs já credenciados continuam (Art. 37); dados só para a finalidade do projeto (Art. 34) |
| Res. CNS 466/2012, IV.8 | dispensa de TCLE dos pacientes "nos casos em que seja inviável a obtenção" ou em que a obtenção "signifique riscos substanciais à privacidade e confidencialidade dos dados do participante", justificadamente solicitada ao sistema CEP. **Ponto em aberto:** a Lei 14.874/2024 não tem dispositivo expresso de dispensa (Art. 18 exige autorização expressa; exceção só para emergência, Art. 18 §6); a aplicação do IV.8 depende do Art. 39 do decreto e de não haver conflito — confirmar com o CEP e com assessoria jurídica |
| Res. CNS 580/2018 (instituição do SUS) | pesquisa não se confunde com atenção à saúde (Art. 2); uso de acervo só com aprovação ética (Art. 3); não interferir na rotina nem no trabalho dos profissionais sem autorização do dirigente (Arts. 5, 6); trabalhadores como participantes respeitam os preceitos da instituição (Art. 7); divulgar resultados à instituição (Art. 9); CEP da instituição e TAI (Art. 10, §1); sigilo do acervo descrito no protocolo (Art. 10 §2) |
| Radiologistas cujos laudos são comparados | definir com o CEP se são participantes (o desfecho compara o trabalho deles). Se forem, TCLE ou justificativa, e resultados nunca divulgados por profissional identificável |
| Registro do estudo | `⟨registro público, a escolher⟩` antes do primeiro exame |
| Normas citadas em outros documentos do repositório e **não conferidas** aqui | Res. CNS 738/2024 (bancos de dados, citada em REG-10 §5 e REG-11 §4); Res. CNS 510/2016 |

## 10. Itens DECIDE-AI (adaptado ao modo silencioso)

Lista conferida na versão BMJ 2022 (PMC9116198): 17 itens específicos de IA (28 subitens) e 10
genéricos (I–X). Resumos abaixo em redação própria.

| Item | Tema | Neste piloto |
|---|---|---|
| 1 | Título identifica avaliação clínica inicial de sistema de apoio por IA | incluir "silencioso / *shadow mode*" no título |
| 2a, 2b | Uso pretendido: condição, população, usuários, lugar no fluxo | uso **de pesquisa** (REG-01); declarar que não há usuário clínico da saída |
| 3a | Pacientes: recrutamento, elegibilidade, tamanho amostral | §3 |
| 3b, 3c | Usuários: recrutamento, familiarização | **adaptado**: os "usuários" são a equipe do piloto (papel `researcher`); médicos assistentes não usam o sistema |
| 4a–4c | Sistema: versão, tipo, treino, desempenho prévio; entradas; saídas e como foram vistas | modelo, pesos sha256, registros de `evidence/measured/` que sustentam o nível; entradas e rejeições; saída oculta (§5) |
| 5a, 5b | Local e fluxo; quem toma a decisão final | §3, §4; decisão final sempre do radiologista, sem acesso à IA |
| 6a, 6b | Definição de erro/mau funcionamento; riscos ao paciente | §6, §7 |
| 7 | Fatores humanos | **em grande parte não aplicável** (sem interação clínica); relatar a interação da equipe do piloto e qualquer vazamento |
| 8 | Ética (ex.: equidade) | subgrupos §6 |
| 9a, 9b | Características basais de pacientes e usuários; dados de entrada faltantes | relatar |
| 10a, 10b | Exposição dos usuários; mudanças de fluxo | relatar zero exposição clínica pretendida e qualquer desvio |
| 11 | Modificações do sistema durante o estudo | proibidas; qualquer uma é desvio relatado |
| 12 | Concordância humano–IA e motivos de divergência | **adaptado**: concordância modelo × laudo (§6), sem a dimensão "o usuário mudou de ideia" |
| 13a, 13b | Erros e danos observados | §6, §7; incluir acionamentos da política de achados discordantes |
| 14a, 14b | Usabilidade; curvas de aprendizado | **não aplicável** a médicos; usabilidade da exportação pela equipe, se relevante |
| 15, 16 | Discussão: uso pretendido; segurança | relatar que nada sobre efeito no cuidado pode ser concluído |
| 17 | Disponibilidade de dados e código | código no repositório; dados conforme o controlador e o CEP |
| I–X | genéricos (resumo, objetivos, governança, desfechos, análise, envolvimento de pacientes, resultados, subgrupos, limitações, conflitos) | relatar todos |

## 11. Referências e estado de conferência

| Referência | Como foi conferida |
|---|---|
| Vasey B et al. Reporting guideline for the early-stage clinical evaluation of decision support systems driven by artificial intelligence: DECIDE-AI. *Nat Med* 2022;28(5):924-933. DOI 10.1038/s41591-022-01772-9 | PubMed (PMID 35585198), resumo; texto do *Nat Med* não acessível (redirecionamento para login) |
| Vasey B et al. DECIDE-AI (mesma diretriz). *BMJ* 2022;377:e070904. DOI 10.1136/bmj-2022-070904 | texto integral no PMC (PMC9116198): lista de itens e glossário (*live* × *offline/shadow mode*) |
| Lei 14.874, de 28/05/2024 — Arts. 2 XXXIV, 14, 15, 18, 18 §6, 19, 51 §1, 61, 65 | planalto.gov.br/ccivil_03/_ato2023-2026/2024/lei/l14874.htm |
| Decreto 12.651, de 07/10/2025 — Arts. 10, 34, 37, 39, 40, 41 | planalto.gov.br/ccivil_03/_ato2023-2026/2025/decreto/d12651.htm |
| Resolução CNS 466/2012 — itens IV.8, XI.2.f | PDF oficial: gov.br/conselho-nacional-de-saude/pt-br/atos-normativos/resolucoes/2012/resolucao-no-466.pdf |
| Resolução CNS 580/2018 (22/03/2018; DOU 16/07/2018) — Arts. 2, 3, 5, 6, 7, 9, 10 | PDF oficial: gov.br/conselho-nacional-de-saude/pt-br/atos-normativos/resolucoes/2018/resolucao-no-580.pdf |
| LGPD, Lei 13.709, de 14/08/2018 — Arts. 5 XVIII, 11 II c, 13 (caput, §1, §2, §4), 37, 38, 46 | planalto.gov.br/ccivil_03/_ato2015-2018/2018/lei/l13709.htm (texto com falhas de codificação na página; dispositivos conferidos por trecho) |
| Res. CNS 738/2024; Res. CNS 510/2016; Res. CFM 2.454/2026 | **não conferidas** neste documento |
| Métodos de detecção de deriva; método de IC para kappa | **não conferido** — parâmetros do protocolo |

---

## Histórico de revisões

| Versão | Data | Alteração |
|---|---|---|
| v0.1 | 2026-11-28 | Criação do modelo; base técnica REQ-195 (implementado no software; integração no hospital pendente); DECIDE-AI aplicado como arcabouço adaptado ao *shadow mode* |
