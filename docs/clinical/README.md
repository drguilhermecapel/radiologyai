# Protocolos clínicos (modelos)

> **Software de pesquisa. Não é dispositivo médico.** Os documentos desta pasta são **modelos de
> protocolo — não submetidos a CEP; não iniciados**. Nenhum estudo com leitores ou com pacientes
> foi conduzido. Estado real do projeto: [`HONEST_STATUS.md`](../../HONEST_STATUS.md).

| Documento | Portão ([`ROADMAP.md`](../../ROADMAP.md) §4) | Pergunta | Estado |
|---|---|---|---|
| [`protocolo-mrmc-G3.md`](protocolo-mrmc-G3.md) | G3 — estudo de leitores MRMC | o médico assistido erra menos? | modelo v0.1; não submetido a CEP; não iniciado |
| [`protocolo-piloto-silencioso-G4.md`](protocolo-piloto-silencioso-G4.md) | G4 — piloto prospectivo silencioso | funciona no fluxo real sem influenciar o cuidado? | modelo v0.1; não submetido a CEP; não iniciado; base técnica REQ-195 (implementado no software; integração no hospital pendente) |

Regras comuns aos dois modelos:

- Só condições em **nível A** em `src/radiologyai/catalog/levels.json` na data do congelamento
  do protocolo; hoje, `cxr.cardiomegaly` e `cxr.lung_opacity`, com as ressalvas transcritas em
  `protocolo-mrmc-G3.md` §0.1.
- Nenhum tamanho de efeito, tamanho amostral, limiar de deriva ou de parada é proposto: são
  parâmetros `⟨…⟩` a fixar antes da submissão, com estatístico.
- Toda referência tem o estado de conferência declarado na tabela final de cada documento;
  o que não foi conferido na fonte está marcado "não conferido".
- Ligações: plano de avaliação clínica [`../regulatory/10-clinical-evaluation-plan.md`](../regulatory/10-clinical-evaluation-plan.md) §4;
  protocolo de evidência [`../protocolo-evidencia.md`](../protocolo-evidencia.md) §10;
  LGPD [`../regulatory/11-cybersecurity-lgpd.md`](../regulatory/11-cybersecurity-lgpd.md).
