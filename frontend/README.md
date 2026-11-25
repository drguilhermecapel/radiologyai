# RadiologyAI — interface web de pesquisa

> **Uso exclusivo em pesquisa — não é dispositivo médico.** Toda saída é
> *achado sugerido para confirmação médica*; nunca é diagnóstico nem afirma
> normalidade. A faixa com esse aviso aparece em todas as telas.

Aplicação React 19 + TypeScript + Vite que conversa só com a API REST do
pacote `radiologyai` (`radiologyai serve`). Os tipos da API são **gerados** de
`docs/OPENAPI_SPEC.yaml` com `openapi-typescript`; a interface não inventa
nenhum número: escores, decisões, mapas de calor, segmentações e medidas vêm
exatamente do resultado calculado pelo servidor.

## Telas

| Tela | O que faz |
|------|-----------|
| Login | Sessão por token opaco (guardado só em `sessionStorage`). Sem usuário padrão: usuários são criados pela CLI. |
| Lista de trabalho | Filtros por estado da análise, tipo de exame, modalidade e revisão; ação "definir tipo de exame" quando o estudo chegou como `unknown`; envio de estudos (DICOM, ZIP, NIfTI, PNG/JPEG). |
| Estudo | Visualizador em canvas + achados + revisão + laudo (ver abaixo). |
| Condições e modelos | Catálogo de condições por exame com o melhor nível de evidência e a base (links); fichas dos modelos com licença do código e dos pesos, **uso comercial** (sinalizado quando "não"), estado dos pesos (hash verificado), limitações, dados de treino e referências. |
| Auditoria | Só administradores: eventos encadeados por hash, filtros e verificação da cadeia feita pelo servidor. |

### Visualizador

* Quadros inteiros (`uint16`/`int16`/`uint8`) do endpoint `/series/{id}/frames/{i}`;
  janela/nível aplicada **no cliente** pela função linear VOI do DICOM
  (PS3.3 C.11.2.1.2.1), com `RescaleSlope/Intercept` (HU na TC).
* Presets: automático (percentis dos pixels reais), janela do DICOM, faixa
  completa e, na TC, pulmão/mediastino/abdome/fígado/osso/cérebro/subdural/AVC.
* Zoom (roda do mouse ancorada no cursor, teclas `+`/`-`), deslocamento,
  ajuste de janela por arrasto, rolagem de cortes/quadros (setas, PageUp/PageDown, controle deslizante).
* Sobreposições: mapa de calor PNG do modelo colorido (inferno/jet/gray, com
  opacidade) e posicionado **pela bbox `placement`** do contrato; mapas de
  rótulos de segmentação com legenda e liga/desliga por rótulo; medidas
  (pontos, linhas, caixas, polígonos) numa camada SVG; caixas-prompt
  desenhadas pelo usuário para modelos interativos (enviadas em `prompts`).

### Achados

* Agrupados por nível de evidência **A / B / C**, cada um com selo e o
  significado em linguagem simples.
* Nível **D** (exploratório) só aparece no **modo pesquisa**, que é um
  interruptor restrito a papéis com `research:view` e mostra uma faixa forte
  enquanto ativo. Achados D nunca são revisáveis nem entram no laudo.
* O número exibido é **"escore"** (com o tipo: logit, sigmoide…) e a relação
  com cada ponto de operação publicado/medido (limiar, sentido e fonte). A
  palavra "probabilidade" só aparece quando o achado traz probabilidade **e**
  identificador de calibração.
* Abstenções aparecem com o motivo; cada achado tem a proveniência (modelo,
  versão, licença, uso comercial, hashes do manifesto e dos pesos).

### Revisão e laudo

* Aceitar / rejeitar / emendar cada achado, com comentário; a revisão é
  amarrada ao hash do resultado mostrado.
* Gerar o laudo, assinar (senha + os dois hashes que o médico viu) e baixar
  PDF, DICOM SR, FHIR e DICOM SEG. Se o servidor detectar integridade violada,
  os downloads somem.

### Idiomas e acessibilidade

PT-BR (padrão) e EN; o compilador garante que o inglês cobre todas as chaves.
Link "pular para o conteúdo", foco visível, rótulos em todos os controles,
regiões `aria-live` para estados assíncronos, contraste AA em tema claro e
escuro, leiaute responsivo (uma coluna abaixo de 1100 px).

## Desenvolvimento

Requer Node 22. Versões fixadas (`save-exact`).

```bash
cd frontend
npm ci
npm run gen:api      # regenera src/api/schema.d.ts a partir de docs/OPENAPI_SPEC.yaml
npm run check:api    # falha se os tipos estiverem desatualizados
npm run typecheck
npm test             # vitest: janela/nível, geometria das sobreposições, níveis de evidência, quadros, rotas, i18n
npm run build        # gera dist/
```

Com o backend rodando (`radiologyai serve`, porta 8000 por padrão):

```bash
RADIOLOGYAI_BACKEND=http://127.0.0.1:8000 npm run dev
```

O Vite repassa `/api` ao backend (também em `npm run preview`).

## Teste de ponta a ponta (backend real)

`tests/e2e/run-e2e.mjs` sobe o backend de verdade (SQLite descartável, usuários
criados pela CLI com senhas aleatórias, worker em processo, pesos reais em
`MEDAI_WEIGHTS_DIR`), serve `frontend/dist` com `vite preview` e dirige um único
Chromium (biblioteca `playwright` do sistema; **não** rode `playwright install`):

login → envio de um DICOM de tórax real → análise → mapa de calor e
segmentação no visualizador → revisão de cada achado → laudo → assinatura →
download e conferência do PDF → catálogo → auditoria com verificação da cadeia.

```bash
cd frontend && npm run build && cd ..
E2E_SCREENS_DIR=/caminho/capturas node tests/e2e/run-e2e.mjs
```

Sem DICOM, sem pesos ou sem `playwright`, o script sai com código 77
(pulado) — nunca finge sucesso. Variáveis: `E2E_DICOM`, `E2E_MODELS`
(ids separados por vírgula ou `auto`; padrão: os modelos principais de tórax), `E2E_PATIENT_AGE` (idade informada no envio; o DICOM de exemplo não traz idade e os modelos de adulto se abstêm sem ela), `E2E_WORKDIR`, `E2E_BACKEND_PORT`,
`E2E_UI_PORT`, `E2E_ANALYSIS_TIMEOUT_MS`, `E2E_VERBOSE`.

## Estrutura

```
src/api/        cliente tipado (openapi-fetch) e tipos gerados
src/i18n/       mensagens PT-BR/EN e provedor
src/lib/        lógica pura testada: windowLevel, overlay, evidence, frames, route
src/viewer/     visualizador em canvas e carga de quadros/mapas
src/components/ painel de achados, laudo, componentes comuns
src/pages/      login, lista de trabalho, estudo, catálogo/modelos, auditoria
```

> Estado atual do catálogo: enquanto nenhuma condição tiver evidência A–C, a
> visão principal mostra "nenhum achado de nível A–C" e o teste de ponta a
> ponta confere os achados calculados ligando o modo pesquisa (nível D); a
> revisão por achado só se aplica a A–C.
