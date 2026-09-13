# Model cards

Um arquivo YAML por versão de modelo. Cada card é validado na carga:

- `weights_sha256` **deve** ser 64 caracteres hexadecimais minúsculos.
  Placeholders são rejeitados — foi exatamente isso que passou despercebido em
  `models/model_registry.json` do sistema legado.
- **Não existe campo de desempenho.** Acurácia, AUC e sensibilidade vivem em
  `artifacts/eval/<run_id>/metrics.json`, produzidos por execução medida.
  O card apenas referencia esses run_ids em `evaluation_runs`.
- `evaluation_runs` vazio significa: **nenhum desempenho foi medido, e nenhum
  pode ser alegado.**

Este diretório está vazio de propósito. O primeiro card entra na Fase 1 do
[ROADMAP](../../../../ROADMAP.md), junto com a primeira medição real
(`xrv-densenet121-pc` avaliado no split oficial de teste do NIH ChestX-ray14).
