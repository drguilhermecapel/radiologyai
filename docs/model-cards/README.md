# Model cards

Um model card descreve **um conjunto de pesos** de um modelo: origem, arquitetura, dados de
treino declarados pelo desenvolvedor, entradas esperadas, rótulos emitidos, pré-condições,
licença de código e de pesos, uso comercial e limitações conhecidas.

Regras deste projeto:

1. **A fonte da verdade é o manifesto/card versionado no pacote** (`src/radiologyai/`), com
   sha256 dos pesos calculado a partir dos bytes reais. Um documento aqui só complementa o
   manifesto com texto; nunca o contradiz.
2. **Nenhuma métrica de desempenho em model card.** Campos `accuracy`, `auc`, `sensitivity`,
   `specificity` são proibidos pelo `scripts/check_honesty.py` (regra R6). Desempenho vive em
   `artifacts/eval/` e no catálogo de condições, com proveniência.
3. Números relatados pelo desenvolvedor (cartão do Hugging Face, README) são citados só como
   **alegação do desenvolvedor**, com a fonte, e nunca promovem nível de evidência — a não ser que
   sejam publicação revisada por pares conferida na fonte, pela barra do
   [protocolo](../protocolo-evidencia.md) §6.
4. Licenças: ver o inventário em [`MODEL_LICENSES.md`](../../MODEL_LICENSES.md).

Os cartões textuais por modelo serão adicionados aqui à medida que cada manifesto entrar no
pacote.
