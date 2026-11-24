# Manifestos de modelo

Um arquivo `<id>.yaml` por modelo; o nome do arquivo é o `id`. Esquema em
`radiologyai/analysis/manifest.py` (`ModelManifest`, `extra="forbid"`).

Regras que o registro impõe ao carregar:

- `status: available` exige `adapter: radiologyai.analysis.adapters.<módulo>:<Classe>`
  e todos os pesos com `sha256` escrito por `radiologyai.analysis.registry.pin` a
  partir dos bytes reais. Nunca digite um hash.
- `status: planned` exige `planned_reason`; nunca é carregado nem baixado. É a forma
  honesta de listar um modelo que ainda não roda de verdade — sem stub.
- Nenhuma métrica de desempenho, em nenhuma profundidade. Evidência vive em
  `artifacts/eval/` e no catálogo de condições.
- Códigos de rótulo: `pydicom.sr.codedict` com o significado oficial, ou `99MEDAI`.
- Licença: `code`, `weights`, `commercial_use` (`yes`/`no`/`unclear`), `evidence`.
  Código GPL/AGPL é recusado; pesos não comerciais são aceitos e sinalizados.
- `route`: `main` | `research` | `interactive` | `dependency` | `none`. Zero-shot é
  sempre `research`. O roteamento por tipo de exame deriva daqui (`routing.yaml`).
