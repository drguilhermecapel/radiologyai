# MedAI Radiologia — project rules

Research-only radiology AI platform. Its core is an **evidence-graded catalog of
conditions**: every suggested finding carries the level of evidence behind it.
Nothing here is a medical device; output is "suggested finding for physician
confirmation".

## The golden rule

Product code never emits a prediction, metric, heatmap or regulatory statement
that a real model or algorithm did not compute from the actual input.

- A model that cannot run raises (`ModelUnavailable`, HTTP 503) or abstains with a
  reason. It is never replaced by a constant, a heuristic or a random value.
- A model that could not be made to run for real is `status: planned` in its
  manifest. No stubs.
- No performance number appears in docs, README or UI text without a link to a
  file under `evidence/` (or a verified citation). No claim of regulatory status.
- Test doubles live in `tests/support/` only and are never importable from `medai`.

`tools/lint_no_fabrication.py` and the test suite enforce this; keep them green.

## Evidence levels (computed, never typed)

| Level | Meaning | Shown |
|---|---|---|
| A | Externally validated by our own measurement (public labelled data the model was not trained on) | main view, report |
| B | Peer-reviewed published validation and/or guideline threshold, quote verified at source | main view, report |
| C | Internally validated by our own measurement (patient-level hold-out) | main view, with badge |
| D | Exploratory (zero-shot, or no per-condition validation) | research mode only, never in the report by itself |

`src/medai/catalog/schema.py::compute_level` is the rule; `protocol.yaml` holds
the pre-registered thresholds. Changing a threshold needs a new `protocol_id`.

## Commands

```bash
uv sync --all-extras                 # CPU wheels: add --extra cpu on machines without a GPU
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run pytest                        # fast suite (no weights, no network)
uv run pytest -m weights             # needs `medai models pull` first
uv run python tools/lint_no_fabrication.py
```

Python 3.11 and 3.12. Code, identifiers and comments in English; documents and
user-facing text in Brazilian Portuguese (UI is bilingual PT-BR/EN).

## Layout

```
src/medai/
  core/        types, errors, settings, artifact store, policy protocol   (frozen)
  models/      manifest schema, adapter protocol, registry (frozen); adapters/, manifests/
  catalog/     conditions, evidence schema, level rule, protocol.yaml      (schema frozen)
  pipeline.py  QA -> predict -> calibrate -> operating point -> decision -> level (frozen)
  routing.py   exam-type inference + routing.yaml                          (frozen)
  io/ preprocess/ qa/ safety/ explain/ measure/ eval/ data/ train/ probes/
  report/ interop/ monitoring/ db/ api/ worker/ cli.py
frontend/      React + TypeScript + Vite
docs/          Portuguese docs, ROADMAP, regulatory skeleton
evidence/      generated measurement records + verified published evidence
tools/         linters and maintenance scripts
tests/         unit/, integration/, e2e/, support/ (fakes and synthetic inputs)
```

## Frozen contracts

`core/types.py`, `core/errors.py`, `core/policy.py`, `core/artifacts.py`,
`models/{manifest,base,registry}.py`, `catalog/schema.py`, `catalog/protocol.yaml`,
`pipeline.py`, `routing.py`, `routing.yaml`, `pyproject.toml`, `uv.lock`,
`tests/conftest.py`.

Everything else is built against these. If a contract is wrong or insufficient,
do not edit it and do not work around it: report a **contract change request**
(what, why, the smallest change) to the lead, who applies it for everyone.

## Rules for adapters (`src/medai/models/adapters/`)

- One adapter wraps one real model. It returns raw scores, label maps and
  measurements (`AdapterOutput`). It never thresholds, calibrates or decides.
- Every label an adapter emits is declared in its manifest and mapped to a
  catalog condition.
- Weights are downloaded only by `medai models pull`, verified against the
  SHA-256 pinned in the manifest, and loaded from the local directory. No network
  access at inference time. Hashes are written by `medai models pin` from the real
  bytes, never typed.
- The manifest states licence (code, weights, commercial use) with its source.
  Excluded licences: GPL/AGPL code linked into the package, and anything whose
  terms forbid research use. Non-commercial weights are allowed, flagged
  `commercial_use: "no"`.
- Coded concepts: only codes present in `pydicom.sr.codedict`, or the private
  scheme `99MEDAI`. Never invent SNOMED/RadLex codes.
- Quantitative CT/MR measurements are emitted only when the acquisition
  preconditions hold; otherwise the measurement abstains with a reason.

## Privacy

- De-identify on ingest by default; store a salted pseudonym, not identifiers.
- No patient identifiers, file names or free-text DICOM fields in logs or errors.
- Evaluation datasets are downloaded to `MEDAI_HOME/datasets`, never versioned.
  Unofficial mirrors of datasets that require a data-use agreement are not used.

## Git

Commit messages in Portuguese (conventional prefix). Never commit weights,
datasets, databases or secrets.
