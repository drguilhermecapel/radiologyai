"""Construção determinística de ``src/radiologyai/catalog/levels.json``.

Lê os registros de ``evidence/`` e as condições de ``conditions.yaml``, agrupa
por trio (condição, modelo, rótulo) e aplica
:func:`~radiologyai.catalog.schema.compute_level`. O arquivo resultante é uma
função pura das entradas: mesma evidência e mesmo catálogo produzem os mesmos
bytes. ``--check`` (CI) compara os bytes e falha se ``levels.json`` estiver
desatualizado.

Trios declarados em ``conditions.yaml`` (``sources``) sem nenhum registro
entram com nível D e sem base: o catálogo mostra que existem e que não têm
evidência.

O JSON de um registro nunca é aceito pelo que diz de si mesmo. Antes de calcular
o nível, cada registro medido é **re-julgado** (:func:`~radiologyai.evidence.
measure.rejudge`): motivos, veredito, métricas e contagens recalculados do
relatório completo contra o protocolo vigente, e vazamento/tipo recalculados das
declarações (conferidas contra o plano, se houver). Qualquer divergência é erro.
Depois, o nível é **limitado a D** (com o motivo em ``limited_by``) quando:

* o modelo não tem manifesto no registro, ou os pesos medidos não constam dele
  com o mesmo sha256 — não é o artefato distribuído (``distributed: false``);
* o manifesto declara ``visibility_default: research`` (termo de uso ou dados de
  treino que restringem o modelo ao modo pesquisa);

e registros medidos com árvore git suja ficam fora do cálculo.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from radiologyai.catalog.schema import (
    Condition,
    EvidenceLevel,
    EvidenceRecord,
    LevelEntry,
    LevelsFile,
    Protocol,
    compute_level,
    visibility_for,
)
from radiologyai.errors import EvidenceError
from radiologyai.evidence.measure import MeasuredEvidence, rejudge
from radiologyai.evidence.store import dump_json

if TYPE_CHECKING:
    from collections.abc import Sequence

    from radiologyai.evidence.plan import Plan

CATALOG_DIR = Path(__file__).resolve().parents[1] / "catalog"
LEVELS_PATH = CATALOG_DIR / "levels.json"
MANIFESTS_DIR = Path(__file__).resolve().parents[1] / "analysis" / "manifests"

_HEADLINE_METRICS = ("auroc", "sensitivity", "specificity", "icc", "dice_mean")


def load_conditions(path: Path = CATALOG_DIR / "conditions.yaml") -> list[Condition]:
    import yaml

    try:
        raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise EvidenceError(f"não foi possível ler {path}: {exc}") from exc
    items = raw.get("conditions") if isinstance(raw, dict) else None
    if not isinstance(items, list):
        raise EvidenceError(f"{path}: a chave 'conditions' deve ser uma lista")
    try:
        return [Condition.model_validate(i) for i in items]
    except ValidationError as exc:
        raise EvidenceError(f"condição inválida em {path}: {exc}") from exc


def evidence_digest(records: list[EvidenceRecord]) -> str:
    """sha256 sobre os registros, ordenados por id, em JSON canônico."""
    digest = hashlib.sha256()
    for rec in sorted(records, key=lambda r: r.id):
        digest.update(
            json.dumps(rec.model_dump(mode="json"), sort_keys=True, ensure_ascii=False).encode()
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _headline(level: EvidenceLevel, basis: list[EvidenceRecord], refuted: bool) -> str | None:
    if not basis:
        return None
    rec = sorted(basis, key=lambda r: r.id)[0]
    if refuted:
        reasons = "; ".join(rec.failure_reasons) or "reprovado no protocolo"
        return f"refutado por medição própria em {rec.dataset}: {reasons}"
    for name in _HEADLINE_METRICS:
        m = rec.metrics.get(name)
        if m is None:
            continue
        ci = (
            f" [{m.ci_low:.3f}, {m.ci_high:.3f}]"
            if m.ci_low is not None and m.ci_high is not None
            else ""
        )
        where = rec.dataset or (rec.source.identifier if rec.source else None) or rec.kind
        return f"{level.value}: {name} {m.value:.3f}{ci} · {where} · n={rec.n_total}"
    return f"{level.value}: {rec.kind} · {rec.id}"


def _manifest_info(model_id: str, manifests_dir: Path) -> dict[str, Any] | None:
    """``{"weights": {arquivo: sha256}, "visibility_default": ...}`` ou ``None``."""
    import yaml

    path = manifests_dir / f"{model_id}.yaml"
    if not path.is_file():
        return None
    try:
        raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise EvidenceError(f"manifesto ilegível {path}: {exc}") from exc
    if not isinstance(raw, dict) or raw.get("id") != model_id:
        raise EvidenceError(f"{path}: manifesto sem id {model_id!r}")
    weights = {
        str(w["filename"]): str(w["sha256"])
        for w in raw.get("weights") or ()
        if isinstance(w, dict) and w.get("filename") and w.get("sha256")
    }
    return {"weights": weights, "visibility_default": raw.get("visibility_default", "main")}


def _published_promotes(rec: EvidenceRecord, protocol: Protocol) -> list[str]:
    """Motivos pelos quais um registro publicado gravado como aprovado não passa na barra.

    Espelha o julgamento da ingestão (``scripts/evidence_ingest_published.py``):
    só a direção que promove é conferida — um registro publicado só conta se a
    barra do protocolo, recalculada dos números gravados, o aprova.
    """
    out: list[str] = []
    src = rec.source
    if src is None or not src.verified_at_source:
        out.append("citação não conferida na fonte")
    if rec.kind == "published" and (src is None or not src.peer_reviewed):
        out.append("fonte não revisada por pares")
    if rec.n_positive is None or rec.n_positive < protocol.published_min_positive:
        out.append(f"positivos {rec.n_positive} < {protocol.published_min_positive}")

    def check(name: str, minimum: float) -> None:
        m = rec.metrics.get(name)
        if m is None or m.value < minimum or m.ci_low is None or m.ci_high is None:
            out.append(f"{name} ausente, sem IC ou abaixo de {minimum}")

    if rec.endpoint == "discrimination":
        check("auroc", protocol.published_auroc_min)
    elif rec.endpoint == "detection":
        check("sensitivity", protocol.published_sens_spec_min)
        check("specificity", protocol.published_sens_spec_min)
    elif rec.endpoint == "segmentation":
        check("dice", protocol.segmentation_dice_min)
    elif rec.kind != "guideline_threshold":
        out.append(f"desfecho {rec.endpoint!r} sem barra de publicação")
    return out


def _check_against_plan(env: MeasuredEvidence, plan: Plan) -> None:
    rec = env.record
    job_id = env.inputs.get("plan_job")
    job = next((j for j in plan.jobs if j.id == job_id), None)
    if job is None or job.status != "measure":
        raise EvidenceError(
            f"registro medido {rec.id!r} não corresponde a item 'measure' do plano "
            f"(plan_job={job_id!r}); medição fora do plano declarado não entra no catálogo"
        )
    declared = env.declared
    assert declared is not None  # rejudge já exigiu
    pairs = (
        ("model_id", rec.model_id, job.model_id),
        ("dataset", rec.dataset, job.dataset_id),
        ("label", rec.label, job.label),
        ("condition_id", rec.condition_id, job.condition_id),
        ("training_data", declared.training_data, tuple(job.training_data)),
        ("selection_data", declared.selection_data, tuple(job.selection_data)),
        ("internal_holdout", declared.internal_holdout, job.internal_holdout),
        (
            "reference_derived_input",
            declared.reference_derived_input,
            job.reference_derived_input,
        ),
    )
    diverging = [f"{n}: registro {a!r} != plano {b!r}" for n, a, b in pairs if a != b]
    if diverging:
        raise EvidenceError(
            f"registro {rec.id!r} diverge do item {job.id!r} do plano: " + "; ".join(diverging)
        )


def build_levels(
    evidence: Sequence[EvidenceRecord | MeasuredEvidence],
    conditions: list[Condition],
    protocol: Protocol,
    *,
    manifests_dir: Path = MANIFESTS_DIR,
    plan: Plan | None = None,
) -> LevelsFile:
    """Calcula ``LevelsFile`` a partir da evidência, re-julgando cada registro.

    ``evidence``: registros publicados (``EvidenceRecord``) e envelopes medidos
    (``MeasuredEvidence``). Registro medido sem envelope não é aceito: sem o
    relatório não há como re-julgá-lo.

    Raises:
        EvidenceError: registro para condição inexistente no catálogo; registro
            medido julgado sob outro ``protocol_id``, sem envelope, divergente do
            re-julgamento ou do plano; publicado gravado como aprovado que não passa
            na barra recalculada.
    """
    known = {c.id for c in conditions}
    records: list[EvidenceRecord] = []
    envelopes: dict[str, MeasuredEvidence] = {}
    for item in evidence:
        rec = item.record if isinstance(item, MeasuredEvidence) else item
        if rec.condition_id not in known:
            raise EvidenceError(
                f"registro {rec.id!r} aponta para condição inexistente {rec.condition_id!r}; "
                "adicione a condição a conditions.yaml ou corrija o registro"
            )
        if rec.provenance is not None and rec.provenance.protocol_id != protocol.protocol_id:
            raise EvidenceError(
                f"registro {rec.id!r} foi julgado sob o protocolo "
                f"{rec.provenance.protocol_id!r}, e o vigente é {protocol.protocol_id!r}: "
                "re-execute a medição sob o protocolo vigente"
            )
        measured = rec.kind in ("measured_external", "measured_internal")
        if measured:
            if not isinstance(item, MeasuredEvidence):
                raise EvidenceError(
                    f"registro medido {rec.id!r} sem envelope (relatório completo): não pode "
                    "ser re-julgado; re-execute a medição"
                )
            mismatches = rejudge(item, protocol)
            if mismatches:
                raise EvidenceError(
                    f"registro {rec.id!r} não confere com o re-julgamento pelo protocolo "
                    f"{protocol.protocol_id}: " + "; ".join(mismatches)
                )
            if plan is not None:
                _check_against_plan(item, plan)
            envelopes[rec.id] = item
        elif rec.passes_protocol is True:
            failing = _published_promotes(rec, protocol)
            if failing:
                raise EvidenceError(
                    f"registro publicado {rec.id!r} gravado como aprovado não passa na barra "
                    "recalculada: " + "; ".join(failing)
                )
        records.append(rec)

    by_triple: dict[tuple[str, str, str], list[EvidenceRecord]] = defaultdict(list)
    for rec in records:
        by_triple[(rec.condition_id, rec.model_id, rec.label)].append(rec)
    for cond in conditions:
        for src in cond.sources:
            by_triple.setdefault((cond.id, src.model_id, src.label), [])

    manifests: dict[str, dict[str, Any] | None] = {}
    entries: list[LevelEntry] = []
    for triple in sorted(by_triple):
        recs = by_triple[triple]
        model_id = triple[1]
        if model_id not in manifests:
            manifests[model_id] = _manifest_info(model_id, manifests_dir)
        info = manifests[model_id]
        limits: list[str] = []

        dirty = [
            r
            for r in recs
            if r.id in envelopes and (d := envelopes[r.id].declared) is not None and d.git_dirty
        ]
        for r in sorted(dirty, key=lambda r: r.id):
            limits.append(
                f"{r.id}: medido com código não commitado (git_dirty); fora do cálculo até "
                "ser re-medido a partir de um commit"
            )
        dirty_ids = {r.id for r in dirty}
        accepted = [r for r in recs if r.id not in dirty_ids]
        level, basis_ids, refuted = (
            compute_level(accepted) if accepted else (EvidenceLevel.D, [], False)
        )
        basis = [r for r in accepted if r.id in set(basis_ids)]

        distributed = True
        if info is None:
            distributed = False
            limits.append(
                f"modelo {model_id!r} sem manifesto no registro de modelos: não é artefato "
                "distribuído; nível limitado a D e fora de qualquer contagem"
            )
        else:
            for r in sorted(recs, key=lambda r: r.id):
                if r.provenance is None:
                    continue
                for fname, sha in sorted(r.provenance.weights_sha256.items()):
                    if info["weights"].get(fname) != sha:
                        distributed = False
                        limits.append(
                            f"{r.id}: pesos medidos {fname} não constam do manifesto "
                            f"{model_id!r} com o mesmo sha256 (a medição não é do artefato "
                            "distribuído)"
                        )
            if info["visibility_default"] == "research":
                limits.append(
                    f"manifesto {model_id!r} declara visibility_default: research (termo de "
                    "uso ou dados de treino restringem o modelo ao modo pesquisa); nível "
                    "limitado a D"
                )
        for r in sorted(basis, key=lambda r: r.id):
            if r.reference_derived_input:
                limits.append(
                    f"{r.id}: entrada derivada do padrão de referência "
                    f"({r.reference_derived_input}); nível A vedado"
                )

        capped_from: EvidenceLevel | None = None
        research_only = info is not None and info["visibility_default"] == "research"
        restricted = not distributed or research_only
        if restricted and level is not EvidenceLevel.D:
            capped_from, level = level, EvidenceLevel.D
        headline = _headline(capped_from or level, basis, refuted)
        if capped_from is not None and headline is not None:
            headline = f"limitado a D (a medição sustentaria {capped_from.value}) · {headline}"
        entries.append(
            LevelEntry(
                condition_id=triple[0],
                model_id=model_id,
                label=triple[2],
                level=level,
                visibility=visibility_for(level),
                basis=tuple(sorted(basis_ids)),
                refuted_by_measurement=refuted,
                headline=headline,
                distributed=distributed,
                limited_by=tuple(limits),
            )
        )
    return LevelsFile(
        protocol_id=protocol.protocol_id,
        evidence_digest=evidence_digest(records),
        entries=tuple(entries),
    )


def render_levels(levels: LevelsFile) -> str:
    """Bytes canônicos de ``levels.json``."""
    return dump_json(levels.model_dump(mode="json"))


def check_levels(expected: LevelsFile, path: Path = LEVELS_PATH) -> tuple[bool, str]:
    """``(atualizado?, mensagem)``. Arquivo ausente conta como desatualizado."""
    if not path.is_file():
        return False, f"{path} não existe; gere com scripts/evidence_build_levels.py"
    current = path.read_text(encoding="utf-8")
    if current != render_levels(expected):
        return False, (
            f"{path} está desatualizado em relação a evidence/ e conditions.yaml; "
            "regere com scripts/evidence_build_levels.py"
        )
    return True, f"{path} atualizado ({len(expected.entries)} trios)"
