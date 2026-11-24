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
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

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
from radiologyai.evidence.store import dump_json

CATALOG_DIR = Path(__file__).resolve().parents[1] / "catalog"
LEVELS_PATH = CATALOG_DIR / "levels.json"

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


def build_levels(
    records: list[EvidenceRecord], conditions: list[Condition], protocol: Protocol
) -> LevelsFile:
    """Calcula ``LevelsFile`` a partir da evidência.

    Raises:
        EvidenceError: registro para condição inexistente no catálogo, ou registro
            medido julgado sob outro ``protocol_id`` (precisa ser re-julgado).
    """
    known = {c.id for c in conditions}
    by_triple: dict[tuple[str, str, str], list[EvidenceRecord]] = defaultdict(list)
    for rec in records:
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
        by_triple[(rec.condition_id, rec.model_id, rec.label)].append(rec)
    for cond in conditions:
        for src in cond.sources:
            by_triple.setdefault((cond.id, src.model_id, src.label), [])

    entries: list[LevelEntry] = []
    for triple in sorted(by_triple):
        recs = by_triple[triple]
        level, basis_ids, refuted = compute_level(recs) if recs else (EvidenceLevel.D, [], False)
        basis = [r for r in recs if r.id in set(basis_ids)]
        entries.append(
            LevelEntry(
                condition_id=triple[0],
                model_id=triple[1],
                label=triple[2],
                level=level,
                visibility=visibility_for(level),
                basis=tuple(sorted(basis_ids)),
                refuted_by_measurement=refuted,
                headline=_headline(level, basis, refuted),
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
