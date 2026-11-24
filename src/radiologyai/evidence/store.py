"""Leitura e escrita do diretório ``evidence/``.

Estrutura::

    evidence/
      measured/<dataset>/<model>/<label>.json   # escritos por scripts/evidence_run.py
      published/**/*.json|*.yaml                # registros publicados conferidos na fonte

Um arquivo medido é um envelope (:class:`~radiologyai.evidence.measure.MeasuredEvidence`)
cujo campo ``record`` é o :class:`~radiologyai.catalog.schema.EvidenceRecord`. Um
arquivo publicado contém um registro, uma lista de registros ou
``{"records": [...]}``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from radiologyai.catalog.schema import EvidenceRecord
from radiologyai.errors import EvidenceError
from radiologyai.evidence.measure import MeasuredEvidence

EVIDENCE_ROOT = Path(__file__).resolve().parents[3] / "evidence"
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")

# scripts/check_honesty.py (R3) exige que toda linha com chave "...sha256" traga
# 64 hex na mesma linha. Uma chave cujo valor é um mapeamento (weights_sha256:
# {nome: hash}) teria "{" nessa posição; o valor é quebrado para a linha seguinte
# (JSON válido) para que cada hash fique sozinho na sua linha e seja verificado.
_SHA_KEY_OPENING = re.compile(r'^(\s*"[^"\n]*sha256[^"\n]*"):\s*([{\[])$', re.M)


def slug(text: str) -> str:
    """Nome de arquivo seguro e estável para dataset, modelo ou rótulo."""
    s = _SAFE.sub("_", text.strip()).strip("._")
    if not s:
        raise EvidenceError(f"nome vazio após normalização: {text!r}")
    return s


def dump_json(data: Any) -> str:
    """JSON determinístico (chaves ordenadas, indentação 2, newline final)."""
    text = json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False)
    return _SHA_KEY_OPENING.sub(r"\1:\n\2", text) + "\n"


def measured_path(root: Path, dataset: str, model: str, label: str) -> Path:
    return root / "measured" / slug(dataset) / slug(model) / f"{slug(label)}.json"


def write_measured(evidence: MeasuredEvidence, root: Path = EVIDENCE_ROOT) -> Path:
    """Grava o envelope em ``measured/<dataset>/<modelo>/<rótulo>.json``."""
    rec = evidence.record
    if rec.dataset is None:
        raise EvidenceError("registro medido sem dataset")
    path = measured_path(root, rec.dataset, rec.model_id, rec.label)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump_json(evidence.model_dump(mode="json")), encoding="utf-8")
    return path


def _records_from(data: Any, path: Path) -> list[EvidenceRecord]:
    if isinstance(data, dict) and "record" in data:
        try:
            return [MeasuredEvidence.model_validate(data).record]
        except ValidationError as exc:
            raise EvidenceError(f"envelope de evidência inválido em {path}: {exc}") from exc
    items: Any
    if isinstance(data, dict) and "records" in data:
        items = data["records"]
    elif isinstance(data, list):
        items = data
    else:
        items = [data]
    if not isinstance(items, list):
        raise EvidenceError(f"{path}: 'records' deve ser uma lista")
    try:
        return [EvidenceRecord.model_validate(item) for item in items]
    except ValidationError as exc:
        raise EvidenceError(f"registro de evidência inválido em {path}: {exc}") from exc


def _read(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise EvidenceError(f"JSON inválido em {path}: {exc}") from exc
    import yaml

    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise EvidenceError(f"YAML inválido em {path}: {exc}") from exc


def evidence_files(root: Path = EVIDENCE_ROOT) -> list[Path]:
    """Arquivos de evidência em ordem determinística."""
    files: list[Path] = []
    measured = root / "measured"
    if measured.is_dir():
        files += sorted(measured.rglob("*.json"))
    published = root / "published"
    if published.is_dir():
        files += sorted(
            p
            for p in published.rglob("*")
            if p.suffix in (".json", ".yaml", ".yml") and p.is_file()
        )
    return files


def load_records(root: Path = EVIDENCE_ROOT) -> list[EvidenceRecord]:
    """Todos os registros de ``measured/`` e ``published/``.

    Raises:
        EvidenceError: arquivo inválido, ou dois registros com o mesmo id.
    """
    records: list[EvidenceRecord] = []
    origin: dict[str, Path] = {}
    for path in evidence_files(root):
        for rec in _records_from(_read(path), path):
            if rec.id in origin:
                raise EvidenceError(
                    f"id de evidência duplicado {rec.id!r}: {origin[rec.id]} e {path}"
                )
            origin[rec.id] = path
            records.append(rec)
    return records
