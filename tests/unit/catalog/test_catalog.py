"""Leitura do catálogo, do protocolo pré-registrado e dos níveis gerados."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import radiologyai.catalog
from radiologyai.catalog import Catalog, Condition, EvidenceLevel, LevelsFile, load_protocol
from radiologyai.catalog.catalog import CONDITIONS_FILE, LEVELS_FILE, PACKAGE_DIR, PROTOCOL_FILE
from radiologyai.errors import CatalogError, RadiologyAIError

CONDITION: dict[str, Any] = {
    "id": "cxr.pneumothorax",
    "name_pt": "Pneumotórax",
    "name_en": "Pneumothorax",
    "group": "cxr",
    "exam_types": ["cxr"],
    "system": "torax",
    "code": {"scheme": "99RADAI", "value": "pneumothorax", "meaning": "Pneumothorax"},
    "sources": [{"model_id": "m", "label": "Pneumothorax"}],
}
TRIPLE = ("cxr.pneumothorax", "m", "Pneumothorax")


def level_entry(level: str = "A", **overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "condition_id": TRIPLE[0],
        "model_id": TRIPLE[1],
        "label": TRIPLE[2],
        "level": level,
        "visibility": "research" if level == "D" else "main",
        "basis": [] if level == "D" else ["r1"],
    }
    base.update(overrides)
    return base


def write_catalog(
    directory: Path, conditions: list[dict[str, Any]], entries: list[dict[str, Any]] | None = None
) -> Path:
    # JSON é YAML válido: evita depender de um serializador YAML no teste.
    (directory / CONDITIONS_FILE).write_text(
        json.dumps({"conditions": conditions}), encoding="utf-8"
    )
    if entries is not None:
        levels = {"protocol_id": "2026-11-v1", "evidence_digest": "a" * 64, "entries": entries}
        (directory / LEVELS_FILE).write_text(json.dumps(levels), encoding="utf-8")
    return directory


def levels_of(*entries: dict[str, Any]) -> LevelsFile:
    return LevelsFile.model_validate(
        {"protocol_id": "2026-11-v1", "evidence_digest": "a" * 64, "entries": list(entries)}
    )


# --- arquivos empacotados -------------------------------------------------------


def test_packaged_protocol_loads() -> None:
    protocol = load_protocol()
    assert protocol.protocol_id == "2026-11-v1"
    assert protocol.frozen_on == "2026-11-24"
    assert 0.5 < protocol.auroc_ci_low_min <= protocol.auroc_min < 1.0
    assert protocol.subgroups == ("sex", "age_band", "view")


def test_packaged_protocol_states_every_threshold_explicitly() -> None:
    """Pré-registro: nenhum limiar do protocolo empacotado depende de default do código."""
    import yaml

    raw = yaml.safe_load((PACKAGE_DIR / PROTOCOL_FILE).read_text(encoding="utf-8"))
    assert set(raw) == set(type(load_protocol()).model_fields)


@pytest.mark.requirement("REQ-091")
def test_packaged_catalog_loads_and_unknown_triple_is_level_d() -> None:
    catalog = Catalog.load()
    ids = [c.id for c in catalog.conditions()]
    assert ids, "o catálogo empacotado não pode estar vazio"
    assert len(ids) == len(set(ids))
    assert not catalog.has("does.not.exist")
    assert catalog.entry("does.not.exist", "m", "x") is None
    assert catalog.level("does.not.exist", "m", "x") is EvidenceLevel.D
    assert catalog.visibility("does.not.exist", "m", "x") == "research"


def test_packaged_levels_are_generated_not_typed() -> None:
    """levels.json é saída de scripts/evidence_build_levels.py; à mão, o --check falha."""
    repo = Path(__file__).resolve().parents[3]
    if (PACKAGE_DIR / LEVELS_FILE).exists():
        result = subprocess.run(
            [sys.executable, str(repo / "scripts" / "evidence_build_levels.py"), "--check"],
            capture_output=True,
            text=True,
            check=False,
            cwd=repo,
        )
        assert result.returncode == 0, result.stdout + result.stderr
    catalog = Catalog.load()
    known = {c.id for c in catalog.conditions()}
    assert all(e.condition_id in known for e in catalog.entries())


def test_import_is_light() -> None:
    """`import radiologyai.catalog` não traz yaml, numpy nem torch."""
    code = (
        "import sys, radiologyai, radiologyai.catalog; "
        "print(','.join(sorted({'torch','numpy','yaml'} & set(sys.modules))))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "", f"import pesado no catálogo: {out.stdout.strip()}"
    assert Path(radiologyai.catalog.__file__).parent == PACKAGE_DIR


# --- catálogo em diretório ------------------------------------------------------


def test_catalog_with_generated_levels(tmp_path: Path) -> None:
    catalog = Catalog.load(write_catalog(tmp_path, [CONDITION], [level_entry("A")]))
    assert catalog.has(TRIPLE[0])
    assert catalog.condition(TRIPLE[0]).name_pt == "Pneumotórax"
    assert [c.id for c in catalog.conditions()] == [TRIPLE[0]]
    assert catalog.level(*TRIPLE) is EvidenceLevel.A
    assert catalog.visibility(*TRIPLE) == "main"
    assert [e.level for e in catalog.entries()] == [EvidenceLevel.A]
    assert catalog.levels_file is not None


@pytest.mark.requirement("REQ-091")
def test_condition_without_generated_level_is_research_only(tmp_path: Path) -> None:
    catalog = Catalog.load(write_catalog(tmp_path, [CONDITION]))
    assert catalog.level(*TRIPLE) is EvidenceLevel.D
    assert catalog.visibility(*TRIPLE) == "research"
    # outro modelo para a mesma condição não herda nível
    with_levels = Catalog.load(write_catalog(tmp_path, [CONDITION], [level_entry("B")]))
    assert with_levels.visibility(TRIPLE[0], "outro-modelo", TRIPLE[2]) == "research"


@pytest.mark.requirement("REQ-091")
def test_hand_edited_levels_cannot_promote_level_d(tmp_path: Path) -> None:
    write_catalog(tmp_path, [CONDITION], [level_entry("D", visibility="main")])
    with pytest.raises(CatalogError, match="visão principal"):
        Catalog.load(tmp_path)


@pytest.mark.requirement("REQ-090")
def test_condition_cannot_declare_a_level(tmp_path: Path) -> None:
    assert "level" not in Condition.model_fields
    write_catalog(tmp_path, [{**CONDITION, "level": "A"}])
    with pytest.raises(CatalogError, match="condição inválida"):
        Catalog.load(tmp_path)


def test_unknown_condition_raises() -> None:
    with pytest.raises(CatalogError, match="desconhecida"):
        Catalog([], None).condition("cxr.nada")
    assert issubclass(CatalogError, RadiologyAIError)


def test_duplicate_condition_id_is_rejected() -> None:
    condition = Condition.model_validate(CONDITION)
    with pytest.raises(CatalogError, match="duplicado"):
        Catalog([condition, condition], None)


def test_levels_for_unknown_condition_are_rejected() -> None:
    with pytest.raises(CatalogError, match="condição desconhecida"):
        Catalog([], levels_of(level_entry("A")))


def test_duplicate_level_entries_are_rejected() -> None:
    condition = Condition.model_validate(CONDITION)
    with pytest.raises(CatalogError, match="nível duplicado"):
        Catalog([condition], levels_of(level_entry("A"), level_entry("D")))


# --- falha alto -----------------------------------------------------------------


def test_missing_files_fail_loudly(tmp_path: Path) -> None:
    with pytest.raises(CatalogError, match="não encontrado"):
        Catalog.load(tmp_path)
    with pytest.raises(CatalogError, match="não encontrado"):
        load_protocol(tmp_path)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("conditions: [", "YAML inválido"),
        ("- a\n- b\n", "mapeamento"),
        ("", "mapeamento"),
        ("conditions: {}\n", "deve ser uma lista"),
        ("other: []\n", "deve ser uma lista"),
    ],
)
def test_malformed_conditions_fail_loudly(tmp_path: Path, content: str, message: str) -> None:
    (tmp_path / CONDITIONS_FILE).write_text(content, encoding="utf-8")
    with pytest.raises(CatalogError, match=message):
        Catalog.load(tmp_path)


@pytest.mark.parametrize("content", ["{not json", '{"protocol_id": "x"}'])
def test_malformed_levels_fail_loudly(tmp_path: Path, content: str) -> None:
    write_catalog(tmp_path, [CONDITION])
    (tmp_path / LEVELS_FILE).write_text(content, encoding="utf-8")
    with pytest.raises(CatalogError, match="níveis inválidos"):
        Catalog.load(tmp_path)


def test_protocol_with_unknown_key_is_rejected(tmp_path: Path) -> None:
    (tmp_path / PROTOCOL_FILE).write_text(
        'protocol_id: "x"\nfrozen_on: "2026-11-24"\nauroc_minimum: 0.5\n', encoding="utf-8"
    )
    with pytest.raises(CatalogError, match="protocolo inválido"):
        load_protocol(tmp_path)
