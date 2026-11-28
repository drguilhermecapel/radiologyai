"""Regra de nível de evidência: calculada a partir dos registros, nunca digitada."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from radiologyai.catalog.schema import (
    EvidenceLevel,
    EvidenceRecord,
    LevelEntry,
    LevelsFile,
    compute_level,
    visibility_for,
)
from radiologyai.errors import EvidenceError, RadiologyAIError

PROVENANCE = {
    "git_sha": "abc",
    "protocol_id": "2026-11-v1",
    "dataset_id": "ds",
    "dataset_sha256": "0" * 64,
    "weights_sha256": {"main": "1" * 64},
    "command": "radiologyai eval run",
    "created_at": datetime(2026, 11, 24, tzinfo=UTC),
}
SOURCE = {
    "title": "paper",
    "identifier": "10.1/x",
    "url": "https://example.org",
    "quote": "citação literal do artigo",
    "peer_reviewed": True,
}


def record(rid: str, kind: str, **overrides: Any) -> EvidenceRecord:
    base: dict[str, Any] = {
        "id": rid,
        "condition_id": "cxr.x",
        "model_id": "m",
        "label": "x",
        "kind": kind,
    }
    if kind.startswith("measured"):
        base.update(provenance=PROVENANCE, n_total=200, n_positive=50, passes_protocol=True)
    else:
        base.update(source={**SOURCE, "verified_at_source": True}, passes_protocol=True)
    base.update(overrides)
    return EvidenceRecord.model_validate(base)


def failed_external(rid: str = "r1") -> EvidenceRecord:
    return record(
        rid, "measured_external", passes_protocol=False, failure_reasons=["auroc abaixo da barra"]
    )


def entry(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "condition_id": "cxr.x",
        "model_id": "m",
        "label": "x",
        "level": "D",
        "visibility": "research",
        "basis": [],
    }
    base.update(overrides)
    return base


# --- a regra ------------------------------------------------------------------


@pytest.mark.requirement("REQ-090")
def test_no_evidence_is_level_d() -> None:
    assert compute_level([]) == (EvidenceLevel.D, [], False)


@pytest.mark.requirement("REQ-090")
def test_clean_passing_external_measurement_is_level_a() -> None:
    level, basis, refuted = compute_level([record("r1", "measured_external")])
    assert (level, basis, refuted) == (EvidenceLevel.A, ["r1"], False)


@pytest.mark.requirement("REQ-090")
@pytest.mark.parametrize("leakage", ["possible", "known"])
def test_contaminated_external_measurement_is_only_level_c(leakage: str) -> None:
    level, basis, refuted = compute_level([record("r1", "measured_external", leakage=leakage)])
    assert (level, basis, refuted) == (EvidenceLevel.C, ["r1"], False)


@pytest.mark.requirement("REQ-090")
def test_internal_measurement_is_level_c() -> None:
    assert compute_level([record("r1", "measured_internal")])[0] is EvidenceLevel.C


@pytest.mark.requirement("REQ-090")
def test_verified_published_evidence_is_level_b() -> None:
    assert compute_level([record("r1", "published")])[0] is EvidenceLevel.B
    assert compute_level([record("r1", "guideline_threshold")])[0] is EvidenceLevel.B


@pytest.mark.requirement("REQ-090")
def test_unverified_published_evidence_does_not_count() -> None:
    unverified = record("r1", "published", source=SOURCE)
    assert compute_level([unverified])[0] is EvidenceLevel.D


@pytest.mark.requirement("REQ-090")
def test_failing_or_undecided_records_do_not_count() -> None:
    failing_published = record(
        "r1", "published", passes_protocol=False, failure_reasons=["amostra pequena"]
    )
    failing_internal = record(
        "r2", "measured_internal", passes_protocol=False, failure_reasons=["auroc abaixo da barra"]
    )
    undecided = record("r3", "measured_external", passes_protocol=None)
    assert compute_level([failing_published, failing_internal, undecided]) == (
        EvidenceLevel.D,
        [],
        False,
    )


@pytest.mark.requirement("REQ-093")
def test_failed_clean_external_measurement_refutes_published_claims() -> None:
    level, basis, refuted = compute_level([failed_external(), record("r2", "published")])
    assert level is EvidenceLevel.D
    assert refuted is True
    assert basis == ["r1"]


@pytest.mark.requirement("REQ-093")
def test_failed_clean_external_measurement_refutes_internal_measurement() -> None:
    level, _, refuted = compute_level([failed_external(), record("r2", "measured_internal")])
    assert (level, refuted) == (EvidenceLevel.D, True)


@pytest.mark.requirement("REQ-093")
def test_undecided_clean_external_measurement_does_not_refute() -> None:
    undecided = record("r1", "measured_external", passes_protocol=None)
    level, _, refuted = compute_level([undecided, record("r2", "published")])
    assert (level, refuted) == (EvidenceLevel.B, False)


@pytest.mark.requirement("REQ-093")
def test_failed_contaminated_external_measurement_does_not_refute() -> None:
    contaminated = record(
        "r1",
        "measured_external",
        leakage="known",
        passes_protocol=False,
        failure_reasons=["auroc abaixo da barra"],
    )
    level, _, refuted = compute_level([contaminated, record("r2", "published")])
    assert (level, refuted) == (EvidenceLevel.B, False)


@pytest.mark.requirement("REQ-090")
def test_one_passing_clean_external_measurement_is_enough() -> None:
    level, basis, refuted = compute_level([failed_external(), record("r2", "measured_external")])
    assert (level, basis, refuted) == (EvidenceLevel.A, ["r2"], False)


@pytest.mark.requirement("REQ-090")
def test_precedence_a_over_b_over_c() -> None:
    records = [
        record("c", "measured_internal"),
        record("b", "published"),
        record("a", "measured_external"),
    ]
    assert compute_level(records)[0] is EvidenceLevel.A
    assert compute_level(records[:2])[0] is EvidenceLevel.B
    assert compute_level(records[:1])[0] is EvidenceLevel.C


@pytest.mark.requirement("REQ-090")
def test_records_of_different_triples_are_rejected() -> None:
    """Evidência de uma condição não pode promover outra."""
    other = record("r2", "measured_external", condition_id="cxr.y")
    with pytest.raises(EvidenceError, match="único trio"):
        compute_level([record("r1", "published"), other])
    assert issubclass(EvidenceError, RadiologyAIError)


# --- não há onde digitar um nível ------------------------------------------------


@pytest.mark.requirement("REQ-090")
def test_evidence_record_has_no_level_field() -> None:
    assert "level" not in EvidenceRecord.model_fields
    with pytest.raises(ValidationError):
        record("r", "published", level="A")


# --- forma dos registros --------------------------------------------------------


def test_measured_record_requires_provenance_and_counts() -> None:
    with pytest.raises(ValidationError):
        record("r", "measured_external", provenance=None)
    with pytest.raises(ValidationError):
        record("r", "measured_external", n_positive=None)


def test_published_record_requires_source_and_failures_need_reasons() -> None:
    with pytest.raises(ValidationError):
        record("r", "published", source=None)
    with pytest.raises(ValidationError):
        record("r", "measured_external", passes_protocol=False)


@pytest.mark.parametrize(
    "provenance",
    [
        {**PROVENANCE, "dataset_sha256": "PLACEHOLDER"},
        {**PROVENANCE, "weights_sha256": {"main": "g" * 64}},
        {**PROVENANCE, "weights_sha256": {}},
    ],
)
def test_provenance_rejects_placeholder_hashes(provenance: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        record("r", "measured_external", provenance=provenance)


def test_records_are_immutable() -> None:
    rec = record("r", "published")
    with pytest.raises(ValidationError):
        rec.passes_protocol = False  # type: ignore[misc]


# --- visibilidade ----------------------------------------------------------------


@pytest.mark.requirement("REQ-091")
def test_visibility_follows_level() -> None:
    assert visibility_for(EvidenceLevel.D) == "research"
    assert all(
        visibility_for(lv) == "main" for lv in (EvidenceLevel.A, EvidenceLevel.B, EvidenceLevel.C)
    )


@pytest.mark.requirement("REQ-091")
def test_level_entry_rejects_level_d_in_main_view() -> None:
    assert LevelEntry.model_validate(entry()).visibility == "research"
    with pytest.raises(ValidationError, match="visão principal"):
        LevelEntry.model_validate(entry(visibility="main"))


def test_level_entry_rejects_incoherent_combinations() -> None:
    with pytest.raises(ValidationError):
        LevelEntry.model_validate(entry(level="A", visibility="research", basis=["r1"]))
    with pytest.raises(ValidationError, match="refutada"):
        LevelEntry.model_validate(
            entry(level="B", visibility="main", basis=["r1"], refuted_by_measurement=True)
        )


def test_levels_file_requires_real_digest() -> None:
    ok = LevelsFile.model_validate(
        {"protocol_id": "2026-11-v1", "evidence_digest": "a" * 64, "entries": [entry()]}
    )
    assert len(ok.entries) == 1
    with pytest.raises(ValidationError):
        LevelsFile.model_validate(
            {"protocol_id": "2026-11-v1", "evidence_digest": "TODO", "entries": []}
        )
