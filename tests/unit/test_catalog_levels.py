from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from medai.catalog.catalog import Catalog, load_protocol
from medai.catalog.schema import EvidenceRecord, compute_level, visibility_for
from medai.core.types import EvidenceLevel

PROVENANCE = {
    "git_sha": "abc",
    "protocol_id": "2026-11-v1",
    "dataset_id": "ds",
    "dataset_sha256": "0" * 64,
    "weights_sha256": {"main": "1" * 64},
    "command": "medai eval run",
    "created_at": datetime(2026, 11, 24, tzinfo=UTC),
}
SOURCE = {
    "title": "paper",
    "identifier": "10.1/x",
    "url": "https://example.org",
    "quote": "AUC 0.91",
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


def test_no_evidence_is_level_d() -> None:
    assert compute_level([]) == (EvidenceLevel.D, [], False)


def test_clean_passing_external_measurement_is_level_a() -> None:
    level, basis, refuted = compute_level([record("r1", "measured_external")])
    assert (level, basis, refuted) == (EvidenceLevel.A, ["r1"], False)


def test_contaminated_external_measurement_is_only_level_c() -> None:
    level, _, _ = compute_level([record("r1", "measured_external", leakage="known")])
    assert level is EvidenceLevel.C


def test_internal_measurement_is_level_c() -> None:
    assert compute_level([record("r1", "measured_internal")])[0] is EvidenceLevel.C


def test_verified_published_evidence_is_level_b() -> None:
    assert compute_level([record("r1", "published")])[0] is EvidenceLevel.B
    assert compute_level([record("r1", "guideline_threshold")])[0] is EvidenceLevel.B


def test_unverified_published_evidence_does_not_count() -> None:
    unverified = record("r1", "published", source=SOURCE)
    assert compute_level([unverified])[0] is EvidenceLevel.D


def test_failed_clean_external_measurement_refutes_published_claims() -> None:
    failed = record(
        "r1", "measured_external", passes_protocol=False, failure_reasons=["auroc below bar"]
    )
    level, basis, refuted = compute_level([failed, record("r2", "published")])
    assert level is EvidenceLevel.D
    assert refuted is True
    assert basis == ["r1"]


def test_one_passing_clean_external_measurement_is_enough() -> None:
    failed = record(
        "r1", "measured_external", passes_protocol=False, failure_reasons=["auroc below bar"]
    )
    assert compute_level([failed, record("r2", "measured_external")])[0] is EvidenceLevel.A


def test_precedence_a_over_b_over_c() -> None:
    records = [
        record("c", "measured_internal"),
        record("b", "published"),
        record("a", "measured_external"),
    ]
    assert compute_level(records)[0] is EvidenceLevel.A
    assert compute_level(records[:2])[0] is EvidenceLevel.B


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


def test_visibility_follows_level() -> None:
    assert visibility_for(EvidenceLevel.D) == "research"
    assert all(
        visibility_for(lv) == "main" for lv in (EvidenceLevel.A, EvidenceLevel.B, EvidenceLevel.C)
    )


def test_packaged_protocol_and_catalog_load() -> None:
    protocol = load_protocol()
    assert protocol.protocol_id == "2026-11-v1"
    assert 0.5 < protocol.auroc_ci_low_min <= protocol.auroc_min < 1.0
    catalog = Catalog.load()
    assert catalog.level("does.not.exist", "m", "x") is EvidenceLevel.D
