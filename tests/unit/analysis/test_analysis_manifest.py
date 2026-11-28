"""Esquema do manifesto: sem métricas, códigos oficiais, licença, regras de status."""

from __future__ import annotations

from typing import Any

import pytest
from analysis_support import manifest_dict

from radiologyai.analysis.manifest import ModelManifest, parse_manifest, validate_code
from radiologyai.catalog.schema import CodedConcept
from radiologyai.errors import ManifestError


def _parse(**overrides: Any) -> ModelManifest:
    return parse_manifest(manifest_dict("fake-x", "Brightness", **overrides))


def test_minimal_manifest_parses() -> None:
    manifest = _parse()
    assert manifest.effective_route == "main"
    assert manifest.is_pinned
    assert manifest.points_for("bright") == []


@pytest.mark.requirement("REQ-105")
@pytest.mark.parametrize("key", ["accuracy", "auc", "sensitivity", "specificity"])
def test_performance_keys_are_rejected_at_top_level(key: str) -> None:
    data = manifest_dict("fake-x", "Brightness")
    data[key] = 0.9
    with pytest.raises(ManifestError, match="desempenho"):
        parse_manifest(data)


@pytest.mark.requirement("REQ-105")
def test_performance_keys_are_rejected_inside_config() -> None:
    with pytest.raises(ManifestError, match="desempenho"):
        _parse(config={"eval": {"nih": {"AUC": 0.81}}})


@pytest.mark.requirement("REQ-105")
def test_official_code_must_exist_with_official_meaning() -> None:
    ok = CodedConcept(scheme="SCT", value="36118008", meaning="Pneumothorax")
    assert validate_code(ok) is ok
    with pytest.raises(ValueError, match="significado"):
        validate_code(CodedConcept(scheme="SCT", value="36118008", meaning="Pneumonia"))
    with pytest.raises(ValueError, match="não existe"):
        validate_code(CodedConcept(scheme="SCT", value="999999999999", meaning="x"))
    with pytest.raises(ValueError, match="esquema"):
        validate_code(CodedConcept(scheme="ICD99", value="x", meaning="x"))


def test_label_code_is_validated_in_manifest() -> None:
    labels = [
        {
            "key": "ptx",
            "source_label": "Pneumothorax",
            "condition_id": "cxr.pneumothorax",
            "code": {"scheme": "SCT", "value": "36118008", "meaning": "Pneumonia"},
        }
    ]
    with pytest.raises(ManifestError, match="significado"):
        _parse(labels=labels)


@pytest.mark.requirement("REQ-105")
@pytest.mark.parametrize("licence", ["GPL-3.0", "AGPL-3.0-only", "gpl v2"])
def test_copyleft_code_is_excluded(licence: str) -> None:
    with pytest.raises(ManifestError, match="GPL"):
        _parse(license={"code": licence, "weights": "x", "commercial_use": "no"})


def test_lgpl_is_not_confused_with_gpl() -> None:
    assert _parse(license={"code": "LGPL-2.1", "weights": "x", "commercial_use": "yes"})


def test_planned_needs_reason_and_may_have_unpinned_weights() -> None:
    weights = [{"key": "w", "url": "https://x.invalid/w", "filename": "w.bin"}]
    with pytest.raises(ManifestError, match="planned_reason"):
        _parse(status="planned", weights=weights)
    planned = _parse(status="planned", weights=weights, planned_reason="sem licença")
    assert not planned.is_pinned


@pytest.mark.parametrize(
    "weight",
    [
        {"key": "w", "url": "https://x.invalid/w", "filename": "../w.bin", "sha256": "a" * 64},
        {"key": "w", "url": "http://x.invalid/w", "filename": "w.bin", "sha256": "a" * 64},
        {"key": "w", "url": "https://x.invalid/w", "filename": "w.bin", "sha256": "ABC"},
    ],
)
def test_weight_entries_are_strict(weight: dict[str, Any]) -> None:
    with pytest.raises(ManifestError):
        _parse(weights=[weight])


def test_zero_shot_models_are_research_only() -> None:
    with pytest.raises(ManifestError, match="zero-shot"):
        _parse(task="zero_shot")
    zs = _parse(task="zero_shot", visibility_default="research")
    assert zs.effective_route == "research"


def test_operating_point_must_reference_declared_label_and_have_source() -> None:
    point = {
        "name": "p",
        "role": "balanced",
        "threshold": 0.5,
        "on": "sigmoid",
        "source": "biblioteca",
    }
    with pytest.raises(ManifestError, match="desconhecido"):
        _parse(operating_points=[{"label": "nope", "point": point}])
    with pytest.raises(ManifestError, match="origem"):
        _parse(operating_points=[{"label": "bright", "point": {**point, "source": " "}}])


def test_derived_and_categorical_label_consistency() -> None:
    code = {"scheme": "99MEDAI", "value": "v", "meaning": "m"}
    with pytest.raises(ManifestError, match="medida"):
        _parse(labels=[{"key": "k", "source_label": "s", "condition_id": "c", "kind": "derived",
                        "score_kind": "measurement", "code": code}])  # fmt: skip
    with pytest.raises(ManifestError, match="classes"):
        _parse(labels=[{"key": "k", "source_label": "s", "condition_id": "c",
                        "kind": "categorical", "code": code}])  # fmt: skip


def test_multi_input_needs_roles() -> None:
    with pytest.raises(ManifestError, match="papel"):
        _parse(inputs={"mode": "multi"})
    with pytest.raises(ManifestError, match="multi"):
        _parse(inputs={"mode": "single", "roles": ["L-CC"]})


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ManifestError):
        _parse(weights_sha256="0" * 64)
