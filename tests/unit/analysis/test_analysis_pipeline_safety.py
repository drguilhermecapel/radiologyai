"""Pipeline: saída de modelo que falha some; nível efetivo é o mais restritivo; A/C sem pontos."""

# ruff: noqa: ARG001 - fixtures usadas pelo efeito colateral; dublês com assinatura fixa

from __future__ import annotations

from typing import Any

import pytest
from analysis_support import manifest_dict, op, projection

from radiologyai.analysis.artifacts import ArtifactStore
from radiologyai.analysis.pipeline import _Collector, analyze
from radiologyai.analysis.types import AnalysisResult, ExamInputs
from radiologyai.calibration.decision import OperatingPoint
from radiologyai.catalog.schema import EvidenceLevel, ExamType

BRIGHT = ("cxr.bright", "fake-bright", "bright")


def exam(level: float) -> ExamInputs:
    return ExamInputs(images=(projection(level=level),), exam_type=ExamType.CXR)


def run(
    make_registry: Any,
    make_catalog: Any,
    store: ArtifactStore,
    manifest: dict[str, Any],
    levels: dict[Any, EvidenceLevel],
    level: float = 220,
    **kwargs: Any,
) -> AnalysisResult:
    return analyze(
        exam(level),
        registry=make_registry(manifest),
        catalog=make_catalog(levels),
        store=store,
        model_ids=[manifest["id"]],
        **kwargs,
    )


@pytest.mark.requirement("REQ-167")
def test_invalid_explanation_is_a_warning_and_model_stays_completed(
    make_registry: Any, make_catalog: Any, store: ArtifactStore
) -> None:
    manifest = manifest_dict(
        "fake-bright",
        "PointPlacementExplanation",
        capabilities=["classify", "explain"],
        operating_points=[{"label": "bright", "point": op("balanced", 0.5)}],
    )
    result = run(make_registry, make_catalog, store, manifest, {BRIGHT: EvidenceLevel.B})
    assert [r.status for r in result.runs] == ["completed"]
    assert result.explanations == ()
    assert result.findings[0].explanation_ids == ()
    assert any("exige bbox" in w for w in result.warnings)


@pytest.mark.requirement("REQ-167")
def test_failure_after_publication_rolls_back_everything_the_model_produced(
    make_registry: Any,
    make_catalog: Any,
    store: ArtifactStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = manifest_dict(
        "fake-bright",
        "Brightness",
        capabilities=["classify", "explain"],
        operating_points=[{"label": "bright", "point": op("balanced", 0.5)}],
    )

    def boom(self: Any, *args: Any, **kwargs: Any) -> None:
        raise OSError("disco cheio")

    monkeypatch.setattr(_Collector, "add_explanations", boom)
    result = run(make_registry, make_catalog, store, manifest, {BRIGHT: EvidenceLevel.B})
    (run_,) = result.runs
    assert run_.status == "failed"
    assert "disco cheio" in (run_.error or "")
    # nenhum achado, máscara, medida ou proveniência de um modelo que falhou
    assert result.findings == ()
    assert result.provenance == ()
    assert result.segmentations == ()
    assert result.status == "failed"


@pytest.mark.requirement("REQ-168")
def test_manifest_research_visibility_wins_over_catalog_level(
    make_registry: Any, make_catalog: Any, store: ArtifactStore
) -> None:
    manifest = manifest_dict(
        "fake-bright",
        "Brightness",
        visibility_default="research",
        operating_points=[{"label": "bright", "point": op("balanced", 0.5)}],
    )
    result = run(make_registry, make_catalog, store, manifest, {BRIGHT: EvidenceLevel.B})
    (finding,) = result.findings
    assert finding.evidence_level is EvidenceLevel.D
    assert finding.visibility == "research"
    assert result.main_findings() == []
    assert any("visibility_default 'research'" in w for w in result.warnings)


@pytest.mark.requirement("REQ-168")
@pytest.mark.parametrize("level", [EvidenceLevel.A, EvidenceLevel.C])
def test_measured_level_without_policy_points_is_score_only(
    make_registry: Any, make_catalog: Any, store: ArtifactStore, level: EvidenceLevel
) -> None:
    manifest = manifest_dict(
        "fake-bright",
        "Brightness",
        operating_points=[{"label": "bright", "point": op("rule_in", 0.3)}],
    )
    result = run(make_registry, make_catalog, store, manifest, {BRIGHT: level})
    (finding,) = result.findings
    assert finding.evidence_level is level
    assert finding.decision == "score_only"
    assert finding.operating_points == ()
    assert any("não foram avaliados" in w for w in result.warnings)


class _FittedPolicy:
    def calibrate(
        self, model_id: str, label: str, raw_score: float, score_kind: str
    ) -> tuple[float, str] | None:
        del model_id, label, raw_score, score_kind
        return None

    def operating_points(self, model_id: str, label: str) -> list[OperatingPoint]:
        del model_id, label
        return [OperatingPoint.model_validate(op("rule_in", 0.3))]


@pytest.mark.requirement("REQ-168")
def test_measured_level_decides_with_protocol_points_from_policy(
    make_registry: Any, make_catalog: Any, store: ArtifactStore
) -> None:
    manifest = manifest_dict(
        "fake-bright",
        "Brightness",
        operating_points=[{"label": "bright", "point": op("rule_in", 0.99)}],
    )
    result = run(
        make_registry,
        make_catalog,
        store,
        manifest,
        {BRIGHT: EvidenceLevel.C},
        policy=_FittedPolicy(),
    )
    (finding,) = result.findings
    assert finding.decision == "positive"
    assert finding.operating_points[0].threshold == pytest.approx(0.3)


@pytest.mark.requirement("REQ-168")
def test_published_level_still_uses_manifest_points_with_origin(
    make_registry: Any, make_catalog: Any, store: ArtifactStore
) -> None:
    manifest = manifest_dict(
        "fake-bright",
        "Brightness",
        operating_points=[{"label": "bright", "point": op("rule_in", 0.3)}],
    )
    result = run(make_registry, make_catalog, store, manifest, {BRIGHT: EvidenceLevel.B})
    assert result.findings[0].decision == "positive"
