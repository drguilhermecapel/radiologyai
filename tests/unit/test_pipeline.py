from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from medai.core.artifacts import ArtifactStore
from medai.core.errors import ExamTypeUnknown
from medai.core.types import (
    AbstentionReason,
    AnalysisResult,
    EvidenceLevel,
    ExamInputs,
    ExamType,
    Modality,
    OperatingPoint,
)
from medai.pipeline import analyze
from tests.conftest import manifest_dict
from tests.support import synth

CODE = {"scheme": "99MEDAI", "value": "x", "meaning": "x"}
BRIGHT = ("cxr.bright", "fake-bright", "bright")


def op(role: str, threshold: float, on: str = "sigmoid", direction: str = "ge") -> dict[str, Any]:
    return {
        "name": role,
        "role": role,
        "threshold": threshold,
        "on": on,
        "direction": direction,
        "source": "test",
    }


def exam(*levels: float, exam_type: ExamType = ExamType.CXR) -> ExamInputs:
    return ExamInputs(
        images=tuple(synth.projection(seed=i, level=lv) for i, lv in enumerate(levels)),
        exam_type=exam_type,
    )


def run(
    make_registry: Any,
    make_catalog: Any,
    store: ArtifactStore,
    manifests: list[dict[str, Any]],
    inputs: ExamInputs,
    levels: dict[Any, EvidenceLevel] | None = None,
    **kwargs: Any,
) -> AnalysisResult:
    registry = make_registry(*manifests)
    return analyze(
        inputs,
        registry=registry,
        catalog=make_catalog(levels or {}),
        store=store,
        model_ids=[m["id"] for m in manifests if not m.get("_dep_only")],
        **kwargs,
    )


def test_unknown_exam_type_is_refused(
    make_registry: Any, make_catalog: Any, store: ArtifactStore
) -> None:
    with pytest.raises(ExamTypeUnknown):
        run(
            make_registry,
            make_catalog,
            store,
            [manifest_dict("fake-bright", "Brightness")],
            exam(100, exam_type=ExamType.UNKNOWN),
        )


def test_score_is_computed_from_the_input(
    make_registry: Any, make_catalog: Any, store: ArtifactStore
) -> None:
    manifests = [manifest_dict("fake-bright", "Brightness")]
    dark = run(make_registry, make_catalog, store, manifests, exam(40))
    light = run(make_registry, make_catalog, store, manifests, exam(200))
    assert dark.status == light.status == "completed"
    assert dark.findings[0].raw_score is not None
    assert light.findings[0].raw_score is not None
    assert dark.findings[0].raw_score < light.findings[0].raw_score
    assert dark.findings[0].raw_score == pytest.approx(
        float(exam(40).images[0].pixels.mean() / 255)
    )


def test_without_operating_point_no_call_is_made(
    make_registry: Any, make_catalog: Any, store: ArtifactStore
) -> None:
    result = run(
        make_registry, make_catalog, store, [manifest_dict("fake-bright", "Brightness")], exam(250)
    )
    finding = result.findings[0]
    assert finding.decision == "score_only"
    assert finding.probability is None
    assert finding.calibration_id is None
    assert finding.op_normalized is None


def test_no_catalog_evidence_means_research_only(
    make_registry: Any, make_catalog: Any, store: ArtifactStore
) -> None:
    result = run(
        make_registry, make_catalog, store, [manifest_dict("fake-bright", "Brightness")], exam(250)
    )
    assert result.findings[0].evidence_level is EvidenceLevel.D
    assert result.main_findings() == []
    assert len(result.research_findings()) == 1


def test_catalog_level_controls_visibility_and_points_control_decision(
    make_registry: Any, make_catalog: Any, store: ArtifactStore
) -> None:
    manifest = manifest_dict(
        "fake-bright",
        "Brightness",
        operating_points=[
            {"label": "bright", "point": op("rule_out", 0.3)},
            {"label": "bright", "point": op("rule_in", 0.7)},
        ],
    )
    levels = {BRIGHT: EvidenceLevel.A}
    decisions = [
        run(make_registry, make_catalog, store, [manifest], exam(lv), levels).findings[0].decision
        for lv in (40, 128, 220)
    ]
    assert decisions == ["negative", "indeterminate", "positive"]
    result = run(make_registry, make_catalog, store, [manifest], exam(220), levels)
    assert result.main_findings()[0].evidence_level is EvidenceLevel.A
    assert result.provenance[0].model_id == "fake-bright"
    assert result.provenance[0].weights_sha256


class _Policy:
    def calibrate(
        self, model_id: str, label: str, raw_score: float, score_kind: str
    ) -> tuple[float, str] | None:
        return raw_score / 2, "cal-test"

    def operating_points(self, model_id: str, label: str) -> list[OperatingPoint]:
        return [OperatingPoint.model_validate(op("balanced", 0.25, on="calibrated"))]


def test_policy_calibration_and_fitted_points_take_precedence(
    make_registry: Any, make_catalog: Any, store: ArtifactStore
) -> None:
    manifest = manifest_dict(
        "fake-bright",
        "Brightness",
        operating_points=[{"label": "bright", "point": op("balanced", 0.99)}],
    )
    result = run(make_registry, make_catalog, store, [manifest], exam(200), policy=_Policy())
    finding = result.findings[0]
    assert finding.calibration_id == "cal-test"
    assert finding.probability == pytest.approx((finding.raw_score or 0) / 2)
    assert finding.decision == "positive"
    assert finding.operating_points[0].on == "calibrated"


def test_unavailable_model_is_recorded_not_replaced(
    make_registry: Any, make_catalog: Any, store: ArtifactStore
) -> None:
    registry = make_registry(manifest_dict("fake-bright", "Brightness"), with_weights=False)
    result = analyze(
        exam(100), registry=registry, catalog=make_catalog(), store=store, model_ids=["fake-bright"]
    )
    assert result.status == "failed"
    assert result.findings == []
    assert result.runs[0].status == "unavailable"
    assert "medai models pull" in (result.runs[0].error or "")


def test_failing_model_is_recorded_and_others_still_run(
    make_registry: Any, make_catalog: Any, store: ArtifactStore
) -> None:
    manifests = [
        manifest_dict("fake-boom", "Exploding"),
        manifest_dict("fake-bright", "Brightness"),
    ]
    result = run(make_registry, make_catalog, store, manifests, exam(100))
    assert result.status == "partial"
    assert {r.model_id: r.status for r in result.runs} == {
        "fake-boom": "failed",
        "fake-bright": "completed",
    }
    assert "model blew up" in (result.runs[0].error or "")
    assert [f.label for f in result.findings] == ["bright"]


def test_not_applicable_model_abstains_without_findings(
    make_registry: Any, make_catalog: Any, store: ArtifactStore
) -> None:
    result = run(
        make_registry,
        make_catalog,
        store,
        [manifest_dict("fake-ood", "NeverApplicable")],
        exam(100),
    )
    assert result.status == "abstained"
    assert result.findings == []
    assert result.runs[0].qa is not None
    assert result.runs[0].qa.reasons == [AbstentionReason.OOD]


def test_static_applicability_rejects_wrong_exam_type(
    make_registry: Any, make_catalog: Any, store: ArtifactStore
) -> None:
    result = run(
        make_registry,
        make_catalog,
        store,
        [manifest_dict("fake-bright", "Brightness")],
        exam(100, exam_type=ExamType.MG),
    )
    assert result.runs[0].status == "abstained"
    assert AbstentionReason.EXAM_TYPE in (result.runs[0].qa.reasons if result.runs[0].qa else [])


def test_undeclared_label_fails_the_run(
    make_registry: Any, make_catalog: Any, store: ArtifactStore
) -> None:
    result = run(
        make_registry,
        make_catalog,
        store,
        [manifest_dict("fake-undeclared", "UndeclaredLabel")],
        exam(100),
    )
    assert result.runs[0].status == "failed"
    assert "undeclared" in (result.runs[0].error or "")
    assert result.findings == []


def _derived_manifests(derived_class: str = "Derived") -> list[dict[str, Any]]:
    seg = manifest_dict(
        "fake-segmenter",
        "Segmenter",
        labels=[],
        task="segmentation",
        capabilities=["segment", "measure"],
    )
    label = {
        "key": "too_bright",
        "source_label": "n/a",
        "condition_id": "cxr.too_bright",
        "kind": "derived",
        "score_kind": "measurement",
        "measurement": "doubled_fraction",
        "code": CODE,
    }
    derived = manifest_dict(
        "fake-derived",
        derived_class,
        labels=[label],
        task="measurement",
        capabilities=["measure"],
        depends_on=["fake-segmenter"],
        operating_points=[{"label": "too_bright", "point": op("guideline", 0.5, on="measurement")}],
    )
    return [seg, derived]


def test_dependencies_segmentations_measurements_and_derived_findings(
    make_registry: Any, make_catalog: Any, store: ArtifactStore
) -> None:
    registry = make_registry(*_derived_manifests())
    levels = {("cxr.too_bright", "fake-derived", "too_bright"): EvidenceLevel.B}
    result = analyze(
        exam(100),
        registry=registry,
        catalog=make_catalog(levels),
        store=store,
        model_ids=["fake-derived"],
    )
    assert [r.model_id for r in result.runs] == ["fake-segmenter", "fake-derived"]
    seg = result.segmentations[0]
    assert seg.id == "fake-segmenter:bright_region"
    assert seg.label_keys == {1: "bright_region"}
    mask = store.read_labelmap(seg.mask)
    expected = exam(100).images[0].pixels
    assert np.array_equal(mask, (expected > expected.mean()).astype(mask.dtype))
    assert {m.name for m in result.measurements} == {"bright_fraction", "doubled_fraction"}
    assert result.measurements[0].derived_from == ["fake-segmenter:bright_region"]
    finding = result.findings[0]
    assert finding.kind == "derived"
    assert finding.measurement_ref == "doubled_fraction"
    assert finding.decision == "positive"
    assert finding.evidence_level is EvidenceLevel.B
    assert finding.raw_score == pytest.approx(2 * float(mask.mean()))


def test_abstained_measurement_gives_abstained_derived_finding(
    make_registry: Any, make_catalog: Any, store: ArtifactStore
) -> None:
    registry = make_registry(*_derived_manifests("DerivedAbstains"))
    result = analyze(
        exam(100),
        registry=registry,
        catalog=make_catalog(),
        store=store,
        model_ids=["fake-derived"],
    )
    finding = result.findings[0]
    assert finding.decision == "abstained"
    assert finding.raw_score is None
    assert finding.abstention_reason is AbstentionReason.ACQUISITION


def test_missing_upstream_blocks_dependents(
    make_registry: Any, make_catalog: Any, store: ArtifactStore
) -> None:
    seg, derived = _derived_manifests()
    seg["adapter"] = seg["adapter"].replace("Segmenter", "Exploding")
    registry = make_registry(seg, derived)
    result = analyze(
        exam(100),
        registry=registry,
        catalog=make_catalog(),
        store=store,
        model_ids=["fake-derived"],
    )
    statuses = {r.model_id: r.status for r in result.runs}
    assert statuses == {"fake-segmenter": "failed", "fake-derived": "abstained"}
    assert result.findings == []
    assert result.status == "failed"


def test_categorical_output(make_registry: Any, make_catalog: Any, store: ArtifactStore) -> None:
    label = {
        "key": "density",
        "source_label": "density",
        "condition_id": "cxr.density",
        "kind": "categorical",
        "score_kind": "softmax",
        "classes": ["low", "high"],
        "code": CODE,
    }
    result = run(
        make_registry,
        make_catalog,
        store,
        [manifest_dict("fake-cat", "Categorical", labels=[label])],
        exam(200),
    )
    finding = result.findings[0]
    assert finding.decision == "category"
    assert finding.category == "high"
    assert finding.class_scores is not None
    assert sum(finding.class_scores.values()) == pytest.approx(1.0)


def test_explanations_only_for_called_findings(
    make_registry: Any, make_catalog: Any, store: ArtifactStore
) -> None:
    manifest = manifest_dict(
        "fake-bright",
        "Brightness",
        capabilities=["classify", "explain"],
        operating_points=[{"label": "bright", "point": op("balanced", 0.5)}],
    )
    positive = run(make_registry, make_catalog, store, [manifest], exam(220))
    negative = run(make_registry, make_catalog, store, [manifest], exam(40))
    assert negative.explanations == []
    explanation = positive.explanations[0]
    assert positive.findings[0].explanation_ids == [explanation.id]
    assert store.read_bytes(explanation.map)[:8] == b"\x89PNG\r\n\x1a\n"


def test_single_input_model_uses_first_applicable_image(
    make_registry: Any, make_catalog: Any, store: ArtifactStore
) -> None:
    manifest = manifest_dict("fake-bright", "Brightness")
    manifest["applicability"]["modalities"] = ["DX"]
    images = (
        synth.projection(seed=1, level=30, modality=Modality.MG),
        synth.projection(seed=2, level=200),
    )
    result = run(
        make_registry,
        make_catalog,
        store,
        [manifest],
        ExamInputs(images=images, exam_type=ExamType.CXR),
    )
    assert result.findings[0].raw_score == pytest.approx(float(images[1].pixels.mean() / 255))


def test_multi_input_roles(make_registry: Any, make_catalog: Any, store: ArtifactStore) -> None:
    label = {"key": "asymmetry", "source_label": "asym", "condition_id": "mg.asym", "code": CODE}
    manifest = manifest_dict(
        "fake-multi",
        "MultiRole",
        labels=[label],
        inputs={"mode": "multi", "roles": ["L-CC", "R-CC"]},
    )
    manifest["applicability"]["exam_types"] = ["mg"]
    left = synth.projection(
        seed=1, level=60, modality=Modality.MG, laterality="L", view_position="CC"
    )
    right = synth.projection(
        seed=2, level=180, modality=Modality.MG, laterality="R", view_position="CC"
    )
    both = run(
        make_registry,
        make_catalog,
        store,
        [manifest],
        ExamInputs(images=(left, right), exam_type=ExamType.MG),
    )
    assert both.findings[0].raw_score == pytest.approx(
        abs(float(left.pixels.mean() - right.pixels.mean())) / 255
    )
    one = run(
        make_registry,
        make_catalog,
        store,
        [manifest],
        ExamInputs(images=(left,), exam_type=ExamType.MG),
    )
    assert one.runs[0].status == "abstained"
    assert one.runs[0].qa is not None
    assert AbstentionReason.MISSING_INPUT in one.runs[0].qa.reasons


def test_no_routed_model_abstains_with_warning(
    make_registry: Any, make_catalog: Any, store: ArtifactStore
) -> None:
    registry = make_registry(manifest_dict("fake-bright", "Brightness"))
    result = analyze(
        exam(100), registry=registry, catalog=make_catalog(), store=store, model_ids=[]
    )
    assert result.status == "abstained"
    assert "no model is routed" in result.warnings[0]


def test_result_round_trips_through_json(
    make_registry: Any, make_catalog: Any, store: ArtifactStore
) -> None:
    registry = make_registry(*_derived_manifests())
    result = analyze(
        exam(100),
        registry=registry,
        catalog=make_catalog(),
        store=store,
        model_ids=["fake-derived"],
    )
    again = AnalysisResult.model_validate_json(result.model_dump_json())
    assert again == result
    assert again.intended_use == "research_use_only"
