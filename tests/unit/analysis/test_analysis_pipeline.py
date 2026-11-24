"""Pipeline: decisão, visibilidade, dependências, abstenção e falhas registradas."""

from __future__ import annotations

from typing import Any

import pytest
from analysis_support import manifest_dict, op, projection

from radiologyai.analysis.artifacts import ArtifactStore
from radiologyai.analysis.pipeline import analyze
from radiologyai.analysis.types import (
    AbstentionReason,
    AnalysisResult,
    ExamInputs,
    Modality,
)
from radiologyai.calibration.decision import OperatingPoint
from radiologyai.catalog.schema import EvidenceLevel, ExamType
from radiologyai.errors import ExamTypeUnknownError

BRIGHT = ("cxr.bright", "fake-bright", "bright")


def exam(*levels: float, exam_type: ExamType = ExamType.CXR) -> ExamInputs:
    return ExamInputs(images=tuple(projection(level=lv) for lv in levels), exam_type=exam_type)


def run(
    make_registry: Any,
    make_catalog: Any,
    store: ArtifactStore,
    manifests: list[dict[str, Any]],
    inputs: ExamInputs,
    levels: dict[Any, EvidenceLevel] | None = None,
    model_ids: list[str] | None = None,
    **kwargs: Any,
) -> AnalysisResult:
    registry = make_registry(*manifests)
    return analyze(
        inputs,
        registry=registry,
        catalog=make_catalog(levels or {}),
        store=store,
        model_ids=model_ids if model_ids is not None else [m["id"] for m in manifests],
        **kwargs,
    )


@pytest.mark.requirement("REQ-104")
def test_unknown_exam_type_is_refused(
    make_registry: Any, make_catalog: Any, store: ArtifactStore
) -> None:
    with pytest.raises(ExamTypeUnknownError):
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
    d, li = dark.findings[0].raw_score, light.findings[0].raw_score
    assert d is not None
    assert li is not None
    assert d < li
    assert d == pytest.approx(float(exam(40).images[0].pixels.mean() / 255))


@pytest.mark.requirement("REQ-101")
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
    assert finding.operating_points == ()


@pytest.mark.requirement("REQ-102")
def test_no_catalog_evidence_means_research_only(
    make_registry: Any, make_catalog: Any, store: ArtifactStore
) -> None:
    result = run(
        make_registry, make_catalog, store, [manifest_dict("fake-bright", "Brightness")], exam(250)
    )
    assert result.findings[0].evidence_level is EvidenceLevel.D
    assert result.main_findings() == []
    assert len(result.research_findings()) == 1


@pytest.mark.requirement("REQ-102")
def test_condition_missing_from_catalog_is_level_d_with_warning(
    make_registry: Any, make_catalog: Any, store: ArtifactStore
) -> None:
    manifest = manifest_dict("fake-bright", "Brightness")
    manifest["labels"][0]["condition_id"] = "cxr.not_in_catalog"
    result = run(make_registry, make_catalog, store, [manifest], exam(200))
    assert result.findings[0].evidence_level is EvidenceLevel.D
    assert result.findings[0].visibility == "research"
    assert any("cxr.not_in_catalog" in w for w in result.warnings)


@pytest.mark.requirement("REQ-101")
@pytest.mark.requirement("REQ-102")
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
    main = result.main_findings()
    assert main[0].evidence_level is EvidenceLevel.A
    assert main[0].op_normalized is not None
    assert main[0].op_normalized > 0.5
    assert result.provenance[0].model_id == "fake-bright"
    assert result.provenance[0].weights_sha256
    assert result.provenance[0].commercial_use == "yes"


class _Policy:
    def calibrate(
        self, model_id: str, label: str, raw_score: float, score_kind: str
    ) -> tuple[float, str] | None:
        del model_id, label, score_kind
        return raw_score / 2, "cal-teste"

    def operating_points(self, model_id: str, label: str) -> list[OperatingPoint]:
        del model_id, label
        return [OperatingPoint.model_validate(op("balanced", 0.25, on="calibrated"))]


@pytest.mark.requirement("REQ-101")
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
    assert finding.calibration_id == "cal-teste"
    assert finding.probability == pytest.approx((finding.raw_score or 0) / 2)
    assert finding.decision == "positive"
    assert finding.operating_points[0].on == "calibrated"


@pytest.mark.requirement("REQ-103")
def test_unavailable_model_is_recorded_not_replaced(
    make_registry: Any, make_catalog: Any, store: ArtifactStore
) -> None:
    registry = make_registry(manifest_dict("fake-bright", "Brightness"), with_weights=False)
    result = analyze(
        exam(100), registry=registry, catalog=make_catalog(), store=store, model_ids=["fake-bright"]
    )
    assert result.status == "failed"
    assert result.findings == ()
    assert result.runs[0].status == "unavailable"
    assert "pull" in (result.runs[0].error or "")


@pytest.mark.requirement("REQ-103")
def test_planned_model_is_recorded_as_planned(
    make_registry: Any, make_catalog: Any, store: ArtifactStore
) -> None:
    planned = manifest_dict(
        "fake-planned",
        "Brightness",
        status="planned",
        planned_reason="pesos sem licença declarada",
        weights=[],
    )
    result = run(make_registry, make_catalog, store, [planned], exam(100))
    assert result.runs[0].status == "planned"
    assert "licença" in (result.runs[0].error or "")
    assert result.findings == ()
    assert result.status == "abstained"


@pytest.mark.requirement("REQ-103")
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
    assert "modelo explodiu" in (result.runs[0].error or "")
    assert [f.label for f in result.findings] == ["bright"]


@pytest.mark.requirement("REQ-103")
def test_not_applicable_model_abstains_without_findings(
    make_registry: Any, make_catalog: Any, store: ArtifactStore
) -> None:
    result = run(
        make_registry,
        make_catalog,
        store,
        [manifest_dict("fake-never", "NeverApplicable")],
        exam(100),
    )
    assert result.status == "abstained"
    assert result.findings == ()
    qa = result.runs[0].qa
    assert qa is not None
    assert AbstentionReason.OOD in qa.reasons


@pytest.mark.requirement("REQ-104")
def test_static_gate_rejects_wrong_exam_type_and_modality(
    make_registry: Any, make_catalog: Any, store: ArtifactStore
) -> None:
    manifest = manifest_dict("fake-bright", "Brightness")
    ct_exam = ExamInputs(
        images=(projection(level=100, modality=Modality.CT),), exam_type=ExamType.CT_HEAD
    )
    result = run(make_registry, make_catalog, store, [manifest], ct_exam)
    qa = result.runs[0].qa
    assert result.runs[0].status == "abstained"
    assert qa is not None
    assert AbstentionReason.EXAM_TYPE in qa.reasons
    assert AbstentionReason.MODALITY in qa.reasons


def test_static_gate_views_and_population_are_conservative(
    make_registry: Any, make_catalog: Any, store: ArtifactStore
) -> None:
    manifest = manifest_dict("fake-bright", "Brightness")
    manifest["applicability"].update({"views": ["PA", "AP"], "adult_only": True})
    lateral = ExamInputs(
        images=(projection(level=100, view_position="LATERAL", patient_age_years=40),),
        exam_type=ExamType.CXR,
    )
    unknown_age = ExamInputs(
        images=(projection(level=100, view_position="PA"),), exam_type=ExamType.CXR
    )
    ok = ExamInputs(
        images=(projection(level=100, view_position="PA", patient_age_years=40),),
        exam_type=ExamType.CXR,
    )
    r1 = run(make_registry, make_catalog, store, [manifest], lateral)
    r2 = run(make_registry, make_catalog, store, [manifest], unknown_age)
    r3 = run(make_registry, make_catalog, store, [manifest], ok)
    assert r1.runs[0].qa is not None
    assert AbstentionReason.VIEW in r1.runs[0].qa.reasons
    assert r2.runs[0].qa is not None
    assert AbstentionReason.POPULATION in r2.runs[0].qa.reasons
    assert r3.runs[0].status == "completed"


def test_first_applicable_image_is_used_and_index_is_kept(
    make_registry: Any, make_catalog: Any, store: ArtifactStore
) -> None:
    manifest = manifest_dict("fake-bright", "Brightness")
    manifest["applicability"]["views"] = ["PA"]
    inputs = ExamInputs(
        images=(
            projection(level=50, view_position="LATERAL"),
            projection(level=200, view_position="PA"),
        ),
        exam_type=ExamType.CXR,
    )
    result = run(make_registry, make_catalog, store, [manifest], inputs)
    assert result.findings[0].image_idx == 1
    assert result.findings[0].raw_score == pytest.approx(
        float(inputs.images[1].pixels.mean() / 255)
    )


def test_undeclared_label_fails_the_model(
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
    assert "não declarado" in (result.runs[0].error or "")
    assert result.findings == ()
    assert result.provenance == ()


def _segmenter() -> dict[str, Any]:
    return manifest_dict(
        "fake-segmenter",
        "Segmenter",
        task="segmentation",
        capabilities=["segment", "measure"],
        labels=[],
        route="dependency",
    )


def _derived(adapter: str = "Derived") -> dict[str, Any]:
    return manifest_dict(
        "fake-derived",
        adapter,
        task="measurement",
        capabilities=["measure"],
        depends_on=["fake-segmenter"],
        labels=[
            {
                "key": "bright_large",
                "source_label": "doubled_fraction",
                "condition_id": "cxr.bright_large",
                "kind": "derived",
                "score_kind": "measurement",
                "measurement": "doubled_fraction",
                "code": {"scheme": "99MEDAI", "value": "bl", "meaning": "bright large"},
            }
        ],
        operating_points=[
            {
                "label": "bright_large",
                "point": {**op("guideline", 0.9, on="measurement"), "source": "diretriz-teste"},
            }
        ],
    )


def test_dependencies_run_first_and_feed_derived_findings(
    make_registry: Any, make_catalog: Any, store: ArtifactStore
) -> None:
    result = run(
        make_registry,
        make_catalog,
        store,
        [_segmenter(), _derived()],
        exam(100),
        model_ids=["fake-derived"],
    )
    assert [r.model_id for r in result.runs] == ["fake-segmenter", "fake-derived"]
    assert result.status == "completed"
    seg = result.segmentations[0]
    assert seg.id == "fake-segmenter:bright_region"
    mask = store.read_labelmap(seg.mask)
    assert mask.shape == (64, 64)
    names = [m.name for m in result.measurements]
    assert names == ["bright_fraction", "doubled_fraction"]
    assert result.measurements[0].derived_from == ("fake-segmenter:bright_region",)
    derived = result.findings[0]
    assert derived.kind == "derived"
    assert derived.measurement_ref == "doubled_fraction"
    assert derived.raw_score == pytest.approx(2 * result.measurements[0].value)  # type: ignore[operator]
    assert derived.decision in ("positive", "negative")
    assert derived.operating_points[0].source == "diretriz-teste"


@pytest.mark.requirement("REQ-103")
def test_upstream_failure_makes_dependents_abstain(
    make_registry: Any, make_catalog: Any, store: ArtifactStore
) -> None:
    seg = _segmenter()
    seg["adapter"] = seg["adapter"].replace("Segmenter", "Exploding")
    result = run(
        make_registry, make_catalog, store, [seg, _derived()], exam(100), model_ids=["fake-derived"]
    )
    by_id = {r.model_id: r for r in result.runs}
    assert by_id["fake-segmenter"].status == "failed"
    assert by_id["fake-derived"].status == "abstained"
    qa = by_id["fake-derived"].qa
    assert qa is not None
    assert AbstentionReason.UPSTREAM in qa.reasons
    assert result.findings == ()


@pytest.mark.requirement("REQ-103")
def test_measurement_abstention_becomes_abstained_finding(
    make_registry: Any, make_catalog: Any, store: ArtifactStore
) -> None:
    result = run(
        make_registry,
        make_catalog,
        store,
        [_segmenter(), _derived("DerivedAbstains")],
        exam(100),
        model_ids=["fake-derived"],
    )
    finding = result.findings[0]
    assert finding.decision == "abstained"
    assert finding.raw_score is None
    assert finding.abstention_reason is AbstentionReason.ACQUISITION
    measurement = result.measurements[-1]
    assert measurement.value is None
    assert measurement.abstention_reason is AbstentionReason.ACQUISITION


def test_categorical_output(make_registry: Any, make_catalog: Any, store: ArtifactStore) -> None:
    manifest = manifest_dict(
        "fake-density",
        "Categorical",
        labels=[
            {
                "key": "density",
                "source_label": "density",
                "condition_id": "mg.density",
                "kind": "categorical",
                "score_kind": "softmax",
                "classes": ["low", "high"],
                "code": {"scheme": "99MEDAI", "value": "d", "meaning": "density"},
            }
        ],
    )
    result = run(make_registry, make_catalog, store, [manifest], exam(220))
    finding = result.findings[0]
    assert finding.decision == "category"
    assert finding.category == "high"
    assert finding.class_scores is not None
    assert set(finding.class_scores) == {"low", "high"}


def test_multi_input_roles(make_registry: Any, make_catalog: Any, store: ArtifactStore) -> None:
    manifest = manifest_dict(
        "fake-multi",
        "MultiRole",
        applicability={"exam_types": ["mg"], "modalities": ["MG"], "image_kinds": ["projection"]},
        inputs={"mode": "multi", "roles": ["L-CC", "R-CC"]},
        labels=[
            {
                "key": "asymmetry",
                "source_label": "asym",
                "condition_id": "mg.asymmetry",
                "code": {"scheme": "99MEDAI", "value": "a", "meaning": "asymmetry"},
            }
        ],
    )
    images = (
        projection(level=60, modality=Modality.MG, laterality="L", view_position="CC"),
        projection(level=180, modality=Modality.MG, laterality="R", view_position="CC"),
    )
    result = run(
        make_registry, make_catalog, store, [manifest], ExamInputs(images, exam_type=ExamType.MG)
    )
    assert result.status == "completed"
    assert result.findings[0].raw_score == pytest.approx(
        abs(images[0].pixels.mean() - images[1].pixels.mean()) / 255, rel=1e-5
    )
    missing = run(
        make_registry,
        make_catalog,
        store,
        [manifest],
        ExamInputs(images[:1], exam_type=ExamType.MG),
    )
    qa = missing.runs[0].qa
    assert missing.runs[0].status == "abstained"
    assert qa is not None
    assert AbstentionReason.MISSING_INPUT in qa.reasons


def test_explanations_only_for_positive_findings_and_stored_by_hash(
    make_registry: Any, make_catalog: Any, store: ArtifactStore
) -> None:
    manifest = manifest_dict(
        "fake-bright",
        "Brightness",
        capabilities=["classify", "explain"],
        operating_points=[{"label": "bright", "point": op("balanced", 0.5)}],
    )
    levels = {BRIGHT: EvidenceLevel.B}
    pos = run(make_registry, make_catalog, store, [manifest], exam(220), levels)
    neg = run(make_registry, make_catalog, store, [manifest], exam(30), levels)
    assert pos.findings[0].decision == "positive"
    assert len(pos.explanations) == 1
    assert pos.findings[0].explanation_ids == (pos.explanations[0].id,)
    png = store.read_bytes(pos.explanations[0].map)
    assert png.startswith(b"\x89PNG")
    assert neg.findings[0].decision == "negative"
    assert neg.explanations == ()


def test_failed_explanation_is_a_warning_not_a_fake_map(
    make_registry: Any, make_catalog: Any, store: ArtifactStore
) -> None:
    manifest = manifest_dict(
        "fake-bright",
        "BadExplanation",
        capabilities=["classify", "explain"],
        operating_points=[{"label": "bright", "point": op("balanced", 0.5)}],
    )
    result = run(make_registry, make_catalog, store, [manifest], exam(220))
    assert result.explanations == ()
    assert result.findings[0].explanation_ids == ()
    assert any("mapa falhou" in w for w in result.warnings)


def test_result_is_frozen_and_serialisable(
    make_registry: Any, make_catalog: Any, store: ArtifactStore
) -> None:
    result = run(
        make_registry, make_catalog, store, [manifest_dict("fake-bright", "Brightness")], exam(99)
    )
    assert result.intended_use == "research_use_only"
    assert "confirmação médica" in result.disclaimer
    again = AnalysisResult.model_validate_json(result.model_dump_json())
    assert again == result
    with pytest.raises(Exception, match="frozen"):
        result.findings[0].decision = "positive"  # type: ignore[misc]


def test_routing_runs_main_and_research_only_when_asked(
    make_registry: Any, make_catalog: Any, store: ArtifactStore
) -> None:
    main = manifest_dict("fake-bright", "Brightness")
    zero_shot = manifest_dict(
        "fake-zs",
        "Brightness",
        task="zero_shot",
        visibility_default="research",
    )
    registry = make_registry(main, zero_shot)
    catalog = make_catalog()
    plain = analyze(exam(120), registry=registry, catalog=catalog, store=store)
    research = analyze(
        exam(120), registry=registry, catalog=catalog, store=store, include_research=True
    )
    assert [r.model_id for r in plain.runs] == ["fake-bright"]
    assert sorted(r.model_id for r in research.runs) == ["fake-bright", "fake-zs"]
    assert research.research_mode is True
