"""Documentos: decisão do médico primeiro, nada lido como exclusão, medida nomeada como medida."""

from __future__ import annotations

import importlib.util
import io
import json
import uuid
from datetime import UTC, datetime

import pytest

HAS = all(importlib.util.find_spec(m) for m in ("reportlab", "highdicom", "pypdf"))
pytestmark = pytest.mark.skipif(not HAS, reason="requer o extra [platform] e pypdf")


def _result(*findings, runs=None, measurements=()):
    from radiologyai.analysis.types import AnalysisResult, ModelProvenance, ModelRun
    from radiologyai.catalog.schema import ExamType

    return AnalysisResult(
        analysis_id=uuid.uuid4(),
        created_at=datetime(2026, 11, 24, tzinfo=UTC),
        exam_type=ExamType.CXR,
        input_sha256=("b" * 64,),
        status="partial" if runs else "completed",
        provenance=(
            ModelProvenance(
                model_id="modelo-x",
                model_version="1.0",
                weights_sha256={"main": "c" * 64},
                manifest_sha256="d" * 64,
                code_version="2.0.0.dev0",
                license_code="Apache-2.0",
                license_weights="teste",
                commercial_use="no",
            ),
        ),
        runs=runs
        or (ModelRun(model_id="modelo-x", status="completed", provenance_idx=0, seconds=0.1),),
        findings=tuple(findings),
        measurements=tuple(measurements),
    )


def _binary(decision="negative", level="C"):
    from radiologyai.analysis.types import Finding
    from radiologyai.calibration.decision import OperatingPoint
    from radiologyai.catalog.schema import CodedConcept, EvidenceLevel, visibility_for

    lvl = EvidenceLevel(level)
    return Finding(
        condition_id="cxr.tb",
        label="tb",
        code=CodedConcept(scheme="99MEDAI", value="cxr.tb", meaning="tb"),
        kind="binary",
        raw_score=0.1,
        score_kind="sigmoid",
        operating_points=(
            OperatingPoint(
                name="p", role="rule_out", threshold=0.2, on="sigmoid", source="protocolo"
            ),
        ),
        decision=decision,
        evidence_level=lvl,
        visibility=visibility_for(lvl),
        provenance_idx=0,
    )


def _content(res, reviews=None, catalog=None, current_level=None):
    from radiologyai.reporting.content import build_report_content

    return build_report_content(
        res,
        result_sha256="a" * 64,
        analysis_id="an-1",
        study_id="st-1",
        study_uid="2.25.1",
        patient_pseudo_id="ANON-0",
        reviews=reviews or {},
        catalog=catalog,
        current_level=current_level,
    )


def _fhir(content):
    from radiologyai.reporting.fhir import build_diagnostic_report
    from radiologyai.reporting.pdf import SignOff

    sign = SignOff(report_id="r", report_sha256="e" * 64)
    return json.loads(build_diagnostic_report(content, sign, issued="2026-11-24T00:00:00Z"))


def _sr_texts_and_codes(content, cr_dataset):
    import pydicom

    from radiologyai.reporting.dicom_sr import build_sr
    from radiologyai.reporting.pdf import SignOff

    data = build_sr(content, SignOff(report_id="r", report_sha256="e" * 64), [cr_dataset], {})
    texts, codes = [], []

    def walk(seq):
        for item in seq:
            if "TextValue" in item:
                texts.append(str(item.TextValue))
            if "ConceptCodeSequence" in item:
                c = item.ConceptCodeSequence[0]
                codes.append(str(c.get("CodeValue") or c.get("LongCodeValue")))
            if "ContentSequence" in item:
                walk(item.ContentSequence)

    walk(pydicom.dcmread(io.BytesIO(data)).ContentSequence)
    return texts, codes


@pytest.mark.requirement("REQ-172")
def test_below_threshold_is_never_coded_as_negative(cr_dataset):
    content = _content(_result(_binary("negative")))
    (f,) = content.findings
    assert "não exclui a condição" in f.decision_pt
    assert "ponto de exclusão" not in f.decision_pt.replace("limiar de exclusão", "")
    obs = _fhir(content)["contained"][1]
    codes = [c["code"] for c in obs["valueCodeableConcept"]["coding"]]
    assert codes == ["decision.below_threshold"]
    assert "decision.negative" not in json.dumps(_fhir(content))
    _, sr_codes = _sr_texts_and_codes(content, cr_dataset)
    assert "decision.below_threshold" in sr_codes
    assert "decision.negative" not in sr_codes


@pytest.mark.requirement("REQ-172")
def test_accepted_below_threshold_stays_non_exclusion():
    from radiologyai.reporting.content import ReviewInfo

    review = ReviewInfo(
        action="accept",
        reviewer_username="dra",
        reviewer_name="Dra",
        reviewed_at="2026-11-24T10:00:00",
    )
    obs = _fhir(_content(_result(_binary("negative")), reviews={0: review}))["contained"][1]
    (coding,) = obs["valueCodeableConcept"]["coding"]
    assert coding["code"] == "physician-accepted.below_threshold"
    assert "does not exclude" in coding["display"]


@pytest.mark.requirement("REQ-172")
def test_derived_score_only_names_the_measurement_not_the_condition():
    from radiologyai.analysis.types import Finding, Measurement
    from radiologyai.catalog.catalog import Catalog
    from radiologyai.catalog.schema import (
        CodedConcept,
        Condition,
        EvidenceLevel,
        ExamType,
        LevelEntry,
        LevelsFile,
    )

    finding = Finding(
        condition_id="echo.reduced_lvef",
        label="lv_ef",
        code=CodedConcept(
            scheme="99MEDAI",
            value="echo.reduced_lvef",
            meaning="left ventricular ejection fraction",
        ),
        kind="derived",
        raw_score=62.0,
        score_kind="measurement",
        decision="score_only",
        evidence_level=EvidenceLevel.C,
        visibility="main",
        measurement_ref="lv_ef",
        provenance_idx=0,
    )
    measure = Measurement(
        name="lv_ef",
        code=CodedConcept(scheme="99MEDAI", value="lv_ef", meaning="lv ef"),
        value=62.0,
        unit="%",
        method="Simpson",
        provenance_idx=0,
    )
    catalog = Catalog(
        [
            Condition(
                id="echo.reduced_lvef",
                name_pt="Fração de ejeção do VE reduzida",
                name_en="reduced LVEF",
                group="echo",
                exam_types=(ExamType.CXR,),
                system="teste",
                code=CodedConcept(scheme="99MEDAI", value="echo.reduced_lvef", meaning="x"),
            )
        ],
        LevelsFile(
            protocol_id="t",
            evidence_digest="0" * 64,
            entries=(
                LevelEntry(
                    condition_id="echo.reduced_lvef",
                    model_id="modelo-x",
                    label="lv_ef",
                    level=EvidenceLevel.C,
                    visibility="main",
                    basis=("r",),
                ),
            ),
        ),
    )
    content = _content(_result(finding, measurements=(measure,)), catalog=catalog)
    (f,) = content.findings
    assert f.measurement_only is True
    assert "reduzida" not in f.condition_name
    assert f.condition_name == "Medida: left ventricular ejection fraction"
    assert "reduzida" not in _fhir(content)["contained"][1]["code"]["text"]


@pytest.mark.requirement("REQ-171")
def test_models_that_did_not_complete_are_stated_in_sr_and_fhir(cr_dataset):
    from radiologyai.analysis.types import ModelRun

    runs = (
        ModelRun(model_id="modelo-x", status="completed", provenance_idx=0, seconds=0.1),
        ModelRun(model_id="modelo-y", status="failed", error="explodiu"),
    )
    content = _content(_result(_binary("positive"), runs=runs))
    assert content.result_status == "partial"
    assert content.completed_models == 1
    assert content.runs_statement
    assert "NÃO foi analisado" in content.runs_statement
    assert content.runs_statement in _fhir(content)["conclusion"]
    texts, _ = _sr_texts_and_codes(content, cr_dataset)
    assert any("NÃO foi analisado" in t for t in texts)


@pytest.mark.requirement("REQ-173")
def test_current_level_stricter_than_stored_wins():
    from radiologyai.catalog.schema import EvidenceLevel

    demoted = _content(_result(_binary(level="B")), current_level=lambda *_: EvidenceLevel.C)
    (f,) = demoted.findings
    assert (f.evidence_level, f.level_at_analysis) == ("C", "B")
    gone = _content(_result(_binary(level="B")), current_level=lambda *_: EvidenceLevel.D)
    assert gone.findings == ()
    assert gone.demoted_findings == 1
    promoted = _content(_result(_binary(level="C")), current_level=lambda *_: EvidenceLevel.A)
    assert promoted.findings[0].evidence_level == "C"  # nunca promove sem novo laudo/análise
