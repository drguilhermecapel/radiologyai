"""Suspensões de decisão reaplicadas a decisões GRAVADAS (REQ-193).

Uma análise gravada ANTES de a suspensão existir não pode reemitir a decisão do
trio suspenso: laudo, SR, FHIR e a visão da API mostram só o escore, com o motivo.
Trios não suspensos ficam iguais.
"""

from __future__ import annotations

import importlib.util
import io
import json
import uuid
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

HAS = all(importlib.util.find_spec(m) for m in ("reportlab", "highdicom", "pypdf"))
pytestmark = pytest.mark.skipif(not HAS, reason="requer o extra [platform] e pypdf")

HELD = ("cxr.tb", "modelo-x", "tb")
FREE = ("cxr.nodule", "modelo-x", "nodule")


def _hold(trio=HELD):
    from radiologyai.analysis.evidence_policy import DecisionHold

    c, m, label = trio
    return DecisionHold(
        condition_id=c,
        model_id=m,
        label=label,
        reason="sinal contraditório entre bases externas",
        decided_on=date(2026, 11, 28),
        decided_by="responsável do projeto",
        evidence_ref="HONEST_STATUS.md",
    )


def _lookup(*holds):
    table = {(h.condition_id, h.model_id, h.label): h for h in holds}

    def held(condition_id, model_id, label):
        return table.get((condition_id, model_id, label))

    return held


def _finding(trio, decision, role="rule_in"):
    from radiologyai.analysis.types import Finding
    from radiologyai.calibration.decision import OperatingPoint
    from radiologyai.catalog.schema import CodedConcept, EvidenceLevel, visibility_for

    c, _, label = trio
    lvl = EvidenceLevel.C
    return Finding(
        condition_id=c,
        label=label,
        code=CodedConcept(scheme="99MEDAI", value=c, meaning=c),
        kind="binary",
        raw_score=0.7,
        score_kind="sigmoid",
        op_normalized=0.8,
        operating_points=(
            OperatingPoint(
                name="p",
                role=role,
                threshold=0.5,
                on="sigmoid",
                source="protocolo",
                evidence_ref="evidence/x.json#report.rule_in",
            ),
        ),
        decision=decision,
        evidence_level=lvl,
        visibility=visibility_for(lvl),
        provenance_idx=0,
    )


def _result(*findings):
    from radiologyai.analysis.types import AnalysisResult, ModelProvenance, ModelRun
    from radiologyai.catalog.schema import ExamType

    return AnalysisResult(
        analysis_id=uuid.uuid4(),
        created_at=datetime(2026, 11, 20, tzinfo=UTC),  # antes da suspensão
        exam_type=ExamType.CXR,
        input_sha256=("b" * 64,),
        status="completed",
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
        runs=(ModelRun(model_id="modelo-x", status="completed", provenance_idx=0, seconds=0.1),),
        findings=tuple(findings),
    )


def _content(res, held=None, reviews=None):
    from radiologyai.reporting.content import build_report_content

    return build_report_content(
        res,
        result_sha256="a" * 64,
        analysis_id="an-1",
        study_id="st-1",
        study_uid="2.25.1",
        patient_pseudo_id="ANON-0",
        reviews=reviews or {},
        catalog=None,
        held=held,
    )


def _stored():
    return _result(_finding(HELD, "positive"), _finding(FREE, "positive"))


@pytest.mark.requirement("REQ-193")
def test_report_content_withdraws_held_decision_only():
    content = _content(_stored(), held=_lookup(_hold()))
    held, free = content.findings
    assert held.decision == "score_only"
    assert held.op_normalized is None
    assert held.operating_point is not None
    assert held.operating_point.startswith("sem decisão (só escore): ")
    assert "decisão suspensa" in held.operating_point
    assert "sinal contraditório" in held.operating_point
    assert held.raw_score == pytest.approx(0.7)  # o escore continua
    assert free.decision == "positive"
    assert free.operating_point is not None
    assert "rule_in" in free.operating_point
    assert content.held_findings == 1
    assert any("achado 0" in w and "decisão suspensa" in w for w in content.warnings)


@pytest.mark.requirement("REQ-193")
def test_without_hold_content_is_unchanged():
    assert _content(_stored()) == _content(_stored(), held=_lookup())


@pytest.mark.requirement("REQ-193")
def test_negative_held_is_not_emitted_as_below_threshold():
    content = _content(_result(_finding(HELD, "negative", "rule_out")), held=_lookup(_hold()))
    (f,) = content.findings
    assert f.decision == "score_only"
    assert "não exclui" not in f.decision_pt


@pytest.mark.requirement("REQ-193")
def test_accepted_review_of_held_finding_has_no_model_verdict():
    from radiologyai.reporting.content import ReviewInfo

    review = ReviewInfo(
        action="accept",
        reviewer_username="medica",
        reviewer_name="Médica",
        reviewed_at="2026-11-21T00:00:00+00:00",
    )
    content = _content(_stored(), held=_lookup(_hold()), reviews={0: review})
    held = content.findings[0]
    assert held.physician_decision == "score_only"
    assert "acima do ponto" not in held.physician_decision_pt


def _fhir(content):
    from radiologyai.reporting.fhir import build_diagnostic_report
    from radiologyai.reporting.pdf import SignOff

    sign = SignOff(report_id="r", report_sha256="e" * 64)
    return json.loads(build_diagnostic_report(content, sign, issued="2026-11-28T00:00:00Z"))


def _sr_codes(content, cr_dataset):
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


@pytest.mark.requirement("REQ-193")
def test_fhir_and_sr_carry_no_verdict_for_held_trio(cr_dataset):
    held_only = _content(_result(_finding(HELD, "positive")), held=_lookup(_hold()))
    fhir = json.dumps(_fhir(held_only), ensure_ascii=False)
    assert "decision.positive" not in fhir
    assert "decision.below_threshold" not in fhir
    assert "decision.score_only" in fhir
    texts, codes = _sr_codes(held_only, cr_dataset)
    assert "decision.positive" not in codes
    assert "decision.score_only" in codes
    assert any("decisão suspensa" in t for t in texts)
    assert "decisão suspensa" in fhir
    # Trio não suspenso: decisão emitida como antes.
    free_only = _content(_result(_finding(FREE, "positive")), held=_lookup(_hold()))
    assert "decision.positive" in json.dumps(_fhir(free_only))
    _, free_codes = _sr_codes(free_only, cr_dataset)
    assert "decision.positive" in free_codes


@pytest.mark.requirement("REQ-193")
def test_stale_report_snapshot_is_detected():
    from radiologyai.reporting.service import stale_holds

    before = _content(_stored())  # laudo criado antes da suspensão
    assert stale_holds(before, _lookup(_hold())) == [0]
    assert stale_holds(before, _lookup()) == []
    after = _content(_stored(), held=_lookup(_hold()))
    assert stale_holds(after, _lookup(_hold())) == []


@pytest.mark.requirement("REQ-193")
def test_api_view_applies_holds_keeping_indices():
    from radiologyai.api.views import apply_holds

    stored = _stored()
    shown = apply_holds(stored, _lookup(_hold()))
    assert [f.decision for f in shown.findings] == ["score_only", "positive"]
    assert shown.findings[0].operating_points == ()
    assert shown.findings[0].decision_note
    assert "decisão suspensa" in shown.findings[0].decision_note
    assert shown.findings[1] == stored.findings[1]
    assert any("decisão gravada (positive) retirada" in w for w in shown.warnings)
    assert apply_holds(stored, _lookup()) is stored


@pytest.mark.requirement("REQ-193")
def test_hold_lookup_uses_engine_policy_then_repository_file():
    from radiologyai.analysis.evidence_policy import (
        EvidencePolicy,
        default_holds,
        hold_lookup,
    )
    from radiologyai.catalog.catalog import Catalog

    policy = EvidencePolicy(Catalog.load(), [], base=Path(), holds={HELD: _hold()})
    assert hold_lookup(policy)(*HELD) == _hold()
    assert hold_lookup(policy)(*FREE) is None
    # Sem EvidencePolicy: vale o arquivo do repositório (padrão conservador).
    repo = default_holds()
    lookup = hold_lookup(None)
    for key, hold in repo.items():
        assert lookup(*key) == hold
