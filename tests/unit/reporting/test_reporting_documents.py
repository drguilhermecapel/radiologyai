"""Laudo preliminar: conteúdo canônico, PDF em português, SR TID 1500, FHIR preliminar."""

from __future__ import annotations

import importlib.util
import io
import json
import uuid
from datetime import UTC, datetime

import pytest

HAS = all(importlib.util.find_spec(m) for m in ("reportlab", "highdicom", "pypdf"))
pytestmark = pytest.mark.skipif(not HAS, reason="requer o extra [platform] e pypdf")

SHA = "a" * 64


def finding(label, level, decision="positive", score=0.8, **kw):
    from radiologyai.analysis.types import Finding
    from radiologyai.calibration.decision import OperatingPoint
    from radiologyai.catalog.schema import CodedConcept, EvidenceLevel, visibility_for

    lvl = EvidenceLevel(level)
    points = ()
    if decision in ("positive", "negative", "indeterminate"):
        points = (
            OperatingPoint(
                name="publicado",
                role="rule_in",
                threshold=0.5,
                on="sigmoid",
                source="DOI 10.0/teste",
            ),
        )
    return Finding(
        condition_id=f"cxr.{label}",
        label=label,
        code=CodedConcept(scheme="99MEDAI", value=f"cxr.{label}", meaning=label),
        kind="binary",
        raw_score=score,
        score_kind="sigmoid",
        operating_points=points,
        decision=decision,
        evidence_level=lvl,
        visibility=visibility_for(lvl),
        provenance_idx=0,
        **kw,
    )


def result(*findings):
    from radiologyai.analysis.types import AnalysisResult, ModelProvenance, ModelRun
    from radiologyai.catalog.schema import ExamType

    return AnalysisResult(
        analysis_id=uuid.uuid4(),
        created_at=datetime(2026, 11, 24, tzinfo=UTC),
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
                license_weights="CC-BY-NC-4.0",
                commercial_use="no",
            ),
        ),
        runs=(
            ModelRun(model_id="modelo-x", status="completed", provenance_idx=0, seconds=0.1),
            ModelRun(model_id="modelo-y", status="unavailable", error="pesos ausentes"),
        ),
        findings=tuple(findings),
    )


def content_for(res, reviews=None):
    from radiologyai.reporting.content import build_report_content

    return build_report_content(
        res,
        result_sha256=SHA,
        analysis_id="an-1",
        study_id="st-1",
        study_uid="2.25.1",
        patient_pseudo_id="ANON-0123456789abcdef",
        reviews=reviews or {},
        catalog=None,
    )


@pytest.mark.requirement("REQ-157")
class TestContentExcludesResearch:
    def test_level_d_never_in_report(self):
        c = content_for(result(finding("a", "B"), finding("d", "D"), finding("c", "C")))
        assert [f.label for f in c.findings] == ["a", "c"]
        assert c.excluded_research_findings == 1

    def test_hash_is_canonical_and_stable(self):
        from radiologyai.reporting.content import canonical_bytes, sha256_hex

        res = result(finding("a", "B"))
        one = sha256_hex(canonical_bytes(content_for(res).model_dump(mode="json")))
        two = sha256_hex(canonical_bytes(content_for(res).model_dump(mode="json")))
        assert one == two
        assert len(one) == 64


@pytest.mark.requirement("REQ-158")
class TestPdf:
    def _text(self, content, **sign):
        from pypdf import PdfReader

        from radiologyai.reporting.pdf import SignOff, render_pdf

        data = render_pdf(content, SignOff(report_id="r1", report_sha256="e" * 64, **sign))
        assert data.startswith(b"%PDF")
        return "".join(p.extract_text() for p in PdfReader(io.BytesIO(data)).pages), data

    def test_header_levels_provenance_and_unsigned_block(self):
        from radiologyai.reporting.content import ReviewInfo

        review = ReviewInfo(
            action="reject",
            reviewer_username="dra",
            reviewer_name="Dra. Teste",
            reviewed_at="2026-11-24T10:00:00+00:00",
        )
        text, _ = self._text(content_for(result(finding("a", "B"), finding("c", "C")), {1: review}))
        assert "Uso exclusivo em pesquisa" in text
        assert "não é laudo diagnóstico" in text
        assert "Nível B" in text
        assert "Nível C" in text
        assert "rejeitado pelo médico" in text
        assert "não revisado" in text
        assert "modelo-x" in text
        assert "c" * 64 in text.replace("\n", "")
        assert "NÃO ASSINADO" in text
        assert "pesos ausentes" in text  # modelo indisponível aparece com o motivo

    def test_no_findings_never_asserts_normality(self):
        text, _ = self._text(content_for(result(finding("d", "D"))))
        lowered = " ".join(text.lower().split())
        assert "não significa exame normal" in lowered
        assert "exame normal." not in lowered.replace("não significa exame normal", "")
        assert "sem alterações" not in lowered

    def test_signed_block_and_determinism(self):
        content = content_for(result(finding("a", "A")))
        sign = {
            "signer_name": "Dra. Teste",
            "signer_username": "dra",
            "professional_id": "CRM-SP 0 (teste)",
            "signed_at": "2026-11-24T10:00:00+00:00",
            "signature_sha256": "f" * 64,
        }
        text, first = self._text(content, **sign)
        _, second = self._text(content, **sign)
        assert "Assinado por" in text
        assert "Dra. Teste" in text
        assert first == second  # invariant=1: mesmo conteúdo, mesmos bytes


@pytest.mark.requirement("REQ-159")
class TestStructuredOutputs:
    def test_fhir_is_preliminary_and_pseudonymous(self):
        from radiologyai.reporting.fhir import build_diagnostic_report
        from radiologyai.reporting.pdf import SignOff

        content = content_for(result(finding("a", "B")))
        doc = json.loads(
            build_diagnostic_report(
                content, SignOff(report_id="r1", report_sha256="e" * 64), issued="2026-11-24"
            )
        )
        assert doc["resourceType"] == "DiagnosticReport"
        assert doc["status"] == "preliminary"
        assert doc["code"]["coding"][0]["code"] == "18748-4"
        patient = doc["contained"][0]
        assert patient["identifier"][0]["value"].startswith("ANON-")
        assert "name" not in patient
        obs = doc["contained"][1]
        assert obs["status"] == "preliminary"
        assert obs["valueCodeableConcept"]["coding"][0]["code"] == "decision.positive"

    def test_sr_codes_only_from_codedict_or_private(self, cr_dataset):
        import pydicom

        from radiologyai.reporting.dicom_sr import build_sr
        from radiologyai.reporting.pdf import SignOff

        content = content_for(result(finding("a", "B")))
        data = build_sr(
            content,
            SignOff(report_id="r1", report_sha256="e" * 64),
            [cr_dataset],
            {0: (str(cr_dataset.SOPClassUID), str(cr_dataset.SOPInstanceUID))},
        )
        ds = pydicom.dcmread(io.BytesIO(data))
        assert ds.Modality == "SR"
        assert ds.VerificationFlag == "UNVERIFIED"
        from pydicom.sr.codedict import codes

        allowed = {"99MEDAI"} | set(codes.schemes())
        found = set()
        languages = []

        def walk(seq):
            for item in seq:
                for attr in (
                    "ConceptNameCodeSequence",
                    "ConceptCodeSequence",
                    "MeasurementUnitsCodeSequence",
                ):
                    for code in getattr(item, attr, []):
                        if code.CodingSchemeDesignator == "RFC5646":
                            languages.append(
                                code.get("CodeValue") or code.get("LongCodeValue")
                            )  # idioma exigido pelo TID 1500
                        else:
                            found.add(code.CodingSchemeDesignator)
                if "ContentSequence" in item:
                    walk(item.ContentSequence)

        walk(ds.ContentSequence)
        assert found, found
        assert found <= allowed, found
        assert languages == ["pt-BR"]
        # decisão é "sugestão" privada, nunca SNOMED Present/Absent
        values = set()

        def walk_values(seq):
            for item in seq:
                for code in getattr(item, "ConceptCodeSequence", []):
                    values.add(
                        (
                            code.CodingSchemeDesignator,
                            code.get("CodeValue") or code.get("LongCodeValue"),
                        )
                    )
                if "ContentSequence" in item:
                    walk_values(item.ContentSequence)

        walk_values(ds.ContentSequence)
        assert ("SCT", "52101004") not in values
        assert ("SCT", "272519000") not in values
        assert ("99MEDAI", "decision.positive") in values

    def test_sr_without_findings_carries_statement_only(self, cr_dataset):
        import pydicom

        from radiologyai.reporting.dicom_sr import build_sr
        from radiologyai.reporting.pdf import SignOff

        content = content_for(result(finding("d", "D")))
        data = build_sr(content, SignOff(report_id="r", report_sha256="e" * 64), [cr_dataset], {})
        texts = []

        def walk(seq):
            for item in seq:
                if "TextValue" in item:
                    texts.append(str(item.TextValue))
                if "ContentSequence" in item:
                    walk(item.ContentSequence)

        ds = pydicom.dcmread(io.BytesIO(data))
        walk(ds.ContentSequence)
        assert any("NÃO significa exame normal" in t for t in texts)
        assert b"decision." not in data
        # texto em português gravado em UTF-8 declarado, sem caracteres de substituição
        assert ds.SpecificCharacterSet == "ISO_IR 192"
        assert "NÃO significa exame normal".encode() in data

    def test_sr_without_dicom_evidence_refused(self):
        from radiologyai.reporting.dicom_sr import ReportFormatError, build_sr
        from radiologyai.reporting.pdf import SignOff

        with pytest.raises(ReportFormatError, match="DICOM"):
            build_sr(content_for(result()), SignOff(report_id="r", report_sha256="e" * 64), [], {})

    def test_seg_labelmap_over_source(self, cr_dataset):
        import numpy as np
        import pydicom

        from radiologyai.reporting.dicom_sr import ReportFormatError, build_seg

        mask = np.zeros((512, 512), np.uint8)
        mask[100:200, 100:300] = 1
        data = build_seg(
            report_id="r1",
            seg_name="regiao",
            model_id="modelo-x",
            model_version="1.0",
            labelmap=mask,
            labels={1: ("regiao", "região")},
            source_images=[cr_dataset],
        )
        seg = pydicom.dcmread(io.BytesIO(data))
        assert seg.Modality == "SEG"
        assert seg.SpecificCharacterSet == "ISO_IR 192"
        (item,) = [s for s in seg.SegmentSequence if s.SegmentNumber == 1]
        meaning = item.SegmentedPropertyTypeCodeSequence[0].CodeMeaning
        assert meaning == "região"
        assert seg.SegmentationType == "LABELMAP"
        with pytest.raises(ReportFormatError, match="vazio"):
            build_seg(
                report_id="r1",
                seg_name="v",
                model_id="m",
                model_version="1",
                labelmap=np.zeros((512, 512), np.uint8),
                labels={1: ("x", "x")},
                source_images=[cr_dataset],
            )
