"""Fluxo completo da plataforma: envio -> análise -> revisão -> laudo -> assinatura -> documentos.

Usa dublês de modelo que calculam a saída a partir dos pixels reais (ver
``platform_support``); o fluxo com pesos reais é exercitado pelo teste
``test_platform_real_models`` quando os pesos existem nesta máquina.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
from platform_support import HAS_PLATFORM, PASSWORD, dicom_bytes, png_bytes

pytestmark = pytest.mark.skipif(not HAS_PLATFORM, reason="requer os extras [api,platform]")


def upload(api, headers, dataset, **form):
    return api.post(
        "/api/v1/studies",
        headers=headers,
        files=[("files", ("exame.dcm", dicom_bytes(dataset), "application/dicom"))],
        data=form,
    )


def run_worker(platform):
    from radiologyai.worker.queue import run_one

    return run_one(platform)


def completed_analysis(api, platform, headers, cr_dataset):
    study = upload(api, headers["technician"], cr_dataset).json()
    created = api.post(
        f"/api/v1/studies/{study['id']}/analyses", headers=headers["technician"], json={}
    )
    assert created.status_code == 202, created.text
    assert created.json()["state"] == "queued"
    assert run_worker(platform) == created.json()["id"]
    body = api.get(f"/api/v1/analyses/{created.json()['id']}", headers=headers["physician"])
    assert body.status_code == 200, body.text
    return study, body.json()


@pytest.mark.requirement("REQ-153")
class TestIngestDeidentifies:
    def test_upload_deidentifies_and_decodes(self, api, headers, cr_dataset, platform):
        response = upload(api, headers["technician"], cr_dataset)
        assert response.status_code == 201, response.text
        study = response.json()
        assert study["exam_type"] == "cxr"
        assert study["exam_type_source"] == "metadata"
        assert study["patient_pseudo_id"].startswith("ANON-")
        assert study["study_uid"] != cr_dataset.StudyInstanceUID
        (series,) = study["series"]
        assert (series["rows"], series["columns"]) == (512, 512)
        # Nenhum identificador direto em lugar algum do banco ou do repositório.
        raw = platform.db.engine.url.database
        from pathlib import Path

        blob = Path(raw).read_bytes()
        for phi in (b"PACIENTE-12345", b"TESTE^SINTETICO", b"Hospital de Teste", b"ACC-999"):
            assert phi not in blob
        for path in platform.blobs.store.base_dir.rglob("*.dcm"):
            data = path.read_bytes()
            assert b"PACIENTE-12345" not in data
            assert b"TESTE^SINTETICO" not in data

    def test_missing_deident_key_refuses_dicom(self, api, headers, cr_dataset, platform):
        platform.settings.__dict__["deident_key"] = None
        response = upload(api, headers["technician"], cr_dataset)
        assert response.status_code == 503
        assert "MEDAI_DEIDENT_KEY" in response.json()["detail"]

    def test_raster_metadata_stripped_and_modality_declared(self, api, headers, platform):
        pixels = np.tile(np.linspace(0, 255, 300, dtype=np.uint8), (300, 1))
        data = png_bytes(pixels, Author="NOME DO PACIENTE")
        response = api.post(
            "/api/v1/studies",
            headers=headers["technician"],
            files=[("files", ("rx.png", data, "image/png"))],
            data={
                "modality": "DX",
                "body_part": "CHEST",
                "view_position": "PA",
                "patient_age_years": "40",
                "exam_type": "cxr",
            },
        )
        assert response.status_code == 201, response.text
        study = response.json()
        assert study["exam_type_source"] == "human"
        assert study["series"][0]["human_overrides"]["view_position"] == "PA"
        for path in platform.blobs.store.base_dir.rglob("*.png"):
            assert b"NOME DO PACIENTE" not in path.read_bytes()

    def test_human_metadata_never_overrides_dicom(self, api, headers, cr_dataset):
        response = upload(api, headers["technician"], cr_dataset, view_position="AP")
        series = response.json()["series"][0]
        assert series["view_position"] == "PA"  # gravado no DICOM; a declaração não sobrescreve


@pytest.mark.requirement("REQ-160")
class TestFullFlow:
    def test_flow_upload_to_signed_documents(self, api, headers, cr_dataset, platform):
        study, analysis = completed_analysis(api, platform, headers, cr_dataset)
        assert analysis["state"] == "completed"
        result = analysis["result"]
        # sha256 do resultado confere com os bytes gravados
        from radiologyai.persistence.models import Analysis

        with platform.db.session() as session:
            row = session.get(Analysis, analysis["id"])
            assert hashlib.sha256(row.result_json.encode()).hexdigest() == analysis["result_sha256"]
        # o escore é a média real dos pixels (dublê), transportada sem alteração
        bright = next(f for f in result["findings"] if f["label"] == "bright")
        pixels = cr_dataset.pixel_array.astype(np.float64)
        lo, hi = -32768.0, 32767.0  # faixa representável (int16, PixelRepresentation=1)
        assert bright["raw_score"] == pytest.approx((pixels.mean() - lo) / (hi - lo), rel=1e-5)
        assert bright["evidence_level"] == "B"
        assert bright["decision"] in ("positive", "negative", "indeterminate")
        # nível D oculto por padrão
        assert analysis["hidden_research_findings"] == 1
        assert all(f["visibility"] == "main" for f in result["findings"])

        idx = analysis["finding_indices"][0]
        review = api.post(
            f"/api/v1/analyses/{analysis['id']}/reviews",
            headers=headers["physician"],
            json={
                "finding_idx": idx,
                "action": "accept",
                "result_sha256": analysis["result_sha256"],
            },
        )
        assert review.status_code == 201, review.text

        report = api.post(
            f"/api/v1/analyses/{analysis['id']}/reports", headers=headers["physician"]
        )
        assert report.status_code == 201, report.text
        rep = report.json()
        assert rep["signed"] is False
        assert rep["integrity"]["content_matches_hash"] is True
        assert [f["label"] for f in rep["content"]["findings"]] == ["bright"]
        assert rep["content"]["excluded_research_findings"] == 1

        signed = api.post(
            f"/api/v1/reports/{rep['id']}/sign",
            headers=headers["physician"],
            json={
                "report_sha256": rep["report_sha256"],
                "result_sha256": rep["result_sha256"],
                "password": PASSWORD,
            },
        )
        assert signed.status_code == 200, signed.text
        assert signed.json()["integrity"]["signature_valid"] is True

        pdf = api.get(f"/api/v1/reports/{rep['id']}/pdf", headers=headers["viewer"])
        assert pdf.status_code == 200
        assert pdf.content.startswith(b"%PDF")
        import io

        from pypdf import PdfReader

        text = "".join(p.extract_text() for p in PdfReader(io.BytesIO(pdf.content)).pages)
        assert "Uso exclusivo em pesquisa" in text
        assert "não é laudo diagnóstico" in text
        assert "Nível B" in text
        assert "Assinado por" in text

        fhir = api.get(f"/api/v1/reports/{rep['id']}/fhir", headers=headers["viewer"])
        assert fhir.headers["content-type"].startswith("application/fhir+json")
        doc = fhir.json()
        assert doc["resourceType"] == "DiagnosticReport"
        assert doc["status"] == "preliminary"
        assert all(o["status"] == "preliminary" for o in doc["contained"][1:])

        sr = api.get(f"/api/v1/reports/{rep['id']}/sr", headers=headers["viewer"])
        assert sr.status_code == 200, sr.text
        import pydicom

        ds = pydicom.dcmread(io.BytesIO(sr.content))
        assert ds.Modality == "SR"
        assert ds.VerificationFlag == "VERIFIED"
        assert ds.CompletionFlag == "PARTIAL"

        seg = api.get(f"/api/v1/reports/{rep['id']}/seg/0", headers=headers["viewer"])
        assert seg.status_code == 200, seg.text
        segds = pydicom.dcmread(io.BytesIO(seg.content))
        assert segds.Modality == "SEG"
