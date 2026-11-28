"""Modo pesquisa, revisão por achado e assinatura vinculada aos hashes."""

from __future__ import annotations

import pytest
from platform_support import HAS_PLATFORM, PASSWORD, dicom_bytes

pytestmark = pytest.mark.skipif(not HAS_PLATFORM, reason="requer os extras [api,platform]")


@pytest.fixture
def analysis(api, headers, platform, cr_dataset):
    from radiologyai.worker.queue import run_one

    study = api.post(
        "/api/v1/studies",
        headers=headers["technician"],
        files=[("files", ("a.dcm", dicom_bytes(cr_dataset), "application/dicom"))],
    ).json()
    created = api.post(f"/api/v1/studies/{study['id']}/analyses", headers=headers["physician"])
    run_one(platform)
    return api.get(f"/api/v1/analyses/{created.json()['id']}", headers=headers["physician"]).json()


def review(api, headers, analysis, idx, action="accept", **extra):
    return api.post(
        f"/api/v1/analyses/{analysis['id']}/reviews",
        headers=headers["physician"],
        json={
            "finding_idx": idx,
            "action": action,
            "result_sha256": analysis["result_sha256"],
            **extra,
        },
    )


@pytest.mark.requirement("REQ-157")
class TestResearchMode:
    def test_level_d_hidden_by_default_for_everyone(self, api, headers, analysis):
        for role in ("admin", "physician", "technician", "viewer"):
            body = api.get(f"/api/v1/analyses/{analysis['id']}", headers=headers[role]).json()
            levels = {f["evidence_level"] for f in body["result"]["findings"]}
            assert "D" not in levels
            assert body["hidden_research_findings"] == 1

    def test_research_view_only_for_allowed_roles(self, api, headers, analysis):
        url = f"/api/v1/analyses/{analysis['id']}?include_research=true"
        for role in ("technician", "viewer"):
            assert api.get(url, headers=headers[role]).status_code == 403
        body = api.get(url, headers=headers["physician"]).json()
        assert body["research_view"] is True
        assert {f["evidence_level"] for f in body["result"]["findings"]} == {"B", "D"}
        assert body["finding_indices"] == [0, 1]

    def test_research_run_requires_allowed_role(self, api, headers, analysis):
        url = f"/api/v1/studies/{analysis['study_id']}/analyses"
        denied = api.post(url, headers=headers["technician"], json={"include_research": True})
        assert denied.status_code == 403
        ok = api.post(url, headers=headers["physician"], json={"include_research": True})
        assert ok.status_code == 202
        assert ok.json()["include_research"] is True

    def test_level_d_finding_cannot_be_reviewed(self, api, headers, analysis):
        research_idx = 1  # 'dark' é nível D no catálogo de teste
        response = review(api, headers, analysis, research_idx)
        assert response.status_code == 409
        assert "nível D" in response.json()["detail"]


@pytest.mark.requirement("REQ-156")
class TestSignatureBinding:
    def _report(self, api, headers, analysis):
        for idx in analysis["finding_indices"]:
            assert review(api, headers, analysis, idx).status_code == 201
        return api.post(
            f"/api/v1/analyses/{analysis['id']}/reports", headers=headers["physician"]
        ).json()

    def test_review_bound_to_seen_result(self, api, headers, analysis):
        stale = dict(analysis, result_sha256="0" * 64)
        assert review(api, headers, stale, analysis["finding_indices"][0]).status_code == 409

    def test_amend_requires_content(self, api, headers, analysis):
        idx = analysis["finding_indices"][0]
        assert review(api, headers, analysis, idx, action="amend").status_code == 422
        ok = review(
            api,
            headers,
            analysis,
            idx,
            action="amend",
            amended_decision="negative",
            amended_text="sem achado à revisão",
        )
        assert ok.status_code == 201

    def test_unreviewed_findings_block_signature(self, api, headers, analysis):
        rep = api.post(
            f"/api/v1/analyses/{analysis['id']}/reports", headers=headers["physician"]
        ).json()
        assert rep["content"]["unreviewed_findings"] == 1
        response = api.post(
            f"/api/v1/reports/{rep['id']}/sign",
            headers=headers["physician"],
            json={
                "report_sha256": rep["report_sha256"],
                "result_sha256": rep["result_sha256"],
                "password": PASSWORD,
            },
        )
        assert response.status_code == 409
        assert "sem revisão" in response.json()["detail"]

    def test_wrong_hash_or_password_refused(self, api, headers, analysis):
        rep = self._report(api, headers, analysis)
        url = f"/api/v1/reports/{rep['id']}/sign"
        good = {"report_sha256": rep["report_sha256"], "result_sha256": rep["result_sha256"]}
        bad_hash = api.post(
            url,
            headers=headers["physician"],
            json={**good, "report_sha256": "1" * 64, "password": PASSWORD},
        )
        assert bad_hash.status_code == 409
        bad_pass = api.post(
            url, headers=headers["physician"], json={**good, "password": "senha-errada-123"}
        )
        assert bad_pass.status_code == 401
        for role in ("admin", "technician", "viewer"):
            assert (
                api.post(
                    url, headers=headers[role], json={**good, "password": PASSWORD}
                ).status_code
                == 403
            )
        ok = api.post(url, headers=headers["physician"], json={**good, "password": PASSWORD})
        assert ok.status_code == 200
        assert ok.json()["integrity"]["signature_valid"] is True
        again = api.post(url, headers=headers["physician"], json={**good, "password": PASSWORD})
        assert again.status_code == 409

    def test_tampering_after_signature_invalidates_and_blocks_documents(
        self, api, headers, analysis, platform
    ):
        from radiologyai.persistence.models import Analysis

        rep = self._report(api, headers, analysis)
        api.post(
            f"/api/v1/reports/{rep['id']}/sign",
            headers=headers["physician"],
            json={
                "report_sha256": rep["report_sha256"],
                "result_sha256": rep["result_sha256"],
                "password": PASSWORD,
            },
        )
        with platform.db.session() as session:
            row = session.get(Analysis, analysis["id"])
            row.result_json = row.result_json.replace('"bright"', '"brilho"', 1)
        got = api.get(f"/api/v1/reports/{rep['id']}", headers=headers["viewer"]).json()
        assert got["integrity"]["result_matches_hash"] is False
        assert got["integrity"]["signature_valid"] is False
        assert (
            api.get(f"/api/v1/reports/{rep['id']}/pdf", headers=headers["viewer"]).status_code
            == 409
        )
        assert (
            api.get(f"/api/v1/analyses/{analysis['id']}", headers=headers["viewer"]).status_code
            == 500
        )

    def test_audit_reconstructs_the_decision(self, api, headers, analysis):
        rep = self._report(api, headers, analysis)
        api.post(
            f"/api/v1/reports/{rep['id']}/sign",
            headers=headers["physician"],
            json={
                "report_sha256": rep["report_sha256"],
                "result_sha256": rep["result_sha256"],
                "password": PASSWORD,
            },
        )
        events = api.get("/api/v1/audit?limit=500", headers=headers["admin"]).json()["items"]
        names = [e["event"] for e in events]
        for expected in (
            "study.ingested",
            "analysis.requested",
            "analysis.completed",
            "review.created",
            "report.created",
            "report.signed",
        ):
            assert expected in names
        completed = next(e for e in events if e["event"] == "analysis.completed")
        assert completed["payload"]["result_sha256"] == analysis["result_sha256"]
        assert completed["payload"]["models"][0]["weights_sha256"]
        assert api.get("/api/v1/audit/verify", headers=headers["admin"]).json()["ok"] is True


@pytest.mark.requirement("REQ-163")
def test_segmentation_overlay_on_same_grid(api, headers, analysis):
    seg = api.get(
        f"/api/v1/analyses/{analysis['id']}/segmentations/0/frames/0?max_dim=128",
        headers=headers["viewer"],
    )
    img = api.get(
        f"/api/v1/series/{analysis['inputs'][0]['series_id']}/frames/0?max_dim=128",
        headers=headers["viewer"],
    )
    assert seg.status_code == img.status_code == 200
    for key in ("x-frame-rows", "x-frame-columns", "x-frame-scale"):
        assert seg.headers[key] == img.headers[key]
    assert seg.headers["x-frame-dtype"] == "uint8"
    assert len(seg.content) == int(seg.headers["x-frame-rows"]) * int(
        seg.headers["x-frame-columns"]
    )
