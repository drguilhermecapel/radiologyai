"""Modo silencioso do piloto prospectivo (REQ-195, portão G4).

As análises rodam, são gravadas e auditadas; a saída de IA não chega a quem
não tem ``shadow:view``; nada sai para o PACS; a equipe de pesquisa vê tudo e
exporta um CSV sem PHI, auditado.
"""

# ruff: noqa: ARG001, ARG002 - fixtures usadas pelo efeito colateral

from __future__ import annotations

import csv
import hashlib
import io

import pytest
from platform_support import HAS_PLATFORM, PASSWORD, dicom_bytes

pytestmark = pytest.mark.skipif(not HAS_PLATFORM, reason="requer os extras [api,platform]")

CLINICAL = ("physician", "technician", "viewer", "admin")


def _shadow(platform, on: bool) -> None:
    platform.settings.__dict__["shadow_mode"] = on


def _blocked(response) -> None:
    assert response.status_code == 403, response.text
    assert response.json()["error"] == "ShadowModeError"


def _study(api, headers, cr_dataset):
    response = api.post(
        "/api/v1/studies",
        headers=headers["technician"],
        files=[("files", ("a.dcm", dicom_bytes(cr_dataset), "application/dicom"))],
    )
    assert response.status_code == 201, response.text
    return response.json()


def _run(api, headers, platform, study, role="technician"):
    from radiologyai.worker.queue import run_one

    created = api.post(f"/api/v1/studies/{study['id']}/analyses", headers=headers[role], json={})
    assert created.status_code == 202, created.text
    assert run_one(platform) == created.json()["id"]
    return created.json()


@pytest.fixture
def team(headers, make_user):
    return {**headers, "researcher": make_user("researcher", username="pesquisa")}


@pytest.fixture
def pilot(api, team, platform, cr_dataset):
    """Estudo com análise, revisão e laudo assinado (feitos antes do piloto); depois, silêncio."""
    study = _study(api, team, cr_dataset)
    created = _run(api, team, platform, study)
    analysis = api.get(f"/api/v1/analyses/{created['id']}", headers=team["physician"]).json()
    for idx in analysis["finding_indices"]:
        r = api.post(
            f"/api/v1/analyses/{analysis['id']}/reviews",
            headers=team["physician"],
            json={
                "finding_idx": idx,
                "action": "accept",
                "result_sha256": analysis["result_sha256"],
            },
        )
        assert r.status_code == 201, r.text
    report = api.post(f"/api/v1/analyses/{analysis['id']}/reports", headers=team["physician"])
    assert report.status_code == 201, report.text
    rep = report.json()
    signed = api.post(
        f"/api/v1/reports/{rep['id']}/sign",
        headers=team["physician"],
        json={
            "report_sha256": rep["report_sha256"],
            "result_sha256": rep["result_sha256"],
            "password": PASSWORD,
        },
    )
    assert signed.status_code == 200, signed.text
    _shadow(platform, True)
    return {"study": study, "analysis": analysis, "report": signed.json()}


def _ai_urls(pilot) -> dict[str, str]:
    a, rep = pilot["analysis"], pilot["report"]
    seg = a["segmentation_indices"][0]
    seg_key = next(k for k in rep["downloads"] if k.startswith("seg/"))
    return {
        "segmentation_frame": f"/api/v1/analyses/{a['id']}/segmentations/{seg}/frames/0",
        "explanation": f"/api/v1/artifacts/{a['result']['explanations'][0]['map']['sha256']}",
        "mask": f"/api/v1/artifacts/{a['result']['segmentations'][0]['mask']['sha256']}",
        "reviews": f"/api/v1/analyses/{a['id']}/reviews",
        "reports": f"/api/v1/analyses/{a['id']}/reports",
        "report": f"/api/v1/reports/{rep['id']}",
        "pdf": f"/api/v1/reports/{rep['id']}/pdf",
        "sr": f"/api/v1/reports/{rep['id']}/sr",
        "fhir": f"/api/v1/reports/{rep['id']}/fhir",
        "seg": f"/api/v1/reports/{rep['id']}/{seg_key}",
    }


def _events(api, team, event):
    return api.get("/api/v1/audit", headers=team["admin"], params={"event": event}).json()["items"]


@pytest.mark.requirement("REQ-195")
class TestStatus:
    def test_status_exposes_shadow_mode(self, api, platform):
        assert api.get("/api/v1/status").json()["shadow_mode"] is False
        _shadow(platform, True)
        assert api.get("/api/v1/status").json()["shadow_mode"] is True

    def test_only_researcher_holds_shadow_view(self):
        from radiologyai.persistence.security import PERMISSIONS, Permission, Role

        assert PERMISSIONS[Permission.SHADOW_VIEW] == frozenset({Role.RESEARCHER})
        for p in (
            Permission.REVIEWS_WRITE,
            Permission.REPORTS_CREATE,
            Permission.REPORTS_SIGN,
            Permission.DICOMWEB_EXPORT,
            Permission.LEGACY_ANALYZE,
            Permission.USERS_MANAGE,
            Permission.AUDIT_READ,
        ):
            assert Role.RESEARCHER not in PERMISSIONS[p], p


@pytest.mark.requirement("REQ-195")
class TestClinicianSeesNothing:
    @pytest.mark.parametrize("role", CLINICAL)
    def test_mixed_resources_are_redacted(self, api, team, pilot, role):
        a = pilot["analysis"]
        auth = team[role]
        out = api.get(f"/api/v1/analyses/{a['id']}", headers=auth)
        assert out.status_code == 200, out.text
        body = out.json()
        assert body["shadow"] is True
        assert body["state"] == "completed"
        for key in ("result", "result_sha256", "result_status", "finding_indices"):
            assert body[key] is None, key
        assert body["reviews"] == []
        assert body["hidden_research_findings"] == 0
        assert a["result_sha256"] not in out.text
        listed = api.get(f"/api/v1/studies/{pilot['study']['id']}/analyses", headers=auth).json()
        assert all(x["shadow"] and x["result"] is None and x["reviews"] == [] for x in listed)
        study = api.get(f"/api/v1/studies/{pilot['study']['id']}", headers=auth).json()
        for summary in study["analyses"]:
            assert summary["shadow"] is True
            assert summary["n_main_findings"] is None
            assert summary["n_research_findings"] is None
            assert summary["result_status"] is None
        worklist = api.get("/api/v1/studies", headers=auth).json()["items"]
        latest = worklist[0]["latest_analysis"]
        assert latest["shadow"] is True
        assert latest["n_main_findings"] is None

    def test_research_query_does_not_bypass_silence(self, api, team, pilot):
        a = pilot["analysis"]
        out = api.get(
            f"/api/v1/analyses/{a['id']}",
            headers=team["physician"],
            params={"include_research": True},
        ).json()
        assert out["shadow"] is True
        assert out["result"] is None

    @pytest.mark.parametrize("role", CLINICAL)
    def test_ai_only_resources_are_403(self, api, team, pilot, role):
        for name, url in _ai_urls(pilot).items():
            response = api.get(url, headers=team[role])
            assert response.status_code == 403, (name, response.status_code)
            assert response.json()["error"] == "ShadowModeError", name
            assert pilot["analysis"]["result_sha256"] not in response.text, name

    def test_physician_acts_on_ai_output_are_blocked(self, api, team, pilot, cr_dataset):
        a, rep = pilot["analysis"], pilot["report"]
        _blocked(
            api.post(
                f"/api/v1/analyses/{a['id']}/reviews",
                headers=team["physician"],
                json={"finding_idx": 0, "action": "accept", "result_sha256": a["result_sha256"]},
            )
        )
        _blocked(api.post(f"/api/v1/analyses/{a['id']}/reports", headers=team["physician"]))
        _blocked(
            api.post(
                f"/api/v1/reports/{rep['id']}/sign",
                headers=team["physician"],
                json={
                    "report_sha256": rep["report_sha256"],
                    "result_sha256": rep["result_sha256"],
                    "password": PASSWORD,
                },
            )
        )
        for role in ("physician", "technician"):
            _blocked(
                api.post(
                    "/api/v1/analyze",
                    headers=team[role],
                    files={"file": ("a.dcm", dicom_bytes(cr_dataset), "application/dicom")},
                )
            )

    def test_exam_input_artifact_still_served(self, api, team, pilot, platform):
        from radiologyai.persistence.models import Artifact

        with platform.db.session() as session:
            sha = session.query(Artifact.sha256).filter(Artifact.kind == "input").first()[0]
        assert api.get(f"/api/v1/artifacts/{sha}", headers=team["viewer"]).status_code == 200

    def test_blocks_and_redactions_are_audited(self, api, team, pilot):
        urls = _ai_urls(pilot)
        api.get(urls["pdf"], headers=team["physician"])
        api.get(f"/api/v1/analyses/{pilot['analysis']['id']}", headers=team["viewer"])
        blocked = _events(api, team, "shadow.blocked")
        assert any(
            e["payload"]["action"] == "report.pdf"
            and e["payload"]["role"] == "physician"
            and e["subject_id"] == pilot["report"]["id"]
            for e in blocked
        )
        redacted = _events(api, team, "shadow.redacted")
        assert redacted[-1]["subject_id"] == pilot["analysis"]["id"]
        assert redacted[-1]["payload"]["role"] == "viewer"
        assert api.get("/api/v1/audit/verify", headers=team["admin"]).json()["ok"] is True


@pytest.mark.requirement("REQ-195")
class TestResearchSeesEverything:
    def test_researcher_full_access(self, api, team, pilot):
        a = pilot["analysis"]
        auth = team["researcher"]
        out = api.get(f"/api/v1/analyses/{a['id']}", headers=auth).json()
        assert out["shadow"] is False
        assert out["result_sha256"] == a["result_sha256"]
        assert out["result"]["findings"] == a["result"]["findings"]
        assert len(out["reviews"]) == len(a["finding_indices"])
        full = api.get(
            f"/api/v1/analyses/{a['id']}", headers=auth, params={"include_research": True}
        ).json()
        assert full["research_view"] is True
        assert full["hidden_research_findings"] == 0
        study = api.get(f"/api/v1/studies/{pilot['study']['id']}", headers=auth).json()
        assert study["analyses"][0]["shadow"] is False
        assert study["analyses"][0]["n_main_findings"] >= 1
        for name, url in _ai_urls(pilot).items():
            response = api.get(url, headers=auth)
            assert response.status_code == 200, (name, response.status_code, response.text[:200])

    def test_analyses_keep_running_and_being_stored(self, api, team, pilot, platform, cr_dataset):
        from radiologyai.persistence.models import Analysis

        created = _run(api, team, platform, pilot["study"])
        assert created["shadow"] is True  # resposta ao técnico, que não vê a saída de IA
        with platform.db.session() as session:
            row = session.get(Analysis, created["id"])
            assert row.state == "completed"
            assert hashlib.sha256(row.result_json.encode()).hexdigest() == row.result_sha256
        completed = _events(api, team, "analysis.completed")
        assert any(e["subject_id"] == created["id"] for e in completed)
        seen = api.get(f"/api/v1/analyses/{created['id']}", headers=team["researcher"]).json()
        assert seen["result"]["findings"]


@pytest.mark.requirement("REQ-195")
class TestNothingLeavesTowardCare:
    def test_pacs_export_blocked_for_everyone(self, api, team, pilot, platform, monkeypatch):
        from radiologyai.interop import dicomweb

        platform.settings.__dict__["dicomweb_url"] = "http://127.0.0.1:9"

        def never(*_args, **_kwargs):
            raise AssertionError("STOW-RS chamado no modo silencioso")

        monkeypatch.setattr(dicomweb.DicomwebGateway, "store", never)
        url = f"/api/v1/reports/{pilot['report']['id']}/export"
        _blocked(api.post(url, headers=team["physician"]))
        for role in ("researcher", "admin", "technician", "viewer"):
            assert api.post(url, headers=team[role]).status_code == 403
        blocked = [e for e in _events(api, team, "shadow.blocked") if e["payload"]["care_export"]]
        assert blocked
        assert blocked[-1]["payload"]["action"] == "report.export"
        assert not _events(api, team, "report.exported")

    def test_care_export_guard_holds_even_with_shadow_view(self, api, platform):
        from radiologyai.api.deps import Principal
        from radiologyai.api.shadow import ShadowModeError, block_care_export
        from radiologyai.persistence.security import Permission

        _shadow(platform, True)
        researcher = Principal("u", "pesquisa", "P", "researcher", None, "s")
        assert researcher.can(Permission.SHADOW_VIEW)
        with pytest.raises(ShadowModeError):
            block_care_export(platform, researcher, action="report.export")


@pytest.mark.requirement("REQ-195")
class TestCsvExport:
    PHI_HINTS = ("patient", "birth", "accession", "description", "institution", "sex", "age")

    def test_csv_has_no_phi_and_is_audited(self, api, team, pilot, cr_dataset):
        from radiologyai.api.routers.shadow import COLUMNS

        response = api.get("/api/v1/shadow/export", headers=team["researcher"])
        assert response.status_code == 200, response.text
        assert response.headers["content-type"].startswith("text/csv")
        text = response.text
        reader = csv.DictReader(io.StringIO(text))
        assert tuple(reader.fieldnames or ()) == COLUMNS
        for column in COLUMNS:
            assert not any(hint in column for hint in self.PHI_HINTS), column
        assert not {"name", "full_name", "patient_pseudo_id", "study_uid"} & set(COLUMNS)
        rows = list(reader)
        a = pilot["analysis"]
        mine = [r for r in rows if r["analysis_id"] == a["id"]]
        total = len(a["result"]["findings"]) + a["hidden_research_findings"]
        assert len(mine) == total
        assert {r["result_sha256"] for r in mine} == {a["result_sha256"]}
        assert {r["study_uid_pseudo"] for r in mine} == {pilot["study"]["study_uid"]}
        assert {r["evidence_level"] for r in mine} >= {"B", "D"}
        assert all(r["model_id"] and r["model_version"] and r["condition_id"] for r in mine)
        for phi in (
            "PACIENTE-12345",
            "TESTE^SINTETICO",
            "19800115",
            "ACC-999",
            "Hospital de Teste",
            str(cr_dataset.StudyInstanceUID),
            pilot["study"]["patient_pseudo_id"],
        ):
            assert phi not in text, phi
        digest = hashlib.sha256(response.content).hexdigest()
        assert response.headers["x-content-sha256"] == digest
        (event,) = _events(api, team, "shadow.exported")
        assert event["payload"]["csv_sha256"] == digest
        assert event["payload"]["rows"] == len(rows)
        assert event["actor_username"] == "pesquisa"
        assert api.get("/api/v1/audit/verify", headers=team["admin"]).json()["ok"] is True

    @pytest.mark.parametrize("role", CLINICAL)
    def test_csv_export_is_research_only(self, api, team, pilot, role):
        assert api.get("/api/v1/shadow/export", headers=team[role]).status_code == 403


@pytest.mark.requirement("REQ-195")
class TestNormalModeUnchanged:
    def test_shadow_off_serves_ai_output_as_before(self, api, team, pilot, platform):
        _shadow(platform, False)
        a = pilot["analysis"]
        out = api.get(f"/api/v1/analyses/{a['id']}", headers=team["viewer"]).json()
        assert out["shadow"] is False
        assert out["result_sha256"] == a["result_sha256"]
        assert out["result"]["findings"]
        study = api.get(f"/api/v1/studies/{pilot['study']['id']}", headers=team["viewer"]).json()
        assert study["analyses"][0]["shadow"] is False
        assert study["analyses"][0]["n_main_findings"] >= 1
        for name, url in _ai_urls(pilot).items():
            response = api.get(url, headers=team["physician"])
            assert response.status_code == 200, (name, response.text[:200])
        assert not _events(api, team, "shadow.blocked")
        assert not _events(api, team, "shadow.redacted")


@pytest.mark.requirement("REQ-195")
class TestResearcherRole:
    def test_matrix_enforced_for_researcher(self, api, team):
        """O papel novo segue a matriz rota a rota, como os demais (REQ-151)."""
        from test_platform_rbac import api_routes, call, fill

        from radiologyai.persistence.security import PERMISSIONS, Permission, Role

        checked = 0
        for method, raw_path, declared in api_routes(api.app):
            if declared in ("public", "authenticated"):
                continue
            response = call(api, method, fill(raw_path), team["researcher"])
            if Role.RESEARCHER in PERMISSIONS[Permission(declared)]:
                assert response.status_code not in (401, 403), (method, raw_path)
            else:
                assert response.status_code == 403, (method, raw_path, response.status_code)
            checked += 1
        assert checked > 30

    def test_me_lists_shadow_view(self, api, team):
        me = api.get("/api/v1/auth/me", headers=team["researcher"]).json()
        assert me["role"] == "researcher"
        assert "shadow:view" in me["permissions"]
        assert "reports:sign" not in me["permissions"]
