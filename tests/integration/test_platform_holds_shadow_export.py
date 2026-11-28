"""Suspensões de decisão nas rotas que reemitem decisões gravadas (REQ-193) e o recorte da
exportação do piloto (REQ-195).

A suspensão é criada DEPOIS da análise (e de um laudo): a API, o laudo novo e o CSV não
podem reemitir a decisão gravada do trio suspenso; o laudo antigo não é emitido nem
assinado. Trios não suspensos ficam iguais.
"""

# ruff: noqa: ARG001, ARG002 - fixtures usadas pelo efeito colateral

from __future__ import annotations

import csv
import io
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest
from platform_support import HAS_PLATFORM, PASSWORD, dicom_bytes

pytestmark = pytest.mark.skipif(not HAS_PLATFORM, reason="requer os extras [api,platform]")

HELD = ("cxr.bright", "fake-pixel-mean", "bright")


def _shadow(platform, on: bool) -> None:
    platform.settings.__dict__["shadow_mode"] = on


def _study(api, headers, cr_dataset):
    response = api.post(
        "/api/v1/studies",
        headers=headers["technician"],
        files=[("files", ("a.dcm", dicom_bytes(cr_dataset), "application/dicom"))],
    )
    assert response.status_code == 201, response.text
    return response.json()


def _run(api, headers, platform, study, *, research=False):
    from radiologyai.worker.queue import run_one

    created = api.post(
        f"/api/v1/studies/{study['id']}/analyses",
        headers=headers["technician"],
        json={"include_research": research} if research else {},
    )
    assert created.status_code == 202, created.text
    assert run_one(platform) == created.json()["id"]
    return created.json()


def _put_hold(platform, trio=HELD) -> None:
    """Simula uma suspensão registrada depois da análise (motor recarregado)."""
    from radiologyai.analysis.evidence_policy import DecisionHold, EvidencePolicy

    c, m, label = trio
    hold = DecisionHold(
        condition_id=c,
        model_id=m,
        label=label,
        reason="sinal contraditório (teste)",
        decided_on=date(2026, 11, 28),
        decided_by="responsável do projeto",
        evidence_ref="HONEST_STATUS.md",
    )
    engine = platform.engine
    engine.policy = EvidencePolicy(engine.catalog, [], base=Path(), holds={trio: hold})


def _events(api, headers, event):
    return api.get("/api/v1/audit", headers=headers["admin"], params={"event": event}).json()[
        "items"
    ]


def _csv(response):
    assert response.status_code == 200, response.text
    return list(csv.DictReader(io.StringIO(response.text)))


@pytest.fixture
def team(headers, make_user):
    return {**headers, "researcher": make_user("researcher", username="pesquisa")}


@pytest.fixture
def before_hold(api, team, platform, cr_dataset):
    """Análise revisada e laudo criado ANTES da suspensão; depois, a suspensão."""
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
    full = api.get(
        f"/api/v1/analyses/{analysis['id']}",
        headers=team["physician"],
        params={"include_research": True},
    )
    assert full.status_code == 200, full.text
    _put_hold(platform)
    return {"study": study, "analysis": analysis, "full": full.json(), "report": report.json()}


@pytest.mark.requirement("REQ-193")
class TestHoldsOnStoredDecisions:
    def test_fixture_had_a_decision_before_the_hold(self, before_hold):
        a = before_hold["analysis"]
        bright = [f for f in a["result"]["findings"] if f["condition_id"] == "cxr.bright"]
        assert bright
        assert all(f["decision"] not in ("score_only", "abstained") for f in bright)

    def test_get_analysis_serves_held_trio_as_score_only(self, api, team, before_hold):
        a = before_hold["analysis"]
        out = api.get(
            f"/api/v1/analyses/{a['id']}",
            headers=team["physician"],
            params={"include_research": True},
        ).json()
        assert out["result_sha256"] == a["result_sha256"]
        by_cond = {f["condition_id"]: f for f in out["result"]["findings"]}
        held = by_cond["cxr.bright"]
        assert held["decision"] == "score_only"
        assert held["operating_points"] == []
        assert "decisão suspensa" in held["decision_note"]
        assert any("retirada" in w for w in out["result"]["warnings"])
        # Trio não suspenso inalterado.
        before = {f["condition_id"]: f for f in before_hold["full"]["result"]["findings"]}
        assert by_cond["cxr.dark"] == before["cxr.dark"]

    def test_new_report_has_no_verdict_for_held_trio(self, api, team, before_hold):
        a = before_hold["analysis"]
        report = api.post(f"/api/v1/analyses/{a['id']}/reports", headers=team["physician"])
        assert report.status_code == 201, report.text
        rep = report.json()
        content = api.get(f"/api/v1/reports/{rep['id']}", headers=team["physician"]).json()
        text = str(content)
        assert "decisão suspensa" in text
        fhir = api.get(f"/api/v1/reports/{rep['id']}/fhir", headers=team["physician"])
        assert fhir.status_code == 200, fhir.text
        assert "decision.positive" not in fhir.text
        assert "decision.below_threshold" not in fhir.text
        assert "decision.score_only" in fhir.text

    def test_report_made_before_hold_is_not_emitted_nor_signed(self, api, team, before_hold):
        rep = before_hold["report"]
        for kind in ("pdf", "sr", "fhir"):
            r = api.get(f"/api/v1/reports/{rep['id']}/{kind}", headers=team["physician"])
            assert r.status_code == 409, (kind, r.text)
            assert "suspensa" in r.json()["detail"]
        signed = api.post(
            f"/api/v1/reports/{rep['id']}/sign",
            headers=team["physician"],
            json={
                "report_sha256": rep["report_sha256"],
                "result_sha256": rep["result_sha256"],
                "password": PASSWORD,
            },
        )
        assert signed.status_code == 409, signed.text

    def test_csv_withdraws_held_decision(self, api, team, platform, before_hold):
        _shadow(platform, True)
        rows = _csv(api.get("/api/v1/shadow/export", headers=team["researcher"]))
        mine = [r for r in rows if r["analysis_id"] == before_hold["analysis"]["id"]]
        held = [r for r in mine if r["condition_id"] == "cxr.bright"]
        free = [r for r in mine if r["condition_id"] != "cxr.bright"]
        assert held
        assert free
        assert {r["decision"] for r in held} == {"score_only"}
        assert all("decisão suspensa" in r["decision_held"] for r in held)
        assert all(r["raw_score"] for r in held)
        assert all(r["decision_held"] == "" for r in free)


@pytest.mark.requirement("REQ-195")
class TestExportScope:
    def test_export_requires_shadow_mode(self, api, team, platform, cr_dataset):
        _run(api, team, platform, _study(api, team, cr_dataset))
        response = api.get("/api/v1/shadow/export", headers=team["researcher"])
        assert response.status_code == 409, response.text
        assert response.json()["error"] == "ShadowModeOffError"
        assert not _events(api, team, "shadow.exported")

    def test_since_until_filter_requested_at_and_are_audited(self, api, team, platform, cr_dataset):
        from radiologyai.persistence.models import Analysis

        first = _run(api, team, platform, _study(api, team, cr_dataset))
        second = _run(api, team, platform, _study(api, team, cr_dataset))
        with platform.db.session() as session:
            t1 = session.get(Analysis, first["id"]).requested_at
            row = session.get(Analysis, second["id"])
            row.requested_at = t1 + timedelta(hours=2)
            t2 = row.requested_at
        _shadow(platform, True)
        mid = (t1 + timedelta(hours=1)).isoformat()
        after = _csv(
            api.get("/api/v1/shadow/export", headers=team["researcher"], params={"since": mid})
        )
        assert {r["analysis_id"] for r in after} == {second["id"]}
        before = _csv(
            api.get("/api/v1/shadow/export", headers=team["researcher"], params={"until": mid})
        )
        assert {r["analysis_id"] for r in before} == {first["id"]}
        # until é exclusivo; since inclusivo.
        exact = _csv(
            api.get(
                "/api/v1/shadow/export",
                headers=team["researcher"],
                params={"since": t2.isoformat(), "until": (t2 + timedelta(seconds=1)).isoformat()},
            )
        )
        assert {r["analysis_id"] for r in exact} == {second["id"]}
        events = _events(api, team, "shadow.exported")
        assert events[0]["payload"]["filter"] == {
            "field": "requested_at",
            "since": datetime.fromisoformat(mid).isoformat(),
            "until": None,
        }
        assert events[1]["payload"]["filter"]["until"] == datetime.fromisoformat(mid).isoformat()

    @pytest.mark.parametrize(
        "params",
        [
            {"since": "2026-11-28T10:00:00"},  # sem fuso
            {"since": "2026-11-28T10:00:00Z", "until": "2026-11-28T09:00:00Z"},
            {"since": "ontem"},
        ],
    )
    def test_bad_bounds_are_rejected(self, api, team, platform, params):
        _shadow(platform, True)
        response = api.get("/api/v1/shadow/export", headers=team["researcher"], params=params)
        assert response.status_code == 422, response.text
        assert not _events(api, team, "shadow.exported")


@pytest.mark.requirement("REQ-195")
def test_redacted_analysis_has_no_error_text(api, team, platform, cr_dataset):
    from radiologyai.persistence.models import Analysis

    created = _run(api, team, platform, _study(api, team, cr_dataset))
    with platform.db.session() as session:
        session.get(Analysis, created["id"]).error = "modelo x: abstenção por QA (teste)"
    _shadow(platform, True)
    out = api.get(f"/api/v1/analyses/{created['id']}", headers=team["physician"]).json()
    assert out["shadow"] is True
    assert out["error"] is None
    team_view = api.get(f"/api/v1/analyses/{created['id']}", headers=team["researcher"]).json()
    assert team_view["error"] == "modelo x: abstenção por QA (teste)"
