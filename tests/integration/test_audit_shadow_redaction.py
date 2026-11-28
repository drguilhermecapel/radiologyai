"""Auditoria no modo silencioso (REQ-195): contagens derivadas da IA redigidas na leitura.

O administrador (``audit:read``) não tem ``shadow:view``. No modo silencioso a
trilha é servida sem os campos de carga que derivam da saída de IA
(``n_findings``, estado do resultado, decisões de revisão...), mas o evento
gravado e encadeado não muda: a cadeia continua íntegra.
"""

# ruff: noqa: ARG001 - fixtures usadas pelo efeito colateral

from __future__ import annotations

import json

import pytest
from platform_support import HAS_PLATFORM, PASSWORD, dicom_bytes

pytestmark = [
    pytest.mark.skipif(not HAS_PLATFORM, reason="requer os extras [api,platform]"),
    pytest.mark.requirement("REQ-195"),
]


def _shadow(platform, on: bool) -> None:
    platform.settings.__dict__["shadow_mode"] = on


@pytest.fixture
def done(api, headers, platform, cr_dataset):
    """Análise concluída, revisada e com laudo assinado (antes do piloto)."""
    from radiologyai.worker.queue import run_one

    study = api.post(
        "/api/v1/studies",
        headers=headers["technician"],
        files=[("files", ("a.dcm", dicom_bytes(cr_dataset), "application/dicom"))],
    ).json()
    created = api.post(
        f"/api/v1/studies/{study['id']}/analyses", headers=headers["technician"], json={}
    )
    assert created.status_code == 202, created.text
    assert run_one(platform) == created.json()["id"]
    a = api.get(f"/api/v1/analyses/{created.json()['id']}", headers=headers["physician"]).json()
    for idx in a["finding_indices"]:
        r = api.post(
            f"/api/v1/analyses/{a['id']}/reviews",
            headers=headers["physician"],
            json={"finding_idx": idx, "action": "accept", "result_sha256": a["result_sha256"]},
        )
        assert r.status_code == 201, r.text
    rep = api.post(f"/api/v1/analyses/{a['id']}/reports", headers=headers["physician"]).json()
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
    return a


def _items(api, headers, event):
    r = api.get("/api/v1/audit", headers=headers["admin"], params={"event": event, "limit": 500})
    assert r.status_code == 200, r.text
    return r.json()["items"]


def _stored(platform, seq):
    from radiologyai.persistence.models import AuditEvent

    with platform.db.session() as session:
        row = session.get(AuditEvent, seq)
        assert row is not None
        return json.loads(row.payload_json), row.record_hash


def test_normal_mode_serves_counts(api, headers, platform, done):
    (completed,) = _items(api, headers, "analysis.completed")
    assert (
        completed["payload"]["n_findings"]
        == len(done["result"]["findings"]) + done["hidden_research_findings"]
    )
    assert "shadow_redacted" not in completed["payload"]


def test_shadow_redacts_ai_counts_for_admin(api, headers, platform, done):
    _shadow(platform, True)
    (completed,) = _items(api, headers, "analysis.completed")
    p = completed["payload"]
    assert p["n_findings"] is None  # retido, nunca 0 ("zero achados" não é "normal")
    assert p["status"] is None
    assert p["runs"] is None
    assert p["result_sha256"] is None
    assert p["shadow_redacted"] == ["n_findings", "result_sha256", "runs", "status"]
    # Proveniência (modelos, entradas) não é saída de IA: continua servida.
    assert p["models"]
    assert p["input_sha256"]
    reviews = _items(api, headers, "review.created")
    assert reviews
    for e in reviews:
        assert e["payload"]["condition_id"] is None
        assert e["payload"]["action"] is None
        assert e["payload"]["finding_idx"] is None
    (created,) = _items(api, headers, "report.created")
    assert created["payload"]["n_findings"] is None
    assert created["payload"]["unreviewed"] is None
    assert created["payload"]["report_sha256"]  # o laudo em si não é contagem da IA
    # Eventos sem campo de IA voltam intactos.
    (login, *_) = _items(api, headers, "auth.login")
    assert "shadow_redacted" not in login["payload"]


def test_redaction_is_read_time_only_chain_still_verifies(api, headers, platform, done):
    _shadow(platform, True)
    (completed,) = _items(api, headers, "analysis.completed")
    stored, record_hash = _stored(platform, completed["seq"])
    assert (
        stored["n_findings"] == len(done["result"]["findings"]) + done["hidden_research_findings"]
    )
    assert "shadow_redacted" not in stored
    assert record_hash  # gravado intacto; só a resposta o retém
    assert completed["record_hash"] == ""
    verify = api.get("/api/v1/audit/verify", headers=headers["admin"]).json()
    assert verify["ok"] is True, verify
    # Desligar o modo silencioso devolve a trilha completa (nada foi apagado).
    _shadow(platform, False)
    (again,) = _items(api, headers, "analysis.completed")
    assert again["payload"] == stored


def test_redact_payload_does_not_mutate_and_skips_absent_fields():
    from radiologyai.api.routers.audit import redact_payload

    payload = {"n_findings": 3, "status": "ok", "models": []}
    out = redact_payload("analysis.completed", payload)
    assert payload == {"n_findings": 3, "status": "ok", "models": []}
    assert out == {
        "n_findings": None,
        "status": None,
        "models": [],
        "shadow_redacted": ["n_findings", "status"],
    }
    other = {"username": "x", "role": "admin"}
    assert redact_payload("user.created", other) is other


def _all(api, headers):
    r = api.get("/api/v1/audit", headers=headers["admin"], params={"limit": 500})
    assert r.status_code == 200, r.text
    return r.json()["items"]


def test_shadow_withholds_hashes_of_redacted_events(api, headers, platform, done):
    """O record_hash é sha256 da carga canônica: servi-lo permitiria força bruta do redigido."""
    _shadow(platform, True)
    items = _all(api, headers)
    by_seq = {e["seq"]: e for e in items}
    redacted = {e["seq"] for e in items if "shadow_redacted" in e["payload"]}
    assert redacted
    for e in items:
        if e["seq"] in redacted:
            assert e["record_hash"] == ""
        else:
            assert len(e["record_hash"]) == 64
        if e["seq"] - 1 in redacted:
            assert e["prev_hash"] == ""  # o prev_hash do seguinte é o mesmo valor
        elif e["seq"] - 1 in by_seq:
            assert e["prev_hash"] == by_seq[e["seq"] - 1]["record_hash"]
    # Predecessor redigido fora da página (paginação/filtro) também retém o prev_hash.
    completed = next(e for e in items if e["event"] == "analysis.completed")
    nxt = api.get(
        "/api/v1/audit",
        headers=headers["admin"],
        params={"after_seq": completed["seq"], "limit": 1},
    ).json()["items"]
    assert nxt[0]["prev_hash"] == ""
    # O último hash da verificação é o record_hash do último evento (aqui, redigido).
    last = items[-1]
    verify = api.get("/api/v1/audit/verify", headers=headers["admin"]).json()
    assert verify["ok"] is True
    assert ("shadow_redacted" in last["payload"]) == (verify["last_hash"] == "")
    # Fora do modo silencioso, tudo volta (nada foi alterado no banco).
    _shadow(platform, False)
    full = _all(api, headers)
    assert all(len(e["record_hash"]) == 64 for e in full)
    assert api.get("/api/v1/audit/verify", headers=headers["admin"]).json()["last_hash"]


def test_shadow_redacts_failed_error_and_review_report_fields(api, headers, platform, done):
    from radiologyai.persistence.audit import append_event

    append_event(
        platform.db,
        "analysis.failed",
        actor_username="worker",
        subject_type="analysis",
        subject_id=done["id"],
        payload={"error": "modelo x: abstenção em 3 achados"},
    )
    _shadow(platform, True)
    (failed,) = _items(api, headers, "analysis.failed")
    assert failed["payload"] == {"error": None, "shadow_redacted": ["error"]}
    assert failed["record_hash"] == ""
    verify = api.get("/api/v1/audit/verify", headers=headers["admin"]).json()
    assert verify["ok"] is True
    assert verify["last_hash"] == ""  # o último evento é o redigido
    (report,) = _items(api, headers, "report.created")
    for key in ("n_findings", "unreviewed", "demoted"):
        assert report["payload"][key] is None
        assert key in report["payload"]["shadow_redacted"]
    for review in _items(api, headers, "review.created"):
        for key in ("condition_id", "action", "amended_decision"):
            assert review["payload"][key] is None
            assert key in review["payload"]["shadow_redacted"]
