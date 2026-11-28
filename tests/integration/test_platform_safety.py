"""Correções da revisão adversarial: saída exploratória, RBAC, laudo, concorrência, auditoria."""

# ruff: noqa: ARG001, ARG005 - fixtures usadas pelo efeito colateral; dublês com assinatura fixa

from __future__ import annotations

import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from platform_support import HAS_PLATFORM, PASSWORD, dicom_bytes

pytestmark = pytest.mark.skipif(not HAS_PLATFORM, reason="requer os extras [api,platform]")


def _study(api, headers, cr_dataset):
    response = api.post(
        "/api/v1/studies",
        headers=headers["technician"],
        files=[("files", ("a.dcm", dicom_bytes(cr_dataset), "application/dicom"))],
    )
    assert response.status_code == 201, response.text
    return response.json()


def _analysis(api, headers, platform, study, role="physician", **body):
    from radiologyai.worker.queue import run_one

    created = api.post(f"/api/v1/studies/{study['id']}/analyses", headers=headers[role], json=body)
    assert created.status_code == 202, created.text
    run_one(platform)
    return api.get(f"/api/v1/analyses/{created.json()['id']}", headers=headers["physician"]).json()


@pytest.fixture
def analysis(api, headers, platform, cr_dataset):
    return _analysis(api, headers, platform, _study(api, headers, cr_dataset))


def _review_all(api, headers, analysis, action="accept", **extra):
    for idx in analysis["finding_indices"]:
        r = api.post(
            f"/api/v1/analyses/{analysis['id']}/reviews",
            headers=headers["physician"],
            json={
                "finding_idx": idx,
                "action": action,
                "result_sha256": analysis["result_sha256"],
                **extra,
            },
        )
        assert r.status_code == 201, r.text


def _report(api, headers, analysis):
    r = api.post(f"/api/v1/analyses/{analysis['id']}/reports", headers=headers["physician"])
    assert r.status_code == 201, r.text
    return r.json()


def _sign(api, headers, rep, password=PASSWORD):
    return api.post(
        f"/api/v1/reports/{rep['id']}/sign",
        headers=headers["physician"],
        json={
            "report_sha256": rep["report_sha256"],
            "result_sha256": rep["result_sha256"],
            "password": password,
        },
    )


# --------------------------------------------------------------------------- #
# Saída exploratória (máscaras e medidas) fora da visão padrão, do laudo e do PACS
# --------------------------------------------------------------------------- #
@pytest.mark.requirement("REQ-169")
class TestResearchArtifactsNeverLeak:
    def test_default_view_hides_research_masks_and_measurements(self, api, headers, analysis):
        for role in ("viewer", "technician", "physician", "admin"):
            body = api.get(f"/api/v1/analyses/{analysis['id']}", headers=headers[role]).json()
            result = body["result"]
            assert [s["name"] for s in result["segmentations"]] == ["bright_region"]
            assert result["measurements"] == []
            assert body["segmentation_indices"] == [0]
            assert body["hidden_research_artifacts"] == 2  # dark_region + dark_fraction

    def test_research_view_marks_research_artifacts(self, api, headers, analysis):
        url = f"/api/v1/analyses/{analysis['id']}?include_research=true"
        body = api.get(url, headers=headers["physician"]).json()
        assert [s["name"] for s in body["result"]["segmentations"]] == [
            "bright_region",
            "dark_region",
        ]
        assert body["research_segmentation_indices"] == [1]
        assert body["research_measurement_names"] == ["dark_fraction"]

    def test_research_mask_frame_requires_research_mode(self, api, headers, analysis):
        base = f"/api/v1/analyses/{analysis['id']}/segmentations"
        assert api.get(f"{base}/0/frames/0", headers=headers["viewer"]).status_code == 200
        for role in ("viewer", "technician"):
            denied = api.get(f"{base}/1/frames/0?include_research=true", headers=headers[role])
            assert denied.status_code == 403
        assert api.get(f"{base}/1/frames/0", headers=headers["physician"]).status_code == 403
        ok = api.get(f"{base}/1/frames/0?include_research=true", headers=headers["physician"])
        assert ok.status_code == 200

    def test_report_offers_and_renders_only_linked_segmentations(self, api, headers, analysis):
        _review_all(api, headers, analysis)
        rep = _report(api, headers, analysis)
        assert [s["idx"] for s in rep["content"]["segmentations"]] == [0]
        assert "seg/0" in rep["downloads"]
        assert "seg/1" not in rep["downloads"]
        assert _sign(api, headers, rep).status_code == 200
        refused = api.get(f"/api/v1/reports/{rep['id']}/seg/1", headers=headers["viewer"])
        assert refused.status_code == 409
        assert "não acompanha este laudo" in refused.json()["detail"]

    def test_rejected_finding_takes_its_segmentation_out(self, api, headers, analysis):
        _review_all(api, headers, analysis, action="reject")
        rep = _report(api, headers, analysis)
        assert rep["content"]["segmentations"] == []
        assert not any(k.startswith("seg/") for k in rep["downloads"])


class _Stow:
    """STOW-RS falso: confirma tudo, ou recusa os SEG (202 + FailedSOPSequence)."""

    def __init__(self, reject_seg: bool) -> None:
        self.reject_seg = reject_seg
        self.stored: list[str] = []

    def handler(self):
        stow = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                return

            def do_POST(self):  # noqa: N802
                import pydicom

                raw = b""
                if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
                    while True:
                        size = int(self.rfile.readline().strip(), 16)
                        if size == 0:
                            self.rfile.readline()
                            break
                        raw += self.rfile.read(size)
                        self.rfile.readline()
                else:
                    raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                boundary = self.headers["Content-Type"].split("boundary=")[1].split(";")[0]
                ok, failed = [], []
                for chunk in raw.split(f"--{boundary.strip(chr(34))}".encode()):
                    if b"\r\n\r\n" not in chunk:
                        continue
                    payload = chunk.split(b"\r\n\r\n", 1)[1].rstrip(b"\r\n")
                    if not payload:
                        continue
                    ds = pydicom.dcmread(io.BytesIO(payload))
                    item = {
                        "00081150": {"vr": "UI", "Value": [str(ds.SOPClassUID)]},
                        "00081155": {"vr": "UI", "Value": [str(ds.SOPInstanceUID)]},
                    }
                    if stow.reject_seg and ds.Modality == "SEG":
                        failed.append({**item, "00081197": {"vr": "US", "Value": [272]}})
                    else:
                        stow.stored.append(str(ds.Modality))
                        ok.append(item)
                body = {}
                if ok:
                    body["00081199"] = {"vr": "SQ", "Value": ok}
                if failed:
                    body["00081198"] = {"vr": "SQ", "Value": failed}
                data = json.dumps(body).encode()
                self.send_response(202 if failed else 200)
                self.send_header("Content-Type", "application/dicom+json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        return Handler


@pytest.fixture(params=[False, True], ids=["confirma", "recusa-seg"])
def stow(request, platform):
    fake = _Stow(reject_seg=request.param)
    server = ThreadingHTTPServer(("127.0.0.1", 0), fake.handler())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    platform.settings.__dict__["dicomweb_url"] = f"http://127.0.0.1:{server.server_address[1]}"
    yield fake
    server.shutdown()


@pytest.mark.requirement("REQ-169")
@pytest.mark.requirement("REQ-178")
def test_export_sends_only_linked_seg_and_never_reports_partial_success(
    api, headers, analysis, stow
):
    _review_all(api, headers, analysis)
    rep = _report(api, headers, analysis)
    assert _sign(api, headers, rep).status_code == 200
    response = api.post(f"/api/v1/reports/{rep['id']}/export", headers=headers["physician"])
    events = api.get("/api/v1/audit", headers=headers["admin"]).json()["items"]
    if stow.reject_seg:
        assert response.status_code == 502, response.text
        assert "exportação parcial" in response.json()["detail"]
        partial = [e for e in events if e["event"] == "report.export_partial"]
        assert len(partial) == 1
        assert partial[0]["payload"]["failed"]
        assert stow.stored == ["SR"]
    else:
        assert response.status_code == 200, response.text
        assert len(response.json()["stored_sop_instance_uids"]) == 2  # SR + 1 SEG (bright)
        assert sorted(stow.stored) == ["SEG", "SR"]
        assert any(e["event"] == "report.exported" for e in events)


# --------------------------------------------------------------------------- #
# RBAC de modelos exploratórios pedidos explicitamente
# --------------------------------------------------------------------------- #
@pytest.mark.requirement("REQ-170")
def test_explicit_research_model_requires_research_mode(api, headers, platform, cr_dataset):
    study = _study(api, headers, cr_dataset)
    url = f"/api/v1/studies/{study['id']}/analyses"
    body = {"model_ids": ["fake-research-seg"], "include_research": False}
    denied = api.post(url, headers=headers["technician"], json=body)
    assert denied.status_code == 403
    assert "fake-research-seg" in denied.json()["detail"]
    ok = api.post(url, headers=headers["physician"], json=body)
    assert ok.status_code == 202
    assert ok.json()["include_research"] is True  # gravada como análise de modo pesquisa
    main = api.post(url, headers=headers["technician"], json={"model_ids": ["fake-pixel-mean"]})
    assert main.status_code == 202


# --------------------------------------------------------------------------- #
# Laudo só de análise com modelo concluído
# --------------------------------------------------------------------------- #
@pytest.mark.requirement("REQ-171")
def test_no_report_when_no_model_completed(api, headers, platform, cr_dataset):
    analysis = _analysis(
        api, headers, platform, _study(api, headers, cr_dataset), model_ids=["fake-planned"]
    )
    assert analysis["state"] == "completed"
    assert analysis["result"]["status"] in ("abstained", "failed")
    refused = api.post(f"/api/v1/analyses/{analysis['id']}/reports", headers=headers["physician"])
    assert refused.status_code == 409
    assert "nenhum modelo concluiu" in refused.json()["detail"]


@pytest.mark.requirement("REQ-171")
def test_report_carries_result_status_and_completed_models(api, headers, analysis):
    _review_all(api, headers, analysis)
    content = _report(api, headers, analysis)["content"]
    assert content["result_status"] == "completed"
    assert content["completed_models"] == 1


# --------------------------------------------------------------------------- #
# Nível reconferido no laudo
# --------------------------------------------------------------------------- #
@pytest.mark.requirement("REQ-173")
def test_level_demoted_after_analysis_leaves_new_report(api, headers, platform, analysis):
    from radiologyai.catalog.catalog import Catalog
    from radiologyai.catalog.schema import EvidenceLevel, LevelEntry, LevelsFile

    _review_all(api, headers, analysis)
    old = platform.engine.catalog
    demoted = LevelsFile(
        protocol_id="teste-2",
        evidence_digest="1" * 64,
        entries=tuple(
            LevelEntry(
                condition_id=e.condition_id,
                model_id=e.model_id,
                label=e.label,
                level=EvidenceLevel.D,
                visibility="research",
                basis=(),
                refuted_by_measurement=True,
            )
            for e in old.entries()
        ),
    )
    platform.engine.catalog = Catalog(old.conditions(), demoted)
    content = _report(api, headers, analysis)["content"]
    assert content["findings"] == []
    assert content["demoted_findings"] == 1
    assert content["segmentations"] == []
    assert any("hoje é exploratório" in w for w in content["warnings"])


# --------------------------------------------------------------------------- #
# Decisão do médico nos documentos
# --------------------------------------------------------------------------- #
@pytest.mark.requirement("REQ-172")
def test_amended_decision_is_the_primary_value_in_pdf_fhir_and_sr(api, headers, analysis):
    import pydicom
    from pypdf import PdfReader

    _review_all(
        api,
        headers,
        analysis,
        action="amend",
        amended_decision="negative",
        amended_text="sem achado à revisão",
    )
    rep = _report(api, headers, analysis)
    assert _sign(api, headers, rep).status_code == 200
    (finding,) = rep["content"]["findings"]
    assert finding["physician_decision"] == "negative"
    fhir = api.get(f"/api/v1/reports/{rep['id']}/fhir", headers=headers["viewer"]).json()
    obs = fhir["contained"][1]
    assert obs["valueCodeableConcept"]["coding"][0]["code"] == "physician-decision.negative"
    suggestion = [
        c for c in obs["component"] if c["code"]["coding"][0]["code"] == "model-suggestion"
    ]
    assert suggestion
    assert "sugestão do modelo" in suggestion[0]["valueCodeableConcept"]["text"]
    pdf = api.get(f"/api/v1/reports/{rep['id']}/pdf", headers=headers["viewer"]).content
    text = " ".join("".join(p.extract_text() for p in PdfReader(io.BytesIO(pdf)).pages).split())
    assert "decisão do médico: emendado pelo médico: negativo segundo o médico" in text
    sr = api.get(f"/api/v1/reports/{rep['id']}/sr", headers=headers["viewer"]).content
    ds = pydicom.dcmread(io.BytesIO(sr))
    values = []

    def walk(seq):
        for item in seq:
            if "ConceptCodeSequence" in item:
                code = item.ConceptCodeSequence[0]
                values.append(str(code.get("CodeValue") or code.get("LongCodeValue")))
            if "ContentSequence" in item:
                walk(item.ContentSequence)

    walk(ds.ContentSequence)
    assert "physician-decision.negative" in values


@pytest.mark.requirement("REQ-172")
def test_rejected_finding_has_no_value_and_does_not_count(api, headers, analysis):
    _review_all(api, headers, analysis, action="reject")
    rep = _report(api, headers, analysis)
    fhir = api.get(f"/api/v1/reports/{rep['id']}/fhir", headers=headers["viewer"]).json()
    obs = fhir["contained"][1]
    assert "valueCodeableConcept" not in obs
    assert obs["dataAbsentReason"]["coding"][0]["code"] == "physician-decision.rejected"
    assert "0 achado(s) sugerido(s)" in fhir["conclusion"]
    assert "1 sugestão(ões) rejeitada(s)" in fhir["conclusion"]


# --------------------------------------------------------------------------- #
# Concorrência e auditoria atômica
# --------------------------------------------------------------------------- #
@pytest.mark.requirement("REQ-176")
def test_concurrent_signatures_only_one_wins(api, headers, platform, analysis):
    from radiologyai.persistence.models import User
    from radiologyai.reporting.service import ReportStateError, sign_report

    _review_all(api, headers, analysis)
    rep = _report(api, headers, analysis)
    with platform.db.session() as session:
        users = [u for u in session.query(User).all() if u.role in ("physician", "admin")]
    barrier = threading.Barrier(len(users))
    outcomes: list[str] = []

    def attempt(user):
        barrier.wait()
        try:
            sign_report(
                platform,
                rep["id"],
                user_id=user.id,
                username=user.username,
                report_sha256=rep["report_sha256"],
                result_sha256=rep["result_sha256"],
            )
            outcomes.append("ok")
        except ReportStateError:
            outcomes.append("refused")

    threads = [threading.Thread(target=attempt, args=(u,)) for u in users]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(outcomes) == ["ok", "refused"]
    events = api.get(
        "/api/v1/audit", headers=headers["admin"], params={"event": "report.signed"}
    ).json()["items"]
    assert len(events) == 1
    final = api.get(f"/api/v1/reports/{rep['id']}", headers=headers["viewer"]).json()
    assert final["integrity"]["signature_valid"] is True
    assert final["signature_sha256"] == events[0]["payload"]["signature_sha256"]


@pytest.mark.requirement("REQ-176")
def test_sign_and_audit_commit_together(api, headers, platform, analysis, monkeypatch):
    import radiologyai.persistence.audit as audit

    _review_all(api, headers, analysis)
    rep = _report(api, headers, analysis)

    def broken(*args, **kwargs):
        raise audit.AuditError("cadeia indisponível")

    monkeypatch.setattr(audit, "_append_in", broken)
    try:
        response = _sign(api, headers, rep)
    except audit.AuditError:
        response = None
    assert response is None or response.status_code >= 500
    monkeypatch.undo()
    after = api.get(f"/api/v1/reports/{rep['id']}", headers=headers["viewer"]).json()
    assert after["signed"] is False  # a assinatura não ficou gravada sem o evento


@pytest.mark.requirement("REQ-176")
def test_result_of_recovered_analysis_never_flips_back_to_completed(
    api, headers, platform, cr_dataset
):
    from radiologyai.persistence.models import Analysis
    from radiologyai.worker.queue import claim_next, recover_interrupted
    from radiologyai.worker.runner import AnalysisJobError, execute, persist_result

    study = _study(api, headers, cr_dataset)
    created = api.post(
        f"/api/v1/studies/{study['id']}/analyses", headers=headers["physician"], json={}
    ).json()
    dead = "host-que-nao-existe:1:abc"  # outro host, sem lease válida? started_at agora
    platform.settings.__dict__["worker_lease_seconds"] = 0.000001
    assert claim_next(platform, dead) == created["id"]
    result = execute(platform, created["id"])
    assert recover_interrupted(platform) == 1  # lease vencida: dado como morto
    with pytest.raises(AnalysisJobError, match="descartado"):
        persist_result(platform, created["id"], result, worker_id=dead)
    with platform.db.session() as session:
        row = session.get(Analysis, created["id"])
        assert (row.state, row.result_json) == ("failed", None)


@pytest.mark.requirement("REQ-176")
def test_live_worker_on_this_host_is_not_recovered(api, headers, platform, cr_dataset):
    import os
    import socket

    from radiologyai.persistence.models import Analysis
    from radiologyai.worker.queue import claim_next, recover_interrupted

    study = _study(api, headers, cr_dataset)
    created = api.post(
        f"/api/v1/studies/{study['id']}/analyses", headers=headers["physician"], json={}
    ).json()
    live = f"{socket.gethostname()[:40]}:{os.getppid()}:viz"  # processo pai: vivo
    assert claim_next(platform, live) == created["id"]
    assert recover_interrupted(platform) == 0
    with platform.db.session() as session:
        assert session.get(Analysis, created["id"]).state == "running"


# --------------------------------------------------------------------------- #
# Reautenticação na assinatura e limitador de login
# --------------------------------------------------------------------------- #
@pytest.mark.requirement("REQ-177")
def test_wrong_sign_password_is_audited_throttled_and_revokes(api, headers, analysis):
    _review_all(api, headers, analysis)
    rep = _report(api, headers, analysis)
    codes = [_sign(api, headers, rep, password="errada-123456").status_code for _ in range(5)]
    assert codes[:4] == [401, 401, 401, 401]
    # na quinta falha as sessões foram revogadas: o token roubado deixou de valer
    assert codes[4] in (401,)
    assert api.get("/api/v1/auth/me", headers=headers["physician"]).status_code == 401
    events = api.get(
        "/api/v1/audit", headers=headers["admin"], params={"event": "auth.reauth_failed"}
    ).json()["items"]
    assert len(events) == 5
    assert events[-1]["payload"]["sessions_revoked"] >= 1


@pytest.mark.requirement("REQ-177")
def test_login_lockout_is_per_client_and_unknown_names_not_stored(api, platform, make_user):
    from radiologyai.persistence.models import AuditEvent
    from radiologyai.persistence.users import (
        AuthenticationError,
        LoginThrottledError,
        authenticate,
    )

    make_user("physician", username="dr.silva")
    for _ in range(5):
        with pytest.raises(AuthenticationError):
            authenticate(platform.db, "dr.silva", "errada", client="10.0.0.66")
    with pytest.raises(LoginThrottledError):
        authenticate(platform.db, "dr.silva", "errada", client="10.0.0.66")
    # o médico, de outra estação, continua entrando
    assert authenticate(platform.db, "dr.silva", PASSWORD, client="10.0.0.10").username
    with pytest.raises(AuthenticationError):
        authenticate(platform.db, "minha-senha-secreta-123", "x", client="10.0.0.10")
    with platform.db.session() as session:
        raw = " ".join(
            f"{e.actor_username} {e.payload_json}" for e in session.query(AuditEvent).all()
        )
    assert "minha-senha-secreta-123" not in raw


# --------------------------------------------------------------------------- #
# Endpoint legado
# --------------------------------------------------------------------------- #
@pytest.mark.requirement("REQ-174")
def test_legacy_analyze_without_research_mode_returns_no_decisions(
    api, headers, platform, cr_dataset, monkeypatch
):
    import radiologyai.api.routers.analyze as legacy

    fake = [{"label": "Effusion", "score": 0.9, "band": "achado_provavel"}]
    monkeypatch.setattr(legacy, "_run_inference", lambda *a, **k: list(fake))
    for role, shown in (("technician", 0), ("physician", 1)):
        response = api.post(
            "/api/v1/analyze",
            headers=headers[role],
            files={"file": ("a.dcm", dicom_bytes(cr_dataset), "application/dicom")},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert len(body["findings"]) == shown
        assert body["hidden_research_findings"] == 1 - shown
        assert all(f["evidence_level"] == "D" for f in body["findings"])
    events = api.get(
        "/api/v1/audit", headers=headers["admin"], params={"event": "legacy.analyze"}
    ).json()["items"]
    assert [e["payload"]["research_view"] for e in events] == [False, True]
    assert api.get("/api/v1/audit/verify", headers=headers["admin"]).json()["ok"] is True


# --------------------------------------------------------------------------- #
# Assinatura com chave do servidor e âncora externa da cadeia
# --------------------------------------------------------------------------- #
@pytest.mark.requirement("REQ-179")
def test_forged_unkeyed_signature_is_invalid(api, headers, platform, analysis):
    import hashlib

    from radiologyai.persistence.models import Report, User, utcnow
    from radiologyai.reporting.content import canonical_bytes

    _review_all(api, headers, analysis)
    rep = _report(api, headers, analysis)
    with platform.db.session() as session:
        doctor = next(u for u in session.query(User).all() if u.role == "physician")
        row = session.get(Report, rep["id"])
        when = utcnow()
        row.signed_by, row.signed_at = doctor.id, when
        row.signature_sha256 = hashlib.sha256(
            canonical_bytes(
                {
                    "report_id": row.id,
                    "report_sha256": row.report_sha256,
                    "result_sha256": row.result_sha256,
                    "signer_id": doctor.id,
                    "signed_at": when.isoformat(),
                }
            )
        ).hexdigest()
    forged = api.get(f"/api/v1/reports/{rep['id']}", headers=headers["viewer"]).json()
    assert forged["integrity"]["signature_valid"] is False
    assert api.get(f"/api/v1/reports/{rep['id']}/sr", headers=headers["viewer"]).status_code == 409


@pytest.mark.requirement("REQ-179")
def test_deleting_newest_audit_events_fails_verification(api, headers, platform, analysis):
    from sqlalchemy import delete, func, select

    from radiologyai.persistence.models import AuditEvent

    first = api.get("/api/v1/audit/verify", headers=headers["admin"]).json()
    assert first["ok"] is True
    assert first["anchored_seq"] == first["events"]
    with platform.db.session() as session:
        last = session.execute(select(func.max(AuditEvent.seq))).scalar_one()
        session.execute(delete(AuditEvent).where(AuditEvent.seq >= last - 1))
    status = api.get("/api/v1/audit/verify", headers=headers["admin"]).json()
    assert status["ok"] is False
    assert "encurtada" in status["detail"]


@pytest.mark.requirement("REQ-168")
def test_catalog_api_applies_manifest_research_restriction(api, headers, platform):
    import yaml
    from platform_support import manifest

    from radiologyai.analysis.registry import Registry
    from radiologyai.catalog.catalog import Catalog
    from radiologyai.catalog.schema import EvidenceLevel, LevelEntry, LevelsFile

    registry = platform.engine.registry
    data = manifest("fake-restricted")
    data["visibility_default"] = "research"
    (registry.manifest_dir / "fake-restricted.yaml").write_text(
        yaml.safe_dump(data), encoding="utf-8"
    )
    platform.engine.registry = Registry(
        manifest_dir=registry.manifest_dir,
        weights_dir=registry.weights_dir,
        settings=registry.settings,
    )
    old = platform.engine.catalog
    entries = (
        *old.levels_file.entries,
        LevelEntry(
            condition_id="cxr.bright",
            model_id="fake-restricted",
            label="bright",
            level=EvidenceLevel.C,
            visibility="main",
            basis=("registro-teste",),
        ),
    )
    platform.engine.catalog = Catalog(
        old.conditions(),
        LevelsFile(protocol_id="t", evidence_digest="2" * 64, entries=entries),
    )
    body = api.get("/api/v1/catalog", headers=headers["viewer"]).json()
    bright = next(c for c in body["conditions"] if c["id"] == "cxr.bright")
    restricted = next(e for e in bright["entries"] if e["model_id"] == "fake-restricted")
    assert restricted["level"] == "C"
    assert restricted["visibility"] == "research"
    assert restricted["restricted_by_manifest"] is True
