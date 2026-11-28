"""Contrato DICOMweb (QIDO/WADO/STOW) contra um servidor falso local, sem rede externa.

O servidor falso fala HTTP de verdade (``http.server``) e devolve respostas no
formato do padrão (DICOM JSON, multipart/related). Ele devolve de propósito
nome e ID do paciente na busca, para provar que a fachada não os repassa.
"""

from __future__ import annotations

import importlib.util
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

HAS = all(
    importlib.util.find_spec(m)
    for m in ("dicomweb_client", "requests", "sqlalchemy", "alembic", "argon2")
)
pytestmark = pytest.mark.skipif(not HAS, reason="requer o extra [platform]")

BOUNDARY = "fronteira-de-teste"


def _dcm_bytes(ds) -> bytes:
    import pydicom

    buf = io.BytesIO()
    pydicom.dcmwrite(buf, ds, enforce_file_format=True)
    return buf.getvalue()


class FakePacs:
    def __init__(self, study_datasets):
        self.studies = {str(study_datasets[0].StudyInstanceUID): study_datasets}
        self.stored: list[bytes] = []
        self.requests: list[str] = []
        self.reject_uids: set[str] = set()  # STOW: recusados em FailedSOPSequence (202)

    def handler(self):
        pacs = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):  # silencioso
                return

            def _send(self, code, body, ctype):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):  # noqa: N802
                pacs.requests.append(f"GET {self.path}")
                path = self.path.split("?")[0]
                if path == "/studies":
                    items = []
                    for uid, dss in pacs.studies.items():
                        items.append(
                            {
                                "0020000D": {"vr": "UI", "Value": [uid]},
                                "00100010": {
                                    "vr": "PN",
                                    "Value": [{"Alphabetic": "FULANO^DE^TAL"}],
                                },
                                "00100020": {"vr": "LO", "Value": ["ID-REAL-123"]},
                                "00081030": {"vr": "LO", "Value": ["RX TORAX PA"]},
                                "00080061": {"vr": "CS", "Value": ["CR"]},
                                "00201208": {"vr": "IS", "Value": [len(dss)]},
                                "00080020": {"vr": "DA", "Value": ["20260315"]},
                            }
                        )
                    self._send(200, json.dumps(items).encode(), "application/dicom+json")
                    return
                parts = path.strip("/").split("/")
                if (
                    len(parts) == 3
                    and parts[0] == "studies"
                    and parts[2] == "instances"
                    and parts[1] in pacs.studies
                ):
                    items = [
                        {"00080018": {"vr": "UI", "Value": [str(ds.SOPInstanceUID)]}}
                        for ds in pacs.studies[parts[1]]
                    ]
                    self._send(200, json.dumps(items).encode(), "application/dicom+json")
                    return
                if len(parts) == 2 and parts[0] == "studies" and parts[1] in pacs.studies:
                    body = b""
                    for ds in pacs.studies[parts[1]]:
                        body += (
                            f"--{BOUNDARY}\r\nContent-Type: application/dicom\r\n\r\n".encode()
                            + _dcm_bytes(ds)
                            + b"\r\n"
                        )
                    body += f"--{BOUNDARY}--\r\n".encode()
                    self._send(
                        200,
                        body,
                        f'multipart/related; type="application/dicom"; boundary={BOUNDARY}',
                    )
                    return
                self._send(404, b"{}", "application/json")

            def do_POST(self):  # noqa: N802
                pacs.requests.append(f"POST {self.path}")
                if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
                    raw = b""
                    while True:
                        size = int(self.rfile.readline().strip(), 16)
                        if size == 0:
                            self.rfile.readline()
                            break
                        raw += self.rfile.read(size)
                        self.rfile.readline()
                else:
                    raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                ctype = self.headers.get("Content-Type", "")
                boundary = ctype.split("boundary=")[1].split(";")[0].strip('"')
                import pydicom

                referenced, failed = [], []
                for chunk in raw.split(f"--{boundary}".encode()):
                    if b"\r\n\r\n" in chunk:
                        payload = chunk.split(b"\r\n\r\n", 1)[1].rstrip(b"\r\n")
                        if payload:
                            ds = pydicom.dcmread(io.BytesIO(payload))
                            item = {
                                "00081150": {"vr": "UI", "Value": [str(ds.SOPClassUID)]},
                                "00081155": {"vr": "UI", "Value": [str(ds.SOPInstanceUID)]},
                            }
                            if str(ds.SOPInstanceUID) in pacs.reject_uids:
                                failed.append({**item, "00081197": {"vr": "US", "Value": [272]}})
                            else:
                                pacs.stored.append(payload)
                                referenced.append(item)
                body = {}
                if referenced:
                    body["00081199"] = {"vr": "SQ", "Value": referenced}
                if failed:
                    body["00081198"] = {"vr": "SQ", "Value": failed}
                code = 202 if failed else 200
                self._send(code, json.dumps(body).encode(), "application/dicom+json")

        return Handler


@pytest.fixture
def pacs(cr_dataset):
    fake = FakePacs([cr_dataset])
    server = ThreadingHTTPServer(("127.0.0.1", 0), fake.handler())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    fake.url = f"http://127.0.0.1:{server.server_address[1]}"
    yield fake
    server.shutdown()


@pytest.mark.requirement("REQ-162")
def test_qido_search_drops_patient_identifiers(pacs):
    from radiologyai.interop.dicomweb import DicomwebGateway

    found = DicomwebGateway(pacs.url).search_studies(modality="CR")
    assert len(found) == 1
    dumped = json.dumps([s.model_dump() for s in found])
    assert "FULANO" not in dumped
    assert "ID-REAL-123" not in dumped
    assert found[0].study_year == "2026"
    assert found[0].modalities_in_study == ("CR",)
    assert any(r.startswith("GET /studies?") for r in pacs.requests)


@pytest.mark.requirement("REQ-162")
def test_wado_retrieve_returns_datasets(pacs, cr_dataset):
    from radiologyai.interop.dicomweb import DicomwebGateway

    datasets = DicomwebGateway(pacs.url).retrieve_study(str(cr_dataset.StudyInstanceUID))
    assert [str(d.SOPInstanceUID) for d in datasets] == [str(cr_dataset.SOPInstanceUID)]


@pytest.mark.requirement("REQ-162")
def test_stow_sends_part10_objects(pacs, cr_dataset):
    import pydicom

    from radiologyai.interop.dicomweb import DicomwebGateway

    DicomwebGateway(pacs.url).store([cr_dataset])
    assert len(pacs.stored) == 1
    assert pydicom.dcmread(io.BytesIO(pacs.stored[0])).SOPInstanceUID == cr_dataset.SOPInstanceUID


@pytest.mark.requirement("REQ-162")
def test_unreachable_server_is_explicit_error():
    import socket

    from radiologyai.interop.dicomweb import DicomwebError, DicomwebGateway

    with socket.socket() as sock:  # porta livre e fechada: conexão recusada na hora
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with pytest.raises(DicomwebError, match="QIDO falhou"):
        DicomwebGateway(f"http://127.0.0.1:{port}", timeout=2).search_studies()


@pytest.mark.requirement("REQ-162")
def test_import_over_api_deidentifies(pacs, cr_dataset, tmp_path):
    """POST /dicomweb/import passa pela ingestão: estudo entra des-identificado."""
    from fastapi.testclient import TestClient

    from radiologyai.api import create_app
    from radiologyai.persistence.config import PlatformSettings
    from radiologyai.persistence.users import create_user, issue_session
    from radiologyai.worker.runtime import Platform

    settings = PlatformSettings(
        database_url=f"sqlite:///{tmp_path / 'p.db'}",
        store_dir=tmp_path / "store",
        deident_key="chave",
        worker_enabled=False,
        dicomweb_url=pacs.url,
    )
    platform = Platform.create(settings)
    with TestClient(create_app(platform=platform)) as client:
        user = create_user(
            platform.db,
            username="tec",
            full_name="T",
            role="technician",
            password="senha-de-teste-1",
        )
        token, _ = issue_session(platform.db, user, 1.0)
        auth = {"Authorization": f"Bearer {token}"}
        listing = client.get("/api/v1/dicomweb/studies", headers=auth)
        assert listing.status_code == 200
        assert "FULANO" not in listing.text
        imported = client.post(
            "/api/v1/dicomweb/import",
            headers=auth,
            json={"study_instance_uid": str(cr_dataset.StudyInstanceUID)},
        )
        assert imported.status_code == 201, imported.text
        body = imported.json()
        assert body["source"] == "dicomweb"
        assert body["study_uid"] != str(cr_dataset.StudyInstanceUID)
        assert body["patient_pseudo_id"].startswith("ANON-")


@pytest.mark.requirement("REQ-178")
def test_stow_outcome_reports_only_confirmed_and_flags_202_failures(pacs, cr_dataset):
    import copy

    from pydicom.uid import generate_uid

    from radiologyai.interop.dicomweb import DicomwebGateway, store_outcome

    other = copy.deepcopy(cr_dataset)
    other.SOPInstanceUID = generate_uid()
    pacs.reject_uids.add(str(other.SOPInstanceUID))
    sent = [str(cr_dataset.SOPInstanceUID), str(other.SOPInstanceUID)]
    outcome = store_outcome(DicomwebGateway(pacs.url).store([cr_dataset, other]), sent)
    assert outcome.stored == (str(cr_dataset.SOPInstanceUID),)
    assert [uid for uid, _ in outcome.failed] == [str(other.SOPInstanceUID)]
    assert "272" in outcome.failed[0][1]


@pytest.mark.requirement("REQ-178")
def test_stow_without_confirmation_is_not_success():
    from pydicom.dataset import Dataset

    from radiologyai.interop.dicomweb import store_outcome

    outcome = store_outcome(Dataset(), ["1.2.3"])
    assert outcome.stored == ()
    assert outcome.failed == (("1.2.3", "não confirmado pelo servidor"),)


@pytest.mark.requirement("REQ-178")
def test_import_over_limit_is_refused_before_download(pacs, cr_dataset, tmp_path):
    import copy

    from fastapi.testclient import TestClient
    from pydicom.uid import generate_uid

    from radiologyai.api import create_app
    from radiologyai.persistence.config import PlatformSettings
    from radiologyai.persistence.users import create_user, issue_session
    from radiologyai.worker.runtime import Platform

    second = copy.deepcopy(cr_dataset)
    second.SOPInstanceUID = generate_uid()
    pacs.studies[str(cr_dataset.StudyInstanceUID)].append(second)
    settings = PlatformSettings(
        database_url=f"sqlite:///{tmp_path / 'p.db'}",
        store_dir=tmp_path / "store",
        deident_key="chave",
        worker_enabled=False,
        dicomweb_url=pacs.url,
        max_zip_members=1,
    )
    platform = Platform.create(settings)
    with TestClient(create_app(platform=platform)) as client:
        user = create_user(
            platform.db,
            username="tec",
            full_name="T",
            role="technician",
            password="senha-de-teste-1",
        )
        token, _ = issue_session(platform.db, user, 1.0)
        response = client.post(
            "/api/v1/dicomweb/import",
            headers={"Authorization": f"Bearer {token}"},
            json={"study_instance_uid": str(cr_dataset.StudyInstanceUID)},
        )
    assert response.status_code == 413, response.text
    assert "mais de 1 instâncias" in response.json()["detail"]
    assert not any(r.startswith("GET /studies/") and r.count("/") == 2 for r in pacs.requests)
