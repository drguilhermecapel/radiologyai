"""API v1 (plataforma): quadros para o visualizador, lista de trabalho, tipo de exame,
prontidão, limite de corpo e OpenAPI versionado."""

from __future__ import annotations

import copy
import importlib.util
import io
import zipfile
from pathlib import Path

import numpy as np
import pytest
from pydicom.uid import generate_uid

HAS = all(
    importlib.util.find_spec(m)
    for m in ("fastapi", "sqlalchemy", "alembic", "argon2", "httpx", "multipart", "PIL")
)
pytestmark = pytest.mark.skipif(not HAS, reason="requer os extras [api,platform,imaging]")
REPO = Path(__file__).resolve().parents[3]


def dcm(ds) -> bytes:
    import pydicom

    buf = io.BytesIO()
    pydicom.dcmwrite(buf, ds, enforce_file_format=True)
    return buf.getvalue()


@pytest.fixture
def client(tmp_path):
    from fastapi.testclient import TestClient

    from radiologyai.analysis.registry import Registry
    from radiologyai.api import create_app
    from radiologyai.catalog.catalog import Catalog
    from radiologyai.config.settings import Settings
    from radiologyai.persistence.config import PlatformSettings
    from radiologyai.persistence.users import create_user, issue_session, reset_throttle
    from radiologyai.worker.runtime import AnalysisEngine, Platform

    (tmp_path / "m").mkdir()
    engine = AnalysisEngine(
        registry=Registry(
            manifest_dir=tmp_path / "m",
            weights_dir=tmp_path / "w",
            settings=Settings(weights_dir=tmp_path / "w"),
        ),
        catalog=Catalog([], None),
    )
    settings = PlatformSettings(
        database_url=f"sqlite:///{tmp_path / 'p.db'}",
        store_dir=tmp_path / "store",
        deident_key="chave",
        worker_enabled=False,
        max_upload_bytes=3_000_000,
        frames_max_dim=256,
        frames_hard_cap=512,
    )
    platform = Platform.create(settings, engine=engine)
    reset_throttle()
    with TestClient(create_app(platform=platform)) as c:
        auth = {}
        for role in ("technician", "admin", "viewer"):
            user = create_user(
                platform.db, username=role, full_name=role, role=role, password="senha-de-teste-1"
            )
            auth[role] = {"Authorization": f"Bearer {issue_session(platform.db, user, 1.0)[0]}"}
        c.roles = auth
        c.platform = platform
        yield c


def upload(client, name, data, **form):
    return client.post(
        "/api/v1/studies",
        headers=client.roles["technician"],
        files=[("files", (name, data, "application/octet-stream"))],
        data=form,
    )


def ct_volume_zip(ct_dataset, n=4) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        series = generate_uid()
        for z in range(n):
            ds = copy.deepcopy(ct_dataset)
            ds.SOPInstanceUID = generate_uid()
            ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
            ds.SeriesInstanceUID = series
            ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
            ds.ImagePositionPatient = [0, 0, float(z) * 1.25]
            ds.InstanceNumber = z + 1
            ds.StudyDescription = "CT CHEST"
            zf.writestr(f"ct/{z}.dcm", dcm(ds))
    return buf.getvalue()


def decode(response) -> np.ndarray:
    dtype = {"uint16": "<u2", "int16": "<i2", "uint8": "u1"}[response.headers["x-frame-dtype"]]
    rows, cols = int(response.headers["x-frame-rows"]), int(response.headers["x-frame-columns"])
    arr = np.frombuffer(response.content, dtype=dtype).reshape(rows, cols)
    return arr * float(response.headers["x-rescale-slope"]) + float(
        response.headers["x-rescale-intercept"]
    )


@pytest.mark.requirement("REQ-163")
class TestFrames:
    def test_projection_downsampled_within_cap(self, client, cr_dataset):
        study = upload(client, "a.dcm", dcm(cr_dataset)).json()
        sid = study["series"][0]["id"]
        meta = client.get(f"/api/v1/series/{sid}", headers=client.roles["viewer"]).json()
        assert meta["default_max_dim"] == 256
        assert meta["max_dim_cap"] == 512
        frame = client.get(f"/api/v1/series/{sid}/frames/0", headers=client.roles["viewer"])
        assert frame.status_code == 200
        assert frame.headers["x-frame-scale"] == "2"  # 512 px -> 256
        assert (frame.headers["x-frame-rows"], frame.headers["x-frame-columns"]) == ("256", "256")
        capped = client.get(
            f"/api/v1/series/{sid}/frames/0?max_dim=100000", headers=client.roles["viewer"]
        )
        assert int(capped.headers["x-frame-rows"]) <= 512
        full = decode(capped)
        expected = cr_dataset.pixel_array.astype(np.float64)
        np.testing.assert_allclose(full, expected, atol=0.5)  # sem redução: valores exatos

    def test_ct_volume_hounsfield_preserved(self, client, ct_dataset):
        study = upload(client, "ct.zip", ct_volume_zip(ct_dataset)).json()
        (series,) = study["series"]
        assert series["kind"] == "volume"
        assert series["n_frames"] == 4
        assert study["exam_type"] == "ct_chest"
        frame = client.get(
            f"/api/v1/series/{series['id']}/frames/2?max_dim=64", headers=client.roles["viewer"]
        )
        assert frame.headers["x-frame-count"] == "4"
        hu = decode(frame)
        expected = ct_dataset.pixel_array.astype(np.float64) - 1024.0
        np.testing.assert_allclose(hu, expected, atol=0.5)
        assert hu.min() < -1000
        assert hu.max() > 1000

    def test_out_of_range_frame_404(self, client, cr_dataset):
        sid = upload(client, "a.dcm", dcm(cr_dataset)).json()["series"][0]["id"]
        assert (
            client.get(f"/api/v1/series/{sid}/frames/1", headers=client.roles["viewer"]).status_code
            == 404
        )


@pytest.mark.requirement("REQ-161")
class TestExamType:
    def test_unknown_requires_human_choice_and_is_audited(self, client, cr_dataset):
        cr_dataset.BodyPartExamined = ""
        study = upload(client, "a.dcm", dcm(cr_dataset)).json()
        assert study["exam_type"] == "unknown"
        blocked = client.post(
            f"/api/v1/studies/{study['id']}/analyses", headers=client.roles["technician"]
        )
        assert blocked.status_code == 409
        assert "humano" in blocked.json()["detail"]
        refused = client.put(
            f"/api/v1/studies/{study['id']}/exam-type",
            headers=client.roles["technician"],
            json={"exam_type": "unknown"},
        )
        assert refused.status_code == 422
        chosen = client.put(
            f"/api/v1/studies/{study['id']}/exam-type",
            headers=client.roles["technician"],
            json={"exam_type": "cxr"},
        )
        assert chosen.status_code == 200
        assert chosen.json()["exam_type"] == "cxr"
        assert chosen.json()["exam_type_source"] == "human"
        ok = client.post(
            f"/api/v1/studies/{study['id']}/analyses", headers=client.roles["technician"]
        )
        assert ok.status_code == 202
        events = client.get(
            f"/api/v1/audit?subject_id={study['id']}", headers=client.roles["admin"]
        ).json()["items"]
        assert [e["event"] for e in events] == ["study.ingested", "study.exam_type_set"]
        assert events[1]["payload"] == {"from": "unknown", "to": "cxr"}


@pytest.mark.requirement("REQ-160")
def test_worklist_filters(client, cr_dataset):
    first = upload(client, "a.dcm", dcm(cr_dataset)).json()
    cr_dataset.StudyInstanceUID = generate_uid()
    cr_dataset.SOPInstanceUID = generate_uid()
    cr_dataset.file_meta.MediaStorageSOPInstanceUID = cr_dataset.SOPInstanceUID
    upload(client, "b.dcm", dcm(cr_dataset))
    client.post(f"/api/v1/studies/{first['id']}/analyses", headers=client.roles["technician"])
    viewer = client.roles["viewer"]
    everything = client.get("/api/v1/studies", headers=viewer).json()
    assert everything["total"] == 2
    queued = client.get("/api/v1/studies?state=queued", headers=viewer).json()
    assert [s["id"] for s in queued["items"]] == [first["id"]]
    assert queued["items"][0]["latest_analysis"]["state"] == "queued"
    assert queued["items"][0]["review_status"] == "pending_review"
    none = client.get("/api/v1/studies?state=no_analysis", headers=viewer).json()
    assert none["total"] == 1
    assert none["items"][0]["review_status"] == "no_analysis"
    assert client.get("/api/v1/studies?modality=CT", headers=viewer).json()["total"] == 0
    page = client.get("/api/v1/studies?limit=1&offset=1", headers=viewer).json()
    assert len(page["items"]) == 1
    assert page["total"] == 2


@pytest.mark.requirement("REQ-152")
def test_body_limit_enforced_while_streaming(client):
    response = upload(client, "grande.dcm", b"\0" * 3_100_000)
    assert response.status_code == 413
    assert response.json()["error"] == "UploadTooLargeError"


@pytest.mark.requirement("REQ-165")
def test_readyz_reports_dependencies(client):
    body = client.get("/api/v1/readyz").json()
    assert body["database_revision"] == body["expected_revision"]
    assert body["worker"] == "disabled"
    assert body["store_writable"] is True
    assert body["ready"] is True
    assert body["deident_key_configured"] is True
    assert client.get("/api/v1/healthz").json() == {"status": "ok"}


@pytest.mark.requirement("REQ-164")
def test_committed_openapi_matches_generated(client):
    """docs/OPENAPI_SPEC.yaml é gerado da aplicação (`radiologyai openapi`) e versionado."""
    import yaml

    from radiologyai.api.openapi import export_openapi

    committed = yaml.safe_load((REPO / "docs" / "OPENAPI_SPEC.yaml").read_text(encoding="utf-8"))
    assert committed == yaml.safe_load(export_openapi(client.app))


@pytest.mark.requirement("REQ-164")
def test_every_json_response_is_typed(client):
    spec = client.app.openapi()
    for path, ops in spec["paths"].items():
        for method, op in ops.items():
            ok = (
                op["responses"].get("200")
                or op["responses"].get("201")
                or op["responses"].get("202")
            )
            if ok is None:  # 204 sem corpo
                continue
            content = ok.get("content", {})
            if "application/json" in content:
                schema = content["application/json"]["schema"]
                assert schema not in ({}, {"type": "object"}), f"{method} {path} sem modelo"
