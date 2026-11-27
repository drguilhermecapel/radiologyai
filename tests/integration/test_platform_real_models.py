"""Plataforma com pesos REAIS: RX de tórax DICOM real (COVID-19-AR, CC BY 4.0) de ponta a ponta.

Pula de forma limpa quando a amostra ou os pesos não existem nesta máquina (CI).
Rode sob o lock pesado (~1 min em 4 núcleos de CPU)::

    /root/medai-cache/bin/heavy .venv/bin/python -m pytest \\
        tests/integration/test_platform_real_models.py -m slow

O teste não fixa escores: verifica que o que a API serve é exatamente o que os
modelos calcularam (hash do resultado), que cada execução registrada foi real
(``completed`` com tempo medido) e que achados exploratórios (nível D) ficam
fora da resposta padrão e de todo laudo.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import math
from typing import Any

import pytest
from cache_paths import MEDAI_CACHE
from platform_support import HAS_PLATFORM, PASSWORD

CACHE = MEDAI_CACHE
WEIGHTS = CACHE / "weights" / "registry"
SAMPLE = (
    CACHE
    / "datasets"
    / "samples-cxr-dicom"
    / "1.3.6.1.4.1.14519.5.2.1.9999.103.2009432504168213786754397912410"
    / "00000001.dcm"
)
CXR_MODELS = ("xrv-view-xinario", "xrv-ae-ood", "xrv-densenet121-res224-pc")
HAVE_REAL = (
    HAS_PLATFORM
    and SAMPLE.is_file()
    and all((WEIGHTS / m).is_dir() for m in CXR_MODELS)
    and all(importlib.util.find_spec(m) for m in ("torch", "torchxrayvision", "pypdf"))
)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not HAVE_REAL, reason="amostra CXR real ou pesos XR ausentes"),
]


@pytest.fixture
def real_platform(platform_settings: Any) -> Any:
    from radiologyai.analysis.registry import Registry
    from radiologyai.catalog.catalog import Catalog
    from radiologyai.worker.runtime import AnalysisEngine, Platform

    engine = AnalysisEngine(registry=Registry(weights_dir=WEIGHTS), catalog=Catalog.load())
    platform = Platform.create(platform_settings, engine=engine)
    platform.db.migrate()
    return platform


@pytest.mark.requirement("REQ-160")
def test_real_cxr_flow_serves_exactly_what_models_computed(real_platform: Any) -> None:
    import pydicom
    import torch
    from fastapi.testclient import TestClient
    from pypdf import PdfReader

    from radiologyai.api import create_app
    from radiologyai.persistence.models import Analysis
    from radiologyai.persistence.users import create_user, issue_session, reset_throttle
    from radiologyai.worker.queue import run_one

    torch.set_num_threads(4)
    reset_throttle()
    headers = {}
    for role, extra in (
        ("technician", {}),
        ("physician", {"professional_id": "CRM fictício (teste)"}),
    ):
        user = create_user(
            real_platform.db,
            username=f"{role}-real",
            full_name=f"Pessoa {role}",
            role=role,
            password=PASSWORD,
            **extra,
        )
        token, _ = issue_session(real_platform.db, user, 1.0)
        headers[role] = {"Authorization": f"Bearer {token}"}

    with TestClient(create_app(platform=real_platform, start_worker=False)) as api:
        up = api.post(
            "/api/v1/studies",
            headers=headers["technician"],
            files=[("files", ("cxr.dcm", SAMPLE.read_bytes(), "application/dicom"))],
            data={"view_position": "AP"},
        )
        assert up.status_code == 201, up.text
        study = up.json()
        assert study["exam_type"] == "cxr"
        assert study["patient_pseudo_id"].startswith("ANON-")

        created = api.post(
            f"/api/v1/studies/{study['id']}/analyses", headers=headers["technician"], json={}
        )
        assert created.status_code == 202, created.text
        assert run_one(real_platform) == created.json()["id"]

        body = api.get(f"/api/v1/analyses/{created.json()['id']}", headers=headers["physician"])
        assert body.status_code == 200, body.text
        analysis = body.json()
        assert analysis["state"] == "completed", analysis["error"]

        # o que a API serve é byte a byte o resultado gravado pelo worker
        with real_platform.db.session() as session:
            row = session.get(Analysis, analysis["id"])
            assert row is not None
            assert row.result_json is not None
            digest = hashlib.sha256(row.result_json.encode()).hexdigest()
        assert digest == analysis["result_sha256"]

        research = api.get(
            f"/api/v1/analyses/{analysis['id']}?include_research=true",
            headers=headers["physician"],
        ).json()["result"]
        runs = {r["model_id"]: r for r in research["runs"]}
        for model_id in CXR_MODELS:
            assert runs[model_id]["status"] == "completed", runs[model_id]
            assert runs[model_id]["seconds"] > 0
        assert research["findings"], "modelos reais concluídos devem produzir escores"
        for f in research["findings"]:
            if f["raw_score"] is not None:
                assert math.isfinite(f["raw_score"])
        n_hidden = sum(1 for f in research["findings"] if f["visibility"] != "main")
        assert analysis["hidden_research_findings"] == n_hidden
        assert all(f["visibility"] == "main" for f in analysis["result"]["findings"])

        for idx in analysis["finding_indices"]:
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

        rep = api.post(
            f"/api/v1/analyses/{analysis['id']}/reports", headers=headers["physician"]
        ).json()
        assert rep["integrity"]["content_matches_hash"] is True
        assert rep["content"]["excluded_research_findings"] == n_hidden
        assert all(f["evidence_level"] in ("A", "B", "C") for f in rep["content"]["findings"])
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

        pdf = api.get(f"/api/v1/reports/{rep['id']}/pdf", headers=headers["physician"])
        text = "".join(p.extract_text() for p in PdfReader(io.BytesIO(pdf.content)).pages)
        assert "Uso exclusivo em pesquisa" in text
        sr = pydicom.dcmread(
            io.BytesIO(
                api.get(f"/api/v1/reports/{rep['id']}/sr", headers=headers["physician"]).content
            )
        )
        assert sr.Modality == "SR"
        assert sr.SpecificCharacterSet == "ISO_IR 192"
        fhir = api.get(f"/api/v1/reports/{rep['id']}/fhir", headers=headers["physician"]).json()
        assert fhir["status"] == "preliminary"
