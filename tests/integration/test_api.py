"""API — o servidor do v1 fabricava métricas a cada requisição. Este não."""

from __future__ import annotations

import importlib.util
import io
import json

import pytest

HAS_API = importlib.util.find_spec("fastapi") is not None
pytestmark = pytest.mark.skipif(not HAS_API, reason="requer o extra [api]")


@pytest.fixture
def client(tmp_path):
    from fastapi.testclient import TestClient

    from radiologyai.api import create_app

    return TestClient(create_app(artifacts_dir=str(tmp_path / "sem_artefatos")))


@pytest.fixture
def client_with_metrics(tmp_path):
    from fastapi.testclient import TestClient

    from radiologyai.api import create_app

    run = tmp_path / "eval" / "run-001"
    run.mkdir(parents=True)
    (run / "metrics.json").write_text(
        json.dumps(
            {
                "run_id": "run-001",
                "model": {"card_id": "xrv-densenet121-pc"},
                "dataset": {"name": "NIH ChestX-ray14", "external_to_training_data": True},
                "macro_auroc": 0.781,
                "per_label": {"Effusion": {"auroc": 0.85, "auroc_ci95": [0.83, 0.87]}},
                "not_evaluated": [],
                "subgroups": {},
                "limitations": ["Rótulos minerados por NLP."],
                "environment": {"git_sha": "abc123"},
            }
        ),
        encoding="utf-8",
    )
    return TestClient(create_app(artifacts_dir=str(tmp_path / "eval")))


def dicom_bytes(dataset) -> bytes:
    import pydicom

    buffer = io.BytesIO()
    pydicom.dcmwrite(buffer, dataset, write_like_original=False)
    return buffer.getvalue()


class TestHealth:
    def test_health_ok(self, client):
        body = client.get("/api/v1/health").json()
        assert body["status"] == "ok"
        assert "NÃO é dispositivo médico" in body["disclaimer"]

    def test_status_declares_no_clinical_validation(self, client):
        body = client.get("/api/v1/status").json()
        assert body["clinically_validated"] is False
        assert "não registrado" in body["regulatory_status"]

    def test_status_marks_unimplemented_modalities(self, client):
        by_code = {m["code"]: m for m in client.get("/api/v1/status").json()["modalities"]}
        assert by_code["XR"]["implemented"] is True
        assert by_code["CT"]["implemented"] is False


@pytest.mark.requirement("REQ-080")
class TestMetricsNeverFabricated:
    def test_no_artifact_means_no_number(self, client):
        """O v1 fabricava clinical_metrics de um array de um elemento."""
        body = client.get("/api/v1/metrics").json()
        assert body["measured"] is False
        assert "macro_auroc" not in body
        assert "Nenhum desempenho foi medido" in body["reason"]

    def test_response_explains_how_to_produce(self, client):
        assert "radiologyai evaluate" in client.get("/api/v1/metrics").json()["how_to_produce"]

    def test_artifact_metrics_are_served_with_provenance(self, client_with_metrics):
        body = client_with_metrics.get("/api/v1/metrics").json()
        assert body["measured"] is True
        assert body["macro_auroc"] == 0.781
        assert body["provenance"]["git_sha"] == "abc123"
        assert body["limitations"]

    def test_leakage_status_exposed(self, client_with_metrics):
        assert (
            client_with_metrics.get("/api/v1/metrics").json()["external_to_training_data"] is True
        )


class TestModels:
    def test_measured_performance_flag_matches_runs(self, client):
        from pathlib import Path

        root = Path(__file__).resolve().parents[2]
        for card in client.get("/api/v1/models").json():
            assert card["has_measured_performance"] is bool(card["evaluation_runs"])
            for run_id in card["evaluation_runs"]:
                assert (root / "artifacts" / "eval" / run_id / "metrics.json").is_file()

    def test_no_accuracy_field_exposed(self, client):
        """O model_registry.json do v1 expunha accuracy: 0.92 fabricado."""
        for card in client.get("/api/v1/models").json():
            assert "accuracy" not in card
            assert "auc" not in card

    def test_sha256_is_hex(self, client):
        for card in client.get("/api/v1/models").json():
            assert len(card["weights_sha256"]) == 64


@pytest.mark.requirement("REQ-031")
class TestScopeGateOverHTTP:
    def test_in_scope_reaches_inference(self, client, cr_dataset):
        """Entrada válida chega ao modelo — 200 com pesos, 503 sem eles."""
        response = client.post(
            "/api/v1/analyze",
            files={"file": ("study.dcm", dicom_bytes(cr_dataset), "application/dicom")},
        )
        assert response.status_code in (200, 503)

    def test_lateral_view_rejected_422(self, client, lateral_dataset):
        response = client.post(
            "/api/v1/analyze",
            files={"file": ("study.dcm", dicom_bytes(lateral_dataset), "application/dicom")},
        )
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert "fora do uso pretendido" in detail["error"]
        assert any("incidência" in r for r in detail["reasons"])

    def test_pediatric_rejected_422(self, client, pediatric_dataset):
        response = client.post(
            "/api/v1/analyze",
            files={"file": ("study.dcm", dicom_bytes(pediatric_dataset), "application/dicom")},
        )
        assert response.status_code == 422
        assert any("pediátrica" in r for r in response.json()["detail"]["reasons"])

    def test_unsupported_modality_rejected(self, client, ct_dataset):
        """TC é declarada mas não implementada: recusa, nunca predição."""
        response = client.post(
            "/api/v1/analyze",
            files={"file": ("study.dcm", dicom_bytes(ct_dataset), "application/dicom")},
        )
        assert response.status_code == 422

    def test_non_dicom_rejected_400(self, client):
        response = client.post(
            "/api/v1/analyze",
            files={"file": ("x.dcm", b"isto nao e dicom" * 50, "application/dicom")},
        )
        assert response.status_code == 400

    def test_unknown_card_404(self, client, cr_dataset):
        response = client.post(
            "/api/v1/analyze",
            files={"file": ("study.dcm", dicom_bytes(cr_dataset), "application/dicom")},
            data={"card_id": "inexistente"},
        )
        assert response.status_code == 404


class TestAnalyzeResponseContract:
    def test_response_carries_audit_fields(self, client, cr_dataset):
        response = client.post(
            "/api/v1/analyze",
            files={"file": ("study.dcm", dicom_bytes(cr_dataset), "application/dicom")},
        )
        if response.status_code != 200:
            pytest.skip("backend indisponível neste ambiente")
        body = response.json()
        for key in ("input_sha256", "weights_sha256", "code_version", "card_id"):
            assert body[key], f"{key} ausente — decisão não reconstruível"

    def test_never_asserts_normality(self, client, cr_dataset):
        response = client.post(
            "/api/v1/analyze",
            files={"file": ("study.dcm", dicom_bytes(cr_dataset), "application/dicom")},
        )
        if response.status_code != 200:
            pytest.skip("backend indisponível neste ambiente")
        body = response.json()
        assert body["normality_asserted"] is False
        assert body["requires_physician_review"] is True
        assert not any(f["label"].lower() in ("normal", "no finding") for f in body["findings"])

    def test_no_clinical_metrics_in_response(self, client, cr_dataset):
        """medai_fastapi_server.py:340 devolvia clinical_metrics fabricadas."""
        response = client.post(
            "/api/v1/analyze",
            files={"file": ("study.dcm", dicom_bytes(cr_dataset), "application/dicom")},
        )
        if response.status_code == 200:
            assert "clinical_metrics" not in response.json()

    def test_unmeasured_model_yields_indeterminate_bands(self, client, cr_dataset):
        """Sem avaliação medida não há ponto de operação, logo tudo é indeterminado."""
        response = client.post(
            "/api/v1/analyze",
            files={"file": ("study.dcm", dicom_bytes(cr_dataset), "application/dicom")},
        )
        if response.status_code != 200:
            pytest.skip("backend indisponível neste ambiente")
        for finding in response.json()["findings"]:
            assert finding["band"] == "nao_avaliavel"
            assert finding["evaluated"] is False


@pytest.fixture
def client_with_repo_artifacts():
    """API apontada para os artefatos REAIS versionados no repositório."""
    from pathlib import Path

    from fastapi.testclient import TestClient

    from radiologyai.api import create_app

    root = Path(__file__).resolve().parents[2]
    return TestClient(create_app(artifacts_dir=str(root / "artifacts" / "eval")))


@pytest.mark.requirement("REQ-062")
class TestThresholdsComeFromMeasuredArtifact:
    """Os limiares da abstenção vêm do artefato medido — nunca de um placeholder."""

    def test_bands_derive_from_baseline_run(self, client_with_repo_artifacts, cr_dataset):
        response = client_with_repo_artifacts.post(
            "/api/v1/analyze",
            files={"file": ("study.dcm", dicom_bytes(cr_dataset), "application/dicom")},
        )
        if response.status_code == 503:
            pytest.skip("backend indisponível neste ambiente")
        assert response.status_code == 200, response.text
        findings = response.json()["findings"]
        evaluated = [f for f in findings if f["evaluated"]]
        assert evaluated, "nenhum achado com medição — o artefato não foi usado"
        for f in evaluated:
            assert f["evaluation_run"] == "xrv-densenet121-pc__20260912T202549Z"
            assert f["band"] in ("achado_provavel", "nao_avaliavel", "achado_improvavel")
            assert f["calibrated"] is False

    def test_run_without_artifact_in_checkout_is_all_indeterminate(self, client, cr_dataset):
        """Card referencia um run; o artefato não está neste artifacts_dir → tudo indeterminado."""
        response = client.post(
            "/api/v1/analyze",
            files={"file": ("study.dcm", dicom_bytes(cr_dataset), "application/dicom")},
        )
        if response.status_code == 503:
            pytest.skip("backend indisponível neste ambiente")
        for f in response.json()["findings"]:
            assert f["band"] == "nao_avaliavel"
            assert "não está neste checkout" in f["reason"]
