"""API — o servidor do v1 fabricava métricas a cada requisição. Este não.

Reescrito para a API com autenticação (v3 da plataforma): os mesmos contratos de
antes (REQ-031, REQ-062, REQ-063, REQ-080) mais a exigência de sessão no
endpoint legado ``/analyze``. Saúde e métricas MEDIDAS continuam públicas.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_support import HAS_PLATFORM, dicom_bytes

pytestmark = pytest.mark.skipif(not HAS_PLATFORM, reason="requer os extras [api,platform]")

REPO = Path(__file__).resolve().parents[2]


def _client(platform_settings, artifacts_dir):
    from fastapi.testclient import TestClient

    from radiologyai.api import create_app
    from radiologyai.worker.runtime import Platform

    platform = Platform.create(platform_settings)
    return TestClient(create_app(platform=platform, artifacts_dir=str(artifacts_dir)))


@pytest.fixture
def client(platform_settings, tmp_path):
    return _client(platform_settings, tmp_path / "sem_artefatos")


@pytest.fixture
def client_with_metrics(platform_settings, tmp_path):
    import json

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
    return _client(platform_settings, tmp_path / "eval")


@pytest.fixture
def client_with_repo_artifacts(platform_settings):
    """API apontada para os artefatos REAIS versionados no repositório."""
    return _client(platform_settings, REPO / "artifacts" / "eval")


def _auth(client, role: str = "physician") -> dict[str, str]:
    from radiologyai.persistence.users import create_user, issue_session

    platform = client.app.state.platform
    user = create_user(
        platform.db,
        username=f"{role}-api",
        full_name="Teste",
        role=role,
        password="senha-de-teste-123",
    )
    token, _ = issue_session(platform.db, user, 1.0)
    return {"Authorization": f"Bearer {token}"}


def post_analyze(client, dataset, headers=None, **data):
    return client.post(
        "/api/v1/analyze",
        headers=headers if headers is not None else _auth(client),
        files={"file": ("study.dcm", dicom_bytes(dataset), "application/dicom")},
        data=data,
    )


class TestHealth:
    def test_health_ok(self, client):
        body = client.get("/api/v1/health").json()
        assert body["status"] == "ok"
        assert "NÃO é dispositivo médico" in body["disclaimer"]

    def test_status_declares_no_clinical_validation(self, client):
        body = client.get("/api/v1/status").json()
        assert body["clinically_validated"] is False
        assert "não registrado" in body["regulatory_status"]

    def test_status_lists_declared_modalities_with_maturity(self, client):
        """Cada modalidade declara se está implementada; nenhuma omite o campo."""
        modalities = client.get("/api/v1/status").json()["modalities"]
        by_code = {m["code"]: m for m in modalities}
        assert by_code["XR"]["implemented"] is True
        assert all(isinstance(m["implemented"], bool) for m in modalities)


@pytest.mark.requirement("REQ-080")
class TestMetricsNeverFabricated:
    def test_no_artifact_means_no_number(self, client):
        """O v1 fabricava clinical_metrics de um array de um elemento."""
        body = client.get("/api/v1/metrics").json()
        assert body["measured"] is False
        assert body["macro_auroc"] is None
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

    def test_no_performance_keys_in_model_cards(self, client):
        """O model_registry.json do v1 expunha accuracy: 0.92 fabricado."""
        forbidden = {"accuracy", "auc", "auroc", "sensitivity", "specificity"}

        def walk(value):
            if isinstance(value, dict):
                assert not forbidden & {str(k).lower() for k in value}, value.keys()
                for sub in value.values():
                    walk(sub)
            elif isinstance(value, list):
                for sub in value:
                    walk(sub)

        cards = client.get("/api/v1/models", headers=_auth(client, "viewer")).json()
        assert cards, "o registro do pacote deve listar modelos"
        walk(cards)
        for card in cards:
            for weight in card["weights"]:
                assert len(weight["sha256"]) == 64 or card["status"] == "planned"
            if card["status"] == "planned":
                assert card["planned_reason"]


@pytest.mark.requirement("REQ-031")
class TestScopeGateOverHTTP:
    def test_requires_session(self, client, cr_dataset):
        assert post_analyze(client, cr_dataset, headers={}).status_code == 401

    def test_viewer_cannot_run_inference(self, client, cr_dataset):
        response = post_analyze(client, cr_dataset, headers=_auth(client, "viewer"))
        assert response.status_code == 403

    def test_in_scope_reaches_inference(self, client, cr_dataset):
        """Entrada válida chega ao modelo — 200 com pesos, 503 sem eles."""
        assert post_analyze(client, cr_dataset).status_code in (200, 503)

    def test_lateral_view_rejected_422(self, client, lateral_dataset):
        response = post_analyze(client, lateral_dataset)
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert "fora do uso pretendido" in detail["error"]
        assert any("incidência" in r for r in detail["reasons"])

    def test_pediatric_rejected_422(self, client, pediatric_dataset):
        response = post_analyze(client, pediatric_dataset)
        assert response.status_code == 422
        assert any("pediátrica" in r for r in response.json()["detail"]["reasons"])

    def test_out_of_scope_modality_never_predicted(self, client, ct_dataset):
        """Modalidade que o card não cobre: recusa (422) ou falha fechada — nunca 200."""
        response = post_analyze(client, ct_dataset)
        assert response.status_code in (404, 422, 503)
        assert "findings" not in response.json()

    def test_non_dicom_rejected_400(self, client):
        response = client.post(
            "/api/v1/analyze",
            headers=_auth(client),
            files={"file": ("x.dcm", b"isto nao e dicom" * 50, "application/dicom")},
        )
        assert response.status_code == 400

    def test_unknown_card_404(self, client, cr_dataset):
        assert post_analyze(client, cr_dataset, card_id="inexistente").status_code == 404


@pytest.mark.requirement("REQ-063")
class TestAnalyzeResponseContract:
    def test_response_carries_audit_fields(self, client, cr_dataset):
        response = post_analyze(client, cr_dataset)
        if response.status_code != 200:
            pytest.skip("backend indisponível neste ambiente")
        body = response.json()
        for key in ("input_sha256", "weights_sha256", "code_version", "card_id"):
            assert body[key], f"{key} ausente — decisão não reconstruível"

    def test_never_asserts_normality(self, client, cr_dataset):
        response = post_analyze(client, cr_dataset)
        if response.status_code != 200:
            pytest.skip("backend indisponível neste ambiente")
        body = response.json()
        assert body["normality_asserted"] is False
        assert body["requires_physician_review"] is True
        assert not any(f["label"].lower() in ("normal", "no finding") for f in body["findings"])

    def test_no_clinical_metrics_in_response(self, client, cr_dataset):
        """medai_fastapi_server.py:340 devolvia clinical_metrics fabricadas."""
        response = post_analyze(client, cr_dataset)
        if response.status_code == 200:
            assert "clinical_metrics" not in response.json()

    def test_unmeasured_model_yields_indeterminate_bands(self, client, cr_dataset):
        """Sem avaliação medida não há ponto de operação, logo tudo é indeterminado."""
        response = post_analyze(client, cr_dataset)
        if response.status_code != 200:
            pytest.skip("backend indisponível neste ambiente")
        for finding in response.json()["findings"]:
            assert finding["band"] == "nao_avaliavel"
            assert finding["evaluated"] is False


@pytest.mark.requirement("REQ-062")
class TestThresholdsComeFromMeasuredArtifact:
    """Os limiares da abstenção vêm do artefato medido — nunca de um placeholder."""

    def test_bands_derive_from_baseline_run(self, client_with_repo_artifacts, cr_dataset):
        response = post_analyze(client_with_repo_artifacts, cr_dataset)
        if response.status_code == 503:
            pytest.skip("backend indisponível neste ambiente")
        assert response.status_code == 200, response.text
        findings = response.json()["findings"]
        evaluated = [f for f in findings if f["evaluated"]]
        assert evaluated, "nenhum achado com medição — o artefato não foi usado"
        for f in evaluated:
            assert f["evaluation_run"] == "xrv-densenet121-pc__20260912T215742Z"
            assert f["band"] in ("achado_provavel", "nao_avaliavel", "achado_improvavel")
            assert f["calibrated"] is False

    def test_run_without_artifact_in_checkout_is_all_indeterminate(self, client, cr_dataset):
        response = post_analyze(client, cr_dataset)
        if response.status_code == 503:
            pytest.skip("backend indisponível neste ambiente")
        for f in response.json()["findings"]:
            assert f["band"] == "nao_avaliavel"
            assert "não está neste checkout" in f["reason"]
