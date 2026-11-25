"""Fila de análises: reivindicação atômica, resultado imutável, falhas com motivo."""

from __future__ import annotations

import hashlib
import importlib.util
import io

import pytest

HAS = all(importlib.util.find_spec(m) for m in ("sqlalchemy", "alembic", "argon2", "PIL"))
pytestmark = pytest.mark.skipif(not HAS, reason="requer os extras [platform,imaging]")


@pytest.fixture
def platform(tmp_path):
    from radiologyai.analysis.registry import Registry
    from radiologyai.catalog.catalog import Catalog
    from radiologyai.config.settings import Settings
    from radiologyai.persistence.config import PlatformSettings
    from radiologyai.worker.runtime import AnalysisEngine, Platform

    (tmp_path / "manifests").mkdir()
    engine = AnalysisEngine(
        registry=Registry(
            manifest_dir=tmp_path / "manifests",
            weights_dir=tmp_path / "w",
            settings=Settings(weights_dir=tmp_path / "w", device="cpu"),
        ),
        catalog=Catalog([], None),
    )
    p = Platform.create(
        PlatformSettings(
            database_url=f"sqlite:///{tmp_path / 'p.db'}",
            store_dir=tmp_path / "store",
            deident_key="chave",
            worker_enabled=False,
        ),
        engine=engine,
    )
    p.db.migrate()
    return p


@pytest.fixture
def study(platform, cr_dataset, tmp_path):
    import pydicom

    from radiologyai.interop.ingest import ingest
    from radiologyai.persistence.users import create_user

    user = create_user(
        platform.db, username="t", full_name="T", role="technician", password="senha-de-teste-1"
    )
    path = tmp_path / "a.dcm"
    buf = io.BytesIO()
    pydicom.dcmwrite(buf, cr_dataset, enforce_file_format=True)
    path.write_bytes(buf.getvalue())
    result = ingest(
        [("a.dcm", path)],
        db=platform.db,
        blobs=platform.blobs,
        settings=platform.settings,
        user_id=user.id,
        username="t",
    )
    return result.study_id, user.id


def enqueue(platform, study):
    from radiologyai.worker.jobs import enqueue_analysis

    study_id, user_id = study
    return enqueue_analysis(platform, study_id, user_id=user_id, username="t").id


def row(platform, analysis_id):
    from radiologyai.persistence.models import Analysis

    with platform.db.session() as session:
        return session.get(Analysis, analysis_id)


@pytest.mark.requirement("REQ-154")
def test_claim_is_exclusive(platform, study):
    from radiologyai.worker.queue import claim_next

    analysis_id = enqueue(platform, study)
    assert claim_next(platform, "w1") == analysis_id
    assert claim_next(platform, "w2") is None
    assert row(platform, analysis_id).state == "running"
    assert row(platform, analysis_id).worker_id == "w1"


@pytest.mark.requirement("REQ-154")
def test_no_models_routed_is_honest_abstention(platform, study):
    """Sem nenhum modelo disponível o resultado diz isso; nada é preenchido no lugar."""
    from radiologyai.analysis.types import AnalysisResult
    from radiologyai.worker.queue import run_one

    analysis_id = enqueue(platform, study)
    assert run_one(platform) == analysis_id
    done = row(platform, analysis_id)
    assert done.state == "completed"
    assert hashlib.sha256(done.result_json.encode()).hexdigest() == done.result_sha256
    result = AnalysisResult.model_validate_json(done.result_json)
    assert result.status == "abstained"
    assert not result.findings
    assert not result.runs
    assert any("nenhum modelo roteado" in w for w in result.warnings)


@pytest.mark.requirement("REQ-154")
def test_result_written_once(platform, study):
    from radiologyai.worker.queue import run_one
    from radiologyai.worker.runner import AnalysisJobError, persist_result

    analysis_id = enqueue(platform, study)
    run_one(platform)
    from radiologyai.analysis.types import AnalysisResult

    result = AnalysisResult.model_validate_json(row(platform, analysis_id).result_json)
    with pytest.raises(AnalysisJobError, match="já com resultado"):
        persist_result(platform, analysis_id, result)


@pytest.mark.requirement("REQ-154")
def test_rerun_creates_new_row(platform, study):
    first, second = enqueue(platform, study), enqueue(platform, study)
    assert first != second
    assert row(platform, first).state == row(platform, second).state == "queued"


@pytest.mark.requirement("REQ-154")
def test_failure_recorded_with_reason(platform, study):
    from radiologyai.persistence.models import Analysis
    from radiologyai.worker.queue import run_one

    analysis_id = enqueue(platform, study)
    with platform.db.session() as session:
        session.get(Analysis, analysis_id).inputs = [
            {"image_idx": 0, "series_id": "nao-existe", "content_sha256": "0" * 64}
        ]
    run_one(platform)
    failed = row(platform, analysis_id)
    assert failed.state == "failed"
    assert failed.result_json is None
    assert "não existe mais" in failed.error


@pytest.mark.requirement("REQ-154")
def test_interrupted_running_becomes_failed(platform, study):
    from radiologyai.worker.queue import INTERRUPTED, claim_next, recover_interrupted

    analysis_id = enqueue(platform, study)
    claim_next(platform, "morto")
    assert recover_interrupted(platform) == 1
    failed = row(platform, analysis_id)
    assert failed.state == "failed"
    assert failed.error == INTERRUPTED


@pytest.mark.requirement("REQ-154")
def test_unknown_exam_type_blocks_enqueue(platform, study):
    from radiologyai.errors import ExamTypeUnknownError
    from radiologyai.persistence.models import Study

    with platform.db.session() as session:
        session.get(Study, study[0]).exam_type = "unknown"
    with pytest.raises(ExamTypeUnknownError):
        enqueue(platform, study)


@pytest.mark.requirement("REQ-154")
def test_worker_thread_processes_queue(platform, study):
    import time

    from radiologyai.worker.queue import Worker

    analysis_id = enqueue(platform, study)
    worker = Worker(platform)
    worker.start()
    try:
        worker.notify()
        deadline = time.monotonic() + 30
        while row(platform, analysis_id).state in ("queued", "running"):
            assert time.monotonic() < deadline
            time.sleep(0.05)
    finally:
        worker.stop()
    assert row(platform, analysis_id).state == "completed"
    assert not worker.alive
