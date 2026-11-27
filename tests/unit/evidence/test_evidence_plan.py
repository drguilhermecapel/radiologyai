"""Plano de medição, razão, pesos conferidos contra manifesto e registros aguardando condição."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from cache_paths import MEDAI_CACHE

from radiologyai.catalog import load_protocol
from radiologyai.errors import EvidenceError
from radiologyai.evidence.leakage import assess_leakage
from radiologyai.evidence.plan import (
    Ledger,
    Plan,
    PlanJob,
    execute_job,
    hash_weights,
    load_plan,
    manifest_hashes,
    render_ledger,
)
from radiologyai.evidence.provenance import dirty_paths, git_state
from radiologyai.evidence.store import AWAITING_DIR, evidence_files, load_records

REPO = Path(__file__).resolve().parents[3]
REPO_PLAN = REPO / "evidence" / "measured" / "plan.yaml"
HAS_PARQUET = all(importlib.util.find_spec(m) for m in ("pandas", "pyarrow"))
HAS_GIT = shutil.which("git") is not None
FIXED = datetime(2026, 11, 27, tzinfo=UTC)
RADDINO = ("mimic-cxr", "chexpert", "nih-cxr14", "padchest", "brax")
SHA_LINE = re.compile(r"sha256[\"']?\s*:\s*[\"']?([^\"'\s,}]+)", re.I)

needs_parquet = pytest.mark.skipif(not HAS_PARQUET, reason="pandas/pyarrow ausentes")


@pytest.fixture(scope="module")
def protocol():
    return load_protocol()


def _job(**kw) -> PlanJob:
    base = {
        "id": "j1",
        "status": "measure",
        "model_id": "m1",
        "dataset_id": "tb-montgomery",
        "label": "tuberculosis",
        "condition_id": "cxr.tuberculosis",
        "predictions": "m1/tb-montgomery/tuberculosis.parquet",
        "score_scale": "logit",
        "training_data": ("tb-shenzhen",),
        "weights": {"probe": "w/probe.safetensors"},
    }
    base.update(kw)
    return PlanJob.model_validate(base)


def _plan(tmp_path: Path, *jobs: PlanJob) -> Plan:
    return Plan.model_validate(
        {
            "format": "radiologyai.evidence/plan-v1",
            "protocol_id": "2026-11-v1",
            "preds_root": str(tmp_path / "preds"),
            "jobs": [j.model_dump() for j in jobs],
        }
    )


# --------------------------------------------------------------------------- #
# Vazamento das chaves novas
# --------------------------------------------------------------------------- #
class TestLeakageKeys:
    @pytest.mark.requirement("REQ-114")
    def test_raddino_corpus_on_nih_is_known(self):
        assert assess_leakage("nih-cxr14", ("tb-shenzhen", *RADDINO))[0] == "known"
        assert assess_leakage("tb-montgomery", ("tb-shenzhen", *RADDINO))[0] == "none"

    @pytest.mark.requirement("REQ-114")
    @pytest.mark.parametrize(
        "dataset", ["busi", "bus-bra", "breast-lesions-usg", "liver-lesion-us-7272660"]
    )
    def test_pmc15m_pretraining_is_possible_for_public_us_sets(self, dataset):
        leak, note = assess_leakage(dataset, ("pmc-15m",))
        assert leak == "possible"
        assert "pmc-15m" in note

    @pytest.mark.requirement("REQ-114")
    def test_medsam2_corpus_on_camus_is_possible(self):
        assert assess_leakage("camus", ("medsam2-training-corpus",))[0] == "possible"

    @pytest.mark.requirement("REQ-114")
    def test_grazpedwri_holdout_lineage(self):
        assert assess_leakage("grazpedwri-dx-sample-test30", ("grazpedwri-dx",))[0] == "known"
        assert assess_leakage("grazpedwri-dx-sample", ("fracatlas", *RADDINO))[0] == "none"


# --------------------------------------------------------------------------- #
# Validação do plano
# --------------------------------------------------------------------------- #
class TestPlanValidation:
    @pytest.mark.requirement("REQ-115")
    def test_pending_requires_reason(self):
        with pytest.raises(ValueError, match="reason"):
            PlanJob.model_validate({"id": "p", "status": "pending"})
        assert PlanJob.model_validate({"id": "p", "status": "pending", "reason": "7/10"})

    @pytest.mark.requirement("REQ-115")
    def test_measure_requires_weights_and_training_declaration(self):
        with pytest.raises(ValueError, match="weights"):
            _job(weights={})
        with pytest.raises(ValueError, match="training_data"):
            _job(training_data=())
        assert _job(training_data=(), no_training_data_declared=True)

    @pytest.mark.requirement("REQ-115")
    def test_agreement_requires_columns(self):
        with pytest.raises(ValueError, match="agreement"):
            _job(endpoint="agreement")

    @pytest.mark.requirement("REQ-115")
    def test_two_jobs_writing_same_record_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="mesmo registro"):
            _plan(tmp_path, _job(id="a"), _job(id="b"))

    @pytest.mark.requirement("REQ-114")
    def test_unknown_dataset_key_rejected_on_load(self, tmp_path):
        import yaml

        plan = _plan(tmp_path, _job(dataset_id="dataset-sem-linhagem"))
        path = tmp_path / "plan.yaml"
        path.write_text(yaml.safe_dump(plan.model_dump(mode="json")), encoding="utf-8")
        with pytest.raises(EvidenceError, match="desconhecida"):
            load_plan(path)

    @pytest.mark.requirement("REQ-115")
    def test_repo_plan_is_valid_and_declares_every_status(self):
        plan = load_plan(REPO_PLAN)
        statuses = {j.status for j in plan.jobs}
        assert statuses == {"measure", "pending", "excluded"}
        assert plan.protocol_id == load_protocol().protocol_id
        for job in plan.jobs:
            if job.status == "measure":
                assert job.weights
                assert job.training_data


# --------------------------------------------------------------------------- #
# Pesos: hash lido e conferido contra o manifesto
# --------------------------------------------------------------------------- #
class TestWeights:
    def _manifest(self, tmp_path: Path, sha: str) -> Path:
        d = tmp_path / "manifests"
        d.mkdir(exist_ok=True)
        (d / "m1.yaml").write_text(
            f"id: m1\nweights:\n- key: probe\n  filename: probe.safetensors\n  sha256: {sha}\n",
            encoding="utf-8",
        )
        return d

    @pytest.mark.requirement("REQ-115")
    def test_hash_matches_manifest(self, tmp_path):
        w = tmp_path / "probe.safetensors"
        w.write_bytes(b"sonda")
        sha = hashlib.sha256(b"sonda").hexdigest()
        mdir = self._manifest(tmp_path, sha)
        job = _job(weights={"probe": str(w)}, manifest="m1")
        assert manifest_hashes("m1", mdir) == {"probe.safetensors": sha}
        assert hash_weights(job, manifests_dir=mdir) == {"probe.safetensors": sha}

    @pytest.mark.requirement("REQ-115")
    def test_hash_mismatch_with_manifest_refused(self, tmp_path):
        w = tmp_path / "probe.safetensors"
        w.write_bytes(b"outra sonda")
        mdir = self._manifest(tmp_path, "c" * 64)
        with pytest.raises(EvidenceError, match="difere do manifesto"):
            hash_weights(_job(weights={"probe": str(w)}, manifest="m1"), manifests_dir=mdir)

    @pytest.mark.requirement("REQ-115")
    def test_no_weight_in_manifest_refused(self, tmp_path):
        w = tmp_path / "outro.bin"
        w.write_bytes(b"x")
        mdir = self._manifest(tmp_path, "c" * 64)
        with pytest.raises(EvidenceError, match="nenhum arquivo"):
            hash_weights(_job(weights={"x": str(w)}, manifest="m1"), manifests_dir=mdir)

    @pytest.mark.requirement("REQ-115")
    def test_missing_weight_file_refused(self, tmp_path):
        with pytest.raises(EvidenceError, match="não encontrado"):
            hash_weights(_job(weights={"probe": str(tmp_path / "nada.pt")}))


# --------------------------------------------------------------------------- #
# Execução de itens
# --------------------------------------------------------------------------- #
def _binary_parquet(path: Path, n: int = 200, seed: int = 0) -> Path:
    import pandas as pd

    rng = np.random.default_rng(seed)
    y = np.r_[np.ones(n // 2, dtype=int), np.zeros(n - n // 2, dtype=int)]
    score = rng.normal(0.0, 1.0, n) + 2.5 * y
    df = pd.DataFrame(
        {
            "patient_id": [f"p{i}" for i in range(n)],
            "image_id": [f"i{i}" for i in range(n)],
            "sex": ["F" if i % 2 else "M" for i in range(n)],
            "y_true": y,
            "y_score": score,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return path


def _ef_parquet(path: Path, n: int = 60) -> Path:
    import pandas as pd

    rng = np.random.default_rng(1)
    ref = rng.uniform(20, 70, n)
    rows = []
    for mode, noise in (("oracle_box", 2.0), ("prompt_free", 15.0)):
        for i in range(n):
            pred = ref[i] + rng.normal(0, noise)
            rows.append(
                {
                    "patient_id": f"p{i}",
                    "image_id": f"p{i}",
                    "mode": mode,
                    "y_true": ref[i],
                    "y_pred": np.nan if (mode == "prompt_free" and i == 0) else pred,
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path)
    return path


def _run(plan, job, protocol, root, known, **kw):
    return execute_job(
        plan,
        job,
        protocol,
        known_conditions=known,
        root=root,
        git_sha="a" * 40,
        git_dirty=False,
        command="evidence_run.py --plan plan.yaml",
        n_bootstrap=50,
        now=lambda: FIXED,
        **kw,
    )


@needs_parquet
class TestExecuteJob:
    @pytest.mark.requirement("REQ-115")
    def test_condition_missing_from_catalog_goes_to_awaiting_and_out_of_levels(
        self, tmp_path, protocol
    ):
        w = tmp_path / "w.pt"
        w.write_bytes(b"pesos")
        _binary_parquet(tmp_path / "preds" / "m1" / "tb-montgomery" / "tuberculosis.parquet")
        job = _job(weights={"w": str(w)}, condition_id="msk_xr.condicao_nova")
        plan = _plan(tmp_path, job)
        root = tmp_path / "evidence"
        entry = _run(plan, job, protocol, root, known={"cxr.tuberculosis"})
        assert entry.awaiting_condition
        assert entry.outcome == "aprovado"
        assert entry.record is not None
        assert entry.record.startswith(f"measured/{AWAITING_DIR}/")
        assert (root / entry.record).is_file()
        assert evidence_files(root) == []
        assert load_records(root) == []

    @pytest.mark.requirement("REQ-115")
    def test_known_condition_written_with_read_hashes(self, tmp_path, protocol):
        w = tmp_path / "w.pt"
        w.write_bytes(b"pesos")
        _binary_parquet(tmp_path / "preds" / "m1" / "tb-montgomery" / "tuberculosis.parquet")
        job = _job(weights={"w": str(w)})
        plan = _plan(tmp_path, job)
        root = tmp_path / "evidence"
        entry = _run(plan, job, protocol, root, known={"cxr.tuberculosis"})
        assert not entry.awaiting_condition
        (rec,) = load_records(root)
        assert rec.provenance is not None
        assert rec.provenance.weights_sha256 == {"w.pt": hashlib.sha256(b"pesos").hexdigest()}
        assert rec.kind == "measured_external"
        assert rec.leakage == "none"
        env = json.loads((root / entry.record).read_text(encoding="utf-8"))
        assert env["inputs"]["plan_job"] == "j1"
        assert len(env["operating_points"]) == 2

    @pytest.mark.requirement("REQ-113")
    def test_agreement_with_filter_counts_abstentions(self, tmp_path, protocol):
        w = tmp_path / "ck.pt"
        w.write_bytes(b"ck")
        _ef_parquet(tmp_path / "preds" / "echo" / "camus" / "ef.parquet")
        common = {
            "endpoint": "agreement",
            "model_id": "echo",
            "dataset_id": "camus",
            "condition_id": "echo.reduced_lvef",
            "predictions": "echo/camus/ef.parquet",
            "reference_column": "y_true",
            "measured_column": "y_pred",
            "training_data": ("medsam2-training-corpus",),
            "weights": {"ck": str(w)},
            "score_scale": "probability",
        }
        box = _job(id="box", label="lv_ef", where={"mode": "oracle_box"}, **common)
        free = _job(id="free", label="lv_ef.free", where={"mode": "prompt_free"}, **common)
        plan = _plan(tmp_path, box, free)
        root = tmp_path / "evidence"
        e_box = _run(plan, box, protocol, root, known={"echo.reduced_lvef"})
        e_free = _run(plan, free, protocol, root, known={"echo.reduced_lvef"})
        assert e_box.outcome == "aprovado"
        assert e_box.leakage == "possible"
        assert e_box.n_total == 60
        assert e_free.n_total == 59
        env = json.loads((root / e_free.record).read_text(encoding="utf-8"))
        assert env["inputs"]["n_abstained_excluded"] == 1
        assert env["inputs"]["n_rows_outside_filter"] == 60

    @pytest.mark.requirement("REQ-115")
    def test_pending_and_excluded_never_compute(self, tmp_path, protocol):
        pending = PlanJob(id="p", status="pending", reason="medição pendente: 7 de 10")
        excluded = PlanJob(id="x", status="excluded", reason="não é avaliação")
        plan = _plan(tmp_path, pending, excluded)
        root = tmp_path / "evidence"
        e1 = _run(plan, pending, protocol, root, known=set())
        e2 = _run(plan, excluded, protocol, root, known=set())
        assert (e1.outcome, e2.outcome) == ("pendente", "excluido")
        assert e1.record is None
        assert e1.n_total is None
        assert not root.exists()

    @pytest.mark.requirement("REQ-115")
    def test_measurement_error_is_failure_not_silence(self, tmp_path, protocol):
        w = tmp_path / "w.pt"
        w.write_bytes(b"pesos")
        job = _job(weights={"w": str(w)})  # parquet inexistente
        plan = _plan(tmp_path, job)
        entry = _run(plan, job, protocol, tmp_path / "evidence", known={"cxr.tuberculosis"})
        assert entry.outcome == "falha"
        assert "não encontrado" in entry.reasons[0]
        assert entry.record is None

    @pytest.mark.requirement("REQ-115")
    def test_ledger_is_deterministic_and_honesty_clean(self, tmp_path, protocol):
        pending = PlanJob(id="p", status="pending", reason="medição pendente")
        plan = _plan(tmp_path, pending)
        entry = _run(plan, pending, protocol, tmp_path / "evidence", known=set())
        ledger = Ledger(protocol_id="2026-11-v1", plan="plan.yaml", entries=(entry,))
        text = render_ledger(ledger)
        assert text == render_ledger(Ledger.model_validate(ledger.model_dump()))
        for line in text.splitlines():
            m = SHA_LINE.search(line)
            assert not m or re.fullmatch(r"[0-9a-f]{64}", m.group(1))


# --------------------------------------------------------------------------- #
# Estado git com escopo de medição
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not HAS_GIT, reason="git ausente")
class TestScopedDirty:
    def _repo(self, tmp_path: Path) -> Path:
        repo = tmp_path / "repo"
        (repo / "src" / "radiologyai" / "evidence").mkdir(parents=True)
        (repo / "src" / "radiologyai" / "api").mkdir(parents=True)
        (repo / "src" / "radiologyai" / "evidence" / "m.py").write_text("x = 1\n")
        (repo / "src" / "radiologyai" / "api" / "a.py").write_text("y = 1\n")
        env = {
            **os.environ,
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@example.invalid",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@example.invalid",
        }
        for cmd in (
            ["init", "-q"],
            ["add", "-A"],
            ["-c", "commit.gpgsign=false", "commit", "-q", "-m", "c"],
        ):
            subprocess.run(["git", *cmd], cwd=repo, check=True, env=env)
        return repo

    @pytest.mark.requirement("REQ-115")
    def test_only_measurement_scope_counts(self, tmp_path):
        repo = self._repo(tmp_path)
        assert git_state(repo)[1] is False
        (repo / "src" / "radiologyai" / "api" / "a.py").write_text("y = 2\n")
        (repo / "src" / "radiologyai" / "api" / "novo.py").write_text("z = 1\n")
        assert dirty_paths(repo) == []
        assert dirty_paths(repo, scope=None) == ["src/radiologyai/api/a.py"]

    @pytest.mark.requirement("REQ-115")
    def test_untracked_module_in_scope_is_dirty(self, tmp_path):
        repo = self._repo(tmp_path)
        (repo / "src" / "radiologyai" / "evidence" / "novo.py").write_text("z = 1\n")
        assert dirty_paths(repo) == ["src/radiologyai/evidence/novo.py"]
        sha, dirty = git_state(repo)
        assert len(sha) == 40
        assert dirty is True


# --------------------------------------------------------------------------- #
# Dados reais: o registro do repositório é reproduzível a partir do plano
# --------------------------------------------------------------------------- #
REAL_JOBS = ("xr-probe-tb@tb-montgomery", "medsam2-echo-lv@camus:box")


def _rebased(plan: Plan) -> Plan:
    """O plano com a raiz do cache trocada pela de ``MEDAI_CACHE``."""
    old = str(Path(plan.preds_root).parent)
    new = str(MEDAI_CACHE)

    def fix(s: str) -> str:
        return new + s[len(old) :] if s.startswith(old) else s

    jobs = [
        j.model_copy(update={"weights": {k: fix(v) for k, v in j.weights.items()}})
        for j in plan.jobs
    ]
    return plan.model_copy(update={"preds_root": fix(plan.preds_root), "jobs": tuple(jobs)})


@needs_parquet
@pytest.mark.skipif(
    not (MEDAI_CACHE / "eval-preds" / "xr-probe-tb").is_dir(),
    reason="predições reais ausentes (MEDAI_CACHE)",
)
@pytest.mark.parametrize("job_id", REAL_JOBS)
@pytest.mark.requirement("REQ-115")
def test_repo_record_reproduces_from_plan(tmp_path, protocol, job_id):
    plan = _rebased(load_plan(REPO_PLAN))
    job = next(j for j in plan.jobs if j.id == job_id)
    entry = execute_job(
        plan,
        job,
        protocol,
        known_conditions={job.condition_id or ""},
        root=tmp_path / "evidence",
        git_sha="a" * 40,
        git_dirty=False,
        command="teste",
    )
    assert entry.record is not None
    fresh = json.loads((tmp_path / "evidence" / entry.record).read_text(encoding="utf-8"))
    committed = json.loads((REPO / "evidence" / entry.record).read_text(encoding="utf-8"))
    for key in ("metrics", "n_total", "n_positive", "passes_protocol", "leakage", "kind"):
        assert fresh["record"][key] == committed["record"][key], key
    prov_f, prov_c = fresh["record"]["provenance"], committed["record"]["provenance"]
    assert prov_f["dataset_sha256"] == prov_c["dataset_sha256"]
    assert prov_f["weights_sha256"] == prov_c["weights_sha256"]
    assert fresh["inputs"]["predictions_sha256"] == committed["inputs"]["predictions_sha256"]
