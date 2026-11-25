"""Fábrica de evidência: vazamento por tabela, julgamento pelo protocolo, níveis gerados."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from radiologyai.catalog import load_protocol
from radiologyai.catalog.schema import (
    Condition,
    EvidenceLevel,
    EvidenceRecord,
    EvidenceSource,
    LevelsFile,
    Metric,
    compute_level,
)
from radiologyai.errors import EvidenceError
from radiologyai.evaluation.protocol_eval import (
    BinaryProtocolReport,
    Counts,
    Estimate,
    SplitSummary,
)
from radiologyai.evidence.leakage import assess_leakage, lineage
from radiologyai.evidence.levels import build_levels, check_levels, render_levels
from radiologyai.evidence.measure import (
    ELIGIBILITY_PREFIX,
    MeasurementContext,
    judge_discrimination,
    measure_agreement,
    measure_binary,
    verdict,
)
from radiologyai.evidence.store import dump_json, load_records, write_measured

REPO = Path(__file__).resolve().parents[3]
HAS_PARQUET = all(importlib.util.find_spec(m) for m in ("pandas", "pyarrow"))
SHA = "b" * 64


@pytest.fixture(scope="module")
def protocol():
    return load_protocol()


def ctx(**kw) -> MeasurementContext:
    base = {
        "model_id": "xrv-densenet121-pc",
        "label": "Effusion",
        "condition_id": "cxr.pleural_effusion",
        "dataset_id": "nih-cxr14",
        "training_data": ("padchest",),
        "reference_standard": "report_nlp",
        "weights_sha256": {"pc.pt": "a" * 64},
        "git_sha": "c" * 40,
        "command": "evidence_run.py --test",
        "dataset_sha256": SHA,
        "created_at": datetime(2026, 11, 24, tzinfo=UTC),
    }
    base.update(kw)
    return MeasurementContext(**base)


def synthetic(n_patients=300, effect=2.0, seed=0):
    rng = np.random.default_rng(seed)
    pids, y, s = [], [], []
    for p in range(n_patients):
        sick = rng.random() < 0.4
        for _ in range(int(rng.integers(1, 3))):
            pids.append(f"p{p}")
            y.append(int(sick))
            s.append(float(1 / (1 + np.exp(-(effect * sick - 1 + rng.normal(0, 1))))))
    return pids, np.array(y), np.array(s)


# --------------------------------------------------------------------------- #
class TestLeakageTable:
    @pytest.mark.requirement("REQ-114")
    def test_derived_datasets_inherit_lineage(self):
        assert assess_leakage("siim-acr-pneumothorax", ["nih-cxr14"])[0] == "known"
        assert assess_leakage("nih-cxr14-google", ["nih-cxr14"])[0] == "known"
        assert assess_leakage("nih-cxr14", ["padchest"])[0] == "none"
        assert assess_leakage("chexpert-valid", ["chexpert"])[0] == "known"
        assert "nih-cxr14" in lineage("rsna-pneumonia")

    @pytest.mark.requirement("REQ-114")
    def test_unknown_key_is_error_never_external(self):
        with pytest.raises(EvidenceError, match="desconhecida"):
            assess_leakage("NIH ChestX-ray14", ["padchest"])
        with pytest.raises(EvidenceError, match="desconhecida"):
            assess_leakage("nih-cxr14", ["PadChest"])

    @pytest.mark.requirement("REQ-114")
    def test_no_substring_matching(self):
        # "nih" é substring de "nih-cxr14", mas não é chave: erro, não "known"/"none".
        with pytest.raises(EvidenceError):
            assess_leakage("nih-cxr14", ["nih"])

    def test_undeclared_training_data_is_possible(self):
        assert assess_leakage("nih-cxr14", [])[0] == "possible"


# --------------------------------------------------------------------------- #
def fake_report(auroc=0.9, lo=0.85, n_pos_pat=50, n_neg_pat=50) -> BinaryProtocolReport:
    c = Counts(
        n_images=200,
        n_patients=100,
        n_pos=80,
        n_neg=120,
        n_pos_patients=n_pos_pat,
        n_neg_patients=n_neg_pat,
    )
    return BinaryProtocolReport(
        protocol_id="2026-11-v1",
        label="X",
        score_scale="probability",
        ci_method="m",
        ci_level=0.95,
        split=SplitSummary(method="m", seed=1, calibration_fraction=0.5, calibration=c, test=c),
        counts=c,
        auroc=Estimate(value=auroc, ci_low=lo, ci_high=0.99),
        auprc=Estimate(value=0.8),
        prevalence=0.4,
        calibration=None,
        calibration_error=None,
        ece_before=None,
        ece_after=None,
        rule_out=None,
        rule_in=None,
        operating_point_errors=(),
        subgroups=(),
        subgroups_unavailable=(),
    )


class TestJudgement:
    @pytest.mark.requirement("REQ-113")
    @pytest.mark.parametrize(
        ("kw", "fragment"),
        [
            ({"auroc": 0.79}, "auroc_min"),
            ({"lo": 0.74}, "auroc_ci_low_min"),
            ({"n_pos_pat": 29}, "min_positive"),
            ({"n_neg_pat": 29}, "min_negative"),
        ],
    )
    def test_each_protocol_bar(self, protocol, kw, fragment):
        reasons = judge_discrimination(fake_report(**kw), protocol)
        assert len(reasons) == 1
        assert fragment in reasons[0]

    @pytest.mark.requirement("REQ-113")
    def test_boundary_values_pass(self, protocol):
        assert judge_discrimination(fake_report(auroc=0.80, lo=0.75, n_pos_pat=30), protocol) == []

    @pytest.mark.requirement("REQ-113")
    def test_measure_binary_decides_from_protocol(self, protocol):
        pids, y, s = synthetic(effect=3.0)
        good = measure_binary(
            ctx(), protocol, patient_ids=pids, y_true=y, y_score=s, n_bootstrap=200
        )
        assert good.record.passes_protocol is True
        assert good.record.failure_reasons == ()
        pids, y, s = synthetic(effect=0.3, seed=1)
        bad = measure_binary(
            ctx(), protocol, patient_ids=pids, y_true=y, y_score=s, n_bootstrap=200
        )
        assert bad.record.passes_protocol is False
        assert any("auroc_min" in r for r in bad.record.failure_reasons)

    @pytest.mark.requirement("REQ-113")
    def test_underpowered_record_is_ineligible_not_refuting(self, protocol):
        # 20 pacientes: abaixo dos mínimos do protocolo §2 -> nem aprova nem refuta.
        pids, y, s = synthetic(n_patients=40, effect=0.3, seed=3)
        ev = measure_binary(ctx(), protocol, patient_ids=pids, y_true=y, y_score=s, n_bootstrap=100)
        rec = ev.record
        assert rec.passes_protocol is None
        assert any("min_positive" in r for r in rec.failure_reasons)
        assert any("NÃO ELEGÍVEL" in c for c in rec.caveats)
        level, _, refuted = compute_level([rec])
        assert level is EvidenceLevel.D
        assert refuted is False

    @pytest.mark.requirement("REQ-113")
    def test_verdict_rule(self):
        assert verdict([]) is True
        assert verdict(["AUROC 0.7 < auroc_min 0.8"]) is False
        assert verdict([f"{ELIGIBILITY_PREFIX}pacientes 3 < min_positive 30", "x"]) is None

    @pytest.mark.requirement("REQ-115")
    def test_record_carries_full_provenance(self, protocol):
        pids, y, s = synthetic()
        ev = measure_binary(ctx(), protocol, patient_ids=pids, y_true=y, y_score=s, n_bootstrap=100)
        rec = ev.record
        assert rec.kind == "measured_external"
        assert rec.leakage == "none"
        assert rec.provenance is not None
        assert rec.provenance.protocol_id == protocol.protocol_id
        assert rec.provenance.dataset_sha256 == SHA
        assert rec.provenance.git_sha == "c" * 40
        assert {"auroc", "auprc", "rule_out_sensitivity", "rule_in_specificity"} <= set(rec.metrics)
        roles = {p.role for p in ev.operating_points}
        assert roles == {"rule_out", "rule_in"}
        assert all("metade de calibração" in p.source for p in ev.operating_points)

    @pytest.mark.requirement("REQ-115")
    def test_dirty_tree_is_flagged(self, protocol):
        pids, y, s = synthetic()
        ev = measure_binary(
            ctx(git_dirty=True), protocol, patient_ids=pids, y_true=y, y_score=s, n_bootstrap=50
        )
        assert any("não commitadas" in c for c in ev.record.caveats)

    @pytest.mark.requirement("REQ-114")
    def test_leakage_from_table_in_record(self, protocol):
        pids, y, s = synthetic()
        ev = measure_binary(
            ctx(training_data=("nih-cxr14", "padchest")),
            protocol,
            patient_ids=pids,
            y_true=y,
            y_score=s,
            n_bootstrap=50,
        )
        assert ev.record.leakage == "known"

    def test_agreement_record(self, protocol):
        rng = np.random.default_rng(0)
        ref = rng.normal(55, 10, 60)
        ev = measure_agreement(
            ctx(label="lvef", condition_id="echo.lvef"),
            protocol,
            patient_ids=[f"p{i}" for i in range(60)],
            measured=ref + rng.normal(0, 2, 60),
            reference=ref,
        )
        assert ev.record.endpoint == "agreement"
        assert ev.record.passes_protocol is True
        assert ev.record.metrics["icc"].value > protocol.icc_min


# --------------------------------------------------------------------------- #
def condition(cid="cxr.pleural_effusion", sources=()):
    return Condition.model_validate(
        {
            "id": cid,
            "name_pt": "Derrame pleural",
            "name_en": "Pleural effusion",
            "group": "cxr",
            "exam_types": ["cxr"],
            "system": "respiratório",
            "code": {"scheme": "SCT", "value": "60046008", "meaning": "Pleural effusion"},
            "sources": list(sources),
        }
    )


def published(rid="pub:1", label="Effusion"):
    return EvidenceRecord(
        id=rid,
        condition_id="cxr.pleural_effusion",
        model_id="xrv-densenet121-pc",
        label=label,
        kind="published",
        metrics={"auroc": Metric(value=0.9)},
        passes_protocol=True,
        source=EvidenceSource(
            title="t",
            url="https://x",
            quote="AUC 0.90",
            peer_reviewed=True,
            verified_at_source=True,
        ),
    )


class TestStoreAndLevels:
    @pytest.mark.requirement("REQ-116")
    def test_write_load_build_deterministic(self, tmp_path, protocol):
        pids, y, s = synthetic(effect=3.0)
        ev = measure_binary(ctx(), protocol, patient_ids=pids, y_true=y, y_score=s, n_bootstrap=100)
        path = write_measured(ev, tmp_path)
        assert path == tmp_path / "measured" / "nih-cxr14" / "xrv-densenet121-pc" / "Effusion.json"
        (tmp_path / "published").mkdir()
        (tmp_path / "published" / "p.json").write_text(
            json.dumps([published().model_dump(mode="json")]), encoding="utf-8"
        )
        records = load_records(tmp_path)
        assert len(records) == 2
        conds = [condition(sources=[{"model_id": "other", "label": "E"}])]
        a = build_levels(records, conds, protocol)
        b = build_levels(list(reversed(records)), conds, protocol)
        assert render_levels(a) == render_levels(b)
        LevelsFile.model_validate_json(render_levels(a))
        levels = {(e.model_id, e.level.value) for e in a.entries}
        assert levels == {("xrv-densenet121-pc", "A"), ("other", "D")}
        entry = next(e for e in a.entries if e.model_id == "xrv-densenet121-pc")
        assert entry.basis == (ev.record.id,)
        assert entry.headline
        assert "auroc" in entry.headline

    @pytest.mark.requirement("REQ-116")
    def test_check_detects_stale(self, tmp_path, protocol):
        conds = [condition()]
        empty = build_levels([], conds, protocol)
        out = tmp_path / "levels.json"
        assert check_levels(empty, out)[0] is False  # ausente
        out.write_text(render_levels(empty), encoding="utf-8")
        assert check_levels(empty, out)[0] is True
        changed = build_levels([published()], conds, protocol)
        assert changed.evidence_digest != empty.evidence_digest
        assert check_levels(changed, out)[0] is False

    @pytest.mark.requirement("REQ-116")
    def test_unknown_condition_and_protocol_mismatch_fail(self, protocol):
        with pytest.raises(EvidenceError, match="condição inexistente"):
            build_levels([published()], [condition("cxr.other")], protocol)
        pids, y, s = synthetic()
        ev = measure_binary(ctx(), protocol, patient_ids=pids, y_true=y, y_score=s, n_bootstrap=50)
        prov = ev.record.provenance.model_copy(update={"protocol_id": "2020-01-v0"})
        old = ev.record.model_copy(update={"provenance": prov})
        with pytest.raises(EvidenceError, match="re-execute"):
            build_levels([old], [condition()], protocol)

    def test_duplicate_ids_rejected(self, tmp_path):
        (tmp_path / "published").mkdir()
        for name in ("a.json", "b.json"):
            (tmp_path / "published" / name).write_text(
                json.dumps(published().model_dump(mode="json")), encoding="utf-8"
            )
        with pytest.raises(EvidenceError, match="duplicado"):
            load_records(tmp_path)

    @pytest.mark.requirement("REQ-115")
    def test_json_passes_honesty_sha256_rule(self, protocol):
        """Cada linha com chave '...sha256' traz o hash de 64 hex na própria linha (R3)."""
        sys.path.insert(0, str(REPO / "scripts"))
        try:
            import check_honesty as guard
        finally:
            sys.path.pop(0)
        pids, y, s = synthetic()
        ev = measure_binary(ctx(), protocol, patient_ids=pids, y_true=y, y_score=s, n_bootstrap=50)
        text = dump_json(ev.model_dump(mode="json"))
        json.loads(text)
        for line in text.splitlines():
            m = guard.SHA_FIELD_RE.search(line)
            if m:
                assert guard.SHA_VALID_RE.match(m.group(1).strip().strip('",')), line


# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not HAS_PARQUET, reason="pandas/pyarrow ausentes (extra eval)")
class TestScripts:
    def _parquet(self, path: Path, multi: bool = False):
        import pandas as pd

        pids, y, s = synthetic(effect=3.0)
        df = pd.DataFrame(
            {
                "patient_id": pids,
                "image_id": [f"i{i}" for i in range(len(pids))],
                "sex": ["M" if i % 2 else "F" for i in range(len(pids))],
                "age": [float(10 + i % 70) for i in range(len(pids))],
                "view": ["PA" if i % 3 else "AP" for i in range(len(pids))],
            }
        )
        yt = y.astype(float)
        yt[:5] = np.nan  # rótulo ausente: excluído, nunca negativo
        if multi:
            df["y_true__Effusion"] = yt
            df["y_score__Effusion"] = s
        else:
            df["y_true"] = yt
            df["y_score"] = s
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
        return path

    @pytest.mark.requirement("REQ-115")
    def test_evidence_run_then_build_levels(self, tmp_path):
        sys.path.insert(0, str(REPO / "scripts"))
        try:
            import evidence_build_levels
            import evidence_run
        finally:
            sys.path.pop(0)
        weights = tmp_path / "w.pt"
        weights.write_bytes(b"pesos")
        preds = self._parquet(tmp_path / "preds" / "m1" / "nih-cxr14" / "Effusion.parquet")
        out = tmp_path / "evidence"
        argv = [
            "--model-id", "m1",
            "--dataset-id", "nih-cxr14",
            "--preds-root", str(tmp_path / "preds"),
            "--label", "Effusion",
            "--condition", "Effusion=cxr.pleural_effusion",
            "--training-data", "padchest",
            "--weights", str(weights),
            "--out", str(out),
            "--n-bootstrap", "50",
            "--allow-dirty",
        ]  # fmt: skip
        assert evidence_run.main(argv) == 0
        files = list((out / "measured").rglob("*.json"))
        assert len(files) == 1
        env = json.loads(files[0].read_text(encoding="utf-8"))
        assert env["inputs"]["n_missing_label_excluded"] == 5
        assert env["record"]["provenance"]["weights_sha256"] == {
            "w.pt": hashlib.sha256(b"pesos").hexdigest()
        }
        assert preds.is_file()

        catalog = tmp_path / "catalog"
        catalog.mkdir()
        (catalog / "protocol.yaml").write_text(
            (REPO / "src/radiologyai/catalog/protocol.yaml").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        import yaml

        (catalog / "conditions.yaml").write_text(
            yaml.safe_dump({"conditions": [condition().model_dump(mode="json")]}),
            encoding="utf-8",
        )
        common = ["--evidence-root", str(out), "--catalog-dir", str(catalog)]
        assert evidence_build_levels.main([*common, "--check"]) == 1
        assert evidence_build_levels.main(common) == 0
        assert evidence_build_levels.main([*common, "--check"]) == 0
        first = (catalog / "levels.json").read_bytes()
        assert evidence_build_levels.main(common) == 0
        assert (catalog / "levels.json").read_bytes() == first

    def test_n_bootstrap_override_refused_for_repo_evidence(self, tmp_path):
        sys.path.insert(0, str(REPO / "scripts"))
        try:
            import evidence_run
        finally:
            sys.path.pop(0)
        preds = self._parquet(tmp_path / "p.parquet", multi=True)
        with pytest.raises(SystemExit, match="n-bootstrap"):
            evidence_run.main(
                [
                    "--model-id", "m1",
                    "--dataset-id", "nih-cxr14",
                    "--predictions", str(preds),
                    "--all-labels",
                    "--condition", "Effusion=cxr.pleural_effusion",
                    "--training-data", "padchest",
                    "--weights-sha256", f"w={'a' * 64}",
                    "--n-bootstrap", "10",
                ]
            )  # fmt: skip

    def test_multilabel_reader(self, tmp_path):
        from radiologyai.evidence.predictions import labels_in, read_label

        preds = self._parquet(tmp_path / "p.parquet", multi=True)
        assert labels_in(preds) == ["Effusion"]
        lp = read_label(preds, "Effusion")
        assert lp.n_missing_label == 5
        assert set(lp.subgroups) == {"sex", "age", "view"}
        assert re.fullmatch(r"[0-9a-f]{64}", lp.dataset_sha256)
        assert lp.dataset_sha256 != lp.predictions_sha256
