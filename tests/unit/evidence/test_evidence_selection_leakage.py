"""Reuso do dataset de avaliação na seleção do modelo é vazamento (REQ-119)."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml

from radiologyai.catalog.schema import EvidenceLevel, EvidenceRecord, compute_level
from radiologyai.evidence import leakage as leak
from radiologyai.evidence.leakage import (
    DatasetInfo,
    assess_leakage,
    assess_training_leakage,
    selection_overlap,
)
from radiologyai.evidence.plan import load_plan

REPO = Path(__file__).resolve().parents[3]
RADDINO = ("mimic-cxr", "chexpert", "nih-cxr14", "padchest", "brax")
SELECTION = ("tb-shenzhen", "tb-montgomery", "fracatlas", "grazpedwri-dx-sample")
PROVENANCE = {
    "git_sha": "a" * 40,
    "protocol_id": "2026-11-v1",
    "dataset_id": "ds",
    "dataset_sha256": "0" * 64,
    "weights_sha256": {"w": "1" * 64},
    "command": "c",
    "created_at": datetime(2026, 11, 27, tzinfo=UTC),
}


def record(rid: str, **kw: Any) -> EvidenceRecord:
    base: dict[str, Any] = {
        "id": rid,
        "condition_id": "cxr.tuberculosis",
        "model_id": "m",
        "label": "tb",
        "kind": "measured_external",
        "provenance": PROVENANCE,
        "n_total": 300,
        "n_positive": 150,
        "passes_protocol": True,
    }
    base.update(kw)
    return EvidenceRecord.model_validate(base)


class TestSelectionLeakage:
    @pytest.mark.requirement("REQ-119")
    def test_shenzhen_reused_for_encoder_selection_is_known(self):
        """O caso do revisor: treino Montgomery, avaliação Shenzhen, codificador escolhido em
        Shenzhen. Pela linhagem do treino seria 'none' (o antigo nível A)."""
        train = ("tb-montgomery", *RADDINO)
        assert assess_training_leakage("tb-shenzhen", train)[0] == "none"
        leakage, note = assess_leakage("tb-shenzhen", train, SELECTION)
        assert leakage == "known"
        assert "seleção" in note
        assert selection_overlap("tb-shenzhen", SELECTION) == ("tb-shenzhen",)

    @pytest.mark.requirement("REQ-119")
    def test_selection_reaches_derived_datasets(self):
        assert selection_overlap("grazpedwri-dx-sample-test30", SELECTION) == (
            "grazpedwri-dx-sample",
        )
        assert assess_leakage("tb-montgomery", ("padchest",), SELECTION)[0] == "known"
        assert assess_leakage("vindr-cxr", ("tb-shenzhen",), SELECTION)[0] == "none"

    @pytest.mark.requirement("REQ-119")
    def test_unknown_selection_key_is_error(self):
        from radiologyai.errors import EvidenceError

        with pytest.raises(EvidenceError, match="desconhecida"):
            assess_leakage("tb-shenzhen", ("tb-montgomery",), ("Shenzhen",))

    @pytest.mark.requirement("REQ-119")
    def test_record_with_selection_overlap_must_be_known(self):
        with pytest.raises(ValueError, match="seleção"):
            record("r", leakage="none", selection_overlap=("tb-shenzhen",))
        ok = record("r", leakage="known", training_leakage="none", selection_overlap=("x",))
        assert ok.base_leakage == "none"

    @pytest.mark.requirement("REQ-119")
    def test_selection_reuse_blocks_a_but_level_is_c(self):
        rec = record("r", leakage="known", training_leakage="none", selection_overlap=("x",))
        assert compute_level([rec]) == (EvidenceLevel.C, ["r"], False)

    @pytest.mark.requirement("REQ-119")
    @pytest.mark.requirement("REQ-093")
    def test_selection_reuse_does_not_rescue_from_refutation(self):
        """Reuso só infla a estimativa: a reprovação continua refutando (fratura no GRAZPEDWRI)."""
        failed = record(
            "ext",
            leakage="known",
            training_leakage="none",
            selection_overlap=("grazpedwri-dx-sample",),
            passes_protocol=False,
            failure_reasons=("AUROC 0.69 < 0.8",),
        )
        internal = record("int", kind="measured_internal", leakage="known")
        level, basis, refuted = compute_level([failed, internal])
        assert (level, basis, refuted) == (EvidenceLevel.D, ["ext"], True)

    @pytest.mark.requirement("REQ-114")
    def test_possible_overlap_follows_training_lineage(self, monkeypatch):
        """Treino derivado de BUSI herda a sobreposição plausível BUSI x BrEaST-Lesions."""
        monkeypatch.setitem(leak.DATASETS, "busi-clean", DatasetInfo("busi-clean", ("busi",)))
        assert assess_leakage("breast-lesions-usg", ("busi-clean",))[0] == "possible"
        assert assess_leakage("breast-lesions-usg", ("busi",))[0] == "possible"


class TestRepositoryDeclarations:
    @pytest.mark.requirement("REQ-119")
    def test_every_raddino_probe_declares_the_selection_sets(self):
        plan = load_plan(REPO / "evidence" / "measured" / "plan.yaml")
        probes = [j for j in plan.jobs if j.status == "measure" and j.weights]
        raddino = [
            j for j in probes if any("xr-probe" in p and "model.safetensors" in p
                                     for p in j.weights.values())
        ]  # fmt: skip
        assert len(raddino) >= 8
        for job in raddino:
            assert set(SELECTION) <= set(job.selection_data), job.id

    @pytest.mark.requirement("REQ-119")
    def test_encoder_selection_results_are_committed_and_hashed(self):
        d = REPO / "evidence" / "selection" / "xr-encoder"
        meta = yaml.safe_load((d / "selection.yaml").read_text(encoding="utf-8"))
        data = (d / "compare.json").read_bytes()
        assert hashlib.sha256(data).hexdigest() == meta["results_sha256"]
        assert [s["key"] for s in meta["selection_data"]] == list(SELECTION)
        for key in SELECTION:
            leak.lineage(key)
