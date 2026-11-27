"""levels.json re-julga cada registro e limita o nível de modelo não distribuído."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from radiologyai.catalog import load_protocol
from radiologyai.catalog.schema import Condition, EvidenceLevel
from radiologyai.errors import EvidenceError
from radiologyai.evidence.levels import build_levels
from radiologyai.evidence.measure import (
    DIRTY_CAVEAT,
    MeasuredEvidence,
    MeasurementContext,
    measure_binary,
    rejudge,
)

WEIGHTS = {"probe.safetensors": "a" * 64}


@pytest.fixture(scope="module")
def protocol():
    return load_protocol()


def ctx(**kw) -> MeasurementContext:
    base = {
        "model_id": "m1",
        "label": "tuberculosis",
        "condition_id": "cxr.tuberculosis",
        "dataset_id": "tb-montgomery",
        "training_data": ("tb-shenzhen",),
        "weights_sha256": WEIGHTS,
        "git_sha": "c" * 40,
        "command": "evidence_run.py --teste",
        "dataset_sha256": "b" * 64,
        "created_at": datetime(2026, 11, 27, tzinfo=UTC),
    }
    base.update(kw)
    return MeasurementContext(**base)


def data(n_pos: int, n_neg: int, effect: float = 3.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = np.r_[np.ones(n_pos, dtype=int), np.zeros(n_neg, dtype=int)]
    s = rng.normal(0.0, 1.0, y.size) + effect * y
    return [f"p{i}" for i in range(y.size)], y, s


def condition() -> Condition:
    return Condition.model_validate(
        {
            "id": "cxr.tuberculosis",
            "name_pt": "Tuberculose",
            "name_en": "Tuberculosis",
            "group": "cxr",
            "exam_types": ["cxr"],
            "system": "respiratório",
            "code": {"scheme": "99MEDAI", "value": "cxr.tuberculosis", "meaning": "TB"},
        }
    )


def manifests(tmp_path: Path, weights=WEIGHTS, visibility: str = "main", model="m1") -> Path:
    d = tmp_path / "manifests"
    d.mkdir(exist_ok=True)
    lines = [f"id: {model}", f"visibility_default: {visibility}", "weights:"]
    for fname, sha in weights.items():
        lines += [f"- key: {fname}", f"  filename: {fname}", f"  sha256: {sha}"]
    (d / f"{model}.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return d


def passing(protocol, **kw) -> MeasuredEvidence:
    pids, y, s = data(160, 160)
    return measure_binary(
        ctx(**kw),
        protocol,
        patient_ids=pids,
        y_true=y,
        y_score=s,
        score_scale="logit",
        n_bootstrap=60,
    )


def tamper(ev: MeasuredEvidence, **record_changes) -> MeasuredEvidence:
    rec = ev.record.model_copy(update=record_changes)
    return ev.model_copy(update={"record": rec})


# --------------------------------------------------------------------------- #
class TestRejudge:
    @pytest.mark.requirement("REQ-180")
    def test_untouched_record_rejudges_clean(self, protocol):
        ev = passing(protocol)
        assert rejudge(ev, protocol) == []
        assert ev.declared is not None
        assert ev.declared.training_data == ("tb-shenzhen",)

    @pytest.mark.requirement("REQ-180")
    def test_hand_flipped_verdict_of_ineligible_record_is_refused(self, protocol, tmp_path):
        """O caso do revisor: 29 positivos (não elegível) editado para passes_protocol=true."""
        pids, y, s = data(58, 80)  # metade de teste: 29 pacientes positivos
        ev = measure_binary(
            ctx(),
            protocol,
            patient_ids=pids,
            y_true=y,
            y_score=s,
            score_scale="logit",
            n_bootstrap=60,
        )
        assert ev.record.passes_protocol is None
        forged = tamper(ev, passes_protocol=True, failure_reasons=())
        assert any("passes_protocol" in m for m in rejudge(forged, protocol))
        with pytest.raises(EvidenceError, match="re-julgamento"):
            build_levels([forged], [condition()], protocol, manifests_dir=manifests(tmp_path))
        # o original, honesto, constrói (D: não elegível não aprova)
        levels = build_levels([ev], [condition()], protocol, manifests_dir=manifests(tmp_path))
        assert levels.entries[0].level is EvidenceLevel.D

    @pytest.mark.requirement("REQ-180")
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("leakage", "none"),
            ("kind", "measured_internal"),
            ("n_positive", 999),
        ],
    )
    def test_any_hand_edit_is_refused(self, protocol, field, value):
        ev = passing(protocol, dataset_id="tb-shenzhen", training_data=("tb-montgomery",))
        if field == "leakage":
            ev = passing(protocol, training_data=("tb-montgomery",))  # leakage known
        forged = tamper(ev, **{field: value})
        assert rejudge(forged, protocol)

    @pytest.mark.requirement("REQ-180")
    def test_edited_metric_is_refused(self, protocol):
        ev = passing(protocol)
        metrics = dict(ev.record.metrics)
        metrics["auroc"] = metrics["auroc"].model_copy(update={"value": 0.99})
        assert any("métricas" in m for m in rejudge(tamper(ev, metrics=metrics), protocol))

    @pytest.mark.requirement("REQ-180")
    def test_envelope_without_declarations_cannot_be_rejudged(self, protocol):
        ev = passing(protocol).model_copy(update={"declared": None})
        with pytest.raises(EvidenceError, match="declarações"):
            rejudge(ev, protocol)

    @pytest.mark.requirement("REQ-180")
    def test_bare_measured_record_is_refused(self, protocol, tmp_path):
        ev = passing(protocol)
        with pytest.raises(EvidenceError, match="sem envelope"):
            build_levels([ev.record], [condition()], protocol, manifests_dir=manifests(tmp_path))

    @pytest.mark.requirement("REQ-180")
    def test_dirty_record_is_left_out_with_reason(self, protocol, tmp_path):
        ev = passing(protocol, git_dirty=True)
        assert DIRTY_CAVEAT in ev.record.caveats
        assert rejudge(ev, protocol) == []
        (entry,) = build_levels(
            [ev], [condition()], protocol, manifests_dir=manifests(tmp_path)
        ).entries
        assert entry.level is EvidenceLevel.D
        assert entry.basis == ()
        assert any("git_dirty" in reason for reason in entry.limited_by)

    @pytest.mark.requirement("REQ-180")
    def test_declarations_must_match_plan(self, protocol, tmp_path):
        from radiologyai.evidence.plan import Plan

        ev = passing(protocol)
        ev = ev.model_copy(update={"inputs": {"plan_job": "j1"}})
        job = {
            "id": "j1",
            "status": "measure",
            "model_id": "m1",
            "dataset_id": "tb-montgomery",
            "label": "tuberculosis",
            "condition_id": "cxr.tuberculosis",
            "predictions": "x.parquet",
            "training_data": ["tb-shenzhen"],
            "weights": {"p": "probe.safetensors"},
        }
        plan = Plan.model_validate(
            {"format": "radiologyai.evidence/plan-v1", "protocol_id": "2026-11-v1",
             "preds_root": "/x", "jobs": [job]}
        )  # fmt: skip
        mdir = manifests(tmp_path)
        assert build_levels([ev], [condition()], protocol, manifests_dir=mdir, plan=plan)
        sneaky = plan.model_copy(
            update={
                "jobs": (plan.jobs[0].model_copy(update={"selection_data": ("tb-montgomery",)}),)
            }
        )
        with pytest.raises(EvidenceError, match="selection_data"):
            build_levels([ev], [condition()], protocol, manifests_dir=mdir, plan=sneaky)
        orphan = ev.model_copy(update={"inputs": {"plan_job": "nao-existe"}})
        with pytest.raises(EvidenceError, match="fora do plano"):
            build_levels([orphan], [condition()], protocol, manifests_dir=mdir, plan=plan)


# --------------------------------------------------------------------------- #
class TestDistribution:
    @pytest.mark.requirement("REQ-185")
    def test_clean_passing_measurement_of_distributed_model_is_a(self, protocol, tmp_path):
        (entry,) = build_levels(
            [passing(protocol)], [condition()], protocol, manifests_dir=manifests(tmp_path)
        ).entries
        assert (entry.level, entry.distributed, entry.limited_by) == (EvidenceLevel.A, True, ())

    @pytest.mark.requirement("REQ-185")
    def test_model_without_manifest_is_capped_to_d_and_flagged(self, protocol, tmp_path):
        (entry,) = build_levels(
            [passing(protocol)], [condition()], protocol, manifests_dir=tmp_path / "vazio"
        ).entries
        assert entry.level is EvidenceLevel.D
        assert entry.visibility == "research"
        assert entry.distributed is False
        assert entry.refuted_by_measurement is False
        assert any("sem manifesto" in r for r in entry.limited_by)
        assert entry.headline is not None
        assert entry.headline.startswith("limitado a D (a medição sustentaria A)")

    @pytest.mark.requirement("REQ-185")
    def test_measured_weights_not_in_manifest_are_not_distributed(self, protocol, tmp_path):
        """O hash da sonda medida precisa constar do manifesto (não só o do codificador)."""
        ev = passing(protocol)
        mdir = manifests(tmp_path, weights={"probe.safetensors": "d" * 64})
        (entry,) = build_levels([ev], [condition()], protocol, manifests_dir=mdir).entries
        assert (entry.level, entry.distributed) == (EvidenceLevel.D, False)
        assert any("não constam do manifesto" in r for r in entry.limited_by)

    @pytest.mark.requirement("REQ-185")
    def test_research_only_manifest_caps_level(self, protocol, tmp_path):
        mdir = manifests(tmp_path, visibility="research")
        (entry,) = build_levels(
            [passing(protocol)], [condition()], protocol, manifests_dir=mdir
        ).entries
        assert entry.level is EvidenceLevel.D
        assert entry.distributed is True
        assert any("visibility_default: research" in r for r in entry.limited_by)

    @pytest.mark.requirement("REQ-185")
    def test_level_entry_refuses_non_distributed_above_d(self):
        from pydantic import ValidationError

        from radiologyai.catalog.schema import LevelEntry

        with pytest.raises(ValidationError, match="não distribuído"):
            LevelEntry(
                condition_id="c", model_id="m", label="l", level=EvidenceLevel.C,
                visibility="main", basis=("r",), distributed=False,
            )  # fmt: skip


# --------------------------------------------------------------------------- #
class TestReferenceDerivedInput:
    @pytest.mark.requirement("REQ-186")
    def test_oracle_input_never_reaches_a(self, protocol, tmp_path):
        ev = passing(protocol, reference_derived_input="caixa = caixa da máscara + 25%")
        assert ev.record.leakage == "none"
        assert ev.record.reference_derived_input
        assert any("derivada do padrão de referência" in c for c in ev.record.caveats)
        (entry,) = build_levels(
            [ev], [condition()], protocol, manifests_dir=manifests(tmp_path)
        ).entries
        assert entry.level is EvidenceLevel.C
        assert any("nível A vedado" in r for r in entry.limited_by)
