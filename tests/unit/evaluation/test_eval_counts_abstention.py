"""Contagem de pacientes negativos, abstenção na concordância e réplicas indefinidas."""

from __future__ import annotations

import numpy as np
import pytest

from radiologyai.catalog import load_protocol
from radiologyai.errors import EvaluationError
from radiologyai.evaluation.agreement import evaluate_agreement
from radiologyai.evaluation.metrics import (
    MIN_DEFINED_REPLICATE_FRACTION,
    bootstrap_interval,
    percentile_ci,
)
from radiologyai.evaluation.protocol_eval import counts_for
from radiologyai.evidence.measure import ELIGIBILITY_PREFIX, judge_agreement, verdict


@pytest.fixture(scope="module")
def protocol():
    return load_protocol()


# --------------------------------------------------------------------------- #
class TestNegativePatients:
    @pytest.mark.requirement("REQ-181")
    def test_patient_with_both_kinds_of_image_counts_only_as_positive(self):
        """30 pacientes, cada um com uma imagem positiva e uma negativa: zero negativos."""
        pids = [f"p{i}" for i in range(30) for _ in range(2)]
        y = np.array([1, 0] * 30)
        c = counts_for(pids, y)
        assert (c.n_pos_patients, c.n_neg_patients) == (30, 0)
        assert c.n_pos_patients + c.n_neg_patients == c.n_patients

    @pytest.mark.requirement("REQ-181")
    def test_strata_are_disjoint(self):
        pids = ["a", "a", "b", "c", "c", "d"]
        y = np.array([1, 0, 0, 0, 0, 1])
        c = counts_for(pids, y)
        assert (c.n_patients, c.n_pos_patients, c.n_neg_patients) == (4, 2, 2)


# --------------------------------------------------------------------------- #
def _ef(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    ref = rng.uniform(20, 70, n)
    return ref + rng.normal(0, 2, n), ref


class TestAgreementAbstention:
    @pytest.mark.requirement("REQ-182")
    def test_model_abstaining_on_hard_cases_cannot_pass(self, protocol):
        """Revisor: abstém-se em 70 de 100 e concorda nos 30 restantes."""
        meas, ref = _ef(30)
        report = evaluate_agreement(
            patient_ids=[f"k{i}" for i in range(30)],
            measured=meas,
            reference=ref,
            seed=1,
            n_bootstrap=100,
            abstained_patient_ids=[f"a{i}" for i in range(70)],
        )
        assert (report.n_cases, report.n_abstained, report.n_cases_with_reference) == (30, 70, 100)
        assert report.n_patients_with_reference == 100
        assert report.icc.value > protocol.icc_min  # nos que sobraram concorda...
        reasons = judge_agreement(report, protocol)
        assert verdict(reasons) is False  # ...mas não aprova
        assert any("abstenção em 70 de 100" in r for r in reasons)

    @pytest.mark.requirement("REQ-182")
    def test_eligibility_counts_patients_with_reference(self, protocol):
        meas, ref = _ef(20)
        report = evaluate_agreement(
            patient_ids=[f"k{i}" for i in range(20)],
            measured=meas,
            reference=ref,
            seed=1,
            n_bootstrap=50,
            abstained_patient_ids=[f"a{i}" for i in range(15)],
        )
        reasons = judge_agreement(report, protocol)
        assert not any(r.startswith(ELIGIBILITY_PREFIX) for r in reasons)  # 35 >= 30
        assert verdict(reasons) is False

    @pytest.mark.requirement("REQ-182")
    def test_protocol_states_zero_tolerance(self, protocol):
        assert protocol.max_abstention_fraction == 0.0

    @pytest.mark.requirement("REQ-182")
    def test_no_abstention_still_passes(self, protocol):
        meas, ref = _ef(40)
        report = evaluate_agreement(
            patient_ids=[f"k{i}" for i in range(40)],
            measured=meas,
            reference=ref,
            seed=1,
            n_bootstrap=50,
        )
        assert report.n_abstained == 0
        assert judge_agreement(report, protocol) == []


# --------------------------------------------------------------------------- #
class TestUndefinedReplicates:
    @pytest.mark.requirement("REQ-183")
    def test_undefined_replicates_are_counted(self):
        samples = np.r_[np.linspace(0.6, 0.9, 98), [np.nan, np.nan]]
        ci = bootstrap_interval(samples, 0.95)
        assert (ci.n_defined, ci.n_replicates, ci.n_undefined) == (98, 100, 2)
        assert ci.low is not None

    @pytest.mark.requirement("REQ-183")
    def test_too_many_undefined_replicates_leave_ci_undefined(self):
        n_bad = int(100 * (1 - MIN_DEFINED_REPLICATE_FRACTION)) + 1
        samples = np.r_[np.linspace(0.6, 0.9, 100 - n_bad), [np.nan] * n_bad]
        ci = bootstrap_interval(samples, 0.95)
        assert (ci.low, ci.high) == (None, None)
        with pytest.raises(EvaluationError, match="réplicas"):
            percentile_ci(samples, 0.95)

    @pytest.mark.requirement("REQ-183")
    def test_agreement_reports_replicate_counts(self):
        """Paciente com muitas linhas iguais: réplicas com variância zero ficam NaN e contadas."""
        meas = np.array([50.0] * 8 + [40.0, 60.0])
        ref = np.array([50.0] * 8 + [41.0, 59.0])
        pids = ["dup"] * 8 + ["x", "y"]
        report = evaluate_agreement(
            patient_ids=pids, measured=meas, reference=ref, seed=3, n_bootstrap=200
        )
        assert report.icc.n_replicates == 200
        assert report.icc.n_defined_replicates is not None
        assert report.icc.n_defined_replicates < 200
        if report.icc.n_defined_replicates < MIN_DEFINED_REPLICATE_FRACTION * 200:
            assert report.icc.ci_low is None

    @pytest.mark.requirement("REQ-183")
    def test_record_caveat_lists_undefined_replicates(self):
        from datetime import UTC, datetime

        from radiologyai.evidence.measure import MeasurementContext, measure_agreement

        protocol = load_protocol().model_copy(update={"bootstrap_iterations": 200})
        meas = np.array([50.0] * 8 + [40.0, 60.0])
        ref = np.array([50.0] * 8 + [41.0, 59.0])
        ev = measure_agreement(
            MeasurementContext(
                model_id="m",
                label="lv_ef",
                condition_id="echo.reduced_lvef",
                dataset_id="camus",
                training_data=("medsam2-training-corpus",),
                weights_sha256={"w": "a" * 64},
                git_sha="c" * 40,
                command="t",
                dataset_sha256="b" * 64,
                created_at=datetime(2026, 11, 27, tzinfo=UTC),
            ),  # fmt: skip
            protocol,
            patient_ids=["dup"] * 8 + ["x", "y"],
            measured=meas,
            reference=ref,
        )
        assert any(
            "Réplicas de bootstrap com estatística indefinida" in c for c in ev.record.caveats
        )
