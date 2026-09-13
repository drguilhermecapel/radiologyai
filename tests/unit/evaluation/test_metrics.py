"""Métricas — determinismo, incerteza e recusa de casos degenerados."""

from __future__ import annotations

import numpy as np
import pytest

from radiologyai.errors import EvaluationError, PatientLeakageError
from radiologyai.evaluation import (
    assert_patient_disjoint,
    auroc,
    auroc_with_ci,
    expected_calibration_error,
    operating_point_at_sensitivity,
)


class TestAUROC:
    def test_perfect_separation(self):
        y = np.array([0, 0, 1, 1])
        s = np.array([0.1, 0.2, 0.8, 0.9])
        assert auroc(y, s) == pytest.approx(1.0)

    def test_inverted_separation(self):
        y = np.array([0, 0, 1, 1])
        s = np.array([0.9, 0.8, 0.2, 0.1])
        assert auroc(y, s) == pytest.approx(0.0)

    def test_all_ties_give_chance(self):
        y = np.array([0, 1, 0, 1])
        s = np.array([0.5, 0.5, 0.5, 0.5])
        assert auroc(y, s) == pytest.approx(0.5)

    def test_single_class_raises_instead_of_nan(self):
        """O legado gravou mean_auc: NaN em test_samples: 1. Aqui é erro."""
        with pytest.raises(EvaluationError, match="AUROC indefinido"):
            auroc(np.array([1, 1, 1]), np.array([0.2, 0.5, 0.9]))

    def test_empty_raises(self):
        with pytest.raises(EvaluationError, match="vazio"):
            auroc(np.array([], dtype=int), np.array([]))

    def test_non_binary_raises(self):
        with pytest.raises(EvaluationError, match="binário"):
            auroc(np.array([0, 1, 2]), np.array([0.1, 0.5, 0.9]))


class TestBootstrapCI:
    def test_ci_brackets_point_estimate(self):
        rng = np.random.default_rng(7)
        y = np.repeat([0, 1], 100)
        s = np.concatenate([rng.normal(0.3, 0.1, 100), rng.normal(0.7, 0.1, 100)])
        r = auroc_with_ci(y, s, n_bootstrap=200, seed=42)
        assert r.ci95_low <= r.auroc <= r.ci95_high

    @pytest.mark.requirement("REQ-050")
    def test_same_seed_is_bit_reproducible(self):
        """Reprodutibilidade é requisito de V&V, não conveniência."""
        rng = np.random.default_rng(7)
        y = np.repeat([0, 1], 50)
        s = np.concatenate([rng.normal(0.3, 0.2, 50), rng.normal(0.7, 0.2, 50)])
        a = auroc_with_ci(y, s, n_bootstrap=100, seed=123)
        b = auroc_with_ci(y, s, n_bootstrap=100, seed=123)
        assert a == b

    def test_different_seed_changes_ci(self):
        rng = np.random.default_rng(7)
        y = np.repeat([0, 1], 50)
        s = np.concatenate([rng.normal(0.3, 0.3, 50), rng.normal(0.7, 0.3, 50)])
        a = auroc_with_ci(y, s, n_bootstrap=100, seed=1)
        b = auroc_with_ci(y, s, n_bootstrap=100, seed=2)
        assert a.auroc == b.auroc, "estimativa pontual não depende de seed"
        assert (a.ci95_low, a.ci95_high) != (b.ci95_low, b.ci95_high)

    def test_reports_support_counts(self):
        y = np.array([0, 0, 0, 1, 1])
        s = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
        r = auroc_with_ci(y, s, n_bootstrap=50, seed=1)
        assert (r.n_pos, r.n_neg) == (2, 3)
        assert "bootstrap" in r.ci_method


class TestOperatingPoint:
    def test_reaches_target_sensitivity(self):
        y = np.array([0] * 50 + [1] * 50)
        s = np.concatenate([np.linspace(0.0, 0.5, 50), np.linspace(0.5, 1.0, 50)])
        op = operating_point_at_sensitivity(y, s, target_sensitivity=0.90)
        assert op.sensitivity >= 0.90

    def test_impossible_target_raises(self):
        y = np.array([0, 1])
        s = np.array([0.5, 0.5])
        op = operating_point_at_sensitivity(y, s, target_sensitivity=1.0)
        assert op.sensitivity == pytest.approx(1.0)

    def test_invalid_target_raises(self):
        y = np.array([0, 1])
        s = np.array([0.1, 0.9])
        with pytest.raises(EvaluationError, match="target_sensitivity"):
            operating_point_at_sensitivity(y, s, target_sensitivity=1.5)


class TestECE:
    def test_perfectly_calibrated_is_low(self):
        y = np.array([0] * 90 + [1] * 10)
        s = np.array([0.05] * 90 + [0.95] * 10)
        assert expected_calibration_error(y, s) < 0.1

    def test_overconfident_is_high(self):
        y = np.array([0] * 50 + [1] * 50)
        s = np.array([0.99] * 100)
        assert expected_calibration_error(y, s) > 0.4


@pytest.mark.requirement("REQ-051")
class TestPatientLeakage:
    def test_disjoint_splits_pass(self):
        assert_patient_disjoint({"train": ["p1", "p2"], "test": ["p3"]})

    def test_shared_patient_raises(self):
        with pytest.raises(PatientLeakageError, match="vazamento por paciente"):
            assert_patient_disjoint({"train": ["p1", "p2"], "test": ["p2", "p3"]})

    def test_error_names_both_splits(self):
        with pytest.raises(PatientLeakageError) as exc:
            assert_patient_disjoint({"train": ["p1"], "val": ["p1"], "test": ["p9"]})
        message = str(exc.value)
        assert "train" in message
        assert "val" in message

    def test_three_way_check(self):
        assert_patient_disjoint({"train": ["a"], "val": ["b"], "test": ["c"]})
