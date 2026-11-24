"""Estatística do protocolo conferida contra valores de referência.

sklearn (AUROC, AUPRC), scipy, o exemplo de ICC de Shrout & Fleiss (1979) e
fórmulas diretas (DeLong por componentes estruturais, Bland-Altman).
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from radiologyai.errors import EvaluationError
from radiologyai.evaluation.agreement import bland_altman, evaluate_agreement, icc
from radiologyai.evaluation.compare import delong_test, delong_variance, paired_cluster_bootstrap
from radiologyai.evaluation.detection import evaluate_detection
from radiologyai.evaluation.metrics import (
    ClusterBootstrap,
    auroc,
    expected_calibration_error,
    operating_point_at_sensitivity,
    percentile_ci,
    threshold_for_sensitivity,
    threshold_for_specificity,
    weighted_auroc,
    weighted_average_precision,
    weighted_ece,
    weighted_rates_at,
)
from radiologyai.evaluation.segmentation import dice_coefficient, evaluate_dice


def clustered_data(n_patients: int = 120, seed: int = 0):
    """Pacientes com 1-8 imagens, efeito aleatório por paciente (imagens correlacionadas)."""
    rng = np.random.default_rng(seed)
    pids, y, s = [], [], []
    for p in range(n_patients):
        k = int(rng.integers(1, 9))
        sick = rng.random() < 0.4
        effect = rng.normal(0, 1.0)
        for _ in range(k):
            label = int(sick and rng.random() < 0.8)
            pids.append(f"p{p}")
            y.append(label)
            s.append(1 / (1 + np.exp(-(1.2 * label + effect + rng.normal(0, 0.5)))))
    return np.array(pids), np.array(y), np.array(s)


class TestWeightedStatsMatchSklearn:
    @pytest.mark.requirement("REQ-118")
    def test_auroc_and_ap_match_sklearn_with_ties(self):
        rng = np.random.default_rng(1)
        y = rng.integers(0, 2, 500)
        s = np.round(rng.random(500) * 0.5 + 0.3 * y, 2)  # muitos empates
        ones = np.ones(500)
        assert weighted_auroc(y, s, ones)[0] == pytest.approx(roc_auc_score(y, s), abs=1e-12)
        assert auroc(y, s) == pytest.approx(roc_auc_score(y, s), abs=1e-12)
        ap = weighted_average_precision(y, s, ones)[0]
        assert ap == pytest.approx(average_precision_score(y, s), abs=1e-12)

    def test_integer_weights_equal_row_replication(self):
        rng = np.random.default_rng(2)
        y = rng.integers(0, 2, 60)
        s = rng.random(60)
        w = rng.integers(0, 4, 60).astype(float)
        idx = np.repeat(np.arange(60), w.astype(int))
        assert weighted_auroc(y, s, w)[0] == pytest.approx(roc_auc_score(y[idx], s[idx]))
        assert weighted_average_precision(y, s, w)[0] == pytest.approx(
            average_precision_score(y[idx], s[idx])
        )
        rates = weighted_rates_at(y, s, 0.5, w)
        pred = s[idx] >= 0.5
        assert rates["sensitivity"][0] == pytest.approx(pred[y[idx] == 1].mean())
        assert rates["specificity"][0] == pytest.approx((~pred)[y[idx] == 0].mean())
        assert weighted_ece(y, s, w)[0] == pytest.approx(expected_calibration_error(y[idx], s[idx]))

    def test_ece_known_value(self):
        # bin (0.7, 0.8]: 4 casos, confiança média 0.75, acerto 0.5 -> |0.5-0.75| * 4/4
        y = np.array([1, 1, 0, 0])
        s = np.array([0.75, 0.75, 0.75, 0.75])
        assert expected_calibration_error(y, s) == pytest.approx(0.25)


class TestThresholds:
    def test_rule_out_is_highest_threshold_reaching_sensitivity(self):
        y = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        s = np.array([0.1, 0.2, 0.3, 0.15, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99])
        # 10 positivos; 9/10 = 0.9 atingido em 0.4 (exclui só 0.15)
        assert threshold_for_sensitivity(y, s, 0.9) == pytest.approx(0.4)
        op = operating_point_at_sensitivity(y, s, 0.9)
        assert op.threshold == pytest.approx(0.4)
        assert op.specificity == pytest.approx(1.0)

    def test_rule_in_is_lowest_threshold_reaching_specificity(self):
        y = np.array([0] * 10 + [1] * 5)
        s = np.concatenate([np.linspace(0.0, 0.9, 10), [0.5, 0.6, 0.95, 0.97, 0.99]])
        t = threshold_for_specificity(y, s, 0.9)
        # especificidade >= 0.9 exige 9 dos 10 negativos abaixo do limiar
        assert t == pytest.approx(0.9)
        neg = s[y == 0]
        assert (neg < t).mean() >= 0.9

    def test_rule_in_unreachable_raises(self):
        y = np.array([0, 0, 1, 1])
        s = np.array([0.9, 0.9, 0.9, 0.9])
        with pytest.raises(EvaluationError, match="nenhum valor de escore"):
            threshold_for_specificity(y, s, 0.9)


class TestClusterBootstrap:
    @pytest.mark.requirement("REQ-110")
    def test_resamples_whole_patients(self):
        pids = np.array(["a", "a", "a", "b", "c", "c"])
        boot = ClusterBootstrap(pids, seed=1)
        for block in boot.weights(50):
            for row in block:
                # imagens do mesmo paciente têm sempre a mesma multiplicidade
                assert row[0] == row[1] == row[2]
                assert row[4] == row[5]
                assert row[[0, 3, 4]].sum() == 3  # 3 pacientes sorteados

    @pytest.mark.requirement("REQ-110")
    def test_stratified_by_patient_positivity(self):
        pids, y, _ = clustered_data()
        boot = ClusterBootstrap(pids, seed=3, strata=y)
        pos_patients = len(set(pids[y == 1]))
        for block in boot.weights(20):
            for row in block:
                drawn_pos = sum(row[np.flatnonzero(pids == p)[0]] for p in set(pids[y == 1]))
                assert drawn_pos == pos_patients

    @pytest.mark.requirement("REQ-110")
    def test_deterministic_and_chunk_independent(self):
        pids, y, s = clustered_data()
        a = np.concatenate(list(ClusterBootstrap(pids, seed=9, strata=y).weights(130)))
        b = np.concatenate(list(ClusterBootstrap(pids, seed=9, strata=y).weights(130)))
        assert np.array_equal(a, b)
        c = np.concatenate(list(ClusterBootstrap(pids, seed=10, strata=y).weights(130)))
        assert not np.array_equal(a, c)

    @pytest.mark.requirement("REQ-110")
    def test_cluster_ci_wider_than_image_ci_for_correlated_images(self):
        """Imagens correlacionadas: o IC por imagem é estreito demais."""
        pids, y, s = clustered_data(n_patients=150, seed=4)
        cl = ClusterBootstrap(pids, seed=5, strata=y)
        img = ClusterBootstrap(np.arange(len(y)).astype(str), seed=5, strata=y)
        width = []
        for boot in (cl, img):
            samples = np.concatenate([weighted_auroc(y, s, w) for w in boot.weights(400)])
            lo, hi = percentile_ci(samples)
            width.append(hi - lo)
        assert width[0] > width[1] * 1.2


class TestDeLong:
    @pytest.mark.requirement("REQ-118")
    def test_single_model_variance_matches_structural_components(self):
        rng = np.random.default_rng(11)
        y = rng.integers(0, 2, 80)
        s = rng.random(80) + 0.4 * y
        auc, var = delong_variance(y, s)
        pos, neg = s[y == 1], s[y == 0]
        psi = (pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])
        v10, v01 = psi.mean(axis=1), psi.mean(axis=0)
        expected = v10.var(ddof=1) / pos.size + v01.var(ddof=1) / neg.size
        assert auc == pytest.approx(roc_auc_score(y, s))
        assert var == pytest.approx(expected, rel=1e-10)

    @pytest.mark.requirement("REQ-118")
    def test_paired_difference(self):
        rng = np.random.default_rng(12)
        y = rng.integers(0, 2, 300)
        a = rng.random(300) + 0.8 * y
        b = a + rng.normal(0, 0.6, 300)
        r = delong_test(y, a, b)
        assert r.auroc_a == pytest.approx(roc_auc_score(y, a))
        assert r.auroc_b == pytest.approx(roc_auc_score(y, b))
        assert r.difference > 0
        assert 0.0 <= r.p_value <= 1.0
        same = delong_test(y, a, a)
        assert same.difference == 0.0
        assert same.p_value == 1.0

    def test_paired_cluster_bootstrap_ci_contains_difference(self):
        pids, y, s = clustered_data(seed=6)
        rng = np.random.default_rng(0)
        s2 = np.clip(s + rng.normal(0, 0.15, s.size), 0, 1)
        r = paired_cluster_bootstrap(
            patient_ids=pids, y_true=y, score_a=s, score_b=s2, seed=1, n_bootstrap=300
        )
        assert r.ci_low <= r.difference <= r.ci_high
        assert r.difference == pytest.approx(roc_auc_score(y, s) - roc_auc_score(y, s2))


class TestAgreement:
    # Shrout & Fleiss (1979), Psychol Bull 86:420, Tabela 2: 6 alvos x 4 juízes.
    SF = np.array(
        [[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8], [7, 1, 2, 6], [10, 5, 6, 9], [6, 2, 4, 7]],
        dtype=float,
    )

    @pytest.mark.requirement("REQ-118")
    @pytest.mark.parametrize(
        ("form", "expected"),
        [("1,1", 0.17), ("2,1", 0.29), ("3,1", 0.71), ("1,k", 0.44), ("2,k", 0.62), ("3,k", 0.91)],
    )
    def test_shrout_fleiss_textbook_values(self, form, expected):
        assert icc(self.SF, form) == pytest.approx(expected, abs=0.005)

    def test_bland_altman_manual(self):
        m = np.array([50.0, 55, 60, 48, 62])
        r = np.array([52.0, 54, 63, 45, 60])
        d = m - r
        bias, lo, hi = bland_altman(m, r)
        assert bias == pytest.approx(d.mean())
        assert hi - bias == pytest.approx(1.96 * d.std(ddof=1))
        assert bias - lo == pytest.approx(1.96 * d.std(ddof=1))

    def test_evaluate_agreement_with_ci(self):
        rng = np.random.default_rng(3)
        ref = rng.normal(55, 10, 80)
        meas = ref + rng.normal(2, 3, 80)
        rep = evaluate_agreement(
            patient_ids=[f"p{i}" for i in range(80)],
            measured=meas,
            reference=ref,
            seed=1,
            n_bootstrap=200,
        )
        assert rep.icc.ci_low <= rep.icc.value <= rep.icc.ci_high
        assert rep.bias.value == pytest.approx((meas - ref).mean())
        assert rep.mae.value == pytest.approx(np.abs(meas - ref).mean())
        assert rep.icc_form == "ICC(2,1)"


class TestSegmentationAndDetection:
    def test_dice_known_values(self):
        a = np.zeros((4, 4), bool)
        b = np.zeros((4, 4), bool)
        a[:2, :2] = True  # 4 px
        b[:2, :] = True  # 8 px, sobreposição 4
        assert dice_coefficient(a, b) == pytest.approx(2 * 4 / 12)
        assert np.isnan(dice_coefficient(np.zeros(3), np.zeros(3)))

    def test_dice_both_empty_excluded_not_perfect(self):
        rep = evaluate_dice(
            patient_ids=["a", "b", "c", "d"],
            dice_per_case=np.array([0.8, 0.6, np.nan, 0.7]),
            seed=1,
            n_bootstrap=100,
        )
        assert rep.n_both_empty == 1
        assert rep.n_cases == 3
        assert rep.dice_mean == pytest.approx(0.7)

    def test_detection_fixed_rule(self):
        y = np.array([1] * 40 + [0] * 60)
        pred = np.array([1] * 36 + [0] * 4 + [0] * 54 + [1] * 6)
        rep = evaluate_detection(
            rule="volume > 0",
            patient_ids=[f"p{i}" for i in range(100)],
            y_true=y,
            y_pred=pred,
            seed=1,
            n_bootstrap=200,
        )
        assert rep.sensitivity.value == pytest.approx(0.9)
        assert rep.specificity.value == pytest.approx(0.9)
        assert rep.sensitivity.ci_low < 0.9 < rep.sensitivity.ci_high
