"""Desenho pré-registrado: split por paciente, pontos fixados na calibração, subgrupos."""

from __future__ import annotations

import numpy as np
import pytest

from radiologyai.catalog import load_protocol
from radiologyai.data.manifest import Manifest, ManifestRow
from radiologyai.evaluation.dataset import (
    MISSING_LABEL,
    age_band,
    label_matrix_with_missing,
    manifest_sha256_v2,
)
from radiologyai.evaluation.metrics import weighted_rates_at
from radiologyai.evaluation.protocol_eval import evaluate_binary, patient_split
from radiologyai.evaluation.runner import EvaluationConfig, evaluate_labelwise


def synthetic(n_patients: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    pids, y, s, sex, age, view = [], [], [], [], [], []
    for p in range(n_patients):
        k = int(rng.integers(1, 4))
        sick = rng.random() < 0.35
        a = float(rng.integers(5, 90))
        sx = "M" if rng.random() < 0.5 else "F"
        for _ in range(k):
            lab = int(sick)
            pids.append(f"p{p:04d}")
            y.append(lab)
            s.append(float(1 / (1 + np.exp(-(1.5 * lab - 0.7 + rng.normal(0, 1))))))
            sex.append(sx)
            age.append(a)
            view.append("PA" if rng.random() < 0.5 else "AP")
    return pids, np.array(y), np.array(s), {"sex": sex, "age": age, "view": view}


@pytest.fixture(scope="module")
def protocol():
    return load_protocol()


class TestPatientSplit:
    @pytest.mark.requirement("REQ-111")
    def test_patient_disjoint_deterministic_order_independent(self):
        pids, y, _, _ = synthetic()
        m1 = patient_split(pids, y, seed=7, calibration_fraction=0.5)
        cal = {p for p, k in zip(pids, m1, strict=True) if k}
        test = {p for p, k in zip(pids, m1, strict=True) if not k}
        assert not cal & test
        perm = np.random.default_rng(1).permutation(len(pids))
        m2 = patient_split([pids[i] for i in perm], y[perm], seed=7, calibration_fraction=0.5)
        assert {pids[i] for i, k in zip(perm, m2, strict=True) if k} == cal
        m3 = patient_split(pids, y, seed=8, calibration_fraction=0.5)
        assert not np.array_equal(m1, m3)

    @pytest.mark.requirement("REQ-111")
    def test_stratified_halves(self):
        pids, y, _, _ = synthetic()
        m = patient_split(pids, y, seed=7, calibration_fraction=0.5)
        pos = {p for p, v in zip(pids, y, strict=True) if v == 1}
        cal_pos = {p for p, k in zip(pids, m, strict=True) if k} & pos
        assert abs(len(cal_pos) - len(pos) / 2) <= 1


@pytest.fixture(scope="module")
def report(protocol):
    pids, y, s, sub = synthetic()
    return evaluate_binary(
        label="X",
        patient_ids=pids,
        y_true=y,
        y_score=s,
        protocol=protocol,
        subgroup_columns=sub,
        n_bootstrap=200,
    ), (pids, y, s)


class TestEvaluateBinary:
    @pytest.mark.requirement("REQ-111")
    def test_points_fitted_on_calibration_reported_on_test(self, report, protocol):
        rep, (pids, y, s) = report
        cal = patient_split(
            pids, y, seed=protocol.random_seed, calibration_fraction=protocol.calibration_fraction
        )
        ro = rep.rule_out
        assert ro is not None
        # sensibilidade de calibração atinge o alvo; a de teste é medida em outros pacientes
        assert ro.calibration_sensitivity >= protocol.rule_out_sensitivity_target
        test_rates = weighted_rates_at(y[~cal], s[~cal], ro.threshold, np.ones((~cal).sum()))
        assert ro.sensitivity.value == pytest.approx(test_rates["sensitivity"][0])
        assert ro.specificity.ci_low <= ro.specificity.value <= ro.specificity.ci_high
        ri = rep.rule_in
        assert ri is not None
        assert ri.calibration_specificity >= protocol.rule_in_specificity_target
        assert ri.threshold >= ro.threshold
        assert rep.counts.n_images == int((~cal).sum())

    @pytest.mark.requirement("REQ-110")
    def test_cis_and_calibration(self, report):
        rep, _ = report
        assert rep.auroc.ci_low <= rep.auroc.value <= rep.auroc.ci_high
        assert rep.auprc.ci_low <= rep.auprc.value <= rep.auprc.ci_high
        assert rep.calibration is not None
        assert rep.ece_before is not None
        assert rep.ece_after is not None
        assert "por paciente" in rep.ci_method

    @pytest.mark.requirement("REQ-112")
    def test_subgroups_with_minor_band(self, report):
        rep, _ = report
        attrs = {(g.attribute, g.value) for g in rep.subgroups}
        assert ("age_band", "<18") in attrs
        assert {("sex", "F"), ("sex", "M"), ("view", "AP"), ("view", "PA")} <= attrs
        for g in rep.subgroups:
            if not g.eligible:
                assert g.note

    @pytest.mark.requirement("REQ-112")
    def test_missing_metadata_declared(self, protocol):
        pids, y, s, _ = synthetic(seed=2)
        rep = evaluate_binary(
            label="X", patient_ids=pids, y_true=y, y_score=s, protocol=protocol, n_bootstrap=50
        )
        assert len(rep.subgroups_unavailable) == 3
        assert all("metadado ausente" in u for u in rep.subgroups_unavailable)

    def test_reproducible(self, protocol):
        pids, y, s, sub = synthetic(n_patients=150, seed=3)
        kw = {
            "label": "X",
            "patient_ids": pids,
            "y_true": y,
            "y_score": s,
            "protocol": protocol,
            "subgroup_columns": sub,
            "n_bootstrap": 100,
        }
        assert evaluate_binary(**kw) == evaluate_binary(**kw)


class TestDatasetFixes:
    @pytest.mark.requirement("REQ-117")
    def test_age_band_has_separate_minor_band(self):
        assert age_band(10) == "<18"
        assert age_band(17.9) == "<18"
        assert age_band(18) == "18-39"
        assert age_band(None) == "desconhecida"

    @pytest.mark.requirement("REQ-117")
    def test_manifest_v2_hash_covers_demographics(self):
        def m(sex):
            return Manifest(
                name="x",
                split="t",
                label_names=("A",),
                rows=[ManifestRow("i1", "p1", {"A": 1}, patient_sex=sex, patient_age_years=40)],
            )

        a, b = m("M"), m("F")
        assert a.sha256() == b.sha256(), "v1 (mantido para runs commitados) ignora demografia"
        assert manifest_sha256_v2(a) != manifest_sha256_v2(b)

    @pytest.mark.requirement("REQ-117")
    def test_missing_label_is_not_negative(self):
        man = Manifest(
            name="CheXpert",
            split="valid",
            label_names=("A", "B"),
            rows=[ManifestRow("i1", "p1", {"A": 1}), ManifestRow("i2", "p2", {"A": 0, "B": 1})],
        )
        mat = label_matrix_with_missing(man)
        assert mat[0, 1] == MISSING_LABEL
        assert man.label_matrix()[0][1] == 0, "comportamento antigo: ausente vira negativo"
        # v2 distingue ausente de negativo
        man2 = Manifest(
            name="CheXpert",
            split="valid",
            label_names=("A", "B"),
            rows=[
                ManifestRow("i1", "p1", {"A": 1, "B": 0}),
                ManifestRow("i2", "p2", {"A": 0, "B": 1}),
            ],
        )
        assert manifest_sha256_v2(man) != manifest_sha256_v2(man2)

    @pytest.mark.requirement("REQ-117")
    def test_runner_excludes_missing_labels(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, (60, 1))
        y[:20, 0] = MISSING_LABEL
        s = rng.random((60, 1))
        evaluated, _ = evaluate_labelwise(
            y, s, ("A",), EvaluationConfig(n_bootstrap=50), patient_ids=[str(i) for i in range(60)]
        )
        e = evaluated["A"]
        assert e["n_pos"] + e["n_neg"] == 40
        assert e["n_missing_label"] == 20
        assert "por paciente" in e["ci_method"]
