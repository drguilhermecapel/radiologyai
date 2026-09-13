"""Calibração e abstenção — o controle de risco primário."""

from __future__ import annotations

import numpy as np
import pytest

from radiologyai.calibration import (
    CRITICAL_FINDINGS,
    AbstentionPolicy,
    BandThresholds,
    fit_temperature,
    summarize,
)
from radiologyai.errors import EvaluationError
from radiologyai.evaluation import expected_calibration_error
from radiologyai.inference.types import Band

SHARPENING = 2.5
"""Fator de afiamento usado para simular superconfianca.

Um modelo superconfiante e aquele cujo logit esta multiplicado por k > 1: a
ordenacao esta certa, mas os escores sao empurrados para os extremos. E
exatamente isso que temperature scaling desfaz, ajustando T ~ k.
"""


def overconfident_data(n: int = 2000, seed: int = 11, k: float = SHARPENING):
    """Gera rotulos a partir de probabilidades verdadeiras e escores afiados.

    Setup canonico de Guo et al. (2017): as probabilidades verdadeiras geram os
    rotulos; o "modelo" reporta sigmoid(k * logit), superconfiante.
    """
    rng = np.random.default_rng(seed)
    logits = rng.normal(-1.2, 1.8, n)
    p_true = 1.0 / (1.0 + np.exp(-logits))
    truth = (rng.random(n) < p_true).astype(int)
    scores = 1.0 / (1.0 + np.exp(-k * logits))
    return truth, scores


class TestTemperatureScaling:
    def test_reduces_calibration_error(self):
        truth, scores = overconfident_data()
        cal = fit_temperature(truth, scores, label="Pneumonia")
        before = expected_calibration_error(truth, scores)
        after = expected_calibration_error(truth, cal.apply(scores))
        assert after < before, f"ECE piorou: {before:.4f} -> {after:.4f}"

    def test_recovers_the_known_sharpening_factor(self):
        """Se o modelo afia o logit por k, a temperatura ajustada deve ≈ k.

        É a verificação mais forte disponível: o valor correto é conhecido.
        """
        truth, scores = overconfident_data(n=4000, k=SHARPENING)
        cal = fit_temperature(truth, scores, label="Pneumonia")
        assert cal.temperature == pytest.approx(SHARPENING, rel=0.20)

    def test_already_calibrated_gets_temperature_near_one(self):
        truth, scores = overconfident_data(n=4000, k=1.0)
        cal = fit_temperature(truth, scores, label="Pneumonia")
        assert cal.temperature == pytest.approx(1.0, abs=0.25)

    @pytest.mark.requirement("REQ-061")
    def test_preserves_ranking_and_therefore_auroc(self):
        """Calibração realinha escores; NÃO muda a ordenação, logo não muda o AUROC."""
        from radiologyai.evaluation import auroc

        truth, scores = overconfident_data()
        cal = fit_temperature(truth, scores, label="Pneumonia")
        assert auroc(truth, cal.apply(scores)) == pytest.approx(auroc(truth, scores))

    def test_reduces_nll(self):
        truth, scores = overconfident_data()
        cal = fit_temperature(truth, scores, label="Pneumonia")
        assert cal.improved
        assert cal.nll_after <= cal.nll_before

    def test_records_fit_provenance(self):
        truth, scores = overconfident_data(n=300)
        d = fit_temperature(truth, scores, label="Effusion").to_dict()
        assert d["label"] == "Effusion"
        assert d["n_fit_samples"] == 300
        assert d["temperature"] > 0

    def test_output_stays_in_unit_range(self):
        truth, scores = overconfident_data()
        out = fit_temperature(truth, scores, label="X").apply(scores)
        assert 0.0 <= out.min() <= out.max() <= 1.0

    def test_too_few_samples_refused(self):
        with pytest.raises(EvaluationError, match="pouco para calibrar"):
            fit_temperature(np.array([0, 1, 0]), np.array([0.1, 0.9, 0.2]), label="X")

    def test_single_class_refused(self):
        with pytest.raises(EvaluationError, match="as duas classes"):
            fit_temperature(np.ones(50, dtype=int), np.full(50, 0.9), label="X")

    def test_deterministic(self):
        truth, scores = overconfident_data()
        a = fit_temperature(truth, scores, label="X")
        b = fit_temperature(truth, scores, label="X")
        assert a.temperature == b.temperature


class TestBandThresholds:
    def test_three_bands(self):
        t = BandThresholds(lower=0.2, upper=0.6)
        assert t.classify(0.9) is Band.LIKELY
        assert t.classify(0.4) is Band.INDETERMINATE
        assert t.classify(0.1) is Band.UNLIKELY

    def test_boundaries_inclusive(self):
        t = BandThresholds(lower=0.2, upper=0.6)
        assert t.classify(0.6) is Band.LIKELY
        assert t.classify(0.2) is Band.UNLIKELY

    def test_invalid_order_rejected(self):
        with pytest.raises(ValueError, match="limiares inválidos"):
            BandThresholds(lower=0.8, upper=0.2)

    def test_out_of_range_rejected(self):
        with pytest.raises(ValueError, match="limiares inválidos"):
            BandThresholds(lower=-0.1, upper=0.5)


@pytest.mark.requirement("REQ-062")
class TestAbstentionPolicy:
    def test_built_from_measured_operating_points(self):
        p = AbstentionPolicy.from_operating_points({"Pneumonia": 0.5, "Effusion": 0.4})
        assert set(p.thresholds) == {"Pneumonia", "Effusion"}

    def test_empty_operating_points_refused(self):
        """Limiares inventados não são política de risco."""
        with pytest.raises(EvaluationError, match="pontos de operação medidos"):
            AbstentionPolicy.from_operating_points({})

    def test_critical_findings_get_wider_band(self):
        """Achado crítico: na dúvida o sistema se abstém em vez de negar."""
        p = AbstentionPolicy.from_operating_points({"Pneumothorax": 0.5, "Fibrosis": 0.5})
        assert "Pneumothorax" in CRITICAL_FINDINGS
        assert "Fibrosis" not in CRITICAL_FINDINGS
        assert (
            p.thresholds["Pneumothorax"].indeterminate_width
            > p.thresholds["Fibrosis"].indeterminate_width
        )

    def test_unknown_label_is_always_indeterminate(self):
        p = AbstentionPolicy.from_operating_points({"Pneumonia": 0.5})
        assert p.classify("Desconhecido", 0.99) is Band.INDETERMINATE

    def test_unevaluated_label_forced_indeterminate(self):
        """Sem medição de desempenho não há base para afirmar nada."""
        p = AbstentionPolicy.from_operating_points({"Pneumonia": 0.5})
        findings = p.apply(["Pneumonia"], [0.99], evaluated=[False])
        assert findings[0].band is Band.INDETERMINATE
        assert findings[0].evaluated is False

    def test_calibration_flag_propagates(self):
        p = AbstentionPolicy.from_operating_points({"Pneumonia": 0.5}, calibrated=False)
        assert p.apply(["Pneumonia"], [0.9])[0].calibrated is False

    def test_length_mismatch_refused(self):
        p = AbstentionPolicy.from_operating_points({"Pneumonia": 0.5})
        with pytest.raises(EvaluationError, match="rótulos para"):
            p.apply(["A", "B"], [0.5])

    def test_abstention_rate(self):
        p = AbstentionPolicy.from_operating_points({"A": 0.5, "B": 0.5}, margin=0.1)
        findings = p.apply(["A", "B"], [0.95, 0.52])
        assert p.abstention_rate(findings) == 0.5

    def test_to_dict_marks_critical(self):
        d = AbstentionPolicy.from_operating_points({"Pneumothorax": 0.5}).to_dict()
        assert d["thresholds"]["Pneumothorax"]["critical"] is True


@pytest.mark.requirement("REQ-063")
class TestNeverAssertsNormality:
    """REG-01 §4: ausência de achado provável não é exame normal."""

    def test_summary_never_says_normal(self):
        p = AbstentionPolicy.from_operating_points({"Pneumonia": 0.5, "Effusion": 0.5})
        text = summarize(p.apply(["Pneumonia", "Effusion"], [0.05, 0.02]))
        assert "normal" not in text.lower().replace("não constitui afirmação de exame normal", "")

    def test_summary_states_it_is_not_normality(self):
        p = AbstentionPolicy.from_operating_points({"Pneumonia": 0.5})
        text = summarize(p.apply(["Pneumonia"], [0.01]))
        assert "NÃO constitui afirmação de exame normal" in text

    def test_summary_always_requires_medical_review(self):
        p = AbstentionPolicy.from_operating_points({"Pneumonia": 0.5})
        for score in (0.01, 0.5, 0.99):
            text = summarize(p.apply(["Pneumonia"], [score]))
            assert "Requer revisão e validação médica" in text

    def test_summary_declares_indeterminate_findings(self):
        p = AbstentionPolicy.from_operating_points({"Pneumonia": 0.5}, margin=0.2)
        text = summarize(p.apply(["Pneumonia"], [0.5]))
        assert "Não avaliável" in text
