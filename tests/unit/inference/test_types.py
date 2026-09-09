"""Tipos de resultado — regras estruturais do uso pretendido."""

from __future__ import annotations

import pytest

from radiologyai.inference.types import Band, Finding, StudyResult

SHA = "c" * 64


def test_three_bands_exist():
    assert {b.value for b in Band} == {
        "achado_provavel",
        "nao_avaliavel",
        "achado_improvavel",
    }


def test_finding_score_bounded():
    with pytest.raises(ValueError, match="less than or equal"):
        Finding(label="X", score=1.5, band=Band.LIKELY, calibrated=False)


def test_finding_is_immutable():
    f = Finding(label="X", score=0.5, band=Band.INDETERMINATE, calibrated=False)
    with pytest.raises(ValueError, match="frozen"):
        f.score = 0.9


def test_calibration_must_be_declared():
    f = Finding(label="X", score=0.85, band=Band.LIKELY, calibrated=False)
    assert f.calibrated is False, "escore não calibrado não é probabilidade de doença"


def test_system_never_asserts_normality():
    """REG-01 §4: ausência de achado provável não é exame normal."""
    result = StudyResult(
        input_sha256=SHA,
        card_id="x",
        weights_sha256=SHA,
        code_version="2.0.0.dev0",
        modality="XR",
        findings=(Finding(label="Pneumonia", score=0.02, band=Band.UNLIKELY, calibrated=True),),
    )
    assert result.normality_asserted is False


def test_audit_fields_required():
    """Todo resultado carrega o que a auditoria precisa para reconstruir a decisão."""
    with pytest.raises(ValueError, match="String should match pattern"):
        StudyResult(
            input_sha256="curto",
            card_id="x",
            weights_sha256=SHA,
            code_version="1",
            modality="XR",
            findings=(),
        )
