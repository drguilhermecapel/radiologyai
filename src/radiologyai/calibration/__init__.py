"""Calibração e abstenção — o controle de risco primário (perigos H-01, H-02, H-10)."""

from __future__ import annotations

from radiologyai.calibration.abstention import (
    CRITICAL_FINDINGS,
    AbstentionPolicy,
    BandThresholds,
    summarize,
)
from radiologyai.calibration.temperature import TemperatureCalibrator, fit_temperature

__all__ = [
    "CRITICAL_FINDINGS",
    "AbstentionPolicy",
    "BandThresholds",
    "TemperatureCalibrator",
    "fit_temperature",
    "summarize",
]
