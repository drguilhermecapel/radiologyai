"""Calibração e abstenção — o controle de risco primário (perigos H-01, H-02, H-10)."""

from __future__ import annotations

from radiologyai.calibration.abstention import (
    CRITICAL_FINDINGS,
    AbstentionPolicy,
    BandThresholds,
    summarize,
)
from radiologyai.calibration.platt import (
    PlattScaler,
    fit_platt_scaler,
    load_platt_calibrators,
    save_platt_calibrators,
)
from radiologyai.calibration.reliability import (
    ReliabilityBin,
    ReliabilityDiagram,
    compute_reliability_diagram,
)
from radiologyai.calibration.temperature import TemperatureCalibrator, fit_temperature

__all__ = [
    "CRITICAL_FINDINGS",
    "AbstentionPolicy",
    "BandThresholds",
    "PlattScaler",
    "ReliabilityBin",
    "ReliabilityDiagram",
    "TemperatureCalibrator",
    "compute_reliability_diagram",
    "fit_platt_scaler",
    "fit_temperature",
    "load_platt_calibrators",
    "save_platt_calibrators",
    "summarize",
]
