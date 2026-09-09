"""Avaliação de desempenho — medida, reprodutível, com incerteza declarada."""

from __future__ import annotations

from radiologyai.evaluation.metrics import (
    AUROCResult,
    OperatingPoint,
    auroc,
    auroc_with_ci,
    expected_calibration_error,
    operating_point_at_sensitivity,
)
from radiologyai.evaluation.splits import assert_patient_disjoint

__all__ = [
    "AUROCResult",
    "OperatingPoint",
    "assert_patient_disjoint",
    "auroc",
    "auroc_with_ci",
    "expected_calibration_error",
    "operating_point_at_sensitivity",
]
