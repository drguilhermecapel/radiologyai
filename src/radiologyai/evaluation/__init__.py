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
from radiologyai.evaluation.predict_dataset import load_png, predict_manifest
from radiologyai.evaluation.runner import EvaluationConfig, check_leakage, run_evaluation
from radiologyai.evaluation.splits import assert_patient_disjoint

__all__ = [
    "AUROCResult",
    "EvaluationConfig",
    "OperatingPoint",
    "assert_patient_disjoint",
    "auroc",
    "auroc_with_ci",
    "check_leakage",
    "expected_calibration_error",
    "load_png",
    "operating_point_at_sensitivity",
    "predict_manifest",
    "run_evaluation",
]
