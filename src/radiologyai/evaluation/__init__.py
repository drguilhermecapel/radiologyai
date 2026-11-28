"""Avaliação de desempenho — medida, reprodutível, com incerteza declarada."""

from __future__ import annotations

from radiologyai.evaluation.dataset import age_band, manifest_sha256_v2
from radiologyai.evaluation.metrics import (
    AUROCResult,
    ClusterBootstrap,
    OperatingPoint,
    auroc,
    auroc_with_ci,
    expected_calibration_error,
    operating_point_at_sensitivity,
    threshold_for_sensitivity,
    threshold_for_specificity,
)
from radiologyai.evaluation.predict_dataset import load_png, predict_manifest
from radiologyai.evaluation.runner import EvaluationConfig, check_leakage, run_evaluation
from radiologyai.evaluation.splits import assert_patient_disjoint

__all__ = [
    "AUROCResult",
    "ClusterBootstrap",
    "EvaluationConfig",
    "OperatingPoint",
    "assert_patient_disjoint",
    "auroc",
    "auroc_with_ci",
    "check_leakage",
    "expected_calibration_error",
    "load_png",
    "age_band",
    "manifest_sha256_v2",
    "operating_point_at_sensitivity",
    "predict_manifest",
    "run_evaluation",
    "threshold_for_sensitivity",
    "threshold_for_specificity",
]
