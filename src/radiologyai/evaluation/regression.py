from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from radiologyai.errors import RegressionGateError

_DISALLOWED_LEAKAGE_STATUSES = frozenset({"leakage", "overlap", "failed"})


@dataclass(frozen=True)
class RegressionResult:
    passed: bool
    reasons: tuple[str, ...]
    current_macro_auroc: float | None
    reference_macro_auroc: float | None
    delta_macro_auroc: float | None


def load_metrics(path: str | Path) -> dict[str, Any]:
    raw_text = Path(path).read_text(encoding="utf-8")
    data = json.loads(raw_text)
    if not isinstance(data, dict):
        raise RegressionGateError(f"Expected JSON object in {path}, got {type(data).__name__}")
    return data


def _extract_finite_float(
    metrics: dict[str, Any],
    key: str,
    label: str,
    reasons: list[str],
) -> float | None:
    value = metrics.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        reasons.append(f"{label}.{key} is missing or non-numeric")
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        reasons.append(f"{label}.{key} must be finite")
        return None
    return numeric


def _extract_positive_int(
    metrics: dict[str, Any],
    key: str,
    label: str,
    reasons: list[str],
) -> int | None:
    value = metrics.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        reasons.append(f"{label}.{key} must be a positive integer")
        return None
    return value


def check_baseline_regression(
    current_metrics: dict[str, Any],
    reference_metrics: dict[str, Any],
    *,
    max_auroc_drop: float = 0.02,
    require_external_dataset: bool = True,
    allow_label_drop: bool = False,
) -> RegressionResult:
    if (
        isinstance(max_auroc_drop, bool)
        or not isinstance(max_auroc_drop, (int, float))
        or not math.isfinite(float(max_auroc_drop))
        or max_auroc_drop < 0.0
    ):
        raise RegressionGateError("max_auroc_drop must be a finite number >= 0.0")

    reasons: list[str] = []

    current_auroc = _extract_finite_float(
        current_metrics,
        "macro_auroc",
        "current_metrics",
        reasons,
    )
    reference_auroc = _extract_finite_float(
        reference_metrics,
        "macro_auroc",
        "reference_metrics",
        reasons,
    )

    delta_auroc: float | None = None
    if current_auroc is not None and reference_auroc is not None:
        delta_auroc = current_auroc - reference_auroc
        if delta_auroc < -float(max_auroc_drop):
            reasons.append(
                f"macro_auroc dropped by {-delta_auroc:.4f} "
                f"(exceeds max_auroc_drop={max_auroc_drop:.4f})"
            )

    current_n_labels = _extract_positive_int(
        current_metrics,
        "n_labels_evaluated",
        "current_metrics",
        reasons,
    )
    if "n_labels_evaluated" in reference_metrics:
        reference_n_labels = _extract_positive_int(
            reference_metrics,
            "n_labels_evaluated",
            "reference_metrics",
            reasons,
        )
        if (
            not allow_label_drop
            and current_n_labels is not None
            and reference_n_labels is not None
            and current_n_labels < reference_n_labels
        ):
            reasons.append(
                f"n_labels_evaluated decreased from {reference_n_labels} to {current_n_labels}"
            )

    if require_external_dataset:
        dataset = current_metrics.get("dataset")
        if not isinstance(dataset, dict):
            reasons.append(
                "current_metrics.dataset must be an object when require_external_dataset=True"
            )
        else:
            if dataset.get("external_to_training_data") is not True:
                reasons.append("current_metrics.dataset.external_to_training_data must be True")
            leakage_status = dataset.get("leakage_status")
            if (
                isinstance(leakage_status, str)
                and leakage_status.strip().lower() in _DISALLOWED_LEAKAGE_STATUSES
            ):
                reasons.append(
                    f"current_metrics.dataset.leakage_status is disallowed: {leakage_status}"
                )

    return RegressionResult(
        passed=len(reasons) == 0,
        reasons=tuple(reasons),
        current_macro_auroc=current_auroc,
        reference_macro_auroc=reference_auroc,
        delta_macro_auroc=delta_auroc,
    )


def compare_metric_files(
    current_path: str | Path,
    reference_path: str | Path,
    *,
    max_auroc_drop: float = 0.02,
    require_external_dataset: bool = True,
    allow_label_drop: bool = False,
) -> RegressionResult:
    current_metrics = load_metrics(current_path)
    reference_metrics = load_metrics(reference_path)
    return check_baseline_regression(
        current_metrics,
        reference_metrics,
        max_auroc_drop=max_auroc_drop,
        require_external_dataset=require_external_dataset,
        allow_label_drop=allow_label_drop,
    )
