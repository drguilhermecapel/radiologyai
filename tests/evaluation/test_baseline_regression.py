from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from radiologyai.cli.main import app
from radiologyai.evaluation.regression import (
    check_baseline_regression,
    compare_metric_files,
    load_metrics,
)


def _valid_metrics(
    *,
    macro_auroc: float = 0.85,
    n_labels_evaluated: int = 14,
    external_to_training_data: bool = True,
    leakage_status: str = "clean",
) -> dict[str, Any]:
    return {
        "macro_auroc": macro_auroc,
        "n_labels_evaluated": n_labels_evaluated,
        "dataset": {
            "external_to_training_data": external_to_training_data,
            "leakage_status": leakage_status,
        },
    }


def test_passes_within_tolerance() -> None:
    reference = _valid_metrics(macro_auroc=0.85)
    current = _valid_metrics(macro_auroc=0.835)

    result = check_baseline_regression(current, reference, max_auroc_drop=0.02)

    assert result.passed is True
    assert result.reasons == ()
    assert result.current_macro_auroc == pytest.approx(0.835)
    assert result.reference_macro_auroc == pytest.approx(0.85)
    assert result.delta_macro_auroc == pytest.approx(-0.015)


def test_rejects_auroc_drop_above_threshold() -> None:
    reference = _valid_metrics(macro_auroc=0.85)
    current = _valid_metrics(macro_auroc=0.82)

    result = check_baseline_regression(current, reference, max_auroc_drop=0.02)

    assert result.passed is False
    assert result.delta_macro_auroc == pytest.approx(-0.03)
    assert any("macro_auroc dropped" in reason for reason in result.reasons)


@pytest.mark.parametrize(
    "bad_value",
    [None, "0.85", True, math.nan, math.inf, -math.inf],
)
def test_rejects_missing_nan_or_inf_macro_auroc(bad_value: Any) -> None:
    reference = _valid_metrics(macro_auroc=0.85)
    current = _valid_metrics()
    if bad_value is None:
        current.pop("macro_auroc")
    else:
        current["macro_auroc"] = bad_value

    result = check_baseline_regression(current, reference)

    assert result.passed is False
    assert result.delta_macro_auroc is None
    assert len(result.reasons) >= 1


def test_label_drop_behavior() -> None:
    reference = _valid_metrics(n_labels_evaluated=14)
    current = _valid_metrics(n_labels_evaluated=12)

    disallowed = check_baseline_regression(
        current,
        reference,
        allow_label_drop=False,
    )
    allowed = check_baseline_regression(
        current,
        reference,
        allow_label_drop=True,
    )

    assert disallowed.passed is False
    assert any("n_labels_evaluated decreased" in r for r in disallowed.reasons)
    assert allowed.passed is True

    zero_labels = _valid_metrics(n_labels_evaluated=0)
    zero_result = check_baseline_regression(
        zero_labels,
        reference,
        allow_label_drop=True,
    )
    assert zero_result.passed is False


@pytest.mark.parametrize("leakage_status", ["leakage", "overlap", "failed"])
def test_rejects_leakage_status(leakage_status: str) -> None:
    reference = _valid_metrics()
    current = _valid_metrics(leakage_status=leakage_status)

    result = check_baseline_regression(current, reference)

    assert result.passed is False
    assert any("leakage_status" in reason for reason in result.reasons)


def test_rejects_internal_dataset_when_required() -> None:
    reference = _valid_metrics()
    current = _valid_metrics(external_to_training_data=False)

    blocked = check_baseline_regression(
        current,
        reference,
        require_external_dataset=True,
    )
    unblocked = check_baseline_regression(
        current,
        reference,
        require_external_dataset=False,
    )

    assert blocked.passed is False
    assert any("external_to_training_data" in r for r in blocked.reasons)
    assert unblocked.passed is True


def test_compare_metric_files_with_tmp_path(tmp_path: Path) -> None:
    reference_file = tmp_path / "reference.json"
    current_file = tmp_path / "current.json"
    reference_file.write_text(
        json.dumps(_valid_metrics(macro_auroc=0.86)),
        encoding="utf-8",
    )
    current_file.write_text(
        json.dumps(_valid_metrics(macro_auroc=0.85)),
        encoding="utf-8",
    )

    result = compare_metric_files(current_file, reference_file)

    assert result.passed is True
    assert result.delta_macro_auroc == pytest.approx(-0.01)


def test_load_metrics_rejects_non_dict_json(tmp_path: Path) -> None:
    bad_file = tmp_path / "list.json"
    bad_file.write_text(json.dumps([1, 2, 3]), encoding="utf-8")

    with pytest.raises(ValueError, match="Expected JSON object"):
        load_metrics(bad_file)


def test_rejects_negative_max_auroc_drop() -> None:
    reference = _valid_metrics()
    current = _valid_metrics()

    with pytest.raises(ValueError, match="max_auroc_drop"):
        check_baseline_regression(current, reference, max_auroc_drop=-0.01)

def test_cli_check_regression_passes(tmp_path: Path) -> None:
    reference_file = tmp_path / "reference.json"
    current_file = tmp_path / "current.json"
    reference_file.write_text(
        json.dumps(_valid_metrics(macro_auroc=0.86)),
        encoding="utf-8",
    )
    current_file.write_text(
        json.dumps(_valid_metrics(macro_auroc=0.85)),
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "check-regression",
            "--current",
            str(current_file),
            "--reference",
            str(reference_file),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["passed"] is True


def test_cli_check_regression_fails(tmp_path: Path) -> None:
    reference_file = tmp_path / "reference.json"
    current_file = tmp_path / "current.json"
    reference_file.write_text(
        json.dumps(_valid_metrics(macro_auroc=0.86)),
        encoding="utf-8",
    )
    current_file.write_text(
        json.dumps(_valid_metrics(macro_auroc=0.80)),
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "check-regression",
            "--current",
            str(current_file),
            "--reference",
            str(reference_file),
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["passed"] is False
    assert payload["reasons"]
