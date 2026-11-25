"""Testes unitários para o diagrama de confiabilidade e métricas de calibração."""

from __future__ import annotations

import numpy as np
import pytest

from radiologyai.calibration import (
    ReliabilityBin,
    ReliabilityDiagram,
    compute_reliability_diagram,
)


def test_perfect_calibration() -> None:
    probs = np.array([0.0, 0.0, 1.0, 1.0])
    labels = np.array([0, 0, 1, 1])

    diagram = compute_reliability_diagram(probs, labels, n_bins=5)

    assert isinstance(diagram, ReliabilityDiagram)
    assert diagram.n_samples == 4
    assert diagram.n_bins == 5
    assert len(diagram.bins) == 5
    assert diagram.ece == pytest.approx(0.0)
    assert diagram.mce == pytest.approx(0.0)
    assert diagram.brier_score == pytest.approx(0.0)
    assert sum(bin_item.count for bin_item in diagram.bins) == 4


def test_miscalibration_metrics_and_bin_counts() -> None:
    # Bin [0.0, 0.5): probs = [0.2, 0.4] -> mean_conf = 0.3, acc = 0.5, gap = 0.2, count = 2
    # Bin [0.5, 1.0]: probs = [0.8, 0.9] -> mean_conf = 0.85, acc = 0.0, gap = 0.85, count = 2
    probs = np.array([0.2, 0.4, 0.8, 0.9])
    labels = np.array([0, 1, 0, 0])

    diagram = compute_reliability_diagram(probs, labels, n_bins=2)

    expected_ece = (2 / 4) * 0.2 + (2 / 4) * 0.85
    expected_mce = 0.85
    expected_brier = float(np.mean((probs - labels) ** 2))

    assert diagram.n_samples == 4
    assert diagram.n_bins == 2
    assert sum(bin_item.count for bin_item in diagram.bins) == diagram.n_samples
    assert diagram.bins[0].count == 2
    assert diagram.bins[1].count == 2
    assert diagram.ece == pytest.approx(expected_ece)
    assert diagram.mce == pytest.approx(expected_mce)
    assert diagram.brier_score == pytest.approx(expected_brier)


def test_to_dict_serialization() -> None:
    diagram = compute_reliability_diagram([0.1, 0.9], [0, 1], n_bins=2)
    payload = diagram.to_dict()

    assert payload["n_samples"] == 2
    assert payload["n_bins"] == 2
    assert payload["ece"] == pytest.approx(0.1)
    assert payload["mce"] == pytest.approx(0.1)
    assert payload["brier_score"] == pytest.approx(0.01)

    bins_payload = payload["bins"]
    assert isinstance(bins_payload, list)
    assert len(bins_payload) == 2
    assert isinstance(diagram.bins[0], ReliabilityBin)
    assert bins_payload[0] == diagram.bins[0].to_dict()
    assert set(bins_payload[0].keys()) == {
        "lower",
        "upper",
        "count",
        "mean_confidence",
        "empirical_accuracy",
        "gap",
    }


def test_validation_empty_inputs() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        compute_reliability_diagram([], [])


def test_validation_incompatible_shapes() -> None:
    with pytest.raises(ValueError, match="does not match"):
        compute_reliability_diagram([0.2, 0.8], [0])


@pytest.mark.parametrize("bad_probs", [[-0.1, 0.5], [0.5, 1.1]])
def test_validation_probabilities_out_of_range(bad_probs: list[float]) -> None:
    with pytest.raises(ValueError, match=r"\[0\.0, 1\.0\]"):
        compute_reliability_diagram(bad_probs, [0, 1])


def test_validation_non_binary_labels() -> None:
    with pytest.raises(ValueError, match="binary"):
        compute_reliability_diagram([0.2, 0.8], [0, 2])


@pytest.mark.parametrize(
    ("probs", "labels"),
    [
        ([np.nan, 0.5], [0, 1]),
        ([np.inf, 0.5], [0, 1]),
        ([0.2, 0.8], [0, np.nan]),
        ([0.2, 0.8], [0, np.inf]),
    ],
)
def test_validation_non_finite_values(probs: list[float], labels: list[float]) -> None:
    with pytest.raises(ValueError, match="finite"):
        compute_reliability_diagram(probs, labels)


@pytest.mark.parametrize("bad_bins", [0, -3])
def test_validation_invalid_n_bins(bad_bins: int) -> None:
    with pytest.raises(ValueError, match="n_bins"):
        compute_reliability_diagram([0.2, 0.8], [0, 1], n_bins=bad_bins)
