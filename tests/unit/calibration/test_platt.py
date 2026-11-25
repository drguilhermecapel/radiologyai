"""Testes unitários para Platt scaling."""

from __future__ import annotations

import numpy as np
import pytest

from radiologyai.calibration import (
    PlattScaler,
    compute_reliability_diagram,
    fit_platt_scaler,
)


def test_platt_reduces_ece_and_nll_under_prevalence_shift() -> None:
    rng = np.random.default_rng(42)
    logits_true = rng.normal(loc=-2.0, scale=1.4, size=1000)
    probs_true = 1.0 / (1.0 + np.exp(-logits_true))
    labels = rng.binomial(1, probs_true).astype(np.float64)

    shifted_probs = 1.0 / (1.0 + np.exp(-(logits_true + 1.8)))
    scaler = fit_platt_scaler(shifted_probs, labels)
    calibrated = scaler.transform(shifted_probs)

    assert calibrated.dtype == np.float64
    assert np.all((calibrated > 0.0) & (calibrated < 1.0))
    assert scaler.n_samples == 1000
    assert scaler.post_nll < scaler.pre_nll * 0.8

    pre_ece = compute_reliability_diagram(shifted_probs, labels).ece
    post_ece = compute_reliability_diagram(calibrated, labels).ece
    assert post_ece < pre_ece * 0.25


def test_platt_deterministic_convergence() -> None:
    probs = np.array([0.1, 0.25, 0.4, 0.7, 0.85, 0.95], dtype=np.float64)
    labels = np.array([0, 0, 1, 0, 1, 1], dtype=np.float64)

    first = fit_platt_scaler(probs, labels)
    second = fit_platt_scaler(probs, labels)
    assert first == second


def test_platt_preserves_ordering_when_slope_positive() -> None:
    scaler = PlattScaler(
        slope=1.5,
        intercept=-0.8,
        n_samples=5,
        pre_nll=0.6,
        post_nll=0.4,
    )
    probs = np.array([0.05, 0.2, 0.45, 0.7, 0.95], dtype=np.float64)
    calibrated = scaler.transform(probs)

    assert scaler.slope > 0.0
    assert np.all(np.diff(calibrated) > 0.0)


def test_platt_to_dict_serialization() -> None:
    scaler = PlattScaler(
        slope=0.95,
        intercept=-1.2,
        n_samples=250,
        pre_nll=0.72,
        post_nll=0.31,
    )
    payload = scaler.to_dict()
    assert payload == {
        "slope": 0.95,
        "intercept": -1.2,
        "n_samples": 250,
        "pre_nll": 0.72,
        "post_nll": 0.31,
    }


@pytest.mark.parametrize(
    ("probs", "labels"),
    [
        ([], []),
        ([0.2, 0.8], [0, 1, 0]),
        ([[0.2, 0.8]], [[0, 1]]),
        ([-0.1, 0.5], [0, 1]),
        ([0.5, 1.1], [0, 1]),
        ([np.nan, 0.5], [0, 1]),
        ([np.inf, 0.5], [0, 1]),
        ([0.2, 0.8], [0.0, np.nan]),
        ([0.2, 0.8], [0, 2]),
        ([0.2, 0.8], [0.0, 0.5]),
        ([0.2, 0.8], [0, 0]),
        ([0.2, 0.8], [1, 1]),
    ],
)
def test_fit_platt_scaler_validation_errors(
    probs: object,
    labels: object,
) -> None:
    with pytest.raises(ValueError), match=r"(probabilities|labels).*must"):
        fit_platt_scaler(probs, labels)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "probs",
    [
        [],
        [-0.01, 0.5],
        [0.5, 1.01],
        [np.nan, 0.5],
        [np.inf, 0.5],
    ],
)
def test_transform_validation_errors(probs: object) -> None:
    scaler = PlattScaler(
        slope=1.0,
        intercept=0.0,
        n_samples=2,
        pre_nll=0.5,
        post_nll=0.5,
    )
    with pytest.raises(ValueError, match="probabilities must"):
        scaler.transform(probs)  # type: ignore[arg-type]
