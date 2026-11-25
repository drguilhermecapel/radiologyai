"""Diagrama de confiabilidade (reliability diagram) e métricas de calibração."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class ReliabilityBin:
    """Representa um intervalo uniforme de probabilidade no diagrama de confiabilidade."""

    lower: float
    upper: float
    count: int
    mean_confidence: float
    empirical_accuracy: float
    gap: float

    def to_dict(self) -> dict[str, float | int]:
        """Serializa o bin para um dicionário JSON-compatível."""
        return {
            "lower": self.lower,
            "upper": self.upper,
            "count": self.count,
            "mean_confidence": self.mean_confidence,
            "empirical_accuracy": self.empirical_accuracy,
            "gap": self.gap,
        }


@dataclass(frozen=True)
class ReliabilityDiagram:
    """Resultado completo do diagrama de confiabilidade e métricas globais."""

    bins: tuple[ReliabilityBin, ...]
    ece: float
    mce: float
    brier_score: float
    n_samples: int
    n_bins: int

    def to_dict(self) -> dict[str, object]:
        """Serializa o diagrama para um dicionário JSON-compatível."""
        return {
            "bins": [bin_item.to_dict() for bin_item in self.bins],
            "ece": self.ece,
            "mce": self.mce,
            "brier_score": self.brier_score,
            "n_samples": self.n_samples,
            "n_bins": self.n_bins,
        }


def compute_reliability_diagram(
    probabilities: ArrayLike,
    labels: ArrayLike,
    *,
    n_bins: int = 10,
) -> ReliabilityDiagram:
    """Calcula bins uniformes em [0.0, 1.0], ECE, MCE e Brier score."""
    if isinstance(n_bins, bool) or n_bins < 1:
        raise ValueError("n_bins must be an integer >= 1.")

    probs = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.float64)

    if probs.ndim == 0 or targets.ndim == 0:
        raise ValueError("probabilities and labels must be at least 1D arrays.")
    if probs.size == 0 or targets.size == 0:
        raise ValueError("probabilities and labels must be non-empty.")
    if probs.shape != targets.shape:
        raise ValueError(
            f"probabilities shape {probs.shape} does not match labels shape {targets.shape}."
        )

    probs = probs.ravel()
    targets = targets.ravel()

    if not np.all(np.isfinite(probs)):
        raise ValueError("probabilities must contain only finite values.")
    if not np.all(np.isfinite(targets)):
        raise ValueError("labels must contain only finite values.")
    if np.any((probs < 0.0) | (probs > 1.0)):
        raise ValueError("probabilities must be in [0.0, 1.0].")
    if not np.all((targets == 0.0) | (targets == 1.0)):
        raise ValueError("labels must be binary in {0, 1}.")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins_list: list[ReliabilityBin] = []

    for idx in range(n_bins):
        lower = float(edges[idx])
        upper = float(edges[idx + 1])
        if idx < n_bins - 1:
            mask = (probs >= lower) & (probs < upper)
        else:
            mask = (probs >= lower) & (probs <= upper)

        count = int(np.sum(mask))
        if count > 0:
            mean_confidence = float(np.mean(probs[mask]))
            empirical_accuracy = float(np.mean(targets[mask]))
            gap = abs(empirical_accuracy - mean_confidence)
        else:
            mean_confidence = 0.0
            empirical_accuracy = 0.0
            gap = 0.0

        bins_list.append(
            ReliabilityBin(
                lower=lower,
                upper=upper,
                count=count,
                mean_confidence=mean_confidence,
                empirical_accuracy=empirical_accuracy,
                gap=gap,
            )
        )

    n_samples = int(probs.size)
    ece = float(sum((b.count / n_samples) * b.gap for b in bins_list))
    mce = float(max((b.gap for b in bins_list), default=0.0))
    brier_score = float(np.mean((probs - targets) ** 2))

    return ReliabilityDiagram(
        bins=tuple(bins_list),
        ece=ece,
        mce=mce,
        brier_score=brier_score,
        n_samples=n_samples,
        n_bins=n_bins,
    )
