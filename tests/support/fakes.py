"""Deterministic test doubles for the adapter protocol (tests ONLY).

Each fake computes its output from the actual input pixels, so the pipeline
tests can assert that different inputs give different results. They are
registered under ``medai.models.adapters._test_fakes`` by ``conftest.py`` at
test time and are never shipped inside the package.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from medai.core.types import AbstentionReason, Primitive, QAReport
from medai.models.base import (
    AdapterOutput,
    ExplanationMap,
    LabelMap,
    ModelAdapter,
    ModelInputs,
    RawMeasurement,
    ResolvedWeights,
    Score,
)


class _Base(ModelAdapter):
    loaded_from: ResolvedWeights | None = None

    def load(self, weights: ResolvedWeights, device: torch.device) -> None:
        self.loaded_from = weights


class Brightness(_Base):
    """Binary score = mean pixel / 255 (a real function of the input)."""

    def predict(self, inputs: ModelInputs) -> AdapterOutput:
        value = float(inputs.single.pixels.mean() / 255.0)
        return AdapterOutput(scores=[Score("bright", value, "sigmoid")])

    def explain(self, inputs: ModelInputs, labels: Sequence[str]) -> list[ExplanationMap]:
        image = inputs.single.pixels
        heat = (image - image.min()) / max(float(image.max() - image.min()), 1e-6)
        h, w = image.shape
        box = Primitive(type="bbox", xy=[(0.0, 0.0), (float(w), float(h))])
        return [
            ExplanationMap(label=label, heatmap=heat.astype(np.float32), placement=box)
            for label in labels
        ]


class Segmenter(_Base):
    """Label map = pixels above the image mean; measurement = bright area fraction."""

    def predict(self, inputs: ModelInputs) -> AdapterOutput:
        image = inputs.single.pixels
        mask = (image > image.mean()).astype(np.uint8)
        return AdapterOutput(
            labelmaps=[LabelMap(name="bright_region", array=mask, labels={1: "bright_region"})],
            measurements=[
                RawMeasurement(
                    key="bright_fraction",
                    value=float(mask.mean()),
                    unit="1",
                    method="fraction of pixels above the image mean",
                    derived_from=["bright_region"],
                )
            ],
        )


class Derived(_Base):
    """Consumes the Segmenter output; emits a measurement that a derived label thresholds."""

    def predict(self, inputs: ModelInputs) -> AdapterOutput:
        upstream = inputs.upstream["fake-segmenter"]
        fraction = upstream.measurements[0].value
        assert fraction is not None
        return AdapterOutput(
            measurements=[
                RawMeasurement(
                    key="doubled_fraction", value=2.0 * fraction, unit="1", method="2 x upstream"
                )
            ]
        )


class DerivedAbstains(_Base):
    def predict(self, inputs: ModelInputs) -> AdapterOutput:
        return AdapterOutput(
            measurements=[
                RawMeasurement(
                    key="doubled_fraction",
                    value=None,
                    unit="1",
                    method="2 x upstream",
                    abstention=AbstentionReason.ACQUISITION,
                )
            ]
        )


class Categorical(_Base):
    def predict(self, inputs: ModelInputs) -> AdapterOutput:
        mean = float(inputs.single.pixels.mean() / 255.0)
        return AdapterOutput(class_scores={"density": {"low": 1.0 - mean, "high": mean}})


class Exploding(_Base):
    def predict(self, inputs: ModelInputs) -> AdapterOutput:
        raise RuntimeError("model blew up")


class NeverApplicable(_Base):
    def applicability(self, inputs: ModelInputs) -> QAReport:
        return QAReport.reject(AbstentionReason.OOD)

    def predict(self, inputs: ModelInputs) -> AdapterOutput:  # pragma: no cover
        raise AssertionError("predict must not be called when not applicable")


class UndeclaredLabel(_Base):
    def predict(self, inputs: ModelInputs) -> AdapterOutput:
        return AdapterOutput(scores=[Score("not_in_manifest", 0.5, "sigmoid")])


class MultiRole(_Base):
    def predict(self, inputs: ModelInputs) -> AdapterOutput:
        left, right = inputs.by_role("L-CC"), inputs.by_role("R-CC")
        diff = float(abs(left.pixels.mean() - right.pixels.mean()) / 255.0)
        return AdapterOutput(scores=[Score("asymmetry", diff, "sigmoid")])
