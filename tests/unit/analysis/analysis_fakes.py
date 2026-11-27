"""Dublês determinísticos do protocolo de adaptador (SOMENTE testes).

Cada dublê calcula a saída a partir dos pixels reais da entrada, de modo que os
testes do pipeline possam afirmar que entradas diferentes dão resultados
diferentes. São registrados em ``radiologyai.analysis.adapters._test_fakes``
pelo ``conftest.py`` durante o teste e nunca são distribuídos no pacote.
"""

# ruff: noqa: ARG002 - dublês implementam a assinatura do contrato sem usar tudo
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from radiologyai.analysis.base import (
    AdapterOutput,
    ExplanationMap,
    LabelMap,
    ModelAdapter,
    ModelInputs,
    RawMeasurement,
    ResolvedWeights,
    Score,
)
from radiologyai.analysis.types import AbstentionReason, Primitive, QAReport


class _Base(ModelAdapter):
    loaded_from: ResolvedWeights | None = None
    loaded_device: str | None = None
    unloaded: bool = False

    def load(self, weights: ResolvedWeights, device: str) -> None:
        self.loaded_from = weights
        self.loaded_device = device

    def unload(self) -> None:
        self.unloaded = True


class Brightness(_Base):
    """Escore binário = média dos pixels / 255 (função real da entrada)."""

    def predict(self, inputs: ModelInputs) -> AdapterOutput:
        value = float(inputs.single.pixels.mean() / 255.0)
        return AdapterOutput(scores=[Score("bright", value, "sigmoid")])

    def explain(self, inputs: ModelInputs, labels: Sequence[str]) -> list[ExplanationMap]:
        image = inputs.single.pixels
        heat = (image - image.min()) / max(float(image.max() - image.min()), 1e-6)
        h, w = image.shape
        box = Primitive(type="bbox", xy=((0.0, 0.0), (float(w), float(h))))
        return [
            ExplanationMap(label=label, heatmap=heat.astype(np.float32), placement=box)
            for label in labels
        ]


class LogitScore(_Base):
    """Escore em logit = (média dos pixels - 128) / 16, para o primeiro rótulo do manifesto."""

    def predict(self, inputs: ModelInputs) -> AdapterOutput:
        value = float((inputs.single.pixels.mean() - 128.0) / 16.0)
        return AdapterOutput(scores=[Score(self.manifest.labels[0].key, value, "logit")])


class Segmenter(_Base):
    """Máscara = pixels acima da média; medida = fração da área clara."""

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
                    method="fração de pixels acima da média",
                    derived_from=["bright_region"],
                )
            ],
            extras={"mask": mask},
        )


class Derived(_Base):
    """Consome a saída do Segmenter; emite medida que um rótulo derivado limiariza."""

    def predict(self, inputs: ModelInputs) -> AdapterOutput:
        upstream = inputs.upstream_output("fake-segmenter")
        fraction = upstream.measurements[0].value
        assert fraction is not None
        return AdapterOutput(
            measurements=[
                RawMeasurement(
                    key="doubled_fraction", value=2.0 * fraction, unit="1", method="2 x montante"
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
                    method="2 x montante",
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
        raise RuntimeError("modelo explodiu")


class NeverApplicable(_Base):
    def applicability(self, inputs: ModelInputs) -> QAReport:
        return QAReport.reject(AbstentionReason.OOD)

    def predict(self, inputs: ModelInputs) -> AdapterOutput:  # pragma: no cover
        raise AssertionError("predict não pode ser chamado quando não aplicável")


class UndeclaredLabel(_Base):
    def predict(self, inputs: ModelInputs) -> AdapterOutput:
        return AdapterOutput(scores=[Score("not_in_manifest", 0.5, "sigmoid")])


class MultiRole(_Base):
    def predict(self, inputs: ModelInputs) -> AdapterOutput:
        left, right = inputs.by_role("L-CC"), inputs.by_role("R-CC")
        diff = float(abs(left.pixels.mean() - right.pixels.mean()) / 255.0)
        return AdapterOutput(scores=[Score("asymmetry", diff, "sigmoid")])


class BadExplanation(Brightness):
    def explain(self, inputs: ModelInputs, labels: Sequence[str]) -> list[ExplanationMap]:
        raise RuntimeError("mapa falhou")


class PointPlacementExplanation(Brightness):
    """Mapa com placement 'point' (o contrato exige bbox): deve virar aviso, nunca falha."""

    def explain(self, inputs: ModelInputs, labels: Sequence[str]) -> list[ExplanationMap]:
        image = inputs.single.pixels
        point = Primitive(type="point", xy=((1.0, 1.0),))
        return [
            ExplanationMap(
                label=label, heatmap=np.zeros_like(image, dtype=np.float32), placement=point
            )
            for label in labels
        ]
