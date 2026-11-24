"""Biomarcadores oportunistas de TC sobre o mapa ``total`` do TotalSegmentator.

Algoritmo real (sem pesos próprios) que consome a saída do modelo a montante
(``depends_on``) e as HU da própria imagem. Cada biomarcador confere suas
pré-condições de aquisição e se abstém com motivo quando não as satisfaz
(REQ-133); os limiares de decisão são os pontos de operação com origem do
manifesto ``ct-biomarkers.yaml`` (REQ-134).
"""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

from radiologyai.analysis.base import (
    AdapterOutput,
    ModelAdapter,
    ModelInputs,
    RawMeasurement,
    ResolvedWeights,
)
from radiologyai.analysis.types import AbstentionReason, QACheck, QAReport
from radiologyai.errors import BackendUnavailableError, ManifestError, UnsupportedInputError

if TYPE_CHECKING:
    from radiologyai.analysis.biomarkers.ct import CTContext
    from radiologyai.analysis.manifest import ModelManifest

__all__ = ["BIOMARKERS", "CTBiomarkerAdapter"]

#: biomarcadores implementados (chave de configuração -> medidas emitidas)
BIOMARKERS: dict[str, tuple[str, ...]] = {
    "l1_trabecular": ("l1_trabecular_hu",),
    "liver_unenhanced": (
        "liver_attenuation_unenhanced_hu",
        "liver_spleen_difference_unenhanced_hu",
    ),
    "liver_portal_venous": ("liver_attenuation_portal_venous_hu",),
    "emphysema": ("lung_laa950_percent",),
    "aorta": (
        "aorta_abdominal_max_diameter_mm",
        "aorta_ascending_max_diameter_mm",
        "aorta_descending_thoracic_max_diameter_mm",
    ),
    "organ_volumes": ("spleen_volume_ml", "liver_volume_ml", "heart_volume_ml"),
}


class CTBiomarkerAdapter(ModelAdapter):
    """Mede biomarcadores em HU/tamanho a partir do mapa de rótulos do modelo a montante."""

    def __init__(self, manifest: ModelManifest) -> None:
        super().__init__(manifest)
        upstream = manifest.config.get("anatomy_model")
        if not isinstance(upstream, str) or upstream not in manifest.depends_on:
            raise ManifestError(f"{manifest.id}: config.anatomy_model deve estar em depends_on")
        self.anatomy_model = upstream
        enabled = manifest.config.get("biomarkers", list(BIOMARKERS))
        unknown = [b for b in enabled if b not in BIOMARKERS]
        if unknown:
            raise ManifestError(f"{manifest.id}: biomarcadores desconhecidos {unknown}")
        self.enabled: tuple[str, ...] = tuple(enabled)

    def load(self, weights: ResolvedWeights, device: str) -> None:
        del weights, device  # algoritmo sem pesos próprios
        if importlib.util.find_spec("scipy") is None:
            raise BackendUnavailableError(
                "scipy não instalado; instale o extra: pip install 'radiologyai[volumetric]'"
            )

    def applicability(self, inputs: ModelInputs) -> QAReport:
        base = super().applicability(inputs)
        checks = list(base.checks)
        reasons = list(base.reasons)
        output = inputs.upstream.get(self.anatomy_model)
        ok = output is not None and bool(output.labelmaps)
        checks.append(QACheck(name=f"upstream:{self.anatomy_model}", passed=ok))
        if not ok:
            reasons.append(AbstentionReason.UPSTREAM)
        if len(inputs.images) == 1:
            image = inputs.images[0]
            geo = image.spacing_mm is not None and image.affine_lps is not None
            checks.append(QACheck(name="geometry", passed=geo))
            if not geo:
                reasons.append(AbstentionReason.GEOMETRY)
        unique = tuple(dict.fromkeys(reasons))
        return QAReport(applicable=not unique, checks=tuple(checks), reasons=unique)

    def _context(self, inputs: ModelInputs) -> CTContext:
        from radiologyai.analysis.biomarkers.ct import CTContext

        image = inputs.single
        output = inputs.upstream_output(self.anatomy_model)
        if not output.labelmaps:
            raise UnsupportedInputError(f"{self.anatomy_model} não produziu mapa de rótulos")
        labelmap = output.labelmaps[0]
        if image.spacing_mm is None or image.affine_lps is None:
            raise UnsupportedInputError("imagem sem geometria")
        sz, sy, sx = image.spacing_mm
        return CTContext(
            hu=image.pixels,
            spacing_zyx=(sz, sy, sx),
            affine_lps=image.affine_lps,
            meta=image.meta,
            labelmap=labelmap.array,
            names={name: idx for idx, name in labelmap.labels.items()},
            source=f"{self.anatomy_model}:{labelmap.name}",
        )

    def predict(self, inputs: ModelInputs) -> AdapterOutput:
        from radiologyai.analysis.biomarkers import ct

        ctx = self._context(inputs)
        out: list[RawMeasurement] = []
        if "l1_trabecular" in self.enabled:
            out.append(ct.l1_trabecular_hu(ctx))
        if "liver_unenhanced" in self.enabled:
            out.append(ct.liver_attenuation(ctx, "none"))
            out.append(ct.liver_spleen_difference(ctx))
        if "liver_portal_venous" in self.enabled:
            out.append(ct.liver_attenuation(ctx, "portal_venous"))
        if "emphysema" in self.enabled:
            out.append(ct.laa950_percent(ctx))
        if "aorta" in self.enabled:
            out.extend(ct.aortic_diameters(ctx))
        if "organ_volumes" in self.enabled:
            out.append(ct.organ_volume(ctx, "spleen_volume_ml", ("spleen",)))
            out.append(ct.organ_volume(ctx, "liver_volume_ml", ("liver",)))
            out.append(ct.organ_volume(ctx, "heart_volume_ml", ("heart",)))
        return AdapterOutput(measurements=out)
