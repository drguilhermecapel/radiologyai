"""Adapter protocol: the only way a model can contribute to an analysis.

FROZEN CONTRACT.

An adapter wraps ONE real pretrained model (or one real algorithm that derives
measurements from other models' outputs). It returns raw scores, label maps and
measurements only. It never decides what is "positive", never invents a
threshold and never produces output it did not compute from the input: the
pipeline applies calibration, operating points, abstention and evidence levels.
If the model cannot run, the adapter raises; it does not substitute a value.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import torch

from medai.core.errors import UnsupportedInput
from medai.core.types import (
    AbstentionReason,
    ExamType,
    ImageData,
    Primitive,
    QACheck,
    QAReport,
    ScoreKind,
)
from medai.models.manifest import ModelManifest


@dataclass(frozen=True, slots=True)
class ResolvedWeights:
    """Weight files that passed the integrity check. Only the registry builds this."""

    files: Mapping[str, Path]  # key -> file, or extracted directory for archives
    sha256: Mapping[str, str]


@dataclass(slots=True)
class Score:
    label: str  # must be a label key declared in the manifest
    value: float
    kind: ScoreKind


@dataclass(slots=True)
class LabelMap:
    """Integer label map in the geometry of ``inputs.images[image_idx]``."""

    name: str
    array: npt.NDArray[np.integer]
    labels: dict[int, str]  # index (>0) -> label key; 0 is background
    image_idx: int = 0


@dataclass(slots=True)
class RawMeasurement:
    key: str  # stable measurement name, e.g. "cardiothoracic_ratio"
    value: float | None
    unit: str
    method: str
    geometry: list[Primitive] = field(default_factory=list)
    derived_from: list[str] = field(default_factory=list)  # LabelMap names
    preconditions: dict[str, str] = field(default_factory=dict)
    abstention: AbstentionReason | None = None  # required when value is None
    note: str | None = None
    image_idx: int = 0


@dataclass(slots=True)
class ExplanationMap:
    label: str
    heatmap: npt.NDArray[np.float32]  # 2D, values in [0, 1], any resolution
    placement: Primitive  # bbox in source-image pixels that the heatmap covers
    method: Literal["cam", "gradcam", "attention", "occlusion"] = "cam"
    image_idx: int = 0


@dataclass(slots=True)
class AdapterOutput:
    scores: list[Score] = field(default_factory=list)
    class_scores: dict[str, dict[str, float]] = field(default_factory=dict)  # label -> class -> p
    labelmaps: list[LabelMap] = field(default_factory=list)
    measurements: list[RawMeasurement] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)  # for downstream adapters only


@dataclass(frozen=True, slots=True)
class ModelInputs:
    """What the pipeline hands to an adapter."""

    images: tuple[ImageData, ...]
    exam_type: ExamType
    roles: Mapping[str, int] = field(default_factory=dict)  # role -> index into images
    upstream: Mapping[str, AdapterOutput] = field(default_factory=dict)  # model id -> output
    prompts: tuple[Primitive, ...] = ()

    @property
    def single(self) -> ImageData:
        if len(self.images) != 1:
            raise UnsupportedInput(f"expected exactly one image, got {len(self.images)}")
        return self.images[0]

    def by_role(self, role: str) -> ImageData:
        if role not in self.roles:
            raise UnsupportedInput(f"required input role {role!r} is missing")
        return self.images[self.roles[role]]


class ModelAdapter(ABC):
    """Base class for every adapter. Subclasses live in ``medai.models.adapters``."""

    def __init__(self, manifest: ModelManifest) -> None:
        self.manifest = manifest
        self.device = torch.device("cpu")
        self._loaded = False

    # -- lifecycle --------------------------------------------------------- #
    @abstractmethod
    def load(self, weights: ResolvedWeights, device: torch.device) -> None:
        """Build the model from integrity-checked weights. No network access here."""

    def unload(self) -> None:  # noqa: B027 - optional hook
        """Release memory. Default: nothing to do."""

    # -- inference --------------------------------------------------------- #
    def applicability(self, inputs: ModelInputs) -> QAReport:
        """Decide whether this model may run on these inputs.

        The default checks the manifest's static applicability. Override to add
        image-content checks (view, quality, out-of-distribution) and call
        ``super().applicability(inputs)`` first.
        """
        return static_applicability(self.manifest, inputs)

    @abstractmethod
    def predict(self, inputs: ModelInputs) -> AdapterOutput:
        """Run the real model. Raise on failure; never return placeholder values."""

    def explain(self, inputs: ModelInputs, labels: Sequence[str]) -> list[ExplanationMap]:
        """Explanations for the given label keys. Default: none available."""
        return []


def static_applicability(manifest: ModelManifest, inputs: ModelInputs) -> QAReport:
    """Manifest-level gate shared by all adapters."""
    spec = manifest.applicability
    checks: list[QACheck] = []
    reasons: list[AbstentionReason] = []

    ok_exam = inputs.exam_type in spec.exam_types
    checks.append(QACheck(name="exam_type", passed=ok_exam, value=inputs.exam_type.value))
    if not ok_exam:
        reasons.append(AbstentionReason.EXAM_TYPE)

    if not inputs.images:
        checks.append(QACheck(name="has_images", passed=False))
        reasons.append(AbstentionReason.MISSING_INPUT)
        return QAReport(applicable=False, checks=checks, reasons=reasons)

    for idx, image in enumerate(inputs.images):
        ok_mod = image.modality in spec.modalities
        ok_kind = image.kind in spec.image_kinds
        checks.append(QACheck(name=f"modality[{idx}]", passed=ok_mod, value=image.modality.value))
        checks.append(QACheck(name=f"image_kind[{idx}]", passed=ok_kind, value=image.kind.value))
        if not ok_mod and AbstentionReason.MODALITY not in reasons:
            reasons.append(AbstentionReason.MODALITY)
        if not ok_kind and AbstentionReason.INPUT not in reasons:
            reasons.append(AbstentionReason.INPUT)
        if spec.min_pixels is not None:
            plane = image.pixels.shape[-3:-1] if image.color else image.pixels.shape[-2:]
            shortest = int(min(plane))
            ok_px = shortest >= spec.min_pixels
            checks.append(
                QACheck(
                    name=f"min_pixels[{idx}]",
                    passed=ok_px,
                    value=shortest,
                    threshold=spec.min_pixels,
                )
            )
            if not ok_px and AbstentionReason.QUALITY not in reasons:
                reasons.append(AbstentionReason.QUALITY)
        if spec.views and image.meta.view_position is not None:
            ok_view = image.meta.view_position.upper() in {v.upper() for v in spec.views}
            checks.append(
                QACheck(name=f"view[{idx}]", passed=ok_view, value=image.meta.view_position)
            )
            if not ok_view and AbstentionReason.VIEW not in reasons:
                reasons.append(AbstentionReason.VIEW)
        if spec.adult_only and image.meta.patient_age_years is not None:
            ok_age = image.meta.patient_age_years >= 18
            checks.append(
                QACheck(
                    name=f"adult[{idx}]",
                    passed=ok_age,
                    value=image.meta.patient_age_years,
                    threshold=18,
                )
            )
            if not ok_age and AbstentionReason.POPULATION not in reasons:
                reasons.append(AbstentionReason.POPULATION)

    if manifest.inputs.mode == "multi":
        missing = [role for role in manifest.inputs.roles if role not in inputs.roles]
        checks.append(
            QACheck(name="required_roles", passed=not missing, value=",".join(missing) or None)
        )
        if missing:
            reasons.append(AbstentionReason.MISSING_INPUT)
    if manifest.inputs.prompts == "required" and not inputs.prompts:
        checks.append(QACheck(name="prompt", passed=False))
        reasons.append(AbstentionReason.MISSING_INPUT)

    return QAReport(applicable=not reasons, checks=checks, reasons=reasons)
