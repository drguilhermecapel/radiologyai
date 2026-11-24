"""Exam-type inference and model routing.

FROZEN CONTRACT (function signatures and ``routing.yaml`` layout).

Routing is explicit: ``routing.yaml`` lists which models run for each exam type.
When the exam type cannot be established from metadata the answer is
``ExamType.UNKNOWN`` and a human must choose - the system never guesses silently.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from medai.core.errors import ConfigurationError
from medai.core.types import ExamType, ImageData, ImageKind, Modality

ROUTING_FILE = Path(__file__).parent / "routing.yaml"


class Route(BaseModel):
    model_config = ConfigDict(extra="forbid")

    main: list[str] = Field(default_factory=list)  # models whose conditions can reach levels A-C
    research: list[str] = Field(default_factory=list)  # run only when research mode is requested
    interactive: list[str] = Field(default_factory=list)  # run only with a user prompt


class RoutingTable(BaseModel):
    model_config = ConfigDict(extra="forbid")

    exam_types: dict[ExamType, Route]

    def models_for(self, exam_type: ExamType, *, include_research: bool = False) -> list[str]:
        route = self.exam_types.get(exam_type, Route())
        return [*route.main, *(route.research if include_research else [])]

    def interactive_for(self, exam_type: ExamType) -> list[str]:
        return list(self.exam_types.get(exam_type, Route()).interactive)

    def all_model_ids(self) -> set[str]:
        ids: set[str] = set()
        for route in self.exam_types.values():
            ids.update(route.main, route.research, route.interactive)
        return ids


def load_routing(path: Path = ROUTING_FILE) -> RoutingTable:
    try:
        return RoutingTable.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except Exception as exc:
        raise ConfigurationError(f"{path}: {exc}") from exc


# --------------------------------------------------------------------------- #
# Exam-type inference from acquisition metadata
# --------------------------------------------------------------------------- #
_CHEST = ("chest", "thorax", "torax", "tórax", "lung", "pulm", "pecho", "rib")
_HEAD = ("head", "brain", "skull", "cranio", "crânio", "cranium", "neuro", "encefalo", "encéfalo")
_ABDOMEN = ("abdom", "pelvis", "pelve", "liver", "figado", "fígado", "kidney", "renal", "urogra")
_MSK = (
    "hand", "wrist", "finger", "elbow", "forearm", "humerus", "shoulder", "clavicle",
    "knee", "ankle", "foot", "toe", "leg", "tibia", "fibula", "femur", "hip", "extremity",
    "mao", "mão", "punho", "dedo", "cotovelo", "antebraco", "antebraço", "ombro", "joelho",
    "tornozelo", "pe ", "pé", "perna", "quadril", "bacia",
)  # fmt: skip
_BREAST = ("breast", "mama", "mamm")
_CARDIAC = ("echo", "eco", "cardiac", "heart", "tte", "cardio")


def _text(image: ImageData) -> str:
    meta = image.meta
    parts = [meta.body_part, meta.study_description, meta.series_description]
    return " ".join(p for p in parts if p).lower()


def _has(text: str, words: tuple[str, ...]) -> bool:
    return any(w in text for w in words)


def infer_exam_type(images: Sequence[ImageData]) -> ExamType:
    """Best-effort exam type from DICOM-level metadata; ``UNKNOWN`` when not certain."""
    if not images:
        return ExamType.UNKNOWN
    modalities = {image.modality for image in images}
    text = " ".join(_text(image) for image in images)

    if modalities <= {Modality.CR, Modality.DX}:
        if _has(text, _CHEST):
            return ExamType.CXR
        if _has(text, _MSK):
            return ExamType.MSK_XR
        return ExamType.UNKNOWN
    if modalities == {Modality.MG}:
        return ExamType.MG
    if modalities == {Modality.CT}:
        if _has(text, _HEAD):
            return ExamType.CT_HEAD
        if _has(text, _CHEST) and not _has(text, _ABDOMEN):
            return ExamType.CT_CHEST
        if _has(text, _ABDOMEN):
            return ExamType.CT_ABDOMEN
        return ExamType.CT_OTHER
    if modalities == {Modality.MR}:
        return ExamType.MR_BRAIN if _has(text, _HEAD) else ExamType.MR_BODY
    if modalities == {Modality.US}:
        if _has(text, _CARDIAC) or all(image.kind is ImageKind.CINE for image in images):
            return ExamType.ECHO if _has(text, _CARDIAC) else ExamType.UNKNOWN
        if _has(text, _BREAST):
            return ExamType.US_BREAST
        if _has(text, _ABDOMEN):
            return ExamType.US_ABDOMEN
        return ExamType.US_OTHER
    return ExamType.UNKNOWN


def role_of(image: ImageData) -> str | None:
    """Input role of an image for multi-input models (mammography views, MR sequences)."""
    meta = image.meta
    if image.modality is Modality.MG:
        view = (meta.view_position or "").upper().replace(" ", "")
        if meta.laterality in ("L", "R") and view in ("CC", "MLO"):
            return f"{meta.laterality}-{view}"
        return None
    if meta.sequence_role:
        return meta.sequence_role.lower()
    if meta.view_position and image.modality is Modality.US:
        return meta.view_position.lower()
    return None
